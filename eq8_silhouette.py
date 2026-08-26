#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
[식 ⑧] 가림 그림자로 선형 부재의 옆 기울기를 되찾는다
========================================================================
문제
----
격자선이 한 줄만 걸린 부재는 각도의 **한 성분** 밖에 못 잰다. 한 줄에서
나온 3D 점은 그 레이저 평면 안에 놓이므로, 평면 안 기울기는 보이고
평면에 수직인 기울기는 전혀 안 보인다(eq7·pipeline_region 참조).
실측에서 실제로 2.00° 기운 부재가 0.0000° 로 나왔다.

그런데 가로선은 그 부재를 21번 가로지른다. 가로선 자체는 깊이를 못
주지만(g=∞), **부재가 뒤 배경에 드리우는 그림자**는 화면에서 또렷하다.

왜 그림자가 생기나
-----------------
카메라는 조사기에서 기선 b 만큼 옆으로 떨어져 있다. 레이저가 부재에
막혀 못 간 자리를 카메라는 볼 수 있고, 그 자리는 레이저가 안 닿았으니
어둡다. 부재 뒤쪽으로 폭이

    w = f·b·(1/Z_부재 − 1/Z_배경)      [px]

인 띠가 비어 보인다. 이 예측이 실측과 맞는지가 이 방법의 안전장치다
(이 표본: 예측 27px, 실측 25~26px).

무엇을 얻나
----------
그림자가 시작하는 자리가 곧 **부재의 실루엣 가장자리** 다. 부재의 깊이
Z 는 세로선에서 이미 알고 있으므로, 가장자리 화소를 그 깊이에 놓으면
3D 점이 된다.

    X(v) = (u_edge − c_x)·Z/f + b,   Y(v) = (v − c_y)·Z/f

높이마다 이 점을 모아 직선을 맞추면 **평면에 수직인 기울기까지** 나온다.
부재가 옆으로 0.5° 기울면 2m 구간에서 가장자리가 8.5px 움직인다 —
가장자리 검출 오차 ±2px 보다 훨씬 크다.

한계 — 반드시 지킬 것
--------------------
· 부재 뒤에 배경이 없으면 그림자가 안 생긴다. 이 표본의 셋째 동바리가
  그랬다(가장자리 42/63 만 잡힘). 못 잡으면 판정보류를 유지한다.
· 그림자 폭이 예측과 다르면 그 가장자리는 버린다. 다른 원인(부재 경계,
  반사 소실)으로 생긴 끊김을 그림자로 오인하면 엉뚱한 각도가 나온다.
· 가장자리는 부재의 **한쪽 옆면** 이다. 부재 지름이 높이에 따라 변하면
  (테이퍼) 그만큼 기울기로 잘못 읽힌다. 동바리·철근은 등단면이라 문제
  없지만, 조서에 근거를 남겨 사람이 확인할 수 있게 한다.
========================================================================
"""
import numpy as np


def laser_signal(rgb):
    """초록 과잉분 — 레이저만 남긴다."""
    a = np.asarray(rgb, dtype=float)
    return a[:, :, 1] - 0.5 * (a[:, :, 0] + a[:, :, 2])


def shadow_width_px(z_member, z_background, f_px, b_m):
    """부재가 배경에 드리우는 그림자 폭 [px]."""
    if z_background <= z_member or z_member <= 0:
        return None
    return float(f_px) * float(b_m) * (1.0 / float(z_member)
                                       - 1.0 / float(z_background))


def _runs(mask):
    """1차원 불리언에서 True 구간 [(start, end), ...] (end 는 배타)."""
    d = np.diff(mask.astype(np.int8))
    s = list(np.where(d == 1)[0] + 1)
    e = list(np.where(d == -1)[0] + 1)
    if mask[0]:
        s = [0] + s
    if mask[-1]:
        e = e + [len(mask)]
    return list(zip(s, e))


def find_shadow_edges(signal, rows, u_hint, z_member, z_background,
                      f_px, b_m, search_px=90.0, width_tol=0.45,
                      thresh_frac=0.5):
    """
    각 행에서 부재의 실루엣 가장자리를 찾는다.

    Parameters
    ----------
    signal : (H,W) 레이저 신호 (laser_signal 출력)
    rows : 훑을 행 번호들 (보통 가로선이 지나는 행)
    u_hint : 부재가 있을 만한 대략의 u — 세로선 위치를 쓴다
    z_member, z_background : 부재·배경 깊이 [m]
    search_px : u_hint 둘레 탐색 반경
    width_tol : 그림자 폭이 예측과 이만큼(상대) 어긋나면 버린다

    Returns
    -------
    dict — edges [(v, u_edge)], expected_width_px, n_checked, n_found
    """
    sig = np.asarray(signal, float)
    H, W = sig.shape
    want = shadow_width_px(z_member, z_background, f_px, b_m)
    out = {"edges": [], "expected_width_px": (round(want, 2) if want else None),
           "n_checked": 0, "n_found": 0, "reason": None}
    if want is None or want < 3.0:
        out["reason"] = ("배경이 부재보다 앞이거나 너무 가까워 그림자가 "
                         "생기지 않는다")
        return out
    thr = float(sig.max()) * float(thresh_frac)
    if thr <= 0:
        out["reason"] = "레이저 신호 없음"
        return out

    lo = max(0, int(u_hint - search_px))
    hi = min(W, int(u_hint + search_px))
    for v in rows:
        v = int(round(v))
        if not (0 <= v < H):
            continue
        out["n_checked"] += 1
        seg = sig[v, lo:hi] > thr
        if seg.size < 5 or not seg.any():
            continue
        # 켜진 구간 사이의 빈틈 중 폭이 예측과 맞는 것을 고른다
        best = None
        rr = _runs(seg)
        for k in range(len(rr) - 1):
            gap_a, gap_b = rr[k][1], rr[k + 1][0]
            w = gap_b - gap_a
            if abs(w - want) > width_tol * want:
                continue
            edge = lo + gap_a - 1          # 그림자 직전 = 부재 가장자리
            if best is None or abs(edge - u_hint) < abs(best - u_hint):
                best = edge
        # 화면 끝에서 잘린 그림자도 받는다 — 마지막 켜진 구간의 끝이
        # 탐색창 끝보다 충분히 안쪽이면 그 뒤는 그림자다
        if best is None and rr:
            tail = lo + rr[-1][1] - 1
            if tail < hi - 1 - want * (1 - width_tol):
                best = tail
        if best is not None:
            out["edges"].append((float(v), float(best)))
            out["n_found"] += 1
    if not out["edges"]:
        out["reason"] = "예측 폭과 맞는 그림자를 못 찾았다"
    return out


def axis_from_edges(edges, z_member, f_px, cx_px, cy_px, b_m, g_hat,
                    max_rms_px=2.5, robust=True):
    """
    실루엣 가장자리 화소들을 부재 깊이에 놓고 축을 맞춘다.

    가장자리는 부재의 한쪽 옆면이므로 축과 나란하다(등단면 가정).
    따라서 방향은 축 방향과 같고, 위치만 반지름만큼 옆으로 밀려 있다 —
    각도만 쓰므로 문제 되지 않는다.

    Returns
    -------
    dict — direction, theta_deg, n_points, rms_px, span_m
    """
    E = np.asarray(edges, dtype=float).reshape(-1, 2)
    if len(E) < 4:
        return {"ok": False, "reason": f"가장자리 점 부족 ({len(E)})"}
    n_all = len(E)
    if robust and len(E) >= 8:
        # 몇 행에서 엉뚱한 끊김을 그림자로 오인할 수 있다. 가장자리는
        # 원래 직선이어야 하므로, 직선에서 크게 벗어난 점을 먼저 버린다.
        vv, uu = E[:, 0], E[:, 1]
        co = np.polyfit(vv, uu, 1)
        r = uu - np.polyval(co, vv)
        mad = float(np.median(np.abs(r - np.median(r))))
        thr = max(3.0 * 1.4826 * mad, 2.0)
        keep = np.abs(r - np.median(r)) <= thr
        if keep.sum() >= 6:
            E = E[keep]
    v, u = E[:, 0], E[:, 1]
    Z = float(z_member)
    X = (u - cx_px) * Z / f_px + b_m
    Y = (v - cy_px) * Z / f_px
    P = np.column_stack([X, Y, np.full_like(X, Z)])

    c = P.mean(axis=0)
    _, s, Vt = np.linalg.svd(P - c, full_matrices=False)
    d = Vt[0]
    if d @ np.asarray(g_hat, float) < 0:
        d = -d
    # 직선에서 얼마나 벗어나는가 — 화소로 되돌려 본다
    r = (P - c) - np.outer((P - c) @ d, d)
    rms_m = float(np.sqrt(np.mean(np.sum(r ** 2, axis=1))))
    rms_px = rms_m * f_px / Z

    g = np.asarray(g_hat, float)
    g = g / np.linalg.norm(g)
    cosang = abs(float(np.dot(d / np.linalg.norm(d), g)))
    theta = float(np.degrees(np.arccos(min(cosang, 1.0))))
    out = {"direction": d.tolist(),
           "theta_deg": round(theta, 4), "n_points": int(len(E)),
           "n_dropped": int(n_all - len(E)),
           "rms_px": round(float(rms_px), 3),
           "span_m": round(float(np.ptp(Y)), 4),
           "u_span_px": round(float(np.ptp(u)), 2)}
    # ── 자기 검증 ── 가장자리는 직선이어야 한다.
    # 실측에서 셋째 동바리의 가장자리가 43px 이나 튀며 잔차 6.2px 를
    # 냈고, 그대로 두면 없는 기울기 0.36° 를 지어냈다(참값 0°). 잔차가
    # 검출 오차(±2px) 수준을 넘으면 가장자리를 잘못 잡은 것이므로
    # 되살리지 않고 판정보류를 유지한다.
    if rms_px > max_rms_px:
        out["ok"] = False
        out["reason"] = (f"가장자리가 직선이 아니다 (잔차 {rms_px:.2f}px > "
                         f"{max_rms_px}px). 그림자 아닌 끊김을 잡았을 "
                         f"가능성이 크다")
        return out
    out["ok"] = True
    return out


def resolve_member(signal, region, other_regions, cp, g_hat,
                   rows=None, search_px=90.0, to_image=None):
    """
    격자선 한 줄짜리 부재 하나를 실루엣으로 되살린다.

    other_regions 에서 이 부재 **뒤에 있는 면**을 골라 배경 깊이로 쓴다.
    그림자 폭이 그 깊이차로 정해지므로, 배경을 잘못 고르면 예측 폭이
    틀려 가장자리를 아예 못 찾는다(그래서 조용히 실패한다 — 안전하다).
    """
    P = np.asarray(region.get("point_xyz"), float)
    if P is None or len(P) < 10:
        return {"ok": False, "reason": "점 부족"}
    uv = np.asarray(region.get("point_uv"), float)
    if to_image is not None:
        uv = np.asarray(to_image(uv), float)
    z_m = float(np.median(P[:, 2]))
    u_hint = float(np.median(uv[:, 0]))
    f = float(cp["f_px"]); b = float(cp["b_m"])
    cx = float(cp["cx_px"]); cy = float(cp["cy_px"])

    # 뒤 배경 — 이 부재보다 뚜렷하게 먼 면 중 가장 가까운 것
    zs = []
    for r in other_regions:
        if r is region or r.get("kind") not in ("plane_vertical",
                                                "plane_horizontal"):
            continue
        Q = np.asarray(r.get("point_xyz"), float)
        if Q is None or not len(Q):
            continue
        zb = float(np.median(Q[:, 2]))
        if zb > z_m * 1.15:
            zs.append(zb)
    if not zs:
        return {"ok": False, "reason": "뒤에 배경 면이 없어 그림자가 안 생긴다"}
    z_bg = min(zs)

    if rows is None:
        rows = np.linspace(uv[:, 1].min(), uv[:, 1].max(), 40)
    sh = find_shadow_edges(signal, rows, u_hint, z_m, z_bg, f, b,
                           search_px=search_px)
    if sh["n_found"] < 4:
        return {"ok": False, "reason": sh.get("reason") or "가장자리 부족",
                "shadow": sh, "z_member": round(z_m, 4),
                "z_background": round(z_bg, 4)}
    ax = axis_from_edges(sh["edges"], z_m, f, cx, cy, b, g_hat)
    ax.update({"shadow": {k: v for k, v in sh.items() if k != "edges"},
               "z_member": round(z_m, 4), "z_background": round(z_bg, 4)})
    return ax


# ============ 자체 검증 ============
if __name__ == "__main__":
    print("[식 ⑧] 가림 그림자 실루엣 검증")
    f, b, cx, cy = 826.0, 0.15, 634.5, 531.5
    g = np.array([0.0, 1.0, 0.0])
    Z = 1.70

    print("  그림자 폭 예측")
    for zb in (2.70, 2.20, 1.80):
        w = shadow_width_px(Z, zb, f, b)
        print(f"    배경 {zb:.2f}m → {w:.1f} px")

    print("  기울기 복원 — 가장자리에 ±2px 잡음을 주고")
    rng = np.random.default_rng(5)
    for tilt in (0.0, 0.5, 1.0, 2.0):
        v = np.linspace(50, 1000, 21)
        Y = (v - cy) * Z / f
        X0 = 0.20 + np.tan(np.radians(tilt)) * (Y - Y.mean())
        u = (X0 - b) * f / Z + cx + rng.normal(0, 2.0, len(v))
        r = axis_from_edges(np.column_stack([v, u]), Z, f, cx, cy, b, g)
        err = abs(r["theta_deg"] - tilt)
        print(f"    참 {tilt:4.2f}° → 복원 {r['theta_deg']:6.4f}°  "
              f"오차 {err:.4f}°  잔차 {r['rms_px']:.2f}px  "
              f"{'PASS' if (r['ok'] and err < 0.3) else 'FAIL'}")

    print("  가짜 가장자리를 막는가 — 몇 행에서 엉뚱한 곳을 잡았을 때")
    v = np.linspace(50, 1000, 21)
    u = (0.20 - b) * f / Z + cx + rng.normal(0, 1.0, len(v))
    u[[3, 9, 15]] += 40.0                       # 세 행만 크게 튀게
    r = axis_from_edges(np.column_stack([v, u]), Z, f, cx, cy, b, g)
    print(f"    튄 행 3개 → ok={r['ok']}  버린 점 {r.get('n_dropped')}개  "
          f"복원 {r['theta_deg']:.4f}° (참 0°)  "
          f"{'PASS' if abs(r['theta_deg']) < 0.3 else 'FAIL'}")
    u2 = (0.20 - b) * f / Z + cx + rng.normal(0, 8.0, len(v))
    r2 = axis_from_edges(np.column_stack([v, u2]), Z, f, cx, cy, b, g)
    print(f"    전체가 어지러우면 → ok={r2['ok']}  "
          f"{'PASS (되살리지 않음)' if not r2['ok'] else 'FAIL'}")
