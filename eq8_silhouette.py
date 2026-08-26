#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
[식 ⑧] 가로선이 남긴 그림자로 부재의 **옆 기울기** 를 되찾는다
========================================================================
문제
----
격자선이 한 줄만 걸린 부재는 각도의 **한 성분** 밖에 못 잰다. 한 줄에서
나온 3D 점은 그 레이저 평면 안에 놓이므로, 위·아래 화소의 깊이 차이로
잡히는 것은 평면 안 기울기뿐이고 평면에 수직인 기울기는 전혀 안 보인다
(eq7·pipeline_region 참조). 실측에서 실제로 2.00° 기운 부재가 0.0000°
로 나왔다.

그런데 가로선은 그 부재를 21번 가로지른다. 가로선 자체는 깊이를 못
주지만(eq7 이득 g=∞), 화소 **위치**는 정확하고, 부재가 뒤 배경에
드리우는 그림자는 화면에서 또렷하다.

왜 그림자가 생기나
-----------------
카메라는 조사기에서 기선 b 만큼 옆으로 떨어져 있다. 레이저가 부재에
막혀 못 간 자리를 카메라는 볼 수 있고, 그 자리는 어둡다. 부재 뒤로 폭

    w = f·b·(1/Z_부재 − 1/Z_배경)      [px]

인 띠가 비어 보인다. 이 예측이 실측과 맞는지가 이 방법의 안전장치다.

끊김은 부재가 아니라 **벽에 생긴 자국** 이다
-------------------------------------------
처음에는 끊김의 끝을 곧바로 부재 실루엣으로 봤는데 아니었다. roll=0
에서 가로선은 시차가 선과 나란해 깊이가 화소를 못 움직이므로, 부재 위
화소와 그 옆 벽 위 화소는 **끊기지 않고 이어진다**. 실제로 끊기는 것은
부재 뒤 벽에서 빛이 못 닿는 구간이고, 그래서 끊김은 부재에서 옆으로
정확히 w 만큼 밀려 있다. 조사기에서 벽으로의 중심투영이므로 폭은

    끊김[px] ≈ 2R·f/Z_부재 = 부재의 화소 지름

이 된다. 두 관계를 뒤집으면 끊김 하나에서 부재의 **지름과 중심을 동시에**
얻는다.

    R[px]     = 끊김폭 / 2
    u_중심(v) = u_끝(v) + 끊김폭/2 − w

중심투영은 두 축을 같은 배율로 늘이므로 **선의 방향은 보존된다** — 벽에서
읽은 자국으로 부재의 기울기를 잴 수 있는 근거가 이것이다. 높이마다 중심을
모아 부재 깊이에 놓고 직선을 맞추면 평면에 수직인 기울기가 나온다.

    X(v) = (u_중심 − c_x)·Z/f + b,   Y(v) = (v − c_y)·Z/f

부재가 옆으로 0.5° 기울면 2m 구간에서 8.5px 움직인다 — 검출 오차 ±2px
보다 훨씬 크다.

단서 두 가지 — 왜 둘 다 두는가
-----------------------------
같은 그림자를 서로 다른 자료에서 읽는다.

  ① member_edges_from_lines / axis_from_line_gaps
     **검출된 가로선 화소열**의 간격을 본다. 선 추적이 능선 중심을 부화소로
     잡아 주므로 정밀하고, 문턱값을 새로 고를 필요가 없다. 지름·중심이
     함께 나와 세그멘테이션 창 폭까지 이 값으로 정한다.

  ② find_shadow_edges / resolve_member
     **원본 신호**를 행마다 훑는다. ①은 추적기가 그 자리를 이어 놓았거나
     다른 끊김과 합쳐지면 아무것도 못 찾는데, 그때 ②가 받는다.

실측 동바리 3본에서 ①이 1본, ②가 2본을 살렸다. 하나만 두면 3본 중
1~2본이 판정보류로 남는다.

한계 — 반드시 지킬 것
--------------------
· 부재 뒤에 배경이 없으면 그림자가 안 생긴다. 화면 끝에 선 부재도
  그림자가 프레임 밖으로 나간다. 못 잡으면 **판정보류를 유지한다** —
  조용히 합격으로 바꾸지 않는다.
· 되돌린 중심이 부재 반폭 밖으로 나오면 남의 끊김을 문 것이므로 버린다.
  실측에서 44px 어긋난 끊김을 물어 없는 폭 154mm 를 지어냈다(참 48.6mm).
· 가장자리 잔차가 검출 오차(±2px) 수준을 넘으면 되살리지 않는다.
  안 그러면 없는 기울기 0.36° 를 지어낸다(참 0°).
· 부재 지름이 높이에 따라 변하면(테이퍼) 그만큼 기울기로 잘못 읽힌다.
  동바리·철근은 등단면이라 문제없지만, 조서에 근거를 남긴다.
========================================================================
"""
import os as _os
import importlib.util as _ilu

import numpy as np


def _load(name):
    spec = _ilu.spec_from_file_location(
        name, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                            f"{name}.py"))
    m = _ilu.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


_LSIG = _load("laser_signal")


def laser_signal(rgb, off=None, channel=None):
    """
    레이저만 남긴다 — 채널은 laser_signal 모듈이 이미지에서 판별한다.

    여기서는 음수를 자르지 **않는다**. 그림자는 신호가 없는 자리라
    0 쪽으로 눌러 버리면 가장자리가 뭉개진다.
    """
    return _LSIG.laser_signal(rgb, off=off, channel=channel, clip=False)


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


def member_edges_from_lines(lines_uv, u_hint, z_member, z_background,
                            f_px, b_m, search_px=140.0, min_gap_px=6.0,
                            center_tol_frac=1.5):
    """
    **검출된 가로선 화소열의 끊김** 에서 부재의 폭과 중심선을 되돌린다.

    끊김은 부재가 아니라 벽에 생긴 그림자다
    -------------------------------------
    처음에는 끊김의 끝을 곧바로 부재 실루엣으로 썼는데, 실측을 보니
    아니었다. 가로선이 부재를 지날 때 부재 위 화소와 그 옆 벽 위 화소는
    **끊기지 않고 이어진다** — roll=0 에서 가로선은 시차가 선과 나란해
    깊이가 화소를 움직이지 못하기 때문이다. 실제로 끊기는 것은 부재
    **뒤 벽**에서 조사기 빛이 못 닿는 구간이다.

    그래서 끊김은 부재에서 옆으로 밀려 있다. 조사기를 (−b,0,0), 부재를
    (X_p, Z_m), 벽을 Z_bg 라 두고 조사기에서 부재 중심을 지나는 광선이
    벽에 닿는 자리를 카메라로 보면

        Δu = f·b·(1/Z_m − 1/Z_bg) = w          (그림자 폭 공식과 같은 값)

    만큼 밀려 보인다. 끊김의 폭은 부재 지름이 조사기에서 벽으로 확대
    투영된 것이 다시 카메라에서 축소되어 결국

        끊김[px] ≈ 2R·f/Z_m = 부재의 화소 지름

    이 된다. 두 관계를 뒤집으면 **끊김 하나에서 부재의 지름과 중심을
    동시에** 얻는다.

        R[px]      = 끊김폭 / 2
        u_중심(v)  = u_왼끝(v) + 끊김폭/2 − w          (b > 0 일 때)

    지름을 가정하지 않는다는 점이 중요하다. Ø48.6 동바리든 각재든 철근이든
    같은 식이 성립하므로, 다른 현장·다른 부재에서도 그대로 쓴다.

    자기 확인 방지
    -------------
    u_hint 는 **높이에 무관한 상수** 하나다(어느 부재의 끊김인지 고를
    때만 쓴다). 높이에 따른 변화는 전부 화소에서 온다. 되돌린 중심이
    u_hint 에서 부재 반폭 이상 벗어나면 남의 끊김을 잡은 것이므로 버린다 —
    실측에서 둘째 동바리가 44px 어긋난 끊김을 물었고, 그대로 두면 없는
    폭 154mm 를 지어냈다(참 48.6mm).

    Returns
    -------
    dict — edges (M,2)[v, u_중심], n_found, radius_px, gap_px, offset_px
    """
    w = shadow_width_px(z_member, z_background, f_px, b_m)
    if w is None or not np.isfinite(w) or abs(w) < 1e-6:
        return {"n_found": 0, "edges": np.empty((0, 2)),
                "reason": "부재와 배경의 깊이차가 없어 그림자가 안 생긴다"}
    near = 0 if w > 0 else 1        # 그림자가 밀리는 쪽 = 끊김의 부재 쪽 끝

    # ── 추적기가 끊는 폭을 이미지 자체에서 보정한다 ──
    # 능선 추적은 신호가 끊기는 자리마다 부분적으로 켜진 화소를 버려서,
    # 끊김을 실제보다 조금 넓게 잰다. 그 양은 선 굵기에서 오므로 같은
    # 이미지 안의 **작은 끊김**(다른 선이 가로지른 자리)으로 잴 수 있다.
    # 보정 없이 쓰면 실측에서 지름이 10% 부풀었다(26.8mm, 참 24.3mm).
    small = []
    for uv in lines_uv.values():
        A = np.asarray(uv, dtype=float).reshape(-1, 2)
        if len(A) < 8:
            continue
        du = np.diff(np.sort(A[:, 0]))
        small.extend(du[(du > 1.2) & (du < float(min_gap_px))].tolist())
    drop_px = float(np.median(small)) if len(small) >= 8 else 0.0

    rows = []                       # (v, u_center, gap, u_near)
    for uv in lines_uv.values():
        A = np.asarray(uv, dtype=float).reshape(-1, 2)
        if len(A) < 8:
            continue
        A = A[np.argsort(A[:, 0])]
        u, v = A[:, 0], A[:, 1]
        du = np.diff(u)
        best = None
        for k in np.nonzero(du > float(min_gap_px))[0]:
            g = float(du[k])
            u_near = float(u[k]) if near == 0 else float(u[k + 1])
            # 끊김폭이 곧 화소 지름이므로 중심을 바로 되돌릴 수 있다.
            u_c = u_near + (0.5 * g if near == 0 else -0.5 * g) - w
            d = abs(u_c - u_hint)
            if d > search_px:
                continue
            if best is None or d < best[0]:
                best = (d, float(0.5 * (v[k] + v[k + 1])), u_c, g, u_near)
        # 되돌린 중심이 부재 반폭 밖이면 남의 끊김이다.
        if best is not None and best[0] <= center_tol_frac * 0.5 * best[3]:
            rows.append(best[1:])
    if len(rows) < 4:
        return {"n_found": len(rows), "edges": np.empty((0, 2)),
                "expected_offset_px": round(float(w), 2),
                "reason": f"부재의 끊김을 확인한 가로선이 {len(rows)}개뿐"}

    C = np.asarray(rows, float)     # [v, u_center, gap, u_near]
    gap = float(np.median(C[:, 2]))
    # 보정은 끊김의 30% 를 넘지 않을 때만 — 그보다 크면 작은 끊김을
    # 잘못 잰 것이므로 손대지 않는다.
    corr = drop_px if 0.0 < drop_px <= 0.30 * gap else 0.0
    return {"n_found": int(len(C)), "edges": C[:, :2],
            "radius_px": round((gap - corr) / 2.0, 2),
            "dropout_px": round(corr, 2),
            "gap_px": round(gap, 2),
            "gap_spread_px": round(float(np.percentile(C[:, 2], 90)
                                         - np.percentile(C[:, 2], 10)), 2),
            "center_offset_px": round(float(np.median(C[:, 1]) - u_hint), 2),
            "expected_offset_px": round(float(w), 2),
            "side": ("좌" if near == 0 else "우")}


def axis_from_line_gaps(lines_uv, u_hint, z_member, z_background,
                        f_px, cx_px, cy_px, b_m, g_hat,
                        search_px=90.0, max_rms_px=2.5, **kw):
    """
    가로선 화소열의 끊김으로 레이저 평면에 **수직인** 기울기를 잰다.

    member_edges_from_lines 가 끊김에서 되돌린 **부재 중심선**을 높이별로
    받아, 부재 깊이에 놓고 직선을 맞춘다(axis_from_edges 와 같은 계산).
    그림자 경계는 조사기에서 벽으로의 중심투영이라 두 축이 같은 배율로
    늘어나므로 **방향은 보존된다** — 그래서 벽에서 읽은 자국으로 부재의
    기울기를 잴 수 있다. 원본 신호를 훑는 resolve_member 와 같은 양을
    재므로 둘 다 되면 서로 검산이 된다.
    """
    sh = member_edges_from_lines(lines_uv, u_hint, z_member, z_background,
                                 f_px, b_m, search_px=search_px, **kw)
    if sh["n_found"] < 4:
        return {"ok": False, "reason": sh.get("reason") or "가장자리 부족",
                "shadow": {k: v for k, v in sh.items() if k != "edges"}}
    out = axis_from_edges(sh["edges"], z_member, f_px, cx_px, cy_px, b_m,
                          g_hat, max_rms_px=max_rms_px)
    out["shadow"] = {k: v for k, v in sh.items() if k != "edges"}
    out["n_lines"] = int(sh["n_found"])
    out["cue"] = "가로선 화소 끊김"
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
