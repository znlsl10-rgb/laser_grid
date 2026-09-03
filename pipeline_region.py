#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pipeline_region.py — 이미지 1장 → 영역별 품질검측
========================================================================
현장 사진 한 장에 벽·바닥·동바리·철근이 함께 들어와도, 영역을 나눠
각 부재에 맞는 검측식을 적용한다.

  1 shot = (rgb_off, rgb_on, IMU ĝ, calib)
   ├─[A ] 선검출        rgb_on  → {lid: [(u,v)…]}
   ├─[C ] 영역분할      rgb_off → label_map / point_labels
   ├─[eq1] 삼각측량             → (X,Y,Z) 조사기 좌표계
   ├─[eq5] 영역 할당            → 침식·깊이불연속 제거 + 의미x기하 융합
   └─ 영역별 검측
       벽·거푸집·조적  → eq2 평면(TLS) → eq3 수직도(중력) + eq4 평활도(면내)
       바닥·슬래브     → eq2 평면(TLS) → eq3 수평도(중력) + eq4 평활도(면내)
       동바리·기둥·철근 → eq2 축(PCA)  → eq3 축 수직도(중력)  [평활도 N/A]
       전 영역        → eq5 불확실도 → KCS 합격/기준초과/측정불가

기존 inspection.py 는 STATIONS 딕셔너리에 "이 스테이션은 수직도" 라고
적어두고 화면 전체를 단일 평면으로 적합했다. 이 모듈은 그 결정을 설정이
아니라 인지 결과에서 가져온다.

실행 (합성 씬 자체검증):
  python3 pipeline_region.py
========================================================================
"""
import numpy as np
import importlib.util as _ilu, os as _os


def _load(name):
    spec = _ilu.spec_from_file_location(
        name, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), f"{name}.py"))
    m = _ilu.module_from_spec(spec); spec.loader.exec_module(m)
    return m


_EQ2 = _load("eq2_plane_fit")
_EQ3 = _load("eq3_orientation")
_EQ4 = _load("eq4_flatness_line")
_EQ5 = _load("eq5_region_assign")
_EQ6 = _load("eq6_straightedge")
_EQ7 = _load("eq7_laser_plane")
_EQ8 = _load("eq8_silhouette")
_SEG = _load("C_영역분할")



# =====================================================================
# 영역 1개 검측
# =====================================================================
def _defects_in_image(fd, camera_params):
    """
    검출된 요철 덩어리를 화면 좌표로 옮긴다.

    eq4 는 요철을 **면내 좌표(u,v)** 로 돌려준다. 평활도 계산에는 그것이
    맞지만, 조서를 읽는 사람은 "벽 어디가 튀어나왔나" 를 그림에서 보고
    싶어 한다. 면내 좌표를 3D 로 되돌린 뒤 카메라로 투영한다.

        P = origin + u·e1 + v·e2        (w 는 mm 단위라 투영에 영향이 없다)
        u_px = f·(X − b)/Z + c_x,  v_px = f·Y/Z + c_y

    투영식에 −b 가 들어가는 것은 카메라가 조사기에서 X 로 b 만큼 떨어져
    있기 때문이다(eq1 과 같은 규약).
    """
    clusters = fd.get("verified_clusters") or []
    if not clusters or fd.get("basis") is None:
        return []
    e1, e2, _ = fd["basis"]
    origin = np.asarray(fd["origin"], float)
    # cand_points_uv 가 클러스터 point_idx 의 기준 배열이다.
    cand_uv = np.asarray(fd.get("cand_points_uv")
                         if fd.get("cand_points_uv") is not None
                         else fd.get("defect_points_uv"))
    f = float(camera_params["f_px"]); b = float(camera_params["b_m"])
    cx = float(camera_params["cx_px"]); cy = float(camera_params["cy_px"])

    out = []
    for c in clusters:
        idx = np.asarray(c["point_idx"], dtype=int)
        if len(cand_uv) and idx.max() < len(cand_uv):
            uv = cand_uv[idx]
        else:
            # 기준 배열을 못 얻으면 덩어리 중심 하나만 표시한다.
            # 잘못된 배열을 인덱싱해 엉뚱한 위치를 그리는 것보다 낫다.
            uv = np.array([c["center_xy"]], dtype=float)
        P = origin + np.outer(uv[:, 0], e1) + np.outer(uv[:, 1], e2)
        Z = P[:, 2]
        ok = Z > 1e-6
        if not ok.any():
            continue
        P = P[ok]; Z = Z[ok]
        up = f * (P[:, 0] - b) / Z + cx
        vp = f * P[:, 1] / Z + cy
        out.append({
            "center_px": [round(float(up.mean()), 1), round(float(vp.mean()), 1)],
            "bbox_px": [round(float(up.min()), 1), round(float(vp.min()), 1),
                        round(float(up.max()), 1), round(float(vp.max()), 1)],
            "depth_mm": round(float(c["depth_mm"]), 2),
            # 폭은 이 덩어리의 3D 점에서 직접 잰다. 클러스터가 들고 온
            # 값은 면내 좌표계 기준이라 덩어리마다 같은 수가 나오곤 했다
            # (실측: 요철 둘 다 3004.9mm — 부재 전체 크기였다).
            "extent_mm": round(float(max(np.ptp(P[:, 0]), np.ptp(P[:, 1]))
                                     * 1000.0), 1),
            "n_points": int(c["n_points"]),
            "z_m": round(float(np.median(Z)), 3),
            # 3D 산점도가 요철 위치를 찍으려면 조사기 좌표가 필요하다.
            # 화소 좌표만으로는 깊이를 되돌릴 수 없다.
            "center_xyz": [round(float(P[:, 0].mean()), 5),
                           round(float(P[:, 1].mean()), 5),
                           round(float(P[:, 2].mean()), 5)],
        })
    out.sort(key=lambda d: -abs(d["depth_mm"]))
    return out


def resolve_single_plane_members(result, rgb, camera_params, g_hat,
                                 to_image=None):
    """
    격자선 한 줄만 걸려 판정보류로 막힌 부재를, 가림 그림자로 되살린다.

    세로선 한 줄은 레이저 평면 **안**의 기울기만 준다. 가로선은 그 부재를
    21번 가로지르지만 깊이를 못 준다 — 그런데 부재가 뒤 배경에 드리우는
    **그림자**의 시작점이 곧 부재의 실루엣 가장자리이고, 부재 깊이는 이미
    세로선에서 알고 있다. 그래서 가장자리를 그 깊이에 놓으면 평면에
    수직인 성분까지 나온다 (eq8).

    두 성분은 서로 직교한다 — 세로선은 레이저 평면 안, 실루엣은 그 평면에
    수직인 방향(등깊이면)에서 잰다. 작은 각도에서는 제곱합으로 합친다.

        θ_total ≈ √(θ_평면안² + θ_평면수직²)

    되살아나지 않으면(뒤에 배경이 없거나 그림자 폭이 예측과 안 맞으면)
    판정보류를 **그대로 둔다**. 조용히 합격으로 바꾸지 않는다.

    camera_params 와 to_image 는 **이미지 화소 기준** 이어야 한다. 검측
    좌표는 센서 기준이고 캡처가 180° 돌아 있기도 해서, 그대로 넣으면
    엉뚱한 자리를 훑는다. to_image 가 그 변환을 맡는다.
    """
    if rgb is None:
        return 0
    regs = result.get("regions") or []
    targets = [r for r in regs
               if r.get("single_plane") and r.get("kind") == "axis_vertical"
               and r.get("status") == "measured"]
    if not targets:
        return 0
    sig = _EQ8.laser_signal(rgb)
    n_ok = 0
    for r in targets:
        try:
            res = _EQ8.resolve_member(sig, r, regs, camera_params, g_hat,
                                      to_image=to_image)
        except Exception as e:
            res = {"ok": False, "reason": f"실패: {e}"}
        r["silhouette"] = res
        if not res.get("ok"):
            continue
        detail = (f"부재가 배경에 드리운 그림자에서 실루엣 가장자리를 "
                  f"{res['n_points']}개 높이에서 찾아(예측 폭 "
                  f"{res['shadow']['expected_width_px']}px)")
        _apply_lateral(r, res, "가림 그림자 실루엣 (eq8)", detail)
        n_ok += 1
    return n_ok


def _apply_lateral(r, res, resolved_by, detail):
    """
    되찾은 **평면수직 성분**을 이미 있는 평면안 성분과 합쳐 판정을 고친다.

    두 성분은 서로 직교한다 — 세로선은 레이저 평면 안, 이쪽 단서는 그
    평면에 수직인 방향(등깊이면)에서 잰다. 작은 각도에서 제곱합이다.

        θ_전체 ≈ √(θ_평면안² + θ_평면수직²)

    합치고 나면 더 이상 "한 성분만 본" 값이 아니므로 single_plane 을
    내리고 정상 판정을 낸다.
    """
    th_plane = float(r.get("theta_deg_plane_only")
                     if r.get("theta_deg_plane_only") is not None
                     else (r.get("theta_deg") or 0.0))
    th_lat = float(res["theta_deg"])
    th = float(np.hypot(th_plane, th_lat))
    r["theta_deg_plane_only"] = round(th_plane, 4)
    r["theta_deg_lateral"] = round(th_lat, 4)
    r["theta_deg"] = round(th, 4)
    r["single_plane"] = False
    kcs = _EQ5.KCS_CLASS.get(r["class"], r["class"])
    j = _EQ3.judge_kcs(th, kcs, member_length_m=None)
    j["resolved_by"] = resolved_by
    j["theta_deg_plane_only"] = round(th_plane, 4)
    j["theta_deg_lateral"] = round(th_lat, 4)
    j["note"] = (
        f"세로선 한 줄로는 레이저 평면 안 성분({th_plane:.4f}°)밖에 못 "
        f"쟀다. {detail} 평면에 수직인 성분 {th_lat:.4f}° 를 되찾았고, "
        f"둘을 제곱합으로 합쳐 {th:.4f}° 다. 직선 잔차 {res['rms_px']}px, "
        f"세로 구간 {res['span_m']}m.")
    r["judge"] = j
    return j


def _background_depth(regs, region, camera_params, u_px, v_px):
    """
    이 부재 **뒤쪽 바로 그 자리**의 배경 깊이를 구한다.

    예전에는 배경 면의 깊이 중앙값 하나를 썼는데, 벽이 조금이라도 비스듬
    하면 부재마다 뒤 거리가 달라 그림자 폭 예측이 틀어진다. 실측에서
    둘째 동바리의 끊김이 그 때문에 44px 어긋나 통째로 버려졌다.

    면은 이미 적합돼 있으므로, 부재가 보이는 화소의 광선을 그 면과 만나게
    하면 바로 그 자리의 깊이가 나온다.

        Z = −(d + n_x·b) / (n_x·û + n_y·v̂ + n_z)
    """
    f = float(camera_params["f_px"]); b = float(camera_params["b_m"])
    cx = float(camera_params["cx_px"]); cy = float(camera_params["cy_px"])
    uh = (float(u_px) - cx) / f
    vh = (float(v_px) - cy) / f
    out = []
    for o in regs:
        if o is region or o.get("kind") not in ("plane_vertical",
                                                "plane_horizontal"):
            continue
        pl = o.get("plane")
        if not pl or len(pl) != 4:
            continue
        n = np.asarray(pl[:3], float); dd = float(pl[3])
        den = n[0] * uh + n[1] * vh + n[2]
        if abs(den) < 1e-9:
            continue
        z = -(dd + n[0] * b) / den
        if np.isfinite(z) and 0.05 < z < 60.0:
            out.append(float(z))
    return out


def measure_member_silhouettes(result, aux_lines_uv, camera_params):
    """
    선형 부재(동바리·철근)의 **실루엣 가장자리**를 가로선 끊김에서 잰다.

    왜 분할보다 먼저 재는가
    ----------------------
    가로선 화소를 어느 부재에 줄지 가르려면 부재가 이미지에서 얼마나
    넓은지 알아야 한다. 예전에는 "축에서 옆으로 50mm 안" 이라는 상수를
    썼는데, 이 값은 Ø48.6 동바리에서만 맞고 각재·기둥·철근에서 전부
    틀린다. 좁게 잡으면 부재 화소가 뒤 벽으로 새고, 넓게 잡으면 벽
    화소를 부재가 훔친다 — 그림에서 가로선이 부재를 뚫고 지나가거나,
    부재가 실제보다 굵어 보인다.

    폭은 상수로 정할 게 아니라 **재는** 것이다. 가로선이 부재를 지나는
    자리에는 부재가 배경에 드리운 그림자만큼 끊김이 있고, 그 끊김의 부재
    쪽 끝이 그 높이에서의 실루엣 가장자리다(eq8). 그 가장자리와 세로선이
    맞은 자리 사이의 거리가 곧 관측된 반폭이다.

        반폭[px] = |u_세로선 − u_가장자리|

    한쪽만 보이므로 반대쪽은 대칭으로 잡는다. 원통 단면이 이미 풀린
    부재(격자선 2줄 이상)는 반지름을 알고 있으니 그쪽을 먼저 쓴다.

    둘 다 없으면 폭을 모른다 — 그때만 보수적인 기본값을 쓰고 그 사실을
    결과에 남긴다(width_source="기본값"). 조용히 상수를 쓰지 않는다.
    """
    regs = result.get("regions") or []
    axials = [r for r in regs if r.get("kind") == "axis_vertical"
              and r.get("status") == "measured"]
    if not axials:
        return 0
    f = float(camera_params["f_px"]); b = float(camera_params["b_m"])
    n_meas = 0
    for r in axials:
        P = np.asarray(r.get("point_xyz"), float).reshape(-1, 3)
        uv = np.asarray(r.get("point_uv"), float).reshape(-1, 2)
        if len(P) < 10 or len(uv) < 10:
            continue
        z_m = float(np.median(P[:, 2]))
        u_v = float(np.median(uv[:, 0]))
        r["z_median_m"] = round(z_m, 4)

        # (1) 단면이 풀렸으면 반지름이 곧 반폭이다.
        rad_mm = float((r.get("axis") or {}).get("radius_est_mm") or 0.0)
        if rad_mm > 1.0:
            r["half_width_m"] = rad_mm / 1000.0
            r["width_source"] = "단면 반지름"
            n_meas += 1
            continue

        # (2) 가로선 끊김에서 실루엣 가장자리를 재 본다.
        got = False
        if aux_lines_uv:
            zs = [z for z in _background_depth(
                      regs, r, camera_params, u_v, float(np.median(uv[:, 1])))
                  if z > z_m * 1.15]
            if zs:
                try:
                    sh = _EQ8.member_edges_from_lines(
                        aux_lines_uv, u_v, z_m, min(zs), f, b)
                except Exception as e:
                    sh = {"n_found": 0, "reason": f"실패: {e}"}
                r["silhouette_edges"] = sh
                if sh.get("n_found", 0) >= 4:
                    # 끊김폭이 곧 부재의 화소 지름이다(eq8 유도 참고).
                    hw_px = float(sh.get("radius_px") or 0.0)
                    if 2.0 <= hw_px <= 0.25 * f:
                        r["half_width_m"] = hw_px * z_m / f
                        r["width_source"] = "가로선 끊김 폭"
                        # 세로선이 부재의 한가운데를 맞으리라는 보장이 없다.
                        # 끊김에서 되돌린 중심을 쓰면 창을 부재에 맞춰
                        # 놓을 수 있다 — 가장자리에 맞은 부재일수록 크게
                        # 다르다.
                        E = np.asarray(sh["edges"], float)
                        r["center_u_px"] = round(float(np.median(E[:, 1])), 2)
                        r["center_offset_px"] = round(
                            float(np.median(E[:, 1]) - u_v), 2)
                        got = True
                        n_meas += 1
        if got:
            continue

        # (3) 못 쟀다 — 일단 비워 두고 아래에서 채운다.
        r["half_width_m"] = None
        r["width_source"] = None

    # 못 잰 부재는 **같은 장면에서 잰 부재의 폭**을 빌린다. 한 현장의
    # 동바리는 대개 같은 규격이고, 이미지와 무관한 상수보다 훨씬 낫다.
    # 그마저 없을 때만 보수적 기본값을 쓰고 그 사실을 남긴다.
    done = [r["half_width_m"] for r in axials if r.get("half_width_m")]
    for r in axials:
        if r.get("half_width_m"):
            continue
        if done:
            r["half_width_m"] = float(np.median(done))
            r["width_source"] = "같은 장면의 다른 부재 폭"
        else:
            r["half_width_m"] = 0.05
            r["width_source"] = "기본값(폭 미측정)"
    return n_meas


def resolve_by_line_gaps(result, aux_lines_uv, camera_params, g_hat):
    """
    검출된 **가로선 화소의 끊김**으로 옆 기울기를 되찾는다.

    가로선은 깊이를 못 준다. 그래서 오래도록 검측에서 빼 왔는데, 못 주는
    것은 깊이일 뿐 화소 위치가 아니다. 가로선이 동바리를 지나는 자리에는
    부재가 배경에 드리운 그림자만큼 끊김이 생기고, 그 끊김의 부재 쪽 끝이
    그 높이에서의 실루엣 가장자리다. 높이를 따라 이 가장자리가 옆으로
    밀리면 그것이 레이저 평면에 수직인 기울기 — 세로선 한 줄이 원리적으로
    못 보던 성분이다. 깊이는 세로선에서 이미 알므로 화소를 미터로 되돌릴
    수 있다.

    ── 하지 않는 것 ──
    "가로선이 부재를 덮은 구간의 **한가운데**" 를 쓰고 싶어지지만, 그건
    안 된다. roll=0 에서 가로선은 시차가 선과 나란해 깊이를 못 주므로,
    부재 위 화소와 뒤 벽 위 화소가 이어져 보인다. 부재의 반대쪽 가장자리
    (그림자가 없는 쪽)는 신호에 아무 흔적을 남기지 않는다. 그래서 폭을
    창(window)으로 잘라 중앙을 잡으면, 그 창은 **세로선에서 나온 축**을
    중심으로 잡은 것이라 중앙이 축을 그대로 되돌려 준다 — 없는 정보를
    지어내는 자기 확인이다. 실제로 그렇게 짜 봤더니 세 본 모두 잔차 0.0px
    에 θ=0.0000° 가 나왔는데, 측정이 아니라 항등식이었다.

    관측 가능한 가장자리는 **그림자 쪽 하나뿐**이고, 여기서는 그것만
    쓴다. u_hint 는 어느 부재를 볼지 고르는 상수 하나일 뿐이라 높이에
    따른 변화를 주지 못한다.

    camera_params 는 **검측(센서) 좌표계** 기준이어야 한다 — aux_lines_uv
    가 그 좌표계이기 때문이다.
    """
    if not aux_lines_uv:
        return 0
    regs = result.get("regions") or []
    targets = [r for r in regs
               if r.get("single_plane") and r.get("kind") == "axis_vertical"
               and r.get("status") == "measured"]
    if not targets:
        return 0
    f = float(camera_params["f_px"]); b = float(camera_params["b_m"])
    cx = float(camera_params["cx_px"]); cy = float(camera_params["cy_px"])
    n_ok = 0
    for r in targets:
        P = np.asarray(r.get("point_xyz"), float).reshape(-1, 3)
        uv = np.asarray(r.get("point_uv"), float).reshape(-1, 2)
        if len(P) < 10 or len(uv) < 10:
            r["line_gap"] = {"ok": False, "reason": "점 부족"}
            continue
        z_m = float(np.median(P[:, 2]))
        # 뒤 배경 — 부재가 보이는 바로 그 자리에서 잰다(벽이 비스듬해도
        # 부재마다 맞는 거리가 나온다).
        zs = [z for z in _background_depth(
                  regs, r, camera_params, float(np.median(uv[:, 0])),
                  float(np.median(uv[:, 1]))) if z > z_m * 1.15]
        if not zs:
            r["line_gap"] = {"ok": False,
                             "reason": "뒤에 배경 면이 없어 그림자가 안 생긴다"}
            continue
        try:
            res = _EQ8.axis_from_line_gaps(
                aux_lines_uv, float(np.median(uv[:, 0])), z_m, min(zs),
                f, cx, cy, b, g_hat)
        except Exception as e:
            res = {"ok": False, "reason": f"실패: {e}"}
        res["z_member"] = round(z_m, 4)
        res["z_background"] = round(float(min(zs)), 4)
        r["line_gap"] = res
        if not res.get("ok"):
            continue
        sh = res.get("shadow") or {}
        # 예측 폭은 find_shadow_edges 쪽 결과에만 있는 값이라, 이 경로
        # (member_edges_from_lines)에서는 없는 게 정상이다. 그대로 찍으면
        # 조서에 "예측 Nonepx" 가 실린다.
        want = sh.get("expected_width_px")
        detail = (f"가로선 {res['n_lines']}줄이 부재를 지나며 생긴 끊김"
                  f"(폭 중앙값 {sh.get('gap_px')}px"
                  + (f", 예측 {want}px" if want else "")
                  + ")의 부재 쪽 끝을 높이별로 뽑아")
        _apply_lateral(r, res, "가로선 화소 끊김 (eq8)", detail)
        n_ok += 1
    return n_ok


def _ray_cylinder(uh, vh, b, axis_pt, axis_dir, radius):
    """
    카메라 광선과 원통면의 **앞쪽** 교점 깊이 Z.

    광선  P(Z) = (b, 0, 0) + Z·(û, v̂, 1)
    원통  축을 지나는 직선에서 거리가 radius 인 점들

    축 방향 성분을 빼고 나면 평면 위의 원-직선 교차가 된다.

        |O⊥ + Z·D⊥|² = R²
        aZ² + 2βZ + γ = 0,  a=|D⊥|², β=O⊥·D⊥, γ=|O⊥|²−R²

    두 근 중 작은 양수(카메라에 가까운 쪽)가 보이는 앞면이다. 광선이
    원통을 스치지 못하면(판별식 < 0) NaN 을 돌려주고, 호출부가 그 화소를
    이 부재에 배정하지 않는다.
    """
    d = np.asarray(axis_dir, float)
    O = np.array([float(b), 0.0, 0.0]) - np.asarray(axis_pt, float)
    D = np.column_stack([uh, vh, np.ones_like(uh)])
    O_par = float(O @ d)
    Op = O - O_par * d                      # (3,)
    D_par = D @ d                           # (N,)
    Dp = D - D_par[:, None] * d[None, :]    # (N,3)
    A = np.einsum("ij,ij->i", Dp, Dp)
    B = Dp @ Op
    C = float(Op @ Op) - float(radius) ** 2
    disc = B * B - A * C
    # 실루엣에 스치는 광선은 버린다. 접점 근처에서는 두 근이 붙어 깊이가
    # 조금만 흔들려도 크게 튄다 — 실측에서 그 때문에 원통 반지름(25.8mm)
    # 보다 깊은 30.2mm 짜리 점이 생겼다. 기하상 앞면의 깊이 폭은 R 을
    # 넘을 수 없으므로, 그런 값이 나온다는 것 자체가 스침의 증거다.
    guard = (0.20 * float(radius) * np.sqrt(np.maximum(A, 0.0))) ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        sq = np.sqrt(np.maximum(disc, 0.0))
        z1 = (-B - sq) / A
        z2 = (-B + sq) / A
    Z = np.where(z1 > 0.05, z1, z2)
    Z = np.where((disc >= guard) & (A > 1e-12) & (Z > 0.05), Z, np.nan)
    return Z


def place_aux_points(results, aux_lines_uv, camera_params, margin_m=0.12):
    """
    깊이를 못 주는 선(굴리지 않은 가로선)의 화소를 **이미 맞춘 면 위에**
    올려 3D 위치를 준다.

    왜 필요한가
    ----------
    가로선은 원리적으로 깊이를 못 준다(eq7 이득 g=∞). 그렇다고 결과
    그림에서 빼 버리면 "격자가 실제로 어디에 걸렸는지" 가 안 보인다.
    검출은 분명히 됐고(21/21), 화소 위치는 정확하다 — 없는 것은 그 점의
    **거리** 뿐이다.

    그래서 거리는 검측이 이미 구한 면에서 빌려 온다. 화소 하나는 카메라
    광선 하나를 정하고, 그 광선이 어느 면과 먼저 만나는지는 풀 수 있다.
    레이저는 가장 앞에 있는 면에 맺히므로 **가장 가까운 교점** 이 답이다.

        P = Z·(û, v̂, 1) + (b, 0, 0)  가 면 위    →  Z 를 푼다

    반드시 지켜야 할 것 — 이 점은 **측정값이 아니라 유도값** 이다. 면에서
    빌려 온 거리이므로 그 면에 대한 잔차가 정의상 0 이고, 평활도·수직도
    계산에 넣으면 결과를 스스로 확인하는 꼴이 된다. 그래서 검측에는
    절대 넣지 않고 aux_ 접두어로 따로 들고 다니며, 조서와 그림에도
    "투영" 이라고 표시한다.
    """
    if not aux_lines_uv:
        return 0
    f = float(camera_params["f_px"]); b = float(camera_params["b_m"])
    cx = float(camera_params["cx_px"]); cy = float(camera_params["cy_px"])

    # 후보 면을 모은다. 두 종류다.
    #
    #  (1) 면 부재(벽·바닥) — 이미 적합된 평면을 그대로 쓴다.
    #  (2) 선형 부재(동바리·철근) — 평면이 없다. 예전에는 그래서 후보에서
    #      아예 빠졌고, 그 결과 **동바리를 가로지르는 가로선 화소가 뒤의
    #      벽으로 던져져 벽 색으로 칠해졌다**. 눈으로 보면 가로선이
    #      동바리를 뚫고 지나간 것처럼 보인다 — 분할이 틀린 것이다.
    #
    #      동바리에도 면을 하나 줄 수 있다. 세로선에서 이미 잡힌 측정점은
    #      원통의 **앞면**에 놓여 있으므로, 축 방향 d 를 품고 시선에
    #      수직인 접평면을 세우고 그 위치를 측정점 평균으로 잡으면 된다.
    #
    #          n = normalize(ẑ − (ẑ·d)d),   t = mean(n·P_측정)
    #
    #      지름을 몰라도 된다 — 앞면 깊이를 가정이 아니라 측정에서 가져오기
    #      때문이다. 곡률 때문에 옆으로 갈수록 조금씩 앞서지만(최대 반지름
    #      정도), 이 점은 어차피 측정값이 아니라 표시용 유도값이다.
    cands = []
    for k, r in enumerate(results):
        if r.get("status") != "measured":
            continue
        P = np.asarray(r.get("point_xyz"), float).reshape(-1, 3)
        if len(P) < 3:
            continue
        pl = r.get("plane")        # (a, b, c, d) — aX+bY+cZ+d = 0
        ax = r.get("axis") or {}
        if pl and len(pl) == 4:
            cands.append({"idx": k, "n": np.asarray(pl[:3], float),
                          "d": float(pl[3]), "shape": "plane",
                          "lo": P.min(axis=0) - margin_m,
                          "hi": P.max(axis=0) + margin_m})
        elif ax.get("direction") is not None:
            dv = np.asarray(ax["direction"], float)
            nv = np.linalg.norm(dv)
            if nv <= 1e-9:
                continue
            dv = dv / nv
            n = np.array([0.0, 0.0, 1.0]) - dv[2] * dv     # 시선의 축직교 성분
            nn = np.linalg.norm(n)
            if nn <= 1e-6:                                  # 축이 시선과 나란함
                continue
            n = n / nn
            t = float(np.mean(P @ n))
            c0 = np.asarray(ax.get("centroid", P.mean(axis=0)), float)
            # 옆으로 얼마까지 이 부재로 볼 것인가 —
            # measure_member_silhouettes 가 **재 놓은** 반폭을 쓴다.
            # (상수를 쓰면 부재 지름이 바뀌는 순간 분할이 틀린다.)
            hw = float(r.get("half_width_m") or 0.0)
            if hw <= 1e-4:
                hw = float(ax.get("radius_est_mm") or 0.0) / 1000.0
            if hw <= 1e-4:
                hw = 0.05
            # ── 접평면이 아니라 **원통면** 에 올린다 ──
            # 세로선 한 줄은 연직 원통을 곧은 직선으로 자른다(연직면 ∩
            # 연직 원통 = 직선). 그래서 세로선만으로는 둥근 게 보일 수가
            # 없다 — 실측 u 편차 0.0000px 로 확인했다. 둥근 것은 부재를
            # 가로지르는 **가로선** 이 보여 줘야 하는데, 그 점을 시선에
            # 수직인 접평면에 올려 놓고 있었다. 그러면 정의상 깊이가
            # 일정해져 원통이 판판한 띠가 된다(실측 Z 폭 0.000mm).
            #
            # 지름은 이미 재 놓았으므로(가로선 끊김 폭) 광선과 원통을
            # 제대로 만나게 할 수 있다.
            #
            #     |O⊥ + Z·D⊥|² = R²      (축 성분을 뺀 나머지)
            #
            # 두 근 중 **가까운 쪽** 이 카메라가 보는 앞면이다. 이러면
            # 가로선 점이 부재를 감싸고, 가장자리에서 깊이가 최대 R 만큼
            # 되돌아온다 — 그게 실제 레이저가 맺힌 모습이다.
            lat = np.cross(dv, n)
            nl = np.linalg.norm(lat)
            lat = lat / nl if nl > 1e-6 else None
            z0 = float(np.median(P[:, 2]))
            off = float(r.get("center_offset_px") or 0.0)
            # 세로선이 맞은 자리는 **표면** 이다. 중심은 옆으로 a 만큼,
            # 깊이로 √(R²−a²) 만큼 뒤에 있다.
            a_m = (off * z0 / f) if lat is not None else 0.0
            c0 = np.asarray(ax.get("centroid", P.mean(axis=0)), float)
            if lat is not None and abs(off) > 0.5:
                c0 = c0 + lat * a_m
            R = hw if hw > 1e-4 else 0.0
            # 원통으로 올리려면 **지름과 중심을 둘 다** 이 부재에서 재야
            # 한다. 지름만 빌려 오고 중심을 세로선 자리로 가정하면, 세로선이
            # 부재 가장자리를 맞은 경우 원통이 옆으로 밀려 앉아 광선이 먼
            # 쪽 면을 스친다 — 실측에서 앞면 깊이 폭이 반지름(25.8mm)보다
            # 큰 30.2mm 로 나왔다. 기하상 불가능한 값이니 그 배치가 틀린
            # 것이다. 못 재면 접평면 그대로 두고 그 사실을 남긴다.
            own = (r.get("width_source") == "가로선 끊김 폭"
                   and r.get("center_offset_px") is not None)
            use_cyl = bool(R > 1e-3 and R < 0.5 and own)
            r["aux_surface"] = (
                "원통면 (지름·중심 실측)" if use_cyl
                else ("접평면 (중심 미측정 — 둥글게 그리면 지어내는 것)"
                      if R > 1e-3 else "접평면 (지름 미상)"))
            if use_cyl:
                back = float(np.sqrt(max(R * R - a_m * a_m, 0.0)))
                c0 = c0 + n * back          # 표면 → 중심으로 뒤로 민다
            s_ax = P @ dv
            cands.append({"idx": k, "n": n, "d": -t,
                          "shape": "cylinder" if use_cyl else "axis",
                          "axis_dir": dv, "axis_pt": c0, "radius": R,
                          "lat_max": hw * 1.1 + 0.004,
                          "s_lo": float(s_ax.min()) - margin_m,
                          "s_hi": float(s_ax.max()) + margin_m})
        else:
            continue
        r.setdefault("aux_point_uv", [])
        r.setdefault("aux_point_xyz", [])
    if not cands:
        return 0

    n_placed = 0
    for lid, uv in aux_lines_uv.items():
        A = np.asarray(uv, float).reshape(-1, 2)
        uh = (A[:, 0] - cx) / f
        vh = (A[:, 1] - cy) / f
        best_z = np.full(len(A), np.inf)
        best_i = np.full(len(A), -1, dtype=int)
        for c in cands:
            if c["shape"] == "cylinder":
                Z = _ray_cylinder(uh, vh, b, c["axis_pt"], c["axis_dir"],
                                  c["radius"])
            else:
                n, d = c["n"], c["d"]
                den = n[0] * uh + n[1] * vh + n[2]
                with np.errstate(divide="ignore", invalid="ignore"):
                    Z = -(d + n[0] * b) / den
            P = np.column_stack([b + Z * uh, Z * vh, Z])
            ok = np.isfinite(Z) & (Z > 0.05) & (Z < 60.0) & (Z < best_z)
            if c["shape"] == "plane":
                ok &= (np.all(P >= c["lo"], axis=1)
                       & np.all(P <= c["hi"], axis=1))
            else:
                # 축에서 옆으로 얼마나 벗어났나 / 축을 따라 어디쯤인가
                q = P - c["axis_pt"]
                s_ax = q @ c["axis_dir"]
                lat = np.linalg.norm(q - np.outer(s_ax, c["axis_dir"]), axis=1)
                s_abs = P @ c["axis_dir"]
                ok &= ((lat <= c["lat_max"]) & (s_abs >= c["s_lo"])
                       & (s_abs <= c["s_hi"]))
            best_z[ok] = Z[ok]
            best_i[ok] = c["idx"]
        for c in cands:
            m = best_i == c["idx"]
            if not m.any():
                continue
            Z = best_z[m]
            P = np.column_stack([b + Z * uh[m], Z * vh[m], Z])
            results[c["idx"]]["aux_point_uv"].append(A[m])
            results[c["idx"]]["aux_point_xyz"].append(P)
            n_placed += int(m.sum())

    for c in cands:
        r = results[c["idx"]]
        r["aux_point_uv"] = (np.vstack(r["aux_point_uv"])
                             if r["aux_point_uv"] else np.empty((0, 2)))
        r["aux_point_xyz"] = (np.vstack(r["aux_point_xyz"])
                              if r["aux_point_xyz"] else np.empty((0, 3)))
        r["n_aux_points"] = int(len(r["aux_point_xyz"]))
    return n_placed


def measure_region(points_3d, cls, g_hat, camera_params,
                   flatness_threshold_mm=1.5, sigma_u_px=0.2,
                   target_sigma_mm=2.0, member_length_m=None, n_lines=None):
    """
    한 영역의 3D 점군에 클래스에 맞는 검측식을 적용한다.

    Returns
    -------
    dict — kind, theta_deg, judge, flatness, uncertainty, n_points, status
        status : "measured" | "rejected"
    """
    pts = np.asarray(points_3d, dtype=float)
    kind = _EQ5.MEASURE_KIND.get(cls)
    kcs_cls = _EQ5.KCS_CLASS.get(cls, cls)
    out = {"class": cls, "kind": kind, "n_points": int(len(pts)),
           "status": "rejected", "reject_reason": None,
           "theta_deg": None, "judge": None,
           "flatness": None, "uncertainty": None}

    if kind is None:
        out["reject_reason"] = f"검측 대상 클래스 아님 ({cls})"
        return out
    if len(pts) < 12:
        out["reject_reason"] = f"점 부족 ({len(pts)} < 12)"
        return out

    # ── 선형 부재: 축 적합 ──
    if kind == "axis_vertical":
        # 얇은 부재는 마스크 실루엣에서 반드시 오염되므로 robust 적합을 쓴다
        ax = _EQ2.fit_axis_ransac(pts)
        out["axis"] = {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                       for k, v in ax.items() if k != "reject_reason"}
        if not ax["is_valid"]:
            out["reject_reason"] = ax["reject_reason"]
            return out
        theta = _EQ3.measure_axis_verticality(ax["direction"], g_hat)
        out["theta_deg"] = round(theta, 4)
        # 격자가 닿은 구간만 보이므로 부재 전체 길이는 알 수 없다.
        # 측정 구간을 부재 길이로 대신 쓰면 mm 판정의 근거가 틀어지므로,
        # 전체 길이는 None 으로 두고 각도로 판정한다
        # (member_length_m 은 도면에서 알 때 호출부가 넘긴다).
        out["judge"] = _EQ3.judge_kcs(theta, kcs_cls,
                                      member_length_m=member_length_m,
                                      measured_span_m=ax["length_m"])

        # 세장비에서 오는 각도 불확실도를 판정에 반영한다.
        # 부재가 짧게만 보이면 원통 단면이 주축을 끌어당겨 각도가 흔들린다.
        # 측정값만 보고 합격을 내주면 그 흔들림이 판정에 그대로 숨는다.
        unc = ax.get("angle_uncertainty_deg")
        j = out["judge"]
        j["cross_section_resolved"] = ax.get("cross_section_resolved", True)
        if ax.get("note"):
            # 단면이 잡히지 않았다는 사실은 판정만큼 중요하다. 지름을
            # 모르면 부재 종류(동바리/기둥/철근)를 고를 수 없고, 그에 따라
            # 적용할 KCS 허용치도 고를 수 없다. 조서 비고에 남긴다.
            j["note"] = ((j.get("note") + " / ") if j.get("note") else "") \
                + ax["note"]
        # ── 격자선 한 줄짜리 부재: 재긴 재되, 절반만 잰다 ──
        # 한 줄에서 나온 3D 점도 깊이는 제대로 나온다. 부재 위쪽 화소와
        # 아래쪽 화소의 Z 가 다르면 그 차이가 곧 기울기다 —
        #
        #       tan θ_평면안 = (Z_아래 − Z_위) / (관측 구간 길이)
        #
        # 이것이 축 적합(fit_axis_pca)이 실제로 하는 일이고, 잡음 없는
        # 검증에서 참 1.00°/2.00° 를 1.0000°/2.0000° 로 되돌린다
        # (Ø48.6mm·1.65m 기준 ΔZ = 29.88mm / 59.74mm, Δu = 2.70px / 5.40px).
        #
        # 다만 그렇게 잡히는 것은 **레이저 평면 안** 성분뿐이다. 한 줄에서
        # 나온 점은 삼각측량의 정의상 그 평면 위에 놓이므로, 축 적합이
        # 찾는 것은 부재의 축이 아니라 표면과 평면의 교선이고, 교선은
        # 평면 안에서만 기울 수 있다.
        #
        #   · 평면 안 기울기(카메라 쪽으로 넘어짐)  → ΔZ 로 보인다
        #   · 평면에 수직인 기울기(옆으로 넘어짐)   → ΔZ = 0.00mm, 안 보인다
        #
        # 그래서 잰 값은 참 기울기의 한 성분(θ·cos φ)이고, 참값은 항상
        # 그보다 크거나 같다(두 성분이 제곱합이므로). 방향이 하나뿐인
        # 부등식이므로 판정도 한쪽으로만 성립한다:
        #
        #   · 잰 값이 이미 허용치를 넘었다 → **기준초과가 확정**이다.
        #     참값은 더 크니 결론이 뒤집힐 수 없다.
        #   · 잰 값이 허용치 이내다 → 합격이라 말할 수 없다. 안 보이는
        #     성분이 허용치를 넘었을 수 있다. 실측에서 동바리 세 본이
        #     "0.0000° 합격" 으로 나왔는데, 그 0 은 측정값이 아니라
        #     기하의 결과였다.
        #
        # 즉 한 줄짜리 측정은 부재를 **떨어뜨릴 수는 있어도 붙여줄 수는
        # 없다**. 아래가 그 비대칭을 그대로 옮긴 것이다.
        single_plane = (n_lines is not None and n_lines <= 1) or (
            not ax.get("cross_section_resolved", True)
            and float(ax.get("radius_est_mm") or 0.0) <= 1e-6)
        out["single_plane"] = bool(single_plane)
        out["n_lines"] = (int(n_lines) if n_lines is not None else None)
        if single_plane:
            j["cross_section_resolved"] = False
            j["measured_component"] = "레이저 평면 안 성분만 (참값의 하한)"
            j["theta_deg_lower_bound"] = round(float(theta), 4)
            j["theta_deg_is_lower_bound"] = True
            head = ("격자선이 한 줄만 걸렸다"
                    + (f"(선 {int(n_lines)}개)" if n_lines is not None else "")
                    + f". 부재 위쪽 화소와 아래쪽 화소의 깊이 차이로 "
                      f"{theta:.4f}° 를 쟀는데, 이것은 **레이저 평면 안**"
                      f"으로 넘어진 성분뿐이다. 평면에 수직으로(옆으로) "
                      f"넘어진 성분은 깊이를 바꾸지 않아 보이지 않는다. "
                      f"참 기울기는 이 값보다 크거나 같다. ")
            if j.get("is_pass") is False:
                # 하한이 이미 허용치를 넘었다 → 참값은 더 크다. 확정이다.
                j["judgement"] = "기준초과"
                tail = (f"다만 이 하한이 이미 허용치를 넘었으므로 "
                        f"**기준초과는 확정**이다 — 보이지 않는 성분이 "
                        f"더해지면 값은 커지기만 한다. 실제 기울기 크기와 "
                        f"방향까지 알려면 격자선이 두 줄 이상 걸리도록 "
                        f"다시 찍을 것")
            else:
                j["is_pass"] = None
                j["judgement"] = "판정보류(단면 미분해)"
                tail = (f"허용치({j.get('allow_deg')}°) 이내로 나왔지만 "
                        f"합격이라 말할 수 없다 — 보이지 않는 성분이 허용치를 "
                        f"넘었을 수 있다. 부재에 격자선이 두 줄 이상 걸리도록 "
                        f"더 가까이서 찍거나 격자를 조밀하게 할 것")
            j["note"] = ((j.get("note") + " / " if j.get("note") else "")
                         + head + tail)

        if unc is not None:
            j["angle_uncertainty_deg"] = unc
            j["slenderness"] = ax.get("slenderness")
            tol = j.get("allow_deg", 0.5)
            if j.get("is_pass") and abs(theta) + unc > tol:
                j["is_pass"] = None
                j["judgement"] = "판정보류(노출길이)"
                j["note"] = (
                    f"측정 {theta:.3f}° 는 허용 {tol}° 이내이나, 부재가 "
                    f"{ax['length_m']*1000:.0f}mm 만 보여(세장비 "
                    f"{ax['slenderness'] if ax.get('slenderness') else '미확인'}) "
                    f"각도 불확실도가 ±{unc:.2f}° 다. "
                    f"합쳐서 {abs(theta)+unc:.2f}° 로 허용치를 넘을 수 있어 "
                    f"합격 판정을 내리지 않는다. 부재를 세로로 더 길게 담아 "
                    f"재촬영할 것")

        out["uncertainty"] = _EQ5.region_uncertainty(
            pts, camera_params, normal=None, sigma_u_px=sigma_u_px,
            target_sigma_mm=target_sigma_mm)
        # 평활도는 원통면이라 성립하지 않음
        out["flatness"] = {"applicable": False,
                           "reason": "선형 부재(원통면) — 평활도 정의 없음"}
        out["status"] = "measured"
        return out

    # ── 면 부재: 방향 무관 TLS 평면 적합 ──
    plane, inliers = _EQ2.fit_plane_tls_ransac(pts, threshold=0.01)
    normal = np.array(plane[:3], dtype=float)
    out["plane"] = [round(float(x), 6) for x in plane]
    out["plane_inliers"] = int(inliers.sum())

    theta = _EQ3.measure_from_gravity(normal, g_hat, kind)
    out["theta_deg"] = round(theta, 4)
    out["judge"] = _EQ3.judge_kcs(theta, kcs_cls, member_length_m=None)

    unc = _EQ5.region_uncertainty(pts, camera_params, normal=normal,
                                  sigma_u_px=sigma_u_px,
                                  target_sigma_mm=target_sigma_mm)
    out["uncertainty"] = unc

    # ── 평활도 ──
    # 두 가지를 함께 낸다. 쓰임이 다르다.
    #   eq4 : 요철의 위치·개수·깊이를 찾는다 (어디가 문제인지)
    #   eq6 : KCS 가 규정한 직선자 처짐량으로 판정한다 (합격인지)
    # 전역 평면 잔차(eq4)로는 시방 판정을 할 수 없다. 넓은 면이 완만히
    # 휘면 잔차는 크지만 3m 자에는 안 걸리고, 좁고 급한 굴곡은 그 반대다.
    fd = _EQ4.detect_defects_region(pts, plane=plane,
                                    threshold_mm=flatness_threshold_mm)
    flat = {"applicable": True,
            "defect_max_dev_mm": round(fd["overall_max_dev_mm"], 3),
            "rms_dev_mm": round(fd["rms_dev_mm"], 3),
            "raw_max_dev_mm": round(fd["raw_max_dev_mm"], 3),
            "defect_clusters": len(fd["verified_clusters"]),
            "defect_count": fd["defect_count"],
            "reject_reason": fd.get("reject_reason")}
    flat["defects"] = _defects_in_image(fd, camera_params)

    kcs = _EQ6.judge_kcs_flatness(
        pts, cls, plane=plane,
        sigma_normal_mm=(None if unc["flatness_measurable"]
                         else unc["sigma_normal_mm"]),
        target_sigma_mm=target_sigma_mm)
    flat["kcs"] = kcs
    flat["judgement"] = kcs["judgement"]
    flat["is_pass"] = kcs["is_pass"]
    if kcs["checks"]:
        c0 = max(kcs["checks"], key=lambda x: x.get("ratio") or 0.0)
        flat["straightedge_length_m"] = c0["length_m"]
        flat["max_gap_mm"] = c0["max_gap_mm"]
        flat["upper_estimate_mm"] = c0["upper_estimate_mm"]
        flat["tolerance_mm"] = c0["tolerance_mm"]
    if not unc["flatness_measurable"]:
        flat["note"] = (f"법선방향 불확실도 σ_n={unc['sigma_normal_mm']}mm > "
                        f"목표 {target_sigma_mm}mm "
                        f"(Z={unc['z_mean_m']}m, 입사각 {unc['incidence_deg']}°)")
    elif kcs.get("note"):
        flat["note"] = kcs["note"]
    out["flatness"] = flat
    out["status"] = "measured"
    return out


def split_incoherent_region(xyz, cls, g_hat, min_points=12,
                            outlier_frac=0.08, plane_threshold_m=0.015):
    """
    한 영역 안에 성질이 다른 면이 섞여 있으면 기하로 되쪼갠다.

    【왜 필요한가】
      세그멘테이션이 두 부재를 **같은 라벨로 병합**하면(예: 바닥을 벽으로
      오분류) 두 면의 점이 한 영역에 들어온다. 그러면 영역 평면적합이 두
      면에 걸쳐지고, 점이 적은 쪽 부재는 outlier 로 밀려 통째로 사라진다.
      실측에서 바닥 480점이 벽 4121점에 병합되자 바닥 검측 결과가 아예
      나오지 않았다. 경계가 몇 px 어긋나는 것과는 차원이 다른 실패다.

      eq5 의 라벨 융합은 이 경우를 못 고친다. 융합은 "이 영역이 무엇인가"를
      바로잡을 뿐, "이 영역이 사실 둘"이라는 것은 판단하지 않기 때문이다.

    【판정 기준】
      대표 모델(면이면 평면, 선형이면 축)에 대한 outlier 비율이
      outlier_frac 을 넘고 그 수가 min_points 이상이면 섞였다고 본다.
      깨끗한 단일 면은 outlier 가 노이즈뿐이라 이 문턱에 닿지 않는다.

    Returns
    -------
    list of (class_name, index_array) — 쪼갤 필요가 없으면 원본 하나만
    """
    pts = np.asarray(xyz, dtype=float)
    n = len(pts)
    keep_all = [(cls, np.arange(n))]
    if n < 2 * min_points:
        return keep_all

    # 형상 자체가 불분명하면 그것이 곧 "섞였다"는 증거다.
    # 대표 모델 적합이 실패했는데 그대로 넘어가면, 병합된 영역이 통째로
    # 기각되어 두 부재를 다 잃는다(실측: 벽 4134점 + 동바리 288점이 한
    # 라벨로 묶이자 축 적합이 무효가 되고 영역 전체가 기각됨).
    ev = _EQ5.geometric_evidence(pts, g_hat)
    need_split = (ev["shape"] not in ("plane_vertical", "plane_horizontal",
                                      "linear_vertical")
                  or float(ev.get("confidence", 0.0)) < 0.5)

    if not need_split:
        kind = _EQ5.MEASURE_KIND.get(cls)
        if kind in ("plane_vertical", "plane_horizontal"):
            try:
                _, inl = _EQ2.fit_plane_tls_ransac(pts,
                                                   threshold=plane_threshold_m)
                n_out = int((~inl).sum())
            except Exception:
                return keep_all
        elif kind == "axis_vertical":
            ax = _EQ2.fit_axis_pca(pts)
            if not ax["is_valid"]:
                n_out = n                      # 적합 실패 → 섞인 것으로 본다
            else:
                c, d = ax["centroid"], ax["direction"]
                rel = pts - c
                radial = np.linalg.norm(rel - np.outer(rel @ d, d), axis=1)
                n_out = int((radial > 3.0 * max(ax["radial_rms_mm"], 1.0)
                             / 1000.0).sum())
        else:
            return keep_all

        if n_out < min_points or n_out / n < outlier_frac:
            return keep_all

    # 섞였다 → 기하 전용 백엔드로 이 영역만 다시 나눈다.
    # 문턱은 위에서 쓴 것과 **같은 값** 을 넘긴다. 여기만 기본값(10mm)으로
    # 두면, 잡음이 그보다 큰 촬영에서 이미 한 면으로 잘 잡힌 영역을 다시
    # 여러 조각으로 갈라 놓는다(실측: 2.7m 벽 σ_Z=15.9mm 에서 벽 하나가
    # 셋으로 쪼개져 조각마다 수직도 0.0000° 로 나왔다).
    try:
        seg = _SEG.segment(None, backend="geom",
                           table={"xyz": pts,
                                  "uv": np.zeros((n, 2)),
                                  "lid": np.array(["R"] * n, dtype=object),
                                  "seq": np.arange(n)},
                           g_hat=g_hat, min_plane_points=min_points,
                           min_linear_points=min_points,
                           plane_threshold_m=plane_threshold_m)
    except Exception:
        return keep_all

    labels, names = seg["point_labels"], seg["class_names"]
    subs = []
    for cid in np.unique(labels):
        sub_cls = names.get(int(cid), "background")
        if sub_cls in _EQ5.IGNORE_CLASSES:
            continue
        idx = np.where(labels == cid)[0]
        if len(idx) >= min_points:
            subs.append((sub_cls, idx))
    if len(subs) < 2:
        return keep_all
    return subs


# =====================================================================
# 한 장 전체 검측
# =====================================================================
def finalize_extents(result):
    """
    각 영역의 끝이 "부재의 끝" 인지 "격자가 닿은 데까지" 인지 가르고,
    선형 부재의 구간이 서로 다른 이유를 한 줄로 정리한다.

    inspect_image 안에 있던 것을 밖으로 뺐다 — 이 판단만 따로 검증할 수
    있어야 한다(회귀 [18]). result 를 제자리에서 고치고 span_note 를
    summary 에 넣어 돌려준다.
    """
    regions = result.get("regions") or []
    # 각 영역의 끝이 "부재의 끝" 인지 "격자가 닿은 데까지" 인지 가른다.
    # 셋 다 같은 화소 구간을 훑었는데 거리가 달라 미터값만 달라진 경우를
    # 부재 길이 차이로 읽으면 안 된다(실측: 동바리 세 본 1.99/1.93/1.87m,
    # 거리 1.705/1.652/1.595m — 비율이 정확히 같다).
    boxes = [r["point_uv_box"] for r in regions if r.get("point_uv_box")]
    if boxes:
        B = np.asarray(boxes, float)
        gx0, gy0 = float(B[:, 0].min()), float(B[:, 1].min())
        gx1, gy1 = float(B[:, 2].max()), float(B[:, 3].max())
        mg = 0.01 * max(gx1 - gx0, gy1 - gy0, 1.0)
        for r in regions:
            bx = r.get("point_uv_box")
            if not bx:
                continue
            lo_x, lo_y = bx[0] <= gx0 + mg, bx[1] <= gy0 + mg
            hi_x, hi_y = bx[2] >= gx1 - mg, bx[3] >= gy1 - mg
            r["extent_limited"] = bool(lo_x or lo_y or hi_x or hi_y)
            # ── 선형 부재는 **양 끝을 따로** 봐야 한다 ──
            # 세 동바리가 1.9938 / 1.9317 / 1.8652 m 로 서로 다르게 나왔는데,
            # 거리(1.705 / 1.652 / 1.595 m)로 나누면 1.16938 / 1.16932 /
            # 1.16941 — 다섯 자리까지 같다. 셋 다 화소 구간이 같았을 뿐이다.
            # 격자는 고정된 화각을 덮으므로 덮는 미터 길이가 거리에 비례한다.
            # 즉 이 값은 부재 높이가 아니라 **격자가 스친 구간** 이고, 양 끝이
            # 격자 경계에 닿아 있으면 부재 길이는 그 값 **이상** 이라는 것밖에
            # 모른다. 그걸 '높이' 로 내보내면 같은 규격 부재가 서로 다른
            # 높이로 읽힌다.
            if r.get("kind") == "axis_vertical":
                r["extent_ends"] = {"위쪽": bool(lo_y), "아래쪽": bool(hi_y)}
                r["length_is_lower_bound"] = bool(lo_y or hi_y)
            else:
                r["length_is_lower_bound"] = bool(r["extent_limited"])

    # ── 선형 부재의 구간이 서로 다른 이유를 조서가 직접 말하게 한다 ──
    # 격자는 고정 화각을 덮으므로 덮는 미터 길이가 거리에 비례한다. 양 끝이
    # 격자 한계에 닿은 부재들의 구간/거리 비가 같으면, 길이가 다른 게 아니라
    # 같은 화소 구간을 서로 다른 거리에서 본 것이다. 사람이 표만 보고
    # "동바리 높이가 제각각" 이라고 읽지 않도록 근거를 숫자로 남긴다.
    span_note = None
    lin = [r for r in regions
           if r.get("kind") == "axis_vertical" and r.get("status") == "measured"
           and r.get("length_is_lower_bound")
           and (r.get("axis") or {}).get("length_m")
           and r.get("point_xyz") is not None and len(r["point_xyz"])]
    if len(lin) >= 2:
        rows = []
        for r in lin:
            L = float(r["axis"]["length_m"])
            z = float(np.median(np.asarray(r["point_xyz"], float)[:, 2]))
            if z > 1e-6:
                rows.append((L, z, L / z))
        if len(rows) >= 2:
            ratio = np.array([t[2] for t in rows], float)
            spread = float(np.ptp(ratio) / max(np.mean(ratio), 1e-9))
            for r, t in zip(lin, rows):
                r["span_per_depth"] = round(float(t[2]), 5)
            if spread < 0.01:
                span_note = (
                    "선형 부재 " + str(len(rows)) + "본의 측정 구간이 서로 "
                    "다른 것은 **거리 차이** 때문이다 — "
                    + " / ".join(f"{L:.4f}m ÷ {z:.3f}m" for L, z, _ in rows)
                    + f" = {ratio.mean():.5f} 로 모두 같다(편차 "
                      f"{spread * 100:.2f}%). 격자가 고정 화각을 덮으므로 "
                      "덮는 길이가 거리에 비례하고, 셋 다 같은 화소 구간을 "
                      "봤다는 뜻이다. 부재 길이가 다르다는 근거가 아니다.")
    result.setdefault("summary", {})["span_note"] = span_note
    return span_note


def roundness_note(result, line_gain=None):
    """
    "동바리가 왜 안 둥근가" 에 대한 한 줄. 조서와 표에 남긴다.

    세로선 한 줄은 연직 원통을 **직선** 으로 자른다(연직면 ∩ 연직 원통).
    둥근 것을 보여 줄 수 있는 건 부재를 가로지르는 가로선인데, 격자를
    굴리지 않으면 가로선의 레이저 평면이 카메라 광심을 지나 깊이를 못
    준다(이득 g=∞). 이건 선검출 실패가 아니라 기하의 한계다 — 이 캡처에서
    가로선은 21/21 검출됐고 |n_x| 가 9.2e-08 이었다.

    광축 둘레로 γ 만큼 굴리면 g = 1/sin γ 로 유한해지고, 그때는 가로선
    화소가 그냥 삼각측량된다. 지름을 가정할 필요가 없다 — 합성 검증에서
    γ=10° 면 참 원통면에서 0.16mm, γ=45° 면 0.13mm 다.
    """
    regs = result.get("regions") or []
    lin = [r for r in regs if r.get("kind") == "axis_vertical"
           and r.get("status") == "measured"]
    if not lin:
        return None
    inferred = [r for r in lin if (r.get("aux_surface") or "").startswith(
        ("원통면", "접평면"))]
    if not inferred:
        return None
    n_ok = 0
    if line_gain:
        n_ok = sum(1 for k, g in line_gain.items()
                   if str(k).startswith("H") and np.isfinite(g)
                   and g <= _EQ7.MAX_DEPTH_GAIN)
    if n_ok:
        return None            # 가로선이 깊이를 주면 그냥 측정된 점이다
    return (
        "선형 부재의 가로선 점은 **측정이 아니라 유도값** 이다. 세로선 한 "
        "줄은 연직 원통을 직선으로 자르므로(연직면 ∩ 연직 원통 = 직선) "
        "둥근 단면을 보여 줄 수 없고, 그걸 보여 줄 가로선은 격자를 굴리지 "
        "않으면 레이저 평면이 카메라 광심을 지나 깊이를 못 준다(이득 g=∞, "
        "이 캡처 |n_x|=9.2e-08). 선검출 문제가 아니다 — 가로선은 전부 "
        "검출됐다. 광축 둘레로 γ 만큼 굴리면 g=1/sin γ 로 유한해져 가로선 "
        "화소가 그냥 삼각측량되고, 동바리 단면이 가정 없이 측정된다 "
        "(합성 검증: γ=10° → 참 원통면에서 0.16mm, γ=45° → 0.13mm).")


def inspect_image(lines_pixels, lines_xyz, camera_params, g_hat,
                  rgb_off=None, seg_backend="gt", seg_kwargs=None,
                  erode_default_px=3, erode_thin_px=1,
                  min_region_points=12, sigma_u_px=0.2,
                  target_sigma_mm=2.0, flatness_threshold_mm=1.5,
                  split_incoherent=True, line_gain=None,
                  aux_lines_uv=None):
    """
    선검출 결과 + 삼각측량 결과 + 세그멘테이션으로 영역별 검측을 수행한다.

    Parameters
    ----------
    lines_pixels : {lid: [(u,v), ...]}   A_선검출 출력 (ON 프레임)
    lines_xyz    : {lid: [(X,Y,Z), ...]} eq1 삼각측량 결과 (같은 순서)
    camera_params: dict — f_px, b_m, cx_px, cy_px
    g_hat        : (3,) 조사기 좌표계 중력 단위벡터 (eq3.gravity_in_laser_frame)
    rgb_off      : (H,W,3) 레이저 OFF 프레임 — 세그멘테이션 입력
    seg_backend  : "gt" | "geom" | "sam" | "vlm"
    seg_kwargs   : 백엔드 인자 (gt → label_map, id_to_semantic)
    line_gain    : {lid: g} | None
        선별 깊이 이득 (eq7). 주면 영역마다 그 영역을 이룬 점들의
        RMS 이득으로 σ_u 를 부풀려 불확실도를 계산한다. 이득이 큰
        선(굴린 격자의 가로선)의 점은 같은 화소 오차라도 깊이가 더
        흔들리므로, 이걸 반영하지 않으면 평활도가 실제보다 좋게 나온다.
    aux_lines_uv : {lid: (N,2)} | None
        깊이를 못 준 선의 화소 (triangulate_lines 의 skipped_uv).
        주면 이미 맞춘 면 위에 올려 그림에만 표시한다 — 검측에는 쓰지
        않는다(면에서 빌려 온 거리라 잔차가 정의상 0 이다).

    Returns
    -------
    dict — regions, summary, segmentation, assign_stats
    """
    seg_kwargs = dict(seg_kwargs or {})
    table = _EQ5.build_point_table(lines_pixels, lines_xyz)

    # ── 평면 판정 문턱을 **이 촬영의 실제 깊이 잡음** 에 맞춘다 ──
    # 고정값(10~15mm)을 쓰면 잡음이 그보다 큰 장면에서 한 면이 여러 평면으로
    # 쪼개진다. 실측: 2.7m 거리의 벽은 σ_Z=15.9mm 인데, 10mm 문턱에서는
    # 벽 하나가 세 조각(6217·6984·2532점)으로 갈리고 조각마다 수직도가
    # 0.0000° 로 나와 판정이 무의미해졌다.
    #
    #   σ_Z = σ_u · Z² / (f·b)      문턱 = 3σ_Z
    #
    # 3σ 는 정규분포에서 성한 점의 99.7% 를 담는 폭이다. 잡음보다 좁게
    # 잡으면 같은 면을 쪼개고, 너무 넓게 잡으면 다른 면을 삼킨다.
    noise_threshold_m = 0.015
    if len(table["xyz"]):
        _f = float(camera_params.get("f_px", 2318.8))
        _b = float(camera_params.get("b_m", 0.150))
        _z = float(np.median(table["xyz"][:, 2]))
        noise_threshold_m = float(np.clip(
            3.0 * float(sigma_u_px) * _z * _z / (_f * _b), 0.008, 0.10))
    split_threshold_m = noise_threshold_m

    if len(table["xyz"]) == 0:
        return {"regions": [], "summary": {"n_regions": 0},
                "error": "삼각측량된 격자점이 없습니다."}

    # ── [C] 영역 분할 ──
    if seg_backend == "geom":
        seg_kwargs.setdefault("table", table)
        seg_kwargs.setdefault("g_hat", g_hat)
        seg_kwargs.setdefault("camera_params", camera_params)
        # 평면 RANSAC 의 inlier 문턱을 **이 촬영의 실제 깊이 잡음** 에 맞춘다.
        #
        # 고정값 10mm 를 쓰면 잡음이 그보다 큰 장면에서 한 면이 여러 평면으로
        # 쪼개진다. 실측: 2.7m 거리의 벽은 σ_Z=15.9mm 인데 두께가 170mm 뿐이라,
        # 10mm 문턱으로는 벽 하나가 세 조각(6217·6984·2532점)으로 갈리고
        # 조각마다 수직도가 0.0000° 로 나와 판정이 무의미해졌다.
        #
        # 문턱은 3σ 로 둔다 — 정규분포에서 성한 점의 99.7% 를 담는 폭이다.
        # 잡음보다 좁게 잡으면 같은 면을 쪼개고, 너무 넓게 잡으면 다른 면을
        # 삼킨다. 아래위로 묶어 극단을 막는다.
        seg_kwargs.setdefault("plane_threshold_m", noise_threshold_m)
    seg = _SEG.segment(rgb_off, backend=seg_backend, **seg_kwargs)

    # ── [eq5] 점 → 영역 ──
    if seg["label_map"] is not None:
        regions, stats = _EQ5.assign_points_to_regions(
            table, seg["label_map"], seg["class_names"],
            erode_default_px=erode_default_px, erode_thin_px=erode_thin_px,
            min_points=min_region_points)
    else:
        regions, stats = _regions_from_point_labels(
            table, seg["point_labels"], seg["class_names"],
            min_points=min_region_points)

    # ── 영역별 검측 ──
    results = []
    n_split = 0
    n_linear_rescued = 0
    for reg in regions:
        pts_all = table["xyz"][reg["idx"]]
        ev0 = _EQ5.geometric_evidence(pts_all, g_hat)
        fu0 = _EQ5.fuse_label(reg["class"], ev0)

        # ── 1) 병합된 영역이면 먼저 되쪼갠다 ──
        # 세그멘테이션이 두 부재를 한 라벨로 묶었을 수 있다. 이 단계를
        # 선형 정제보다 **먼저** 해야 한다. 순서를 바꾸면, 라벨이 선형인
        # 영역에서 축 바깥 점이 통째로 버려져 같이 묶여 있던 다른 부재가
        # 재분할에 도달하지 못한다(실측: 바닥 471점이 동바리 라벨에 묶이자
        # 축 정제가 이를 버려 바닥 검측이 통째로 사라짐).
        parts = ([(fu0["final_class"], np.arange(len(pts_all)))]
                 if not split_incoherent
                 else split_incoherent_region(
                     pts_all, fu0["final_class"], g_hat,
                     min_points=min_region_points,
                     plane_threshold_m=split_threshold_m))
        if len(parts) > 1:
            n_split += 1

        for part_cls, sub_idx in parts:
            pts = pts_all[sub_idx]
            # 이 조각이 원래 table 의 몇 번 점인지 끝까지 들고 간다.
            # 세그멘테이션 결과 이미지를 그리려면 각 점의 화소 좌표가
            # 필요한데, 아래 선형 정제가 점을 걸러내므로 인덱스를 같이
            # 걸러야 짝이 맞는다.
            keep = np.asarray(reg["idx"])[np.asarray(sub_idx)]

            # ── 2) 조각별 선형 부재 정제 ──
            # 동바리·철근처럼 가는 부재는 마스크가 몇 px 만 밖으로 밀려도
            # 뒤쪽 면의 점이 딸려 들어오고, 부재가 가늘어 그 비중이 크다.
            # 그러면 형상 판별이 선형에서 평면으로 뒤집히고, "면↔선형은
            # 기하 우선" 규칙이 올바른 의미 라벨을 버려 부재가 사라진다
            # (실측: 마스크 +8px 팽창에서 동바리 288점에 벽 33점이 섞여 소실).
            rescued = False
            if part_cls in _EQ5.LINEAR_VERTICAL_CLASSES:
                ax0 = _EQ2.fit_axis_ransac(pts)
                cand = pts[ax0["inlier_mask"]] if ax0["is_valid"] else None
                if cand is not None and len(cand) >= min_region_points:
                    # 정제 결과가 **실제로 1D 부재인지** 확인해야 한다.
                    # 축 적합은 평면을 얇게 저민 조각에도 성공한다. 바닥
                    # 조각에 축을 맞추면 축이 면 안에 누워 수직도가 89° 로
                    # 나온다(실측: 라벨 오분류 시 동바리 오차 88.79°).
                    # 원통은 3번째 주축에도 두께가 남지만 판 조각은 납작해
                    # geometric_evidence 의 선형 판정으로 가려낼 수 있다.
                    if _EQ5.geometric_evidence(cand, g_hat)["shape"] == "linear_vertical":
                        if ax0.get("inlier_frac", 1.0) < 0.999:
                            n_linear_rescued += 1
                        pts = cand
                        keep = keep[ax0["inlier_mask"]]
                        rescued = True

            ev = _EQ5.geometric_evidence(pts, g_hat)
            if rescued:
                fu = {"semantic_class": part_cls, "geom_shape": ev["shape"],
                      "geom_confidence": round(float(ev.get("confidence", 0.0)), 3),
                      "final_class": part_cls, "source": "semantic",
                      "agreed": ev["shape"] == "linear_vertical",
                      "note": ("선형 부재 — robust 축 적합으로 오염점 제거 후 "
                               "의미 라벨 유지")}
            elif len(parts) == 1:
                fu = fu0
            else:
                fu = _EQ5.fuse_label(part_cls, ev)
            final_cls = fu["final_class"]

            # 이 영역을 이룬 점들의 깊이 이득 (RMS). 분산이 g² 로
            # 커지므로 평균이 아니라 제곱평균제곱근이 맞는 집계다.
            g_rms = 1.0
            if line_gain:
                gs = np.array([line_gain.get(l, 1.0)
                               for l in table["lid"][keep]], dtype=float)
                gs = gs[np.isfinite(gs)]
                if len(gs):
                    g_rms = float(np.sqrt(np.mean(gs ** 2)))
            n_lines = int(len(set(str(x) for x in table["lid"][keep])))
            r = measure_region(pts, final_cls, g_hat, camera_params,
                               flatness_threshold_mm=flatness_threshold_mm,
                               sigma_u_px=sigma_u_px * g_rms,
                               target_sigma_mm=target_sigma_mm,
                               n_lines=n_lines)
            r["depth_gain_rms"] = round(g_rms, 3)
            r["region_id"] = int(reg["class_id"])
            r["label_fusion"] = {k: v for k, v in fu.items() if k != "note"}
            r["label_fusion_note"] = fu["note"]
            r["geom_shape"] = ev["shape"]
            r["from_split"] = bool(len(parts) > 1)
            # 결과 이미지·엑셀에서 쓰는 화소 좌표 (검측에는 쓰지 않는다)
            r["point_uv"] = table["uv"][keep]
            r["point_idx"] = keep
            # 이 부재를 이룬 3D 점 그대로. 3D 산점도·좌표 내보내기가 쓴다.
            # 검측에 들어간 점과 **같은 배열** 이어야 그림과 조서가 어긋나지
            # 않는다(선형 정제로 걸러낸 점은 여기에도 없다).
            r["point_xyz"] = pts
            r["point_lid"] = table["lid"][keep]
            r["n_lines"] = n_lines
            # 화면 가장자리에 닿았으면 이 부재의 크기는 **하한** 이다.
            # 부재가 거기서 끝난 것이 아니라 화면이 끝난 것이라, 그 값을
            # 부재 길이로 읽으면 안 된다. 실측에서 동바리 세 본의 "높이"가
            # 1.99/1.93/1.87m 로 달랐는데, 거리에 정확히 비례한 값이었다 —
            # 셋 다 화면에 잘린 같은 화소 구간이었을 뿐이다.
            # 경계는 이미지 테두리가 아니라 **격자가 실제로 닿은 범위**로
            # 잰다. 추적은 신호가 있는 구간만 훑으므로 화면 끝까지 가지
            # 않는 경우가 흔하고, 그때도 부재는 그 너머로 이어진다.
            r["point_uv_box"] = [float(table["uv"][keep][:, 0].min()),
                                 float(table["uv"][keep][:, 1].min()),
                                 float(table["uv"][keep][:, 0].max()),
                                 float(table["uv"][keep][:, 1].max())]
            results.append(r)

    # 깊이를 못 주는 선(가로선)의 화소도 결과 그림에는 나와야 한다.
    # 검측에는 넣지 않고 aux_ 로 따로 들고 간다.
    n_aux = n_cross = 0
    if aux_lines_uv:
        # 가르기 전에 부재 폭을 먼저 잰다 — 폭을 모르면 가를 수 없다.
        try:
            measure_member_silhouettes({"regions": results}, aux_lines_uv,
                                       camera_params)
        except Exception:
            pass
        try:
            n_aux = place_aux_points(results, aux_lines_uv, camera_params)
        except Exception:
            n_aux = 0
        # 가로선 화소는 그림에만 쓰는 게 아니다. 부재를 가로지른 자리의
        # 중심이 세로선 한 줄로는 못 보던 옆 기울기를 준다.
        try:
            n_cross = resolve_by_line_gaps({"regions": results},
                                           aux_lines_uv, camera_params,
                                           g_hat)
        except Exception:
            n_cross = 0

    span_note = finalize_extents({"regions": results,
                                 "summary": {}})
    measured = [r for r in results if r["status"] == "measured"]
    summary = {
        "n_regions": len(results),
        "n_measured": len(measured),
        "n_rejected": len(results) - len(measured),
        "classes": sorted({r["class"] for r in results}),
        "label_corrections": sum(1 for r in results
                                 if r["label_fusion"]["source"] == "geometric"),
        "regions_split": n_split,
        "linear_members_rescued": n_linear_rescued,
        "aux_points": int(n_aux),
        "crossing_resolved": int(n_cross),
        "span_note": span_note,
        "roundness_note": roundness_note({"regions": results}, line_gain),
        "flatness_unmeasurable": sum(
            1 for r in measured
            if (r["flatness"] or {}).get("judgement") == "측정불가"),
    }
    return {"regions": results, "summary": summary,
            "segmentation": {"backend": seg["backend"], "meta": seg["meta"]},
            "assign_stats": stats}


def _regions_from_point_labels(table, point_labels, class_names, min_points=12):
    """geom 백엔드처럼 점 단위 라벨을 주는 경우의 영역 구성."""
    labels = np.asarray(point_labels)
    disc = _EQ5.mark_depth_discontinuity(table)
    stats = {"total": len(labels), "out_of_image": 0, "ignored_class": 0,
             "eroded_away": 0, "discontinuity": int(disc.sum()), "assigned": 0}
    regions = []
    for cid in np.unique(labels):
        cls = class_names.get(int(cid), "background")
        keep = (labels == cid) & ~disc
        if cls in _EQ5.IGNORE_CLASSES:
            stats["ignored_class"] += int((labels == cid).sum())
            continue
        if keep.sum() < min_points:
            continue
        idx = np.where(keep)[0]
        stats["assigned"] += len(idx)
        regions.append({"class": cls, "class_id": int(cid), "idx": idx,
                        "n_points": int(len(idx))})
    return regions, stats


# =====================================================================
# 촬영 결과 → 검측 (Isaac 비의존)
# =====================================================================
def triangulate_lines(lines_pixels, line_angles, camera_params,
                      max_depth_gain=None):
    """
    검출된 선의 화소점을 3D 로 되돌린다 (eq7 평면식).

    V선·H선을 함께 푸는 이유
    ----------------------
    예전에는 eq1 의 V선 전용식만 써서 H선을 통째로 버렸다. 그 근거는
    "기선이 X축이라 H선은 시차가 선과 나란해 깊이가 안 풀린다" 였고,
    회전이 없는 격자에서는 그 말이 맞다. 다만 그것은 H선의 성질이 아니라
    **격자를 굴리지 않았을 때만** 성립하는 조건이다.

    eq7 은 선을 각도가 아니라 레이저 평면 법선 n 으로 다룬다.

        Z = − n_x·b / (n_x·û + n_y·v̂ + n_z)

    이 식에서 깊이가 풀리는지는 오직 n_x 가 0 이 아닌지에 달렸고, 그
    민감도는 이득 g = √(n_x²+n_y²)/|n_x| 하나로 표현된다. V선은 g=1,
    회전 없는 H선은 g=∞(원리적으로 불가), 45° 굴린 격자는 양쪽 다
    g=√2 다. 그래서 판정 기준을 "H선인가" 가 아니라 "이 선의 g 가
    쓸 만한가" 로 바꾼다. 하드웨어가 격자를 굴려 오면 코드를 고치지
    않아도 가로선이 그대로 깊이 표본이 된다.

    버린 선은 skipped 에 이유와 함께 남긴다. 조용히 사라지면 "왜 점이
    절반인가" 를 조서에서 되짚을 수 없다.

    Returns
    -------
    lines_xyz : {lid: [(X,Y,Z), ...]}
    lines_uv  : {lid: [(u,v), ...]}   삼각측량에 성공한 점만 (순서 일치)
    info      : dict
        skipped     : [(lid, 사유), ...]
        skipped_uv  : {lid: (N,2)}  버린 선의 화소. 깊이는 못 주지만
                      결과 그림에는 찍어야 한다 — 실제로 검출된 점이고,
                      가로선까지 나와야 격자가 어디에 걸렸는지 보인다.
        line_gain   : {lid: g}          쓰인 선의 깊이 이득
        family      : eq7.family_summary  V/H 계열 요약
        n_by_family : {"V": 점수, "H": 점수}
    """
    f = float(camera_params["f_px"]); b = float(camera_params["b_m"])
    cx = float(camera_params["cx_px"]); cy = float(camera_params["cy_px"])
    gmax = _EQ7.MAX_DEPTH_GAIN if max_depth_gain is None else float(max_depth_gain)

    # 법선은 line_angles 에 이미 들어 있으면 그것을 쓴다(캘리브레이션이
    # 평면을 직접 잰 경우). 없으면 발사각과 장비 회전각에서 만든다.
    tilt = np.radians(float(camera_params.get("laser_tilt_deg", 0.0) or 0.0))
    roll = np.radians(float(camera_params.get("laser_roll_deg", 0.0) or 0.0))
    planes = _EQ7.line_planes(line_angles, tilt_rad=tilt, roll_rad=roll)

    lines_xyz, lines_uv, skipped = {}, {}, []
    skipped_uv = {}
    line_gain, n_by_family = {}, {}
    for lid, pts in lines_pixels.items():
        pl = planes.get(lid)
        if pl is None:
            skipped.append((lid, "발사각 미상"))
            continue
        g = pl["gain"]
        if not np.isfinite(g):
            skipped.append((lid, "깊이 정보 없음 (평면이 기선을 품음, g=∞)"))
            skipped_uv[lid] = np.asarray(pts, float).reshape(-1, 2)
            continue
        if g > gmax:
            skipped.append((lid, f"깊이 이득 과대 (g={g:.1f} > {gmax:g})"))
            skipped_uv[lid] = np.asarray(pts, float).reshape(-1, 2)
            continue
        arr = np.asarray(pts, dtype=float).reshape(-1, 2)
        if len(arr) == 0:
            continue
        xyz, keep = _EQ7.triangulate_plane(arr, pl["normal"], f, b, cx, cy)
        if len(xyz) < 5:
            skipped.append((lid, f"유효점 부족 ({len(xyz)})"))
            continue
        lines_xyz[lid] = xyz.tolist()
        lines_uv[lid] = arr[keep].tolist()
        line_gain[lid] = float(g)
        fam = lid[0]
        n_by_family[fam] = n_by_family.get(fam, 0) + len(xyz)

    info = {"skipped": skipped, "skipped_uv": skipped_uv,
            "line_gain": line_gain,
            "family": _EQ7.family_summary(planes),
            "n_by_family": n_by_family, "max_depth_gain": gmax}
    return lines_xyz, lines_uv, info


def inspect_capture(lines_pixels, line_angles, camera_params, R_world_cam,
                    label_map=None, id_to_semantic=None, rgb_off=None,
                    g_hat=None, backend=None, **kw):
    """
    한 번의 촬영 결과를 받아 영역별 검측까지 수행한다.

    inspection.py(Isaac)와 오프라인 검증 양쪽에서 같은 코드를 쓰기 위해
    Isaac 의존성을 두지 않는다.

    Parameters
    ----------
    R_world_cam : (3,3) or None
        카메라 자세행렬. 주면 여기서 중력을 유도한다(g_hat 미지정 시).
    label_map, id_to_semantic : Isaac 시맨틱 어노테이터 출력
        있으면 backend='gt', 없으면 backend='geom' 으로 자동 폴백.
    """
    if g_hat is None:
        if R_world_cam is None:
            raise ValueError("g_hat 또는 R_world_cam 중 하나는 필요합니다 "
                             "(중력 기준 없이는 수직·수평도를 정의할 수 없음).")
        g_hat = _EQ3.gravity_from_camera_rotation(R_world_cam)

    lines_xyz, lines_uv, tri_info = triangulate_lines(
        lines_pixels, line_angles, camera_params)
    if not lines_xyz:
        return {"regions": [], "summary": {"n_regions": 0},
                "error": "깊이를 풀 수 있는 선이 없습니다 (eq7 이득 g 초과).",
                "triangulation": tri_info}

    if backend is None:
        backend = "gt" if label_map is not None else "geom"
    seg_kwargs = ({"label_map": label_map, "id_to_semantic": id_to_semantic}
                  if backend == "gt" else {})

    kw.setdefault("line_gain", tri_info["line_gain"])
    kw.setdefault("aux_lines_uv", tri_info.get("skipped_uv"))
    res = inspect_image(lines_uv, lines_xyz, camera_params, g_hat,
                        rgb_off=rgb_off, seg_backend=backend,
                        seg_kwargs=seg_kwargs, **kw)
    res["gravity_laser_frame"] = [round(float(x), 6) for x in np.asarray(g_hat)]
    res["triangulation"] = tri_info
    res["skipped_lines"] = tri_info["skipped"]
    res["n_triangulated"] = int(sum(len(v) for v in lines_xyz.values()))
    return res


# =====================================================================
# 보고
# =====================================================================
def format_report(result):
    """영역별 검측 결과를 사람이 읽는 표로 만든다."""
    lines = []
    s = result["summary"]
    seg = result.get("segmentation", {})
    lines.append(f"세그멘테이션 backend={seg.get('backend')}  "
                 f"영역 {s['n_regions']}개 (검측 {s['n_measured']} / "
                 f"기각 {s['n_rejected']})  라벨교정 {s['label_corrections']}건")
    st = result.get("assign_stats", {})
    if st:
        lines.append(f"  점 배분: 전체 {st['total']} → 할당 {st['assigned']} "
                     f"(화면밖 {st['out_of_image']}, 비검측 {st['ignored_class']}, "
                     f"침식 {st['eroded_away']}, 깊이불연속 {st['discontinuity']})")
    lines.append("")
    hdr = (f"  {'클래스':<14}{'검측':<10}{'각도(°)':>9}{'판정':>10}"
           f"{'자처짐(mm)':>12}{'평활판정':>10}{'σ_n(mm)':>9}{'점수':>7}")
    lines.append(hdr)
    lines.append("  " + "-" * (len(hdr) - 2))
    kind_ko = {"plane_vertical": "수직도", "plane_horizontal": "수평도",
               "axis_vertical": "축수직도"}
    for r in result["regions"]:
        if r["status"] != "measured":
            lines.append(f"  {r['class']:<14}{'기각':<10}"
                         f"{'-':>9}{'-':>10}{'-':>12}{'-':>10}{'-':>9}"
                         f"{r['n_points']:>7}   ← {r['reject_reason']}")
            continue
        j = r["judge"] or {}
        # judgement 가 있으면 그것이 최종 표기다. is_pass 만 보면
        # 판정보류(is_pass=None)가 "기준초과" 로 뒤바뀐다 — 판정을 하지
        # 않은 것과 기준을 넘은 것은 전혀 다른 상태다.
        verdict = j.get("judgement") or ("합격" if j.get("is_pass")
                                         else "기준초과")
        f = r["flatness"] or {}
        fmax = (f"{f.get('max_gap_mm', 0.0):.2f}" if f.get("applicable")
                else "N/A")
        fjud = f.get("judgement", "N/A") if f.get("applicable") else "N/A"
        sn = r["uncertainty"]["sigma_normal_mm"]
        # 표에는 짧은 표기만 넣는다 — "판정보류(단면 미분해)" 를 그대로
        # 넣으면 칸을 넘겨 각도와 붙어 버린다. 괄호 안 사유는 아래 줄에.
        short, _, why = verdict.partition("(")
        lines.append(f"  {r['class']:<14}{kind_ko.get(r['kind'], r['kind']):<10}"
                     f"{r['theta_deg']:>9.4f}{short:>10}"
                     f"{fmax:>12}{fjud:>10}{sn:>9.2f}{r['n_points']:>7}")
        if why:
            lines.append(f"      ↳ 사유: {why.rstrip(')')}"
                         + (f" / {j['resolved_by']}"
                            if j.get("resolved_by") else ""))
        elif j.get("resolved_by"):
            lines.append(f"      ↳ 옆 성분 복원: {j['resolved_by']} "
                         f"(평면안 {j.get('theta_deg_plane_only')}° + "
                         f"평면수직 {j.get('theta_deg_lateral')}°)")
        if r["label_fusion"]["source"] == "geometric":
            lines.append(f"      ↳ 라벨 교정: {r['label_fusion_note']}")
        if f.get("judgement") == "측정불가":
            lines.append(f"      ↳ {f['note']}")
    sn = (result.get("summary") or {}).get("span_note")
    if sn:
        lines.append("")
        lines.append("  [측정 구간] " + sn)
    rn = (result.get("summary") or {}).get("roundness_note")
    if rn:
        lines.append("")
        lines.append("  [가로선 점] " + rn)
    return "\n".join(lines)


# ============ 자체 검증 ============
if __name__ == "__main__":
    _SYN = _load("synth_scene")
    print("=" * 78)
    print("영역별 품질검측 파이프라인 — 합성 씬 자체검증")
    print("=" * 78)
    scene = _SYN.build_scene()
    print(_SYN.describe(scene))
    print()

    for backend in ("gt", "geom"):
        kw = ({"label_map": scene["label_map"],
               "id_to_semantic": scene["id_to_semantic"]}
              if backend == "gt" else {})
        res = inspect_image(scene["lines_pixels"], scene["lines_xyz"],
                            scene["camera_params"], scene["g_hat"],
                            rgb_off=scene["rgb_off"],
                            seg_backend=backend, seg_kwargs=kw)
        print(f"── backend = {backend} " + "─" * 55)
        print(format_report(res))
        print()
        print(_SYN.score(scene, res))
        print()
