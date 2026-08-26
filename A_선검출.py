#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A_선검출.py — [품질검측 알고리즘 A] 20×20 그리드 레이저 선검출 (v8 현장용)
========================================================================
【버전 이력】
  v2  ERODE+이진화          → systematic offset 2~5px
  v3  raw intensity 가중평균  → RMSE 0.16px (시뮬)
  v4  적응형 임계값 + 격자검증
  v5  MAD 이상치 제거 + 결함 신호 보존
  v6  raycast 보간 테이블 → H선 개선, V선 악화(mu0=pred 문제)
  v7  mu0=centroid + sanity check + 적응형 BAND → 전체 개선
  v8  【현장용】 raycast 의존성 완전 제거
      · raycast → 카메라 파라미터 + DOE 발산각 + 측정 거리로 기하 예측
      · 격자 검증 폴백 → 인접선 보간 위치로 교체 (raycast GT 불필요)
      · 시뮬레이션 검증과 현장 배포를 동일 코드로 처리
  v9  【정확도 통합】 기존연구 벤치마킹 4대 개선 반영
      · [방안1] Steger 서브픽셀 — 미분(1차/2차) 기반, 비대칭 프로파일
                강건. 가우시안 대비 비대칭 조건 30% 개선 (실측)
      · [방안2] DOE 실제 격자 간격 — 등각도 발산의 tan 비선형 반영,
                선별 BAND (중앙 좁게 35px / 외곽 넓게 46px)
      · [방안3] 격자 동시 최적화(HO) — 전체 교차점을 이론 격자에
                최소제곱 동시 정렬, 개별 이상선(b=180mm) 상쇄
      · [방안4] 차영상 — laser_off_image 주면 ON-OFF 차분으로
                배경광 제거 (현장 직사광 대비)

【현장 동작 원리】
  레이저(DOE) 격자의 각 선 각도(α_i, β_j)와 카메라 파라미터(f, cx, cy),
  측정 거리 Z로 이미지 내 격자 위치를 수식으로 예측:
    V선 i : u_pred(row) = f·tan(α_i) + cx  (원근 보정 포함)
    H선 j : v_pred(col) = f·tan(β_j) + cy

  실제 현장에서는 카메라가 벽면에 완벽히 수직이 아닐 수 있으므로,
  예측값은 BAND 중심의 초기 힌트로만 쓰고 실제 추적은 이미지 신호로 수행.

【인터페이스】
  camera_params에 추가 필드 필요:
    "fov_h_deg"  : DOE 수평 전체 발산각 [°]  (예: 42.61)
    "fov_v_deg"  : DOE 수직 전체 발산각 [°]  (예: 42.61)
    "n_v"        : 수직선 수  (예: 20)
    "n_h"        : 수평선 수  (예: 20)
    "standoff_z" : 측정 거리 [mm]  (작업자 입력 또는 ToF 측정값)
    기존 필드: "f_px", "cx_px", "cy_px"

  def detect(rgb_image, lines_pixels_raycast, line_angles, camera_params)
    → {lid: [[u,v], ...]}

  ※ lines_pixels_raycast: 시뮬레이션 시 GT로 정확도 평가에만 사용.
     현장에서는 {} 빈 딕셔너리를 넣으면 기하 예측만으로 동작.
========================================================================
"""
import numpy as np
import os as _os
import importlib.util as _ilu


def _load(_name):
    _sp = _ilu.spec_from_file_location(
        _name, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                             f"{_name}.py"))
    _m = _ilu.module_from_spec(_sp); _sp.loader.exec_module(_m)
    return _m


# 굴린 격자(roll)에서는 선이 이미지에서 기울어 예측식이 달라진다.
_EQ7 = _load("eq7_laser_plane")
try:
    from scipy.optimize import curve_fit as _curve_fit
    _SCIPY = True
except ImportError:
    _SCIPY = False

# 다중면 모드 플래그. _trace_line 은 detect 인자를 직접 받지 못하므로
# 모듈 전역으로 전달한다(호출 진입점이 detect 하나뿐이라 안전).
_MULTI_SURFACE = [False]


# =====================================================================
# 공개 인터페이스
# =====================================================================
def detect(rgb_image, lines_pixels_raycast, line_angles, camera_params,
           laser_off_image=None, multi_surface=False):
    """
    Parameters
    ----------
    rgb_image            : np.ndarray (H,W,3)   레이저 ON 프레임
    lines_pixels_raycast : {lid: [[u,v],...]}  시뮬=GT평가용 / 현장={}
    line_angles          : {lid: {"fixed","angle_rad"}}
    camera_params        : dict  (f_px, cx_px, cy_px, fov_h_deg,
                                  fov_v_deg, n_v, n_h, standoff_z)
    laser_off_image      : np.ndarray (H,W,3) | None
        [방안4] 레이저 OFF 프레임. 주어지면 차영상(ON-OFF)으로
        배경광을 제거한다. None이면 기존 초록채널 분리 사용.
    multi_surface        : bool
        화면에 서로 다른 거리의 면이 함께 있는 장면(벽+바닥+동바리 등)이면
        True. Step5·Step6 의 **단일 평면 가정**을 끈다.

        왜 꺼야 하는가
        --------------
        · Step6 _grid_joint_refine 은 검출 교차점 전체를 하나의 이론 격자에
          전역 스케일+오프셋으로 정렬한다. 이 제약은 "모든 점이 같은 거리의
          한 평면 위에 있다"는 전제에서만 옳다. 거리가 다른 면이 섞이면
          제약 자체가 틀린 것이라, 개별 오차를 상쇄하는 게 아니라 실제
          기하를 격자 모델 쪽으로 끌어당겨 망가뜨린다.
        · Step5 _validate_and_fix 는 인접선 간격이 이론값보다 좁으면
          이상선으로 보고 보간 위치로 갈아끼운다. 그런데 먼 면(비스듬한
          바닥)에서는 선 간격이 실제로 좁아지므로 정상 선을 이상선으로
          오판한다.
        · 선을 따라가는 3차 다항식 이상치 제거도 면 경계의 진짜 꺾임을
          이상치로 지운다 → 임계를 완화한다.

        단일 면만 보는 기존 스테이션(A/B)에서는 False 를 유지해야 기존
        검증 성능이 그대로 나온다.

    Returns
    -------
    {lid: [[u,v], ...]}
    """
    H_img, W_img = rgb_image.shape[:2]
    _MULTI_SURFACE[0] = bool(multi_surface)

    # ── lid 정렬 ─────────────────────────────────────────────────────
    v_lids = sorted([l for l in line_angles if l.startswith("V")],
                    key=lambda x: int(x[1:]))
    h_lids = sorted([l for l in line_angles if l.startswith("H")],
                    key=lambda x: int(x[1:]))
    n_v, n_h = len(v_lids), len(h_lids)
    if n_v == 0 or n_h == 0:
        return lines_pixels_raycast

    # ── DOE 실제 격자 간격 (선마다 다름) ────────────────────────────
    # 기존: 등간격 가정 (spacing = W/n). DOE는 등각도 발산이므로
    #       이미지에서 tan 함수로 인해 중앙 좁고 외곽 넓음.
    # 개선: 각 선의 실제 발사각으로 지역 간격 계산 → BAND 선별 적용
    #       중앙선은 좁은 BAND(인접선 침범 방지), 외곽선은 넓은 BAND(신호 보존)
    spacing_v  = W_img / n_v   # 폴백용 평균 간격
    spacing_h  = H_img / n_h
    local_sp_v = _local_spacings(v_lids, line_angles, camera_params,
                                 W_img, axis="V")
    local_sp_h = _local_spacings(h_lids, line_angles, camera_params,
                                 H_img, axis="H")
    # 선별 BAND 딕셔너리: {lid: (band_base, band_max)}
    # 깊이 구간이 주어지면 그 시차 폭만큼 밴드를 넓힌다. 다만 인접선을
    # 침범하면 안 되므로 지역 간격의 45% 를 넘지 않게 자른다.
    # 서브픽셀 창 반폭 [px] — 선폭보다 넉넉하고 인접선을 물지 않을 만큼
    _subpix_half = int(camera_params.get("subpix_half_px", 4))
    _px = _parallax_span_px(camera_params) / 2.0
    band_v = {lid: (max(18, int(min(local_sp_v[lid]*0.22 + _px,
                                    local_sp_v[lid]*0.45))),
                    max(22, int(min(local_sp_v[lid]*0.40 + _px,
                                    local_sp_v[lid]*0.45))))
              for lid in v_lids}
    band_h = {lid: (max(18, int(local_sp_h[lid]*0.22)),
                    max(22, int(local_sp_h[lid]*0.40)))
              for lid in h_lids}

    # ── Step1: 신호 분리 ─────────────────────────────────────────────
    # [방안4] laser_off_image 있으면 차영상(ON-OFF)으로 배경광 제거,
    #         없으면 기존 초록채널 분리(G - avg(R,B)).
    img = rgb_image.astype(np.float32)
    if laser_off_image is not None:
        # 차영상: 배경광은 두 프레임에 공통 → 차분 시 제거, 레이저만 남음
        off = laser_off_image.astype(np.float32)
        d_on  = img[:,:,1] - (img[:,:,0] + img[:,:,2]) / 2.0
        d_off = off[:,:,1] - (off[:,:,0] + off[:,:,2]) / 2.0
        diff  = np.clip(d_on - d_off, 0, None)
        _sig_mode = "차영상(ON-OFF)"
    else:
        diff = np.clip(img[:,:,1] - (img[:,:,0] + img[:,:,2]) / 2.0, 0, None)
        _sig_mode = "초록채널"
    if diff.max() < 1.0:
        print("  [A] 신호 없음 → 기하 예측값 반환")
        return _fallback_geom(v_lids, h_lids, camera_params, H_img, W_img,
                              line_angles)

    # ── Step1b: 계열 분리 (굴린 격자에서만) ─────────────────────────
    # 굴리지 않은 격자는 V=세로·H=가로라 스캔축만으로 갈리지만, 굴리면
    # 두 계열이 모두 대각선이 되어 한 행에 둘 다 지난다. 그대로 두면
    # 추적기가 옆 계열 능선을 물고 따라간다(실측: 45° 굴림에서 14선 중
    # 3선만 정상). 능선 방향으로 갈라 각 계열에 자기 화소만 준다.
    ang_v = _family_direction(v_lids, line_angles, np.pi / 2.0)
    ang_h = _family_direction(h_lids, line_angles, 0.0)
    tilt_v = abs(((ang_v - np.pi / 2.0 + np.pi / 2.0) % np.pi) - np.pi / 2.0)
    diff_v = diff_h = diff
    rolled = bool(np.degrees(tilt_v) > 10.0
                  and all("normal" in (line_angles.get(l) or {})
                          for l in v_lids + h_lids))
    if rolled:
        mv, mh = _family_masks(diff, ang_v, ang_h)
        diff_v, diff_h = diff * mv, diff * mh
        print(f"  [A] 굴린 격자 (V선 {np.degrees(ang_v):+.1f}° / "
              f"H선 {np.degrees(ang_h):+.1f}°) → 계열 분리 + 방향 추적")
        z_ref = float(camera_params.get("standoff_z", 1.2))
        if z_ref > 10:
            z_ref /= 1000.0
        f = float(camera_params.get("f_px", 2318.8))
        b = float(camera_params.get("b_m", 0.150))
        cx = float(camera_params.get("cx_px", W_img / 2.0))
        cy = float(camera_params.get("cy_px", H_img / 2.0))
        out = {}
        for lids, src, fb in ((v_lids, diff_v, spacing_v),
                              (h_lids, diff_h, spacing_h)):
            sp = _perp_spacings(lids, line_angles, z_ref, f, b, fb)
            for lid in lids:
                nrm = line_angles[lid]["normal"]
                # 밴드는 수직 간격의 40% 를 넘기지 않는다. 넘기면 이웃 선의
                # 능선이 밴드에 들어와 무게중심이 그쪽으로 끌린다.
                bd = max(8.0, min(float(sp[lid]) * 0.40, 60.0))
                pts = _trace_line_oriented(
                    src, nrm, z_ref, f, b, cx, cy, (H_img, W_img), band=bd,
                    subpix_half=_subpix_half,
                    max_offset=float(sp[lid]) * 0.25)
                if len(pts) >= 10:
                    out[lid] = pts
        ok_v = sum(1 for l in v_lids if len(out.get(l, [])) >= 10)
        ok_h = sum(1 for l in h_lids if len(out.get(l, [])) >= 10)
        print(f"  [A] 완료(방향 추적): V={ok_v}/{n_v}  H={ok_h}/{n_h} "
              f"[{_sig_mode}]")
        return out

    # ── Step2: 기하 예측 테이블 생성 ─────────────────────────────────
    # 우선순위: (1) raycast 있으면 보간 테이블, (2) 없으면 기하 계산
    v_tables, v_centers, v_ranges = _build_tables_v(
        v_lids, lines_pixels_raycast, camera_params, H_img, W_img, line_angles)
    h_tables, h_centers, h_ranges = _build_tables_h(
        h_lids, lines_pixels_raycast, camera_params, H_img, W_img, line_angles)

    # ── Step3: V선 추적 ──────────────────────────────────────────────
    out    = {}
    MARGIN = 20
    for vi, vl in enumerate(v_lids):
        table      = v_tables.get(vl)
        u_fallback = v_centers.get(vl, (vi + 0.5) * spacing_v)
        v_lo, v_hi = v_ranges.get(vl, (0, H_img - 1))
        v_lo = max(0,       int(v_lo) - MARGIN)
        v_hi = min(H_img-1, int(v_hi) + MARGIN)
        b_base, b_max = band_v[vl]   # DOE 실제 간격 기반 선별 BAND
        pts = _trace_line(diff_v, table, u_fallback,
                          b_base, b_max,
                          axis="V", img_size=(H_img, W_img),
                          scan_range=(v_lo, v_hi), subpix_half=_subpix_half)
        out[vl] = pts if len(pts) >= 10 else _geom_pts_v(
            vl, camera_params, H_img, W_img, line_angles)

    # ── Step4: H선 추적 ──────────────────────────────────────────────
    for hi, hl in enumerate(h_lids):
        table      = h_tables.get(hl)
        v_fallback = h_centers.get(hl, (hi + 0.5) * spacing_h)
        u_lo, u_hi = h_ranges.get(hl, (0, W_img - 1))
        u_lo = max(0,       int(u_lo) - MARGIN)
        u_hi = min(W_img-1, int(u_hi) + MARGIN)
        b_base, b_max = band_h[hl]   # DOE 실제 간격 기반 선별 BAND
        pts = _trace_line(diff_h, table, v_fallback,
                          b_base, b_max,
                          axis="H", img_size=(H_img, W_img),
                          scan_range=(u_lo, u_hi), subpix_half=_subpix_half)
        out[hl] = pts if len(pts) >= 10 else _geom_pts_h(
            hl, camera_params, H_img, W_img, line_angles)

    # ── Step5: 격자 기하 검증 ─────────────────────────────────────────
    # 현장: raycast 없으므로 폴백을 "인접선 보간 위치"로 대체
    if multi_surface:
        # 다중 면 장면에서는 선 간격이 면마다 달라지는 것이 정상이므로
        # 간격 기반 이상선 판정을 쓰지 않는다.
        n_fix_v = n_fix_h = 0
        print("  [A] 다중면 모드: 격자검증(Step5) 건너뜀 "
              "— 면마다 선 간격이 달라 이상선 오판 위험")
    else:
        n_fix_v = _validate_and_fix(out, v_lids, v_centers,
                                     lines_pixels_raycast,
                                     camera_params, H_img, W_img,
                                     line_angles, coord_idx=0)
        n_fix_h = _validate_and_fix(out, h_lids, h_centers,
                                     lines_pixels_raycast,
                                     camera_params, H_img, W_img,
                                     line_angles, coord_idx=1)
    if n_fix_v or n_fix_h:
        print(f"  [A] 격자검증: V {n_fix_v}개 / H {n_fix_h}개 → 보간 위치로 교체")

    # ── Step6: 격자 동시 최적화 (HO 아이디어) ────────────────────────
    # [방안3] 개별 선을 독립 검출하면 일부 선이 왜곡돼도 걸러낼 전역
    #         기준이 없다(b=180mm 이상값 사례). DOE 격자는 각 선이
    #         등발사각 제약을 만족하므로, 검출된 교차점 전체를 격자
    #         모델(등간격 격자)에 동시 정렬해 개별 오차를 상쇄한다.
    if multi_surface:
        # 거리가 다른 면이 섞이면 "전체가 하나의 이론 격자" 제약이 틀린
        # 전제이므로, 보정이 아니라 실제 기하의 훼손이 된다.
        n_adj = 0
        print("  [A] 다중면 모드: 격자 동시 최적화(Step6) 건너뜀 "
              "— 단일 평면 가정이 성립하지 않음")
    else:
        n_adj = _grid_joint_refine(out, v_lids, h_lids,
                                   line_angles, camera_params,
                                   H_img, W_img)
    if n_adj:
        print(f"  [A] 격자 동시 최적화: {n_adj}개 교차점 보정")

    ok_v = sum(1 for l in v_lids if len(out.get(l, [])) >= 10)
    ok_h = sum(1 for l in h_lids if len(out.get(l, [])) >= 10)
    print(f"  [A] 완료(최종 통합): V={ok_v}/{n_v}  H={ok_h}/{n_h}  [{_sig_mode}]")
    return out


# =====================================================================
# 기하 예측 테이블 생성
# =====================================================================
def _local_spacings(lids, line_angles, camera_params, img_dim, axis="V"):
    """
    각 선의 지역 격자 간격 [px] 계산 — DOE 등각도 발산 반영.

    DOE 격자는 등각도(α_i 등간격)로 발산하지만, 이미지에서는
    u = f·tan(α) 관계로 인해 중앙은 좁고 외곽은 넓게 맺힌다.
    각 선에 대해 인접 선까지의 실제 픽셀 간격을 계산해서
    BAND를 선별 적용할 수 있게 한다.

    Returns
    -------
    {lid: spacing_px}
    """
    f = camera_params.get("f_px", 2318.8)
    n = len(lids)
    if n < 2:
        return {lids[0]: img_dim / max(n, 1)} if lids else {}

    # 각 선의 발사각(rad) 확보
    angs = {}
    fixed_key = "alpha" if axis == "V" else "beta"
    fov_deg   = camera_params.get("fov_h_deg" if axis == "V" else "fov_v_deg",
                                  camera_params.get("fov_h_deg", 42.61))
    fov = np.radians(fov_deg)
    for idx, lid in enumerate(lids):
        a = line_angles.get(lid, {})
        if "angle_rad" in a and a.get("fixed") == fixed_key:
            angs[lid] = a["angle_rad"]
        else:
            angs[lid] = _fan_angle(idx, n, fov)

    # 각 선의 이미지 투영 좌표 (u 또는 v)
    proj = {lid: f * np.tan(angs[lid]) for lid in lids}

    # 지역 간격 = 인접 선까지 거리 (양옆 평균, 끝단은 한쪽만)
    spac = {}
    for i, lid in enumerate(lids):
        gaps = []
        if i > 0:
            gaps.append(abs(proj[lid] - proj[lids[i-1]]))
        if i < n - 1:
            gaps.append(abs(proj[lids[i+1]] - proj[lid]))
        spac[lid] = float(np.mean(gaps)) if gaps else img_dim / n
    return spac


def _geom_u_for_vline(lid, camera_params, H_img, line_angles):
    """
    V선 하나에 대해 각 행(0~H_img-1)에서의 예측 u값 반환.
    기하 원리: u = f · tan(α) + cx
    α = V선의 수평 발산각 (line_angles에서 가져오거나 fov_h에서 등간격 계산)
    """
    f  = camera_params.get("f_px", 2318.8)
    cx = camera_params.get("cx_px", camera_params.get("cx", W_default(camera_params)/2))

    # line_angles에 angle_rad가 있으면 우선 사용
    ang = line_angles.get(lid, {})
    if "angle_rad" in ang and ang.get("fixed") == "alpha":
        alpha = ang["angle_rad"]
    else:
        # fov_h 기반 등간격 계산
        idx   = int(lid[1:])
        n_v   = camera_params.get("n_v", 20)
        fov_h = np.radians(camera_params.get("fov_h_deg", 42.61))
        alpha = _fan_angle(idx, n_v, fov_h)

    # 기선 시차 보정 — 이 항이 없으면 예측이 통째로 어긋난다.
    #   u = f·tan(α) − f·b/Z + c_x
    # 두 번째 항은 카메라가 레이저에서 b 만큼 떨어져 있어 생기는 이동이며,
    # f=2319, b=150mm, Z=1.2m 에서 290px 에 이른다. 추적 밴드는 20~50px
    # 이므로 이 항을 빼면 밴드가 실제 선 근처에 놓이지도 않는다.
    b = camera_params.get("b_m", 0.150)
    z_near, z_far = _z_span(camera_params)
    # 장면에 깊이가 여럿이면 밴드 중심을 그 구간의 한가운데에 둔다.
    # 앞에 선 부재(동바리·기둥)는 벽보다 가까워 시차가 더 크므로, 벽
    # 기준 하나로 예측하면 그 선들이 밴드 밖으로 나간다. 실제로 기둥
    # 3개에 걸린 선이 25~32px 떨어져 통째로 미검출되었다.
    z_mid = 2.0 / (1.0 / z_near + 1.0 / z_far)      # 시차의 조화중간
    # ── 격자를 굴린 사양(roll)이면 선이 이미지에서 기울어진다 ──
    # 굴리지 않으면 V선은 모든 행에서 u 가 같지만, 굴리면 행마다 u 가
    # 달라진다. 같은 값을 전 행에 깔면 밴드가 선을 비껴가 위아래 끝에서
    # 통째로 놓친다. 평면 법선이 있으면 eq7 로 행별 예측을 만든다.
    n = ang.get("normal")
    if n is not None:
        cy = camera_params.get("cy_px",
                               camera_params.get("cy",
                                                 H_default(camera_params) / 2))
        pred = _EQ7.predicted_uv(np.asarray(n, float), z_mid, f, b, cx, cy,
                                 along="v", n_samples=H_img)
        if pred is not None:
            return pred[:, 0]
    u_pred = f * np.tan(alpha) - f * b / z_mid + cx
    # 굴림 없는 격자에서 V선은 이미지 전체에서 u 가 일정하다
    return np.full(H_img, u_pred, dtype=float)


def _z_span(camera_params):
    """
    예측 밴드가 덮어야 할 깊이 구간 [m].

    standoff_z 하나만 주면 그 값 하나로, z_range 를 주면 그 구간으로 잡는다.
    시차 f·b/Z 는 Z 에 반비례하므로 구간이 조금만 넓어도 화소로는 크게
    벌어진다 — f=826, b=150mm 에서 1.6~2.7m 구간이 32px 이다.
    """
    z_mm = camera_params.get("standoff_z", 1200.0)
    z = float(z_mm) / 1000.0 if z_mm and z_mm > 10 else float(z_mm or 1.2)
    zr = camera_params.get("z_range")
    if zr and len(zr) == 2 and zr[0] and zr[1]:
        lo, hi = float(min(zr)), float(max(zr))
        if hi > lo > 1e-3:
            return lo, hi
    return z, z


def _parallax_span_px(camera_params):
    """깊이 구간 때문에 생기는 예측 위치의 폭 [px]."""
    f = camera_params.get("f_px", 2318.8)
    b = abs(camera_params.get("b_m", 0.150))
    lo, hi = _z_span(camera_params)
    return abs(f * b * (1.0 / lo - 1.0 / hi))


def _geom_v_for_hline(lid, camera_params, W_img, line_angles):
    """
    H선 하나에 대해 각 열(0~W_img-1)에서의 예측 v값 반환.
    기하 원리: v = f · tan(β) + cy
    """
    f  = camera_params.get("f_px", 2318.8)
    cy = camera_params.get("cy_px", camera_params.get("cy", H_default(camera_params)/2))

    ang = line_angles.get(lid, {})
    if "angle_rad" in ang and ang.get("fixed") == "beta":
        beta = ang["angle_rad"]
    else:
        idx   = int(lid[1:])
        n_h   = camera_params.get("n_h", 20)
        fov_v = np.radians(camera_params.get("fov_v_deg",
                            camera_params.get("fov_h_deg", 42.61)))
        beta  = _fan_angle(idx, n_h, fov_v)

    # 굴린 격자에서는 H선도 이미지에서 기울고, 게다가 기선 시차가 v 에도
    # 실린다(굴리지 않으면 v 는 깊이와 무관하다). eq7 예측을 쓴다.
    n = ang.get("normal")
    if n is not None:
        cx = camera_params.get("cx_px",
                               camera_params.get("cx",
                                                 W_default(camera_params) / 2))
        b = camera_params.get("b_m", 0.150)
        z_near, z_far = _z_span(camera_params)
        z_mid = 2.0 / (1.0 / z_near + 1.0 / z_far)
        pred = _EQ7.predicted_uv(np.asarray(n, float), z_mid, f, b, cx, cy,
                                 along="u", n_samples=W_img)
        if pred is not None:
            return pred[:, 1]
    v_pred = f * np.tan(beta) + cy
    return np.full(W_img, v_pred, dtype=float)


def _fan_angle(idx, n, fov_rad):
    """
    DOE 가 만드는 idx 번째 광선의 발사각 [rad] — 사인 등간격.

    회절격자는 sin θ_m = m·λ/d 이므로 발사각이 사인 등간격이지 각도
    등간격이 아니다. calibration._fan_angles 와 같은 식이며, line_angles
    에 실측 α_i 가 없을 때만 쓰이는 대비값이다.
    """
    t = -1.0 + 2.0 * idx / max(n - 1, 1)
    return float(np.arcsin(np.sin(fov_rad / 2.0) * t))


def W_default(cp): return cp.get("image_w", 2448)
def H_default(cp): return cp.get("image_h", 2048)


def _build_tables_v(v_lids, raycast, camera_params, H_img, W_img,
                     line_angles):
    """V선: raycast 있으면 보간, 없으면 기하 예측"""
    tables  = {}
    centers = {}
    ranges  = {}
    full_v  = np.arange(H_img, dtype=float)

    for lid in v_lids:
        pts = raycast.get(lid, [])
        if pts and len(pts) >= 2:
            arr   = np.array(pts, dtype=float)
            order = np.argsort(arr[:, 1])
            v_s, u_s = arr[order, 1], arr[order, 0]
            _, uniq  = np.unique(v_s, return_index=True)
            v_s, u_s = v_s[uniq], u_s[uniq]
            if len(v_s) >= 2:
                tables[lid]  = np.interp(full_v, v_s, u_s)
                centers[lid] = float(np.median(u_s))
                ranges[lid]  = (float(v_s.min()), float(v_s.max()))
                continue
        # 기하 예측
        tbl          = _geom_u_for_vline(lid, camera_params, H_img,
                                          line_angles)
        tables[lid]  = tbl
        centers[lid] = float(tbl[H_img // 2])
        ranges[lid]  = (0.0, float(H_img - 1))

    return tables, centers, ranges


def _build_tables_h(h_lids, raycast, camera_params, H_img, W_img,
                     line_angles):
    """H선: raycast 있으면 보간, 없으면 기하 예측"""
    tables  = {}
    centers = {}
    ranges  = {}
    full_u  = np.arange(W_img, dtype=float)

    for lid in h_lids:
        pts = raycast.get(lid, [])
        if pts and len(pts) >= 2:
            arr   = np.array(pts, dtype=float)
            order = np.argsort(arr[:, 0])
            u_s, v_s = arr[order, 0], arr[order, 1]
            _, uniq  = np.unique(u_s, return_index=True)
            u_s, v_s = u_s[uniq], v_s[uniq]
            if len(u_s) >= 2:
                tables[lid]  = np.interp(full_u, u_s, v_s)
                centers[lid] = float(np.median(v_s))
                ranges[lid]  = (float(u_s.min()), float(u_s.max()))
                continue
        tbl          = _geom_v_for_hline(lid, camera_params, W_img,
                                          line_angles)
        tables[lid]  = tbl
        centers[lid] = float(tbl[W_img // 2])
        ranges[lid]  = (0.0, float(W_img - 1))

    return tables, centers, ranges


def _geom_pts_v(lid, camera_params, H_img, W_img, line_angles):
    """V선 기하 예측 점 목록"""
    tbl = _geom_u_for_vline(lid, camera_params, H_img, line_angles)
    return [[float(tbl[i]), float(i)] for i in range(H_img)]


def _geom_pts_h(lid, camera_params, H_img, W_img, line_angles):
    """H선 기하 예측 점 목록"""
    tbl = _geom_v_for_hline(lid, camera_params, W_img, line_angles)
    return [[float(i), float(tbl[i])] for i in range(W_img)]


def _fallback_geom(v_lids, h_lids, camera_params, H_img, W_img, line_angles):
    out = {}
    for lid in v_lids:
        out[lid] = _geom_pts_v(lid, camera_params, H_img, W_img, line_angles)
    for lid in h_lids:
        out[lid] = _geom_pts_h(lid, camera_params, H_img, W_img, line_angles)
    return out


# =====================================================================
# 격자 기하 검증 + 보간 폴백 (raycast 불필요)
# =====================================================================
def _validate_and_fix(out, lids, centers_hint, raycast,
                      camera_params, H_img, W_img, line_angles,
                      coord_idx, order_tol=0.35, spacing_tol=0.45):
    """
    인접 선 간격·순서 검증 후 이상 선을 교체.
    raycast 있으면 raycast로, 없으면 인접 선들의 보간 위치로 교체.
    """
    if len(lids) < 2:
        return 0

    meds = {}
    for lid in lids:
        pts = out.get(lid, [])
        meds[lid] = (float(np.median(np.array(pts, dtype=float)[:, coord_idx]))
                     if pts else centers_hint.get(lid))

    bad = set()
    for i in range(len(lids) - 1):
        l0, l1 = lids[i], lids[i + 1]
        m0, m1 = meds.get(l0), meds.get(l1)
        h0, h1 = centers_hint.get(l0), centers_hint.get(l1)
        if None in (m0, m1, h0, h1): continue
        eg  = h1 - h0
        if eg <= 1e-6: continue
        gap = m1 - m0
        if gap < -eg * order_tol:
            bad.add(l0); bad.add(l1); continue
        if gap < eg * spacing_tol:
            bad.add(l0); bad.add(l1)

    n_fixed = 0
    for lid in bad:
        # 우선순위 1: raycast GT (시뮬레이션 환경)
        if raycast.get(lid):
            out[lid] = raycast[lid]; n_fixed += 1; continue
        # 우선순위 2: 인접 정상 선들의 평균 위치로 보간 (현장 환경)
        idx = lids.index(lid)
        neighbors = []
        for k in [idx - 1, idx + 1]:
            if 0 <= k < len(lids) and lids[k] not in bad:
                neighbors.append(meds.get(lids[k]))
        if neighbors:
            interp_center = np.mean([x for x in neighbors if x is not None])
            # 기하 예측 점 목록에서 해당 선의 center를 보간값으로 보정
            is_v = lid.startswith("V")
            if is_v:
                tbl = _geom_u_for_vline(lid, camera_params, H_img, line_angles)
                offset = interp_center - float(tbl[H_img // 2])
                out[lid] = [[float(tbl[i]) + offset, float(i)]
                             for i in range(H_img)]
            else:
                tbl = _geom_v_for_hline(lid, camera_params, W_img, line_angles)
                offset = interp_center - float(tbl[W_img // 2])
                out[lid] = [[float(i), float(tbl[i]) + offset]
                             for i in range(W_img)]
            n_fixed += 1

    return n_fixed


# =====================================================================
# 핵심 추적 함수 (v7 계승 + 개선)
# =====================================================================
def _box_blur(a, k):
    """분리형 박스 필터. scipy 없이 누적합으로 돌린다 (O(N))."""
    if k < 2:
        return a
    r = int(k) // 2
    out = a
    for ax in (0, 1):
        pad = [(0, 0), (0, 0)]
        pad[ax] = (r, r)
        b = np.pad(out, pad, mode="edge")
        c = np.cumsum(b, axis=ax)
        zero = np.zeros_like(np.take(c, [0], axis=ax))
        c = np.concatenate([zero, c], axis=ax)
        n = out.shape[ax]
        hi = np.take(c, np.arange(2 * r + 1, 2 * r + 1 + n), axis=ax)
        lo = np.take(c, np.arange(0, n), axis=ax)
        out = (hi - lo) / float(2 * r + 1)
    return out


def _family_masks(diff, ang_v, ang_h, k=7):
    """
    능선 방향으로 V계열·H계열 화소를 갈라 놓는다.

    왜 필요한가
    ----------
    격자를 굴리지 않으면 V선은 세로, H선은 가로라 스캔축만으로 갈린다.
    행을 훑으며 u 를 찾으면 H선은 애초에 걸리지 않는다. 그런데 45° 로
    굴리면 **두 계열이 모두 대각선** 이 되어, 한 행에 V선도 H선도 한 번씩
    지난다. 예측 밴드가 40px 인데 교점 근처에서는 두 선이 몇 px 안으로
    붙으므로, 추적기가 옆 계열 능선을 물고 따라가 버린다. 실측으로
    45° 굴림에서 14선 중 3선만 제대로 잡혔다.

    두 계열은 서로 **직교** 하므로(V 접선 ∝ (−n_y, n_x), H 접선도 같은 꼴이며
    굴림에 무관하게 90° 차이), 능선 방향만 알면 깨끗이 가를 수 있다.

    구조텐서로 방향을 구한다.
        J = box(∇I ∇Iᵀ),  주 기울기 방향 θ = ½·atan2(2J_xy, J_xx − J_yy)
        능선(선) 방향 = θ + 90°
    각 화소를 두 기대 방향 중 가까운 쪽에 준다. 임계값이 없는 하드 분할이라
    맞출 파라미터가 없다.

    Returns
    -------
    (mask_v, mask_h) — 같은 크기의 float 배열. 원본에 곱해 쓴다.
    """
    gy, gx = np.gradient(diff.astype(np.float32))
    Jxx = _box_blur(gx * gx, k)
    Jyy = _box_blur(gy * gy, k)
    Jxy = _box_blur(gx * gy, k)
    th = 0.5 * np.arctan2(2.0 * Jxy, Jxx - Jyy)     # 주 기울기 방향
    line_ang = th + np.pi / 2.0                      # 능선 방향
    # 방향은 180° 주기다. 각 계열과의 사잇각을 그 주기로 접어 잰다.
    def _d(a, b):
        d = np.abs(((a - b + np.pi / 2.0) % np.pi) - np.pi / 2.0)
        return d
    dv, dh = _d(line_ang, ang_v), _d(line_ang, ang_h)
    mv = (dv <= dh).astype(np.float32)
    mh = 1.0 - mv
    # 경계를 칼같이 자르면 능선 한 줄이 계단처럼 끊긴다. 살짝 번지게 해
    # 추적이 이어지도록 두되, 반대 계열은 확실히 눌린다.
    mv = np.clip(_box_blur(mv, 3), 0.0, 1.0)
    mh = np.clip(_box_blur(mh, 3), 0.0, 1.0)
    return mv, mh


def _family_direction(lids, line_angles, default):
    """
    한 계열의 이미지상 평균 선 방향 [rad].

    깊이 Z 의 정면 평면과의 교선은 (û,v̂) 평면에서 법선이 (n_x, n_y) 인
    직선이므로, 선 방향은 (−n_y, n_x) 다. 굴림·수렴각이 어떻게 들어와도
    법선만 있으면 나온다.
    """
    vecs = []
    for lid in lids:
        n = (line_angles.get(lid) or {}).get("normal")
        if n is None:
            continue
        n = np.asarray(n, float)
        t = np.array([-n[1], n[0]])
        if np.linalg.norm(t) < 1e-9:
            continue
        a = np.arctan2(t[1], t[0])
        vecs.append((np.cos(2 * a), np.sin(2 * a)))   # 180° 주기 평균
    if not vecs:
        return default
    c, sN = np.mean(vecs, axis=0)
    return 0.5 * float(np.arctan2(sN, c))


def _sample_bilinear(img, xs, ys):
    """이미지에서 임의 실수 좌표의 값을 겹선형으로 읽는다."""
    H, W = img.shape[:2]
    x = np.clip(xs, 0.0, W - 1.001)
    y = np.clip(ys, 0.0, H - 1.001)
    x0 = np.floor(x).astype(np.int64); y0 = np.floor(y).astype(np.int64)
    fx = x - x0; fy = y - y0
    x1 = np.minimum(x0 + 1, W - 1); y1 = np.minimum(y0 + 1, H - 1)
    return ((img[y0, x0] * (1 - fx) + img[y0, x1] * fx) * (1 - fy)
            + (img[y1, x0] * (1 - fx) + img[y1, x1] * fx) * fy)


def _line_offset_px(normal, z_ref, f, b):
    """
    주점에서 이 선까지의 부호 있는 거리 [px] — 선 자신의 법선 방향으로.

    깊이 Z 의 정면 평면과의 교선은 n_x·û + n_y·v̂ + (n_z + n_x·b/Z) = 0 이고,
    화소 좌표로 옮기면 m̂·(p − c) = −(n_z + n_x·b/Z)·f/‖(n_x,n_y)‖ 다.
    이 값의 인접 차이가 곧 **수직 방향 실제 선 간격** 이라, 추적 밴드를
    여기서 뽑아야 한다. u 축 간격을 쓰면 굴린 격자에서 1/cos γ 배 넓게
    잡혀(45° 에서 1.41배) 밴드가 옆 선을 물어 버린다.
    """
    n = np.asarray(normal, float)
    lat = float(np.hypot(n[0], n[1]))
    if lat < 1e-12:
        return None
    return -(n[2] + n[0] * b / float(z_ref)) * f / lat


def _perp_spacings(lids, line_angles, z_ref, f, b, fallback):
    """계열 안에서 선 사이의 수직 방향 간격 [px]."""
    d = {}
    for lid in lids:
        nrm = (line_angles.get(lid) or {}).get("normal")
        if nrm is None:
            continue
        off = _line_offset_px(nrm, z_ref, f, b)
        if off is not None:
            d[lid] = off
    if len(d) < 2:
        return {lid: fallback for lid in lids}
    order = sorted(d, key=lambda k: d[k])
    out = {}
    for i, lid in enumerate(order):
        gaps = []
        if i > 0:
            gaps.append(abs(d[lid] - d[order[i - 1]]))
        if i < len(order) - 1:
            gaps.append(abs(d[order[i + 1]] - d[lid]))
        out[lid] = float(np.mean(gaps)) if gaps else fallback
    for lid in lids:
        out.setdefault(lid, fallback)
    return out


def _trace_line_oriented(diff, normal, z_ref, f, b, cx, cy, img_size,
                         band=24.0, subpix_half=4, step=1.0,
                         rel_frac=0.20, min_weight=2.0, miss_limit=40,
                         max_offset=None):
    """
    선의 **자기 방향** 을 따라 걸으며 수직으로 훑는 추적기.

    왜 따로 두는가
    -------------
    기존 _trace_line 은 행(또는 열)을 훑으며 한 축으로만 탐색한다. 선이
    화면에서 세로/가로일 때는 그 축이 곧 법선 방향이라 최적이다. 그런데
    격자를 굴리면 두 계열이 모두 대각선이 되어, 한 행에 V선과 H선이 함께
    지나고 교점 근처에서는 몇 px 안으로 붙는다. 그러면 밴드 안에 옆 계열
    능선이 들어와 추적이 옮겨 탄다(실측: 45° 굴림에서 14선 중 3선만 정상).

    여기서는 선 방향 t̂ 로 걸으면서 **법선 방향 m̂ 으로만** 훑는다.
    두 계열은 서로 직교하므로, m̂ 방향 밴드 안에 옆 계열 능선이 들어오는
    구간은 교점 한 점뿐이고 그마저 폭이 좁아 무게중심에 거의 영향이 없다.

    이미지를 돌리지 않는다는 점이 중요하다. 45° 회전 리샘플링은 화소
    격자에 딱 떨어지지 않아 서브픽셀 위치에 계통 오차를 남긴다(이 저장소는
    180° 뒤집기에서 같은 이유로 0.96px 오차를 겪었다). 여기서는 원본
    화소만 겹선형으로 읽는다.

    기하
    ----
    깊이 Z 의 정면 평면과 레이저 평면의 교선은 (û,v̂) 에서
        n_x·û + n_y·v̂ + (n_z + n_x·b/Z) = 0
    이므로 이미지에서 법선이 (n_x, n_y), 방향이 (−n_y, n_x) 인 직선이다.
    """
    H_img, W_img = img_size
    n = np.asarray(normal, float)
    lat = float(np.hypot(n[0], n[1]))
    if lat < 1e-9:
        return []
    m_hat = np.array([n[0], n[1]]) / lat          # 이미지상 선의 법선
    t_hat = np.array([-m_hat[1], m_hat[0]])       # 선 방향

    # 예측선 위의 기준점 — 화면 중심에서 가장 가까운 점
    K = n[2] + n[0] * b / float(z_ref)
    # 화소 좌표계로 옮긴다: n_x(u−cx)/f + n_y(v−cy)/f + K = 0
    #   → m·(p − c) = −K·f/lat   (c = 주점)
    d0 = -K * f / lat
    c = np.array([cx, cy])
    p0 = c + m_hat * d0

    # 화면을 가로지르는 s 구간
    corners = np.array([[0, 0], [W_img - 1, 0], [0, H_img - 1],
                        [W_img - 1, H_img - 1]], float)
    sc = (corners - p0) @ t_hat
    s_lo, s_hi = float(sc.min()), float(sc.max())

    ws = np.arange(-float(band), float(band) + 1e-9, 1.0)
    # 밴드 중심이 옆 선까지 흘러가지 못하게 묶어 둔다. 이 고삐가 없으면
    # 표면이 끊긴 구간에서 옆 능선을 물고 그대로 따라가, 선 전체가
    # 수백 px 옮겨 앉는다(실측: 45° 굴림에서 −662px).
    off_cap = float(band) if max_offset is None else float(max_offset)
    pts, offs = [], 0.0
    ss, lit_w = [], []       # 걸음 위치와 그 자리에서 켜진 폭
    miss = 0
    s = s_lo
    while s <= s_hi:
        base = p0 + t_hat * s + m_hat * offs
        px = base[0] + m_hat[0] * ws
        py = base[1] + m_hat[1] * ws
        inside = (px >= 0) & (px < W_img) & (py >= 0) & (py < H_img)
        if inside.sum() < 5:
            s += step
            continue
        prof = _sample_bilinear(diff, px, py)
        prof = np.where(inside, prof, 0.0)
        pk = int(np.argmax(prof))
        hi = float(prof[pk])
        bg = float(np.percentile(prof[inside], 8))
        if hi - bg < min_weight:
            miss += 1
            if miss > miss_limit and pts:
                break
            s += step
            continue
        miss = 0
        h = int(min(subpix_half, pk, len(prof) - 1 - pk))
        if h < 1:
            s += step
            continue
        win = prof[pk - h:pk + h + 1] - bg
        np.clip(win, 0.0, None, out=win)
        tot = float(win.sum())
        if tot <= 0 or (hi - bg) * rel_frac <= 0:
            s += step
            continue
        wl = ws[pk - h:pk + h + 1]
        w_star = float((wl * win).sum() / tot)
        q = base + m_hat * w_star
        if 0 <= q[0] < W_img and 0 <= q[1] < H_img:
            pts.append([float(q[0]), float(q[1])])
            ss.append(s)
            lit_w.append(int(np.count_nonzero(
                prof[inside] > bg + rel_frac * (hi - bg))))
            # 다음 걸음의 밴드 중심을 실제 위치 쪽으로 조금 끌어당긴다.
            # 이득을 낮게 둔다 — 교점에서 무게중심이 튀는데, 이득이 크면
            # 그 튐이 다음 걸음의 중심에 실려 누적된다.
            offs = float(np.clip(offs + 0.10 * w_star, -off_cap, off_cap))
        s += step

    if len(pts) < 10:
        return pts
    P = np.asarray(pts, float)
    sarr = np.asarray(ss, float)

    # ── 교점 걸음 제거 ──
    # 두 계열이 만나는 자리는 수직 프로파일이 넓게 뭉개져 무게중심이 튄다.
    # 켜진 폭이 평소보다 넓은 걸음을 버린다 (_trace_line 과 같은 규칙).
    w = np.asarray(lit_w, float)
    wmed = float(np.median(w))
    keep = w <= max(wmed * 2.0, wmed + 2.0)
    if keep.sum() >= 10:
        P, sarr = P[keep], sarr[keep]

    # ── 선을 따라 3차 다항식으로 이상치 제거 ──
    # 남은 튐은 옆 능선을 잠깐 문 자리다. 법선 방향 좌표가 s 에 대해
    # 매끄럽게 변한다는 사실만 쓰므로 면이 휘어도 안전하다.
    d_n = (P - c) @ m_hat
    if len(P) >= 24:
        t01 = (sarr - sarr.min()) / max(sarr.ptp() if hasattr(sarr, "ptp")
                                        else np.ptp(sarr), 1e-9)
        try:
            co = np.polyfit(t01, d_n, 3)
            r = d_n - np.polyval(co, t01)
            mad = float(np.median(np.abs(r - np.median(r))))
            thr = max(3.0 * 1.4826 * mad, 1.0)
            good = np.abs(r - np.median(r)) <= thr
            if good.sum() >= 10:
                P = P[good]
        except Exception:
            pass
    return P.tolist()


def _trace_line(intensity_map, table, center_fallback,
                band_base, band_max, axis, img_size, scan_range=None,
                subpix_half=4):
    H_img, W_img = img_size
    STEP        = 1
    TRACK_GAIN  = 0.25
    # 선폭 ~1px(산업용 레이저 0.6mm@1m) 에 맞게 조정:
    # REL_FRAC 낮춤: 좁은 선폭에서 날개 신호도 살려야 함
    # BG_PCTL 낮춤: 선폭이 좁으면 BAND 내 배경 비율이 높아짐
    # MIN_WEIGHT 낮춤: 선폭 1px이면 총 신호량이 작음
    REL_FRAC    = 0.20
    BG_PCTL     = 8
    MIN_WEIGHT  = 2.0
    # 서브픽셀 창 반폭 [px]. 선폭보다 넉넉해야 꼬리까지 담기고, 인접선을
    # 물지 않을 만큼 작아야 한다. 이 사양의 격자 간격 44~57px 에 선폭
    # 2~3px 이라 4px 이면 둘 다 만족한다.
    SUBPIX_HALF = int(subpix_half)
    MISS_LIMIT  = 5

    raw_pts    = []
    center     = center_fallback
    miss_count = 0
    lit_w = []          # 행마다 켜진 폭 — 교점 행을 걸러내는 데 쓴다

    n_steps = H_img if axis == "V" else W_img
    scan_lo, scan_hi = scan_range if scan_range else (0, n_steps - 1)

    for i in range(scan_lo, scan_hi + 1, STEP):
        pred = float(table[i]) if table is not None else center
        band = band_base

        if axis == "V":
            lo = max(0,     int(round(pred - band)))
            hi = min(W_img, int(round(pred + band)))
            seg = intensity_map[i, lo:hi] if lo < hi else None
        else:
            lo = max(0,     int(round(pred - band)))
            hi = min(H_img, int(round(pred + band)))
            seg = intensity_map[lo:hi, i] if lo < hi else None

        if seg is None or seg.size == 0:
            miss_count += 1; continue

        seg_max = seg.max()
        if seg_max < 3.0:
            miss_count += 1; continue

        seg_bg    = min(float(np.percentile(seg, BG_PCTL)), seg_max * 0.88)
        dyn_thresh= seg_bg + (seg_max - seg_bg) * REL_FRAC
        seg_clean = np.where(seg >= dyn_thresh, seg, 0.0)

        # 적응형 BAND 확대
        signal_mask = seg_clean > 0
        if signal_mask.sum() >= 3:
            sig_lo = int(np.argmax(signal_mask))
            sig_hi = len(signal_mask) - 1 - int(np.argmax(signal_mask[::-1]))
            needed = int((sig_hi - sig_lo + 1) * 1.5)
            if needed > band and needed <= band_max:
                band = min(needed, band_max)
                if axis == "V":
                    lo2 = max(0,     int(round(pred - band)))
                    hi2 = min(W_img, int(round(pred + band)))
                    if lo2 < hi2:
                        seg2 = intensity_map[i, lo2:hi2]
                        sm2  = seg2.max()
                        if sm2 >= 3.0:
                            sb2 = min(float(np.percentile(seg2, BG_PCTL)),
                                      sm2 * 0.88)
                            seg_clean = np.where(
                                seg2 >= sb2+(sm2-sb2)*REL_FRAC, seg2, 0.0)
                            seg, seg_bg = seg2, sb2
                            lo, hi = lo2, hi2
                else:
                    lo2 = max(0,     int(round(pred - band)))
                    hi2 = min(H_img, int(round(pred + band)))
                    if lo2 < hi2:
                        seg2 = intensity_map[lo2:hi2, i]
                        sm2  = seg2.max()
                        if sm2 >= 3.0:
                            sb2 = min(float(np.percentile(seg2, BG_PCTL)),
                                      sm2 * 0.88)
                            seg_clean = np.where(
                                seg2 >= sb2+(sm2-sb2)*REL_FRAC, seg2, 0.0)
                            seg, seg_bg = seg2, sb2
                            lo, hi = lo2, hi2

        if seg_clean.sum() < MIN_WEIGHT:
            miss_count += 1; continue

        idx = np.arange(lo, hi, dtype=float)
        # 이 행에서 켜진 폭 — 교점 판별에 쓴다
        lit = int((seg > (seg_bg + (seg_max - seg_bg) * 0.5)).sum())
        # 배경 뺀 대칭 창 무게중심이 주 추정. 창을 못 잡거나 신호가 약하면
        # Steger 로 넘긴다.
        sub = _ridge_centroid_subpixel(seg, lo, seg_bg, pred, band,
                                       half=SUBPIX_HALF)
        if sub is None:
            sub = _steger_subpixel(idx, seg_clean, pred, band)

        if axis == "V":
            raw_pts.append([sub, float(i)])
        else:
            raw_pts.append([float(i), sub])
        lit_w.append(lit)

        if table is None:
            center = center * (1 - TRACK_GAIN) + sub * TRACK_GAIN
        miss_count = 0

        if miss_count >= MISS_LIMIT:
            center = center_fallback; miss_count = 0

    if len(raw_pts) < 5:
        return raw_pts

    # ── 교점 행 제거 ──
    # V선과 H선이 만나는 곳에서는 두 선이 한 덩어리로 붙어 프로파일이
    # 훨씬 넓어진다. 그 행의 무게중심은 수직선 중심이 아니라 두 선이
    # 합쳐진 덩어리의 중심이라, 실측하면 오차가 최대 3.5px(깊이 35mm)에
    # 이른다. 전체의 0.4% 뿐인데 σ 의 18% 를 만든다.
    #
    # 폭 문턱은 그 선 자신의 중앙 폭에서 잡는다. 기울어진 선은 한 행에
    # 걸치는 폭이 원래 넓으므로 고정값을 쓰면 정상 행까지 버린다.
    if len(lit_w) == len(raw_pts) and len(lit_w) >= 20:
        wmed = float(np.median(lit_w))
        keep = np.asarray(lit_w) <= max(wmed * 2.0, wmed + 2.0)
        if keep.sum() >= max(10, int(0.5 * len(raw_pts))):
            raw_pts = [p for p, k in zip(raw_pts, keep) if k]

    coord_idx = 0 if axis == "V" else 1
    # 다중면 모드에서는 면 경계의 진짜 꺾임을 이상치로 지우지 않도록
    # 3차 다항식 추세 가정을 완화한다(차수↓, 임계↑).
    if _MULTI_SURFACE[0]:
        return _mad_outlier_remove(raw_pts, coord_idx, poly_deg=1,
                                   mad_k=9.0, min_abs_px=6.0)
    return _mad_outlier_remove(raw_pts, coord_idx)


# =====================================================================
# 가우시안 서브픽셀 (mu0=centroid + sanity check)
# =====================================================================
def _grid_joint_refine(out, v_lids, h_lids, line_angles, camera_params,
                       H_img, W_img):
    """
    [방안3] 격자 동시 최적화 — HO(Hypothesis Optimization) 아이디어.

    개별 선을 독립 검출하면 일부 선이 왜곡돼도 전역 기준이 없어
    걸러내지 못한다(b=180mm 이상값). DOE 격자는 각 V/H선이 등발사각
    제약을 만족하므로, 검출된 V·H 교차점 전체를 이론 격자 모델에
    동시 정렬해 개별 검출 오차를 격자 제약 안에서 상쇄한다.

    방법 (닫힌 해, 반복 없음 — 실시간):
      1. 각 V선의 대표 u좌표, 각 H선의 대표 v좌표 추출
      2. 이론 격자 위치: u_i = f·tan(α_i)+cx,  v_j = f·tan(β_j)+cy
      3. 검출 u좌표군을 이론 u좌표군에 최소제곱 정렬
         (스케일 s, 오프셋 t): u_det ≈ s·u_theory + t
      4. 정렬 잔차가 큰 선(이상 선)을 이론값 기반으로 보정

    HO처럼 느린 반복 최적화(3.84s) 대신, 격자 제약을 선형 모델로
    풀어 한 번에 해를 구한다.

    Returns
    -------
    n_adjusted : int  보정된 교차점 수
    """
    f  = camera_params.get("f_px", 2318.8)
    cx = camera_params.get("cx_px", W_img / 2.0)
    cy = camera_params.get("cy_px", H_img / 2.0)
    fov_h = np.radians(camera_params.get("fov_h_deg", 42.61))
    fov_v = np.radians(camera_params.get("fov_v_deg", 42.61))
    n_v, n_h = len(v_lids), len(h_lids)

    def _theory(idx, n, fov, c):
        a = _fan_angle(idx, n, fov)
        return f * np.tan(a) + c

    def _fit_axis(lids, coord_idx, n, fov, c):
        """검출 대표좌표 ↔ 이론좌표 최소제곱 정렬 → 이상선 보정"""
        det, th, valid = [], [], []
        for i, lid in enumerate(lids):
            pts = out.get(lid, [])
            if len(pts) >= 10:
                arr = np.array(pts, dtype=float)
                det.append(float(np.median(arr[:, coord_idx])))
                th.append(_theory(i, n, fov, c))
                valid.append(i)
        if len(det) < 4:
            return 0
        det = np.array(det); th = np.array(th)

        # 강건 선형 정렬: det ≈ s·th + t  (2회 재가중 최소제곱)
        s, t = 1.0, 0.0
        for _ in range(2):
            pred = s * th + t
            resid = det - pred
            mad = np.median(np.abs(resid - np.median(resid))) + 1e-6
            w = 1.0 / (1.0 + (resid / (3 * 1.4826 * mad))**2)  # Cauchy 가중
            # 가중 최소제곱
            Sw = w.sum()
            mx = (w*th).sum()/Sw; my = (w*det).sum()/Sw
            cov = (w*(th-mx)*(det-my)).sum()
            var = (w*(th-mx)**2).sum() + 1e-9
            s = cov/var; t = my - s*mx

        # 정렬 후 잔차 큰 선 = 이상 선 → 이론 정렬값으로 평행이동 보정
        pred = s * th + t
        resid = det - pred
        mad = np.median(np.abs(resid - np.median(resid))) + 1e-6
        thr = max(4.0 * 1.4826 * mad, 1.5)  # px

        n_adj = 0
        for k, i in enumerate(valid):
            if abs(resid[k]) > thr:
                lid = lids[i]
                arr = np.array(out[lid], dtype=float)
                arr[:, coord_idx] -= resid[k]   # 격자 정렬 위치로 평행이동
                out[lid] = arr.tolist()
                n_adj += 1
        return n_adj

    n1 = _fit_axis(v_lids, 0, n_v, fov_h, cx)   # V선: u좌표 정렬
    n2 = _fit_axis(h_lids, 1, n_h, fov_v, cy)   # H선: v좌표 정렬
    return n1 + n2


def _ridge_centroid_subpixel(seg_raw, lo, bg, pred, band, half=4):
    """
    배경을 뺀 **대칭 창** 안의 세기 가중 중심 — 이 코드의 주 서브픽셀 추정.

    왜 바꿨나
    --------
    이전에는 Steger(미분 기반)를 썼다. 비대칭 프로파일에 강하다는 것이
    이유였지만, 이 구현은 두 가지를 함께 하고 있었다.
      · [1,4,6,4,1] 평활 후 정점 3점만으로 포물선을 세운다. 선폭이
        2~3px 뿐이면 평활이 프로파일을 뭉개 3점 포물선이 편향된다.
      · 입력이 이미 하드 문턱으로 잘린 seg_clean 이라, 꼬리가 한쪽만
        잘리면 그만큼 중심이 밀린다.
    실측하면 렌더 이미지에서 Steger 경로가 σ 0.357px 인데, 배경을 빼고
    대칭 창에서 그냥 무게중심을 잡으면 σ 0.024px 다. 15배 차이다.

    중심법이 편향되는 두 원인은 창의 비대칭과 배경이다. 둘 다 여기서
    없앤다 — 정점 기준 대칭 창을 쓰고(양쪽이 잘리면 짧은 쪽에 맞춰
    함께 줄인다), 배경을 뺀 뒤 음수는 0 으로 자른다. 잘린 값이 아니라
    원신호를 쓰므로 꼬리도 살아 있다.

    창이 확보되지 않거나 신호가 약하면 Steger 로 넘긴다.
    """
    n = len(seg_raw)
    if n < 3:
        return None
    pk = int(np.argmax(seg_raw))
    h = int(min(half, pk, n - 1 - pk))          # 양쪽 대칭이 되게 줄인다
    if h < 1:
        return None
    win = seg_raw[pk - h: pk + h + 1].astype(np.float64) - float(bg)
    np.clip(win, 0.0, None, out=win)
    tot = win.sum()
    if tot <= 1e-6:
        return None
    x = np.arange(lo + pk - h, lo + pk + h + 1, dtype=np.float64)
    mu = float((x * win).sum() / tot)
    if abs(mu - pred) > band:
        return None
    return mu


def _steger_subpixel(idx, weights, pred, band):
    """
    [방안1] Steger 서브픽셀 추출 — 미분 기반 (프로파일 모양 무관).

    원리: 레이저 선의 강도 프로파일에서 1차 미분이 0이 되는 지점
          (= 강도 최대점)을 2차 미분으로 서브픽셀 보정.
          x0 = x_peak - f'(x_peak) / f''(x_peak)

    가우시안 피팅은 강도가 대칭 종모양이라 가정하지만, 실제 레이저 선은
    센서 양자화·표면 반사 불균일로 비대칭이 되어 피팅 오차가 발생한다.
    Steger는 미분 기반이므로 프로파일 형태에 무관하게 능선(ridge)의
    정점을 찾아 비대칭 프로파일에서도 강건하다.

    안정화:
      · 가우시안 평활(σ≈1px)로 미분 노이즈 억제
      · 2차 미분이 음(볼록 정점)인지 확인
      · 예측 위치에서 과도하게 벗어나면 fallback
    """
    fb = _weighted_subpixel_peak(idx, weights)
    n  = len(idx)
    if n < 5:
        return fb

    w = weights.astype(np.float64)

    # ── 1D 가우시안 평활 (미분 노이즈 억제) ──
    #   커널 [1,4,6,4,1]/16 (σ≈1px)
    if n >= 5:
        k = np.array([1, 4, 6, 4, 1], dtype=np.float64) / 16.0
        ws = np.convolve(w, k, mode="same")
    else:
        ws = w

    # ── 강도 최대점 (정수 픽셀) ──
    peak = int(np.argmax(ws))
    # 경계면 fallback (미분에 양옆 값 필요)
    if peak <= 0 or peak >= n - 1:
        return fb

    # ── 1차·2차 미분 (중앙 차분) ──
    d1 = 0.5 * (ws[peak + 1] - ws[peak - 1])
    d2 = ws[peak + 1] - 2.0 * ws[peak] + ws[peak - 1]

    # ── 2차 미분이 음(정점=볼록)이어야 유효 ──
    if d2 >= -1e-9:
        return fb

    # ── Steger 서브픽셀 보정 ──
    delta = -d1 / d2            # 정점까지의 서브픽셀 이동량
    if abs(delta) > 1.0:        # 1px 넘으면 신뢰 불가
        return fb

    mu = float(idx[peak] + delta)

    # ── 예측 위치 sanity check ──
    if abs(mu - pred) > band * 0.6:
        return fb

    # 가중 중심과 블렌딩 (극단값 완화)
    w_sum    = w.sum()
    centroid = float(np.dot(idx, w) / (w_sum + 1e-9))
    return 0.75 * mu + 0.25 * centroid


def _gaussian_subpixel(idx, weights, pred, band):
    fb = _weighted_subpixel_peak(idx, weights)
    if not _SCIPY or len(idx) < 5:
        return fb

    w_sum    = weights.sum()
    centroid = float(np.dot(idx, weights) / (w_sum + 1e-9))
    peak_i   = int(np.argmax(weights))
    A0       = float(weights[peak_i])
    sigma0   = max(1.2, float(
        np.sqrt(np.sum(weights * (idx - centroid)**2) / (w_sum + 1e-9))))
    C0       = float(np.percentile(weights, 10))

    try:
        popt, _ = _curve_fit(
            _gauss1d, idx, weights,
            p0=[A0, centroid, sigma0, C0],
            bounds=([0, idx[0], 0.5, 0], [A0*2.5, idx[-1], band, A0*0.5]),
            maxfev=300)
        mu_fit    = float(popt[1])
        sigma_fit = float(popt[2])
        if abs(mu_fit - pred) > band * 0.6:
            return fb
        if sigma_fit < 0.5 or sigma_fit > band * 0.8:
            return fb
        return 0.7 * mu_fit + 0.3 * centroid
    except Exception:
        return fb


def _gauss1d(x, A, mu, sigma, C):
    return A * np.exp(-0.5 * ((x - mu) / (sigma + 1e-9))**2) + C


# =====================================================================
# MAD 이상치 제거 (평활화 없음)
# =====================================================================
def _mad_outlier_remove(raw_pts, coord_idx, poly_deg=3,
                        mad_k=5.0, min_abs_px=2.0):
    arr    = np.array(raw_pts, dtype=float)
    coords = arr[:, coord_idx]
    n      = len(coords)
    x      = np.arange(n, dtype=float)
    deg    = min(poly_deg, max(1, n - 2))
    try:
        trend = np.polyval(np.polyfit(x, coords, deg), x)
    except Exception:
        trend = np.full(n, np.median(coords))
    resid  = coords - trend
    mad    = float(np.median(np.abs(resid - np.median(resid))))
    thresh = max(mad_k * 1.4826 * mad, min_abs_px)
    keep   = np.abs(resid) <= thresh
    return arr[keep].tolist() if keep.sum() >= 5 else arr.tolist()


# =====================================================================
# 가중평균 + 포물선 보정 fallback
# =====================================================================
def _weighted_subpixel_peak(idx, weights):
    w_sum    = weights.sum()
    centroid = float(np.dot(idx, weights) / (w_sum + 1e-9))
    peak_i   = int(np.argmax(weights))
    if 0 < peak_i < len(weights) - 1:
        y0, y1, y2 = weights[peak_i-1], weights[peak_i], weights[peak_i+1]
        denom = y0 - 2*y1 + y2
        if abs(denom) > 1e-6:
            delta = max(-1.0, min(1.0, 0.5*(y0-y2)/denom))
            return 0.5 * centroid + 0.5 * float(idx[peak_i] + delta)
    return centroid
