#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
회귀 검증 — 기존 PDF 검증값이 그대로 재현되는지 + 영역별 검측이 정답과 맞는지
========================================================================
Isaac Sim 없이 도는 검증만 모았다. 렌더링·선검출까지 포함한 검증은
Isaac 환경에서 inspection.py 를 돌려야 한다.

실행:  python3 tests/test_regression.py
========================================================================
"""
import sys, os
import json
import numpy as np
import importlib.util as ilu

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)


def _load_test(name):
    """tests/ 안의 도우미 모듈."""
    spec = ilu.spec_from_file_location(
        name, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           f"{name}.py"))
    m = ilu.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _load(name):
    spec = ilu.spec_from_file_location(name, os.path.join(ROOT, f"{name}.py"))
    m = ilu.module_from_spec(spec); spec.loader.exec_module(m)
    return m


EQ1 = _load("eq1_triangulation")
EQ2 = _load("eq2_plane_fit")
EQ3 = _load("eq3_orientation")
EQ5 = _load("eq5_region_assign")
PIPE = _load("pipeline_region")
SYN = _load("synth_scene")
CALIB = _load("calibration")
DET = _load("A_선검출")
EQ7 = _load("eq7_laser_plane")
REPORT = _load("report")
RP = _load("run_pipeline")
EQ8 = _load("eq8_silhouette")
# synth_scene 은 자기 인스턴스의 calibration 을 본다. _load 는 호출마다
# 새 모듈을 만들므로, 위의 CALIB 에 use_profile 을 걸어도 합성 씬은 옛
# 프로파일로 남는다. 씬을 만드는 검증은 이쪽을 써야 한다.
SYN_CALIB = SYN._CALIB

_FAILS = []


def check(name, cond, detail=""):
    mark = "PASS" if cond else "FAIL"
    print(f"  [{mark}] {name}" + (f"   {detail}" if detail else ""))
    if not cond:
        _FAILS.append(name)
    return cond


def test_hardware_spec():
    """
    캘리브레이션 값이 PDF 2.2 하드웨어 사양과 어긋나지 않는지.

    이 그룹이 있는 이유 — 이전 버전은 렌즈 12mm 를 "PDF 사양" 이라고
    적어 두었지만 PDF 는 초점거리를 명시한 적이 없다. 12mm 로는 DOE
    격자(120cm 936mm)가 센서에 담기지 않고 고정 초점 심도도 작업거리를
    못 덮는데, 검사가 없어 조용히 넘어갔다.

    이제 사양 프로파일이 둘이므로 검사도 둘로 나뉜다.
      · legacy   주신 원본 v4 의 값. 실제 사양이 확정되기 전 기준선이며
                 이미 뽑아 둔 Isaac 렌더가 이 값으로 만들어졌다.
      · pdf      사양표 원문과 한 글자도 어긋나지 않아야 한다
      · improved 바뀐 항목은 근거가 있어야 하고, 물리 조건(격자 수용·
                 심도·목표 정밀도)은 pdf 와 똑같이 만족해야 한다

    세 프로파일 모두 물리 조건은 통과해야 한다. 검측식(eq1~eq6)은 이
    값들에 의존하지 않으므로, 실제 하드웨어 사양이 들어오면 프로파일
    한 줄만 바꾸면 된다.
    """
    print("\n[0] 하드웨어 사양 정합성 — PDF 2.2")
    C = CALIB
    keep = C.ACTIVE_PROFILE

    # ── legacy 는 원본 v4 값을 그대로 재현해야 한다 ──
    C.use_profile("legacy")
    check(f"[legacy] f_px {C.F_PX:.1f} = 원본 1593.0", abs(C.F_PX - 1593.0) < 0.1)
    check(f"[legacy] 발산각 {C.FOV_DEG}° = 원본 60.82°", C.FOV_DEG == 60.82)
    check(f"[legacy] 격자 {C.N_VERTICAL}+{C.N_HORIZONTAL} = 원본 21+21",
          (C.N_VERTICAL, C.N_HORIZONTAL) == (21, 21))
    check(f"[legacy] 발사각 모델 {C.DOE_ANGLE_MODEL} = 원본 등각도",
          C.DOE_ANGLE_MODEL == "equal_angle")
    a = [C.make_line_angles()[f"V{i}"]["angle_rad"] for i in range(21)]
    check("[legacy] V선 간격이 정확히 등간격 (수렴각 0)",
          bool(np.allclose(np.diff(a), np.radians(60.82 / 20))))

    # ── PDF 프로파일은 사양표와 일치해야 한다 ──
    C.use_profile("pdf")
    check(f"[pdf] 해상도 {C.IMAGE_W}×{C.IMAGE_H} = 사양",
          (C.IMAGE_W, C.IMAGE_H) == (2448, 2048))
    check(f"[pdf] 화소 {C.PIXEL_PITCH_UM}µm ≥ 사양 3.45µm",
          C.PIXEL_PITCH_UM >= 3.45)
    check(f"[pdf] 기선 {C.BASELINE_M*1000:.0f}mm = 사양 150mm",
          abs(C.BASELINE_M - 0.150) < 1e-9)
    check(f"[pdf] 격자 수직{C.N_VERTICAL}+수평{C.N_HORIZONTAL} = "
          f"{C.N_VERTICAL*C.N_HORIZONTAL}교점 = 사양 400교점",
          C.N_VERTICAL * C.N_HORIZONTAL == 400)
    check(f"[pdf] 센서 대각 {C.SENSOR_DIAG_MM:.2f}mm ≥ 2/3\u2033(11.0mm)",
          C.SENSOR_DIAG_MM >= 10.9)

    # ── 두 프로파일 모두 만족해야 하는 물리 조건 ──
    proj = C.projection_mm_at(1.2)
    check(f"120cm 투사폭 {proj:.0f}mm = 사양 936mm (프로파일 무관)",
          abs(proj - 936.0) < 1.0)
    for name in ("legacy", "pdf", "improved"):
        C.use_profile(name)
        r = C.check_consistency(verbose=False)
        check(f"[{name}] 격자가 {C.WORK_Z_MIN_M}~{C.WORK_Z_MAX_M}m 내내 센서 안 "
              f"(u {r['u_range_near'][0]:.0f}..{r['u_range_far'][1]:.0f}, "
              f"v {r['v_range'][0]:.0f}..{r['v_range'][1]:.0f})", r["fits"])
        near, far = C.depth_of_field()
        check(f"[{name}] 고정초점 {C.FOCUS_DISTANCE_M}m 심도 "
              f"{near:.3f}~{far:.3f}m 가 작업거리를 덮음",
              near <= C.WORK_Z_MIN_M and far >= C.WORK_Z_MAX_M)
        sz = C.sigma_z_mm(C.WORK_Z_MAX_M)
        check(f"[{name}] 최원거리 깊이잡음 {sz:.2f}mm ≤ 목표 ±{C.TARGET_SIGMA_MM}mm",
              sz <= C.TARGET_SIGMA_MM)
        icp = C.isaac_camera_params()
        f_usd = (icp["focal_length_mm"] * icp["resolution"][0]
                 / icp["horizontal_aperture_mm"])
        check(f"[{name}] Isaac 카메라 설정 → f_px {f_usd:.1f} = {C.F_PX:.1f}",
              abs(f_usd - C.F_PX) < 0.5)
        best = C.find_best_tilt()
        if name == "legacy":
            # 원본은 레이저 축을 카메라와 평행하게 두었다(수렴각 개념 없음).
            # 격자는 들어오지만 한쪽 여유를 그만큼 낭비한다. 값을 바꾸면
            # 이미 뽑아 둔 렌더와 어긋나므로 지적만 하고 그대로 둔다.
            r2 = C.check_consistency(verbose=False)
            now = min(r2["u_range_near"][0], C.IMAGE_W - r2["u_range_far"][1])
            check(f"[legacy] 수렴각 0° — 가장자리 여유 {now:.0f}px "
                  f"(최적 {best['tilt_deg']}° 였다면 {best['margin_px']:.0f}px)",
                  now >= C.EDGE_MARGIN_PX)
        else:
            check(f"[{name}] 수렴각 {C.LASER_TILT_DEG}° 가 탐색 최적값과 일치",
                  abs(best["tilt_deg"] - C.LASER_TILT_DEG) < 0.02)

    # ── 개선안이 실제로 개선인지 ──
    rows = {r["name"]: r for r in C.compare_profiles()}
    pdf, imp = rows["pdf"], rows["improved"]
    gain = pdf["sigma_z_mm"] / imp["sigma_z_mm"]
    check(f"개선안 깊이잡음 {pdf['sigma_z_mm']}mm → {imp['sigma_z_mm']}mm "
          f"({gain:.2f}배 개선)", gain >= 2.5)
    dens = pdf["pitch_mm"] / imp["pitch_mm"]
    check(f"개선안 격자 피치 {pdf['pitch_mm']}mm → {imp['pitch_mm']}mm "
          f"({dens:.2f}배 조밀)", dens >= 1.9)
    check(f"개선안 격자 피치 {imp['pitch_mm']}mm < 동바리 Ø48.6mm "
          f"— 부재당 V선 2개 이상", imp["pitch_mm"] < 48.6 / 2.0)
    check("개선안이 레이저 출력을 올리지 않음 (눈 안전등급 재평가 불필요)",
          C.LASER_POWER_MW == (30, 49))

    C.use_profile(keep)

    # ── 초점거리 근거가 살아 있는지 (활성 프로파일 기준) ──
    bad = []
    for fmm in (10.0, 12.0, 16.0):
        cp = {**C.CAMERA_PARAMS, "f_px": C.focal_px(fmm)}
        n, fr = C.depth_of_field(fmm)
        if C.check_consistency(cp, verbose=False)["fits"] and \
           n <= C.WORK_Z_MIN_M and fr >= C.WORK_Z_MAX_M:
            bad.append(fmm)
    check(f"10/12/16mm 는 격자 수용·심도에서 탈락 (통과 {len(bad)}건)"
          + (f" — {bad}" if bad else ""), not bad)

    # ── DOE 사인등간격 모델 ──
    s = C._fan_angles(C.N_VERTICAL, C.FOV_DEG, "equal_sine")
    a = C._fan_angles(C.N_VERTICAL, C.FOV_DEG, "equal_angle")
    check(f"DOE 사인등간격 — 포락선 ±{np.degrees(s[-1]):.2f}° 는 등각도와 동일",
          abs(s[0] - a[0]) < 1e-12 and abs(s[-1] - a[-1]) < 1e-12)
    # 두 모델의 차이가 얼마나 큰 문제인지 — 각도가 아니라 깊이로 환산해야
    # 판단이 선다. α 를 잘못 알면 u 가 맞아도 Z 가 틀어진다.
    #     Z = f·b / (f·tanα − (u−c_x))  →  dZ/dα ≈ Z²/b · sec²α
    d = float(np.max(np.abs(s - a)))
    dz_mm = d * 1.2 * 1.2 / C.BASELINE_M * 1000.0
    check(f"DOE 모델 오선택 시 1.2m 깊이오차 {dz_mm:.0f}mm — 목표 "
          f"±{C.TARGET_SIGMA_MM}mm 를 크게 넘으므로 실측 α_i 가 필수",
          dz_mm > C.TARGET_SIGMA_MM)


def test_eq1_triangulation():
    """PDF 5.1: 삼각측량 깊이복원오차 ≈ 0mm"""
    print("\n[1] eq1 삼각측량 — PDF 검증값 재현")
    f, b, cx, cy = 1593.0, 0.150, 1224.0, 1024.0
    for Z_true in (0.5, 1.0, 1.5, 2.0):
        alpha, beta = np.radians(10.0), np.radians(5.0)
        X_t, Y_t = Z_true * np.tan(alpha), Z_true * np.tan(beta)
        u = f * (X_t - b) / Z_true + cx
        v = f * Y_t / Z_true + cy
        _, _, Z = EQ1.triangulate_point(u, v, alpha, beta, f, b, cx, cy)
        err_mm = abs(Z - Z_true) * 1000
        check(f"Z={Z_true}m 복원오차 {err_mm:.6f}mm < 0.001mm", err_mm < 1e-3)


def test_eq3_backward_compat():
    """v1 좌표축 규약이 v2 기본값으로 그대로 재현되는지"""
    print("\n[2] eq3 하위호환 — v1 규약 재현")

    def rx(d):
        r = np.radians(d)
        return np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)],
                         [0, np.sin(r), np.cos(r)]])

    n_wall = rx(0.5) @ np.array([0, 0, -1.0])
    tv = EQ3.measure_verticality(n_wall)
    check(f"수직도 기본 ĝ: {tv:.4f}° (참값 0.5°)", abs(tv - 0.5) < 1e-6)

    n_floor = rx(0.3) @ np.array([0, 0, -1.0])
    th = EQ3.measure_horizontality(n_floor)
    check(f"수평도 바닥규약: {th:.4f}° (참값 0.3°)", abs(th - 0.3) < 1e-6)

    # 회전 불변성 — 점군과 중력을 같이 돌리면 결과가 변하지 않아야 한다
    Rt = rx(37.0)
    tv2 = EQ3.measure_verticality(Rt @ n_wall, g_hat=Rt @ EQ3.G_UPRIGHT)
    check(f"장비 37° 기울임 불변: {tv2:.4f}° (참값 0.5°)", abs(tv2 - 0.5) < 1e-6)


def test_gravity_paths_agree():
    """IMU 경로와 카메라 자세 경로가 같은 ĝ 를 내는지"""
    print("\n[3] 중력 경로 일치 — IMU vs 카메라 자세")
    for pitch in (0.0, 22.0, 90.0):
        td = np.radians(pitch)
        view = np.array([0.0, np.cos(td), -np.sin(td)])
        up = np.array([0.0, 0.0, 1.0])
        if abs(float(view @ up)) > 0.9:
            up = np.array([0.0, 1.0, 0.0])
        right = np.cross(view, up); right /= np.linalg.norm(right)
        down = np.cross(view, right); down /= np.linalg.norm(down)
        R = np.column_stack([right, down, view])
        g_cam = EQ3.gravity_from_camera_rotation(R)
        g_expect = np.array([0.0, np.cos(td), np.sin(td)])
        err = float(np.linalg.norm(g_cam - g_expect))
        check(f"pitch {pitch:4.1f}° → ĝ 오차 {err:.2e}", err < 1e-9)


def test_tls_plane_vs_legacy():
    """경사면에서 TLS 가 필요한 이유 + 정면에서 회귀 없음"""
    print("\n[4] eq2 TLS 평면적합 — 경사면 정확도 / 정면 무회귀")
    rng = np.random.default_rng(3)
    grazing = np.column_stack([rng.uniform(-.6, .6, 400), np.full(400, 0.8),
                               rng.uniform(.5, 2.5, 400)]) \
        + rng.normal(0, 5e-4, (400, 3))

    def ang(n, truth):
        n = np.asarray(n[:3], float); n /= np.linalg.norm(n)
        return float(np.degrees(np.arccos(min(1.0, abs(float(n @ truth))))))

    old, _ = EQ2.fit_plane_ransac(grazing, threshold=0.01)
    new, _ = EQ2.fit_plane_tls_ransac(grazing, threshold=0.005)
    e_old, e_new = ang(old, [0, 1, 0]), ang(new, [0, 1, 0])
    check(f"경사면 TLS 법선오차 {e_new:.4f}° ≤ 0.05° (기존 {e_old:.2f}°)",
          e_new <= 0.05)
    check(f"기존 방식이 실제로 실패함을 확인 ({e_old:.2f}° > 0.5°)", e_old > 0.5)

    front = np.column_stack([rng.uniform(-.5, .5, 400), rng.uniform(-.4, .4, 400),
                             np.full(400, 1.2)]) + rng.normal(0, 5e-4, (400, 3))
    o2, _ = EQ2.fit_plane_ransac(front, threshold=0.01)
    n2, _ = EQ2.fit_plane_tls_ransac(front, threshold=0.005)
    d = abs(ang(o2, [0, 0, 1]) - ang(n2, [0, 0, 1]))
    check(f"정면 벽 회귀 없음 (두 방식 차 {d:.4f}° < 0.01°)", d < 0.01)


def test_axis_fit():
    """동바리 축 적합 정확도"""
    print("\n[5] eq2 축 적합 — 동바리 Ø48.6mm")
    rng = np.random.default_rng(0)
    for tilt in (0.0, 0.6, 1.2, 3.0):
        d = EQ3.normalize(np.array(
            [0.0, -np.cos(np.radians(tilt)), np.sin(np.radians(tilt))]))
        e1, e2 = EQ2.plane_tangent_basis(d)
        t = rng.uniform(0, 2.4, 300); ph = rng.uniform(-1.0, 1.0, 300)
        P = (np.outer(t, d) + 0.0243 * (np.cos(ph)[:, None] * e1
                                        + np.sin(ph)[:, None] * e2)
             + rng.normal(0, 8e-4, (300, 3)))
        res = EQ2.fit_axis_pca(P)
        theta = EQ3.measure_axis_verticality(res["direction"],
                                             g_hat=[0, 1, 0])
        err = abs(theta - tilt)
        check(f"기울기 {tilt}° → 측정 {theta:.4f}° 오차 {err:.4f}° < 0.1°",
              err < 0.1)


def test_region_pipeline():
    """합성 씬 전 구간 — 두 세그멘테이션 백엔드"""
    print("\n[6] 영역별 검측 파이프라인 — 합성 씬 (벽+바닥+동바리)")
    truth_gap = SYN.straightedge_truth_mm()
    scene = SYN.build_scene()
    gt = scene["gt"]
    want = {"wall": "wall_verticality_deg",
            "floor": "floor_horizontality_deg",
            "shoring": "shoring_verticality_deg"}

    for backend in ("gt", "geom"):
        res = PIPE.inspect_capture(
            scene["lines_pixels"], scene["line_angles"],
            scene["camera_params"], scene["R_world_cam"],
            label_map=(scene["label_map"] if backend == "gt" else None),
            id_to_semantic=(scene["id_to_semantic"] if backend == "gt" else None),
            rgb_off=scene["rgb_off"], backend=backend)

        # inspect_capture 가 카메라 자세에서 유도한 ĝ 가 씬의 ĝ 와 같아야 한다
        gerr = float(np.linalg.norm(
            np.array(res["gravity_laser_frame"]) - scene["g_hat"]))
        check(f"[{backend}] 카메라 자세 유도 ĝ 오차 {gerr:.2e}", gerr < 1e-6)

        best = {}
        n_wall_regions = 0
        for r in res["regions"]:
            if r["status"] != "measured":
                continue
            c = r["class"]
            if c == "wall":
                n_wall_regions += 1
            if c not in best or r["n_points"] > best[c]["n_points"]:
                best[c] = r

        # 벽은 하나여야 한다. 앞에 선 동바리가 드리운 폭 ~5cm 의 가림
        # 그림자가 벽을 두 조각으로 나누면 각도는 멀쩡한데 직선자
        # 프로파일만 절반으로 줄어 평활도가 조용히 낮게 나온다
        # (실측 3.91 → 2.58mm). C_영역분할._merge_occlusion_split 참조.
        check(f"[{backend}] 벽이 가림 그림자로 쪼개지지 않음 "
              f"(측정된 벽 영역 {n_wall_regions}개)", n_wall_regions == 1)

        for cls, key in want.items():
            r = best.get(cls)
            if not check(f"[{backend}] {cls} 영역 검출됨", r is not None):
                continue
            err = abs(r["theta_deg"] - gt[key])
            check(f"[{backend}] {cls} {r['theta_deg']:.4f}° "
                  f"(정답 {gt[key]}°) 오차 {err:.4f}° ≤ 0.5°", err <= 0.5)

        w = best.get("wall")
        f = (w or {}).get("flatness") or {}
        if f.get("applicable"):
            gap = f.get("max_gap_mm", 0.0)
            up = f.get("upper_estimate_mm", gap)
            depth = f.get("defect_max_dev_mm", 0.0)
            bump = gt["wall_bump_mm"]
            # 자 처짐량은 요철 폭이 측정 분해능보다 좁으면 하한값이 된다.
            # 실제 렌즈(12mm)의 좁은 시야에서는 바닥을 담으려 장비를 기울여야
            # 하고, 그러면 벽을 30° 이상 사각으로 보게 되어 면내 점밀도가
            # 낮아진다. σ=5cm 요철은 그 분해능 아래로 내려간다.
            # 따라서 참값 포괄이 아니라 **하한값 성질**을 검증한다.
            # 참값은 융기 높이(6mm)가 아니라 직선자 처짐(3.99mm)이다.
            # synth_scene.GT_STRAIGHTEDGE_MM 주석에 유도가 있다.
            #
            # 참값은 프로파일마다 다르다. 화각이 다르면 벽에서 재는 구간이
            # 달라지고, 직선자 처짐은 그 구간 길이에 달려 있기 때문이다.
            # synth_scene.GT_STRAIGHTEDGE_MM 주석 참조.
            tol = 0.35
            check(f"[{backend}] 벽 자 처짐 {gap:.2f}mm — 참값 {truth_gap}mm 대비 "
                  f"{gap-truth_gap:+.2f}mm (허용 ±{tol}mm, "
                  f"프로파일 {CALIB.ACTIVE_PROFILE})", abs(gap - truth_gap) <= tol)
            # 요철 깊이는 정점 근방 원시잔차라 분해능 영향이 적다
            check(f"[{backend}] 벽 요철 깊이 {depth:.2f}mm 가 정답 {bump}mm 근방",
                  0.6 * bump <= depth <= 1.4 * bump)
            check(f"[{backend}] 벽 평활도 판정 = {f['judgement']}",
                  f["judgement"] in ("합격", "판정보류(분해능)"))


def test_segmentation_robustness():
    """세그멘테이션이 훼손돼도 각도 판정이 유지되는지"""
    print("\n[8] 세그멘테이션 훼손 강건성")
    EXP = _load_test("scene_perturb")
    scene = SYN.build_scene()
    gt = scene["gt"]
    want = {"wall": "wall_verticality_deg",
            "floor": "floor_horizontality_deg",
            "shoring": "shoring_verticality_deg"}

    def worst_err(row):
        if row["missing"]:
            return float("inf")
        return max(row["errors_deg"].values())

    # 마스크 팽창 — 얇은 동바리가 벽 점에 오염된다
    for k in (2, 8, 16):
        lm = EXP.perturb_mask(scene["label_map"], "dilate", k,
                              np.random.default_rng(0))
        r = EXP.run_once(scene, lm, "gt")
        check(f"마스크 +{k}px 팽창 — 세 부재 모두 측정, 최악 오차 "
              f"{worst_err(r):.4f}° ≤ 0.5°", worst_err(r) <= 0.5)

    # 라벨 오분류 — 두 부재가 한 라벨로 병합되는 최악의 경우
    #
    # 두 가지를 구분해서 본다. 틀린 값을 내놓는 것과, 근거가 모자라 아예
    # 재지 않는 것은 성질이 다르다. 검측 장비에서 전자는 사고로 이어지고
    # 후자는 재촬영으로 끝난다. 그래서 "측정한 값은 반드시 맞을 것" 을
    # 무조건 조건으로 두고, "몇 개를 measured 로 남기는가" 는 훼손 정도에
    # 따라 다르게 요구한다.
    #
    # PDF 실사양(DOE 42.61°, 20×20)에서는 1.2m 격자 피치가 49.3mm 라
    # Ø48.6mm 동바리에 걸리는 V선이 한두 개뿐이다. 라벨이 67% 넘게
    # 뒤섞이면 그 몇 개마저 흩어져 축적합이 성립하지 않는 경우가 생긴다.
    # 이때 측정을 포기하는 것이 옳은 동작이다.
    wrong, dropped = [], []
    for p in (0.34, 0.67, 1.0):
        for t in range(3):
            lm = EXP.perturb_mask(scene["label_map"], "mislabel", p,
                                  np.random.default_rng(1000 + t))
            r = EXP.run_once(scene, lm, "gt")
            errs = r["errors_deg"]
            if errs and max(errs.values()) > 0.5:
                wrong.append(f"p={p} 시행{t}")
            if r["missing"]:
                dropped.append((p, t, r["missing"]))
    check(f"라벨 오분류 9종 — 측정한 값은 모두 정확 (오답 {len(wrong)}건)"
          + (f" — {wrong}" if wrong else ""), not wrong)
    mild = [d for d in dropped if d[0] <= 0.34]
    check(f"라벨 34% 오분류까지는 세 부재 모두 측정 (누락 {len(mild)}건)"
          + (f" — {mild}" if mild else ""), not mild)
    if dropped:
        print(f"       (참고) 심한 훼손에서 측정 포기 {len(dropped)}건: "
              f"{[(p, m) for p, _, m in dropped]}")

    # 부재 누락 — 남은 부재는 영향받지 않아야 한다
    lm = scene["label_map"].copy(); lm[lm == 3] = 0        # 동바리 삭제
    r = EXP.run_once(scene, lm, "gt")
    others_ok = all(r["errors_deg"].get(c, 9) <= 0.5 for c in ("wall", "floor"))
    check("동바리 마스크 삭제 시 벽·바닥은 영향 없음", others_ok)


def test_boundary_rejection():
    """경계 오염 제거 — 깊이 불연속"""
    print("\n[7] eq5 경계 정제")
    tbl = {"uv": np.zeros((40, 2)), "seq": np.arange(40),
           "lid": np.array(["V0"] * 40, dtype=object),
           "xyz": np.column_stack([np.zeros(40), np.zeros(40),
                                   np.r_[np.full(20, 1.2), np.full(20, 2.0)]])}
    bad = EQ5.mark_depth_discontinuity(tbl)
    check(f"벽1.2m→바닥2.0m 경계에서 2점 제거 (실제 {int(bad.sum())})",
          bad.sum() == 2 and bad[19] and bad[20])

    m = np.zeros((60, 60), bool); m[20:40, 20:40] = True
    check(f"마스크 3px 침식 400→{int(EQ5.erode_mask(m, 3).sum())}px",
          EQ5.erode_mask(m, 3).sum() == 196)


def test_eq7_laser_plane():
    """[9] eq7 레이저 평면 — V·H 를 한 식으로, 깊이 이득 g"""
    print("\n[9] eq7 레이저 평면 일반화 — 가로선(H) 을 쓸 수 있는 조건")
    f, b, cx, cy = 1593.0, 0.150, 1224.0, 1024.0
    rng = np.random.default_rng(4)

    # (1) V선에서 eq1 과 완전히 같은 값이어야 한다.
    #     여기가 어긋나면 기존 검증값이 통째로 무의미해진다.
    worst = 0.0
    for _ in range(500):
        al = np.radians(rng.uniform(-30, 30))
        Z = rng.uniform(0.8, 6.0)
        u = f * (Z * np.tan(al) - b) / Z + cx
        v = f * (Z * rng.uniform(-0.5, 0.5)) / Z + cy
        xyz, _ = EQ7.triangulate_plane([[u, v]], EQ7.plane_normal("alpha", al),
                                       f, b, cx, cy)
        ref = EQ1.triangulate_point(u, v, al, 0.0, f, b, cx, cy)
        worst = max(worst, float(np.max(np.abs(xyz[0] - np.array(ref)))))
    check(f"eq1 과 동일 (V선 500점) 최대차 {worst*1e9:.3f} nm", worst < 1e-9)

    # (2) 이득 공식 g = √(n_x²+n_y²)/|n_x| 이 실제 잡음 증폭과 맞는가
    for label, fixed, ang, roll in (("V roll 0°", "alpha", 10.0, 0.0),
                                    ("H roll45°", "beta", 20.0, 45.0),
                                    ("H roll20°", "beta", 20.0, 20.0)):
        n = EQ7.plane_normal(fixed, np.radians(ang),
                             roll_rad=np.radians(roll))
        m = np.array([n[0], n[1]]); m = m / np.linalg.norm(m)
        Zc = 2.7
        K = n[2] + n[0] * b / Zc
        if abs(n[1]) > 1e-9:
            uh0 = 0.05; vh0 = -(n[0] * uh0 + K) / n[1]
        else:
            uh0 = -K / n[0]; vh0 = 0.02
        e = rng.normal(0, 0.3, 20000)
        uv = np.column_stack([cx + f * uh0 + e * m[0],
                              cy + f * vh0 + e * m[1]])
        xyz, _ = EQ7.triangulate_plane(uv, n, f, b, cx, cy)
        sig = float(np.std(xyz[:, 2])) * 1000.0
        pred = EQ7.depth_gain(n) * 0.3 * Zc ** 2 / (f * b) * 1000.0
        check(f"이득 g={EQ7.depth_gain(n):.3f} ({label}) — 실측 σ_Z "
              f"{sig:.2f}mm vs 예측 {pred:.2f}mm",
              abs(sig - pred) < 0.05 * pred)

    # (3) 굴리지 않은 H선은 원리적으로 깊이가 없어야 한다.
    #     이것이 "가로선을 검출해도 깊이에 못 쓴다" 의 근거다.
    n0 = EQ7.plane_normal("beta", np.radians(20.0))
    check("굴림 0° 의 H선 이득 = ∞ (평면이 기선을 품음)",
          not np.isfinite(EQ7.depth_gain(n0)),
          f"n_x={n0[0]:.3e}")
    n45 = EQ7.plane_normal("beta", np.radians(20.0), roll_rad=np.radians(45))
    check(f"굴림 45° 의 H선 이득 {EQ7.depth_gain(n45):.3f} = √2 → 사용 가능",
          abs(EQ7.depth_gain(n45) - np.sqrt(2)) < 1e-9
          and EQ7.depth_gain(n45) <= EQ7.MAX_DEPTH_GAIN)

    # (4) 프로파일별 — 굴림이 있어야만 가로선이 깊이 표본이 된다
    keep = CALIB.ACTIVE_PROFILE
    for name, want_h in (("legacy", False), ("pdf", False),
                         ("improved", False), ("diagonal", True)):
        CALIB.use_profile(name)
        la = CALIB.make_line_angles()
        pl = EQ7.line_planes(la)
        nh = sum(1 for l, p in pl.items() if l[0] == "H" and p["usable"])
        nv = sum(1 for l, p in pl.items() if l[0] == "V" and p["usable"])
        check(f"[{name}] V선 깊이가능 {nv}개 / H선 깊이가능 {nh}개",
              nv > 0 and ((nh > 0) == want_h))
    CALIB.use_profile(keep)


def test_h_lines_in_pipeline():
    """[10] 가로선이 실제로 삼각측량·검측까지 들어가는가"""
    print("\n[10] 가로선(H) 파이프라인 통합")
    keep = CALIB.ACTIVE_PROFILE
    keep_p = dict(CALIB.SPEC_PROFILES["diagonal"])
    keep_s = dict(SYN_CALIB.SPEC_PROFILES["diagonal"])
    keep_sp = SYN_CALIB.ACTIVE_PROFILE
    try:
        # 굴리지 않은 사양 — 가로선은 사유와 함께 걸러져야 한다
        CALIB.use_profile("legacy")
        la = CALIB.make_line_angles()
        cp = dict(CALIB.CAMERA_PARAMS)
        lp = {lid: [[cp["cx_px"] + 10.0 * k, cp["cy_px"] + 3.0 * k]
                    for k in range(30)] for lid in la}
        xyz, uv, info = PIPE.triangulate_lines(lp, la, cp)
        h_used = [l for l in xyz if l.startswith("H")]
        h_skip = [(l, w) for l, w in info["skipped"] if l.startswith("H")]
        check(f"[legacy] 가로선 {len(h_skip)}개가 사유와 함께 제외됨",
              len(h_used) == 0 and len(h_skip) == CALIB.N_HORIZONTAL,
              (h_skip[0][1] if h_skip else ""))
        check("[legacy] 세로선은 그대로 삼각측량됨",
              len([l for l in xyz if l.startswith("V")]) > 0)

        # 굴린 사양 — 합성 씬 전체를 돌려 가로선 점이 검측까지 들어가는지
        SYN_CALIB.SPEC_PROFILES["diagonal"].update(
            {"n_vertical": 20, "n_horizontal": 20, "laser_roll_deg": 45.0})
        SYN_CALIB.use_profile("diagonal")
        CALIB.use_profile("diagonal")
        SYN.CAMERA_PARAMS.clear()
        SYN.CAMERA_PARAMS.update(SYN_CALIB.CAMERA_PARAMS)
        SYN.GRID.update({"n_vertical": 20, "n_horizontal": 20,
                         "fov_deg": SYN_CALIB.FOV_DEG,
                         "samples_per_line": 250})
        sc = SYN.build_scene(seed=2026, sigma_u_px=SYN_CALIB.SIGMA_U_PX)
        nh_emit = sum(1 for l in sc["lines_pixels"] if l.startswith("H"))
        check(f"[diagonal] 합성 씬이 가로선 {nh_emit}개를 실제로 쏨",
              nh_emit > 0)
        res = PIPE.inspect_capture(
            sc["lines_pixels"], sc["line_angles"], sc["camera_params"],
            sc["R_world_cam"], label_map=sc["label_map"],
            id_to_semantic=sc["id_to_semantic"], rgb_off=sc["rgb_off"],
            backend="gt")
        nf = res["triangulation"]["n_by_family"]
        check(f"[diagonal] 삼각측량 점 V {nf.get('V', 0):,} + "
              f"H {nf.get('H', 0):,}", nf.get("H", 0) > 0)
        wall = [r for r in res["regions"]
                if r["class"] == "wall" and r["status"] == "measured"]
        check("[diagonal] 벽 영역이 측정됨", len(wall) == 1)
        if wall:
            lids = np.asarray(wall[0]["point_lid"], dtype=object)
            frac = float(np.mean([str(x).startswith("H") for x in lids]))
            check(f"[diagonal] 벽 점의 {frac*100:.0f}% 가 가로선에서 옴",
                  frac > 0.2)
            err = abs(wall[0]["theta_deg"] - SYN.GT_WALL_TILT_DEG)
            check(f"[diagonal] 벽 수직도 {wall[0]['theta_deg']:.4f}° "
                  f"(정답 {SYN.GT_WALL_TILT_DEG}°) 오차 {err:.4f}°",
                  err <= 0.5)
    finally:
        CALIB.SPEC_PROFILES["diagonal"].update(keep_p)
        CALIB.use_profile(keep)
        SYN_CALIB.SPEC_PROFILES["diagonal"].update(keep_s)
        SYN_CALIB.use_profile(keep_sp)
        SYN.CAMERA_PARAMS.clear()
        SYN.CAMERA_PARAMS.update(SYN_CALIB.CAMERA_PARAMS)
        SYN.GRID.update({"n_vertical": SYN_CALIB.N_VERTICAL,
                         "n_horizontal": SYN_CALIB.N_HORIZONTAL,
                         "fov_deg": SYN_CALIB.FOV_DEG,
                         "samples_per_line": 250})


def test_pointcloud_export():
    """[11] 부재별 3D 좌표 산출"""
    print("\n[11] 부재별 3D 좌표 — 점이 어디에 찍혔는가")
    sc = SYN.build_scene(seed=2026, sigma_u_px=CALIB.SIGMA_U_PX)
    res = PIPE.inspect_capture(
        sc["lines_pixels"], sc["line_angles"], sc["camera_params"],
        sc["R_world_cam"], label_map=sc["label_map"],
        id_to_semantic=sc["id_to_semantic"], rgb_off=sc["rgb_off"],
        backend="gt")
    got = [r for r in res["regions"] if r.get("point_xyz") is not None]
    check(f"모든 영역이 3D 점을 들고 있다 ({len(got)}/{len(res['regions'])})",
          len(got) == len(res["regions"]) and len(got) > 0)
    # 검측에 쓴 점과 같은 배열이어야 한다 — 어긋나면 그림이 거짓말을 한다
    ok = all(len(r["point_xyz"]) == r["n_points"] for r in got)
    check("3D 점 수 = 검측에 쓴 점 수", ok)

    # 중력 정렬 좌표: 벽은 서 있고 바닥은 누워 있어야 한다
    R = REPORT.world_frame(sc["g_hat"])
    check("중력 정렬 기저가 정규직교",
          float(np.abs(R.T @ R - np.eye(3)).max()) < 1e-9)
    summ = REPORT.region_xyz_summary(res, g_hat=sc["g_hat"])
    byc = {d["클래스"]: d for d in summ}
    if "wall" in byc and "floor" in byc:
        w, fl = byc["wall"], byc["floor"]
        check(f"벽은 측정구간 {w['측정구간_m']}m > 깊이폭 {w['깊이폭_m']}m (서 있다)",
              w["측정구간_m"] > w["깊이폭_m"])
        check(f"바닥은 깊이폭 {fl['깊이폭_m']}m > 측정구간 {fl['측정구간_m']}m (누워 있다)",
              fl["깊이폭_m"] > fl["측정구간_m"])
    rows = REPORT.region_xyz_rows(res, g_hat=sc["g_hat"], stride=50)
    check(f"좌표 표 {len(rows):,}행 — 부재 구분 열 포함",
          len(rows) > 0 and "클래스" in rows[0] and "높이_m" in rows[0])


def _plane_grid_pixels(roll_deg, nv=14, nh=14, z0=1.20, tilt_deg=3.0,
                       n_samples=6000):
    """살짝 기운 단일 평면에 격자를 쏘고 화소를 해석적으로 만든다."""
    CALIB.SPEC_PROFILES["diagonal"].update(
        {"n_vertical": nv, "n_horizontal": nh, "laser_roll_deg": roll_deg})
    CALIB.use_profile("diagonal")
    f, b = CALIB.F_PX, CALIB.BASELINE_M
    cx, cy = CALIB.CX_PX, CALIB.CY_PX
    W, H = CALIB.IMAGE_W, CALIB.IMAGE_H
    la = CALIB.make_line_angles()
    t = np.radians(tilt_deg)
    m = np.array([np.sin(t), 0.0, -np.cos(t)]); d0 = z0 * np.cos(t)
    ts = np.linspace(-np.radians(CALIB.FOV_DEG) / 2,
                     np.radians(CALIB.FOV_DEG) / 2, n_samples)
    pix = {}
    for lid, info in la.items():
        n = np.asarray(info["normal"], float)
        a = np.array([0.0, 0.0, 1.0]) - n * n[2]; a /= np.linalg.norm(a)
        c = np.cross(n, a); c /= np.linalg.norm(c)
        dirs = np.cos(ts)[:, None] * a + np.sin(ts)[:, None] * c
        den = dirs @ m
        ok = np.abs(den) > 1e-9
        sarr = np.full(len(dirs), np.nan); sarr[ok] = -d0 / den[ok]
        good = np.isfinite(sarr) & (sarr > 0.2) & (sarr < 5.0)
        P = dirs[good] * sarr[good, None]
        if len(P) < 50:
            continue
        u = f * (P[:, 0] - b) / P[:, 2] + cx
        v = f * P[:, 1] / P[:, 2] + cy
        ins = (u >= 2) & (u < W - 2) & (v >= 2) & (v < H - 2)
        if ins.sum() < 50:
            continue
        pix[lid] = np.column_stack([u[ins], v[ins]])
    cp = dict(f_px=f, b_m=b, cx_px=cx, cy_px=cy, image_w=W, image_h=H,
              n_v=nv, n_h=nh, fov_h_deg=CALIB.FOV_DEG,
              fov_v_deg=CALIB.FOV_DEG, standoff_z=z0,
              z_range=[z0 * 0.93, z0 * 1.07])
    return pix, la, cp


def _render_lines(pix, W, H, width=1.5):
    """선 화소를 가우시안으로 그려 넣은 초록 레이저 영상."""
    acc = np.zeros((H, W), np.float32)
    for P in pix.values():
        for u, v in P:
            u0, v0 = int(u), int(v)
            for dv in range(-2, 3):
                for du in range(-2, 3):
                    x, y = u0 + du, v0 + dv
                    if 0 <= x < W and 0 <= y < H:
                        d = np.hypot(x - u, y - v)
                        acc[y, x] = max(acc[y, x],
                                        float(np.exp(-0.5 * (d / width) ** 2)))
    img = np.zeros((H, W, 3), np.float32)
    img[:, :, 1] = acc * 255.0
    return np.clip(img, 0, 255).astype(np.uint8)


def test_rolled_grid_detection():
    """[12] 굴린 격자에서 선검출이 되는가 (렌더 → 검출 → 정답 대조)"""
    print("\n[12] 굴린 격자 선검출 — 두 계열이 모두 대각선일 때")
    keep = CALIB.ACTIVE_PROFILE
    keep_p = dict(CALIB.SPEC_PROFILES["diagonal"])
    try:
        for roll in (0.0, 45.0):
            pix, la, cp = _plane_grid_pixels(roll)
            img = _render_lines(pix, cp["image_w"], cp["image_h"])
            det = DET.detect(img, {}, la, cp, multi_surface=True)
            c = np.array([cp["cx_px"], cp["cy_px"]])
            stat = {}
            for lid, g in pix.items():
                d = np.asarray(det.get(lid, []), float)
                if len(d) < 10:
                    stat.setdefault(lid[0], []).append(None)
                    continue
                n = np.asarray(la[lid]["normal"], float)
                lat = np.hypot(n[0], n[1])
                mh = np.array([n[0], n[1]]) / lat
                th = np.array([-mh[1], mh[0]])
                tg, ng = (g - c) @ th, (g - c) @ mh
                td, nd = (d - c) @ th, (d - c) @ mh
                o = np.argsort(tg)
                msk = (td >= tg[o][0]) & (td <= tg[o][-1])
                if msk.sum() < 10:
                    stat.setdefault(lid[0], []).append(None)
                    continue
                e = nd[msk] - np.interp(td[msk], tg[o], ng[o])
                stat.setdefault(lid[0], []).append(
                    (float(np.median(e)), float(np.std(e))))
            for pre in "VH":
                v = stat.get(pre) or []
                ok = [x for x in v if x is not None and abs(x[0]) < 1.0
                      and x[1] < 1.0]
                bias = (np.median([x[0] for x in ok]) if ok else float("nan"))
                sc = (np.median([x[1] for x in ok]) if ok else float("nan"))
                check(f"굴림 {roll:g}° {pre}선 {len(ok)}/{len(v)} 검출 "
                      f"— 법선방향 편차 {bias:+.3f}px, 산포 {sc:.3f}px",
                      len(v) > 0 and len(ok) == len(v))
    finally:
        CALIB.SPEC_PROFILES["diagonal"].update(keep_p)
        CALIB.use_profile(keep)


def test_run_pipeline_inputs():
    """[13] 단일 입구 — IMU·거리·규약 복원"""
    print("\n[13] run_pipeline 입력 처리")

    # IMU 해석 — 부호가 뒤집히면 수직/수평 판정 자체가 뒤집힌다
    g, src, assumed = RP.gravity_from_imu(None)
    check(f"IMU 없음 → 똑바로 섰다고 가정 ĝ={g.tolist()}",
          assumed and np.allclose(g, [0, 1, 0]))
    g, _, _ = RP.gravity_from_imu({"pitch_deg": 34.0})
    want = np.array([0.0, np.cos(np.radians(34)), np.sin(np.radians(34))])
    check(f"하향 34° → ĝ={np.round(g, 4).tolist()}",
          np.allclose(g, want, atol=1e-9))
    g, _, _ = RP.gravity_from_imu({"gravity": [0, 2, 0]})
    check("중력벡터는 정규화된다", np.allclose(g, [0, 1, 0]))
    g, _, _ = RP.gravity_from_imu({"accel": [0, -1, 0]})
    check("가속도계는 부호를 뒤집는다 (비력의 반대)",
          np.allclose(g, [0, 1, 0]))

    # 양자화 하한 — 이진 이미지에서 하한이 잡히는가
    binary = np.zeros((80, 80, 3), np.uint8)
    binary[:, 30:32, 1] = 255
    qz = RP.quantization_floor(binary)
    check(f"이진 이미지 → σ 하한 {qz.get('sigma_floor_px')}px",
          qz["binary"] and abs(qz["sigma_floor_px"] - 1 / np.sqrt(12)) < 1e-3)
    ramp = np.zeros((80, 80, 3), np.uint8)
    for k, v in enumerate((60, 160, 255, 160, 60)):
        ramp[:, 28 + k, 1] = v
    check("밝기 기울기가 있으면 하한을 걸지 않는다",
          not RP.quantization_floor(ramp)["binary"])

    # 격자에서 거리·화소 규약 복원 — 정답을 알고 있는 합성 격자로
    keep = CALIB.ACTIVE_PROFILE
    try:
        CALIB.use_profile("legacy")
        cp = dict(CALIB.CAMERA_PARAMS)
        W, H = cp["resolution"]
        cp.update({"image_w": W, "image_h": H, "n_v": CALIB.N_VERTICAL,
                   "n_h": CALIB.N_HORIZONTAL, "fov_h_deg": CALIB.FOV_DEG,
                   "fov_v_deg": CALIB.FOV_DEG, "laser_tilt_deg": 0.0,
                   "laser_roll_deg": 0.0})
        la = CALIB.make_line_angles()
        f, b = cp["f_px"], cp["b_m"]
        cx, cy = cp["cx_px"], cp["cy_px"]
        for z_true, flip in ((1.55, False), (1.55, True), (2.70, False)):
            img = np.zeros((H, W, 3), np.uint8)
            for i in range(CALIB.N_VERTICAL):
                u = f * np.tan(la[f"V{i}"]["angle_rad"]) - f * b / z_true + cx
                if flip:
                    u = 2 * cx - u
                k = int(round(u))
                if 1 <= k < W - 1:
                    img[:, k:k + 2, 1] = 255
            pose = RP.estimate_grid_pose(img, cp, la)
            ok = (pose.get("ok") and pose["flipped"] == flip
                  and abs(pose["z_est_m"] - z_true) < 0.02)
            check(f"격자에서 복원: 참 {z_true}m/뒤집힘 {flip} → "
                  f"{pose.get('z_est_m')}m/{pose.get('flipped')}", ok)
    finally:
        CALIB.use_profile(keep)


def test_single_line_member():
    """[14] 격자선 한 줄만 걸린 부재는 합격을 못 준다"""
    print("\n[14] 단면 미분해 — 한 줄짜리 부재")
    rng = np.random.default_rng(3)
    g = np.array([0.0, 1.0, 0.0])
    cp = {"f_px": 1593.0, "b_m": 0.15, "cx_px": 1224.0, "cy_px": 1024.0,
          "image_w": 2448, "image_h": 2048}

    # 레이저 평면 한 장 위의 연직선 — 실제 동바리를 한 줄이 스친 모습
    y = np.linspace(-1.0, 1.0, 600)
    one = np.column_stack([np.full_like(y, -0.77), y, np.full_like(y, 1.70)])
    r = PIPE.measure_region(one, "shoring", g, cp, n_lines=1)
    j = r.get("judge") or {}
    check(f"한 줄 → 판정 '{j.get('judgement')}' (합격 아님)",
          j.get("judgement") == "판정보류(단면 미분해)"
          and j.get("is_pass") is not True,
          f"측정 {r.get('theta_deg')}°")
    check("한 줄 표시가 결과에 남는다", r.get("single_plane") is True)

    # 평면에 **수직인** 기울기는 한 줄로는 안 보인다 — 그래서 합격 금지다
    tilt = np.radians(2.0)
    two = one.copy()
    two[:, 0] += np.sin(tilt) * (y - y.mean())      # 평면 밖으로 눕힌다
    th_true = np.degrees(np.arccos(abs(
        np.dot(np.array([np.sin(tilt), np.cos(tilt), 0.0]), g))))
    proj = one.copy()                              # 한 줄이 보는 것은 이것뿐
    r2 = PIPE.measure_region(proj, "shoring", g, cp, n_lines=1)
    check(f"실제 {th_true:.2f}° 기울어도 한 줄에서는 "
          f"{r2['theta_deg']:.4f}° 로 보인다 — 그래서 합격 금지가 맞다",
          abs(r2["theta_deg"]) < 0.01 and th_true > 1.0)

    # ── 한 줄짜리 판정의 비대칭 ──
    # 잰 값은 참값의 **하한**이다(참값 = √(잰값² + 못 본 성분²)).
    # 그래서 부등식이 한쪽으로만 성립한다: 하한이 이미 허용치를 넘었으면
    # 참값은 더 크니 기준초과가 **확정**이고, 이내면 아무 말도 못 한다.
    # 즉 한 줄짜리 측정은 부재를 떨어뜨릴 수는 있어도 붙여줄 수는 없다.
    big = one.copy()
    big[:, 2] += np.tan(np.radians(2.0)) * (y - y.mean())   # 평면 '안'으로
    r4 = PIPE.measure_region(big, "shoring", g, cp, n_lines=1)
    j4 = r4.get("judge") or {}
    check(f"한 줄이라도 하한 {r4['theta_deg']:.3f}° 가 허용 "
          f"{j4.get('allow_deg')}° 를 넘으면 '{j4.get('judgement')}' 확정",
          j4.get("judgement") == "기준초과" and j4.get("is_pass") is False
          and j4.get("theta_deg_is_lower_bound") is True)
    check("허용치 이내인 한 줄은 여전히 판정보류",
          (r.get("judge") or {}).get("is_pass") is None)

    # 두 줄 이상이면 정상 판정으로 돌아온다
    t = np.linspace(-1.0, 1.0, 600)
    axis = np.array([np.sin(np.radians(0.3)), np.cos(np.radians(0.3)), 0.0])
    ph = rng.uniform(-np.pi, np.pi, 600)
    cyl = (np.outer(t, axis)
           + 0.0243 * np.column_stack([np.cos(ph), np.zeros(600), np.sin(ph)])
           + np.array([0.2, 0.0, 1.6]))
    r3 = PIPE.measure_region(cyl, "shoring", g, cp, n_lines=3)
    j3 = r3.get("judge") or {}
    check(f"세 줄 걸린 부재 → '{j3.get('judgement')}' "
          f"({r3['theta_deg']:.3f}°, 정답 0.3°)",
          j3.get("judgement") != "판정보류(단면 미분해)"
          and abs(r3["theta_deg"] - 0.3) < 0.5)


def test_silhouette_recovery():
    """[15] 가림 그림자로 한 줄짜리 부재의 옆 기울기를 되찾는다"""
    print("\n[15] eq8 실루엣 — 한 줄짜리 부재 되살리기")
    f, b, cx, cy = 826.0, 0.15, 634.5, 531.5
    g = np.array([0.0, 1.0, 0.0])
    Z = 1.70
    rng = np.random.default_rng(9)

    w = EQ8.shadow_width_px(Z, 2.70, f, b)
    check(f"그림자 폭 예측 {w:.1f}px = f·b·(1/Z−1/Zb)",
          abs(w - f * b * (1 / Z - 1 / 2.70)) < 1e-9)
    check("배경이 부재보다 앞이면 그림자 없음",
          EQ8.shadow_width_px(Z, 1.2, f, b) is None)

    for tilt in (0.0, 0.5, 1.5):
        v = np.linspace(50, 1000, 21)
        Y = (v - cy) * Z / f
        X0 = 0.20 + np.tan(np.radians(tilt)) * (Y - Y.mean())
        u = (X0 - b) * f / Z + cx + rng.normal(0, 2.0, len(v))
        r = EQ8.axis_from_edges(np.column_stack([v, u]), Z, f, cx, cy, b, g)
        check(f"옆 기울기 {tilt}° → 복원 {r['theta_deg']:.4f}° "
              f"(잔차 {r['rms_px']:.2f}px)",
              r["ok"] and abs(r["theta_deg"] - tilt) < 0.3)

    # 가짜 가장자리를 지어내지 않는가 — 이것이 없으면 없는 기울기를 만든다
    v = np.linspace(50, 1000, 21)
    u = (0.20 - b) * f / Z + cx + rng.normal(0, 1.0, len(v))
    u[[3, 9, 15]] += 40.0
    r = EQ8.axis_from_edges(np.column_stack([v, u]), Z, f, cx, cy, b, g)
    check(f"몇 행만 튀면 버리고 이어간다 (버린 점 {r.get('n_dropped')}개, "
          f"{r['theta_deg']:.4f}°)",
          r["ok"] and r.get("n_dropped", 0) >= 3 and abs(r["theta_deg"]) < 0.3)
    u2 = (0.20 - b) * f / Z + cx + rng.normal(0, 8.0, len(v))
    r2 = EQ8.axis_from_edges(np.column_stack([v, u2]), Z, f, cx, cy, b, g)
    check("가장자리가 통째로 어지러우면 되살리지 않는다 "
          f"(잔차 {r2['rms_px']:.2f}px)", not r2["ok"])

    # 그림자에서 가장자리를 실제로 뽑는가 — 합성 행으로
    W, H = 1269, 1063
    sig = np.zeros((H, W), np.float32)
    edge_true = 400
    gap = int(round(EQ8.shadow_width_px(Z, 2.70, f, b)))
    rows = list(range(100, 1000, 45))
    for vv in rows:
        sig[vv, :edge_true + 1] = 255.0
        sig[vv, edge_true + 1 + gap:] = 255.0
    got = EQ8.find_shadow_edges(sig, rows, edge_true - 20, Z, 2.70, f, b)
    check(f"합성 그림자에서 가장자리 {got['n_found']}/{got['n_checked']}행 검출",
          got["n_found"] == len(rows)
          and all(abs(e[1] - edge_true) <= 1 for e in got["edges"]))


def test_line_gap_recovery():
    """[16] 가로선 화소의 끊김에서 부재 폭·중심·옆 기울기를 되돌린다"""
    print("\n[16] 가로선 끊김 — 폭과 옆 기울기")
    f, b, cx, cy = 826.0, 0.15, 634.5, 531.5
    g = np.array([0.0, 1.0, 0.0])
    Zm, Zbg = 1.70, 2.70
    R_m = 0.0243
    R_px = R_m * f / Zm
    w = EQ8.shadow_width_px(Zm, Zbg, f, b)
    drop = 4.0                      # 추적기가 끊는 폭(합성에도 넣는다)
    X_c0 = 0.20

    def build(tilt_deg, n=21, decoy=False):
        """가로선 화소열을 만든다 — 부재 그림자만큼 끊어서."""
        lines = {}
        for i, vv in enumerate(np.linspace(120.0, 980.0, n)):
            Y = (vv - cy) * Zm / f
            X = X_c0 + np.tan(np.radians(tilt_deg)) * Y
            u_c = (X - b) * f / Zm + cx
            lo = u_c + w - R_px - 0.5 * drop
            hi = u_c + w + R_px + 0.5 * drop
            u = np.arange(60.0, 1200.0, 0.25)
            u = u[(u < lo) | (u > hi)]
            # 다른 선이 가로지른 자리 — 작은 끊김(보정용 표본)
            for uk in (150.0, 300.0, 900.0, 1050.0):
                u = u[(u < uk) | (u > uk + drop)]
            if decoy:               # 남의 부재 그림자 — 3R 떨어진 자리
                d0 = u_c + w + 6 * R_px
                u = u[(u < d0) | (u > d0 + 2 * R_px + drop)]
            lines[f"H{i}"] = np.column_stack([u, np.full_like(u, vv)])
        return lines

    lines = build(0.0)
    sh = EQ8.member_edges_from_lines(lines, (X_c0 - b) * f / Zm + cx,
                                     Zm, Zbg, f, b)
    check(f"끊김을 {sh['n_found']}/21 줄에서 찾는다", sh["n_found"] == 21)
    check(f"끊김폭에서 지름 복원 {sh['radius_px']:.2f}px (참 {R_px:.2f}px, "
          f"추적손실 {sh['dropout_px']}px 보정)",
          abs(sh["radius_px"] - R_px) < 0.06 * R_px)

    for tilt in (0.0, 0.5, 1.5):
        r = EQ8.axis_from_line_gaps(build(tilt), (X_c0 - b) * f / Zm + cx,
                                    Zm, Zbg, f, cx, cy, b, g)
        check(f"옆 기울기 {tilt}° → 복원 {r.get('theta_deg')}° "
              f"(잔차 {r.get('rms_px')}px)",
              r.get("ok") and abs(r["theta_deg"] - tilt) < 0.05)

    # ── 자기 확인이 아님을 못 박는다 ──
    # 부재 위치 힌트를 흔들어도 답이 그대로여야 한다. 힌트가 답에 스미면
    # 그건 측정이 아니라 항등식이다(창 중앙을 쓰던 옛 방식이 그랬다).
    base = EQ8.axis_from_line_gaps(build(1.0), (X_c0 - b) * f / Zm + cx,
                                   Zm, Zbg, f, cx, cy, b, g)
    moved = [EQ8.axis_from_line_gaps(build(1.0),
                                     (X_c0 - b) * f / Zm + cx + dd,
                                     Zm, Zbg, f, cx, cy, b, g)["theta_deg"]
             for dd in (-20.0, 20.0)]
    check(f"힌트를 ±20px 옮겨도 답이 끌려가지 않는다 "
          f"({base['theta_deg']}° vs {moved[0]}°/{moved[1]}°, 참 1.0°)",
          all(abs(m - 1.0) < 0.02 for m in moved))

    # 남의 끊김은 물지 않는다 — 되돌린 중심이 부재 반폭 밖이면 버린다
    sh2 = EQ8.member_edges_from_lines(build(0.0, decoy=True),
                                      (X_c0 - b) * f / Zm + cx, Zm, Zbg, f, b)
    check(f"6R 떨어진 남의 끊김에 속지 않는다 (폭 {sh2['radius_px']:.2f}px)",
          sh2["n_found"] == 21 and abs(sh2["radius_px"] - R_px) < 0.06 * R_px)

    # 배경이 없으면 그림자가 없으니 조용히 실패해야 한다
    bad = EQ8.axis_from_line_gaps(lines, (X_c0 - b) * f / Zm + cx,
                                  Zm, Zm, f, cx, cy, b, g)
    check("깊이차가 없으면 되살리지 않는다", not bad.get("ok"))


def test_pointcloud_plot():
    """[17] 3D 점군 그림 — 부재 구분이 색으로 반영되는가"""
    print("\n[17] 3D 점군 시각화 (plot_points3d)")
    try:
        import matplotlib                                    # noqa: F401
    except ImportError:
        print("  [건너뜀] matplotlib 없음 — PIL 판으로 폴백됨")
        return
    P3 = _load("plot_points3d")
    g = np.array([0.0, 1.0, 0.0])
    rng = np.random.default_rng(11)

    def region(cls, kind, c, n=300, aux=0):
        P = rng.normal(0, 0.02, (n, 3)) + np.asarray(c, float)
        r = {"class": cls, "kind": kind, "status": "measured",
             "point_xyz": P, "point_lid": np.array([f"V{i%3}" for i in range(n)],
                                                   dtype=object),
             "judge": {"is_pass": True}, "flatness": {}}
        if aux:
            r["aux_point_xyz"] = rng.normal(0, 0.02, (aux, 3)) \
                + np.asarray(c, float)
        return r

    res = {"regions": [region("wall", "plane_vertical", (0, 0, 2.5), 400, 200),
                       region("shoring", "axis_vertical", (-0.5, 0, 1.6), 300, 90),
                       region("shoring", "axis_vertical", (0.5, 0, 1.6), 300, 70)]}

    gs = P3.groups_from_result(res, g_hat=g, by="member")
    labels = [x["label"] for x in gs]
    check(f"부재별 묶음 {len(gs)}개 (측정 3 + 가로선 3)", len(gs) == 6)
    check("같은 종류 부재는 번호로 갈린다 — 엑셀 9번 시트와 같은 번호",
          any("shoring #2" in l for l in labels)
          and any("shoring #3" in l for l in labels))
    cols = [tuple(np.round(x["color"], 4)) for x in gs if not x["aux"]]
    check(f"부재마다 색이 다르다 ({len(set(cols))}/3)", len(set(cols)) == 3)
    aux = [x for x in gs if x["aux"]]
    check("가로선 점은 '투영' 으로 표시된다",
          len(aux) == 3 and all("투영" in x["label"] for x in aux))

    n_meas = sum(len(x["xyz"]) for x in gs if not x["aux"])
    check(f"솎지 않는다 — 측정점 {n_meas}개 전부", n_meas == 1000)
    n_str = sum(len(x["xyz"]) for x in P3.groups_from_result(
        res, g_hat=g, by="member", stride=10) if not x["aux"])
    check(f"stride=10 이면 {n_str}개로 준다", n_str == 100)

    gl = P3.groups_from_result(res, g_hat=g, by="line")
    check(f"--by line 이면 레퍼런스와 같은 V·H 색칠 ({len(gl)}묶음)",
          len(gl) == 2 and {x["color"] for x in gl}
          == {"tab:blue", "tab:red"})

    # frame 은 좌표를 바꾼다 — camera 는 삼각측량이 낸 값 그대로,
    # gravity 는 중력 정렬(가로·깊이·높이). 섞이면 벽이 기울어 보인다.
    raw = np.asarray(res["regions"][0]["point_xyz"], float)
    cam = P3.groups_from_result(res, g_hat=g, by="member",
                                frame="camera")[0]["xyz"]
    grv = P3.groups_from_result(res, g_hat=g, by="member",
                                frame="gravity")[0]["xyz"]
    check("frame=camera 는 삼각측량 좌표 그대로", np.allclose(cam, raw))
    check("frame=gravity 는 중력 정렬 좌표 — 깊이축이 Z 에서 분리된다",
          not np.allclose(grv, raw)
          and abs(np.ptp(grv[:, 1]) - np.ptp(raw[:, 2])) < 1e-9)

    import tempfile
    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "pc.png")
        got = P3.save_pointcloud_mpl(out, res, g_hat=g, title="검증")
        check(f"그림 파일이 나온다 ({os.path.getsize(out) // 1024}KB)"
              if got else "그림 파일이 나온다",
              got == out and os.path.getsize(out) > 20_000)
        out2 = os.path.join(td, "pc_iso.png")
        check("views=iso 도 나온다",
              P3.save_pointcloud_mpl(out2, res, g_hat=g, views="iso") == out2)


def test_member_span_reporting():
    """[18] 같은 규격 부재가 거리 때문에 다른 길이로 읽히지 않는가"""
    print("\n[18] 측정 구간 — '동바리 높이가 제각각' 문제")
    g = np.array([0.0, 1.0, 0.0])
    cp = {"f_px": 1593.0, "b_m": 0.15, "cx_px": 1224.0, "cy_px": 1024.0,
          "image_w": 2448, "image_h": 2048}
    # 같은 화소 구간(v 92~1955)을 서로 다른 거리에서 본 동바리 3본.
    # 격자가 고정 화각을 덮으므로 덮는 미터 길이가 거리에 비례한다.
    v = np.linspace(92.0, 1955.0, 400)
    spans, zs = [], (1.705, 1.652, 1.595)
    for z in zs:
        Y = (v - cp["cy_px"]) * z / cp["f_px"]
        spans.append(float(np.ptp(Y)))
    ratio = [sp / z for sp, z in zip(spans, zs)]
    check(f"구간/거리 비가 같다 ({ratio[0]:.5f} / {ratio[1]:.5f} / "
          f"{ratio[2]:.5f})",
          max(ratio) - min(ratio) < 1e-6 * max(ratio))
    check(f"그런데 구간 자체는 다르다 ({spans[0]:.4f} / {spans[1]:.4f} / "
          f"{spans[2]:.4f} m) — 이걸 '높이' 로 내보내면 오해가 생긴다",
          max(spans) - min(spans) > 0.10)

    # 파이프라인이 이 상황을 '하한' 으로 표시하고 이유를 남기는가
    res = {"regions": []}
    for k, z in enumerate(zs):
        Y = (v - cp["cy_px"]) * z / cp["f_px"]
        P = np.column_stack([np.full_like(Y, -0.5 + 0.6 * k), Y,
                             np.full_like(Y, z)])
        res["regions"].append({
            "class": "shoring", "kind": "axis_vertical", "status": "measured",
            "point_xyz": P, "point_uv": np.column_stack([
                np.full_like(v, 300.0 + 700.0 * k), v]),
            "point_uv_box": [300.0 + 700.0 * k, 92.0,
                             300.0 + 700.0 * k, 1955.0],
            "axis": {"length_m": float(np.ptp(Y))},
            "judge": {"is_pass": True}, "flatness": {}})
    PIPE.finalize_extents(res)
    lb = [r.get("length_is_lower_bound") for r in res["regions"]]
    check(f"양 끝이 격자 한계면 길이는 하한으로 표시 ({lb})", all(lb))
    ends = res["regions"][0].get("extent_ends") or {}
    check(f"어느 끝이 잘렸는지 남는다 ({ends})",
          ends.get("위쪽") and ends.get("아래쪽"))
    note = (res.get("summary") or {}).get("span_note")
    check("조서가 '거리 차이 때문' 이라고 직접 적는다",
          bool(note) and "거리 차이" in note and "부재 길이가 다르다는 근거가 "
          "아니다" in note)

    # 부재 전체가 화면 안이면 하한이 아니라 확정이어야 한다
    res2 = {"regions": [dict(res["regions"][0])]}
    res2["regions"][0] = dict(res2["regions"][0])
    res2["regions"][0]["point_uv_box"] = [300.0, 400.0, 300.0, 1600.0]
    res2["regions"].append(dict(res["regions"][1]))
    PIPE.finalize_extents(res2)
    check("부재가 격자 안에서 끝나면 '확정'",
          res2["regions"][0].get("length_is_lower_bound") is False)


def test_cylinder_aux_surface():
    """[19] 동바리에 맺힌 가로선 점이 원통을 감싸는가"""
    print("\n[19] 원통면 배치 — '동바리가 둥글게 안 나온다'")
    f, b, cx, cy = 1593.0, 0.15, 1224.0, 1024.0
    R, Zc = 0.0243, 1.70                       # Ø48.6 동바리, 중심 깊이

    # 연직 원통을 가로지르는 광선들 — 부재 폭 안에서 u 를 훑는다
    lat = np.linspace(-0.85 * R, 0.85 * R, 41)
    u = (lat - b) * f / (Zc - R) + cx          # 대략 앞면에 맞도록
    uh = (u - cx) / f
    vh = np.zeros_like(uh)
    d = np.array([0.0, 1.0, 0.0])
    c0 = np.array([0.0, 0.0, Zc])
    Z = PIPE._ray_cylinder(uh, vh, b, c0, d, R)
    ok = np.isfinite(Z)
    check(f"광선 {int(ok.sum())}/{len(Z)} 이 원통을 맞는다", ok.sum() >= 30)
    P = np.column_stack([b + Z[ok] * uh[ok], Z[ok] * vh[ok], Z[ok]])
    q = P - c0
    rad = np.linalg.norm(q - np.outer(q @ d, d), axis=1)
    check(f"모든 점이 축에서 정확히 R 만큼 ({rad.mean()*1000:.3f}±"
          f"{rad.std()*1000:.3f}mm, R={R*1000:.1f}mm)",
          abs(rad.mean() - R) < 1e-6 and rad.std() < 1e-6)
    check(f"앞면이다 (깊이 < 중심)", float(P[:, 2].max()) < Zc + 1e-9)
    span = float(np.ptp(P[:, 2]))
    check(f"깊이가 휜다 — 폭 {span*1000:.1f}mm (평면이면 0, 반지름 넘으면 "
          f"기하 불가)", 0.002 < span <= R + 1e-9)

    # 스치는 광선은 버린다 — 접점 근처에서 깊이가 튀는 것을 막는다
    graze = PIPE._ray_cylinder(np.array([(0.0 - b) * f / Zc / f + R / Zc]),
                               np.zeros(1), b, c0, d, R)
    check("실루엣을 스치는 광선은 배정하지 않는다",
          not np.isfinite(graze[0]) or abs(graze[0] - Zc) < R)

    # 원통을 아예 빗나가면 NaN
    miss = PIPE._ray_cylinder(np.array([0.5]), np.zeros(1), b, c0, d, R)
    check("빗나간 광선은 NaN", not np.isfinite(miss[0]))

    # 지름·중심을 못 잰 부재는 둥글게 그리지 않는다 (지어내지 않기)
    u_c = cx + f * (0.0 - b) / (Zc - R)

    def region(width_source, center_off):
        y = np.linspace(-0.5, 0.5, 200)
        P = np.column_stack([np.zeros_like(y), y,
                             np.full_like(y, Zc - R)])
        r = {"class": "shoring", "kind": "axis_vertical", "status": "measured",
             "point_xyz": P, "point_uv": np.column_stack(
                 [np.full_like(y, u_c), np.linspace(200, 1800, 200)]),
             "axis": {"direction": [0.0, 1.0, 0.0],
                      "centroid": [0.0, 0.0, Zc - R]},
             "point_uv_box": [u_c, 200.0, u_c, 1800.0],
             "half_width_m": R, "width_source": width_source,
             "judge": {"is_pass": True}, "flatness": {}}
        if center_off is not None:
            r["center_offset_px"] = center_off
        return r

    cp = {"f_px": f, "b_m": b, "cx_px": cx, "cy_px": cy}
    # 화소는 X=0 인 부재를 보는 광선이어야 한다. X = b + Z·û 이므로
    # û = (0 − b)/Z — 주점(cx)이 아니라 그만큼 옆이다. 이걸 틀리면
    # 광선이 부재를 150mm 빗나가고 "원통면인데 안 휘네" 로 보인다.
    lines = {"H0": np.column_stack([
        np.linspace(u_c - 18, u_c + 18, 60), np.full(60, cy)])}
    for src, off, want in (("가로선 끊김 폭", 0.0, True),
                           ("같은 장면의 다른 부재 폭", None, False),
                           ("기본값(폭 미측정)", None, False)):
        rr = [region(src, off)]
        PIPE.place_aux_points(rr, lines, cp)
        A = rr[0].get("aux_point_xyz")
        curved = A is not None and len(A) and float(np.ptp(A[:, 2])) > 1e-6
        check(f"폭 근거 '{src}' → {'원통면' if want else '접평면'} "
              f"({rr[0].get('aux_surface')})", bool(curved) == want)

    # 왜 안 둥근가 — 선검출이 아니라 기하의 한계임을 조서가 말하는가
    res = {"regions": [{"class": "shoring", "kind": "axis_vertical",
                        "status": "measured",
                        "aux_surface": "접평면 (지름 미상)"}]}
    note = PIPE.roundness_note(res, {"H0": float("inf"), "V0": 1.0})
    check("가로선이 깊이를 못 주면 이유를 적는다",
          bool(note) and "선검출 문제가 아니다" in note and "1/sin" in note)
    check("가로선이 깊이를 주면 그런 말을 하지 않는다",
          PIPE.roundness_note(res, {"H0": 1.41, "V0": 1.41}) is None)

    # 격자를 굴리면 지름 가정 없이 원통면이 복원되는가
    EQ = EQ7
    Rr, Zc2 = 0.0243, 1.70
    for gam, want in ((0.0, False), (10.0, True), (45.0, True)):
        n = EQ.plane_normal("beta", np.radians(-6.0),
                            roll_rad=np.radians(gam))
        if not np.isfinite(EQ.depth_gain(n)):
            check(f"굴림 {gam:.0f}° → 가로선으로 깊이 복원 불가", not want)
            continue
        th = np.linspace(-1.15, 1.15, 300)
        X = Rr * np.sin(th)
        Z0 = Zc2 - Rr * np.cos(th)
        Y = -(n[0] * X + n[2] * Z0) / n[1]
        P = np.column_stack([X, Y, Z0])
        P = P[np.abs(Y) < 1.0]
        u = 1593.0 * (P[:, 0] - 0.15) / P[:, 2] + 1224.0
        v = 1593.0 * P[:, 1] / P[:, 2] + 1024.0
        xyz, _k = EQ.triangulate_plane(np.column_stack([u, v]), n,
                                       1593.0, 0.15, 1224.0, 1024.0)
        q = xyz - np.array([0.0, 0.0, Zc2])
        q[:, 1] = 0.0
        rms = float(np.sqrt(np.mean(
            (np.linalg.norm(q, axis=1) - Rr) ** 2)))
        check(f"굴림 {gam:.0f}° → 참 원통면 잔차 {rms*1000:.3f}mm "
              f"(지름 가정 없음)", want and rms < 1e-4)



def test_hardware_contract():
    """[20] 실장비 입력 규약 — 무엇이 없으면 막고, 무엇이면 경고인가"""
    print("\n[20] 하드웨어 입력 규약 (hardware.py)")
    HW = _load("hardware")
    import tempfile
    from PIL import Image

    bad, _w, _i = HW.check_params(HW.template())
    check(f"빈 서식은 필수 {len(bad)}건을 다 잡는다",
          len(bad) == len(HW.REQUIRED))

    good = {"camera": {"f_px": 1593.0, "cx_px": 1224.0, "cy_px": 1024.0,
                       "sensor_W": 2448, "sensor_H": 2048},
            "baseline_m": 0.15,
            "grid": {"n_vertical": 21, "n_horizontal": 21, "fov_deg": 60.8},
            "laser": {"roll_deg": 0.0, "tilt_deg": 0.0}}
    bad, warn, info = HW.check_params(good)
    check("갖춘 사양은 통과", not bad)
    check(f"기선·초점거리로 깊이잡음을 미리 알려준다 "
          f"({info.get('깊이잡음_mm@1.7m_σu0.3px')}mm @1.7m)",
          0.5 < (info.get("깊이잡음_mm@1.7m_σu0.3px") or 0) < 20)
    check("굴림 0° 면 '가로선이 깊이를 못 준다' 고 경고",
          any("굴림" in w for w in warn))
    g2 = dict(good); g2["laser"] = {"roll_deg": 20.0, "tilt_deg": 0.0}
    check("굴림 20° 면 그 경고가 없다",
          not any("굴림" in w for w in HW.check_params(g2)[1]))

    # 초점거리를 빼면 막아야 한다 — 깊이가 통째로 배율만큼 틀리기 때문
    g3 = {"camera": {"cx_px": 1224.0, "cy_px": 1024.0,
                     "sensor_W": 2448, "sensor_H": 2048},
          "baseline_m": 0.15}
    check("f_px 가 없으면 막는다",
          any("camera.f_px" in m for m in HW.check_params(g3)[0]))

    with tempfile.TemporaryDirectory() as td:
        a = np.zeros((160, 240, 3), np.uint8)
        a[:, ::20, 1] = 255
        Image.fromarray(a).save(os.path.join(td, "laser_on.png"))
        import json as _j
        _j.dump(good, open(os.path.join(td, "camera_params.json"), "w"))
        r = HW.check_capture(td, verbose=False)
        check("이미지 + 사양만 있으면 '검측 가능'", r["ok"])
        d = r["이미지"]["laser_on"]
        check(f"이진 렌더를 잡아낸다 ({d['이진(안티에일리어싱 없음)']})",
              d["이진(안티에일리어싱 없음)"] is True)
        check("OFF 프레임·IMU 가 없으면 경고로 남긴다",
              any("OFF" in w for w in r["경고"])
              and any("IMU" in w for w in r["경고"]))
        os.remove(os.path.join(td, "camera_params.json"))
        check("사양 파일이 없으면 막는다",
              not HW.check_capture(td, verbose=False)["ok"])
        os.remove(os.path.join(td, "laser_on.png"))
        _j.dump(good, open(os.path.join(td, "camera_params.json"), "w"))
        check("이미지가 없으면 막는다",
              not HW.check_capture(td, verbose=False)["ok"])

    # ── 점검기의 핵심: 사양의 숫자가 이미지의 것인가 ──
    # 파일이 다 있어도 사양이 다른 장비의 것이면 검측은 그대로 돌아가고
    # 결과만 조용히 틀린다. 그 상태를 잡는지 본다.
    import shutil as _sh
    import tempfile as _td
    _t = _td.mkdtemp()
    try:
        d = HW.demo(os.path.join(_t, "예제"), roll_deg=20.0, verbose=False)
        r = HW.check_capture(d, verbose=False)
        gi = r["기하"]
        check(f"규약을 만족하는 촬영은 '검측 가능' "
              f"(측정 f {gi.get('측정 초점거리_px')}px / 사양 "
              f"{gi.get('사양 초점거리_px')}px)", r["쓸만함"])
        check(f"굴림을 이미지에서 직접 잰다 "
              f"({gi.get('측정 굴림_deg')}° vs 사양 {gi.get('사양 굴림_deg')}°)",
              abs((gi.get("측정 굴림_deg") or 0) - 20.0) <= 2.0)

        pj = os.path.join(d, "camera_params.json")
        orig = json.load(open(pj, encoding="utf-8"))

        def _mut(fn):
            c = json.loads(json.dumps(orig))
            fn(c)
            json.dump(c, open(pj, "w", encoding="utf-8"))
            out = HW.check_capture(d, verbose=False)
            json.dump(orig, open(pj, "w", encoding="utf-8"))
            return out

        rr = _mut(lambda c: c["camera"].update(f_px=c["camera"]["f_px"] * 1.2))
        check("초점거리를 20% 틀리게 적으면 잡는다 (선 간격은 거리와 무관)",
              bool(rr["불일치"]) and not rr["쓸만함"])
        rr = _mut(lambda c: c["laser"].update(roll_deg=0.0))
        check("굴림을 틀리게 적으면 잡는다", bool(rr["불일치"]))
        rr = _mut(lambda c: c["grid"].update(n_vertical=15))
        check("선 수를 틀리게 적으면 잡는다", bool(rr["불일치"]))
        rr = _mut(lambda c: c.update(baseline_m=c["baseline_m"] * 2))
        check("실측 거리를 함께 주면 기선 오류도 잡는다", bool(rr["불일치"]))

        # 기선은 사진 한 장으로는 검증할 수 없다 — b 와 Z 가 b/Z 로만
        # 식에 들어오기 때문이다. 못 잡는 것이 정상이고, 그렇게 안내해야 한다.
        rr = _mut(lambda c: (c.pop("측정거리_m", None),
                             c.update(baseline_m=c["baseline_m"] * 2)))
        check("실측 거리가 없으면 기선은 검증 못 한다고 안내한다",
              not rr["불일치"]
              and any("기선을 검증할 수 없다" in w for w in rr["경고"]))
    finally:
        _sh.rmtree(_t, ignore_errors=True)

    # ── 파이프라인은 규약을 만족하는 촬영만 받는다 ──
    try:
        RP.run(image="/dev/null", params=None, verbose=False)
        _got = ""
    except ValueError as e:
        _got = str(e)
    except Exception as e:
        _got = f"다른 예외: {e}"
    check("사양 없이 파이프라인을 부르면 막고 점검기를 알려 준다",
          "camera_params.json" in _got and "hardware.py" in _got)

    # ── 규약서·검사기·파서가 같은 자리를 보는가 ──
    # 이 셋이 어긋나면 업체가 값을 제대로 적어 놓고도 0 으로 돌아간다.
    # 굴림이 0 이면 가로선이 깊이를 못 주므로 결과가 통째로 달라진다.
    import tempfile as _tf
    base = {"camera": {"f_px": 1593.0, "cx_px": 1224.0, "cy_px": 1024.0,
                       "sensor_W": 2448, "sensor_H": 2048},
            "baseline_m": 0.15,
            "grid": {"n_vertical": 5, "n_horizontal": 5, "fov_deg": 60.0}}

    def _parse(extra):
        d = json.loads(json.dumps(base))
        d.update(extra)
        with _tf.NamedTemporaryFile("w", suffix=".json", delete=False) as fp:
            json.dump(d, fp)
            path = fp.name
        cp, meta = RP._params_from_file(path, 2448, 2048)
        la = RP._line_angles(cp)
        os.unlink(path)
        return cp, meta, la

    for nm, extra in (
            ("laser 블록", {"laser": {"roll_deg": 20.0, "tilt_deg": 6.0}}),
            ("grid 안", {"grid": dict(base["grid"], laser_roll_deg=20.0,
                                     laser_tilt_deg=6.0)}),
            ("최상위", {"laser_roll_deg": 20.0, "laser_tilt_deg": 6.0})):
        cp, _m, la = _parse(extra)
        gh = float(np.median([v["depth_gain"] for k, v in la.items()
                              if k.startswith("H")]))
        check(f"굴림·수렴각을 '{nm}' 에 적어도 읽는다 "
              f"(roll {cp['laser_roll_deg']:.0f}°, H선 이득 {gh:.2f})",
              abs(cp["laser_roll_deg"] - 20.0) < 1e-9
              and abs(cp["laser_tilt_deg"] - 6.0) < 1e-9
              and np.isfinite(gh))
    cp0, _m0, la0 = _parse({})
    check("안 적으면 0 으로 두고 가로선 이득은 무한(깊이 못 줌)",
          cp0["laser_roll_deg"] == 0.0
          and not np.isfinite(float(np.median(
              [v["depth_gain"] for k, v in la0.items()
               if k.startswith("H")]))))

    cpn, mn, lan = _parse({"lines": {"V0": {"normal": [3.0, 0.0, 4.0]}}})
    check(f"선별 평면 법선을 직접 주면 그대로 쓴다 "
          f"({mn.get('평면 법선 직접 지정')}, 정규화 "
          f"{[round(x, 3) for x in lan['V0']['normal']]})",
          np.allclose(lan["V0"]["normal"], [0.6, 0.0, 0.8])
          and abs(lan["V0"]["depth_gain"] - 1.0) < 1e-9)


def _colab_hint(HW):
    """코랩 밖에서 hardware.colab() 을 부르면 안내를 주는가."""
    try:
        HW.colab()
    except RuntimeError as e:
        return "경로를 직접" in str(e)
    except Exception:
        return False
    return False


def test_sample_capture():
    """[21] 예제 입력 데이터셋 — 격자 모양·거리 복원·참값 회수"""
    print("\n[21] 예제 입력 데이터셋 (make_sample_capture.py)")
    MK = _load("make_sample_capture")
    RP = _load("run_pipeline")
    DET = _load("A_선검출")
    PIPE = _load("pipeline_region")
    import io as _io
    import contextlib

    W, H = MK.W, MK.H
    on, off, Pmap, sid, planes = MK.render()
    check("[21] 렌더가 네 면을 모두 담는다",
          all((sid == k).sum() > 5000 for k in (0, 1, 2, 3)),
          ", ".join(f"{n}={int((sid == k).sum())}"
                    for k, n in ((0, "벽"), (1, "바닥"),
                                 (2, "동바리1"), (3, "동바리2"))))

    # ── 격자가 유한한 사각형이어야 한다 ──
    # 평면식만 쓰면 선이 화면 끝까지 이어져 21×21 칸이 어디서 끝나는지
    # 보이지 않는다. 실제 DOE 는 유한한 부채꼴이다.
    g = (on[:, :, 1].astype(float)
         - 0.5 * (on[:, :, 0].astype(float) + on[:, :, 2].astype(float)))
    ys, xs = np.nonzero(g > 40)
    margin = min(xs.min(), W - 1 - xs.max(), ys.min(), H - 1 - ys.max())
    check("[21] 격자가 화면 끝까지 이어지지 않는다 (유한한 부채꼴)",
          margin > 30,
          f"가장 좁은 여백 {margin}px  (u {xs.min()}~{xs.max()}, "
          f"v {ys.min()}~{ys.max()})")
    check("[21] 격자는 21×21 선",
          MK.N_V == 21 and MK.N_H == 21, f"V{MK.N_V} × H{MK.N_H}")
    size = 2.0 * MK.GRID_REF_Z * np.tan(np.radians(MK.FOV_DEG) / 2.0)
    check("[21] 기준거리에서 격자 한 변이 규격대로",
          abs(size - MK.GRID_SIZE_M) < 1e-6,
          f"{MK.GRID_REF_Z:.2f}m 에서 {size:.3f}m × {size:.3f}m "
          f"(칸 {size / (MK.N_V - 1) * 1000:.0f}mm)")
    check("[21] 카메라 화각이 레이저 부채꼴보다 넓다 (그래야 격자가 안에 앉는다)",
          MK.CAM_FOV_DEG > MK.FOV_DEG + 5.0,
          f"카메라 {MK.CAM_FOV_DEG:.1f}° > 레이저 {MK.FOV_DEG:.2f}°")
    check("[21] 반듯한 구성 — 굴림·수렴각 0",
          MK.ROLL_DEG == 0.0 and MK.TILT_DEG == 0.0,
          f"굴림 {MK.ROLL_DEG}° / 수렴각 {MK.TILT_DEG}°")

    def _cp(roll):
        return {"f_px": MK.F_PX, "cx_px": MK.CX, "cy_px": MK.CY,
                "b_m": MK.BASE_M, "n_v": MK.N_V, "n_h": MK.N_H,
                "fov_h_deg": MK.FOV_DEG, "fov_v_deg": MK.FOV_DEG,
                "laser_tilt_deg": MK.TILT_DEG, "laser_roll_deg": roll,
                "image_w": W, "image_h": H, "resolution": [W, H],
                "standoff_z": 1.2}

    # ── 거리 복원: 부재가 선을 가려도 흔들리지 않아야 한다 ──
    # 읽은 선을 정렬해 순서대로 짝지으면 가려진 선 하나에 짝이 통째로
    # 밀려 Z 가 선 간격 하나만큼 틀린다. 합의 방식이라야 한다.
    cp = _cp(MK.ROLL_DEG)
    la = RP._line_angles(cp)
    n_read = len(RP.read_grid_from_image(on)[0])
    pose = RP.estimate_grid_pose(on, cp, la)
    z_true = float(MK.WALL_P0[2])
    check("[21] 부재가 선을 가려도 거리를 정확히 푼다 (합의 방식)",
          bool(pose.get("ok")) and abs(pose["z_est_m"] - z_true) < 0.02,
          f"Z={pose.get('z_est_m')} m (벽 참값 {z_true:.2f} m) — "
          f"읽은 선 {n_read}/{MK.N_V}개, 맞은 선 {pose.get('n_matched')}개")
    check("[21] 실제로 선이 가려져 개수가 모자란 상황이다 (이 검증이 뜻을 가지려면)",
          n_read < MK.N_V, f"읽은 선 {n_read} < 발사한 선 {MK.N_V}")

    # ── 굴린 구성: 축 투영만으로는 격자를 못 읽는다 ──
    on_r, _, _, _, pl_r = MK.render(roll_deg=20.0)
    cp_r = _cp(20.0)
    la_r = RP._line_angles(cp_r)
    n_plain = len(RP.read_grid_from_image(on_r)[0])
    n_roll = len(RP.read_grid_from_image(
        on_r, roll_rad=np.radians(20.0), cx=MK.CX, cy=MK.CY)[0])
    check("[21] 굴린 격자는 굴림 보정을 해야 읽힌다",
          n_plain < 5 <= n_roll,
          f"굴림 미보정 {n_plain}개 → 보정 {n_roll}개")
    pose_r = RP.estimate_grid_pose(on_r, cp_r, la_r)
    check("[21] 굴린 격자에서도 거리를 푼다",
          bool(pose_r.get("ok")) and abs(pose_r["z_est_m"] - z_true) < 0.06,
          f"Z={pose_r.get('z_est_m')} m (참값 {z_true:.2f} m)")

    # ── 추적 밴드가 배경 **앞** 의 부재를 덮어야 한다 ──
    band = RP._depth_band(cp, la, pose["z_est_m"])
    front = min(p["xz"][1] - p["r"] for p in MK.POSTS)
    check("[21] 깊이 구간이 배경 앞의 부재까지 덮는다",
          band[0] < front and band[1] > z_true,
          f"구간 {band[0]:.2f}~{band[1]:.2f} m, 동바리 앞면 {front:.3f} m, "
          f"벽 {z_true:.2f} m")
    cp["standoff_z"], cp["z_range"] = pose["z_est_m"], band

    buf = _io.StringIO()
    with contextlib.redirect_stdout(buf):
        det = DET.detect(on, {}, la, cp, multi_surface=True)
    cam = np.array([MK.BASE_M, 0.0, 0.0])

    def _hit(uv):
        D = np.stack([(uv[:, 0] - MK.CX) / MK.F_PX,
                      (uv[:, 1] - MK.CY) / MK.F_PX, np.ones(len(uv))], axis=1)
        D /= np.linalg.norm(D, axis=1, keepdims=True)
        return MK.trace(cam, D)

    n_post = 0
    for uv in det.values():
        uv = np.asarray(uv, float)
        if len(uv):
            _, _, s2 = _hit(uv)
            n_post += int(((s2 == 2) | (s2 == 3)).sum())
    # 매끄러움 컷이 부재를 지우면 여기가 100점대로 떨어진다
    check("[21] 배경 앞 부재의 화소가 살아남는다 (매끄러움 컷에 지워지지 않음)",
          n_post > 900, f"동바리 검출 점 {n_post}개")

    lx, luv, tri = PIPE.triangulate_lines(det, la, cp)
    err = []
    for lid in lx:
        uv = np.asarray(luv[lid], float)
        Pt, _, _ = _hit(uv)
        m = np.isfinite(Pt[:, 2])
        err.append((np.asarray(lx[lid], float)[m, 2] - Pt[m, 2]) * 1000.0)
    e = np.concatenate(err)
    bias = float(np.median(e))
    mad = 1.4826 * float(np.median(np.abs(e - bias)))
    # 화소 중심 규약이 0.5px 어긋나면 여기가 수십 mm 로 뛴다
    check("[21] 깊이에 계통 오차가 없다 (화소 중심 규약 일치)",
          abs(bias) < 3.0, f"치우침 {bias:+.2f}mm / 산포(MAD) {mad:.2f}mm")

    # ── 코랩 진입점 ──
    # 코랩은 폴더를 통째로 못 올린다. zip 한 겹과 낱개 업로드 두 경우 모두
    # 촬영 폴더를 찾아내야 한다.
    HW = _load("hardware")
    import tempfile
    import zipfile
    import shutil
    src = os.path.join(ROOT, "samples", "example_capture")
    if os.path.isdir(src):
        td = tempfile.mkdtemp()
        try:
            zp = os.path.join(td, "촬영.zip")
            with zipfile.ZipFile(zp, "w") as zf:
                for f in os.listdir(src):
                    zf.write(os.path.join(src, f), f"내촬영_2026/{f}")
                zf.writestr("__MACOSX/._laser_on.png", b"junk")   # 맥 압축 부산물
            ex = os.path.join(td, "ex")
            with zipfile.ZipFile(zp) as zf:
                zf.extractall(ex)
            root = HW._find_capture_dir(ex)
            check("[21] zip 한 겹 안의 촬영 폴더를 찾는다",
                  os.path.basename(root) == "내촬영_2026"
                  and HW.check_capture(root, verbose=False)["ok"],
                  f"찾은 폴더 {os.path.basename(root)}")
            flat = os.path.join(td, "flat")
            os.makedirs(flat)
            for f in ("laser_on.png", "camera_params.json"):
                shutil.copy(os.path.join(src, f), flat)
            check("[21] 낱개로 올린 파일도 한 촬영 폴더로 본다",
                  HW._find_capture_dir(flat) == flat
                  and HW.check_capture(flat, verbose=False)["ok"])
        finally:
            shutil.rmtree(td, ignore_errors=True)
    check("[21] 코랩이 아니면 무엇을 하라는지 알려 준다",
          _colab_hint(HW), "폴더를 직접 지정하라는 안내")

    # ── 폴더가 실제로 규약을 통과하고 참값을 되찾는가 ──
    HW = _load("hardware")
    root = os.path.join(ROOT, "samples")
    for name in ("example_capture", "minimal_capture", "rolled_capture"):
        d = os.path.join(root, name)
        if not os.path.isdir(d):
            continue
        r = HW.check_capture(d, verbose=False)
        check(f"[21] samples/{name} 가 규약 검사를 통과한다", bool(r["ok"]),
              f"막힘 {len(r['필수문제'])}건 / 경고 {len(r['경고'])}건")


def main():
    print("=" * 70)
    print("레이저 그리드 품질검측 — 회귀 검증")
    print("=" * 70)
    for t in (test_hardware_spec, test_eq1_triangulation, test_eq3_backward_compat,
              test_gravity_paths_agree, test_tls_plane_vs_legacy,
              test_axis_fit, test_region_pipeline,
              test_segmentation_robustness, test_boundary_rejection,
              test_eq7_laser_plane, test_h_lines_in_pipeline,
              test_pointcloud_export, test_rolled_grid_detection,
              test_run_pipeline_inputs, test_single_line_member,
              test_silhouette_recovery, test_line_gap_recovery,
              test_pointcloud_plot, test_member_span_reporting,
              test_cylinder_aux_surface, test_hardware_contract,
              test_sample_capture):
        t()
    print("\n" + "=" * 70)
    if _FAILS:
        print(f"실패 {len(_FAILS)}건:")
        for f in _FAILS:
            print(f"  - {f}")
        return 1
    print("전체 통과")
    return 0


if __name__ == "__main__":
    sys.exit(main())
