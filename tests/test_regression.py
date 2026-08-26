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
import numpy as np
import importlib.util as ilu

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)


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
    EXP = _load("experiment_segmentation")
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
        check(f"벽은 높이폭 {w['높이폭_m']}m > 깊이폭 {w['깊이폭_m']}m (서 있다)",
              w["높이폭_m"] > w["깊이폭_m"])
        check(f"바닥은 깊이폭 {fl['깊이폭_m']}m > 높이폭 {fl['높이폭_m']}m (누워 있다)",
              fl["깊이폭_m"] > fl["높이폭_m"])
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
              test_run_pipeline_inputs):
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
