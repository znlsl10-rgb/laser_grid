#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
synth_scene.py — Isaac Sim 없이 도는 해석적 다중면 합성 씬
========================================================================
벽 + 바닥 + 동바리가 **한 프레임에 함께** 들어오는 장면을 수식으로 만든다.
Isaac Sim 이 없는 환경(CI, 개발 노트북)에서도 영역별 검측 파이프라인
전체를 정답과 대조해 검증하기 위한 것이다.

Isaac 씬과의 역할 분담
  · synth_scene  : 검측식·영역분할·융합 로직의 정답 대조 (빠름, 결정적)
  · Isaac 씬     : 렌더링·재질·조명·선검출까지 포함한 현장 리얼리즘 검증

【생성물】
  lines_pixels   {lid: [(u,v)…]}   V선 21개 × 250샘플 (픽셀 노이즈 포함)
  lines_xyz      {lid: [(X,Y,Z)…]} eq1 로 역산한 3D 점
  label_map      (H,W) int         정답 세그멘테이션 마스크
  id_to_semantic {id: "class:…"}   Isaac 어노테이터와 같은 형식
  g_hat          (3,)              조사기 좌표계 중력
  gt             정답값 (벽 수직도, 바닥 수평도, 동바리 수직도, 요철 깊이)

【좌표계】 조사기 좌표계 — X 우, Y 하, Z 전방 (eq1 규약)
  장비를 아래로 θ_d 만큼 숙여 벽과 바닥을 함께 담는다.
  이때 중력은 ĝ = (0, cos θ_d, sin θ_d) 이다.
    θ_d = 0°  → ĝ=(0,1,0)  정면 (기존 벽 스테이션)
    θ_d = 90° → ĝ=(0,0,1)  수직 하방 (기존 바닥 스테이션)
  즉 기존 두 스테이션은 이 연속 스펙트럼의 양 끝점이다.

V선만 생성하는 이유
  삼각측량(eq1)은 기선이 X축 방향이라 점의 발사각 α 를 알아야 한다.
  V선은 α 가 선마다 고정이므로 선을 따라 조밀 샘플링해도 모두 α 를
  안다. H선은 α 가 샘플마다 달라 단독으로는 깊이를 풀 수 없다
  (실장비도 V×H 교점에서만 α 를 회복한다).
========================================================================
"""
import numpy as np
import importlib.util as _ilu, os as _os


def _load(name):
    spec = _ilu.spec_from_file_location(
        name, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), f"{name}.py"))
    m = _ilu.module_from_spec(spec); spec.loader.exec_module(m)
    return m


_EQ7 = _load("eq7_laser_plane")

# ── 정답값 (Isaac 씬의 HORIZ=0.3, VBACK=0.5 규약을 그대로 따름) ──
GT_WALL_TILT_DEG = 0.5      # 벽이 연직에서 벗어난 각
GT_FLOOR_TILT_DEG = 0.3     # 바닥이 수평에서 벗어난 각
GT_SHORING_TILT_DEG = 1.2   # 동바리가 연직에서 벗어난 각
GT_BUMP_MM = 6.0            # 벽면 융기 높이
GT_BUMP_SIGMA_M = 0.05

# 직선자 처짐의 참값 — 융기 "높이" 와 다른 값이다.
#
# 오래 잘못 비교하던 부분이라 근거를 남긴다. 6mm 융기가 있는 평면에
# 직선자를 대면 자는 융기 꼭대기와 프로파일 끝점에 걸쳐진다. 그때
# 재는 틈은 융기 높이가 아니라
#
#     gap(d) = A·(1 − d/D) − A·exp(−d²/2σ²)
#
# 의 최대값이다 (A=6mm, σ=50mm, D = 자가 걸리는 반폭). 즉 참값은 표면만이
# 아니라 **얼마나 넓게 쟀는지**에 달려 있다. 화각이 다르면 벽에서 측정되는
# 구간이 달라지므로 참값도 프로파일마다 다르다.
#
#   legacy   화각 60.82° → 프로파일 1.41m
#   pdf      화각 42.61° → 프로파일 1.18m
#   improved 화각 42.61° → 프로파일 1.15m
#
# 아래 값은 각 프로파일에서 잡음 0 · V선 150~250개로 돌린 극한값이다.
# 알고리즘이 수렴하는 곳이 곧 그 기하의 참값이며, improved 의 4.0mm 는
# 해석식(D=430mm)과도 일치한다. 즉 계통 편향은 없다.
#
# 이전 문서가 "평활도를 정답의 60~70% 로 과소보고" 라고 적은 것은
# 6.0mm(융기 높이)와 비교한 탓이며, 실제로는 참값을 맞히고 있었다.
#   diagonal 화각 42.61° → 프로파일 1.15m (improved 와 같은 하드웨어)
#
# diagonal 은 improved 와 하드웨어가 같은데 참값이 4.01 → 4.07 로 달라진다.
# 격자를 45° 굴리면 벽에서 점이 놓이는 자리가 달라져 직선자가 걸치는
# 반폭 D 가 조금 넓어지기 때문이다. 위 gap(d) 식이 D 에 의존하므로
# 참값도 따라 움직인다 — 알고리즘 오차가 아니라 기하가 바뀐 것이다.
GT_STRAIGHTEDGE_MM = {"legacy": 2.60, "pdf": 4.54, "improved": 4.01,
                      "diagonal": 4.07}


def straightedge_truth_mm(profile=None):
    """활성 사양 프로파일에서의 직선자 처짐 참값 [mm]"""
    return GT_STRAIGHTEDGE_MM[_CALIB.ACTIVE_PROFILE if profile is None else profile]

# 장비를 아래로 숙인 각. 실제 렌즈(12mm)의 VFOV 는 32.81° 로 좁아,
# 22° 로는 바닥이 화면에 들어오지 않는다(실측: 바닥 점 0개).
# 40° 에서 벽·바닥·동바리가 모두 잡힌다.
# 34° 에서 벽 2372 · 바닥 345 · 동바리 263 점으로 균형이 맞는다.
# 더 눕히면 바닥 점은 늘지만 벽 입사각이 커져 평활도 분해능이 나빠진다.
DEVICE_PITCH_DEG = 34.0
WALL_DIST_M = 1.20          # 벽까지 거리
FLOOR_DROP_M = 0.90         # 장비에서 바닥까지 (중력 방향)
SHORING_RADIUS_M = 0.0243   # Ø48.6mm 파이프서포트
SHORING_LENGTH_M = 2.40
# 동바리 설치 위치. 12mm 렌즈의 좁은 VFOV(32.81°)에서는 배치에 따라
# 부재가 짧게만 담겨 축 방향이 결정되지 않는다. X=0.20, Z=1.15 에서
# 약 580mm(세장비 24)가 보여 측정이 성립한다.
SHORING_X_M = 0.20
SHORING_Z_M = 1.15

# 철근 — 동바리보다 훨씬 가늘다. D25 이형철근 공칭 Ø25.4mm.
# 격자 피치가 이 지름의 두 배를 넘으면 부재 하나에 V선이 한 줄도 안
# 걸릴 수 있다(legacy 70.4mm / pdf 49.3mm / improved 24.0mm). 즉 이
# 부재는 사양 프로파일에 따라 아예 측정이 안 될 수도 있고, 그 사실을
# 보여주는 것이 이 부재를 씬에 두는 이유다.
GT_REBAR_TILT_DEG = 0.8
REBAR_RADIUS_M = 0.0127
REBAR_LENGTH_M = 2.00
REBAR_X_M = -0.26
REBAR_Z_M = 1.02
# 기본은 끈다. 켜면 격자점 분포가 달라져 기존 회귀 기준값이 흔들린다.
WITH_REBAR = False

# 캘리브레이션은 calibration.py 단일 출처
_CALIB = _load("calibration")
CAMERA_PARAMS = dict(_CALIB.CAMERA_PARAMS)
GRID = {"n_vertical": _CALIB.N_VERTICAL,
        "n_horizontal": _CALIB.N_HORIZONTAL,
        "fov_deg": _CALIB.GRID_PARAMS["fov_deg"],
        "samples_per_line": _CALIB.GRID_PARAMS["samples_per_line"]}
SIGMA_U_PX = _CALIB.SIGMA_U_PX

CLASS_IDS = {0: "class:BACKGROUND", 1: "class:wall",
             2: "class:floor", 3: "class:shoring", 4: "class:rebar"}


def _rx(deg):
    r = np.radians(deg)
    return np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)],
                     [0, np.sin(r), np.cos(r)]])


def _unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


# =====================================================================
# 씬 기하
# =====================================================================
def _build_geometry(with_rebar=None):
    """조사기 좌표계에서 벽·바닥·동바리(·철근)를 정의한다."""
    td = np.radians(DEVICE_PITCH_DEG)
    g = np.array([0.0, np.cos(td), np.sin(td)])          # 중력 (아래)

    # 벽: 이상적 법선은 중력에 수직이고 카메라를 향한다
    n_wall_ideal = _unit([0.0, np.sin(td), -np.cos(td)])
    n_wall = _unit(_rx(GT_WALL_TILT_DEG) @ n_wall_ideal)
    p_wall = np.array([0.0, 0.0, WALL_DIST_M])            # 벽 위의 한 점

    # 바닥: 이상적 법선은 중력과 평행
    n_floor = _unit(_rx(GT_FLOOR_TILT_DEG) @ g)
    p_floor = FLOOR_DROP_M * g

    # 동바리: 축은 중력 방향에서 GT_SHORING_TILT_DEG 만큼 기움
    axis = _unit(_rx(GT_SHORING_TILT_DEG) @ g)
    # 바닥면 위, 벽보다 앞쪽(Z=0.95) 오른쪽(X=0.30) 지점에 세운다.
    # 바닥 평면식 n·(p - p_floor)=0 에서 Y를 풀어 정확히 바닥에 놓는다.
    bx, bz = SHORING_X_M, SHORING_Z_M
    by = (n_floor @ p_floor - n_floor[0] * bx - n_floor[2] * bz) / n_floor[1]
    base = np.array([bx, by, bz])
    geo = {"g": g,
           "wall": {"n": n_wall, "p": p_wall, "half": (0.85, 0.85)},
           "floor": {"n": n_floor, "p": p_floor, "half": (0.85, 1.30)},
           "shoring": {"axis": axis, "base": base,
                       "r": SHORING_RADIUS_M, "len": SHORING_LENGTH_M}}

    if WITH_REBAR if with_rebar is None else with_rebar:
        r_axis = _unit(_rx(GT_REBAR_TILT_DEG) @ g)
        rx, rz = REBAR_X_M, REBAR_Z_M
        ry = (n_floor @ p_floor - n_floor[0] * rx - n_floor[2] * rz) / n_floor[1]
        geo["rebar"] = {"axis": r_axis, "base": np.array([rx, ry, rz]),
                        "r": REBAR_RADIUS_M, "len": REBAR_LENGTH_M}
    return geo


def _plane_basis(n):
    seed = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1.0, 0])
    e1 = _unit(np.cross(n, seed)); e2 = _unit(np.cross(n, e1))
    return e1, e2


def _wall_bump(local_uv):
    """벽면 국소좌표에서의 융기 깊이 (m). 평활도 정답."""
    d2 = local_uv[..., 0] ** 2 + local_uv[..., 1] ** 2
    return (GT_BUMP_MM / 1000.0) * np.exp(-d2 / (2 * GT_BUMP_SIGMA_M ** 2))


def _cylinder_hit(dirs, origin, s, code, t_best, cid):
    """유한 원통과의 교차. 동바리·철근이 같은 식을 쓴다."""
    a, b0, r, L = s["axis"], s["base"], s["r"], s["len"]
    d_perp = dirs - np.outer(dirs @ a, a)
    o = np.asarray(origin, float) - b0
    o_perp = o - (o @ a) * a
    A = np.einsum("ij,ij->i", d_perp, d_perp)
    B = 2.0 * (d_perp @ o_perp)
    C = float(o_perp @ o_perp) - r * r
    disc = B * B - 4 * A * C
    ok = (disc > 0) & (A > 1e-12)
    if not ok.any():
        return
    sq = np.sqrt(disc[ok])
    t1 = (-B[ok] - sq) / (2 * A[ok])
    t2 = (-B[ok] + sq) / (2 * A[ok])
    tc = np.where(t1 > 1e-6, t1, t2)
    hit = np.asarray(origin, float) + dirs[ok] * tc[:, None]
    axial = (hit - b0) @ a
    valid = (tc > 1e-6) & (axial >= -L) & (axial <= 0.05)   # 바닥에서 위로
    idx = np.where(ok)[0][valid]
    tv = tc[valid]
    better = tv < t_best[idx]
    t_best[idx[better]] = tv[better]
    cid[idx[better]] = code


def _intersect(dirs, geo):
    """
    원점에서 나가는 광선 다발과 씬의 교차 (벡터화).

    Returns
    -------
    t   : (N,) 교차 거리, 미교차는 inf
    cid : (N,) 클래스 id (0=배경)
    """
    N = len(dirs)
    t_best = np.full(N, np.inf)
    cid = np.zeros(N, dtype=np.int32)

    for name, code in (("wall", 1), ("floor", 2)):
        s = geo[name]
        n, p, (h1, h2) = s["n"], s["p"], s["half"]
        denom = dirs @ n
        ok = np.abs(denom) > 1e-9
        t = np.full(N, np.inf)
        t[ok] = (p @ n) / denom[ok]
        ok &= (t > 1e-6) & np.isfinite(t)
        if not ok.any():
            continue
        hit = dirs[ok] * t[ok, None]
        e1, e2 = _plane_basis(n)
        rel = hit - p
        uu, vv = rel @ e1, rel @ e2
        inside = (np.abs(uu) <= h1) & (np.abs(vv) <= h2)

        if name == "wall":
            # 융기만큼 카메라 쪽으로 당겨진다 (법선 방향 -bump)
            t_adj = t[ok].copy()
            bump = _wall_bump(np.stack([uu, vv], -1))
            # 광선 방향과 법선의 사잇각을 고려해 거리 보정
            cosang = np.abs(denom[ok])
            t_adj -= bump / np.maximum(cosang, 1e-6)
            t[ok] = t_adj

        idx = np.where(ok)[0][inside]
        better = t[idx] < t_best[idx]
        sel = idx[better]
        t_best[sel] = t[sel]; cid[sel] = code

    # 동바리·철근 (유한 원통)
    _cylinder_hit(dirs, np.zeros(3), geo["shoring"], 3, t_best, cid)
    if "rebar" in geo:
        _cylinder_hit(dirs, np.zeros(3), geo["rebar"], 4, t_best, cid)
    return t_best, cid


# =====================================================================
# 씬 생성
# =====================================================================
def build_scene(seed=2026, sigma_u_px=SIGMA_U_PX, label_map_stride=2,
                with_rebar=None):
    """
    합성 씬 한 장을 만든다.

    label_map_stride : int
        정답 마스크를 1/stride 해상도로 만든 뒤 확대한다. 2448×2048 전
        화소 레이캐스트는 느리므로 기본 2로 둔다(경계 ±1px 오차는
        eq5 의 마스크 침식이 어차피 흡수한다).
    """
    rng = np.random.default_rng(seed)
    geo = _build_geometry(with_rebar)
    f = CAMERA_PARAMS["f_px"]; b = CAMERA_PARAMS["b_m"]
    cx = CAMERA_PARAMS["cx_px"]; cy = CAMERA_PARAMS["cy_px"]
    W, H = CAMERA_PARAMS["resolution"]

    # ── 레이저 선 투사 ──
    # 발사각은 calibration 단일 출처 (레이저 수렴각이 α 에 포함된다)
    roll_deg = float(_CALIB.GRID_PARAMS.get("laser_roll_deg", 0.0) or 0.0)
    _ang = _CALIB.make_line_angles(GRID["n_vertical"], GRID["n_horizontal"],
                                   GRID["fov_deg"],
                                   _CALIB.GRID_PARAMS["laser_tilt_deg"])
    alphas = np.array([_ang[f"V{i}"]["angle_rad"]
                       for i in range(GRID["n_vertical"])])
    fov = np.radians(GRID["fov_deg"])
    betas = np.linspace(-fov / 2, fov / 2, GRID["samples_per_line"])

    # 격자를 굴리지 않은 사양에서는 H선이 깊이를 못 주므로(eq7 이득 g=∞)
    # 예전처럼 V선만 쏜다. 굴린 사양에서는 두 계열 모두 깊이를 주므로
    # 둘 다 쏜다. 이 분기를 두는 이유는 성능이 아니라 **재현성** 이다 —
    # 굴리지 않은 프로파일의 광선 생성 경로를 건드리면 이미 검증해 둔
    # 합성 씬 정답값(GT_STRAIGHTEDGE_MM 등)이 미세하게 달라진다.
    emit = [(f"V{i}", "alpha", float(a)) for i, a in enumerate(alphas)]
    if roll_deg:
        emit += [(f"H{j}", "beta", float(_ang[f"H{j}"]["angle_rad"]))
                 for j in range(GRID["n_horizontal"])]

    lines_pixels, lines_xyz, point_class = {}, {}, {}
    line_angles = {}
    for lid, fixed, al in emit:
        n_plane = np.asarray(_ang[lid]["normal"], float)
        if roll_deg:
            # 평면 위에서 광선을 훑는다. 평면 안에서 +Z 에 가장 가까운
            # 방향을 중심축 a 로 잡고, 그와 직교하는 평면 내 방향 c 로
            # 부채꼴을 편다. 굴림·수렴각이 어떻게 들어와도 같은 코드가 받는다.
            a_ax = np.array([0.0, 0.0, 1.0]) - n_plane * float(n_plane[2])
            a_ax /= np.linalg.norm(a_ax)
            c_ax = np.cross(n_plane, a_ax)
            c_ax /= np.linalg.norm(c_ax)
            dirs = (np.cos(betas)[:, None] * a_ax
                    + np.sin(betas)[:, None] * c_ax)
        else:
            dirs = np.stack([np.full_like(betas, np.tan(al)), np.tan(betas),
                             np.ones_like(betas)], axis=1)
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        t, cid = _intersect(dirs, geo)
        hit = np.isfinite(t) & (cid > 0)
        if hit.sum() < 5:
            continue
        P = dirs[hit] * t[hit, None]

        # 카메라 투영 (카메라는 +X 로 b 만큼 떨어져 있다)
        u = f * (P[:, 0] - b) / P[:, 2] + cx + rng.normal(0, sigma_u_px, hit.sum())
        v = f * P[:, 1] / P[:, 2] + cy + rng.normal(0, sigma_u_px, hit.sum())
        inim = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        if inim.sum() < 5:
            continue
        u, v, cls_hit = u[inim], v[inim], cid[hit][inim]
        P = P[inim]

        # ── 카메라 가림(occlusion) 검사 ──
        # 레이저는 원점에서, 카메라는 +X 로 b 만큼 떨어진 곳에서 본다.
        # 기선 150mm 의 시차 때문에, 레이저가 맞춘 점이라도 카메라 시선에서
        # 다른 물체(동바리 등)에 가려지면 촬영되지 않는다. 이 검사를 빼면
        # 동바리 뒤 벽면 점이 동바리 마스크 안으로 투영되어 영역을 오염시킨다.
        cam_o = np.array([b, 0.0, 0.0])
        vec = P - cam_o
        dist = np.linalg.norm(vec, axis=1)
        t_first, _ = _intersect_from(vec / dist[:, None], cam_o, geo)
        visible = t_first >= dist - 2e-3          # 2mm 여유 (자기 자신 히트)
        if visible.sum() < 5:
            continue
        u, v, cls_hit = u[visible], v[visible], cls_hit[visible]

        # eq7 평면식으로 역산 (실제 파이프라인과 동일 경로).
        # 굴리지 않은 V선에서는 eq1 과 나노미터 아래까지 같은 값이다
        # (eq7 자체 검증에서 2000점 대조 확인).
        P3, keep = _EQ7.triangulate_plane(np.column_stack([u, v]), n_plane,
                                          f, b, cx, cy)
        if len(P3) < 5:
            continue
        lines_pixels[lid] = np.column_stack([u[keep], v[keep]]).tolist()
        lines_xyz[lid] = P3.tolist()
        point_class[lid] = np.asarray(cls_hit, dtype=np.int32)[keep]
        line_angles[lid] = {"fixed": fixed, "angle_rad": al,
                            "normal": n_plane.tolist(),
                            "depth_gain": _EQ7.depth_gain(n_plane)}

    # ── 정답 라벨맵 (카메라 시점 레이캐스트) ──
    st = int(label_map_stride)
    us = np.arange(0, W, st, dtype=np.float64)
    vs = np.arange(0, H, st, dtype=np.float64)
    UU, VV = np.meshgrid(us, vs)
    # 카메라 원점은 (b,0,0) 이므로 광선 원점을 옮겨 교차시킨다.
    d_cam = np.stack([(UU - cx) / f, (VV - cy) / f, np.ones_like(UU)], -1)
    d_cam /= np.linalg.norm(d_cam, axis=-1, keepdims=True)
    small = np.zeros(UU.shape, dtype=np.int32)
    cam_o = np.array([b, 0.0, 0.0])
    for r0 in range(0, small.shape[0], 128):
        blk = d_cam[r0:r0 + 128].reshape(-1, 3)
        _, cid = _intersect_from(blk, cam_o, geo)
        small[r0:r0 + 128] = cid.reshape(-1, small.shape[1])
    label_map = np.repeat(np.repeat(small, st, axis=0), st, axis=1)[:H, :W]

    # 렌더 이미지는 세그멘테이션 백엔드 인터페이스용 더미
    rgb_off = (label_map[..., None] * np.array([60, 40, 30], np.uint8)) \
        .astype(np.uint8)

    # 카메라 자세행렬 (inspection.py 규약: 열 = [right, down, forward]).
    # 장비를 아래로 θ_d 숙인 자세이며, eq3.gravity_in_laser_frame 없이
    # 이 행렬만으로도 같은 ĝ 가 나와야 한다(두 경로 교차검증).
    td = np.radians(DEVICE_PITCH_DEG)
    view_w = np.array([0.0, np.cos(td), -np.sin(td)])
    right_w = _unit(np.cross(view_w, np.array([0.0, 0.0, 1.0])))
    down_w = _unit(np.cross(view_w, right_w))
    R_world_cam = np.column_stack([right_w, down_w, view_w])

    return {"lines_pixels": lines_pixels, "lines_xyz": lines_xyz,
            "point_class": point_class,
            "line_angles": line_angles, "R_world_cam": R_world_cam,
            "label_map": label_map, "id_to_semantic": CLASS_IDS,
            "rgb_off": rgb_off, "camera_params": dict(CAMERA_PARAMS),
            "g_hat": geo["g"], "geometry": geo,
            "gt": {"wall_verticality_deg": GT_WALL_TILT_DEG,
                   "floor_horizontality_deg": GT_FLOOR_TILT_DEG,
                   "shoring_verticality_deg": GT_SHORING_TILT_DEG,
                   "rebar_verticality_deg": GT_REBAR_TILT_DEG,
                   "wall_bump_mm": GT_BUMP_MM,
                   "device_pitch_deg": DEVICE_PITCH_DEG,
                   "sigma_u_px": sigma_u_px}}


def _intersect_from(dirs, origin, geo):
    """원점이 0이 아닌 광선 다발 (카메라 시점 레이캐스트용)."""
    N = len(dirs)
    t_best = np.full(N, np.inf); cid = np.zeros(N, dtype=np.int32)

    for name, code in (("wall", 1), ("floor", 2)):
        s = geo[name]; n, p, (h1, h2) = s["n"], s["p"], s["half"]
        denom = dirs @ n
        ok = np.abs(denom) > 1e-9
        t = np.full(N, np.inf)
        t[ok] = ((p - origin) @ n) / denom[ok]
        ok &= (t > 1e-6) & np.isfinite(t)
        if not ok.any():
            continue
        hit = origin + dirs[ok] * t[ok, None]
        e1, e2 = _plane_basis(n)
        rel = hit - p
        uu, vv = rel @ e1, rel @ e2
        inside = (np.abs(uu) <= h1) & (np.abs(vv) <= h2)
        idx = np.where(ok)[0][inside]
        better = t[idx] < t_best[idx]
        t_best[idx[better]] = t[idx[better]]; cid[idx[better]] = code

    _cylinder_hit(dirs, origin, geo["shoring"], 3, t_best, cid)
    if "rebar" in geo:
        _cylinder_hit(dirs, origin, geo["rebar"], 4, t_best, cid)
    return t_best, cid


# =====================================================================
# 설명·채점
# =====================================================================
def describe(scene):
    gt = scene["gt"]
    g = scene["g_hat"]
    n_pts = sum(len(v) for v in scene["lines_xyz"].values())
    cnt = {}
    for arr in scene["point_class"].values():
        for c, k in zip(*np.unique(arr, return_counts=True)):
            cnt[int(c)] = cnt.get(int(c), 0) + int(k)
    name = {1: "벽", 2: "바닥", 3: "동바리"}
    dist = "  ".join(f"{name.get(c, c)} {k}" for c, k in sorted(cnt.items()))
    return (f"합성 씬: 장비 {gt['device_pitch_deg']}° 하향, "
            f"ĝ=({g[0]:.3f}, {g[1]:.3f}, {g[2]:.3f})\n"
            f"  정답  벽 수직도 {gt['wall_verticality_deg']}° / "
            f"바닥 수평도 {gt['floor_horizontality_deg']}° / "
            f"동바리 수직도 {gt['shoring_verticality_deg']}° / "
            f"벽 융기 {gt['wall_bump_mm']}mm\n"
            f"  격자점 {n_pts}개  ({dist})  픽셀노이즈 σ_u={gt['sigma_u_px']}px")


def score(scene, result):
    """검측 결과를 정답과 대조한다."""
    gt = scene["gt"]
    want = {"wall": ("wall_verticality_deg", "수직도"),
            "floor": ("floor_horizontality_deg", "수평도"),
            "shoring": ("shoring_verticality_deg", "축수직도")}
    lines = ["  정답 대조 (허용 ±0.5°)"]
    # 한 클래스가 여러 영역으로 나뉠 수 있으므로 점 수가 가장 많은 영역을 대표로 본다
    got = {}
    n_by_cls = {}
    for r in result["regions"]:
        if r["status"] != "measured":
            continue
        c = r["class"]
        n_by_cls[c] = n_by_cls.get(c, 0) + 1
        if c not in got or r["n_points"] > got[c]["n_points"]:
            got[c] = r
    all_ok = True
    for cls, (key, ko) in want.items():
        r = got.get(cls)
        if r is None:
            lines.append(f"    {cls:8s} {ko:6s} → 영역 없음  ✗")
            all_ok = False
            continue
        err = abs(r["theta_deg"] - gt[key])
        ok = err <= 0.5
        all_ok &= ok
        extra = (f"  (영역 {n_by_cls[cls]}개 중 최대)"
                 if n_by_cls.get(cls, 1) > 1 else "")
        lines.append(f"    {cls:8s} {ko:6s} 측정 {r['theta_deg']:7.4f}° "
                     f"정답 {gt[key]:5.2f}°  오차 {err:6.4f}°  "
                     f"{'✓' if ok else '✗'}{extra}")
    w = got.get("wall")
    f = (w or {}).get("flatness") or {}
    if f.get("applicable"):
        lines.append(
            f"    {'wall':8s} 평활도  자 처짐 {f.get('max_gap_mm', 0):.2f}mm "
            f"(상한 {f.get('upper_estimate_mm', 0):.2f}mm, 허용 "
            f"{f.get('tolerance_mm', 0)}mm) / eq4 요철깊이 "
            f"{f.get('defect_max_dev_mm', 0):.2f}mm  "
            f"정답 융기 {gt['wall_bump_mm']:.1f}mm → {f['judgement']}")
    lines.append(f"  → {'전체 통과' if all_ok else '실패 항목 있음'}")
    return "\n".join(lines)


if __name__ == "__main__":
    import time
    t0 = time.time()
    sc = build_scene()
    print(describe(sc))
    print(f"  라벨맵 {sc['label_map'].shape} 생성 {time.time()-t0:.1f}s")
    u, c = np.unique(sc["label_map"], return_counts=True)
    print(f"  라벨맵 화소 분포: " +
          "  ".join(f"{CLASS_IDS[int(i)].split(':')[-1]} {k}" for i, k in zip(u, c)))
