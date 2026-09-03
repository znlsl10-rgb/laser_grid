#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_sample_capture.py — 하드웨어 업체용 **예제 입력 데이터셋** 생성기
========================================================================
규약서(docs/하드웨어_인터페이스_규약.md)가 말하는 파일을 실제로 하나씩
만들어 둔다. 업체가 "이런 모양으로 주면 되는구나" 를 눈으로 확인하고,
자기 장비 출력과 나란히 놓고 대조할 수 있어야 한다.

왜 그림판으로 그리지 않고 렌더하는가
-----------------------------------
예제가 **물리적으로 앞뒤가 맞아야** 쓸모가 있다. 아무 격자나 그려 놓으면
검사기는 통과해도 검측 파이프라인을 태웠을 때 엉뚱한 값이 나오고, 그러면
업체가 "우리 데이터가 문제인가, 코드가 문제인가" 를 가릴 수 없다.

그래서 장면을 세우고 광선을 쏴서 만든다.

  · 카메라 화소마다 광선을 쏘아 벽·바닥·동바리 중 **가장 가까운** 교점
  · 그 점이 레이저 평면에서 얼마나 떨어져 있는지로 밝기를 준다
    (각거리 기준 가우시안 → 안티에일리어싱이 저절로 들어간다)
  · 조사기 원점에서 그 점까지 **가리는 것이 있으면 어둡게** (그림자)
  · 센서 잡음을 얹는다 — 이진(0/255)이 아닌, 실제로 쓸 만한 예제

그래서 이 예제로 파이프라인을 돌리면 깊이·각도가 참값 근처로 나온다.
생성 직후 스스로 그것을 확인한다(--verify).

좌표계 (규약서 1절과 같음)
-------------------------
    원점  = 조사기(레이저) 출사점
    카메라 = (b, 0, 0)          ← +X 쪽으로 기선 b 만큼
    X 오른쪽 +, Y 아래 +, Z 전방 +
    레이저 평면은 모두 원점을 지난다:  n·P = 0
    화소 투영:  u = f·(X−b)/Z + c_x,   v = f·Y/Z + c_y

실행
----
    python3 make_sample_capture.py                 # samples/ 아래에 생성
    python3 make_sample_capture.py --out ./보낼폴더
    python3 make_sample_capture.py --verify        # 만든 뒤 파이프라인까지 검증
========================================================================
"""
import argparse
import json
import os as _os
import sys as _sys
import importlib.util as _ilu

import numpy as np


def _load(name):
    spec = _ilu.spec_from_file_location(
        name, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                            f"{name}.py"))
    m = _ilu.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


EQ7 = _load("eq7_laser_plane")

# ── 예제 장비 사양 ──
# 실제 장비(2448×2048)보다 작게 잡는다. 형식을 보여 주는 것이 목적이고,
# 저장소에 넣을 파일이라 가볍게 두는 편이 낫다. 배율 혼동이 없도록
# sensor_W/H 를 이미지 크기와 같게 둔다.
W, H = 1224, 1024
F_PX = 796.5                     # 2448×2048·1593px 장비를 반으로 줄인 값
CX, CY = (W - 1) / 2.0, (H - 1) / 2.0   # 화소 중심 규약에서의 화면 중심
BASE_M = 0.150
N_V, N_H = 15, 15
FOV_DEG = 60.0
ROLL_DEG = 20.0                  # 굴림 — 가로선도 깊이를 주는 권장 구성
TILT_DEG = 0.0
WAVELENGTH_NM = 520

# ── 예제 장면 ──
# 벽 하나, 바닥 하나, 동바리 두 본. 한 본은 연직, 한 본은 1.2° 기울여
# "합격" 과 "재서 값이 나오는" 두 경우가 다 보이게 한다.
WALL_P0 = np.array([0.0, 0.0, 2.50])
WALL_N = np.array([0.06, 0.0, -1.0])
FLOOR_P0 = np.array([0.0, 1.15, 0.0])
FLOOR_N = np.array([0.0, -1.0, 0.05])
POSTS = [
    {"name": "동바리1(연직)", "xz": (-0.36, 1.62), "r": 0.0243,
     "tilt_deg": 0.0, "azim_deg": 0.0},
    {"name": "동바리2(1.2° 기울임)", "xz": (0.30, 1.78), "r": 0.0243,
     "tilt_deg": 1.2, "azim_deg": 35.0},
]
GRAVITY = np.array([0.0, 1.0, 0.0])      # 장비를 똑바로 세워 찍은 예제


# ============ 장면 기하 ============
def _unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def _post_axis(p):
    th = np.radians(p["tilt_deg"])
    az = np.radians(p["azim_deg"])
    return _unit([np.sin(th) * np.cos(az), np.cos(th), np.sin(th) * np.sin(az)])


def _hit_plane(O, D, p0, n):
    """광선 O+tD 와 평면의 교점 t. 뒤쪽·평행이면 inf."""
    n = _unit(n)
    den = D @ n
    with np.errstate(divide="ignore", invalid="ignore"):
        t = ((p0 - O) @ n) / den
    return np.where((np.abs(den) > 1e-9) & (t > 1e-4), t, np.inf)


def _hit_cylinder(O, D, c0, d, r):
    """광선과 무한 원통의 앞면 교점 t. 안 맞으면 inf."""
    d = _unit(d)
    Oo = O - c0
    Op = Oo - (Oo @ d) * d
    Dp = D - np.outer(D @ d, d)
    A = np.einsum("ij,ij->i", Dp, Dp)
    B = Dp @ Op
    C = float(Op @ Op) - r * r
    disc = B * B - A * C
    with np.errstate(divide="ignore", invalid="ignore"):
        sq = np.sqrt(np.maximum(disc, 0.0))
        t1 = (-B - sq) / A
        t2 = (-B + sq) / A
    t = np.where(t1 > 1e-4, t1, t2)
    return np.where((disc > 0) & (A > 1e-12) & (t > 1e-4), t, np.inf)


def trace(O, D):
    """
    광선들을 장면에 쏜다.

    Returns
    -------
    P   : (N,3) 교점 (안 맞으면 nan)
    Nrm : (N,3) 그 자리 법선
    sid : (N,)  0=벽 1=바닥 2,3=동바리
    """
    hits = [(_hit_plane(O, D, WALL_P0, WALL_N), "wall"),
            (_hit_plane(O, D, FLOOR_P0, FLOOR_N), "floor")]
    for p in POSTS:
        c0 = np.array([p["xz"][0], 0.0, p["xz"][1]])
        hits.append((_hit_cylinder(O, D, c0, _post_axis(p), p["r"]), "post"))
    T = np.stack([h[0] for h in hits], axis=1)
    sid = np.argmin(T, axis=1)
    t = T[np.arange(len(T)), sid]
    ok = np.isfinite(t)
    P = np.where(ok[:, None], O + t[:, None] * D, np.nan)

    Nrm = np.zeros_like(P)
    Nrm[sid == 0] = _unit(WALL_N)
    Nrm[sid == 1] = _unit(FLOOR_N)
    for k, p in enumerate(POSTS):
        m = sid == (2 + k)
        if not m.any():
            continue
        c0 = np.array([p["xz"][0], 0.0, p["xz"][1]])
        d = _post_axis(p)
        q = P[m] - c0
        rad = q - np.outer(q @ d, d)
        Nrm[m] = rad / np.maximum(np.linalg.norm(rad, axis=1, keepdims=True),
                                  1e-9)
    sid = np.where(ok, sid, -1)
    return P, Nrm, sid


def _shadowed(P):
    """조사기 원점에서 P 로 가는 길이 동바리에 막히는가."""
    O = np.zeros(3)
    L = np.linalg.norm(P, axis=1)
    D = P / np.maximum(L[:, None], 1e-9)
    blocked = np.zeros(len(P), bool)
    for p in POSTS:
        c0 = np.array([p["xz"][0], 0.0, p["xz"][1]])
        t = _hit_cylinder(O, D, c0, _post_axis(p), p["r"])
        blocked |= np.isfinite(t) & (t < L - 2e-3)
    return blocked


# ============ 렌더 ============
def render(seed=7, noise=2.0, ambient=True):
    """예제 촬영 두 장(ON/OFF)과 정답값을 만든다."""
    rng = np.random.default_rng(seed)
    cam = np.array([BASE_M, 0.0, 0.0])

    # 화소 좌표 규약: 배열 인덱스 j 가 곧 좌표 j (화소 중심). 파이프라인의
    # 선검출(A_선검출.full_u = arange(W))과 같은 규약이다. 여기서 +0.5 를
    # 두면 예제가 코드와 0.5px 어긋나 벽에서 30mm 계통 오차로 나타난다.
    uu, vv = np.meshgrid(np.arange(W, dtype=np.float32),
                         np.arange(H, dtype=np.float32))
    uh = (uu.ravel() - CX) / F_PX
    vh = (vv.ravel() - CY) / F_PX
    D = np.stack([uh, vh, np.ones_like(uh)], axis=1)
    D /= np.linalg.norm(D, axis=1, keepdims=True)

    P, Nrm, sid = trace(cam, D)
    valid = sid >= 0

    # ── 장면 밝기 (레이저 OFF 프레임) ──
    # 램버트 음영 + 완만한 얼룩. 콘크리트 느낌이면 충분하다.
    scene = np.zeros(len(P), np.float32)
    if ambient:
        lit = _unit([0.25, -1.0, -0.45])
        lam = np.clip(-(Nrm @ lit), 0.0, 1.0)
        dist = np.linalg.norm(np.nan_to_num(P), axis=1)
        fall = np.clip(2.2 / np.maximum(dist, 0.3), 0.3, 1.6)
        base = np.where(sid == 1, 96.0, 108.0)
        base = np.where(sid >= 2, 132.0, base)
        scene = base * (0.45 + 0.55 * lam) * fall
        blob = rng.normal(0, 1, (H // 16 + 2, W // 16 + 2)).astype(np.float32)
        blob = np.repeat(np.repeat(blob, 16, 0), 16, 1)[:H, :W].ravel()
        scene = scene * (1.0 + 0.06 * blob)
    scene = np.where(valid, scene, 8.0)

    # ── 레이저 ──
    planes = EQ7.line_planes(_line_angles(), tilt_rad=np.radians(TILT_DEG),
                             roll_rad=np.radians(ROLL_DEG))
    Pn = np.nan_to_num(P)
    rng_len = np.maximum(np.linalg.norm(Pn, axis=1), 1e-6)
    # 선폭: 화면에서 약 2.0px → 각거리 표준편차 σ_a = 0.85/f
    sig_a = 0.85 / F_PX
    laser = np.zeros(len(P), np.float32)
    for pl in planes.values():
        n = np.asarray(pl["normal"], float)
        ang = np.abs(Pn @ n) / rng_len          # 평면까지의 각거리
        laser = np.maximum(laser, np.exp(-0.5 * (ang / sig_a) ** 2))
    laser *= valid
    laser[_shadowed(Pn)] = 0.0                  # 가림 그림자
    # 거리에 따른 감쇠 — 먼 벽이 조금 어둡다
    laser *= np.clip(2.0 / rng_len, 0.35, 1.0)

    def to_img(gray, green):
        a = np.zeros((len(P), 3), np.float32)
        a[:, 0] = gray * 0.92
        a[:, 1] = gray + green
        a[:, 2] = gray * 0.88
        a += rng.normal(0, noise, a.shape)
        return np.clip(a, 0, 255).astype(np.uint8).reshape(H, W, 3)

    off = to_img(scene, np.zeros_like(scene))
    on = to_img(scene, 175.0 * laser)
    return on, off, P.reshape(H, W, 3), sid.reshape(H, W), planes


def _line_angles():
    """예제 격자의 선별 발사각 — calibration 과 같은 균등 분할."""
    out = {}
    a = np.radians(FOV_DEG) / 2.0
    for i, ang in enumerate(np.linspace(-a, a, N_V)):
        out[f"V{i}"] = {"fixed": "alpha", "angle_rad": float(ang)}
    for j, ang in enumerate(np.linspace(-a, a, N_H)):
        out[f"H{j}"] = {"fixed": "beta", "angle_rad": float(ang)}
    return out


# ============ 정답값 ============
def make_truth(planes, n_per_line=400):
    """
    선마다 참 3D 점과 그 화소를 적는다.

    현장에는 없는 파일이다 — 시뮬레이션·검교정 환경에서만 만들 수 있다.
    있으면 파이프라인이 선검출 정확도와 깊이 오차를 대조해 조서에 싣는다.
    """
    cam = np.array([BASE_M, 0.0, 0.0])
    out = {}
    for lid, pl in planes.items():
        n = np.asarray(pl["normal"], float)
        # 평면 위를 훑도록 화소를 만든다 — 평면 ∩ 장면의 참 교점을 얻는다
        if abs(n[0]) > abs(n[1]):
            vs = np.linspace(0, H - 1, n_per_line)
            vh = (vs - CY) / F_PX
            uh = -(n[1] * vh + n[2]) / n[0]
            us = uh * F_PX + CX
        else:
            us = np.linspace(0, W - 1, n_per_line)
            uh = (us - CX) / F_PX
            vh = -(n[0] * uh + n[2]) / n[1]
            vs = vh * F_PX + CY
        keep = (us >= 0) & (us < W) & (vs >= 0) & (vs < H)
        us, vs = us[keep], vs[keep]
        if len(us) < 8:
            continue
        uh = (us - CX) / F_PX
        vh = (vs - CY) / F_PX
        D = np.stack([uh, vh, np.ones_like(uh)], axis=1)
        D /= np.linalg.norm(D, axis=1, keepdims=True)
        P, _N, sid = trace(cam, D)
        lit = (sid >= 0) & ~_shadowed(np.nan_to_num(P))
        if lit.sum() < 8:
            continue
        pts = [{"uv": [float(u), float(v)],
                "xyz_world": [float(x), float(y), float(z)]}
               for u, v, (x, y, z) in zip(us[lit], vs[lit], P[lit])]
        out[lid] = {"fixed": pl.get("fixed", "alpha"),
                    "angle_deg": float(np.degrees(
                        _line_angles()[lid]["angle_rad"])),
                    "points": pts}
    return out


# ============ 파일 쓰기 ============
def camera_params(with_normals=False, planes=None):
    d = {
        "camera": {"f_px": F_PX, "cx_px": CX, "cy_px": CY,
                   "sensor_W": W, "sensor_H": H},
        "baseline_m": BASE_M,
        "grid": {"n_vertical": N_V, "n_horizontal": N_H, "fov_deg": FOV_DEG},
        "laser": {"roll_deg": ROLL_DEG, "tilt_deg": TILT_DEG,
                  "wavelength_nm": WAVELENGTH_NM},
        # 정답 대조(truth.json)를 쓰려면 월드↔장비 자세가 필요하다.
        # 이 예제는 월드 = 조사기 좌표계라 항등변환이다.
        "rig_transform": {
            "laser_pos_world": [0.0, 0.0, 0.0],
            "camera_pos_world": [BASE_M, 0.0, 0.0],
            "camera_forward_world": [0.0, 0.0, 1.0],
        },
        "sensor_size": [W, H],
        "screenshot_size": [W, H],
        "case_name": "example_capture",
    }
    if with_normals and planes:
        d["lines"] = {lid: {"normal": [round(float(x), 6)
                                       for x in pl["normal"]]}
                      for lid, pl in planes.items()}
    return d


README = """레이저 그리드 검측 — 예제 입력 데이터셋
=========================================================
규약서: docs/하드웨어_인터페이스_규약.md
생성:   python3 make_sample_capture.py

이 폴더는 "장비가 촬영 한 번에 내보내야 하는 것" 의 예시입니다.
파일 이름·JSON 키·값의 단위를 그대로 따라 주시면 됩니다.

파일
----
laser_on.png        [필수] 레이저 ON 프레임. 무손실 PNG.
laser_off.png       [권장] 레이저 OFF 프레임. ON 과 같은 노출로 연속 촬영.
camera_params.json  [필수] 내부 파라미터·기선·격자 사양.
imu.json            [권장] 조사기 좌표계 중력 방향.
truth.json          [선택] 정답 화소·3D 좌표. 현장에는 없는 파일이며,
                           시뮬레이션·검교정 환경에서만 만들 수 있습니다.
                           있으면 선검출 정확도와 깊이 오차를 조서에 싣습니다.

바로 확인해 보기
----------------
    python3 hardware.py <이 폴더>       # 규약 충족 여부
    python3 colab_run.py --image <이 폴더>/laser_on.png

이 예제의 장면 (참값)
---------------------
{scene}

주의
----
· 해상도는 {W}×{H} 로 줄여 두었습니다. 형식을 보여 주는 것이 목적이며,
  실제 장비(예: 2448×2048)는 그 값을 camera_params.json 에 적으면 됩니다.
· 격자 굴림 roll_deg = {roll}° 로 두었습니다. 0° 면 가로선이 깊이를 주지
  못해 원형 부재(동바리)의 단면을 측정할 수 없습니다. 규약서 3절 참고.
· 선이 이진(0/255)이 아니라 안티에일리어싱된 상태입니다. 이진으로 찍히면
  선검출 정밀도가 0.289px 에서 막힙니다.
"""


def write_set(out_dir, minimal=False, with_normals=False, verbose=True):
    from PIL import Image
    _os.makedirs(out_dir, exist_ok=True)
    on, off, P, sid, planes = render()

    Image.fromarray(on).save(_os.path.join(out_dir, "laser_on.png"))
    files = ["laser_on.png"]
    if not minimal:
        Image.fromarray(off).save(_os.path.join(out_dir, "laser_off.png"))
        files.append("laser_off.png")

    cp = camera_params(with_normals=with_normals, planes=planes)
    if minimal:
        cp.pop("rig_transform", None)
    json.dump(cp, open(_os.path.join(out_dir, "camera_params.json"), "w",
                       encoding="utf-8"), ensure_ascii=False, indent=2)
    files.append("camera_params.json")

    if not minimal:
        json.dump({"gravity": [round(float(x), 6) for x in GRAVITY],
                   "note": "조사기 좌표계 기준 중력 방향. 장비를 똑바로 "
                           "세워 찍은 예제라 [0,1,0] 이다."},
                  open(_os.path.join(out_dir, "imu.json"), "w",
                       encoding="utf-8"), ensure_ascii=False, indent=2)
        files.append("imu.json")

        truth = make_truth(planes)
        json.dump(truth, open(_os.path.join(out_dir, "truth.json"), "w",
                              encoding="utf-8"))
        files.append("truth.json")

    # 참값은 "검측이 내놓아야 할 값" 으로 적는다. 벽의 좌우 비틀림처럼
    # 검사 항목이 아닌 값을 "기울어짐" 으로 적으면, 조서의 수직도 0.01°
    # 를 보고 틀렸다고 오해하게 된다.
    scene = "\n".join(
        ["  벽    : Z ≈ %.2f m.  수직도(연직 기준) %.2f° — 즉 **연직**."
         % (WALL_P0[2], np.degrees(np.arctan2(abs(WALL_N[1]),
                                              abs(WALL_N[2])))),
         "          (좌우로 %.1f° 비틀어 두었지만 이는 수직도 항목이 아님)"
         % np.degrees(np.arctan2(abs(WALL_N[0]), abs(WALL_N[2]))),
         "  바닥  : 장비보다 %.2f m 아래.  수평도 %.2f°"
         % (FLOOR_P0[1], np.degrees(np.arctan2(abs(FLOOR_N[2]),
                                               abs(FLOOR_N[1]))))]
        + ["  %s : Ø%.1f mm, 중심 (X %.2f, Z %.2f) m, 축 수직도 %.1f°"
           % (p["name"], 2000 * p["r"], p["xz"][0], p["xz"][1], p["tilt_deg"])
           for p in POSTS]
        + ["",
           "  이 폴더로 파이프라인을 돌리면 위 값이 그대로 나와야 합니다.",
           "  확인:  python3 make_sample_capture.py --verify"])
    txt = README.format(scene=scene, W=W, H=H, roll=ROLL_DEG)
    if minimal:
        # 최소 세트에는 없는 파일을 목록에 남기면 "왜 빠졌지" 를 부른다.
        keep = [ln for ln in txt.splitlines()
                if not ln.startswith(("laser_off.png", "imu.json",
                                      "truth.json", "                    "))]
        txt = "\n".join(keep).replace(
            "예제 입력 데이터셋",
            "예제 입력 데이터셋 (최소 구성 — 필수 파일만)")
        txt = txt.replace(
            "이 폴더는 \"장비가 촬영 한 번에 내보내야 하는 것\" 의 예시입니다.",
            "이 폴더는 **필수 두 가지만** 담은 최소 구성입니다. 이대로도\n"
            "검측은 돌아가지만, 권장 파일(laser_off.png, imu.json)이 없으면\n"
            "판정이 참고값으로 낮아집니다 — 전체 구성은 ../example_capture 참고.")
    open(_os.path.join(out_dir, "README.txt"), "w", encoding="utf-8").write(txt)
    files.append("README.txt")

    if verbose:
        print(f"[생성] {out_dir}")
        for f in files:
            n = _os.path.getsize(_os.path.join(out_dir, f))
            print(f"    {f:<22} {n/1024:8.1f} KB")
    return out_dir


# ============ 검증 ============
def scene_truth():
    """이 장면의 참값 — 파이프라인이 되찾아야 할 값."""
    out = [
        {"이름": "벽", "class": "wall", "Z_m": float(WALL_P0[2]),
         "각도_deg": np.degrees(np.arctan2(abs(WALL_N[1]),
                                          abs(WALL_N[2])))},
        {"이름": "바닥", "class": "floor",
         "Z_m": None,
         "각도_deg": np.degrees(np.arctan2(abs(FLOOR_N[2]),
                                          abs(FLOOR_N[1])))},
    ]
    for pp in POSTS:
        # 레이저가 맺히는 곳은 축이 아니라 **앞면** 이다. 보이는 앞면은
        # Z_축−R 부터 Z_축 까지 걸치므로 중앙값은 축보다 반지름쯤 앞이다.
        # 이걸 빼놓고 축 깊이와 견주면 늘 R 만큼 어긋난 것처럼 보인다.
        out.append({"이름": pp["name"], "class": "shoring",
                    "Z_m": float(pp["xz"][1]) - 0.75 * float(pp["r"]),
                    "Z_설명": f"축 {pp['xz'][1]:.3f}m − 앞면 보정(R={pp['r']*1000:.1f}mm)",
                    "각도_deg": float(pp["tilt_deg"])})
    return out


def _match(regions, want):
    """참값 하나에 가장 잘 맞는 검측 영역을 고른다 (분류 + 거리)."""
    cand = [g for g in regions if g.get("status") == "measured"
            and g.get("class") == want["class"]]
    if not cand:
        return None
    if want["Z_m"] is None:
        return max(cand, key=lambda g: len(np.asarray(g["point_xyz"])))
    return min(cand, key=lambda g: abs(
        float(np.median(np.asarray(g["point_xyz"], float)[:, 2])) - want["Z_m"]))


def verify(folder, tol_deg=0.30, tol_z_mm=60.0):
    """
    만든 예제가 규약을 통과하고, 파이프라인이 참값을 되찾는지 **판정** 한다.

    예제 데이터셋은 "형식이 맞다" 로는 부족하다. 하드웨어 업체가 이 폴더를
    기준으로 자기 출력을 맞출 것이므로, 이 입력을 넣으면 알려진 답이
    나온다는 것까지 확인되어야 한다. 그래서 통과/실패를 찍는다.
    """
    HW = _load("hardware")
    ok_all = True
    print()
    r = HW.check_capture(folder, verbose=False)
    print(f"  [규약 검사] {'통과' if r['ok'] else '막힘'}"
          f"  (막힘 {len(r['필수문제'])}건 / 경고 {len(r['경고'])}건)")
    for m in r["필수문제"]:
        print(f"      막힘: {m}")
        ok_all = False
    for m in r["경고"]:
        print(f"      경고: {m[:88]}")

    d = r["이미지"].get("laser_on", {})
    print(f"  [이미지] {d.get('크기')} {d.get('형식')} / "
          f"{d.get('레이저 채널')} / 이진={d.get('이진(안티에일리어싱 없음)')}"
          f" / 과포화={d.get('과포화 화소 비율')}")

    import io as _io
    import contextlib
    RP = _load("run_pipeline")
    buf = _io.StringIO()
    # 검증 산출물(조서·그림)은 입력 폴더 밖에 쓴다. 예제 폴더는 "장비가
    # 내보내야 하는 것" 만 담아야 업체가 보고 그대로 따라 만들 수 있다.
    import tempfile
    tmp = tempfile.mkdtemp(prefix="verify_")
    out = _os.path.join(tmp, "검증결과")
    with contextlib.redirect_stdout(buf):
        res = RP.run(image=_os.path.join(folder, "laser_on.png"),
                     params=_os.path.join(folder, "camera_params.json"),
                     imu=json.load(open(_os.path.join(folder, "imu.json"),
                                        encoding="utf-8")),
                     scene_image=_os.path.join(folder, "laser_off.png"),
                     out=out, verbose=True)
    log = buf.getvalue()
    for key in ("V선(", "H선(", "추적 깊이 구간", "삼각측량 점"):
        for ln in log.splitlines():
            if key in ln:
                print("  " + ln.strip()[:96])
                break

    # ── 화소 단위 대조 ──
    # 검출된 화소마다 그 시선을 장면에 그대로 쏘아 참 깊이를 구한다.
    # 정답 파일을 거치지 않으므로 이산화 오차가 섞이지 않는다.
    PIPE = _load("pipeline_region")
    regions = res["result"]["regions"]
    cam = np.array([BASE_M, 0.0, 0.0])
    err = []
    for gg in regions:
        uv = gg.get("point_uv")
        if uv is None or not len(uv):
            continue
        uv = np.asarray(uv, float)
        D = np.stack([(uv[:, 0] - CX) / F_PX, (uv[:, 1] - CY) / F_PX,
                      np.ones(len(uv))], axis=1)
        D /= np.linalg.norm(D, axis=1, keepdims=True)
        Pt, _, sid = trace(cam, D)
        z = np.asarray(gg["point_xyz"], float)[:, 2]
        m = np.isfinite(Pt[:, 2])
        err.append((z[m] - Pt[m, 2]) * 1000.0)
    if err:
        e = np.concatenate(err)
        mad = 1.4826 * float(np.median(np.abs(e - np.median(e))))
        print(f"  [깊이] 점 {len(e):,}  치우침 {np.median(e):+.2f}mm  "
              f"산포(MAD) {mad:.2f}mm  |오차|>50mm {np.mean(np.abs(e) > 50):.3%}")
        if abs(float(np.median(e))) > 5.0:
            print("      [실패] 깊이 치우침이 5mm 를 넘는다")
            ok_all = False

    # ── 검측 결과 대조 ──
    print("  [검측]  (참값 대비)")
    for want in scene_truth():
        g = _match(regions, want)
        if g is None:
            print(f"      [실패] {want['이름']:<14} 검측 영역을 못 찾음")
            ok_all = False
            continue
        th = float(g["theta_deg"])
        Z = float(np.median(np.asarray(g["point_xyz"], float)[:, 2]))
        dth = abs(th - want["각도_deg"])
        line = (f"      {want['이름']:<14} θ={th:7.4f}° "
                f"(참 {want['각도_deg']:.4f}°, 차 {dth:.4f}°)  Z={Z:.3f} m")
        bad = dth > tol_deg
        if want["Z_m"] is not None:
            dz = abs(Z - want["Z_m"]) * 1000.0
            line += f" (참 {want['Z_m']:.3f} m, 차 {dz:.0f} mm)"
            if want.get("Z_설명"):
                line += f"  [{want['Z_설명']}]"
            bad = bad or dz > tol_z_mm
        print(("      [실패]" + line[12:]) if bad else line)
        ok_all = ok_all and not bad

    import shutil
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"\n  결과: {'예제 데이터셋 검증 통과' if ok_all else '검증 실패'}")
    return ok_all


def main(argv=None):
    ap = argparse.ArgumentParser(description="예제 입력 데이터셋 생성")
    ap.add_argument("--out", default=None, help="생성 위치 (기본 samples/)")
    ap.add_argument("--verify", action="store_true",
                    help="만든 뒤 규약 검사 + 파이프라인 검증까지")
    a = ap.parse_args(argv)

    root = a.out or _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                  "samples")
    full = write_set(_os.path.join(root, "example_capture"))
    print()
    mini = write_set(_os.path.join(root, "minimal_capture"), minimal=True)
    ok = True
    if a.verify:
        ok = verify(full)
    print()
    print(f"규약 검사:  python3 hardware.py {full}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
