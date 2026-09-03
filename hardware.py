#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
촬영 데이터 점검기 — 이 데이터를 검측에 써도 되는가
========================================================================
장비 제작·납품 업체에 **이 파일 하나만** 건네면 됩니다. 저장소를 받을
필요도, 다른 파일을 같이 둘 필요도 없습니다. 필요한 것은 파이썬과
numpy, Pillow 뿐입니다.

    python3 hardware.py <촬영폴더>              한 벌 점검
    python3 hardware.py <폴더1> <폴더2> ...     여러 벌 한꺼번에
    python3 hardware.py <촬영.zip>              zip 도 그대로
    python3 hardware.py --template              camera_params.json 빈 서식
    python3 hardware.py --demo                  맞는 예제를 만들어 보여 준다
    python3 hardware.py                         자체 검증

코랩에서는 이 파일을 올린 뒤 셀에서:

    import hardware; hardware.colab()

파일 여러 개를 한꺼번에 고르거나 zip 을 올리면 됩니다.

────────────────────────────────────────────────────────────────────────
이 점검기가 보는 것 — 세 단계
────────────────────────────────────────────────────────────────────────
1. 파일이 있는가        빠진 것이 있으면 그 자리에서 막는다.
2. 이미지가 쓸 만한가   저장 형식·노출·선폭·채널 분리.
3. **사양이 이미지와 맞는가**  ← 이 점검기의 핵심

3번이 없으면 점검기가 아니라 파일 목록일 뿐입니다. camera_params.json 에
적힌 초점거리·기선·격자 수·굴림각이 **실제로 그 이미지를 찍은 값인지**
확인합니다. 방법은 이렇습니다.

    격자를 이미지에서 직접 읽는다 (예측 없이)
    → 사양대로라면 선이 어디 있어야 하는지 계산한다
    → 두 벌이 겹치는 거리 Z 가 있는가

f 가 5% 틀리거나, 기선을 잘못 적었거나, 선 수·발산각이 데이터시트와
다르면 **겹치는 Z 가 없습니다.** 파일은 다 있는데 숫자가 안 맞는 상태를
여기서 잡습니다. 이걸 못 잡으면 검측은 그대로 돌아가고 결과만 조용히
틀립니다.

────────────────────────────────────────────────────────────────────────
촬영 한 벌에 있어야 하는 것
────────────────────────────────────────────────────────────────────────
필수
  laser_on.png        레이저 ON 프레임. 무손실(png/tiff).
                      jpg 는 압축 잡음이 선 중심을 흔들어 정밀도를 깎는다.
  camera_params.json  아래 값들.

강력 권장
  laser_off.png       레이저 OFF 프레임. ON 과 **같은 노출**로, 수십 µs
                      안에 연속 촬영. 두 장을 빼면 배경광이 지워져
                      햇빛 드는 현장에서도 선이 살아난다.
  imu.json            {"gravity": [x,y,z]} 또는 {"pitch_deg":, "roll_deg":}.
                      **조사기 좌표계** 기준. 없으면 장비가 똑바로 섰다고
                      가정하고, 판정을 참고값으로 낮춘다.

선택
  truth.json          정답 화소·좌표. 검교정 환경에서만 만들 수 있다.

────────────────────────────────────────────────────────────────────────
camera_params.json — 값마다 '어떻게 얻는가' 가 다르다
────────────────────────────────────────────────────────────────────────
    {
      "camera": {"f_px": 1593.0, "cx_px": 1224.0, "cy_px": 1024.0,
                 "sensor_W": 2448, "sensor_H": 2048},
      "baseline_m": 0.150,
      "grid": {"n_vertical": 21, "n_horizontal": 21, "fov_deg": 60.82},
      "laser": {"roll_deg": 20.0, "tilt_deg": 0.0, "wavelength_nm": 520}
    }

  f_px, cx_px, cy_px   체커보드 캘리브레이션. **추정 금지** —
                       f 가 1% 틀리면 깊이가 1% 틀린다(1.7m 에서 17mm).
  sensor_W/H           센서 화소 수. 이미지가 축소본이면 배율을 여기서 잡는다.
  baseline_m           카메라 광심 ↔ 조사기 출사점 거리. **조립 후 실측.**
                       1mm 틀리면 깊이가 0.7% 틀린다(150mm 기준).
  grid.n_vertical/     DOE 가 만드는 선 수. 데이터시트 값.
  n_horizontal
  grid.fov_deg         격자 전체 발산각. 데이터시트 값.
  laser.roll_deg       **광축 둘레로 DOE 를 얼마나 돌려 끼웠는가.**
                       0 이면 가로선의 레이저 평면이 카메라 광심을 지나
                       깊이를 전혀 못 준다(이득 g=∞). 원형 부재(동바리)의
                       단면을 재려면 20° 이상 필요하다 (g = 1/sin γ).
  laser.tilt_deg       조사기 광축이 카메라 광축과 이루는 수렴각.
  laser.wavelength_nm  참고값. 채널은 이미지에서 자동 판별한다.

발사각 대신 **평면 법선** 을 직접 재 오면 더 좋습니다. 평면 하나에
자유도 3 이고 굴림·수렴각·렌즈 왜곡이 그 안에 다 흡수됩니다.

    "lines": {"V0": {"normal": [nx, ny, nz]}, ..., "H0": {...}}

────────────────────────────────────────────────────────────────────────
촬영 조건
────────────────────────────────────────────────────────────────────────
  · 무손실 저장, 안티에일리어싱 켠 채로. 선이 이진(0/255)으로 찍히면
    선 중심이 0.5화소 격자에 갇혀 정밀도가 0.289px 아래로 못 간다.
    깊이 잡음은 σ_Z = σ_u·Z²/(f·b) 이므로 그대로 바닥이 된다.
  · 과포화 금지. 선 중심이 평평해지면 능선을 못 찾는다.
  · ON/OFF 두 프레임 사이 장비가 움직이면 차영상이 어긋난다.
  · 한 화면에 담기는 깊이 범위에는 한계가 있다. 깊이가 변하면 선이
    시차만큼 움직이는데 그 움직임이 이웃 선 간격을 넘으면 어느 선인지
    구분되지 않는다:   b·(1/Z_near − 1/Z_far) < 발산각/(선수−1)
========================================================================
"""
import os as _os
import sys as _sys
import json as _json
import math as _math

import numpy as np

__all__ = ["check_capture", "check_many", "check_params", "check_image",
           "check_geometry", "template", "colab", "demo", "main"]

# =====================================================================
# 규약
# =====================================================================
SOURCE = {"measured": "실측(캘리브레이션)", "spec": "데이터시트",
          "assumed": "가정 — 확인 필요"}

REQUIRED = ("camera.f_px", "camera.cx_px", "camera.cy_px",
            "camera.sensor_W", "camera.sensor_H", "baseline_m")
RECOMMENDED = ("grid.n_vertical", "grid.n_horizontal", "grid.fov_deg")
# 굴림·수렴각은 세 자리 중 어디에 적어도 된다(검측 파이프라인이 다 읽는다).
ROLL_KEYS = ("laser.roll_deg", "grid.laser_roll_deg", "laser_roll_deg")
TILT_KEYS = ("laser.tilt_deg", "grid.laser_tilt_deg", "laser_tilt_deg")

IMG_ON = ("laser_on", "cast", "_on", "on_", "grid", "레이저")
IMG_OFF = ("laser_off", "cam", "scene", "_off", "off_", "배경", "장면")
IMG_EXT = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
LOSSY_EXT = (".jpg", ".jpeg")

# 가로선이 깊이를 주려면 이 이득 아래여야 한다. g = 1/sin γ 이므로
# 3.0 은 굴림 19.5° 에 해당한다.
MAX_DEPTH_GAIN = 3.0
CHANNELS = ("R", "G", "B")


def _dig(d, path):
    cur = d
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def _first(d, keys):
    for k in keys:
        v = _dig(d, k)
        if v is not None:
            return float(v)
    return None


def _num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) \
        and np.isfinite(float(v))


# =====================================================================
# 레이저 신호 — 어느 채널이 레이저인가
# =====================================================================
def _excess(a, c):
    """채널 c 의 과잉분 — 나머지 두 채널의 평균을 뺀다."""
    other = [k for k in range(3) if k != c]
    return a[:, :, c] - 0.5 * (a[:, :, other[0]] + a[:, :, other[1]])


def pick_channel(rgb, min_contrast=6.0, min_ratio=1.8):
    """
    어느 채널이 레이저인가. (이름, 인덱스, 점수, 근거).

    점수는 과잉분의 **상위 분위수 − 중앙값** 이다. 평균이나 최대값으로
    보면 안 된다 — 선이 차지하는 화소는 몇 %뿐이라 평균은 배경에 묻히고
    최대값은 핫픽셀 하나에 끌려간다. 중앙값을 빼면 벽 자체의 색조가
    상쇄되고 선이 얹은 만큼만 남는다.

    두 문턱을 다 넘어야 단색으로 인정한다. 절대 대비만 보면 백색 레이저·
    흑백 카메라에서 잡음 중 가장 큰 채널이 뽑혀 엉뚱한 신호를 만든다.
    """
    a = np.asarray(rgb, dtype=np.float32)
    if a.ndim == 2 or a.shape[2] < 3:
        return None, None, 0.0, "흑백 — 채널 분리 없음"
    scores = [float(np.percentile(_excess(a, c), 99.5)
                    - np.median(_excess(a, c))) for c in range(3)]
    best = int(np.argmax(scores))
    rest = max(scores[c] for c in range(3) if c != best)
    why = (" / ".join(f"{CHANNELS[c]} {scores[c]:+.1f}" for c in range(3))
           + f", 배수 {scores[best] / max(rest, 1e-6):.1f}")
    if scores[best] < float(min_contrast):
        return None, None, scores[best], f"대비 부족 ({why})"
    if scores[best] < float(min_ratio) * max(rest, 1e-6):
        return None, None, scores[best], f"단색 아님 ({why})"
    return CHANNELS[best], best, scores[best], why


def laser_signal(rgb, off=None):
    """레이저만 남긴 2D 배열과 그 근거."""
    a = np.asarray(rgb, np.float32)
    if a.ndim == 2:
        a = np.repeat(a[:, :, None], 3, axis=2)
    name, idx, score, why = pick_channel(a)
    if idx is None:
        sig = a.mean(axis=2)
        mode = "밝기(단색 채널 없음)"
    else:
        sig = _excess(a, idx)
        mode = f"{name}채널 과잉분"
    if off is not None:
        b = np.asarray(off, np.float32)
        if b.ndim == 2:
            b = np.repeat(b[:, :, None], 3, axis=2)
        if b.shape == a.shape:
            sig = sig - (b.mean(axis=2) if idx is None else _excess(b, idx))
            mode += " − OFF"
    return sig, {"mode": mode, "why": why, "score": score}


# =====================================================================
# 기하 — 사양이 이미지와 맞는지 보려면 이만큼이 필요하다
# =====================================================================
def _rot_y(t):
    c, s = _math.cos(t), _math.sin(t)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rot_z(t):
    c, s = _math.cos(t), _math.sin(t)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def plane_normal(fixed, angle_rad, tilt_rad=0.0, roll_rad=0.0):
    """
    선 하나의 레이저 평면 법선 (조사기 좌표계, 단위벡터).

    V평면은 Y축과 방향 (sinα,0,cosα) 를 품으므로 법선이 (cosα, 0, −sinα),
    H평면은 X축과 (0,sinβ,cosβ) 를 품으므로 (0, −cosβ, sinβ).
    회전은 수렴각(Y축) → 굴림(광축) 순 — 장비에서 DOE 를 얹은 뒤 모듈을
    돌리는 순서와 같다.
    """
    a = float(angle_rad)
    if fixed == "alpha":
        n = np.array([_math.cos(a), 0.0, -_math.sin(a)])
    else:
        n = np.array([0.0, -_math.cos(a), _math.sin(a)])
    if tilt_rad:
        n = _rot_y(float(tilt_rad)) @ n
    if roll_rad:
        n = _rot_z(float(roll_rad)) @ n
    return n / max(float(np.linalg.norm(n)), 1e-12)


def depth_gain(normal, b_vec=(1.0, 0.0, 0.0)):
    """
    깊이 잡음 배수 g — 이상적인 세로선을 1 로 둔 무차원 값.

        Z = − (n·b) / (n_x û + n_y v̂ + n_z)

    이므로 깊이는 n·b 를 통해서만 식에 들어온다. 선에 수직인 화소 오차
    하나가 깊이에 실리는 비율이 g = |b|·√(n_x²+n_y²)/|n·b| 이고, 이것은
    **화면에서 선과 기선이 이루는 각 θ** 하나로 g = 1/|sin θ| 이다.
    n·b = 0 이면 g=∞ — 그 선은 깊이를 전혀 못 준다.
    """
    n = np.asarray(normal, float)
    b = np.asarray(b_vec, float)
    nb = float(n @ b)
    lat = float(_math.hypot(n[0], n[1]))
    if abs(nb) < 1e-12:
        return float("inf")
    return float(np.linalg.norm(b)) * lat / abs(nb)


def fan_angles(n, fov_deg, model="equal_angle"):
    """DOE 가 만드는 n 개 광선의 발사각 [rad]. 바깥 두 선이 ±fov/2."""
    half = _math.radians(float(fov_deg)) / 2.0
    t = np.linspace(-1.0, 1.0, int(n))
    if model == "equal_sine":
        return np.arcsin(_math.sin(half) * t)
    return half * t


def _line_runs(mask, min_gap=3):
    """1차원 불리언 프로파일에서 선 중심 위치."""
    idx = np.where(mask)[0]
    if not len(idx):
        return np.array([])
    brk = np.where(np.diff(idx) > min_gap)[0]
    return np.array([g.mean() for g in np.split(idx, brk + 1) if len(g)])


def _occupancy_peaks(coord, lit, occupancy, min_capacity=0.25):
    """
    1화소 폭 띠마다 "켜진 화소 / 그 띠에 들어오는 화소" 를 재어 선을 찾는다.

    띠 길이가 화면 모서리에서 짧아지므로 분모(수용량)를 같이 세어 비율로
    본다. 굴리지 않은 격자라면 이것이 그냥 열 평균과 같다.
    """
    lo = int(_math.floor(float(coord.min())))
    hi = int(_math.ceil(float(coord.max()))) + 1
    edges = np.arange(lo, hi + 1, dtype=float)
    cap, _ = np.histogram(coord, bins=edges)
    got, _ = np.histogram(coord[lit], bins=edges)
    med = float(np.median(cap[cap > 0])) if (cap > 0).any() else 0.0
    keep = cap >= min_capacity * max(med, 1.0)
    ratio = np.zeros(len(cap), float)
    ratio[keep] = got[keep] / cap[keep]
    return _line_runs(ratio >= occupancy) + lo


def read_grid(sig, roll_rad=0.0, cx=None, cy=None, occupancy=0.35):
    """
    격자를 **예측 없이** 이미지에서 읽는다. (V선 위치, H선 위치).

    사양이 틀린 상태를 잡아내는 것이 목적이므로 사양으로 만든 예측을
    씨앗으로 쓰면 안 된다. 축 투영 + 임계만 쓴다.

    임계를 분위수로 잡으면 안 된다 — 선이 차지하는 화소는 전체의 몇 %뿐
    이라 90% 분위수가 배경값이 되고 화면 전체가 선으로 잡힌다. 신호의
    중앙값과 최대값 사이에서 잡는다.

    굴린 격자에서는 좌표를 −γ 만큼 되돌려 읽는다. 굴림은 레이저 평면
    법선을 광축 둘레로 돌리므로 **모든** V선이 화면에서 정확히 γ 만큼
    기울고(발사각과 무관), 되돌린 좌표에서는 굴리지 않은 식이 그대로
    성립한다. 이미지가 아니라 좌표를 돌리므로 리샘플링 오차가 없다.
    """
    med = float(np.median(sig))
    hi = float(np.percentile(sig, 99.5))
    if hi - med < 1e-6:
        return np.array([]), np.array([])
    m = sig >= med + 0.4 * (hi - med)
    H, W = m.shape
    if abs(float(roll_rad)) < 1e-9:
        return (_line_runs(m.mean(axis=0) >= occupancy),
                _line_runs(m.mean(axis=1) >= occupancy))
    cx = 0.5 * (W - 1) if cx is None else float(cx)
    cy = 0.5 * (H - 1) if cy is None else float(cy)
    uu, vv = np.meshgrid(np.arange(W, dtype=np.float32),
                         np.arange(H, dtype=np.float32))
    du, dv = uu - cx, vv - cy
    c, s = _math.cos(float(roll_rad)), _math.sin(float(roll_rad))
    lit = m.ravel()
    return (_occupancy_peaks((cx + du * c + dv * s).ravel(), lit, occupancy),
            _occupancy_peaks((cy - du * s + dv * c).ravel(), lit, occupancy))


def measure_roll(sig, cx=None, cy=None, span=45.0, step=1.0):
    """
    격자가 화면에서 실제로 얼마나 기울어 있는가 [deg].

    사양에 적힌 굴림각을 믿지 않고 이미지에서 직접 잰다. 좌표를 −γ 로
    되돌렸을 때 한 계열이 가장 좁은 띠에 모이는 γ 를 찾는다. 모임의
    정도는 히스토그램의 집중도(∑h² / (∑h)²)로 본다.

    V선과 H선은 직교하므로 답이 90° 주기다. ±45° 안에서만 찾아 한 값으로
    정한다 — 격자를 40° 넘게 굴려 끼우는 일은 없다.
    """
    med = float(np.median(sig))
    hi = float(np.percentile(sig, 99.5))
    if hi - med < 1e-6:
        return None, 0.0
    m = sig >= med + 0.4 * (hi - med)
    H, W = m.shape
    cx = 0.5 * (W - 1) if cx is None else float(cx)
    cy = 0.5 * (H - 1) if cy is None else float(cy)
    ys, xs = np.nonzero(m)
    if len(xs) < 200:
        return None, 0.0
    if len(xs) > 200000:                       # 큰 이미지는 솎아 쓴다
        k = np.linspace(0, len(xs) - 1, 200000).astype(int)
        ys, xs = ys[k], xs[k]
    du, dv = xs - cx, ys - cy
    best = (None, -1.0)
    for g in np.arange(-span, span + 1e-9, step):
        r = _math.radians(float(g))
        up = du * _math.cos(r) + dv * _math.sin(r)
        h, _ = np.histogram(up, bins=np.arange(up.min(), up.max() + 2, 1.0))
        tot = float(h.sum())
        if tot <= 0:
            continue
        conc = float((h.astype(float) ** 2).sum()) / (tot * tot) * len(h)
        if conc > best[1]:
            best = (float(g), conc)
    return best


def fit_focal(u_lines, alphas, f_hint=None, tol_frac=0.22):
    """
    읽은 선 위치에서 **초점거리를 직접 잰다.**

        d_i = f·tan α_i  −  f·b·cos γ / Z  + c_x
              └ 간격을 정하는 항 ┘  └ 전체를 통째로 옮기는 항 ┘

    선 **사이 간격** 은 f 와 발사각만으로 정해지고 거리·기선과 무관하다.
    그래서 읽은 위치를 tan α 에 맞추면 기울기가 곧 f 다. 사양의 f 와
    견주면 "이 이미지를 찍은 초점거리가 맞는가" 를 바로 알 수 있다 —
    맞은 선 수를 세는 것보다 훨씬 또렷하다.

    선이 몇 개 빠져 있어도 풀려야 한다
    --------------------------------
    앞에 선 부재가 선을 가리면 21개 중 19개만 읽힌다. 순서대로 1:1 로
    짝지으면 그 순간 전부 한 칸씩 밀려 f 가 엉뚱하게 나온다. 그래서
    **읽은 선 두 개와 발사각 두 개** 로 (f, C) 후보를 만들고, 그 후보에
    가장 많은 선이 맞는 것을 고른다. 빠진 선은 그냥 안 맞는 선이 된다.

    기울기가 음수로 나오면 화소가 180° 돌아 저장된 것이다.
    """
    u = np.sort(np.asarray(u_lines, float))
    t = np.tan(np.sort(np.asarray(alphas, float)))
    n, m = len(u), len(t)
    if n < 3 or m < 3:
        return None
    gap_t = float(np.min(np.diff(t)))
    jj, kk = np.triu_indices(n, 1)
    pp, qq = np.triu_indices(m, 1)
    dt = t[qq] - t[pp]
    fc = (u[kk] - u[jj])[:, None] / dt[None, :]
    Cc = u[jj][:, None] - fc * t[pp][None, :]
    fc, Cc = fc.ravel(), Cc.ravel()
    lo, hi = ((0.2 * f_hint, 5.0 * f_hint) if f_hint else (1.0, 1e6))
    keep = np.isfinite(fc) & np.isfinite(Cc) & (fc > lo) & (fc < hi)
    if not keep.any():                       # 뒤집힌 경우까지 살펴본다
        keep = np.isfinite(fc) & np.isfinite(Cc) & (np.abs(fc) > lo)
        if not keep.any():
            return None
    fc, Cc = fc[keep], Cc[keep]
    best = None
    for s0 in range(0, len(fc), 4000):       # 메모리를 위해 나눠 센다
        f_b, C_b = fc[s0:s0 + 4000], Cc[s0:s0 + 4000]
        pred = f_b[:, None] * t[None, :] + C_b[:, None]        # (B,m)
        d = np.abs(pred[:, None, :] - u[None, :, None])        # (B,n,m)
        near = d.min(axis=2)                                   # (B,n)
        tol = tol_frac * np.abs(f_b) * gap_t
        hit = near <= tol[:, None]
        cnt = hit.sum(axis=1)
        i = int(np.argmax(cnt))
        key = (int(cnt[i]), -float(near[i][hit[i]].mean() if cnt[i] else 9e9))
        if best is None or key > best[0]:
            best = (key, float(f_b[i]), float(C_b[i]))
    if best is None or best[0][0] < 3:
        return None
    # 맞은 선만으로 최소제곱 다듬기
    f0, C0 = best[1], best[2]
    pred = f0 * t + C0
    d = np.abs(pred[None, :] - u[:, None])
    j = d.argmin(axis=1)
    ok = d.min(axis=1) <= tol_frac * abs(f0) * gap_t
    if ok.sum() >= 3:
        A = np.stack([t[j[ok]], np.ones(int(ok.sum()))], axis=1)
        sol = np.linalg.lstsq(A, u[ok], rcond=None)[0]
        f0, C0 = float(sol[0]), float(sol[1])
        resid = float(np.max(np.abs(u[ok] - A @ sol)))
    else:
        resid = float("nan")
    return {"f_px": f0, "offset_px": C0, "잔차_px": resid,
            "맞은 선": int(ok.sum()), "읽은 선": int(n)}


def solve_standoff(u_lines, params, cx, f, b, roll_rad, tilt_rad,
                   n_v, fov_deg, model="equal_angle", z_lo=0.3, z_hi=8.0):
    """
    "읽은 선"과 "사양대로의 예측"이 겹치는 거리 Z 를 찾는다.

    선 i 의 레이저 평면이 깊이 Z 의 정면 평면과 만나는 자취는, 주점에서
    잰 **수직 거리** 로 쓰면

        d_i(Z) = − (n_z + (n·b)/Z) · f / √(n_x² + n_y²)

    이다. 굴림만 있는 경우 이것은 f·tanα − f·b·cosγ/Z 로 줄어든다
    (시차 항에 cos γ 가 붙는다 — 빼먹으면 Z 가 1/cosγ 배로 나온다).

    미지수는 Z 하나뿐이므로 Z 를 훑으며 **읽은 선이 예측에 몇 개나
    맞는지** 세고, 가장 많이 맞는 Z 를 고른 뒤 맞은 짝으로만 다시 푼다.
    순서대로 짝지으면 안 된다 — 앞에 선 부재가 선 하나만 가려도 짝이
    통째로 밀려 Z 가 선 간격 하나만큼 틀린다.

    **겹치는 Z 가 없다는 것은 사양의 숫자가 이 이미지의 것이 아니라는
    뜻이다.** 이 점검기가 잡으려는 것이 바로 그 상태다.
    """
    u = np.sort(np.asarray(u_lines, float))
    if len(u) < 4 or not n_v or n_v < 2:
        return None
    al = fan_angles(int(n_v), float(fov_deg), model)
    base = []
    for a in al:
        n = plane_normal("alpha", float(a), tilt_rad, roll_rad)
        lat = _math.hypot(n[0], n[1])
        if lat < 1e-9:
            return None
        base.append((-n[2] * f / lat, -n[0] * b * f / lat))   # d = c0 + c1/Z
    c0 = np.array([x[0] for x in base]) + cx
    c1 = np.array([x[1] for x in base])
    gap = float(np.min(np.diff(np.sort(c0)))) if len(c0) > 1 else 40.0
    tol = max(0.25 * abs(gap), 1.5)
    best = None
    for iz in np.linspace(1.0 / z_hi, 1.0 / z_lo, 1400):
        pred = c0 + c1 * iz
        d = np.abs(u[:, None] - pred[None, :])
        near = d.min(axis=1)
        hit = near <= tol
        n_hit = int(hit.sum())
        if n_hit < max(4, int(0.5 * len(u))):
            continue
        j = d[hit].argmin(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            zz = c1[j] / (u[hit] - c0[j])
        zz = zz[np.isfinite(zz) & (zz > 0)]
        if len(zz) < 4:
            continue
        z = float(np.median(zz))
        if not (z_lo <= z <= z_hi):
            continue
        resid = float(np.median(np.abs(u[hit] - (c0[j] + c1[j] / z))))
        key = (n_hit, -resid)
        if best is None or key > best[0]:
            best = (key, {"z_m": round(z, 4), "맞은 선": n_hit,
                          "읽은 선": int(len(u)),
                          "잔차_px": round(resid, 2),
                          "허용_px": round(tol, 1), "모델": model,
                          "선간격_px": round(abs(gap), 1)})
    return best[1] if best else None


# =====================================================================
# 검사 — 1. 사양 값
# =====================================================================
def template():
    """업체에 건네는 camera_params.json 빈 서식."""
    return {
        "camera": {"f_px": None, "cx_px": None, "cy_px": None,
                   "sensor_W": None, "sensor_H": None},
        "baseline_m": None,
        "grid": {"n_vertical": None, "n_horizontal": None, "fov_deg": None},
        "laser": {"roll_deg": 0.0, "tilt_deg": 0.0, "wavelength_nm": None},
        "_출처": {
            "f_px/cx_px/cy_px": "체커보드 캘리브레이션 — 추정 금지",
            "sensor_W/H": "센서 사양",
            "baseline_m": "조립 후 실측 (1mm 오차 = 깊이 0.7%)",
            "grid.*": "DOE 데이터시트",
            "laser.roll_deg": "DOE 를 광축 둘레로 돌려 끼운 각. 0 이면 "
                              "가로선이 깊이를 못 준다 (20° 이상 권장)",
            "lines": "선택 — 선별 레이저 평면 법선을 직접 쟀다면 "
                     "{\"V0\": {\"normal\": [nx,ny,nz]}, ...} 로 넣으면 "
                     "각도 모델보다 정확하다"},
    }


def check_params(params):
    """camera_params 를 훑어 (문제목록, 경고목록, 요약)."""
    bad, warn, info = [], [], {}
    if not isinstance(params, dict):
        return ["camera_params 가 dict 가 아니다"], [], {}
    for k in REQUIRED:
        v = _dig(params, k)
        if v is None:
            bad.append(f"필수 값 없음: {k}")
        elif not _num(v):
            bad.append(f"필수 값이 숫자가 아님: {k} = {v!r}")
    for k in RECOMMENDED:
        if _dig(params, k) is None:
            warn.append(f"권장 값 없음: {k}")
    if all(_dig(params, k) is None for k in TILT_KEYS):
        warn.append("권장 값 없음: 수렴각 (laser.tilt_deg 등) — 0 으로 가정")

    f = _dig(params, "camera.f_px")
    b = params.get("baseline_m")
    W = _dig(params, "camera.sensor_W")
    if _num(f) and _num(b) and float(b):
        info["f_px"] = float(f)
        info["baseline_m"] = float(b)
        z = 1.7
        info["깊이잡음_mm@1.7m_σu0.3px"] = round(
            0.3 * z * z / (float(f) * float(b)) * 1000.0, 2)
        if float(b) <= 0:
            bad.append(f"기선이 0 이하다: {b}")
        elif float(b) < 0.05:
            warn.append(f"기선이 {float(b)*1000:.0f}mm 로 짧다 — 깊이 잡음이 "
                        f"기선에 반비례한다")
        elif float(b) > 1.0:
            warn.append(f"기선이 {float(b)*1000:.0f}mm 로 크다 — 값의 단위가 "
                        f"m 가 맞는지 확인할 것")
    # 초점거리와 센서 폭이 말이 되는 화각을 만드는가
    if _num(f) and _num(W) and float(f) > 0:
        fov = 2.0 * _math.degrees(_math.atan(float(W) / 2.0 / float(f)))
        info["카메라 화각_deg"] = round(fov, 1)
        if not (10.0 <= fov <= 150.0):
            bad.append(f"f_px 와 sensor_W 가 만드는 카메라 화각이 {fov:.0f}° "
                       f"— 단위(화소 vs mm)를 확인할 것")
    roll = _first(params, ROLL_KEYS)
    tilt = _first(params, TILT_KEYS)
    if roll is not None:
        info["laser_roll_deg"] = float(roll)
        g = depth_gain(plane_normal("beta", 0.0, _math.radians(tilt or 0.0),
                                    _math.radians(float(roll))))
        info["가로선 깊이이득"] = ("∞" if not np.isfinite(g) else round(g, 2))
        if not np.isfinite(g) or g > MAX_DEPTH_GAIN:
            warn.append(
                f"격자 굴림이 {float(roll):.1f}° 라 가로선 깊이이득이 "
                f"{'∞' if not np.isfinite(g) else f'{g:.1f}'} — 상한 "
                f"{MAX_DEPTH_GAIN:.0f} 을 넘어 가로선을 깊이에 못 쓴다. "
                f"세로선만으로도 수직도·수평도는 나오지만 원형 부재(동바리)의 "
                f"단면(지름·중심)은 측정되지 않는다. "
                f"g = 1/sin γ 이므로 20° 이상 돌려 끼울 것")
    else:
        warn.append("laser.roll_deg 가 없다 — 0 으로 가정한다 "
                    "(그러면 가로선이 깊이를 못 준다)")

    lines = params.get("lines")
    if isinstance(lines, dict) and lines:
        info["법선을 직접 준 선"] = sum(
            1 for v in lines.values()
            if isinstance(v, dict) and v.get("normal"))
    return bad, warn, info


# =====================================================================
# 검사 — 2. 이미지
# =====================================================================
def _open_rgb(path):
    from PIL import Image
    im = Image.open(path)
    return im, np.asarray(im.convert("RGB"), np.float32)


def check_image(path):
    """이미지 한 장이 검측에 쓸 만한가."""
    out = {"path": path}
    try:
        from PIL import Image                                  # noqa: F401
    except ImportError:
        return {"path": path, "error": "Pillow 가 없다: pip install pillow"}
    try:
        im, a = _open_rgb(path)
    except Exception as e:
        return {"path": path, "error": f"열 수 없음: {e}"}

    out["크기"] = f"{im.width}×{im.height}"
    out["형식"] = im.format
    sig, meta = laser_signal(a)
    out["레이저 채널"] = meta["mode"]
    out["채널 대비"] = meta["why"]

    hi = float(np.percentile(sig, 99.9))
    med = float(np.median(sig))
    thr = med + 0.5 * (hi - med)
    lit = sig[sig > thr]
    out["점등 화소 비율"] = round(float(len(lit)) / sig.size, 4)
    out["이진(안티에일리어싱 없음)"] = bool(
        len(lit) and float((lit > hi * 0.9).mean()) > 0.9)
    out["과포화 화소 비율"] = round(float((a.max(axis=2) >= 254.0).mean()), 5)
    out["신호 대비"] = round(hi - med, 1)
    # 선폭 — 켜진 화소 수를 선 길이로 나눈 대략값. 1px 미만이면 계단이
    # 생겨 중심을 못 잡고, 6px 를 넘으면 초점이 나갔거나 번진 것이다.
    v, h = read_grid(sig)
    n_lines = len(v) + len(h)
    if n_lines:
        span = float(sig.shape[0] * len(v) + sig.shape[1] * len(h))
        out["선폭_px"] = round(float(len(lit)) / max(span, 1.0), 2)
    out["읽은 선"] = f"세로 {len(v)} / 가로 {len(h)}"
    return out


# =====================================================================
# 검사 — 3. 사양이 이미지와 맞는가 (이 점검기의 핵심)
# =====================================================================
def check_geometry(path, params):
    """
    camera_params 의 숫자가 **실제로 이 이미지를 찍은 값인지** 본다.

    파일이 다 있고 이미지도 깨끗한데 사양이 다른 장비의 것이면, 검측은
    그대로 돌아가고 결과만 조용히 틀린다. 여기서 잡는다.
    """
    bad, warn, info = [], [], {}
    try:
        im, a = _open_rgb(path)
    except Exception as e:
        return [f"이미지를 열 수 없다: {e}"], [], {}
    sig, _meta = laser_signal(a)
    H, W = sig.shape

    f = _dig(params, "camera.f_px")
    cx = _dig(params, "camera.cx_px")
    cy = _dig(params, "camera.cy_px")
    sw = _dig(params, "camera.sensor_W")
    sh = _dig(params, "camera.sensor_H")
    b = params.get("baseline_m")
    n_v = _dig(params, "grid.n_vertical")
    n_h = _dig(params, "grid.n_horizontal")
    fov = _dig(params, "grid.fov_deg")
    roll = _first(params, ROLL_KEYS) or 0.0
    tilt = _first(params, TILT_KEYS) or 0.0

    # ── 센서 크기 ↔ 실제 이미지 ──
    if _num(sw) and _num(sh):
        info["사양 센서"] = f"{int(sw)}×{int(sh)}"
        rx, ry = float(sw) / W, float(sh) / H
        if abs(rx - ry) > 0.02 * max(rx, ry):
            bad.append(f"이미지 {W}×{H} 와 사양 센서 {int(sw)}×{int(sh)} 의 "
                       f"가로세로 배율이 다르다 ({rx:.3f} vs {ry:.3f}) — "
                       f"잘라낸 이미지라면 잘라낸 뒤 값으로 다시 적을 것")
        elif abs(rx - 1.0) > 0.02:
            info["축소 배율"] = round(rx, 3)
            warn.append(f"이미지가 사양 센서의 1/{rx:.2f} 축소본이다 — "
                        f"검측은 f·cx·cy 를 이 배율로 함께 줄여 쓴다. "
                        f"원본을 그대로 주는 편이 정밀도에 낫다")
    # ── 주점 ──
    if _num(cx) and _num(cy):
        scale = (float(sw) / W) if (_num(sw) and W) else 1.0
        offx = abs(float(cx) / max(scale, 1e-9) - 0.5 * (W - 1)) / W
        offy = abs(float(cy) / max(scale, 1e-9) - 0.5 * (H - 1)) / H
        info["주점 이탈"] = f"가로 {offx*100:.1f}% / 세로 {offy*100:.1f}%"
        if max(offx, offy) > 0.15:
            warn.append(f"주점이 화면 중심에서 {max(offx, offy)*100:.0f}% "
                        f"떨어져 있다 — 캘리브레이션 결과가 맞는지 확인할 것")

    # ── 격자가 실제로 얼마나 기울어 있는가 ──
    got_roll, conc = measure_roll(sig, cx=0.5 * (W - 1), cy=0.5 * (H - 1))
    if got_roll is None:
        bad.append("레이저 격자를 이미지에서 못 읽었다 — 노출과 채널을 "
                   "확인할 것")
        return bad, warn, info
    info["측정 굴림_deg"] = round(got_roll, 1)
    info["사양 굴림_deg"] = round(float(roll), 1)
    d_roll = abs(((float(roll) - got_roll + 45.0) % 90.0) - 45.0)
    if d_roll > 3.0:
        bad.append(f"사양의 굴림 {float(roll):.1f}° 와 이미지에서 잰 "
                   f"{got_roll:.1f}° 가 {d_roll:.1f}° 다르다 — 둘 중 하나가 "
                   f"틀렸다. 굴림이 틀리면 가로선 깊이가 통째로 어긋난다")

    # ── 읽은 선 수 ↔ 사양의 선 수 ──
    v, h = read_grid(sig, _math.radians(got_roll),
                     cx=0.5 * (W - 1), cy=0.5 * (H - 1))
    info["읽은 선"] = f"세로 {len(v)} / 가로 {len(h)}"
    for nm, got, want in (("세로", len(v), n_v), ("가로", len(h), n_h)):
        if not _num(want):
            continue
        want = int(want)
        if got == 0:
            bad.append(f"{nm}선을 하나도 못 읽었다 (사양 {want}개)")
        elif got > want:
            bad.append(f"{nm}선이 사양보다 많이 읽혔다 — 읽음 {got} > 사양 "
                       f"{want}. 선 수나 발산각이 데이터시트와 다르다")
        elif got < want * 0.6:
            warn.append(f"{nm}선이 사양 {want}개 중 {got}개만 읽혔다 — "
                        f"화면 밖으로 나갔거나 가려졌을 수 있다")

    # ── 사양의 숫자가 이 이미지의 것인가 ──
    if _num(f) and _num(b) and _num(n_v) and _num(fov):
        scale = (float(sw) / W) if (_num(sw) and W) else 1.0
        fpx = float(f) / max(scale, 1e-9)
        cxp = (float(cx) / max(scale, 1e-9)) if _num(cx) else 0.5 * (W - 1)
        rr = _math.radians(got_roll)

        # (1) 초점거리 — 선 사이 간격이 정한다. 거리·기선과 무관하다.
        best = None
        for model in ("equal_angle", "equal_sine"):
            al = fan_angles(int(n_v), float(fov), model)
            ff = fit_focal(v, al, f_hint=fpx)
            if ff and (best is None or ff["잔차_px"] < best[1]["잔차_px"]):
                best = (model, ff, al)
        if best is None:
            # 선 수가 사양과 다르면 1:1 로 짝지을 수 없다 → 합의 방식으로
            r = solve_standoff(v, params, cxp, fpx, float(b), rr,
                               _math.radians(tilt), int(n_v), float(fov))
            if r is None:
                bad.append(
                    "이미지에서 읽은 격자와 사양대로의 예측이 **어떤 거리에서도 "
                    "겹치지 않는다.** 초점거리·선 수·발산각 중 하나가 이 "
                    "이미지의 값이 아니다")
            else:
                info["복원 거리_m"] = r["z_m"]
                info["맞은 선"] = f"{r['맞은 선']}/{r['읽은 선']}"
                info["예측 잔차_px"] = r["잔차_px"]
        else:
            model, ff, al = best
            info["DOE 각도모델"] = model
            f_meas = ff["f_px"]
            info["측정 초점거리_px"] = round(f_meas, 1)
            info["사양 초점거리_px"] = round(fpx, 1)
            info["직선 잔차_px"] = round(ff["잔차_px"], 2)
            info["맞은 선"] = f"{ff['맞은 선']}/{ff['읽은 선']}"
            if f_meas < 0:
                bad.append(
                    "선 순서가 사양과 반대다 — 화소가 180° 돌아 저장된 것 "
                    "같다. 저장 방향(원점이 좌상단인지)을 확인할 것")
            else:
                err = abs(f_meas - fpx) / max(fpx, 1e-9)
                info["초점거리 차이"] = f"{err*100:.1f}%"
                if err > 0.03:
                    bad.append(
                        f"사양의 초점거리 {fpx:.0f}px 와 **이미지에서 잰** "
                        f"{f_meas:.0f}px 가 {err*100:.1f}% 다르다. 선 사이 "
                        f"간격은 f 와 발사각만으로 정해지므로 거리·기선과는 "
                        f"무관하다 — f_px, grid.fov_deg, grid.n_vertical "
                        f"셋 중 하나가 이 이미지의 값이 아니다. "
                        f"깊이가 그대로 {err*100:.0f}% 틀린다")
                elif err > 0.01:
                    warn.append(
                        f"측정 초점거리 {f_meas:.0f}px 가 사양 {fpx:.0f}px 와 "
                        f"{err*100:.1f}% 다르다 — 체커보드 캘리브레이션 "
                        f"결과를 다시 확인할 것 (깊이가 그만큼 틀린다)")
            gap = float(np.min(np.diff(np.sort(np.tan(al) * max(f_meas, 1e-9)))))
            if ff["잔차_px"] > 0.15 * abs(gap):
                bad.append(
                    f"읽은 선이 발사각 모델에 맞지 않는다 (최대 잔차 "
                    f"{ff['잔차_px']:.1f}px, 선 간격 {abs(gap):.0f}px). "
                    f"DOE 발사각이 등각·등사인 어느 쪽도 아니라면 선별 평면 "
                    f"법선(lines.*.normal)을 직접 재서 넣을 것")

            # (2) 거리와 기선 — 둘은 b/Z 로만 식에 들어와 따로 못 가른다.
            par = cxp - ff["offset_px"]          # = f·b·cos γ / Z
            if par > 1e-6 and f_meas > 0:
                bz = par / (f_meas * _math.cos(rr))       # = b/Z
                info["기선/거리 비 b/Z"] = round(bz, 5)
                z_from_b = float(b) / bz
                info["복원 거리_m"] = round(z_from_b, 3)
                known = _first(params, ("검증.측정거리_m", "측정거리_m",
                                        "known_distance_m"))
                if known:
                    dz = abs(z_from_b - float(known)) / float(known)
                    info["측정거리 대조"] = (f"복원 {z_from_b:.3f}m / 실측 "
                                       f"{float(known):.3f}m ({dz*100:.1f}%)")
                    if dz > 0.03:
                        bad.append(
                            f"실측 거리 {float(known):.3f}m 인 촬영인데 사양 "
                            f"값으로 복원하면 {z_from_b:.3f}m 다 "
                            f"({dz*100:.1f}% 차이). 초점거리가 맞다면 "
                            f"**기선(baseline_m)** 이 틀렸다는 뜻이다 — "
                            f"조립 후 실측값을 다시 확인할 것")
                else:
                    warn.append(
                        f"기선을 검증할 수 없다. 깊이 식에는 b 와 Z 가 b/Z "
                        f"로만 들어와서, 사진 한 장으로는 '기선 2배'와 "
                        f"'거리 2배'가 구분되지 않는다(이 촬영은 b/Z = "
                        f"{bz:.4f}). 검증하려면 **거리를 자로 잰 촬영** 을 "
                        f"한 벌 넣고 camera_params.json 에 "
                        f"\"측정거리_m\": 1.50 처럼 적어 줄 것")
            else:
                warn.append("시차 항이 0 이하로 나왔다 — 카메라가 조사기의 "
                            "반대편(−x)에 있거나 주점 값이 어긋났을 수 있다")

        # 한 화면에 담을 수 있는 깊이 범위
        gapr = _math.radians(float(fov)) / max(int(n_v) - 1, 1)
        budget = gapr / float(b)
        zf = info.get("복원 거리_m")
        if zf:
            znear = 1.0 / (1.0 / float(zf) + budget)
            info["깊이 예산"] = (f"Δ(1/Z) < {budget:.3f} /m  →  배경 "
                              f"{float(zf):.2f}m 이면 앞쪽 {znear:.2f}m 까지")
    return bad, warn, info



# =====================================================================
# 촬영 한 벌
# =====================================================================
def _find_capture_dir(root):
    """풀어 놓은 폴더에서 촬영 폴더(이미지가 든 가장 얕은 폴더)를 찾는다."""
    best = None
    for dirpath, dirnames, filenames in _os.walk(root):
        dirnames[:] = [d for d in dirnames
                       if not d.startswith(("__MACOSX", ".", "_"))]
        if any(fn.lower().endswith(IMG_EXT) for fn in filenames):
            depth = dirpath[len(root):].count(_os.sep)
            if best is None or depth < best[0]:
                best = (depth, dirpath)
    return best[1] if best else root


def _capture_dirs(root):
    """한 폴더 아래의 촬영 폴더들 — 이미지가 든 폴더 하나가 한 벌이다."""
    out = []
    for dirpath, dirnames, filenames in _os.walk(root):
        dirnames[:] = sorted(d for d in dirnames
                             if not d.startswith(("__MACOSX", ".", "_")))
        if any(fn.lower().endswith(IMG_EXT) for fn in filenames):
            out.append(dirpath)
    return out or [root]


def check_capture(folder, verbose=True):
    """
    촬영 폴더 하나를 검사한다.

    Returns
    -------
    dict — ok(필수 항목 충족), 쓸만함(사양↔이미지 정합까지), 판정,
           필수문제 / 불일치 / 경고, 파일, 이미지, 사양, 기하
    """
    res = {"folder": folder, "필수문제": [], "불일치": [], "경고": [],
           "파일": {}, "이미지": {}, "사양": {}, "기하": {}}
    if not _os.path.isdir(folder):
        res["필수문제"].append(f"폴더가 없다: {folder}")
        res["ok"] = res["쓸만함"] = False
        res["판정"] = "검측 불가"
        return res

    names = sorted(_os.listdir(folder))
    imgs = [n for n in names if n.lower().endswith(IMG_EXT)]
    jsons = [n for n in names if n.lower().endswith(".json")]

    on = next((n for n in imgs if any(k in n.lower() for k in IMG_ON)), None)
    off = next((n for n in imgs if any(k in n.lower() for k in IMG_OFF)), None)
    if off is not None and off == on:
        off = None
    if on is None and imgs:
        on = imgs[0]
        res["경고"].append(f"이름으로 ON 프레임을 못 골랐다 — {on} 를 쓴다. "
                          f"laser_on / laser_off 로 이름 지을 것")
    if on is None:
        res["필수문제"].append("레이저 ON 이미지가 없다")
    res["파일"]["laser_on"] = on
    res["파일"]["laser_off"] = off
    if off is None:
        res["경고"].append(
            "레이저 OFF 프레임이 없다 — 배경광이 센 현장(햇빛)에서는 "
            "차영상 없이 선을 놓칠 수 있다. 같은 노출로 수십 µs 안에 "
            "연속 촬영할 것")
    if on and on.lower().endswith(LOSSY_EXT):
        res["경고"].append(
            f"{on} 이 손실 압축(jpg)이다 — 압축 잡음이 선 중심을 흔들어 "
            f"정밀도를 깎는다. png 나 tiff 로 저장할 것")

    params = truth = imu = None
    for n in jsons:
        try:
            d = _json.load(open(_os.path.join(folder, n), encoding="utf-8"))
        except Exception as e:
            res["경고"].append(f"{n} 을 읽지 못했다: {e}")
            continue
        if params is None and isinstance(d, dict) and (
                "camera" in d or "f_px" in d or "baseline_m" in d):
            params, res["파일"]["camera_params"] = d, n
        elif truth is None and isinstance(d, dict) and d and \
                str(next(iter(d)))[:1] in "VH":
            truth, res["파일"]["truth"] = d, n
        elif imu is None and isinstance(d, dict) and any(
                k in d for k in ("gravity", "accel", "pitch_deg", "roll_deg")):
            imu, res["파일"]["imu"] = d, n
    if params is None:
        res["필수문제"].append(
            "camera_params.json 이 없다 — 초점거리·기선을 모르면 깊이가 "
            "통째로 배율만큼 틀린다. 서식은 python3 hardware.py --template")
    else:
        bad, warn, info = check_params(params)
        res["필수문제"] += bad
        res["경고"] += warn
        res["사양"] = info
    if imu is None:
        res["경고"].append(
            "IMU 값이 없다 — 장비가 똑바로 섰다고 가정한다. 숙여 찍었다면 "
            "바닥이 기운 벽으로 읽힌다")
    if truth is None:
        res["경고"].append("정답값이 없다 — 선검출 정확도·깊이 오차를 "
                          "대조할 수 없다(현장에서는 정상)")

    if on:
        d = check_image(_os.path.join(folder, on))
        res["이미지"]["laser_on"] = d
        if d.get("error"):
            res["필수문제"].append(f"laser_on: {d['error']}")
        else:
            if d.get("이진(안티에일리어싱 없음)"):
                res["경고"].append(
                    "선이 이진(0/255)으로 찍혔다 — 선 중심이 0.5화소 격자에 "
                    "갇혀 정밀도가 0.289px 아래로 못 간다. 안티에일리어싱을 "
                    "켜거나 실촬영본을 쓸 것")
            if (d.get("과포화 화소 비율") or 0) > 0.02:
                res["경고"].append(
                    f"과포화 화소가 {d['과포화 화소 비율']*100:.1f}% — 선 "
                    f"중심이 평평해져 능선을 못 찾는다. 노출을 줄일 것")
            if (d.get("점등 화소 비율") or 0) < 1e-4:
                res["필수문제"].append(
                    "레이저 신호가 거의 없다 — 채널 판별 결과와 노출을 "
                    "확인할 것")
            wpx = d.get("선폭_px")
            if wpx is not None and wpx > 6.0:
                res["경고"].append(
                    f"선이 {wpx:.1f}px 로 두껍다 — 초점이 나갔거나 노출이 "
                    f"과하다. 두꺼우면 중심 추정이 흔들린다")
    if off:
        d2 = check_image(_os.path.join(folder, off))
        res["이미지"]["laser_off"] = d2
        d1 = res["이미지"].get("laser_on") or {}
        if d1.get("크기") and d2.get("크기") and d1["크기"] != d2["크기"]:
            res["필수문제"].append(
                f"ON({d1['크기']}) 과 OFF({d2['크기']}) 의 크기가 다르다 — "
                f"차영상을 만들 수 없다")
        elif (d2.get("신호 대비") or 0) > 0.5 * (d1.get("신호 대비") or 1):
            res["경고"].append(
                "OFF 프레임에도 레이저가 남아 있다 — 정말 소등된 프레임인지, "
                "두 파일이 뒤바뀌지 않았는지 확인할 것")

    # ── 사양이 이미지와 맞는가 ──
    if on and params is not None and not res["필수문제"]:
        try:
            gbad, gwarn, ginfo = check_geometry(
                _os.path.join(folder, on), params)
            res["불일치"] += gbad
            res["경고"] += gwarn
            res["기하"] = ginfo
        except Exception as e:
            res["경고"].append(f"사양↔이미지 정합 검사를 못 돌렸다: {e}")

    res["ok"] = not res["필수문제"]
    res["쓸만함"] = res["ok"] and not res["불일치"]
    res["판정"] = ("검측 가능" if res["쓸만함"]
                 else ("사양과 이미지가 어긋난다 — 그대로 쓸 수 없다"
                       if res["ok"] else "검측 불가 — 필수 항목이 빠졌다"))
    if verbose:
        _print(res)
    return res


def check_many(paths, verbose=True):
    """
    여러 촬영을 한꺼번에 검사한다.

    폴더·zip·낱개 파일을 섞어 줘도 된다. **이미지가 든 폴더 하나가 한 벌**
    이므로, zip 안에 촬영 폴더가 여럿이면 각각 따로 점검한다.
    """
    import shutil
    import tempfile
    import zipfile

    todo, tmps = [], []
    loose = None
    for p in paths:
        p = _os.path.abspath(p)
        if _os.path.isdir(p):
            todo += _capture_dirs(p)
        elif p.lower().endswith(".zip"):
            td = tempfile.mkdtemp(prefix="capture_")
            tmps.append(td)
            with zipfile.ZipFile(p) as zf:
                zf.extractall(td)
            todo += _capture_dirs(td)
        elif _os.path.isfile(p):
            if loose is None:
                loose = tempfile.mkdtemp(prefix="capture_")
                tmps.append(loose)
                todo.append(loose)
            shutil.copy(p, loose)
        else:
            todo.append(p)                       # 없는 경로 — 그대로 보고
    seen, out = set(), []
    for d in todo:
        if d in seen:
            continue
        seen.add(d)
        out.append(check_capture(d, verbose=verbose))
    if verbose and len(out) > 1:
        _print_summary(out)
    return {"captures": out, "ok": all(r["ok"] for r in out),
            "쓸만함": all(r["쓸만함"] for r in out), "_tmp": tmps}


# =====================================================================
# 보고
# =====================================================================
def _print(res):
    line = "=" * 74
    print(line)
    print(f"촬영 점검 — {_os.path.basename(res['folder'].rstrip('/')) or res['folder']}")
    print(line)
    print("  [파일]")
    for k, ko in (("laser_on", "레이저 ON (필수)"),
                  ("laser_off", "레이저 OFF (권장)"),
                  ("camera_params", "카메라 사양 (필수)"),
                  ("imu", "IMU (권장)"), ("truth", "정답값 (선택)")):
        v = res["파일"].get(k)
        print(f"    {ko:<20} {v if v else '— 없음'}")
    for title, d in (("사양", res["사양"]), ("사양 ↔ 이미지", res["기하"])):
        if d:
            print(f"  [{title}]")
            for k, v in d.items():
                print(f"    {k:<20} {v}")
    for k, d in res["이미지"].items():
        print(f"  [{k}]")
        for kk, vv in d.items():
            if kk != "path":
                print(f"    {kk:<20} {vv}")
    for key, head in (("필수문제", "막힘 — 이대로는 검측을 못 돌린다"),
                      ("불일치", "불일치 — 파일은 있으나 사양이 이미지와 안 맞는다"),
                      ("경고", "경고 — 돌아가지만 정확도·판정에 영향")):
        if res[key]:
            print(f"  [{head}]")
            for m in res[key]:
                print(f"    · {m}")
    print("-" * 74)
    print(f"  결과: {res['판정']}")
    print(line)


def _print_summary(rows):
    print()
    print("=" * 74)
    print(f"전체 요약 — 촬영 {len(rows)}벌")
    print("=" * 74)
    print(f"  {'촬영':<26}{'막힘':>5}{'불일치':>7}{'경고':>5}   판정")
    for r in rows:
        nm = _os.path.basename(r["folder"].rstrip("/")) or r["folder"]
        print(f"  {nm[:26]:<26}{len(r['필수문제']):>5}{len(r['불일치']):>7}"
              f"{len(r['경고']):>5}   {r['판정']}")
    n_ok = sum(1 for r in rows if r["쓸만함"])
    print("-" * 74)
    print(f"  {n_ok}/{len(rows)} 벌이 검측 가능")
    print("=" * 74)


def save_report(res_or_list, path="점검결과.json"):
    """업체가 그대로 회신할 수 있는 결과 파일."""
    rows = res_or_list if isinstance(res_or_list, list) else [res_or_list]
    doc = {"점검기": "hardware.py", "촬영수": len(rows),
           "전체판정": ("검측 가능" if all(r["쓸만함"] for r in rows)
                    else "보완 필요"),
           "촬영": [{k: v for k, v in r.items() if not k.startswith("_")}
                  for r in rows]}
    with open(path, "w", encoding="utf-8") as fp:
        _json.dump(doc, fp, ensure_ascii=False, indent=2, default=str)
    return path


# =====================================================================
# 코랩
# =====================================================================
def colab(paths=None, verbose=True, report="점검결과.json"):
    """
    코랩에서 한 줄로 점검한다.

        import hardware
        hardware.colab()                 # 업로드 창 — 파일 여러 개 / zip
        hardware.colab('내촬영')          # 이미 올려 둔 폴더
        hardware.colab(['A', 'B.zip'])   # 여러 벌 한꺼번에

    코랩의 업로드는 파일 단위라 폴더를 통째로 올릴 수 없다. 그래서 zip
    하나든, 파일 여러 개를 한꺼번에 고르든 둘 다 받는다. 낱개로 올린
    파일들은 한 촬영으로 묶고, zip 은 풀어서 안쪽 촬영 폴더를 찾는다.
    """
    if paths:
        if isinstance(paths, str):
            paths = [paths]
        out = check_many(list(paths), verbose=verbose)
    else:
        try:
            from google.colab import files
        except ImportError:
            raise RuntimeError(
                "코랩이 아니다. 경로를 직접 줄 것: hardware.colab('촬영폴더') "
                "또는  python3 hardware.py 촬영폴더")
        print("촬영 한 벌을 올리세요 — zip 하나, 또는 파일 여러 개를 한꺼번에.")
        print("  필수: laser_on.png, camera_params.json")
        print("  권장: laser_off.png, imu.json      선택: truth.json")
        up = files.upload()
        if not up:
            print("올린 파일이 없다.")
            return None
        import tempfile
        td = tempfile.mkdtemp(prefix="upload_")
        got = []
        for name, data in up.items():
            dst = _os.path.join(td, _os.path.basename(name))
            with open(dst, "wb") as fp:
                fp.write(data)
            got.append(dst)
            # 코랩은 작업 폴더에도 사본을 남긴다 — 치운다
            if _os.path.exists(name) and \
                    _os.path.abspath(name) != _os.path.abspath(dst):
                try:
                    _os.remove(name)
                except OSError:
                    pass
        out = check_many(got, verbose=verbose)

    if report:
        rows = out["captures"]
        p = save_report(rows, report)
        print(f"\n결과 파일: {p}  — 이 파일을 회신해 주시면 됩니다.")
        try:
            from google.colab import files as _f
            _f.download(p)
        except Exception:
            pass
    return out


# =====================================================================
# 예제 만들기 — "맞는 데이터" 가 어떻게 생겼는지
# =====================================================================
def demo(out_dir="예제촬영", roll_deg=20.0, verbose=True):
    """
    규약을 만족하는 촬영 한 벌을 만들어 둔다.

    업체가 자기 데이터와 견줄 기준이 있어야 한다. 여기서 만드는 것은
    광선추적이 아니라 **레이저 평면식 그대로** 그린 격자다 — 사양 파일의
    숫자와 이미지가 정의상 맞으므로, 이 점검기를 통과하는 것이 무엇인지
    보여 주는 용도다.
    """
    from PIL import Image
    W, H = 1224, 1024
    f, b = 942.4, 0.150
    n_v = n_h = 21
    fov = 45.24
    Z = 1.50
    cx, cy = 0.5 * (W - 1), 0.5 * (H - 1)
    rr = _math.radians(float(roll_deg))

    uu, vv = np.meshgrid(np.arange(W, dtype=np.float32),
                         np.arange(H, dtype=np.float32))
    uh, vh = (uu - cx) / f, (vv - cy) / f
    lit = np.zeros((H, W), np.float32)
    sig_px = 0.9                                   # 선폭 σ [px]
    for fixed, n_lines in (("alpha", n_v), ("beta", n_h)):
        for a in fan_angles(n_lines, fov):
            n = plane_normal(fixed, float(a), 0.0, rr)
            lat = _math.hypot(n[0], n[1])
            if lat < 1e-9:
                continue
            # 깊이 Z 의 정면 평면 위에서 이 선까지의 화소 거리
            d = (n[0] * uh + n[1] * vh + n[2] + n[0] * b / Z) * f / lat
            lit = np.maximum(lit, np.exp(-0.5 * (d / sig_px) ** 2))
    rng = np.random.default_rng(3)
    base = 96.0 + 8.0 * rng.normal(0, 1, (H, W)).astype(np.float32)
    a_on = np.stack([base * 0.92, base + 150.0 * lit, base * 0.88], axis=2)
    a_off = np.stack([base * 0.92, base, base * 0.88], axis=2)
    _os.makedirs(out_dir, exist_ok=True)
    for nm, arr in (("laser_on.png", a_on), ("laser_off.png", a_off)):
        Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).save(
            _os.path.join(out_dir, nm))
    _json.dump({"camera": {"f_px": f, "cx_px": cx, "cy_px": cy,
                           "sensor_W": W, "sensor_H": H},
                "baseline_m": b,
                "grid": {"n_vertical": n_v, "n_horizontal": n_h,
                         "fov_deg": fov},
                "laser": {"roll_deg": float(roll_deg), "tilt_deg": 0.0,
                          "wavelength_nm": 520},
                # 기선은 사진 한 장으로 검증할 수 없다(b 와 Z 가 b/Z 로만
                # 식에 들어온다). 거리를 자로 잰 촬영을 한 벌 넣고 이 값을
                # 적어 주면 그때 비로소 기선을 검증할 수 있다.
                "측정거리_m": Z},
               open(_os.path.join(out_dir, "camera_params.json"), "w",
                    encoding="utf-8"), ensure_ascii=False, indent=2)
    _json.dump({"gravity": [0.0, 1.0, 0.0],
                "note": "조사기 좌표계 기준 중력 방향"},
               open(_os.path.join(out_dir, "imu.json"), "w",
                    encoding="utf-8"), ensure_ascii=False, indent=2)
    if verbose:
        print(f"[생성] {out_dir}  (배경 {Z:.2f} m, 굴림 {roll_deg:.0f}°)")
    return out_dir


# =====================================================================
# CLI
# =====================================================================
def main(argv=None):
    a = list(_sys.argv[1:] if argv is None else argv)
    if a and a[0] == "--template":
        out = a[1] if len(a) > 1 else "camera_params.json"
        _json.dump(template(), open(out, "w", encoding="utf-8"),
                   ensure_ascii=False, indent=2)
        print(f"서식을 썼다: {out}")
        return 0
    if a and a[0] == "--demo":
        d = demo(a[1] if len(a) > 1 else "예제촬영")
        return 0 if check_capture(d)["쓸만함"] else 1
    if not a:
        print(__doc__)
        return _selftest()
    r = check_many(a)
    if r["captures"]:
        save_report(r["captures"])
        print("\n결과 파일: 점검결과.json  — 이 파일을 회신해 주시면 됩니다.")
    return 0 if r["쓸만함"] else 1


# =====================================================================
# 자체 검증 — 점검기가 제 일을 하는가
# =====================================================================
def _selftest():
    import shutil
    import tempfile
    print("=" * 74)
    print("hardware.py 자체 검증")
    print("=" * 74)
    ok = True

    def t(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}"
              + (f"   {detail}" if detail else ""))

    bad, _w, _i = check_params(template())
    t(f"빈 서식은 필수 {len(REQUIRED)}건을 다 잡는다", len(bad) == len(REQUIRED))

    good = {"camera": {"f_px": 1593.0, "cx_px": 1224.0, "cy_px": 1024.0,
                       "sensor_W": 2448, "sensor_H": 2048},
            "baseline_m": 0.15,
            "grid": {"n_vertical": 21, "n_horizontal": 21, "fov_deg": 60.8},
            "laser": {"roll_deg": 0.0, "tilt_deg": 0.0}}
    bad, warn, info = check_params(good)
    t("갖춘 사양은 통과", not bad)
    t("깊이잡음을 미리 알려준다", 0.5 < info.get("깊이잡음_mm@1.7m_σu0.3px", 0) < 20,
      f"{info.get('깊이잡음_mm@1.7m_σu0.3px')}mm @1.7m")
    t("굴림 0° 면 가로선을 못 쓴다고 경고", any("굴림" in w for w in warn),
      f"가로선 이득 {info.get('가로선 깊이이득')}")
    g20 = _json.loads(_json.dumps(good))
    g20["laser"]["roll_deg"] = 20.0
    t("굴림 20° 면 그 경고가 없다",
      not any("굴림" in w for w in check_params(g20)[1]))
    t("f_px 가 없으면 막는다",
      any("camera.f_px" in m for m in check_params(
          {"camera": {"cx_px": 1, "cy_px": 1, "sensor_W": 2, "sensor_H": 2},
           "baseline_m": 0.15})[0]))
    t("단위를 mm 로 적으면 화각이 이상해져 잡힌다",
      any("화각" in m for m in check_params(
          dict(good, camera=dict(good["camera"], f_px=8.0)))[0]))

    td = tempfile.mkdtemp()
    try:
        d = demo(_os.path.join(td, "예제촬영"), roll_deg=20.0, verbose=False)
        r = check_capture(d, verbose=False)
        t("규약을 만족하는 예제는 '검측 가능'", r["쓸만함"],
          f"{r['판정']} / 측정 f {r['기하'].get('측정 초점거리_px')}px, "
          f"복원거리 {r['기하'].get('복원 거리_m')}m")
        t("측정 굴림이 사양과 일치", r["기하"].get("측정 굴림_deg") is not None
          and abs(r["기하"]["측정 굴림_deg"] - 20.0) <= 2.0,
          f"측정 {r['기하'].get('측정 굴림_deg')}° / 사양 20°")

        # ── 점검기가 실제로 어긋남을 잡는가 ──
        p = _os.path.join(d, "camera_params.json")
        orig = _json.load(open(p, encoding="utf-8"))
        for name, mut in (
                ("초점거리를 20% 틀리게 적으면", lambda c: c["camera"].update(
                    f_px=c["camera"]["f_px"] * 1.2)),
                ("선 수를 다르게 적으면", lambda c: c["grid"].update(
                    n_vertical=15)),
                ("굴림을 0 으로 적으면", lambda c: c["laser"].update(
                    roll_deg=0.0)),
                ("실측 거리와 함께 기선을 2배로 적으면", lambda c: c.update(
                    baseline_m=c["baseline_m"] * 2))):
            c = _json.loads(_json.dumps(orig))
            mut(c)
            _json.dump(c, open(p, "w", encoding="utf-8"))
            rr = check_capture(d, verbose=False)
            t(name + " 잡아낸다", bool(rr["불일치"]) and not rr["쓸만함"],
              (rr["불일치"][0][:60] + "…") if rr["불일치"] else "못 잡음")

        # 사진 한 장으로는 기선을 검증할 수 없다 — 깊이 식에 b 와 Z 가
        # b/Z 로만 들어오기 때문이다. 못 잡는 것이 정상이고, 대신 그렇게
        # 안내해야 한다. (실측 거리를 주면 위 항목처럼 잡힌다.)
        c = _json.loads(_json.dumps(orig))
        c.pop("측정거리_m", None)
        c["baseline_m"] *= 2
        _json.dump(c, open(p, "w", encoding="utf-8"))
        rr = check_capture(d, verbose=False)
        t("실측 거리가 없으면 기선은 검증 못 한다고 안내한다",
          not rr["불일치"] and any("기선을 검증할 수 없다" in w
                                for w in rr["경고"]),
          "b 와 Z 는 b/Z 로만 식에 들어온다")
        _json.dump(orig, open(p, "w", encoding="utf-8"))

        _os.remove(_os.path.join(d, "camera_params.json"))
        t("사양 파일을 빼면 막힌다", not check_capture(d, verbose=False)["ok"])
    finally:
        shutil.rmtree(td, ignore_errors=True)
    print("=" * 74)
    print("전체 통과" if ok else "실패 있음")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
