#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
calibration.py — 캘리브레이션 데이터 단일 출처
========================================================================
검측 알고리즘이 쓰는 상수를 한곳에 모은다. 이전에는 inspection.py 와
synth_scene.py 가 같은 값을 따로 들고 있었고 A_선검출·eq5 는 또 다른
기본값을 갖고 있어, 한쪽만 고치면 조용히 어긋났다.

【데이터 구분 — PDF 3.1】
  A. 매 촬영 데이터   촬영할 때마다 하드웨어가 보내는 것
  B. 캘리브레이션     출고 시 1회 측정해 모듈에 저장하는 것
  B 의 정확도가 곧 측정 정확도다. 기선 실측값과 레이저 발사각이
  부정확하면 알고리즘이 정확해도 결과가 틀어진다.

【값의 출처를 세 등급으로 구분한다】
  spec    PDF 2.2 사양표에 직접 적힌 값, 또는 그것만으로 유도되는 값
  design  PDF 에 없어 이 코드가 정한 값. 근거를 주석에 남긴다.
  assumed 가정값 — 출고 전 실측으로 대체해야 함

  assumed 로 남은 항목이 현재 측정 신뢰도의 상한이다.

【PDF 사양표를 그대로 옮기면 이렇다 (2.2)】
  카메라   2448×2048 (5MP), 글로벌 셔터 CMOS, 2/3″ 이상, 픽셀 ≥3.45µm
           노출 ≤50µs, 외부 HW 트리거, USB3/UVC
  렌즈     저왜곡 F2.0급, 초점 ~1.2m 고정 잠금      ← 초점거리 값은 없음
  레이저   녹색 520nm LD, 30~49mW, DOE 단일소자
           수직 20 + 수평 20 (400 교점 정방형 격자)
           격자 투사 범위 120cm 에서 약 936×936mm
  기선     카메라–레이저 광축 150mm
  IMU      6축, 촬영 순간 중력벡터
  측정거리 권장 1~1.5m,  목표 평활도 ±2mm / 수직·수평 ±0.5°  (1.1)

  이 표에서 자유롭게 남은 값은 사실상 렌즈 초점거리 하나뿐이다.
  아래 LENS_FOCAL_MM 주석이 그것을 어떻게 정했는지 적는다.
========================================================================
"""
import os as _os
import numpy as np

# =====================================================================
# 하드웨어 사양 (PDF 2.2) — 표에 적힌 값
# =====================================================================
PIXEL_PITCH_UM = 3.45       # spec  "픽셀 ≥3.45µm" 의 하한 = Sony IMX264
IMAGE_W = 2448              # spec  2448 × 2048 (5MP)
IMAGE_H = 2048              # spec
BASELINE_M = 0.150          # spec  카메라–레이저 광축 150mm
LENS_FNUMBER = 2.0          # spec  "저왜곡 F2.0급"
FOCUS_DISTANCE_M = 1.2      # spec  "초점 ~1.2m 고정 잠금"
LASER_WAVELENGTH_NM = 520   # spec  녹색 LD
LASER_POWER_MW = (30, 49)   # spec  출력 30~49mW

# DOE 격자 — PDF 2.2 "수직 20 + 수평 20 (400 교점)"
N_VERTICAL = 20             # spec
N_HORIZONTAL = 20           # spec

# DOE 발산각 — PDF 2.2 "120cm 에서 936×936mm" 에서 곧바로 나온다.
#   FOV = 2·atan(468 / 1200) = 42.61°
# 사양표에 각도로 적혀 있지는 않지만 다른 해석의 여지가 없으므로 spec 이다.
PDF_FOV_DEG = float(np.degrees(2 * np.arctan(468.0 / 1200.0)))   # 42.612
FOV_DEG = PDF_FOV_DEG        # 활성 프로파일이 덮어쓴다 (use_profile)

# 측정 거리 (PDF 1.1 "권장 1~1.5m")
WORK_Z_MIN_M = 1.0          # spec
WORK_Z_MAX_M = 1.5          # spec
EDGE_MARGIN_PX = 50.0       # design 센서 가장자리 여유

# 정확도 목표 (PDF 1.1)
TARGET_SIGMA_MM = 2.0       # spec  평활도 ±2mm
TARGET_ANGLE_DEG = 0.5      # spec  수직·수평도 ±0.5°

# ---------------------------------------------------------------------
# 렌즈 초점거리 — PDF 가 값을 주지 않은 유일한 광학 상수
# ---------------------------------------------------------------------
# PDF 는 "저왜곡 F2.0급, 초점 ~1.2m 고정 잠금" 이라고만 쓴다. 초점거리는
# 나머지 사양이 사실상 하나로 몰아준다.
#
#   (1) 격자를 전부 담을 것
#       DOE 는 42.61° 로 고정되어 있다. 센서(8.446×7.066mm)에 이 각도를
#       담으려면 세로 쪽이 먼저 막힌다.
#           f ≤ (H/2 − margin)·pitch / tan(21.31°) = 8.62mm
#       계산해 보면 표준 초점거리 중 6mm·8mm 만 통과하고 10mm 부터는
#       수평·수직 모두 벗어난다. 즉 8mm 가 상한이자 최선이다.
#
#   (2) 초점 1.2m 고정으로 1.0~1.5m 를 다 볼 것
#       고정 초점이므로 피사계심도가 작업거리를 덮어야 선이 흐려지지
#       않는다. 착란원을 2px(6.9µm)로 두면
#           8mm  F2.0 → 0.95 ~ 1.61m   작업거리 1.0~1.5m 를 덮는다
#           12mm F2.0 → 1.08 ~ 1.35m   양 끝이 초점 밖으로 나간다
#       PDF 가 "초점 ~1.2m 고정" 이라고 쓴 이상 12mm 는 성립하지 않는다.
#
#   (3) 목표 정밀도를 지킬 것
#       σ_Z = σ_u·Z²/(f·b) 는 1.5m 에서 1.29mm 로 목표 ±2mm 안이다.
#
# 세 조건을 동시에 만족하는 표준 초점거리는 8mm 뿐이다. 2/3″ 커버·저왜곡·
# F2.0급 8mm 렌즈는 머신비전에서 흔한 규격이다.
LENS_FOCAL_MM = 8.0         # design (PDF 사양 (1)(2)(3) 에서 유도)

# ---------------------------------------------------------------------
# 레이저 축 수렴각 — 격자를 센서 안에 담기 위한 기구 설계값
# ---------------------------------------------------------------------
# 격자의 이미지상 위치는  u = f·tan(α) − f·b/Z + c_x  이다. 기선 b 때문에
# 격자 전체가 거리에 따라 왼쪽으로 밀리며, 그 양은 1.0m 에서 348px 이다.
# 레이저 축을 카메라와 평행하게 두면 이 이동량만큼 센서 한쪽이 통째로
# 낭비되어 42.61° 격자가 근거리에서 왼쪽으로 잘린다.
#
# 레이저를 카메라 쪽으로 δ 만큼 기울이면 격자가 작업거리 한가운데에서
# 화면 중앙에 온다. 삼각측량 기하는 그대로이고 발사각의 기준축만 바뀌므로
# α_i 에 δ 를 더해 쓰면 된다. 아래 값은 1.0m·1.5m 양 끝에서 가장자리
# 여유가 최대가 되도록 0.01° 간격으로 찾은 값이다 (find_best_tilt()).
LASER_TILT_DEG = 6.18       # design 수렴각 (카메라 쪽으로)

# 선검출 정밀도 (불확실도 산정용)
SIGMA_U_PX = 0.2            # assumed 서브픽셀 반복성. 실장비 측정 필요

SENSOR_COLOR = "color"      # spec  PDF 3.1 "RGB 영상 … 컬러"
OPTICAL_FILTER_NM = None    # PDF 에 없음
LASER_POWER_MW_SPEC = LASER_POWER_MW


# =====================================================================
# 사양 프로파일 — PDF 원안과 정확도 개선안
# =====================================================================
# PDF 사양은 목표 정확도를 만족한다. 다만 여유가 항목마다 크게 다르다.
# 합성 씬으로 측정한 현행 오차는 이렇다.
#
#   면 수직·수평도   0.013°   목표 0.5° 의 1/38  — 여유가 크다
#   동바리 수직도    0.149°   목표 0.5° 의 1/3   — 여유가 작다
#   평활도 불확실도  ±0.61mm  목표 ±2mm 의 1/3   — 여유가 작다
#   깊이 잡음 σ_Z    0.83mm   (1.2m)
#
# 뒤의 두 항목이 병목이고, 원인이 같다. 1.2m 에서 격자 피치가 49.3mm 라
# Ø48.6mm 동바리에 V선이 한두 개밖에 걸리지 않고, 벽 프로파일도 49mm
# 간격으로만 찍힌다. 즉 정확도를 올리는 첫 번째 지렛대는 광학 정밀도가
# 아니라 **공간 표본 밀도**다.
#
# 개선안이 바꾸는 것과 그 이유:
#
#  (a) V선 20 → 40, H선 20 유지 (400 → 800 교점)
#      깊이를 주는 것은 V선뿐이다. 기선이 X축이라 H선은 시차가 선과
#      나란해 삼각측량이 풀리지 않는다(실장비도 V×H 교점에서만 α 를
#      회복한다). 따라서 H선을 늘려도 측정점은 하나도 늘지 않는다.
#      V선만 두 배로 늘리면 선 수는 40 → 60 (1.5배) 인데 측정 표본은
#      2배가 된다. 40+40 (선 80개) 보다 광량 부담이 작다.
#
#  (b) 컬러 → 모노 + 520nm 대역통과 필터
#      컬러 센서는 베이어 배열이라 녹색선이 적·청 화소 위에서 신호를
#      잃고, 디모자이크가 선 단면을 뭉갠다. 모노는 이 둘이 없다.
#      대역통과 필터는 배경광을 30배 줄여 직사광 아래 SNR 을 살린다.
#      두 가지로 σ_u 를 0.2 → 0.1px 로 잡았다. PDF 의 차영상 모드와
#      상충하지 않고 오히려 그 부담을 덜어준다.
#      대가: RGB 문맥 영상을 잃는다. 세그멘테이션은 흑백으로 해야 한다.
#
#  (c) 화소 3.45 → 2.74µm, 2448 → 3072 (센서 크기·렌즈·화각 그대로)
#      σ_Z 는 초점거리가 아니라 화소 수로 정해진다.
#          σ_Z = σ_u · Z · W_시야 / (N_화소 · b)
#      같은 2/3″ 에 작은 화소를 쓰면 렌즈도 심도도 그대로 두고 화소
#      수만 1.26배 올릴 수 있다. Pregius S(2.74µm) 세대가 이에 해당한다.
#
#  (d) 기선 150 → 180mm
#      σ_Z ∝ 1/b. 외형 210mm 폭 안에서 렌즈 경통 여유를 남긴 최대값이다.
#
# 바꾸지 않은 것과 그 이유:
#   · DOE 발산각 42.61°  — 좁히면 σ_Z 는 좋아지지만 담는 면적이 준다.
#     KCS 3m 직선자 기준은 이미 한 장으로 커버가 안 되므로(1.5m 에서
#     1.17m) 화각을 더 줄이는 방향은 손해다.
#   · 렌즈 8mm F2.0 초점 1.2m 고정 — 심도가 작업거리를 덮는 유일한 값.
#   · 레이저 출력 30~49mW — 선이 40 → 60 개로 늘어 선당 광량은 0.67배가
#     되지만, 모노(약 2배)와 대역통과 필터(배경 1/30)가 그 이상을
#     보상한다. 출력을 올리지 않으므로 눈 안전등급 재평가가 필요 없다.
#   · 측정거리 1.0~1.5m — PDF 권장 유지.
SPEC_PROFILES = {
    # 주신 원본 코드(v4)의 값. 실제 하드웨어 사양이 확정되기 전까지의
    # 기준선이며, 이미 Isaac 으로 뽑아 둔 렌더가 이 값으로 만들어졌다.
    # 검측식(eq1~eq6)은 이 값에 의존하지 않으므로, 사양이 들어오면
    # 아래 딕셔너리 한 줄만 바꾸면 된다.
    "legacy": {
        "label": "원본 v4",
        "pixel_pitch_um": 3.45, "image_w": 2448, "image_h": 2048,
        "sensor_color": "color", "optical_filter_nm": None,
        "baseline_m": 0.150, "n_vertical": 21, "n_horizontal": 21,
        # 원본은 f_px = 1593 을 직접 못박았다. 화소 3.45µm 로 역산하면
        # 5.496mm 렌즈에 해당한다.
        "lens_focal_mm": 1593.0 * 3.45 / 1000.0,
        "fov_deg": 60.82, "doe_model": "equal_angle",
        "sigma_u_px": 0.2, "laser_tilt_deg": 0.0,
    },
    "pdf": {
        "label": "PDF 원안",
        "pixel_pitch_um": 3.45, "image_w": 2448, "image_h": 2048,
        "sensor_color": "color", "optical_filter_nm": None,
        "baseline_m": 0.150, "n_vertical": 20, "n_horizontal": 20,
        "lens_focal_mm": 8.0, "fov_deg": PDF_FOV_DEG,
        "doe_model": "equal_sine",
        "sigma_u_px": 0.2, "laser_tilt_deg": 6.18,
    },
    "improved": {
        "label": "정확도 개선안",
        "pixel_pitch_um": 2.74, "image_w": 3072, "image_h": 2560,
        "sensor_color": "mono", "optical_filter_nm": 520,
        "baseline_m": 0.180, "n_vertical": 40, "n_horizontal": 20,
        "lens_focal_mm": 8.0, "fov_deg": PDF_FOV_DEG,
        "doe_model": "equal_sine",
        "sigma_u_px": 0.1, "laser_tilt_deg": 7.40,
    },
}
# 기본값은 원본 v4 다. 실제 하드웨어 사양이 확정되지 않았고, 이미 뽑아 둔
# Isaac 렌더가 이 값으로 만들어졌기 때문이다. 사양이 들어오면 여기를 바꾸거나
# 환경변수 LASER_GRID_PROFILE 로 전환한다.
ACTIVE_PROFILE = _os.environ.get("LASER_GRID_PROFILE", "legacy")


# =====================================================================
# 사양에서 유도되는 값
# =====================================================================
def focal_px(focal_mm=None, pitch_um=None):
    """
    렌즈 초점거리와 화소 피치에서 픽셀 단위 초점거리를 구한다.

        f_px = f_mm / pixel_pitch_mm

    삼각측량식의 f 는 픽셀 단위여야 한다. 같은 렌즈라도 센서 화소가
    작을수록 f_px 는 커지고, 그만큼 깊이 분해능이 좋아진다.

    인자를 비우면 현재 활성 프로파일 값을 쓴다. 기본값으로 못 박으면
    use_profile() 로 프로파일을 바꿔도 옛 값이 그대로 남는다.
    """
    focal_mm = LENS_FOCAL_MM if focal_mm is None else focal_mm
    pitch_um = PIXEL_PITCH_UM if pitch_um is None else pitch_um
    return focal_mm / (pitch_um / 1000.0)


def projection_mm_at(z_m, fov_deg=None):
    """거리 z_m 에서 격자가 덮는 폭 [mm]"""
    fov = np.radians(FOV_DEG if fov_deg is None else fov_deg)
    return 2.0 * z_m * 1000.0 * np.tan(fov / 2.0)


# DOE 발사각 분포 모델 — 프로파일이 정한다 (use_profile 이 덮어쓴다)
#   "equal_sine"  회절격자의 실제 거동. m 차 회절광은 sin α_m = m·λ/d 이므로
#                 발사각은 사인 등간격이다. 평면 벽에 맺힌 격자는 가장자리로
#                 갈수록 간격이 벌어진다.
#   "equal_angle" 각도 등간격. 원본 v4 가 쓰던 근사.
# 두 모델의 바깥 포락선은 같고 안쪽 배치만 다르다. 42.61°·20선에서 위치 차가
# 최대 0.19° 이고, 깊이로 환산하면 1.2m 에서 32mm 다. 출고 시 실측 α_i 가
# 이 모델을 대체한다.
DOE_ANGLE_MODEL = "equal_sine"


CAMERA_PARAMS = {}
GRID_PARAMS = {}


def use_profile(name=None):
    """
    사양 프로파일을 적용해 파생값을 다시 계산한다.

    CAMERA_PARAMS·GRID_PARAMS 는 새 객체로 갈지 않고 제자리에서 갱신한다.
    synth_scene·inspection 이 import 시점에 이 딕셔너리를 복사해 가므로,
    객체를 갈아치우면 이미 복사해 간 쪽이 옛 값을 쥔 채 남는다.
    """
    global ACTIVE_PROFILE, PIXEL_PITCH_UM, IMAGE_W, IMAGE_H, BASELINE_M
    global N_VERTICAL, N_HORIZONTAL, LENS_FOCAL_MM, SIGMA_U_PX, LASER_TILT_DEG
    global SENSOR_COLOR, OPTICAL_FILTER_NM, FOV_DEG, DOE_ANGLE_MODEL
    global F_PX, CX_PX, CY_PX, SENSOR_W_MM, SENSOR_H_MM, SENSOR_DIAG_MM

    name = ACTIVE_PROFILE if name is None else name
    if name not in SPEC_PROFILES:
        raise ValueError(f"알 수 없는 사양 프로파일: {name}")
    p = SPEC_PROFILES[name]
    ACTIVE_PROFILE = name
    PIXEL_PITCH_UM = p["pixel_pitch_um"]
    IMAGE_W, IMAGE_H = p["image_w"], p["image_h"]
    BASELINE_M = p["baseline_m"]
    N_VERTICAL, N_HORIZONTAL = p["n_vertical"], p["n_horizontal"]
    LENS_FOCAL_MM = p["lens_focal_mm"]
    SIGMA_U_PX = p["sigma_u_px"]
    LASER_TILT_DEG = p["laser_tilt_deg"]
    SENSOR_COLOR = p["sensor_color"]
    OPTICAL_FILTER_NM = p["optical_filter_nm"]
    FOV_DEG = p["fov_deg"]
    DOE_ANGLE_MODEL = p["doe_model"]

    F_PX = focal_px()
    CX_PX = IMAGE_W / 2.0              # assumed 센서 정중앙. 캘리브레이션 필요
    CY_PX = IMAGE_H / 2.0              # assumed 동일
    # 센서 물리 크기 — Isaac Sim 카메라의 aperture 에 그대로 들어간다.
    SENSOR_W_MM = IMAGE_W * PIXEL_PITCH_UM / 1000.0
    SENSOR_H_MM = IMAGE_H * PIXEL_PITCH_UM / 1000.0
    SENSOR_DIAG_MM = float(np.hypot(SENSOR_W_MM, SENSOR_H_MM))

    CAMERA_PARAMS.clear()
    CAMERA_PARAMS.update({"f_px": round(F_PX, 1), "b_m": BASELINE_M,
                          "cx_px": CX_PX, "cy_px": CY_PX,
                          "resolution": [IMAGE_W, IMAGE_H]})
    GRID_PARAMS.clear()
    GRID_PARAMS.update({"n_vertical": N_VERTICAL, "n_horizontal": N_HORIZONTAL,
                        "fov_deg": FOV_DEG, "laser_tilt_deg": LASER_TILT_DEG,
                        "samples_per_line": 250})
    return dict(CAMERA_PARAMS)


use_profile(ACTIVE_PROFILE)


def _fan_angles(n, fov_deg, model=None):
    """DOE 가 만드는 n 개 광선의 발사각 [rad]. 바깥 두 선이 ±fov/2 이다."""
    model = DOE_ANGLE_MODEL if model is None else model
    half = np.radians(fov_deg) / 2.0
    t = np.linspace(-1.0, 1.0, n)
    if model == "equal_sine":
        return np.arcsin(np.sin(half) * t)
    if model == "equal_angle":
        return half * t
    raise ValueError(f"알 수 없는 DOE 모델: {model}")


def make_line_angles(n_v=None, n_h=None, fov_deg=None, laser_tilt_deg=None,
                     model=None):
    """
    V선·H선의 발사각을 만든다. 카메라 좌표계 기준이다.

    수렴각 δ 는 α 에 그대로 더한다. 레이저를 Y축으로 δ 만큼 돌리면 광선
    (tanα₀, tanβ₀, 1) 의 수평 성분이 정확히 tan(α₀+δ) 가 되기 때문이다
    (탄젠트 덧셈정리).

    한계 — β 의 결합
      같은 회전에서 tanβ 는 1/(cosδ − sinδ·tanα₀) 배로 살짝 늘어난다.
      즉 실제 격자는 미세한 사다리꼴이며, 발산각 42.61°·δ=6.18° 에서
      가장자리 기준 약 ±4% 다. 깊이 Z 는 α 와 u 로만 정해지므로 영향이
      없고, H선 예측 위치에만 반영된다. 실장비에서는 캘리브레이션이
      이 결합을 그대로 측정해 담는다.
    """
    n_v = N_VERTICAL if n_v is None else n_v
    n_h = N_HORIZONTAL if n_h is None else n_h
    fov = FOV_DEG if fov_deg is None else fov_deg
    tilt = np.radians(LASER_TILT_DEG if laser_tilt_deg is None else laser_tilt_deg)
    a = {}
    for i, ang in enumerate(_fan_angles(n_v, fov, model) + tilt):
        a[f"V{i}"] = {"fixed": "alpha", "angle_rad": float(ang)}
    for j, ang in enumerate(_fan_angles(n_h, fov, model)):
        a[f"H{j}"] = {"fixed": "beta", "angle_rad": float(ang)}
    return a


def predicted_u(alpha_rad, z_m, camera_params=None):
    """
    발사각 α 의 V선이 거리 z_m 에서 이미지의 어디에 맺히는지.

        u = f·tan(α) − f·b/Z + c_x

    두 번째 항이 기선 때문에 생기는 시차 이동이다. 이 항을 빼먹으면
    예측 위치가 1.2m 에서 435px 어긋나, 추적 밴드(20~50px) 밖으로 나간다.
    """
    cp = camera_params or CAMERA_PARAMS
    return cp["f_px"] * np.tan(alpha_rad) - cp["f_px"] * cp["b_m"] / z_m + cp["cx_px"]


# 값의 출처 등급 — 문서·보고서가 이 표를 그대로 쓴다.
# 값의 출처는 프로파일마다 다르다. legacy 는 하드웨어 사양이 아니라
# 시뮬레이션 튜닝값이므로 전부 assumed 이고, pdf 는 사양표 그대로,
# improved 는 사양에서 유도한 설계값이다.
_PROV_COMMON = {
    "cx_px":  ("assumed", "센서 정중앙 가정. 체커보드 캘리브레이션 필요"),
    "cy_px":  ("assumed", "센서 정중앙 가정. 체커보드 캘리브레이션 필요"),
    "beta_j": ("assumed", "동일"),
    "R_t":    ("assumed", "R=I, t=(b,0,0). 스테레오 캘리브레이션 필요"),
    "R_ic":   ("assumed", "단위행렬. IMU–카메라 캘리브레이션 필요"),
    "b_a":    ("assumed", "미구현. 가속도계 bias 보정 필요"),
}
_PROV_BY_PROFILE = {
    "legacy": {
        "f_px":    ("assumed", "원본 v4 튜닝값 1593px. 5.50mm 렌즈에 해당"),
        "b_m":     ("spec",    "PDF 2.2 광축 150mm. 조립 후 실측 필요"),
        "fov_deg": ("assumed", "원본 v4 값. PDF 의 936mm(42.61°)와 다름"),
        "n_lines": ("assumed", "원본 v4 21+21. PDF 는 20+20(400교점)"),
        "sensor":  ("spec",    "PDF 3.1 RGB 컬러"),
        "tilt":    ("assumed", "원본은 레이저 축을 카메라와 평행하게 둠"),
        "alpha_i": ("assumed", "등각도 분할. DOE 실측 α_i 로 대체 필요"),
        "sigma_u": ("assumed", "0.2px 가정. 선검출 반복성 측정 필요"),
    },
    "pdf": {
        "f_px":    ("design",  "PDF 에 초점거리 없음. 8mm = 격자수용·심도·정밀도에서 유도"),
        "b_m":     ("spec",    "PDF 2.2 광축 150mm. 조립 후 실측 필요"),
        "fov_deg": ("spec",    "PDF 2.2 120cm 에서 936mm = 42.61°"),
        "n_lines": ("spec",    "PDF 2.2 수직20 + 수평20 = 400 교점"),
        "sensor":  ("spec",    "PDF 3.1 RGB 컬러"),
        "tilt":    ("design",  "레이저 축 수렴각. 기선 시차 이동량을 상쇄한다"),
        "alpha_i": ("assumed", "회절 사인등간격 모델. DOE 실측 α_i 로 대체 필요"),
        "sigma_u": ("assumed", "0.2px 가정. 선검출 반복성 측정 필요"),
    },
    "improved": {
        "f_px":    ("design",  "PDF 에 초점거리 없음. 8mm = 격자수용·심도·정밀도에서 유도"),
        "b_m":     ("design",  "PDF 원안 150mm. 외형 210mm 내 최대 180mm"),
        "fov_deg": ("spec",    "PDF 2.2 120cm 에서 936mm = 42.61°"),
        "n_lines": ("design",  "V선만 깊이를 준다. V만 2배 (PDF 원안 20+20)"),
        "sensor":  ("design",  "모노+대역통과로 σ_u 개선 (PDF 원안 컬러·무필터)"),
        "tilt":    ("design",  "레이저 축 수렴각. 기선 시차 이동량을 상쇄한다"),
        "alpha_i": ("assumed", "회절 사인등간격 모델. DOE 실측 α_i 로 대체 필요"),
        "sigma_u": ("assumed", "모노+대역통과 기준 목표값. 실장비 측정 필요"),
    },
}


def provenance(profile=None):
    """활성(또는 지정) 프로파일에서 각 값의 출처 등급과 근거."""
    name = ACTIVE_PROFILE if profile is None else profile
    out = dict(_PROV_COMMON)
    out.update(_PROV_BY_PROFILE[name])
    return out




# =====================================================================
# 해상도 환산
# =====================================================================
def scale_to_resolution(width_px, camera_params=None):
    """
    다른 해상도의 이미지에 맞춰 f, c_x, c_y 를 환산한다.

    f_px 는 센서 해상도에 비례한다. 2448px 기준값을 축소된 이미지에
    그대로 쓰면 예측 격자가 화면 밖으로 나가 검출이 통째로 실패한다
    (실측: 471px 이미지에서 21선 중 5선만 화면 안, 평균 332px 오차).
    """
    cp = dict(camera_params or CAMERA_PARAMS)
    k = float(width_px) / float(cp["resolution"][0])
    h = int(round(cp["resolution"][1] * k))
    return {**cp,
            "f_px":  cp["f_px"] * k,
            "cx_px": cp["cx_px"] * k,
            "cy_px": cp["cy_px"] * k,
            "resolution": [int(width_px), h]}


# =====================================================================
# 정합성 검사
# =====================================================================
def check_consistency(camera_params=None, grid_params=None,
                      z_min=None, z_max=None, margin_px=None, verbose=True):
    """
    격자가 작업거리 전 구간에서 센서 안에 들어오는지 확인한다.

    단순히 시야각만 비교해서는 안 된다. 격자의 이미지상 위치는 기선 때문에
    거리에 따라 f·b/Z 만큼 좌우로 이동하며, 1.0m 에서 그 양이 센서 폭의
    21% 에 이른다. 따라서 가장 가까운 거리와 가장 먼 거리 양쪽에서
    확인해야 한다.

    Returns
    -------
    dict — fits, u_range_near, u_range_far, v_range, usable_z_m
    """
    cp = camera_params or CAMERA_PARAMS
    gp = grid_params or GRID_PARAMS
    z0 = WORK_Z_MIN_M if z_min is None else z_min
    z1 = WORK_Z_MAX_M if z_max is None else z_max
    m = EDGE_MARGIN_PX if margin_px is None else margin_px
    W, H = cp["resolution"]
    f, b, cx, cy = cp["f_px"], cp["b_m"], cp["cx_px"], cp["cy_px"]

    ang = make_line_angles(gp["n_vertical"], gp["n_horizontal"],
                           gp["fov_deg"], gp.get("laser_tilt_deg", 0.0))
    al = np.array([ang[f"V{i}"]["angle_rad"] for i in range(gp["n_vertical"])])
    be = np.array([ang[f"H{j}"]["angle_rad"] for j in range(gp["n_horizontal"])])

    u_near = f * np.tan(al) - f * b / z0 + cx
    u_far = f * np.tan(al) - f * b / z1 + cx
    v = f * np.tan(be) + cy

    fits = (u_near.min() >= m and u_far.max() <= W - m
            and v.min() >= m and v.max() <= H - m)

    # 이 설계로 쓸 수 있는 거리 범위 (가장 왼쪽/오른쪽 선 기준)
    lo = f * b / max(cx + f * np.tan(al.min()) - m, 1e-9)
    hi = f * b / max(cx + f * np.tan(al.max()) - (W - m), 1e-9)
    usable = [round(float(lo), 2), round(float(hi), 2) if hi > 0 else None]

    r = {"fits": bool(fits), "margin_px": m,
         "work_z_m": [z0, z1],
         "u_range_near": [round(float(u_near.min()), 1), round(float(u_near.max()), 1)],
         "u_range_far": [round(float(u_far.min()), 1), round(float(u_far.max()), 1)],
         "v_range": [round(float(v.min()), 1), round(float(v.max()), 1)],
         "usable_z_m": usable,
         "projection_mm": {z: round(float(projection_mm_at(z, gp["fov_deg"])), 0)
                           for z in (z0, z1)}}

    if verbose and not fits:
        print(f"  [캘리브레이션 경고] 격자가 센서를 벗어난다")
        print(f"    Z={z0}m  u = {r['u_range_near'][0]} .. {r['u_range_near'][1]}"
              f"   (허용 {m:.0f} .. {W-m:.0f})")
        print(f"    Z={z1}m  u = {r['u_range_far'][0]} .. {r['u_range_far'][1]}")
        print(f"    v = {r['v_range'][0]} .. {r['v_range'][1]}"
              f"   (허용 {m:.0f} .. {H-m:.0f})")
        print(f"    → 발산각을 줄이거나 레이저 수렴각을 조정할 것")
    return r


def sigma_z_mm(z_m, camera_params=None, sigma_u_px=None):
    """
    깊이 잡음  σ_Z = σ_u · Z² / (f · b)  [mm]

    f 를 화소수로 풀어 쓰면 무엇이 정확도를 정하는지 분명해진다.
    화각을 고정하면 f_px = N_화소 · Z / W_시야 이므로

        σ_Z = σ_u · Z · W_시야 / (N_화소 · b)

    즉 렌즈 초점거리는 독립 변수가 아니다. 담을 면적(W)을 정하면
    초점거리는 따라오고, 남는 지렛대는 σ_u · N_화소 · b 세 개뿐이다.
    """
    cp = camera_params or CAMERA_PARAMS
    su = SIGMA_U_PX if sigma_u_px is None else sigma_u_px
    return su * z_m ** 2 / (cp["f_px"] * cp["b_m"]) * 1000.0


def depth_of_field(focal_mm=None, f_number=None, focus_m=None, coc_px=2.0,
                   pitch_um=None):
    """
    고정 초점 렌즈가 선명하게 담는 거리 범위 [m].

    PDF 는 "초점 ~1.2m 고정 잠금" 이라 조절 장치가 없다. 따라서 작업거리
    1.0~1.5m 가 통째로 심도 안에 들어와야 하며, 그러지 않으면 양 끝에서
    레이저선이 번져 서브픽셀 중심이 흔들린다. 착란원은 화소 몇 개인지로
    준다 (2px = 6.9µm).

        H = f²/(N·c) + f
        near = H·s/(H + (s−f)),   far = H·s/(H − (s−f))
    """
    f = LENS_FOCAL_MM if focal_mm is None else focal_mm
    N = LENS_FNUMBER if f_number is None else f_number
    s = (FOCUS_DISTANCE_M if focus_m is None else focus_m) * 1000.0
    c = coc_px * (PIXEL_PITCH_UM if pitch_um is None else pitch_um) / 1000.0
    H = f * f / (N * c) + f
    near = H * s / (H + (s - f))
    far = H * s / (H - (s - f)) if H > (s - f) else float("inf")
    return near / 1000.0, far / 1000.0


def find_best_tilt(focal_mm=None, fov_deg=None, n_v=None, n_h=None,
                   z_min=None, z_max=None, step_deg=0.01, max_deg=15.0):
    """
    가장자리 여유가 최대가 되는 레이저 수렴각을 찾는다.

    작업거리 양 끝에서 좌우 네 여유(근거리 좌·우, 원거리 좌·우) 중 최소값을
    최대로 만드는 δ 를 고른다. LASER_TILT_DEG 는 이 함수가 낸 값이다.
    """
    cp = dict(CAMERA_PARAMS)
    if focal_mm is not None:
        cp["f_px"] = focal_px(focal_mm)
    f, b, cx = cp["f_px"], cp["b_m"], cp["cx_px"]
    W = cp["resolution"][0]
    z0 = WORK_Z_MIN_M if z_min is None else z_min
    z1 = WORK_Z_MAX_M if z_max is None else z_max
    a0 = _fan_angles(N_VERTICAL if n_v is None else n_v,
                     FOV_DEG if fov_deg is None else fov_deg)
    best = (0.0, -1e9)
    for d in np.arange(0.0, max_deg, step_deg):
        al = a0 + np.radians(d)
        margins = []
        for z in (z0, z1):
            u = f * np.tan(al) - f * b / z + cx
            margins += [u.min(), W - u.max()]
        m = min(margins)
        if m > best[1]:
            best = (float(d), float(m))
    return {"tilt_deg": round(best[0], 2), "margin_px": round(best[1], 1)}


def isaac_camera_params():
    """
    Isaac Sim(USD) 카메라에 넣을 물리 파라미터.

    USD 카메라는 초점거리와 aperture 를 mm 로 받고 화각을 그것으로 정한다.
    센서 실물 크기를 그대로 넣으면 f_px 가 정확히 재현되고, 렌더된 이미지의
    픽셀 좌표가 삼각측량식의 u, v 와 같은 뜻을 갖는다.

        f_px = focal_length · resolution_x / horizontal_aperture

    aperture 를 36mm 같은 임의값으로 두고 초점거리를 역산해도 f_px 는
    같지만, 그러면 f-stop·초점거리가 실물과 달라져 심도·보케를 켰을 때
    사양과 다른 이미지가 나온다.
    """
    return {
        "focal_length_mm":      LENS_FOCAL_MM,
        "horizontal_aperture_mm": round(SENSOR_W_MM, 4),
        "vertical_aperture_mm":   round(SENSOR_H_MM, 4),
        "f_stop":               LENS_FNUMBER,
        "focus_distance_m":     FOCUS_DISTANCE_M,
        "resolution":           [IMAGE_W, IMAGE_H],
        "clipping_range_m":     [0.01, 50.0],
    }


def fov_mm_at(z_m, camera_params=None):
    """거리 z_m 에서 카메라가 담는 시야 (가로, 세로) [mm]"""
    cp = camera_params or CAMERA_PARAMS
    return (2 * z_m * 1000 * cp["cx_px"] / cp["f_px"],
            2 * z_m * 1000 * cp["cy_px"] / cp["f_px"])


def compare_profiles(z=1.2):
    """두 프로파일의 파생값을 나란히 낸다."""
    keep = ACTIVE_PROFILE
    rows = []
    for name in ("legacy", "pdf", "improved"):
        use_profile(name)
        r = check_consistency(verbose=False)
        near, far = depth_of_field()
        rows.append({
            "name": name, "label": SPEC_PROFILES[name]["label"],
            "sensor": f"{IMAGE_W}×{IMAGE_H} @{PIXEL_PITCH_UM}µm {SENSOR_COLOR}",
            "filter": (f"{OPTICAL_FILTER_NM}nm 대역통과"
                       if OPTICAL_FILTER_NM else "없음"),
            "f_px": round(F_PX, 1), "b_mm": BASELINE_M * 1000,
            "lines": f"V{N_VERTICAL} + H{N_HORIZONTAL}",
            "intersections": N_VERTICAL * N_HORIZONTAL,
            "sigma_u": SIGMA_U_PX,
            "tilt": LASER_TILT_DEG,
            "pitch_mm": round(projection_mm_at(z) / (N_VERTICAL - 1), 1),
            "sigma_z_mm": round(sigma_z_mm(z), 3),
            "sigma_z_far_mm": round(sigma_z_mm(WORK_Z_MAX_M), 3),
            "dof": (round(near, 2), round(far, 2)),
            "fits": r["fits"], "margin_px": round(min(
                r["u_range_near"][0], IMAGE_W - r["u_range_far"][1],
                r["v_range"][0], IMAGE_H - r["v_range"][1]), 0),
        })
    use_profile(keep)
    return rows


def summary():
    """현재 캘리브레이션 값과 출처를 표로 출력한다."""
    lines = [f"캘리브레이션 데이터 (B) — 출고 시 1회 측정   "
             f"[프로파일: {ACTIVE_PROFILE} — {SPEC_PROFILES[ACTIVE_PROFILE]['label']}]",
             "-" * 78]
    rows = [
        ("f_px",    f"{CAMERA_PARAMS['f_px']} px", "f",       "f_px"),
        ("주점",     f"{CX_PX:.1f}, {CY_PX:.1f} px", "c_x,c_y", "cx_px"),
        ("기선",     f"{BASELINE_M} m",            "b",       "b_m"),
        ("DOE 발산각", f"{GRID_PARAMS['fov_deg']:.2f}°", "—",  "fov_deg"),
        ("격자선 수",  f"수직{N_VERTICAL} + 수평{N_HORIZONTAL}", "—", "n_lines"),
        ("센서 종류",  f"{SENSOR_COLOR}" + (f" + {OPTICAL_FILTER_NM}nm 필터"
                                           if OPTICAL_FILTER_NM else ""),
         "—", "sensor"),
        ("레이저 수렴각", f"{LASER_TILT_DEG}°",         "δ",       "tilt"),
        ("V선 발사각", f"{DOE_ANGLE_MODEL} {N_VERTICAL}분할", "α_i", "alpha_i"),
        ("H선 발사각", f"{DOE_ANGLE_MODEL} {N_HORIZONTAL}분할", "β_j", "beta_j"),
        ("카메라–레이저 자세", "R=I, t=(b,0,0)",     "R, t",    "R_t"),
        ("IMU–카메라 자세", "단위행렬",              "R_ic",    "R_ic"),
        ("가속도계 bias", "미구현",                  "b_a",     "b_a"),
        ("선검출 픽셀오차", f"{SIGMA_U_PX} px",       "σ_u",     "sigma_u"),
    ]
    prov = provenance()
    for name, val, sym, key in rows:
        grade, note = prov[key]
        lines.append(f"  {name:<18} {sym:<8} {val:<22} [{grade:<7}] {note}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(summary())
    print()
    print("사양 프로파일 비교  (LASER_GRID_PROFILE 환경변수로 전환)")
    print("-" * 78)
    rows = compare_profiles()
    keys = [("label", "프로파일"), ("sensor", "센서"), ("filter", "광학필터"),
            ("lines", "격자선"), ("b_mm", "기선mm"), ("sigma_u", "σ_u px"),
            ("f_px", "f_px"), ("pitch_mm", "피치@1.2m"),
            ("sigma_z_mm", "σ_Z@1.2m"), ("sigma_z_far_mm", "σ_Z@1.5m"),
            ("dof", "심도 m"), ("margin_px", "격자여유px")]
    w = max(len(str(r[k])) for _, _ in [(0, 0)] for r in rows
            for k, _t in keys) + 3
    print("  " + f"{'항목':<14}" + "".join(f"{r['name']:<{w}}" for r in rows))
    for k, title in keys:
        print("  " + f"{title:<14}" + "".join(f"{str(r[k]):<{w}}" for r in rows))
    pdf_r = next(r for r in rows if r["name"] == "pdf")
    imp_r = next(r for r in rows if r["name"] == "improved")
    print(f"  → 개선안은 PDF 원안 대비 σ_Z {pdf_r['sigma_z_mm']/imp_r['sigma_z_mm']:.2f}배, "
          f"공간 표본 밀도 {pdf_r['pitch_mm']/imp_r['pitch_mm']:.2f}배")
    print()
    print("PDF 2.2 사양표 원문 — 개선안이 지킨 것과 바꾼 것")
    print("-" * 78)
    P = SPEC_PROFILES["pdf"]
    def _cmp(name, pdf_val, now_val, note=""):
        same = str(pdf_val) == str(now_val)
        mark = "유지" if same else "변경"
        print(f"  {name:<14}{str(pdf_val):<24}{str(now_val):<24}[{mark}] {note}")
    print(f"  {'항목':<14}{'PDF 원문':<24}{'현재 프로파일':<24}")
    _cmp("해상도", f"{P['image_w']}×{P['image_h']}", f"{IMAGE_W}×{IMAGE_H}",
         "화소수만 늘림. 센서 크기·화각 동일")
    _cmp("화소 피치", f"{P['pixel_pitch_um']}µm (>=3.45)",
         f"{PIXEL_PITCH_UM}µm", "「>=3.45µm」 하한 조건에서 벗어남")
    _cmp("센서 종류", P["sensor_color"], SENSOR_COLOR, "RGB 문맥영상 상실")
    _cmp("광학필터", "없음",
         f"{OPTICAL_FILTER_NM}nm 대역통과" if OPTICAL_FILTER_NM else "없음",
         "PDF 에 없던 항목 추가")
    _cmp("기선", f"{P['baseline_m']*1000:.0f}mm", f"{BASELINE_M*1000:.0f}mm",
         "외형 210mm 내에서 확대")
    _cmp("격자선", f"V{P['n_vertical']} + H{P['n_horizontal']}",
         f"V{N_VERTICAL} + H{N_HORIZONTAL}", "깊이를 주는 V선만 증가")
    _cmp("교점 수", P["n_vertical"] * P["n_horizontal"],
         N_VERTICAL * N_HORIZONTAL, "")
    _cmp("DOE 발산각", f"{FOV_DEG:.2f}°", f"{FOV_DEG:.2f}°",
         "PDF 「120cm 936mm」 그대로")
    _cmp("레이저", f"{LASER_WAVELENGTH_NM}nm "
         f"{LASER_POWER_MW[0]}~{LASER_POWER_MW[1]}mW",
         f"{LASER_WAVELENGTH_NM}nm {LASER_POWER_MW[0]}~{LASER_POWER_MW[1]}mW",
         "출력 그대로 → 눈 안전등급 재평가 불필요")
    _cmp("렌즈", f"F{LENS_FNUMBER} 초점 {FOCUS_DISTANCE_M}m 고정",
         f"F{LENS_FNUMBER} 초점 {FOCUS_DISTANCE_M}m 고정", "초점거리는 PDF 에 없음")
    _cmp("작업거리", f"{WORK_Z_MIN_M}~{WORK_Z_MAX_M}m",
         f"{WORK_Z_MIN_M}~{WORK_Z_MAX_M}m", "PDF 권장 유지")

    print()
    print("PDF 에 없어 이 코드가 정한 값과 그 근거")
    print("-" * 78)
    print(f"  렌즈 초점거리 {LENS_FOCAL_MM}mm")
    print(f"    {'초점거리':<8}{'격자 수용':<12}{'심도(초점1.2m)':<20}{'σ_Z@1.5m':<10}판정")
    for fmm in (6.0, 8.0, 10.0, 12.0, 16.0):
        f = focal_px(fmm)
        cp = {**CAMERA_PARAMS, "f_px": f}
        r = check_consistency(cp, verbose=False)
        n, fr = depth_of_field(fmm)
        cover = (n <= WORK_Z_MIN_M and fr >= WORK_Z_MAX_M)
        sz = sigma_z_mm(WORK_Z_MAX_M, cp)
        ok = r["fits"] and cover and sz <= TARGET_SIGMA_MM
        print(f"    {fmm:<8.0f}{'들어옴' if r['fits'] else '벗어남':<12}"
              f"{f'{n:.2f} ~ {fr:.2f} m' + ('' if cover else ' (부족)'):<20}"
              f"{f'{sz:.2f}mm':<10}{'채택' if ok else '탈락'}")
    bt = find_best_tilt()
    print(f"  레이저 수렴각 {LASER_TILT_DEG}°  "
          f"(탐색 결과 {bt['tilt_deg']}°, 그때 가장자리 여유 {bt['margin_px']}px)")
    print(f"  가장자리 마진 {EDGE_MARGIN_PX:.0f}px")
    print()
    print("Isaac Sim 카메라 설정값")
    print("-" * 78)
    for k, v in isaac_camera_params().items():
        print(f"  {k:<24}{v}")
    print(f"  → f_px = {LENS_FOCAL_MM} × {IMAGE_W} / {SENSOR_W_MM:.4f} = {F_PX:.1f}")
    print()
    print("거리별 시야 · 격자 투사폭 · 깊이 노이즈 (σ_u = 0.2px, b = 150mm)")
    print("-" * 78)
    print(f"  {'거리':<7}{'카메라 시야':<22}{'격자 투사폭':<13}"
          f"{'mm/px':<9}{'격자 피치':<11}{'σ_Z':<8}")
    for z in (0.5, 1.0, 1.2, 1.5, 2.0, 3.0):
        w, h = fov_mm_at(z)
        pitch = projection_mm_at(z) / (N_VERTICAL - 1)
        print(f"  {z:<7.1f}{f'{w:.0f} × {h:.0f} mm':<22}"
              f"{f'{projection_mm_at(z):.0f} mm':<13}{w/IMAGE_W:<9.4f}"
              f"{f'{pitch:.1f} mm':<11}{sigma_z_mm(z):<8.2f}")
    print()
    print("정합성 검사 — 격자가 작업거리 전 구간에서 센서 안에 드는가")
    print("-" * 78)
    r = check_consistency()
    W, H = IMAGE_W, IMAGE_H
    m = EDGE_MARGIN_PX
    print(f"  작업거리 {r['work_z_m'][0]} ~ {r['work_z_m'][1]} m,  마진 {m:.0f}px")
    print(f"    Z={r['work_z_m'][0]}m  u = {r['u_range_near'][0]:7.1f} .. "
          f"{r['u_range_near'][1]:7.1f}   (허용 {m:.0f} .. {W-m:.0f})")
    print(f"    Z={r['work_z_m'][1]}m  u = {r['u_range_far'][0]:7.1f} .. "
          f"{r['u_range_far'][1]:7.1f}")
    print(f"    v = {r['v_range'][0]:7.1f} .. {r['v_range'][1]:7.1f}"
          f"   (허용 {m:.0f} .. {H-m:.0f})")
    print(f"  판정: {'격자 전부 센서 안' if r['fits'] else '벗어남'}")
    print(f"  이 설계로 쓸 수 있는 거리: "
          f"{r['usable_z_m'][0]} ~ {r['usable_z_m'][1]} m")
    n, fr = depth_of_field()
    print(f"  피사계심도 {n:.2f} ~ {fr:.2f} m "
          f"({'작업거리 포함' if n <= WORK_Z_MIN_M and fr >= WORK_Z_MAX_M else '작업거리 미포함'})")
