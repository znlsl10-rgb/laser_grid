#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_pipeline.py — 이미지 한 장 → 엑셀 조서 한 개
========================================================================
레이저 그리드 품질검측의 **단일 입구**다. 코랩에서 파일 몇 개만 올리면
끝까지 돌아 조서가 하나 나온다.

입력
----
  레이저 이미지   (필수)  격자가 찍힌 사진/렌더 (PNG·JPG)
  카메라 사양     (선택)  camera_params.json. 없으면 활성 프로파일을
                        이미지 해상도에 맞춰 쓴다
  IMU             (선택)  중력 방향. 없으면 **장비가 똑바로 서 있다**고
                        보고 진행한다 (ĝ = (0,1,0))
  정답값          (선택)  cast_pixels.json. 있으면 선검출 정확도와
                        깊이 오차를 표로 낸다
  장면 사진       (선택)  레이저 OFF 프레임(CAM.png). 결과 그림 배경으로만

출력 — 엑셀 파일 하나
  1.요약            입력·가정·결과 한눈에
  2.설계값          사양과 그 출처 등급
  3.선검출(1단계)    선별 검출 결과 + 정답 대조 + 대조 그림
  4.깊이검증(2단계)  화소→3D 깊이, 정답이 있으면 mm 오차표
  5.세그멘테이션     색↔부재 대응 + 그림
  6.검측결과        부재별 수직도·수평도·평활도 판정
  7~8              평활도 근거 · 요철 위치
  9.3D좌표(부재별)   **3D 산점도 (부재별 색)** + 점수·중심·크기
  10.3D좌표(점목록)  좌표 목록
  11.유의사항       이 값을 어디까지 믿을 수 있는가

코랩에서
--------
    !git clone https://github.com/znlsl10-rgb/laser_grid.git
    %cd laser_grid
    !pip install -q -r requirements.txt
    from run_pipeline import run
    res = run(image="LASER.png", truth="cast_pixels.json")   # 나머지는 선택
    print(res["xlsx"])

명령줄에서
---------
    python3 run_pipeline.py --image LASER.png \\
        [--params camera_params.json] [--imu imu.json] \\
        [--truth cast_pixels.json] [--scene CAM.png] [--out 조서.xlsx]

【정확도에 대해 — 미리 알아 둘 것】
  선검출의 한계는 알고리즘이 아니라 **입력 이미지**가 정한다. 선이
  안티에일리어싱 없이 이진(0/255)으로 그려져 있으면 선 중심이 0.5px
  격자에 갇혀, 어떤 추정기를 써도 σ = 1/√12 = 0.289px 아래로 못 내려간다.
  이 파이프라인은 그 하한을 자동으로 재서 조서에 적는다. 실측에서 검출
  σ 가 0.263px 로 이미 하한에 닿아 있었다 — 더 줄이려면 렌더에
  안티에일리어싱을 켜거나 실촬영본을 넣어야 한다(그때 σ 0.164px).
========================================================================
"""
import argparse
import json
import os as _os
import importlib.util as _ilu
import numpy as np


def _load(name):
    spec = _ilu.spec_from_file_location(
        name, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                            f"{name}.py"))
    m = _ilu.module_from_spec(spec); spec.loader.exec_module(m)
    return m


CALIB = _load("calibration")
EQ7 = _load("eq7_laser_plane")
DETECT = _load("A_선검출")
PIPE = _load("pipeline_region")
REPORT = _load("report")
PLOT3D = _load("plot_points3d")
XLS = _load("report_excel")
LC = _load("load_capture")


# =====================================================================
# 입력 해석
# =====================================================================
def read_image(path):
    from PIL import Image
    return np.asarray(Image.open(path).convert("RGB"))


def gravity_from_imu(imu=None):
    """
    IMU 입력을 조사기 좌표계 중력 단위벡터 ĝ 로 바꾼다.

    좌표계는 X=오른쪽, Y=아래, Z=정면이다. 장비가 똑바로 서서 수평을
    보면 중력은 그대로 아래, 곧 ĝ = (0, 1, 0) 이다.

    받는 형식 (아무거나 하나)
      {"gravity": [x, y, z]}      조사기 좌표계 중력 벡터. 가장 확실하다.
      {"pitch_deg": θ, "roll_deg": φ}
                                  θ = 아래로 숙인 각(+가 아래), φ = 광축
                                  둘레 회전. ĝ = R_z(−φ)·(0, cosθ, sinθ)
      {"accel": [x, y, z]}        정지한 가속도계 읽음. 비력은 중력의
                                  반대이므로 부호를 뒤집는다.
      None                        **똑바로 서 있다고 가정** → (0, 1, 0)

    왜 부호를 따지는가
      수직도는 면 법선과 ĝ 의 사잇각이다. ĝ 가 뒤집히면 각도의 부호가
      아니라 "수직이냐 수평이냐" 의 판정 자체가 뒤집힌다. 그래서 가정을
      쓸 때는 조서에 그 사실을 반드시 남긴다.

    Returns
    -------
    (g_hat (3,), 설명 문자열, 가정이었는지 bool)
    """
    if imu is None:
        return np.array([0.0, 1.0, 0.0]), "가정: 장비가 똑바로 서 있음", True
    if isinstance(imu, str):
        with open(imu, encoding="utf-8") as fp:
            imu = json.load(fp)
    if isinstance(imu, (list, tuple, np.ndarray)) and len(imu) == 3:
        imu = {"gravity": list(imu)}
    if not isinstance(imu, dict):
        raise ValueError(f"IMU 형식을 모르겠다: {type(imu)}")

    if "gravity" in imu:
        g = np.asarray(imu["gravity"], float)
        src = "IMU 중력벡터"
    elif "accel" in imu:
        g = -np.asarray(imu["accel"], float)
        src = "IMU 가속도계 (비력의 반대)"
    elif "pitch_deg" in imu or "roll_deg" in imu:
        th = np.radians(float(imu.get("pitch_deg", 0.0)))
        ph = np.radians(float(imu.get("roll_deg", 0.0)))
        g = np.array([0.0, np.cos(th), np.sin(th)])
        c, s = np.cos(-ph), np.sin(-ph)
        g = np.array([c * g[0] - s * g[1], s * g[0] + c * g[1], g[2]])
        src = (f"IMU 자세 (하향 {np.degrees(th):.2f}°, "
               f"회전 {np.degrees(ph):.2f}°)")
    else:
        raise ValueError("IMU 에 gravity / accel / pitch_deg·roll_deg "
                         "중 하나는 있어야 한다")
    n = float(np.linalg.norm(g))
    if n < 1e-9:
        raise ValueError("IMU 중력 벡터의 크기가 0 이다")
    return g / n, src, False


def _params_from_file(path, img_w, img_h):
    """
    camera_params.json → 검측용 카메라 파라미터.

    Isaac 내보내기 형식과 평평한 형식을 모두 받는다.
      Isaac : {"camera": {"f_px","cx_px","cy_px"}, "baseline_m",
               "sensor_size", "screenshot_size", "grid": {...}}
      평평   : {"f_px","cx_px","cy_px","b_m","n_v","n_h","fov_deg", ...}

    f 와 c 는 **이미지 화소 단위** 로 환산해 돌려준다. 삼각측량식
    Z = f·b/(f·tanα − Δu) 는 f 와 Δu 가 같은 배율로 함께 줄면 값이
    변하지 않으므로, 축소된 화면 캡처로도 거리는 정확히 나온다.
    줄어드는 것은 정밀도이지 정확도가 아니다.
    """
    with open(path, encoding="utf-8") as fp:
        d = json.load(fp)
    cam = d.get("camera", d)
    f = float(cam["f_px"]); cx = float(cam["cx_px"]); cy = float(cam["cy_px"])
    b = float(d.get("baseline_m", d.get("b_m", CALIB.BASELINE_M)))
    sensor = d.get("sensor_size") or [cam.get("sensor_W", img_w),
                                      cam.get("sensor_H", img_h)]
    su = float(sensor[0]) / float(img_w)
    sv = float(sensor[1]) / float(img_h)
    grid = d.get("grid", d)
    cp = {"f_px": f / su, "b_m": b, "cx_px": cx / su, "cy_px": cy / sv,
          "resolution": [img_w, img_h], "image_w": img_w, "image_h": img_h,
          "n_v": int(grid.get("n_vertical", grid.get("n_v", CALIB.N_VERTICAL))),
          "n_h": int(grid.get("n_horizontal",
                              grid.get("n_h", CALIB.N_HORIZONTAL))),
          "fov_h_deg": float(grid.get("fov_deg", CALIB.FOV_DEG)),
          "fov_v_deg": float(grid.get("fov_deg", CALIB.FOV_DEG)),
          "laser_tilt_deg": float(grid.get("laser_tilt_deg",
                                           CALIB.LASER_TILT_DEG)),
          "laser_roll_deg": float(grid.get("laser_roll_deg",
                                           CALIB.LASER_ROLL_DEG))}
    cp["sensor_w"], cp["sensor_h"] = float(sensor[0]), float(sensor[1])
    return cp, {"출처": _os.path.basename(path),
                "센서→이미지 배율": (round(su, 4), round(sv, 4))}


def _params_from_profile(img_w, img_h):
    """사양 프로파일을 이미지 해상도에 맞춰 낸다."""
    cp = CALIB.scale_to_resolution(img_w)
    cp = {**cp, "image_w": img_w, "image_h": img_h,
          "n_v": CALIB.N_VERTICAL, "n_h": CALIB.N_HORIZONTAL,
          "fov_h_deg": CALIB.FOV_DEG, "fov_v_deg": CALIB.FOV_DEG,
          "laser_tilt_deg": CALIB.LASER_TILT_DEG,
          "laser_roll_deg": CALIB.LASER_ROLL_DEG}
    cp["cy_px"] = img_h / 2.0
    cp["resolution"] = [img_w, img_h]
    return cp, {"출처": f"사양 프로파일 '{CALIB.ACTIVE_PROFILE}'",
                "주의": "실측 캘리브레이션이 아니다. 사양표 값이므로 "
                        "절대 거리에 계통 오차가 남을 수 있다"}


def quantization_floor(rgb):
    """
    입력 이미지가 이진(안티에일리어싱 없음)인지 보고, 그렇다면 선 중심
    추정의 하한 σ 를 돌려준다.

    이진으로 그려진 선은 켜진 화소가 전부 같은 값이라, 중심을 어떻게
    구하든 결과가 0.5px 격자 위에만 떨어진다. 그 격자에 균등분포한
    오차의 표준편차가 1/√12 = 0.289px 다. **알고리즘으로 못 내려가는
    바닥** 이므로, 검출 σ 가 이 값 근처면 그 이상은 입력을 고쳐야 한다.
    """
    a = np.asarray(rgb, float)
    g = a[:, :, 1] - 0.5 * (a[:, :, 0] + a[:, :, 2])
    mx = float(g.max())
    if mx < 1.0:
        return {"binary": False, "reason": "레이저 신호를 못 찾음"}
    lit = g[g > mx * 0.15]
    frac = float((lit > mx * 0.9).mean()) if len(lit) else 0.0
    binary = bool(frac > 0.9)
    out = {"binary": binary, "saturated_frac": round(frac, 4)}
    if binary:
        out["sigma_floor_px"] = round(1.0 / np.sqrt(12.0), 4)
        out["note"] = ("켜진 화소가 전부 같은 값이다(안티에일리어싱 없음). "
                       "선 중심이 0.5px 격자에 갇혀 σ 0.289px 아래로는 "
                       "원리적으로 못 내려간다. 렌더에 AA 를 켜거나 "
                       "실촬영본을 쓰면 개선된다.")
        # 선폭도 같이 잰다 — 폭이 굵을수록 켜진 화소가 많아 중심이
        # 안정되지만, 이진에서는 폭이 커져도 격자 간격은 그대로다.
        row = g[g.shape[0] // 2]
        on = row > mx * 0.5
        d = np.diff(on.astype(int))
        st, en = np.where(d == 1)[0], np.where(d == -1)[0]
        n = min(len(st), len(en))
        if n:
            out["line_width_px"] = float(np.median(en[:n] - st[:n]))
    else:
        out["note"] = "밝기 기울기가 있다 — 서브픽셀 정보가 살아 있다."
    return out


def improvement_forecast(sigma_u_img_px, cp, z_m, sensor_w=None, binary=True):
    """
    "이 오차를 어떻게 줄이나" 에 대한 **숫자로 된 답**.

    깊이 잡음은 σ_Z = σ_u · Z²/(f·b) 다. σ_u 는 센서 화소 단위여야 하므로,
    화면을 축소해 저장했으면 그 배율만큼 손해를 본다. 통제 실험(합성 격자,
    같은 장면·같은 추정기)에서 잰 값:

        1269×1063 이진   σ_u 0.577 센서px → 5.80mm      ← 지금 입력
        1269×1063 AA     σ_u 0.398        → 4.00mm
        2448×2048 이진   σ_u 0.295        → 2.97mm
        2448×2048 AA     σ_u 0.204        → 2.05mm

    즉 **원해상도로 저장만 해도 약 2배**, 거기에 안티에일리어싱을 켜면
    약 2.8배 좋아진다. 둘 다 촬영·렌더 설정이지 알고리즘이 아니다.
    """
    f = float(cp["f_px"]); b = float(cp["b_m"])
    z = float(z_m)
    mm = lambda su_img: su_img * z * z / (f * b) * 1000.0
    now = mm(sigma_u_img_px)
    out = {"현재": round(now, 2)}
    scale = 1.0
    if sensor_w and cp.get("image_w"):
        scale = float(sensor_w) / float(cp["image_w"])
    if scale > 1.02:
        out["원해상도로 저장"] = round(now / scale, 2)
        out["해상도 배율"] = round(scale, 3)
    if binary:
        aa = 0.204 / 0.295          # 통제 실험에서 잰 AA 이득
        base = out.get("원해상도로 저장", now)
        out["+ 안티에일리어싱"] = round(base * aa, 2)
    return out


def _median_depth_of(lines_xyz):
    Z = np.concatenate([np.asarray(v, float)[:, 2] for v in lines_xyz.values()])
    return float(np.median(Z))


def _line_planes_from_truth(truth, cp):
    """
    정답 데이터가 있으면 레이저 평면을 **데이터에서** 맞춘다.

    발사각 하나로 평면을 세우면 "V평면은 Y축을 품는다" 같은 가정이 따라
    들어온다. 정답 점은 정의상 그 평면 위에 있으므로 평면을 직접 맞추면
    가정 없이 법선이 나온다 (실측 잔차 0.0006~0.33mm).

    xyz_world 가 없으면 화소만으로는 평면을 세울 수 없으므로 None.
    """
    if not truth:
        return None
    first = next(iter(truth.values()))
    if not first.get("points") or "xyz_world" not in first["points"][0]:
        return None
    return None          # 자세를 모르면 세계좌표를 조사기좌표로 못 옮긴다


def estimate_grid_pose(rgb, cp, line_angles, z_lo=0.4, z_hi=8.0):
    """
    이미지에서 격자를 직접 읽어 **측정거리와 화소 규약** 을 추정한다.

    왜 필요한가
    ----------
    기본 경로에는 정답도 자세도 없다. 그런데 예측식
        u_i = f·tan(α_i) − f·b/Z + c_x
    의 두 번째 항(기선 시차)이 거리 Z 에 따라 100px 넘게 움직이고, 게다가
    내보내기에 따라 화소가 180° 돌아 있기도 하다. 둘 중 하나만 틀려도
    추적 밴드가 실제 선을 비껴가 검출이 통째로 어긋난다(실측: 이 표본을
    가정값 1.2m·정방향으로 돌리면 깊이가 1.55m 대신 1.20m 로 나왔다).

    어떻게 푸는가
    ------------
    격자를 예측 없이 읽는다(축 투영 + 임계). 그 다음 V선 위치 {u_j} 와
    발사각 {α_j} 를 짝지어 Z 를 역산한다.

        d_j = f·tan(α_j) + c_x − u_j = f·b/Z      →   Z = f·b / median(d_j)

    Z 가 물리적으로 말이 되고(0.4~8m) 선별 편차가 작으면 그 규약이 맞는
    것이다. 정방향과 180° 반전 두 가지를 다 풀어 보고 잔차가 작은 쪽을
    고른다. 시차 항이 부호를 가지므로 두 규약이 대칭이 아니고, 그래서
    구분이 된다.

    Returns
    -------
    dict — z_est_m, flipped, resid_m, n_matched, ok
    """
    INS = _load("inspect_png")
    u_img, _ = INS.read_grid_from_image(rgb)
    f, b = float(cp["f_px"]), float(cp["b_m"])
    cx, cy = float(cp["cx_px"]), float(cp["cy_px"])
    alphas = np.array(sorted(
        v["angle_rad"] for k, v in line_angles.items() if k.startswith("V")))
    best = None
    for flipped in (False, True):
        u = np.sort(2 * cx - u_img if flipped else u_img)
        if len(u) < 5:
            continue
        # 개수가 다를 수 있다(끝선이 화면 밖). 순서대로 겹치는 만큼만 본다.
        n = min(len(u), len(alphas))
        for off in range(0, max(1, len(alphas) - n + 1)):
            a = alphas[off:off + n]
            d = f * np.tan(a) + cx - u[:n]
            if np.median(d) <= 0:
                continue
            z = f * b / np.median(d)
            if not (z_lo <= z <= z_hi):
                continue
            zz = f * b / d[d > 0]
            if len(zz) < 5:
                continue
            resid = float(np.std(zz))
            cand = {"z_est_m": round(float(z), 4), "flipped": flipped,
                    "resid_m": round(resid, 4), "n_matched": int(len(zz)),
                    "offset": off}
            if best is None or resid < best["resid_m"]:
                best = cand
    if best is None:
        return {"ok": False, "reason": "격자에서 거리를 풀지 못함",
                "n_lines_read": int(len(u_img))}
    best["ok"] = True
    best["n_lines_read"] = int(len(u_img))
    return best


def _line_angles(cp, truth_path=None):
    """검측에 쓸 선별 발사각·평면 법선."""
    return CALIB.make_line_angles(
        n_v=cp["n_v"], n_h=cp["n_h"], fov_deg=cp["fov_h_deg"],
        laser_tilt_deg=cp.get("laser_tilt_deg", 0.0),
        laser_roll_deg=cp.get("laser_roll_deg", 0.0))


# =====================================================================
# 본체
# =====================================================================
def _is_full_export(params, truth):
    """
    Isaac 내보내기 한 벌(사양 + 정답 3D + 장비 자세)이 다 있는가.

    다 있으면 발사각·기선 부호·자세를 **데이터에서** 복원할 수 있다.
    이름표(angle_deg)는 내보내기마다 규약이 달라 그대로 믿으면 안 된다 —
    이 표본만 해도 화소가 180° 돌아 있고 H선 번호가 코드와 반대라,
    이름표로 예측을 세우면 격자가 화면 반대편을 가리킨다.
    """
    if not (params and truth):
        return False
    try:
        with open(params, encoding="utf-8") as fp:
            d = json.load(fp)
        with open(truth, encoding="utf-8") as fp:
            t = json.load(fp)
    except Exception:
        return False
    if "rig_transform" not in d:
        return False
    first = next(iter(t.values()), None)
    return bool(first and first.get("points")
                and "xyz_world" in first["points"][0])


def run(image, params=None, imu=None, truth=None, scene_image=None,
        out=None, profile=None, backend="geom", standoff_m=None,
        site=None, pc_stride=20, smooth=0, verbose=True):
    """
    이미지 한 장을 끝까지 돌려 엑셀 조서 하나를 만든다.

    두 갈래로 갈린다 — 가진 정보에 따라 정확도가 다르기 때문이다.

      [정밀 경로]  사양 + 정답 3D + 장비 자세가 모두 있을 때.
                  발사각·기선 부호·카메라 자세를 데이터에서 복원하므로
                  가정이 거의 없다. 선검출 정확도와 깊이 오차를 mm 로 낸다.
      [기본 경로]  이미지만, 또는 사양까지만 있을 때.
                  발사각은 사양 프로파일의 이름값을 쓰고, 중력은 IMU 가
                  없으면 똑바로 서 있다고 가정한다. 무엇을 가정했는지는
                  조서 11번 시트에 그대로 적힌다.

    Parameters
    ----------
    image : str            레이저 격자 이미지 (필수)
    params : str | None    camera_params.json
    imu : str | dict | None  IMU. 없으면 똑바로 서 있다고 가정
    truth : str | None     cast_pixels.json (정답 화소·3D)
    scene_image : str | None  레이저 OFF 장면 사진 (배경용)
    out : str | None       엑셀 경로. 기본 <이미지이름>_품질검측조서.xlsx
    standoff_m : float | None  대표 측정거리. 없으면 사양 기본값

    Returns
    -------
    dict — xlsx, result, detection, depth, images
    """
    def say(*a):
        if verbose:
            print(*a)

    if profile:
        CALIB.use_profile(profile)

    rgb = read_image(image)
    H, W = rgb.shape[:2]
    name = _os.path.splitext(_os.path.basename(image))[0]
    # out 은 조서 **파일** 경로다. 그런데 코랩에서는 폴더를 주기 쉽고,
    # 그러면 확장자 없는 파일이 하나 생겨 openpyxl 도 pandas 도 다시 못
    # 연다(엑셀도 못 연다). 조용히 그런 파일을 남기지 말고 바로잡는다.
    fname = f"{name}_품질검측조서.xlsx"
    if not out:
        out = _os.path.join(_os.path.dirname(_os.path.abspath(image)) or ".",
                            fname)
    elif _os.path.isdir(out) or out.endswith(("/", _os.sep)):
        out = _os.path.join(out, fname)
    elif _os.path.splitext(out)[1].lower() not in (".xlsx", ".xlsm"):
        out = out + ".xlsx"
    out_dir = _os.path.dirname(_os.path.abspath(out)) or "."
    _os.makedirs(out_dir, exist_ok=True)

    qz = quantization_floor(rgb)
    say("=" * 70)
    say(f"입력 이미지: {image}  ({W}×{H})")
    say("=" * 70)
    if qz.get("binary"):
        say(f"  [입력 품질]   이진 렌더 — 선검출 σ 하한 "
            f"{qz['sigma_floor_px']}px (알고리즘으로 못 넘는 바닥)")

    full = _is_full_export(params, truth)
    say(f"  경로          {'정밀 (데이터에서 발사각·자세 복원)' if full else '기본 (사양값·가정 사용)'}")

    if full:
        return _run_full(image, params, truth, scene_image, imu, out, out_dir,
                         name, rgb, qz, backend, site, pc_stride, say, smooth)
    return _run_plain(image, params, truth, scene_image, imu, out, out_dir,
                      name, rgb, qz, backend, site, pc_stride, standoff_m, say,
                      smooth)


def _run_full(image, params, truth, scene_image, imu, out, out_dir, name,
              rgb, qz, backend, site, pc_stride, say, smooth=0):
    """
    정밀 경로 — load_capture 의 검증된 경로를 그대로 탄다.

    여기서 코드를 새로 짜지 않는 이유는 정확도가 세부에 달려 있기 때문이다.
    화면→센서 배율, 180° 뒤집힌 화소 규약, 부호 있는 기선, Kabsch 자세
    복원 — 하나라도 어긋나면 깊이가 미터 단위로 틀어진다. 이미 맞춰 둔
    경로를 재사용하는 편이 안전하다.
    """
    cap = LC.load_folder(None, stride=1, params_path=params, truth_path=truth)
    cp = cap["camera_params"]
    for k, v in cap["diag"].items():
        say(f"  {k:<24}{v}")

    g_hat, g_src, g_assumed = cap["g_hat"], "데이터에서 복원한 카메라 자세", False
    if imu is not None:
        g_hat, g_src, g_assumed = gravity_from_imu(imu)
        say(f"  중력 ĝ (IMU 우선)        {np.round(g_hat, 4).tolist()}  ({g_src})")

    say("\n  [1단계] 선검출")
    det_eval = LC.evaluate_line_detection(None, cap, image_path=image)
    if det_eval and not det_eval.get("error"):
        say(f"    계통 {det_eval['err_bias_px']:+.3f}px  "
            f"σ {det_eval['err_noise_px']:.3f}px"
            + (f"  (양자화 하한 {qz['sigma_floor_px']}px)"
               if qz.get("binary") else ""))
        for _, d in (det_eval.get("families") or {}).items():
            say(f"    {d['이름']:<12}검출 {d['검출']}/{d['선 수']}"
                f"   선내 {d['선내잡음_px']} / 선간 {d['선간편차_px']}"
                f" → 통합 {d['통합오차_px']} px"
                f"   깊이가능 {d['깊이가능']}/{d['선 수']}")
    else:
        raise RuntimeError(f"선검출 실패: {(det_eval or {}).get('error')}")

    say("\n  [2단계] 3D 복원 (검출 화소 → 깊이)")
    lines_in = LC.detected_lines_sensor(cap, det_eval)
    if smooth:
        half = float(qz.get("sigma_floor_px", 0.29)) * np.sqrt(12.0) / 2.0
        half *= det_eval["scale_to_sensor"]
        lines_in = DETECT.smooth_along_lines(lines_in, half, win=int(smooth))
        say(f"    선따라 평활  창 {int(smooth)}점, 보정 한계 ±{half:.2f}px")
    lines_xyz, lines_uv, tri = PIPE.triangulate_lines(
        lines_in, cap["line_angles"], cp)
    n3d = sum(len(v) for v in lines_xyz.values())
    nf = tri["n_by_family"]
    say(f"    삼각측량 점 {n3d:,}   (V {nf.get('V', 0):,} + H {nf.get('H', 0):,})")
    depth = LC.verify_depth(lines_uv, lines_xyz, cap)
    if depth:
        depth["개선예측_mm"] = improvement_forecast(
            float(det_eval["err_noise_px"]), {**cp, "image_w": rgb.shape[1],
                                              "f_px": det_eval["f_px_image"]},
            _median_depth_of(lines_xyz), sensor_w=cp["resolution"][0],
            binary=qz.get("binary", False))
        say(f"    깊이 오차   치우침 {depth['z_bias_mm']:+.2f}mm / "
            f"산포 {depth['z_noise_mm']:.2f}mm / RMS {depth['z_rms_mm']:.2f}mm"
            f"  ({depth['n_points']:,}점)")

    su = float(det_eval["err_noise_px"]) * det_eval["scale_to_sensor"]
    su_src = "이 이미지에서 실측"
    say(f"    σ_u        {su:.3f} px  ({su_src})")

    say("\n  [3단계] 영역분할 → 수직도·수평도·평활도")
    res = PIPE.inspect_image(lines_uv, lines_xyz, cp, g_hat,
                             seg_backend=backend, sigma_u_px=su,
                             line_gain=tri["line_gain"],
                             aux_lines_uv=tri.get("skipped_uv"))
    res["triangulation"] = tri
    res["pixel_source"] = "detected"
    res["depth_check"] = depth

    # 한 줄만 걸린 부재를 가림 그림자로 되살린다. 검측 좌표는 센서 기준·
    # 표준 규약이고 이미지는 화면 기준·원본 규약이라 변환을 넘긴다.
    su = det_eval["scale_to_sensor"]; sv = det_eval["scale_to_sensor_v"]
    flip = bool(cap["diag"]["uv 180° 뒤집힘"])
    cxs, cys = cp["cx_px"], cp["cy_px"]

    def _to_img(a):
        a = np.asarray(a, float)
        if flip:
            a = np.stack([2 * cxs - a[..., 0], 2 * cys - a[..., 1]], axis=-1)
        return a / np.array([su, sv])

    cp_img = {"f_px": cp["f_px"] / su, "b_m": cap.get("b_raw", cp["b_m"]),
              "cx_px": cxs / su, "cy_px": cys / sv}
    n_res = PIPE.resolve_single_plane_members(res, rgb, cp_img, g_hat,
                                              to_image=_to_img)
    if n_res:
        say(f"    가림 그림자로 {n_res}개 부재의 옆 기울기 복원")
    say(); say(PIPE.format_report(res))

    # 배경은 원본 규약, 검측 좌표는 표준 규약이다. 배경을 돌리는 대신
    # 그릴 좌표를 되돌린다 (리샘플링 없음).
    flip = bool(cap["diag"]["uv 180° 뒤집힘"])
    cxp, cyp = cp["cx_px"], cp["cy_px"]
    uv_tf = ((lambda a: np.stack([2 * cxp - np.asarray(a, float)[..., 0],
                                  2 * cyp - np.asarray(a, float)[..., 1]],
                                 axis=-1)) if flip else None)
    base = _base_for_overlay(scene_image, image, tuple(cp["resolution"]))
    imgs = _outputs(out_dir, name, res, g_hat, base,
                    (cp["resolution"][1], cp["resolution"][0]), uv_tf,
                    pc_stride, say)
    try:
        imgs["선검출대조"] = LC.save_detection_overlay(
            None, cap, det_eval,
            _os.path.join(out_dir, f"{name}_선검출대조.png"),
            image_path=image)
    except Exception as e:
        say(f"    [경고] 대조 그림 실패: {e}")

    meta = {"현장": site or "-", "입력 이미지": _os.path.basename(image),
            "이미지 크기": f"{rgb.shape[1]}×{rgb.shape[0]}",
            "경로": "정밀 (발사각·자세를 데이터에서 복원)",
            "중력 기준": g_src,
            "정답 데이터": _os.path.basename(truth),
            "σ_u (px)": f"{su:.3f}  ({su_src})"}
    meta.update({k: str(v) for k, v in cap["diag"].items()})
    caveats = _caveats(g_assumed, params, True, qz, su, depth, backend)
    xlsx = XLS.save_excel(out, res, meta=meta,
                          seg_image_path=imgs.get("세그멘테이션"),
                          extra_caveats=caveats, detection=det_eval,
                          g_hat=g_hat, pointcloud_image=imgs.get("3D점군"),
                          detection_image=imgs.get("선검출대조"),
                          depth_check=depth, pixel_source="detected",
                          pc_stride=max(1, int(pc_stride)))
    say(); say(f"  엑셀 조서:  {xlsx}")
    return {"xlsx": xlsx, "result": res, "detection": det_eval,
            "depth": depth, "sigma_u_px": su, "g_hat": g_hat, "images": imgs}


def _run_plain(image, params, truth, scene_image, imu, out, out_dir, name,
               rgb, qz, backend, site, pc_stride, standoff_m, say, smooth=0):
    """
    기본 경로 — 이미지(+사양)만 있을 때.

    발사각은 사양 프로파일의 이름값을 쓰고, 중력은 IMU 가 없으면 똑바로
    서 있다고 가정한다. 무엇을 가정했는지는 조서에 그대로 남긴다.
    """
    H, W = rgb.shape[:2]
    truth_data = None
    if truth:
        with open(truth, encoding="utf-8") as fp:
            truth_data = json.load(fp)
        say(f"  정답 데이터    {_os.path.basename(truth)} "
            f"— 선 {len(truth_data)}개")
    if params:
        cp, p_meta = _params_from_file(params, W, H)
    else:
        cp, p_meta = _params_from_profile(W, H)
    cp["standoff_z"] = float(standoff_m or 1.2)
    say(f"  카메라 사양   {p_meta['출처']}")
    say(f"    f={cp['f_px']:.1f}px  c=({cp['cx_px']:.1f},{cp['cy_px']:.1f})  "
        f"b={cp['b_m']*1000:.0f}mm  격자 V{cp['n_v']}+H{cp['n_h']}")

    g_hat, g_src, g_assumed = gravity_from_imu(imu)
    say(f"  중력 ĝ        {np.round(g_hat, 4).tolist()}   ({g_src})")

    line_angles = _line_angles(cp)
    for _, d in EQ7.family_summary(EQ7.line_planes(line_angles)).items():
        say(f"    {d['이름']:<12}{d['선 수']}개 중 깊이가능 {d['깊이가능']}개"
            + (f"  (이득 g≈{d['이득_중앙']})" if d["이득_중앙"] else "  (g=∞)"))

    # ── 거리·화소 규약을 이미지에서 먼저 푼다 ──
    # 추적 밴드의 중심은 f·b/Z 만큼 움직이고, 내보내기에 따라 화소가 180°
    # 돌아 있기도 하다. 둘 중 하나만 틀려도 밴드가 실제 선을 비껴간다.
    pose = estimate_grid_pose(rgb, cp, line_angles)
    flipped = False
    if pose.get("ok"):
        flipped = bool(pose["flipped"])
        if standoff_m is None:
            cp["standoff_z"] = pose["z_est_m"]
        say(f"  격자에서 복원   거리 {pose['z_est_m']:.3f} m  "
            f"(선 {pose['n_matched']}개, 편차 {pose['resid_m']*1000:.0f}mm)"
            + ("   화소 180° 뒤집힘" if flipped else ""))
        # 깊이 구간을 넉넉히 잡아 밴드가 앞뒤 면을 다 덮게 한다
        cp["z_range"] = [pose["z_est_m"] * 0.7, pose["z_est_m"] * 1.4]
    else:
        say(f"  [경고] 격자에서 거리를 못 풀었다 ({pose.get('reason')}). "
            f"가정값 {cp['standoff_z']}m 로 진행한다.")

    # 뒤집힌 캡처에서는 예측을 원본 규약으로 세운다. 회전 하나에 발사각과
    # 기선이 함께 뒤집힌다 — 각도만 뒤집고 기선을 그대로 두면 예측이
    # 2·f·b/Z 만큼(이 표본에서 160px) 어긋난다.
    la_det, cp_det = line_angles, cp
    if flipped:
        la_det = {lid: {**v, "angle_rad": -v["angle_rad"],
                        "normal": EQ7.plane_normal(
                            v["fixed"], -v["angle_rad"],
                            tilt_rad=np.radians(cp.get("laser_tilt_deg", 0.0)),
                            roll_rad=np.radians(cp.get("laser_roll_deg", 0.0))
                        ).tolist()}
                  for lid, v in line_angles.items()}
        cp_det = {**cp, "b_m": -cp["b_m"]}

    say("\n  [1단계] 선검출")
    detected = DETECT.detect(rgb, {}, la_det, cp_det, multi_surface=True)
    say(f"    검출 선 {sum(1 for v in detected.values() if len(v) >= 10)}"
        f"/{len(line_angles)}   점 {sum(len(v) for v in detected.values()):,}")

    # 검출 화소를 표준 규약으로 되돌린다(이미지가 아니라 좌표를 돌린다 —
    # 주점이 이미지 중심과 다르면 리샘플링이 계통 오차를 남긴다).
    uv_tf = None
    detected_raw = detected
    if flipped:
        cxp, cyp = cp["cx_px"], cp["cy_px"]
        detected = {lid: np.stack([2 * cxp - np.asarray(d, float)[:, 0],
                                   2 * cyp - np.asarray(d, float)[:, 1]],
                                  axis=1)
                    for lid, d in detected.items() if len(d)}
        detected_raw = detected
        uv_tf = (lambda a: np.stack(
            [2 * cxp - np.asarray(a, float)[..., 0],
             2 * cyp - np.asarray(a, float)[..., 1]], axis=-1))

    say("\n  [2단계] 3D 복원 (검출 화소 → 깊이)")
    if smooth:
        half = float(qz.get("sigma_floor_px", 0.29)) * np.sqrt(12.0) / 2.0
        detected = DETECT.smooth_along_lines(detected, half, win=int(smooth))
        say(f"    선따라 평활  창 {int(smooth)}점, 보정 한계 ±{half:.2f}px")
    lines_xyz, lines_uv, tri = PIPE.triangulate_lines(detected, line_angles, cp)
    n3d = sum(len(v) for v in lines_xyz.values())
    if n3d == 0:
        raise RuntimeError("삼각측량된 점이 없다. --params 사양이나 "
                           "--standoff 측정거리를 확인할 것.")
    nf = tri["n_by_family"]
    say(f"    삼각측량 점 {n3d:,}   (V {nf.get('V', 0):,} + H {nf.get('H', 0):,})")
    Z = np.concatenate([np.asarray(v, float)[:, 2] for v in lines_xyz.values()])
    say(f"    깊이 범위   {Z.min():.3f} ~ {Z.max():.3f} m  "
        f"(중앙 {np.median(Z):.3f} m)")

    su = (float(qz["sigma_floor_px"]) if qz.get("binary")
          else CALIB.SIGMA_U_PX)
    su_src = ("이진 렌더의 양자화 하한" if qz.get("binary") else "설계 가정")
    say(f"    σ_u        {su:.3f} px  ({su_src})")

    # 정답이 있으면 사양이 없어도 정확도는 잴 수 있다
    det_eval = depth = None
    truth_skip = None
    if truth_data:
        sw, sh, truth_skip = truth_frame(truth_data, cp, W, H)
        cp["truth_sensor_w"], cp["truth_sensor_h"] = sw, sh
    if truth_data and truth_skip:
        say(f"    [정답 대조 생략] {truth_skip}")
    elif truth_data:
        det_eval = _compare_to_truth(detected_raw, truth_data, cp, rgb, qz,
                                     flipped)
        if det_eval and not det_eval.get("error"):
            say(f"    정답 대조   계통 {det_eval['err_bias_px']:+.3f}px  "
                f"σ {det_eval['err_noise_px']:.3f}px"
                + (f"  (양자화 하한 {qz['sigma_floor_px']}px)"
                   if qz.get("binary") else ""))
            su = float(det_eval["err_noise_px"])
            su_src = "이 이미지에서 실측"
        depth = _depth_vs_truth(lines_uv, lines_xyz, truth_data, cp, flipped)
        if depth:
            depth["개선예측_mm"] = improvement_forecast(
                su, cp, _median_depth_of(lines_xyz),
                sensor_w=cp.get("sensor_w"), binary=qz.get("binary", False))
            say(f"    깊이 오차   치우침 {depth['z_bias_mm']:+.2f}mm / "
                f"산포 {depth['z_noise_mm']:.2f}mm / "
                f"RMS {depth['z_rms_mm']:.2f}mm  ({depth['n_points']:,}점)")

    say("\n  [3단계] 영역분할 → 수직도·수평도·평활도")
    scene_rgb = read_image(scene_image) if scene_image else None
    res = PIPE.inspect_image(lines_uv, lines_xyz, cp, g_hat,
                             rgb_off=scene_rgb, seg_backend=backend,
                             sigma_u_px=su, line_gain=tri["line_gain"],
                             aux_lines_uv=tri.get("skipped_uv"))
    res["triangulation"] = tri
    res["pixel_source"] = "detected"
    res["depth_check"] = depth
    say(); say(PIPE.format_report(res))

    base = _base_for_overlay(scene_image, image, (W, H))
    imgs = _outputs(out_dir, name, res, g_hat, base, (H, W), uv_tf,
                    pc_stride, say)

    meta = {"현장": site or "-", "입력 이미지": _os.path.basename(image),
            "이미지 크기": f"{W}×{H}",
            "경로": "기본 (사양값·가정 사용)",
            "격자에서 복원한 거리": (f"{pose['z_est_m']:.3f} m "
                             f"(선 {pose['n_matched']}개, "
                             f"편차 {pose['resid_m']*1000:.0f}mm)"
                             if pose.get("ok") else "복원 실패"),
            "화소 180° 뒤집힘": str(flipped),
            "카메라 사양 출처": p_meta["출처"],
            "중력 기준": g_src,
            "IMU": ("없음 — 똑바로 서 있다고 가정" if g_assumed else "지정됨"),
            "정답 데이터": "없음",
            "σ_u (px)": f"{su:.3f}  ({su_src})"}
    meta.update({k: str(v) for k, v in p_meta.items() if k != "출처"})
    if det_eval and not det_eval.get("error"):
        try:
            imgs["선검출대조"] = _overlay(
                rgb, detected_raw, truth_data, cp,
                _os.path.join(out_dir, f"{name}_선검출대조.png"),
                flipped=flipped, uv_tf=uv_tf)
        except Exception as e:
            say(f"    [경고] 대조 그림 실패: {e}")
    meta["정답 데이터"] = (_os.path.basename(truth) if truth else "없음")
    meta["σ_u (px)"] = f"{su:.3f}  ({su_src})"
    if truth_skip:
        caveats_extra = [truth_skip]
    else:
        caveats_extra = []
    caveats = (_caveats(g_assumed, params, det_eval, qz, su, depth, backend)
               + caveats_extra)
    xlsx = XLS.save_excel(out, res, meta=meta,
                          seg_image_path=imgs.get("세그멘테이션"),
                          extra_caveats=caveats, detection=det_eval,
                          g_hat=g_hat, pointcloud_image=imgs.get("3D점군"),
                          detection_image=imgs.get("선검출대조"),
                          depth_check=depth, pixel_source="detected",
                          pc_stride=max(1, int(pc_stride)))
    say(); say(f"  엑셀 조서:  {xlsx}")
    return {"xlsx": xlsx, "result": res, "detection": det_eval,
            "depth": depth, "sigma_u_px": su, "g_hat": g_hat, "images": imgs}


def _outputs(out_dir, name, res, g_hat, base, shape, uv_tf, pc_stride, say):
    """세그멘테이션·3D 점군·좌표 CSV 를 낸다."""
    imgs = {}
    try:
        imgs["세그멘테이션"] = REPORT.save_segmentation(
            _os.path.join(out_dir, f"{name}_세그멘테이션.png"), res,
            base_image=base, shape=shape, uv_transform=uv_tf)
    except Exception as e:
        say(f"    [경고] 세그멘테이션 그림 실패: {e}")
    try:
        # 축척·눈금이 있는 matplotlib 판을 먼저 시도한다. 등척 3D 에 실제
        # 미터 눈금이 붙어 있어야 "이 벽이 정말 수직인가" 를 눈으로 잰다.
        # matplotlib 이 없으면 의존성 없는 PIL 판으로 내려간다.
        imgs["3D점군"] = PLOT3D.save_pointcloud_mpl(
            _os.path.join(out_dir, f"{name}_3D점군.png"), res, g_hat,
            title=f"3D 점군 — {name}")
        if not imgs["3D점군"]:
            imgs["3D점군"] = REPORT.save_pointcloud_3d(
                _os.path.join(out_dir, f"{name}_3D점군.png"), res, g_hat,
                title=f"3D 점군 — {name}")
        imgs["3D좌표csv"] = REPORT.save_pointcloud_csv(
            _os.path.join(out_dir, f"{name}_3D좌표.csv"), res,
            g_hat=g_hat, stride=max(1, int(pc_stride)))
    except Exception as e:
        say(f"    [경고] 3D 산출 실패: {e}")
    return imgs


# =====================================================================
# 정답 대조
# =====================================================================
def truth_frame(truth, cp, img_w, img_h):
    """
    정답 화소가 어느 해상도로 적혀 있는지 정한다.

    이것을 추측으로 때우면 안 된다. 배율이 1% 만 틀려도 화면 가장자리에서
    10px 이 어긋나고, 그러면 "선검출 오차" 로 보고되는 값이 실은 좌표계
    차이가 된다(실측: 잘못 추측했을 때 σ 가 0.26px → 2.68px 로 뻥튀기).

    그래서 근거가 있을 때만 환산한다.
      · 사양 파일에 sensor_size 가 있으면 그 값 (확실)
      · 정답 좌표가 이미 이미지 안에 들어오면 그대로 (환산 불필요)
      · 둘 다 아니면 **비교를 포기하고 이유를 남긴다**

    Returns
    -------
    (sensor_w, sensor_h, 사유문자열 | None)
    """
    mu = mv = 0.0
    for ln in truth.values():
        uv = np.array([p["uv"] for p in ln["points"]], float)
        mu = max(mu, float(uv[:, 0].max())); mv = max(mv, float(uv[:, 1].max()))
    if cp.get("sensor_w"):
        return float(cp["sensor_w"]), float(cp["sensor_h"]), None
    if mu <= img_w * 1.02 and mv <= img_h * 1.02:
        return 0.0, 0.0, None                     # 환산 불필요
    return 0.0, 0.0, (
        f"정답 화소가 이미지({img_w}×{img_h})보다 큰 좌표계"
        f"(최대 {mu:.0f}×{mv:.0f})로 적혀 있는데, 그 해상도를 알려 줄 "
        f"사양 파일이 없다. --params 로 sensor_size 를 주면 대조할 수 있다.")


def _truth_uv(ln, cp, img_w, img_h, flipped=False):
    """
    정답 선의 화소를 **검측과 같은 좌표계** 로 옮긴다.

    두 가지를 맞춘다.
      · 센서 해상도로 적혀 있으면 이미지 해상도로 줄인다
      · 캡처가 180° 돌아 있으면 주점 기준으로 되돌린다 (2c − u)
    맞추지 않으면 오차가 아니라 좌표계 차이를 재게 된다 — 이 표본에서는
    그 차이가 970px 였다.
    """
    uv = np.array([p["uv"] for p in ln["points"]], float)
    sw = float(cp.get("truth_sensor_w") or 0.0)
    if sw > 0:
        uv = uv * np.array([img_w / sw, img_h / float(cp["truth_sensor_h"])])
    if flipped:
        uv = np.stack([2 * cp["cx_px"] - uv[:, 0],
                       2 * cp["cy_px"] - uv[:, 1]], axis=1)
    return uv


def _compare_to_truth(detected, truth, cp, rgb, qz, flipped=False):
    """
    검출 화소를 정답 화소와 맞대 정확도를 낸다.

    ID 가 아니라 **위치** 로 짝짓는다. 내보내기마다 선 번호 규약이 달라
    ID 로 빼면 없는 오차가 생긴다. 알고 싶은 것은 "찾았는가, 얼마나
    정확한가" 이므로 위치가 맞는 짝을 고르고 번호 일치는 따로 센다.
    """
    H, W = rgb.shape[:2]
    det_key = {}
    for lid, pts in detected.items():
        d = np.asarray(pts, float)
        if len(d) >= 3:
            ax = 0 if lid.startswith("V") else 1
            det_key[lid] = (d, float(np.median(d[:, ax])))

    rows, errs = [], []
    n_id_ok = {"V": 0, "H": 0}
    n_id_tot = {"V": 0, "H": 0}
    for lid, ln in sorted(truth.items(),
                          key=lambda kv: (kv[0][0], int(kv[0][1:]))):
        axis = lid[0]
        if axis not in ("V", "H"):
            continue
        n_id_tot[axis] += 1
        gt = _truth_uv(ln, cp, W, H, flipped)
        ax = 0 if ln.get("fixed", "alpha") == "alpha" else 1
        gkey = float(np.median(gt[:, ax]))
        cand = [(abs(k - gkey), l, d) for l, (d, k) in det_key.items()
                if l[0] == axis]
        row = {"lid": lid, "fixed": ln.get("fixed", "alpha"),
               "n_gt": len(gt), "gt_pos": round(gkey, 1),
               "z_m": None, "matched": None, "id_ok": False,
               "n_det": 0, "err_med": None, "err_noise": None,
               "err_rms": None, "err_p95": None, "err_max": None,
               "note": None}
        if not cand:
            row["note"] = "검출선 없음"
            rows.append(row); continue
        gap, mlid, d = min(cand, key=lambda t: t[0])
        row["matched"] = mlid
        row["match_gap_px"] = round(gap, 2)
        row["id_ok"] = (mlid == lid)
        if row["id_ok"]:
            n_id_ok[axis] += 1
        if gap > LC.MATCH_TOL_PX:
            row["note"] = f"미검출 — 가장 가까운 선이 {gap:.1f}px 떨어짐"
            rows.append(row); continue
        sc = 1 - ax                              # 스캔축: V선은 v, H선은 u
        o = np.argsort(gt[:, sc])
        inr = (d[:, sc] >= gt[o, sc][0]) & (d[:, sc] <= gt[o, sc][-1])
        if inr.sum() < 5:
            row["note"] = "정답 구간과 겹치는 검출점이 없음"
            rows.append(row); continue
        e = d[inr, ax] - np.interp(d[inr, sc], gt[o, sc], gt[o, ax])
        row.update(n_det=int(inr.sum()),
                   err_med=float(np.median(e)),
                   err_noise=float(np.std(e - np.median(e))),
                   err_rms=float(np.sqrt(np.mean(e ** 2))),
                   err_p95=float(np.percentile(np.abs(e), 95)),
                   err_max=float(np.abs(e).max()))
        rows.append(row)
        if ax == 0:
            errs.append(e)
    if not errs:
        return {"error": "정답과 짝지어진 세로선이 없다"}

    E = np.concatenate(errs)
    v_rows = [r for r in rows if r["fixed"] == "alpha"]
    ok = [r for r in v_rows if r["err_rms"] is not None]
    missed = [r for r in v_rows if r["err_rms"] is None]
    bias = float(np.median(E)); noise = float(np.std(E - np.median(E)))
    z = float(cp.get("standoff_z", 1.2))
    mm = z * z / (cp["f_px"] * cp["b_m"]) * 1000.0

    families = {}
    for pre, fx, nm, axn in (("V", "alpha", "V선(세로)", "u"),
                             ("H", "beta", "H선(가로)", "v")):
        fr = [r for r in rows if r["fixed"] == fx]
        if not fr:
            continue
        fok = [r for r in fr if r["err_rms"] is not None]
        med = [r["err_med"] for r in fok]
        nz = [r["err_noise"] for r in fok]
        pooled = (float(np.sqrt(np.mean(
            [r["err_noise"] ** 2 + (r["err_med"] - np.median(med)) ** 2
             for r in fok]))) if fok else None)
        sw = []
        for r in fr:
            g = _truth_uv(truth[r["lid"]], cp, rgb.shape[1], rgb.shape[0],
                          flipped)
            sw.append(float(np.ptp(g[:, 0 if fx == "alpha" else 1])))
        gains = [line_gain for line_gain in
                 [_gain_of(r["lid"], cp) for r in fr]]
        gfin = [g for g in gains if np.isfinite(g) and g < 1e6]
        families[pre] = {
            "이름": nm, "측정축": axn, "선 수": len(fr), "검출": len(fok),
            "미검출": [r["lid"] for r in fr if r["err_rms"] is None],
            "계통편차_px": (round(float(np.median(med)), 4) if med else None),
            "선간편차_px": (round(float(np.std(med)), 4)
                        if len(med) > 1 else None),
            "선내잡음_px": (round(float(np.median(nz)), 4) if nz else None),
            "통합오차_px": (round(pooled, 4) if pooled is not None else None),
            "정답선_변동폭_px": (round(float(np.median(sw)), 4) if sw else None),
            "깊이이득_중앙": (round(float(np.median(gfin)), 3) if gfin else None),
            "깊이가능": sum(1 for g in gains
                         if np.isfinite(g) and g <= EQ7.MAX_DEPTH_GAIN)}

    return {
        "image": "입력 이미지", "image_size": [rgb.shape[1], rgb.shape[0]],
        "flipped": False, "scale_to_sensor": 1.0, "scale_to_sensor_v": 1.0,
        "f_px_image": round(cp["f_px"], 1), "f_px_sensor": cp["f_px"],
        "z_ref_m": round(z, 3),
        "n_lines_gt": len(truth), "n_lines_det": len(detected),
        "rows": rows,
        "err_med_px": round(float(np.median(E)), 4),
        "err_rms_px": round(float(np.sqrt(np.mean(E ** 2))), 4),
        "err_p95_px": round(float(np.percentile(np.abs(E), 95)), 3),
        "err_bias_px": round(bias, 4), "err_noise_px": round(noise, 4),
        "err_rms_sensor_px": round(float(np.sqrt(np.mean(E ** 2))), 4),
        "bias_sensor_px": round(bias, 4), "noise_sensor_px": round(noise, 4),
        "depth_err_mm": round(float(np.sqrt(np.mean(E ** 2))) * mm, 3),
        "depth_bias_mm": round(abs(bias) * mm, 3),
        "depth_noise_mm": round(noise * mm, 3),
        "mm_per_px_depth": round(mm, 3),
        "n_v_matched": len(ok), "n_v_total": len(v_rows),
        "n_v_missed": len(missed),
        "missed_lines": [r["lid"] for r in missed],
        "id_ok": {k: (n_id_ok[k], n_id_tot[k]) for k in ("V", "H")},
        "sigma_u_design_px": CALIB.SIGMA_U_PX,
        "quantization_px": qz.get("sigma_floor_px"),
        "quantization_mm": (round(qz["sigma_floor_px"] * mm, 3)
                            if qz.get("sigma_floor_px") else None),
        "families": families, "max_depth_gain": EQ7.MAX_DEPTH_GAIN,
    }


def _gain_of(lid, cp):
    la = CALIB.make_line_angles(
        n_v=cp["n_v"], n_h=cp["n_h"], fov_deg=cp["fov_h_deg"],
        laser_tilt_deg=cp.get("laser_tilt_deg", 0.0),
        laser_roll_deg=cp.get("laser_roll_deg", 0.0))
    info = la.get(lid)
    return EQ7.depth_gain(info["normal"]) if info else float("inf")


def _depth_vs_truth(lines_uv, lines_xyz, truth, cp, flipped=False):
    """
    검출 화소로 푼 깊이를 정답과 맞댄다.

    정답에 xyz_world 가 있어도 카메라 자세를 모르면 조사기 좌표로 옮길
    수 없다. 그래서 **정답 화소를 같은 식으로 삼각측량한 값** 을 기준으로
    삼는다. 이러면 삼각측량 모델은 공통이고 화소 차이만 남으므로, 재는
    것이 정확히 "선검출 오차가 깊이를 얼마나 흔드는가" 가 된다.
    """
    f, b = cp["f_px"], cp["b_m"]
    cx, cy = cp["cx_px"], cp["cy_px"]
    la = CALIB.make_line_angles(
        n_v=cp["n_v"], n_h=cp["n_h"], fov_deg=cp["fov_h_deg"],
        laser_tilt_deg=cp.get("laser_tilt_deg", 0.0),
        laser_roll_deg=cp.get("laser_roll_deg", 0.0))
    dz, dd, per_line = [], [], {}
    for lid, uv in lines_uv.items():
        ln = truth.get(lid)
        info = la.get(lid)
        if ln is None or info is None or lid not in lines_xyz:
            continue
        g_uv = _truth_uv(ln, cp, cp["image_w"], cp["image_h"],
                         flipped)
        gx, keep = EQ7.triangulate_plane(g_uv, info["normal"], f, b, cx, cy)
        if len(gx) < 5:
            continue
        g_uv = g_uv[keep]
        ax = 0 if ln.get("fixed", "alpha") == "alpha" else 1
        sc = 1 - ax
        o = np.argsort(g_uv[:, sc])
        uv = np.asarray(uv, float)
        got = np.asarray(lines_xyz[lid], float)
        n = min(len(uv), len(got)); uv, got = uv[:n], got[:n]
        inr = ((uv[:, sc] >= g_uv[o, sc][0]) & (uv[:, sc] <= g_uv[o, sc][-1]))
        if inr.sum() < 5:
            continue
        ref = np.column_stack([np.interp(uv[inr, sc], g_uv[o, sc], gx[o, k])
                               for k in range(3)])
        e = got[inr] - ref
        dz.append(e[:, 2]); dd.append(np.linalg.norm(e, axis=1))
        per_line[lid] = round(float(np.median(np.abs(e[:, 2]))) * 1000.0, 3)
    if not dz:
        return None
    Z = np.concatenate(dz) * 1000.0
    D = np.concatenate(dd) * 1000.0
    return {"n_points": int(len(Z)),
            "z_bias_mm": round(float(np.median(Z)), 3),
            "z_noise_mm": round(float(np.std(Z - np.median(Z))), 3),
            "z_rms_mm": round(float(np.sqrt(np.mean(Z ** 2))), 3),
            "z_p95_mm": round(float(np.percentile(np.abs(Z), 95)), 3),
            "dist_med_mm": round(float(np.median(D)), 3),
            "dist_p95_mm": round(float(np.percentile(D, 95)), 3),
            "per_line_mm": per_line,
            "기준": "정답 화소를 같은 식으로 삼각측량한 값"}


# =====================================================================
# 그림
# =====================================================================
def _base_for_overlay(scene_image, laser_image, size):
    """
    결과 그림의 배경.

    레이저 OFF 장면 사진이 있으면 그것을, 없으면 레이저 이미지에서 선을
    지워 쓴다. 배경에 격자가 남아 있으면 검출점과 뒤섞여 무엇이 결과인지
    구분되지 않으므로, 어느 쪽이든 색을 빼고 흑백으로 깐다.
    """
    from PIL import Image
    src = scene_image or laser_image
    im = Image.open(src).convert("RGB").resize(size, Image.BICUBIC)
    a = np.asarray(im, float)
    a = LC._remove_laser(a)
    ex = a[:, :, 1] - np.maximum(a[:, :, 0], a[:, :, 2])
    a[:, :, 1] -= np.clip(ex, 0.0, None)
    gray = 0.299 * a[:, :, 0] + 0.587 * a[:, :, 1] + 0.114 * a[:, :, 2]
    return np.clip(np.repeat(gray[:, :, None], 3, axis=2), 0, 255).astype(
        np.uint8)


def _overlay(rgb, detected, truth, cp, path, zoom=6, crop=140,
             flipped=False, uv_tf=None):
    """검출선(자홍)과 정답선(청록)을 확대해 나란히 보여 준다."""
    from PIL import Image, ImageDraw
    H, W = rgb.shape[:2]
    im = Image.fromarray((np.asarray(rgb, float) * 0.55).astype(np.uint8))
    d = ImageDraw.Draw(im)
    # 배경은 원본 이미지이므로, 그릴 좌표를 원본 규약으로 되돌린다.
    back = (uv_tf if uv_tf is not None else (lambda a: np.asarray(a, float)))
    for lid, ln in truth.items():
        g = back(_truth_uv(ln, cp, W, H, flipped))
        for u, v in np.asarray(g)[::7]:
            d.ellipse([u - 1, v - 1, u + 1, v + 1], fill=(34, 211, 238))
    for lid, pts in detected.items():
        for u, v in np.asarray(back(pts), float)[::7]:
            d.ellipse([u - 1, v - 1, u + 1, v + 1], fill=(236, 72, 153))
    cx0, cy0 = W // 2, H // 2
    box = (max(0, cx0 - crop), max(0, cy0 - crop),
           min(W, cx0 + crop), min(H, cy0 + crop))
    zi = im.crop(box).resize(((box[2] - box[0]) * zoom,
                              (box[3] - box[1]) * zoom), Image.NEAREST)
    out = Image.new("RGB", (im.width + zi.width + 24,
                            max(im.height, zi.height)), (16, 16, 20))
    out.paste(im, (0, 0)); out.paste(zi, (im.width + 24, 0))
    dd = ImageDraw.Draw(out)
    f = REPORT._korean_font(max(16, out.height // 45))
    t = ("자홍 = 검출,  청록 = 정답 (오른쪽은 중앙 확대)" if f
         else "magenta = detected, cyan = truth")
    dd.text((12, out.height - 32), t, fill=(235, 235, 235), font=f)
    dd.rectangle([box[0], box[1], box[2], box[3]], outline=(250, 204, 21),
                 width=2)
    out.save(path)
    return path


def _caveats(g_assumed, params, truth, qz, su, depth, backend):
    c = []
    if g_assumed:
        c.append("IMU 가 없어 **장비가 똑바로 서 있다고 가정**했다(ĝ=(0,1,0)). "
                 "수직도·수평도는 면 법선과 중력의 사잇각이므로, 장비가 실제로 "
                 "기울어 있었다면 그 각도가 그대로 판정에 더해진다. 1° 기울면 "
                 "판정도 1° 틀린다 — 허용치가 ±0.5° 임을 생각하면 작지 않다.")
    if not params:
        c.append(f"카메라 사양을 실측 캘리브레이션이 아니라 사양 프로파일"
                 f"('{CALIB.ACTIVE_PROFILE}')에서 가져왔다. 발사각 α_i·주점·"
                 f"기선이 실제 장비와 다르면 절대 거리에 계통 오차가 남는다. "
                 f"각도(수직도·수평도)는 상대량이라 영향이 작지만 평활도는 "
                 f"거리 오차를 그대로 받는다.")
    if not truth:
        c.append("정답 데이터가 없어 선검출 정확도와 깊이 오차를 재지 못했다. "
                 "표의 σ_n 은 측정치가 아니라 가정에 따른 예상치다.")
    if qz.get("binary"):
        c.append(f"입력 이미지가 이진(안티에일리어싱 없음)이다. 켜진 화소가 "
                 f"전부 같은 값이라 선 중심이 0.5px 격자에 갇히고, 어떤 "
                 f"추정기를 써도 σ {qz['sigma_floor_px']}px 아래로 못 "
                 f"내려간다. 이 하한이 곧 깊이 정밀도의 바닥이다 — 렌더에 "
                 f"안티에일리어싱을 켜거나 실촬영본을 쓰면 개선된다.")
    if depth and depth["z_noise_mm"] > 2.0:
        c.append(f"점별 깊이 산포가 {depth['z_noise_mm']:.1f}mm 로 평활도 목표 "
                 f"±2mm 를 넘는다. 평활도는 점별 깊이 오차가 그대로 결과이므로 "
                 f"이 촬영에서는 측정불가로 나오는 것이 정상이다. 각도는 수만 "
                 f"점을 평균하므로 영향이 훨씬 작다.")
    if backend == "geom":
        c.append("기하 전용 백엔드는 동바리/기둥/철근, 벽/거푸집/조적을 "
                 "구분하지 못한다. 부재 종류에 따라 KCS 허용치가 달라지므로 "
                 "종류는 사람이 확인해야 한다.")
    return c


def main():
    ap = argparse.ArgumentParser(
        description="레이저 그리드 품질검측 — 이미지 한 장 → 엑셀 조서 하나")
    ap.add_argument("--image", required=True, help="레이저 격자 이미지 (필수)")
    ap.add_argument("--params", default=None, help="camera_params.json (선택)")
    ap.add_argument("--imu", default=None,
                    help="IMU json (선택). 없으면 똑바로 서 있다고 가정")
    ap.add_argument("--truth", default=None,
                    help="cast_pixels.json 정답값 (선택)")
    ap.add_argument("--scene", default=None,
                    help="레이저 OFF 장면 사진 (선택, 배경용)")
    ap.add_argument("--out", default=None, help="엑셀 조서 경로")
    ap.add_argument("--profile", default=None,
                    help="사양 프로파일 (legacy/pdf/improved/diagonal)")
    ap.add_argument("--backend", default="geom", choices=["geom", "sam", "vlm"])
    ap.add_argument("--standoff", type=float, default=None,
                    help="대표 측정거리 [m]")
    ap.add_argument("--site", default=None, help="현장명")
    ap.add_argument("--smooth", type=int, default=0,
                    help="선을 따라 N점 평활 (기본 0=끔). 보정량은 양자화 "
                         "반폭으로 묶인다. 비스듬한 면에서 깊이 잡음이 "
                         "줄지만(실측 4.32→2.93mm) 창 너비보다 좁은 요철은 "
                         "뭉개진다")
    ap.add_argument("--pc-stride", type=int, default=20,
                    help="3D 좌표 표에 N개마다 한 점 (기본 20)")
    a = ap.parse_args()
    run(image=a.image, params=a.params, imu=a.imu, truth=a.truth,
        scene_image=a.scene, out=a.out, profile=a.profile,
        backend=a.backend, standoff_m=a.standoff, site=a.site,
        pc_stride=a.pc_stride, smooth=a.smooth)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
