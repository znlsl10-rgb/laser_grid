#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hardware.py — 실장비에서 무엇을 받아야 하는가 (규약 + 검사기)
========================================================================
실장비가 오면 이 파일이 첫 관문이다. 촬영 한 벌을 주고

    python3 hardware.py <촬영폴더>

를 돌리면 "검측을 돌려도 되는가" 를 항목별로 답한다. 문서가 아니라
**검사기** 로 둔 이유는, 규약을 글로만 적어 두면 실제 파일이 규약과
다를 때 파이프라인 한참 안쪽에서 엉뚱한 숫자로 터지기 때문이다.
여기서 걸리면 그 자리에서 무엇이 없는지 알 수 있다.

────────────────────────────────────────────────────────────────────────
1. 촬영 한 벌에 있어야 하는 것
────────────────────────────────────────────────────────────────────────
필수
  laser_on.png        레이저 ON 프레임. 무손실(png/tiff) 권장.
                      jpg 는 압축 잡음이 선 중심을 흔들어 σ_u 가 커진다.
  camera_params.json  아래 2절의 값들.

강력 권장
  laser_off.png       레이저 OFF 프레임. ON 과 **같은 노출**로, 수십 µs
                      안에 연속으로 찍어야 한다. 두 장을 빼면 배경광이
                      지워져 햇빛 드는 현장에서도 선이 살아난다.
                      (이게 없으면 채널 과잉분만으로 분리 — 실내 전용)
  imu.json            {"gravity": [x,y,z]} 또는 {"pitch_deg":, "roll_deg":}.
                      **조사기 좌표계** 기준. 없으면 장비가 똑바로 섰다고
                      가정하고, 판정은 참고값으로 낮춘다.

선택 (검증용, 현장에는 없어도 됨)
  truth.json          정답 화소·좌표. 있으면 선검출 정확도와 깊이 오차를
                      대조해 조서에 싣는다.

────────────────────────────────────────────────────────────────────────
2. camera_params.json — 값마다 '어떻게 얻는가' 가 다르다
────────────────────────────────────────────────────────────────────────
    {
      "camera": {"f_px": 1593.0, "cx_px": 1224.0, "cy_px": 1024.0,
                 "sensor_W": 2448, "sensor_H": 2048},
      "baseline_m": 0.150,
      "grid": {"n_vertical": 21, "n_horizontal": 21, "fov_deg": 60.82},
      "laser": {"roll_deg": 0.0, "tilt_deg": 0.0, "wavelength_nm": 520}
    }

  f_px, cx_px, cy_px   체커보드 캘리브레이션. **추정하면 안 된다** —
                       f 가 1% 틀리면 깊이가 1% 틀린다(1.7m 에서 17mm).
  sensor_W/H           센서 화소 수. 이미지가 축소본이면 배율을 여기서 잡는다.
  baseline_m           카메라 광심 ↔ 조사기 출사점 거리. **조립 후 실측**.
                       b 가 1mm 틀리면 깊이가 0.7% 틀린다(150mm 기준).
  grid.n_vertical/     DOE 가 만드는 선 수. 데이터시트 값.
  n_horizontal
  grid.fov_deg         격자 전체 발산각. 데이터시트 값.
  laser.roll_deg       **광축 둘레로 DOE 를 얼마나 돌려 끼웠는가.**
                       0 이면 세로선이 화면에서 똑바로 세로다. 이때
                       가로선은 깊이를 못 준다(레이저 평면이 카메라 광심을
                       지난다). γ 를 주면 g=1/sin γ 로 유한해져 가로선도
                       삼각측량된다 — 동바리 단면이 가정 없이 측정된다.
                       10° 만 돌려도 합성 검증에서 참 원통면 0.16mm.
  laser.tilt_deg       조사기 광축이 카메라 광축과 이루는 수렴각.
  laser.wavelength_nm  참고값. 채널은 이미지에서 자동 판별하므로 필수 아님.

더 나은 길 — 발사각 대신 **평면 법선**
  DOE 선 i 의 레이저 평면 법선 n_i 를 직접 재면 각도 모델(α_i, β_j)보다
  가정이 적다. 평면 하나에 자유도 3 이고, 굴림·수렴각·렌즈 왜곡이 그
  안에 다 흡수된다. camera_params.json 에

      "lines": {"V0": {"normal": [nx, ny, nz]}, ..., "H0": {...}}

  를 넣으면 파이프라인이 각도 모델을 건너뛰고 이 값을 그대로 쓴다
  (eq7.line_planes 가 normal 이 있으면 우선한다).

────────────────────────────────────────────────────────────────────────
3. 촬영 조건 — 지키지 않으면 정확도가 여기서 정해진다
────────────────────────────────────────────────────────────────────────
  · 무손실 저장, 안티에일리어싱 켠 채로. 선이 이진(0/255)으로 찍히면
    선 중심이 0.5화소 격자에 갇혀 σ_u 가 1/√12 = 0.289px 아래로 못 간다.
    깊이 잡음은 σ_Z = σ_u·Z²/(f·b) 이므로 그대로 바닥이 된다.
  · 과포화 금지. 선 중심이 평평해지면 능선을 못 찾는다.
  · ON/OFF 두 프레임 사이 장비가 움직이면 차영상이 어긋난다. 수십 µs.
  · 부재 하나에 세로선이 **2줄 이상** 걸리게. 한 줄이면 단면이 안 풀려
    수직도를 절반만 재고 판정보류가 된다.

업체에 넘기는 규약서(좌표계 정의·납품 체크리스트 포함):
    docs/하드웨어_인터페이스_규약.md

이 파일을 직접 실행하면 자체 검증이 돈다:  python3 hardware.py
========================================================================
"""
import os as _os
import sys as _sys
import json as _json

import numpy as np

# 값의 출처 등급 — 조서 2번 시트가 이 등급을 그대로 싣는다.
SOURCE = {"measured": "실측(캘리브레이션)", "spec": "데이터시트",
          "assumed": "가정 — 확인 필요"}

REQUIRED = ("camera.f_px", "camera.cx_px", "camera.cy_px",
            "camera.sensor_W", "camera.sensor_H", "baseline_m")
RECOMMENDED = ("grid.n_vertical", "grid.n_horizontal", "grid.fov_deg")
# 굴림·수렴각은 세 자리 중 어디에 적어도 된다(파서가 다 읽는다).
ROLL_KEYS = ("laser.roll_deg", "grid.laser_roll_deg", "laser_roll_deg")
TILT_KEYS = ("laser.tilt_deg", "grid.laser_tilt_deg", "laser_tilt_deg")

IMG_ON = ("laser_on", "cast", "_on", "on_", "grid", "레이저")
IMG_OFF = ("laser_off", "cam", "scene", "_off", "off_", "배경", "장면")


def _dig(d, path):
    cur = d
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def template():
    """실장비 담당자에게 건네는 camera_params.json 빈 서식."""
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
                              "가로선이 깊이를 못 준다",
            "lines": "선택 — 선별 레이저 평면 법선을 직접 쟀다면 "
                     "{\"V0\": {\"normal\": [nx,ny,nz]}, ...} 로 넣으면 "
                     "각도 모델보다 정확하다"},
    }


def check_params(params):
    """camera_params 를 훑어 (문제목록, 경고목록, 요약) 을 돌려준다."""
    bad, warn, info = [], [], {}
    if not isinstance(params, dict):
        return ["camera_params 가 dict 가 아니다"], [], {}
    for k in REQUIRED:
        v = _dig(params, k)
        if v is None:
            bad.append(f"필수 값 없음: {k}")
        elif not isinstance(v, (int, float)) or not np.isfinite(float(v)):
            bad.append(f"필수 값이 숫자가 아님: {k} = {v!r}")
    for k in RECOMMENDED:
        if _dig(params, k) is None:
            warn.append(f"권장 값 없음: {k}")
    if all(_dig(params, k) is None for k in TILT_KEYS):
        warn.append("권장 값 없음: 수렴각 (laser.tilt_deg 등) — 0 으로 가정")

    f = _dig(params, "camera.f_px")
    b = params.get("baseline_m")
    if isinstance(f, (int, float)) and isinstance(b, (int, float)) and b:
        info["f_px"] = float(f)
        info["baseline_m"] = float(b)
        # 깊이 잡음: σ_Z = σ_u·Z²/(f·b). 대표 거리 1.7m, σ_u 0.3px 기준.
        z = 1.7
        info["깊이잡음_mm@1.7m_σu0.3px"] = round(
            0.3 * z * z / (float(f) * float(b)) * 1000.0, 2)
        if float(b) < 0.05:
            warn.append(f"기선이 {float(b)*1000:.0f}mm 로 짧다 — 깊이 잡음이 "
                        f"기선에 반비례한다")
    # 파서가 읽는 세 자리를 그대로 본다 — 규약서와 검사기와 파서가
    # 서로 다른 자리를 보면, 값을 적어 놓고도 0 으로 도는 사고가 난다.
    roll = _dig(params, "laser.roll_deg")
    if roll is None:
        roll = _dig(params, "grid.laser_roll_deg")
    if roll is None:
        roll = params.get("laser_roll_deg")
    if roll is not None:
        info["laser_roll_deg"] = float(roll)
        if abs(float(roll)) < 1.0:
            warn.append(
                "격자 굴림이 0° 다 — 가로선이 깊이를 못 준다(레이저 평면이 "
                "카메라 광심을 지남). 동바리 같은 원형 부재의 단면을 재려면 "
                "DOE 를 광축 둘레로 10° 이상 돌려 끼울 것 "
                "(g=1/sin γ, 10° → 합성 검증 잔차 0.16mm)")
    else:
        warn.append("laser.roll_deg 가 없다 — 0 으로 가정한다")

    lines = params.get("lines")
    if isinstance(lines, dict) and lines:
        n_norm = sum(1 for v in lines.values()
                     if isinstance(v, dict) and v.get("normal"))
        info["법선을 직접 준 선"] = n_norm
    return bad, warn, info


def check_image(path):
    """이미지 한 장이 검측에 쓸 만한가."""
    out = {"path": path}
    try:
        from PIL import Image
    except ImportError:
        return {"path": path, "error": "Pillow 없음"}
    try:
        im = Image.open(path)
    except Exception as e:
        return {"path": path, "error": f"열 수 없음: {e}"}
    out["크기"] = f"{im.width}×{im.height}"
    out["형식"] = im.format
    a = np.asarray(im.convert("RGB"), np.float32)
    try:
        _here = _os.path.dirname(_os.path.abspath(__file__))
        if _here not in _sys.path:
            _sys.path.insert(0, _here)
        import laser_signal as LS
        sig, meta = LS.laser_signal(a, info=True)
        out["레이저 채널"] = meta["mode"]
        out["채널 대비"] = meta.get("why")
    except Exception as e:
        sig = a[:, :, 1] - 0.5 * (a[:, :, 0] + a[:, :, 2])
        out["레이저 채널"] = f"판별 실패({e}) — 초록 가정"

    hi = float(np.percentile(sig, 99.9))
    med = float(np.median(sig))
    lit = sig[sig > med + 0.5 * (hi - med)]
    out["점등 화소 비율"] = round(float(len(lit)) / sig.size, 4)
    # 이진 렌더인가 — 켜진 화소가 거의 다 최대값이면 안티에일리어싱이 없다
    out["이진(안티에일리어싱 없음)"] = bool(
        len(lit) and float((lit > hi * 0.9).mean()) > 0.9)
    # 과포화 — 8bit 최대값에 붙은 화소 비율
    out["과포화 화소 비율"] = round(
        float((a.max(axis=2) >= 254.0).mean()), 5)
    return out


def check_capture(folder, verbose=True):
    """
    촬영 폴더 하나를 검사한다.

    Returns
    -------
    dict — ok(bool), 필수문제, 경고, 찾은 파일, 이미지 진단, 사양 요약
    """
    def say(*a):
        if verbose:
            print(*a)

    res = {"folder": folder, "필수문제": [], "경고": [], "파일": {},
           "이미지": {}, "사양": {}}
    if not _os.path.isdir(folder):
        res["필수문제"].append(f"폴더가 없다: {folder}")
        res["ok"] = False
        return res

    names = sorted(_os.listdir(folder))
    imgs = [n for n in names
            if n.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff",
                                   ".bmp"))]
    jsons = [n for n in names if n.lower().endswith(".json")]

    on = next((n for n in imgs
               if any(k in n.lower() for k in IMG_ON)), None)
    off = next((n for n in imgs
                if any(k in n.lower() for k in IMG_OFF)), None)
    if on is None and imgs:
        on = imgs[0]
        res["경고"].append(f"이름으로 ON 프레임을 못 골랐다 — {on} 를 쓴다")
    if on is None:
        res["필수문제"].append("레이저 ON 이미지가 없다")
    res["파일"]["laser_on"] = on
    res["파일"]["laser_off"] = off
    if off is None:
        res["경고"].append(
            "레이저 OFF 프레임이 없다 — 배경광이 센 현장(햇빛)에서는 "
            "차영상 없이 선을 놓칠 수 있다. 같은 노출로 수십 µs 안에 "
            "연속 촬영할 것")

    params = truth = imu = None
    for n in jsons:
        try:
            d = _json.load(open(_os.path.join(folder, n), encoding="utf-8"))
        except Exception:
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
            "통째로 배율만큼 틀린다. 없이 돌리면 판정은 참고값으로 낮춰진다")
    else:
        bad, warn, info = check_params(params)
        res["필수문제"] += bad
        res["경고"] += warn
        res["사양"] = info
    if imu is None:
        res["경고"].append(
            "IMU 값이 없다 — 장비가 똑바로 섰다고 가정한다. 숙여 찍었다면 "
            "바닥이 기운 벽으로 읽힌다(실측: 바닥 캡처가 4.4594° 기준초과로 "
            "나왔는데 사양 파일과 함께 돌리면 0.0390° 합격)")
    if truth is None:
        res["경고"].append("정답값이 없다 — 선검출 정확도·깊이 오차를 "
                          "대조할 수 없다(현장에서는 정상)")

    if on:
        d = check_image(_os.path.join(folder, on))
        res["이미지"]["laser_on"] = d
        if d.get("이진(안티에일리어싱 없음)"):
            res["경고"].append(
                "선이 이진으로 찍혔다 — 선 중심이 0.5화소 격자에 갇혀 "
                "σ_u 가 0.289px 아래로 못 간다. 안티에일리어싱을 켜거나 "
                "실촬영본을 쓸 것")
        if (d.get("과포화 화소 비율") or 0) > 0.02:
            res["경고"].append(
                f"과포화 화소가 {d['과포화 화소 비율']*100:.1f}% — 선 중심이 "
                f"평평해져 능선을 못 찾는다. 노출을 줄일 것")
        if (d.get("점등 화소 비율") or 0) < 1e-4:
            res["필수문제"].append(
                "레이저 신호가 거의 없다 — 채널 판별 결과와 노출을 확인할 것")
    if off:
        res["이미지"]["laser_off"] = check_image(_os.path.join(folder, off))

    res["ok"] = not res["필수문제"]
    if verbose:
        _print(res)
    return res


def _print(res):
    print("=" * 74)
    print(f"촬영 점검 — {res['folder']}")
    print("=" * 74)
    print("  [파일]")
    for k, ko in (("laser_on", "레이저 ON (필수)"),
                  ("laser_off", "레이저 OFF (권장)"),
                  ("camera_params", "카메라 사양 (필수)"),
                  ("imu", "IMU (권장)"), ("truth", "정답값 (선택)")):
        v = res["파일"].get(k)
        print(f"    {ko:<20} {v if v else '— 없음'}")
    if res["사양"]:
        print("  [사양]")
        for k, v in res["사양"].items():
            print(f"    {k:<28} {v}")
    for k, d in res["이미지"].items():
        print(f"  [{k}]")
        for kk, vv in d.items():
            if kk == "path":
                continue
            print(f"    {kk:<28} {vv}")
    if res["필수문제"]:
        print("  [막힘 — 이대로는 검측을 못 돌린다]")
        for m in res["필수문제"]:
            print(f"    · {m}")
    if res["경고"]:
        print("  [경고 — 돌아가지만 정확도·판정에 영향]")
        for m in res["경고"]:
            print(f"    · {m}")
    print("-" * 74)
    print("  결과: " + ("검측 가능" if res["ok"] else "필수 항목이 빠졌다"))
    print("=" * 74)


def main(argv=None):
    a = list(_sys.argv[1:] if argv is None else argv)
    if a and a[0] == "--template":
        out = a[1] if len(a) > 1 else "camera_params.json"
        _json.dump(template(), open(out, "w", encoding="utf-8"),
                   ensure_ascii=False, indent=2)
        print(f"서식을 썼다: {out}")
        return 0
    if not a:
        print(__doc__)
        return _selftest()
    ok = True
    for folder in a:
        ok &= bool(check_capture(folder)["ok"])
    return 0 if ok else 1


# ============ 자체 검증 ============
def _selftest():
    import tempfile
    from PIL import Image
    print("=" * 74)
    print("hardware.py 자체 검증")
    print("=" * 74)
    ok = True

    bad, warn, info = check_params(template())
    ok &= len(bad) == len(REQUIRED)
    print(f"  빈 서식 → 필수 문제 {len(bad)}건 (기대 {len(REQUIRED)})  "
          f"{'PASS' if len(bad) == len(REQUIRED) else 'FAIL'}")

    good = {"camera": {"f_px": 1593.0, "cx_px": 1224.0, "cy_px": 1024.0,
                       "sensor_W": 2448, "sensor_H": 2048},
            "baseline_m": 0.15,
            "grid": {"n_vertical": 21, "n_horizontal": 21, "fov_deg": 60.8},
            "laser": {"roll_deg": 0.0, "tilt_deg": 0.0}}
    bad, warn, info = check_params(good)
    ok &= not bad
    has_roll_warn = any("굴림" in w for w in warn)
    ok &= has_roll_warn
    print(f"  갖춘 사양 → 필수 문제 0건, 굴림 0° 경고 "
          f"{'있음' if has_roll_warn else '없음'}  "
          f"{'PASS' if not bad and has_roll_warn else 'FAIL'}")
    print(f"    깊이잡음 @1.7m σu0.3px = "
          f"{info.get('깊이잡음_mm@1.7m_σu0.3px')} mm")

    good["laser"]["roll_deg"] = 20.0
    _b, warn2, _i = check_params(good)
    ok &= not any("굴림" in w for w in warn2)
    print(f"  굴림 20° → 그 경고 사라짐  "
          f"{'PASS' if not any('굴림' in w for w in warn2) else 'FAIL'}")

    with tempfile.TemporaryDirectory() as td:
        # 이진 렌더 흉내 — 검은 배경에 순수 초록선. 배경이 흔들리면
        # 과잉분도 흔들려 "이진" 으로 안 잡히므로 배경을 0 으로 둔다.
        a = np.zeros((200, 300, 3), np.uint8)
        a[:, ::20, 1] = 255
        a[::20, :, 1] = 255
        Image.fromarray(a).save(_os.path.join(td, "laser_on.png"))
        _json.dump(good, open(_os.path.join(td, "camera_params.json"), "w"))
        r = check_capture(td, verbose=False)
        ok &= r["ok"]
        d = r["이미지"]["laser_on"]
        ok &= d["이진(안티에일리어싱 없음)"]
        ok &= "G채널" in d["레이저 채널"]
        ok &= any("OFF" in w for w in r["경고"])
        ok &= any("IMU" in w for w in r["경고"])
        print(f"  최소 촬영 폴더 → 검측가능={r['ok']}, "
              f"채널={d['레이저 채널']}, 이진={d['이진(안티에일리어싱 없음)']}, "
              f"경고 {len(r['경고'])}건  {'PASS' if ok else 'FAIL'}")

        _os.remove(_os.path.join(td, "camera_params.json"))
        r2 = check_capture(td, verbose=False)
        ok &= (not r2["ok"]) and any("camera_params" in m
                                     for m in r2["필수문제"])
        print(f"  사양 파일을 빼면 → 막힘  "
              f"{'PASS' if not r2['ok'] else 'FAIL'}")
    print("=" * 74)
    print("전체 통과" if ok else "실패 있음")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
