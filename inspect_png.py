#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
inspect_png.py — 렌더/촬영 이미지 한 장을 검측 파이프라인에 넣는다
========================================================================
Isaac 에서 뽑은 PNG(또는 실장비 촬영본)를 그대로 받아
  A_선검출 → eq1 삼각측량 → C_영역분할 → 영역별 검측
까지 돌리고 조서를 출력한다. Isaac 런타임 없이 돈다.

두 가지 모드
  --check   이미지에서 격자를 읽어 **어떤 사양으로 찍힌 것인지** 알려준다.
            선 개수·간격·화면 점유 범위를 활성 프로파일의 예측과 대조한다.
            사양이 확정되기 전에는 이걸 먼저 돌려 값을 맞추는 게 순서다.
  (기본)    검측까지 수행하고 부재별 판정을 낸다.

왜 이 파일이 따로 있나
  inspection.py 는 Isaac 씬을 세우고 스테이션을 돌며 촬영까지 하는
  파이프라인이라 Isaac 없이는 임포트조차 되지 않는다. 이미 뽑아 둔
  이미지를 다시 넣어 보려면 그 앞단이 필요 없다.

실행 예
  python3 inspect_png.py grid.png --check
  python3 inspect_png.py grid_on.png --off grid_off.png --standoff 1.2 --pitch 34
========================================================================
"""
import argparse
import numpy as np
import importlib.util as _ilu, os as _os


def _load(name):
    spec = _ilu.spec_from_file_location(
        name, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                            f"{name}.py"))
    m = _ilu.module_from_spec(spec); spec.loader.exec_module(m)
    return m


CALIB = _load("calibration")
EQ3 = _load("eq3_orientation")
DETECT = _load("A_선검출")
PIPE = _load("pipeline_region")
REPORT = _load("report")
XLS = _load("report_excel")


def read_image(path):
    from PIL import Image
    im = Image.open(path).convert("RGB")
    return np.asarray(im)


def camera_params_for(image_shape, standoff_m):
    """
    활성 프로파일의 캘리브레이션을 이미지 해상도에 맞춰 낸다.

    f_px 는 센서 해상도 기준값이다. 다른 크기로 저장된 이미지에 그대로
    쓰면 예측 격자가 화면 밖으로 나간다. calibration.scale_to_resolution
    이 그 환산을 담당한다 — 다만 환산한 값으로는 절대 측정을 할 수 없고
    (원본 해상도의 서브픽셀 정보가 이미 사라졌다) 격자 확인용이다.
    """
    h, w = image_shape[:2]
    cp = dict(CALIB.CAMERA_PARAMS)
    scaled = (w != cp["resolution"][0])
    if scaled:
        cp = CALIB.scale_to_resolution(w)
    cp.update({"n_v": CALIB.N_VERTICAL, "n_h": CALIB.N_HORIZONTAL,
               "fov_h_deg": CALIB.FOV_DEG, "fov_v_deg": CALIB.FOV_DEG,
               "image_w": w, "image_h": h,
               "standoff_z": float(standoff_m)})
    return cp, scaled


# =====================================================================
# --check : 이미지가 어떤 사양으로 찍힌 것인지 읽는다
# =====================================================================
def _line_positions(mask_1d, min_gap=3):
    """1차원 투영 프로파일에서 선 중심 위치를 뽑는다."""
    idx = np.where(mask_1d)[0]
    if len(idx) == 0:
        return np.array([])
    breaks = np.where(np.diff(idx) > min_gap)[0]
    groups = np.split(idx, breaks + 1)
    return np.array([g.mean() for g in groups if len(g) >= 1])


def read_grid_from_image(rgb, occupancy=0.35):
    """
    격자 이미지에서 V/H 선의 픽셀 위치를 읽는다.

    선검출(A_선검출)은 예측 격자를 씨앗으로 쓰므로, 사양이 틀린 상태에서는
    예측이 어긋나 아무것도 못 찾는다. 확인 단계에서는 예측 없이 이미지만
    보고 읽어야 한다. 그래서 여기서는 축 투영 + 임계만 쓴다.

    임계를 분위수로 잡으면 안 된다. 선이 차지하는 화소는 전체의 몇 %뿐이라
    90% 분위수가 배경값이 되고, 그러면 화면 전체가 선으로 잡힌다. 신호의
    중앙값과 최대값 사이에서 잡아야 한다.
    """
    a = np.asarray(rgb, float)
    # 레이저는 녹색이다. 배경이 흰 종이·벽이면 G−(R+B)/2 는 0 근처이므로
    # 밝기 반전도 함께 본다(선이 배경보다 어두운 스캔본 대비).
    green = a[:, :, 1] - 0.5 * (a[:, :, 0] + a[:, :, 2])
    gray = a.mean(axis=2)
    signal = green if float(np.ptp(green)) > float(np.ptp(gray)) * 0.3 \
        else (gray.max() - gray)
    med = float(np.median(signal))
    hi = float(np.percentile(signal, 99.5))
    if hi - med < 1e-6:
        return np.array([]), np.array([])
    m = signal >= med + 0.4 * (hi - med)
    # 열별·행별 점유율. V선이 지나는 열은 대부분의 행이 켜져 있고,
    # 그렇지 않은 열은 H선이 지나는 몇 행만 켜져 있다.
    v = _line_positions(m.mean(axis=0) >= occupancy)
    h = _line_positions(m.mean(axis=1) >= occupancy)
    return v, h


def predicted_grid(cp, standoff_m):
    """활성 프로파일이 예측하는 V선 u 위치와 H선 v 위치."""
    ang = CALIB.make_line_angles()
    f, b = cp["f_px"], cp["b_m"]
    cx, cy = cp["cx_px"], cp["cy_px"]
    u = np.array([f * np.tan(ang[f"V{i}"]["angle_rad"]) - f * b / standoff_m + cx
                  for i in range(CALIB.N_VERTICAL)])
    v = np.array([f * np.tan(ang[f"H{j}"]["angle_rad"]) + cy
                  for j in range(CALIB.N_HORIZONTAL)])
    return u, v


def check(path, standoff_m):
    rgb = read_image(path)
    h, w = rgb.shape[:2]
    cp, scaled = camera_params_for(rgb.shape, standoff_m)
    dv, dh = read_grid_from_image(rgb)
    pu, pv = predicted_grid(cp, standoff_m)

    print("=" * 70)
    print(f"이미지 확인 — {_os.path.basename(path)}")
    print("=" * 70)
    print(f"  해상도        {w} × {h}")
    print(f"  활성 프로파일   {CALIB.ACTIVE_PROFILE} "
          f"({CALIB.SPEC_PROFILES[CALIB.ACTIVE_PROFILE]['label']})")
    if scaled:
        print(f"  [주의] 프로파일 해상도 {CALIB.IMAGE_W}×{CALIB.IMAGE_H} 와 다르다. "
              f"f_px 를 {cp['f_px']:.1f} 로 환산해 비교한다.")
        print(f"         환산본으로는 격자 확인만 가능하고 측정은 할 수 없다.")
    print()
    print(f"  {'':<14}{'이미지에서 읽음':<20}{'프로파일 예측':<20}")
    print("  " + "-" * 62)
    print(f"  {'V선 개수':<14}{len(dv):<20}{len(pu):<20}"
          + ("" if len(dv) == len(pu) else "  ← 다르다"))
    print(f"  {'H선 개수':<14}{len(dh):<20}{len(pv):<20}"
          + ("" if len(dh) == len(pv) else "  ← 다르다"))
    if len(dv) >= 2 and len(pu) >= 2:
        print(f"  {'V선 범위':<14}{f'{dv.min():.0f} .. {dv.max():.0f} px':<20}"
              f"{f'{pu.min():.0f} .. {pu.max():.0f} px':<20}")
        print(f"  {'V선 간격':<14}"
              f"{f'{np.diff(dv).mean():.1f} px (중앙 {np.median(np.diff(dv)):.1f})':<20}"
              f"{f'{np.diff(pu).mean():.1f} px':<20}")
    if len(dh) >= 2 and len(pv) >= 2:
        print(f"  {'H선 범위':<14}{f'{dh.min():.0f} .. {dh.max():.0f} px':<20}"
              f"{f'{pv.min():.0f} .. {pv.max():.0f} px':<20}")
    print()

    # 간격이 균일한지 — 균일하면 DOE 각도 모델이 반영되지 않은 렌더다
    if len(dv) >= 5:
        d = np.diff(dv)
        ratio = float(d.max() / max(d.min(), 1e-9))
        pd = np.diff(pu)
        pratio = float(pd.max() / max(pd.min(), 1e-9))
        print(f"  V선 간격 불균일도 (최대/최소)")
        print(f"    이미지    {ratio:.3f}")
        print(f"    프로파일   {pratio:.3f}   "
              f"(u = f·tanα 라 가장자리로 갈수록 넓어져야 정상)")
        if ratio < 1.05 <= pratio:
            print("    → 이미지 격자 간격이 거의 균일하다. 발사각이 아니라")
            print("      균등 분할 텍스처로 그려진 격자일 가능성이 크다.")
            print("      그 이미지로는 삼각측량 검증이 성립하지 않는다.")
    print()
    print("  격자선 개수가 맞지 않으면 calibration.SPEC_PROFILES 의")
    print("  n_vertical / n_horizontal / fov_deg 를 이미지에 맞춰 고친 뒤")
    print("  다시 확인한다. 검측식(eq1~eq6)은 이 값에 의존하지 않는다.")
    return 0


# =====================================================================
# 기본 : 검측
# =====================================================================
def inspect(path, off_path, standoff_m, pitch_deg, backend, out_dir=None,
            excel=None, seg_png=None, site=None):
    rgb = read_image(path)
    rgb_off = read_image(off_path) if off_path else None
    cp, scaled = camera_params_for(rgb.shape, standoff_m)
    if scaled:
        print("[중단] 이미지 해상도가 프로파일과 다르다. 환산본으로는 측정할 수 "
              "없다.\n       --check 로 격자를 먼저 맞추거나, 프로파일 해상도로 "
              "다시 렌더한다.")
        return 2

    line_angles = CALIB.make_line_angles()
    detected = DETECT.detect(rgb, {}, line_angles, cp,
                             laser_off_image=rgb_off, multi_surface=True)
    n_pts = sum(len(v) for v in detected.values())
    print(f"선검출: {len(detected)}선 / {n_pts}점")
    if n_pts == 0:
        print("[중단] 검출된 선이 없다. --check 로 사양을 먼저 맞춘다.")
        return 2

    lines_xyz, lines_uv, tri_info = PIPE.triangulate_lines(
        detected, line_angles, cp)
    n3d = sum(len(v) for v in lines_xyz.values())
    nf = tri_info["n_by_family"]
    print(f"삼각측량: 선 {len(lines_xyz)}개 / {n3d}점 "
          f"(V {nf.get('V', 0)}점 + H {nf.get('H', 0)}점)")
    for lid, why in tri_info["skipped"][:3]:
        print(f"  제외 {lid}: {why}")
    if len(tri_info["skipped"]) > 3:
        print(f"  ... 그 밖에 {len(tri_info['skipped'])-3}개 선 제외")
    if n3d == 0:
        print("[중단] 삼각측량된 점이 없다.")
        return 2

    # 장비를 아래로 pitch_deg 숙였을 때의 조사기 좌표계 중력
    td = np.radians(pitch_deg)
    g_hat = np.array([0.0, np.cos(td), np.sin(td)])

    res = PIPE.inspect_image(lines_uv, lines_xyz, cp, g_hat,
                             rgb_off=rgb_off, seg_backend=backend,
                             sigma_u_px=CALIB.SIGMA_U_PX)
    print()
    print(PIPE.format_report(res))

    # ── 산출물 ──
    out_dir = out_dir or _os.path.dirname(_os.path.abspath(path))
    base = _os.path.splitext(_os.path.basename(path))[0]
    seg_png = seg_png or _os.path.join(out_dir, f"{base}_seg.png")
    excel = excel or _os.path.join(out_dir, f"{base}_검측결과.xlsx")

    seg = REPORT.save_segmentation(seg_png, res,
                                   base_image=(rgb_off if rgb_off is not None
                                               else rgb),
                                   shape=rgb.shape)
    meta = {"입력 이미지": _os.path.basename(path),
            "레이저 OFF 이미지": (_os.path.basename(off_path) if off_path
                          else "없음 (초록채널 분리)"),
            "측정 거리(m)": standoff_m,
            "장비 하향각(°)": pitch_deg,
            "이미지 해상도": f"{rgb.shape[1]} × {rgb.shape[0]}",
            "검출 선 수": len(detected),
            "삼각측량 점 수": n3d}
    if site:
        meta = {"현장": site, **meta}
    caveats = []
    if backend == "geom":
        caveats.append("기하 전용 백엔드는 동바리/기둥/철근과 벽/거푸집/조적을 "
                       "구분하지 못한다. 적용 KCS 허용치가 달라지므로 "
                       "부재 종류는 사람이 확인해야 한다.")
    for k, v in CALIB.provenance().items():
        if v[0] == "assumed" and k in ("f_px", "fov_deg", "n_lines", "alpha_i"):
            caveats.append(f"{k} 가 실측이 아닌 가정값이다 — {v[1]}")

    xl = XLS.save_excel(excel, res, meta=meta, seg_image_path=seg,
                        extra_caveats=caveats)
    print()
    print(f"세그멘테이션 이미지: {seg}")
    print(f"엑셀 조서:          {xl}")
    return 0


def main():
    ap = argparse.ArgumentParser(description="렌더/촬영 이미지 한 장 검측")
    ap.add_argument("image", help="레이저 ON 이미지 (PNG 등)")
    ap.add_argument("--off", default=None, help="레이저 OFF 이미지 (차영상·세그멘테이션용)")
    ap.add_argument("--standoff", type=float, default=1.2, help="측정 거리 [m]")
    ap.add_argument("--pitch", type=float, default=0.0,
                    help="장비를 아래로 숙인 각 [°]. 0 이면 정면")
    ap.add_argument("--backend", default="geom", choices=["geom", "sam", "vlm"],
                    help="세그멘테이션 백엔드 (실이미지에는 정답 마스크가 없다)")
    ap.add_argument("--profile", default=None,
                    help="사양 프로파일 (legacy / pdf / improved)")
    ap.add_argument("--check", action="store_true",
                    help="검측 대신 이미지의 격자를 읽어 사양과 대조")
    ap.add_argument("--out", default=None, help="산출물 폴더 (기본: 이미지와 같은 폴더)")
    ap.add_argument("--excel", default=None, help="엑셀 조서 경로")
    ap.add_argument("--seg-png", default=None, help="세그멘테이션 이미지 경로")
    ap.add_argument("--site", default=None, help="현장명 (조서 머리말)")
    a = ap.parse_args()
    if a.profile:
        CALIB.use_profile(a.profile)
    if a.check:
        return check(a.image, a.standoff)
    return inspect(a.image, a.off, a.standoff, a.pitch, a.backend,
                   out_dir=a.out, excel=a.excel, seg_png=a.seg_png,
                   site=a.site)


if __name__ == "__main__":
    raise SystemExit(main())
