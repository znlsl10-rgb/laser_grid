#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
load_capture.py — Isaac raycast 내보내기 폴더를 검측 입력으로
========================================================================
한 폴더에 아래가 들어 있는 형식을 읽는다.

  camera_params.json   f_px·주점·기선·격자·rig_transform·screenshot_size
  cast_pixels.json     {선ID: {fixed, angle_deg, points:[{uv, alpha_deg,
                       beta_deg, xyz_world}, ...]}}
  CAM.png / CAST.png   화면 캡처 (선택 — 결과 이미지 배경으로만 쓴다)

두 가지를 데이터에서 직접 복원한다. 헤더 값을 그대로 믿으면 틀린다.

1) 해상도
   uv 는 screenshot_size(1269×1063) 기준인데 f_px·주점은 sensor_size
   (2448×2048) 기준이다. 그대로 쓰면 삼각측량이 통째로 어긋난다.
   실제로 동봉된 xyz_result.json 이 이 상태에서 만들어져 벽까지 거리가
   1.5m 대신 16m 로 나와 있다. uv 를 센서 단위로 올려 맞춘다. uv 가
   정수가 아니라 해석적 실수값이라 이 환산으로 잃는 정보는 없다.

2) 카메라 자세
   camera_forward_world 만 있고 롤(광축 둘레 회전)이 없다. 아래를 보는
   촬영에서는 forward 와 월드 up 이 거의 나란해져 롤을 세울 수가 없다.
   게다가 캡처마다 uv 축 방향이 다르다(이 표본에서 벽체는 180° 돌아
   있고 동바리는 그렇지 않다). 그래서 자세를 헤더에서 만들지 않고
   (xyz_world, uv) 대응에서 Kabsch 로 맞춘다. 어떤 규약으로 내보냈든
   데이터 자신과 일치하는 자세가 나온다.

H선을 쓰지 않는 이유
   내보내기에는 H선 점에도 alpha_deg 가 들어 있다. 그러나 실장비는
   H선 위 임의 점의 α 를 알 수 없다(V×H 교점에서만 회복된다). 그것을
   쓰면 시뮬레이션에서만 되는 검증이 되므로 V선만 쓴다.

실행:
  python3 load_capture.py <폴더> [--pitch 자동] [--out 출력폴더]
  python3 load_capture.py <상위폴더> --all
========================================================================
"""
import argparse, json, os
import numpy as np
import importlib.util as _ilu


def _load(name):
    spec = _ilu.spec_from_file_location(
        name, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           f"{name}.py"))
    m = _ilu.module_from_spec(spec); spec.loader.exec_module(m)
    return m


CALIB = _load("calibration")
PIPE = _load("pipeline_region")
DETECT = _load("A_선검출")
_EQ7 = _load("eq7_laser_plane")

# 검출선과 정답선을 같은 선으로 볼 최대 거리 [px].
# 이 사양의 격자 간격은 화면에서 45~55px 이므로, 5px 이면 이웃 선과
# 혼동될 여지 없이 "찾았다/못 찾았다" 를 가른다.
MATCH_TOL_PX = 5.0
REPORT = _load("report")
XLS = _load("report_excel")


# =====================================================================
# 자세 복원
# =====================================================================
def fit_camera_rotation(P_world, uv, cam_pos, f, cx, cy):
    """
    (월드 3D 점, 화소 좌표) 대응에서 카메라 회전을 맞춘다.

    카메라 위치는 이미 알고 있으므로 남은 미지수는 회전뿐이다. 각 점의
    월드 방향 단위벡터와 화소가 가리키는 카메라 좌표계 단위벡터를
    맞추는 직교 프로크루스테스 문제이고, SVD 로 닫힌 해가 나온다.

        d_i = normalize(u−c_x, v−c_y, f)      카메라 좌표계 시선
        w_i = normalize(P_i − C)              월드 시선
        R  = argmin Σ |R·d_i − w_i|²

    반사(det<0)는 회전이 아니므로 마지막 특이벡터를 뒤집어 막는다.
    """
    d = np.stack([uv[:, 0] - cx, uv[:, 1] - cy, np.full(len(uv), f)], axis=1)
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    w = P_world - np.asarray(cam_pos, float)
    w /= np.linalg.norm(w, axis=1, keepdims=True)
    U, _, Vt = np.linalg.svd(w.T @ d)
    S = np.eye(3)
    S[2, 2] = np.sign(np.linalg.det(U @ Vt))
    R = U @ S @ Vt                       # 카메라→월드
    resid = np.degrees(np.arccos(np.clip(
        np.einsum("ij,ij->i", (R @ d.T).T, w), -1, 1)))
    return R, float(np.median(resid))


# =====================================================================
# 폴더 읽기
# =====================================================================
def load_folder(path, world_up=(0.0, 0.0, 1.0), stride=1,
                params_path=None, truth_path=None):
    """
    내보내기 폴더 하나를 검측 입력 묶음으로 바꾼다.

    params_path·truth_path 를 주면 폴더 규약 대신 그 파일을 쓴다. 코랩처럼
    파일을 하나씩 올리는 곳에서 폴더 구조를 강요하지 않기 위해서다.

    Returns
    -------
    dict — lines_pixels, line_angles, camera_params, g_hat, diag, meta
    """
    cp_raw = json.load(open(
        params_path or os.path.join(path, "camera_params.json"),
        encoding="utf-8"))
    cast = json.load(open(
        truth_path or os.path.join(path, "cast_pixels.json"),
        encoding="utf-8"))

    SW, SH = cp_raw["sensor_size"]
    sw, sh = cp_raw.get("screenshot_size", [SW, SH])
    su, sv = SW / float(sw), SH / float(sh)      # 화면→센서 (가로·세로 따로)
    f = float(cp_raw["camera"]["f_px"])
    cx = float(cp_raw["camera"]["cx_px"])
    cy = float(cp_raw["camera"]["cy_px"])
    b = float(cp_raw["baseline_m"])

    # ── V선·H선을 모두 모은다 ──
    # 예전에는 V선만 실었다. 근거는 "H선은 깊이가 안 풀린다" 였는데, 그
    # 판단을 여기서 내리면 H선이 실제로 어떤 상태인지 조서에 남지 않는다.
    # 전부 싣고, 깊이에 쓸지 말지는 eq7 의 이득 g 가 데이터로 판정하게 한다
    # (triangulate_lines 가 g=∞ 인 선을 사유와 함께 걸러낸다).
    lines_pixels, line_angles, P_all, uv_all = {}, {}, [], []
    n_h = 0
    line_world = {}
    for lid, ln in cast.items():
        pts = ln["points"][::stride]
        uv = np.array([[p["uv"][0] * su, p["uv"][1] * sv] for p in pts])
        P = np.array([p["xyz_world"] for p in pts], float)
        P_all.append(P); uv_all.append(uv)          # 자세 적합엔 V·H 모두 쓴다
        if ln["fixed"] != "alpha":
            n_h += 1
        lines_pixels[lid] = uv
        line_world[lid] = P
        line_angles[lid] = {
            "fixed": ln["fixed"],
            "angle_rad": float(np.radians(ln["angle_deg"]))}
    P_all = np.concatenate(P_all); uv_all = np.concatenate(uv_all)

    rt = cp_raw.get("rig_transform") or {}
    C = np.array(rt["camera_pos_world"], float)
    L = np.array(rt["laser_pos_world"], float)
    R, resid_deg = fit_camera_rotation(P_all, uv_all, C, f, cx, cy)
    R_raw = R.copy()          # 내보내기 원본 화소 규약에 맞는 자세

    # 기선 방향 확인 — eq1 은 카메라가 조사기의 +x 쪽에 있다고 본다.
    t_cam = (C - L) @ R                  # 조사기→카메라, 카메라 좌표계
    flipped = False
    if t_cam[0] < 0:
        # 내보내기가 180° 돌아 있다. 화소를 주점 기준으로 뒤집고 자세의
        # x·y 축도 같이 뒤집어 규약을 되돌린다(광축 둘레 180° 회전).
        flipped = True
        for lid in lines_pixels:
            lines_pixels[lid] = np.stack(
                [2 * cx - lines_pixels[lid][:, 0],
                 2 * cy - lines_pixels[lid][:, 1]], axis=1)
        R = R @ np.diag([-1.0, -1.0, 1.0])
        t_cam = (C - L) @ R

    g_hat = np.array(world_up, float) * -1.0 @ R    # 조사기 좌표계 중력
    g_hat /= np.linalg.norm(g_hat)

    # ── 발사각을 라벨이 아니라 데이터에서 구한다 ──
    # 내보내기의 angle_deg 는 화소와 부호가 맞지 않는 경우가 있다. 이 표본은
    # H선 번호가 코드와 반대로 매겨져 있고(정답 H0 이 화면 아래), 벽체는
    # u 축까지 뒤집혀 있다. 라벨을 믿고 예측을 세우면 선검출이 엉뚱한 곳을
    # 뒤진다. 각 선의 raycast 점에서 직접 구하면 규약과 무관하게 맞는다.
    #
    #   Pl = (P_world − L) @ R,   tanα = Pl_x/Pl_z,   tanβ = Pl_y/Pl_z
    # 원본 규약에서의 부호 있는 기선. 180° 돌아 있으면 −b 가 된다.
    # 발사각만 부호를 뒤집고 기선을 그대로 두면 예측이 2·f·b/Z (이 표본에서
    # 160px) 만큼 어긋난다. 실제로 그렇게 두자 벽체에서 21선 중 8선을
    # 놓쳤다. 회전 하나에 두 항이 함께 뒤집힌다:
    #   u_raw = 2c_x − u_std = f·tan(−α) + f·b/Z + c_x
    b_raw = float(((C - L) @ R_raw)[0])
    angles_raw = {}
    for lid, ln in cast.items():
        P = np.array([p["xyz_world"] for p in ln["points"][::100]], float)
        Pl = (P - L) @ R_raw
        ok = Pl[:, 2] > 1e-6
        if not ok.any():
            continue
        Pl = Pl[ok]
        a = float(np.median(np.arctan2(Pl[:, 0], Pl[:, 2])))
        bta = float(np.median(np.arctan2(Pl[:, 1], Pl[:, 2])))
        angles_raw[lid] = {"fixed": ln["fixed"],
                           "angle_rad": a if ln["fixed"] == "alpha" else bta,
                           "label_deg": ln["angle_deg"],
                           "data_deg": float(np.degrees(
                               a if ln["fixed"] == "alpha" else bta))}

    # ── 선마다 레이저 평면을 **데이터에서** 맞춘다 ──
    # 발사각 하나로 평면을 세우면 "V평면은 Y축을 품는다" 같은 가정이 따라
    # 들어온다. raycast 점은 정의상 그 평면 위에 있으므로, 레이저 원점을
    # 지나는 평면을 직접 맞추면 가정 없이 법선이 나온다. 격자를 굴려서
    # 오든 비스듬히 오든 같은 코드가 받는다.
    #
    #   Pl = (P_world − L) @ R  (조사기 좌표계) → SVD 최소 특이벡터 = 법선
    #
    # 이렇게 얻은 법선의 n_x 가 0 이면 그 선은 원리적으로 깊이가 없다.
    # 이 표본의 H선이 정확히 그 경우이고, 가정이 아니라 측정으로 확인된다.
    for lid, P in line_world.items():
        Pl = (P - L) @ R
        # 원점을 지나는 평면이므로 중심화하지 않는다(중심화하면 원점
        # 통과 제약이 풀려 다른 평면이 나온다).
        _, sv_, Vt = np.linalg.svd(Pl, full_matrices=False)
        n = Vt[-1]
        n = n / np.linalg.norm(n)
        resid_mm = float(np.max(np.abs(Pl @ n))) * 1000.0
        info = line_angles[lid]
        info["normal"] = n.tolist()
        info["depth_gain"] = _EQ7.depth_gain(n)
        info["plane_resid_mm"] = round(resid_mm, 4)

    camera_params = {"f_px": f, "b_m": b, "cx_px": cx, "cy_px": cy,
                     "resolution": [SW, SH]}
    diag = {
        "화면→센서 배율": (round(su, 4), round(sv, 4)),
        "자세 적합 잔차(°)": round(resid_deg, 4),
        "uv 180° 뒤집힘": flipped,
        "기선 벡터(카메라좌표, m)": [round(x, 4) for x in t_cam],
        "기선 x성분 대비 잔여(mm)": round(
            float(np.hypot(t_cam[1], t_cam[2])) * 1000, 1),
        "V선 수": len(lines_pixels) - n_h, "H선 수": n_h,
        "격자점 수(V+H)": int(sum(len(v) for v in lines_pixels.values())),
        "중력(조사기좌표)": [round(x, 4) for x in g_hat],
        "장비 하향각(°)": round(float(np.degrees(np.arctan2(
            g_hat[2], g_hat[1]))), 2),
    }
    meta = {"case": cp_raw.get("case_name"),
            "captured_at": cp_raw.get("captured_at"),
            "grid": cp_raw.get("grid"), "sensor": [SW, SH],
            "screenshot": [sw, sh]}
    # 라벨과 실제 발사각이 얼마나 어긋나는지
    lbl = [abs(v["data_deg"] - v["label_deg"]) for v in angles_raw.values()]
    diag["발사각 라벨 vs 실측 최대차(°)"] = round(max(lbl), 3) if lbl else None

    return {"lines_pixels": lines_pixels, "line_angles": line_angles,
            "camera_params": camera_params, "g_hat": g_hat, "R_cam": R,
            "R_raw": R_raw, "angles_raw": angles_raw, "b_raw": b_raw,
            "diag": diag, "meta": meta, "raw": cp_raw, "cast": cast,
            "stride": int(stride)}


# =====================================================================
# 선검출 정확도 — 화소 단계가 맞아야 3D 가 맞는다
# =====================================================================
def _flip_about_principal(im, cx, cy, resample=None):
    """
    주점 (cx, cy) 를 중심으로 이미지를 180° 돌린다.

    PIL 의 ROTATE_180 은 u → (w−1) − u 로 **이미지 중심** 기준이다.
    화소 좌표는 주점 기준으로 2·c − u 로 되돌리므로, 주점이 이미지
    중심과 다르면 그만큼 어긋난다. 이 표본에서는 폭 1269, 주점 634.5 라
    정확히 1px 차이가 났고, 그것이 벽체 캡처의 u 오차 중앙값 −0.96px 로
    그대로 나타났다. 센서 환산 1px 은 1.5m 에서 깊이 10mm 다.
    """
    from PIL import Image
    if resample is None:
        resample = Image.BICUBIC
    # AFFINE 은 출력(x,y) → 입력(ax+by+c, dx+ey+f) 로 역방향 사상이다.
    return im.transform(im.size, Image.AFFINE,
                        (-1, 0, 2.0 * cx, 0, -1, 2.0 * cy), resample=resample)


def evaluate_line_detection(path, cap, image_name="CAST.png",
                            image_path=None):
    """
    렌더 이미지에 A_선검출 을 돌려 raycast 정답 화소와 대조한다.

    왜 재야 하는가
    -------------
    삼각측량은 Z = f·b / (f·tanα − (u − c_x)) 다. 분모가 화소 차이라,
    u 가 흔들린 만큼 그대로 깊이가 흔들린다. 민감도는

        dZ = Z² / (f·b) · du

    이고 이 장비의 legacy 사양(f=1593px, b=150mm)에서는 2.7m 거리에서
    1px 이 12mm 다. 즉 뒤쪽 검측식이 아무리 정확해도 화소 단계에서
    1px 이 틀리면 그 자리에서 끝난다.

    무엇과 대조하나
    --------------
    이 내보내기에는 cast_pixels.json 에 raycast 로 구한 정답 화소가 들어
    있다. 같은 장면의 렌더 이미지(CAST.png)에 선검출을 돌리면 검출값과
    정답을 직접 뺄 수 있다. 검측 파이프라인은 정답 화소를 그대로 썼으므로
    이 비교가 곧 "실장비에서 무엇이 더 나빠지는가" 를 보여준다.

    비교 방법
    --------
    V선은 행(v)을 따라가며 u 를 본다. 정답과 검출의 샘플 밀도가 다르므로
    인덱스로 맞추지 않고, 정답을 검출점의 v 위치에 선형보간해 재추출한 뒤
    같은 v 에서 u 차이를 잰다.
    """
    cast = cap["cast"]
    cp_raw = cap["raw"]
    fp = image_path or os.path.join(path, image_name)
    if not os.path.exists(fp):
        return None
    try:
        from PIL import Image
    except ImportError:
        return None

    # 원본 화소 그대로 쓴다. 이전 판은 180° 뒤집힌 캡처의 이미지를 돌려
    # 맞췄는데, 주점 기준 반전(2c−u)은 화소 격자 위로 정확히 떨어지지 않아
    # 리샘플링이 들어가고 그만큼(−0.96px) 계통 오차가 생겼다. 원본 이미지와
    # 원본 정답 화소는 이미 서로 맞으므로(실측 −0.03px, σ 0.28px) 아무것도
    # 돌리지 않는 것이 정확하다. 대신 발사각을 원본 규약으로 넣는다.
    im = Image.open(fp).convert("RGB")
    img = np.asarray(im)
    w, h = im.size
    SW, SH = cp_raw["sensor_size"]
    su, sv = SW / float(w), SH / float(h)
    f_img = float(cp_raw["camera"]["f_px"]) / su
    cx_img = float(cp_raw["camera"]["cx_px"]) / su
    cy_img = float(cp_raw["camera"]["cy_px"]) / sv
    b = float(cp_raw["baseline_m"])
    grid = cp_raw.get("grid") or {}
    z_ref = _median_depth(cap)
    flipped = bool(cap["diag"]["uv 180° 뒤집힘"])

    cp = {"f_px": f_img, "b_m": cap.get("b_raw", b), "cx_px": cx_img,
          "cy_px": cy_img,
          "resolution": [w, h], "image_w": w, "image_h": h,
          "n_v": grid.get("n_vertical", 21), "n_h": grid.get("n_horizontal", 21),
          "fov_h_deg": grid.get("fov_deg"), "fov_v_deg": grid.get("fov_deg"),
          "standoff_z": z_ref, "z_range": _depth_span(cap)}
    # 발사각은 라벨이 아니라 데이터에서 구한 값을 쓴다. 이 표본은 H선 번호가
    # 코드와 반대로 매겨져 있고 벽체는 u 축까지 뒤집혀 있어, 라벨을 그대로
    # 넣으면 예측 격자가 화면 반대편을 가리킨다.
    line_angles = {lid: {"fixed": v["fixed"], "angle_rad": v["angle_rad"]}
                   for lid, v in cap["angles_raw"].items()}
    try:
        det = DETECT.detect(img, {}, line_angles, cp, multi_surface=True)
    except Exception as e:
        return {"error": f"선검출 실패: {e}"}

    # ── 검출선 ↔ 정답선을 **위치로** 짝짓는다 ──
    # ID 로 짝지으면 안 된다. 이 내보내기는 H선 번호가 코드와 반대로
    # 매겨져 있어(정답 H0 이 화면 아래, 코드 H0 이 화면 위) ID 로 빼면
    # 970px 짜리 오차가 나온다. 실제로 알고 싶은 것은 "선을 찾았는가,
    # 얼마나 정확한가" 이므로 위치로 짝짓고, 번호가 맞는지는 따로 센다.
    def _key(arr, fixed):
        return float(np.median(arr[:, 0 if fixed == "alpha" else 1]))

    det_info = {}
    for lid, pts in det.items():
        d = np.array(pts, float)
        if len(d) >= 3:
            det_info[lid] = (d, _key(d, "alpha" if lid.startswith("V") else "beta"))

    rows, all_err = [], []
    n_id_ok = {"V": 0, "H": 0}
    n_id_tot = {"V": 0, "H": 0}
    for lid, ln in sorted(cast.items(),
                          key=lambda kv: (kv[0][0], int(kv[0][1:]))):
        gt = np.array([[p["uv"][0], p["uv"][1]] for p in ln["points"]], float)
        z_line = float(np.median([_depth_of(p, cap) for p in ln["points"][::200]]))
        axis = lid[0]
        n_id_tot[axis] += 1
        g_key = _key(gt, ln["fixed"])

        # 같은 방향의 검출선 중 가장 가까운 것
        cand = [(abs(k - g_key), l, d) for l, (d, k) in det_info.items()
                if l[0] == axis]
        row = {"lid": lid, "fixed": ln["fixed"], "n_gt": len(gt),
               "z_m": round(z_line, 3), "gt_pos": round(g_key, 1)}
        if not cand:
            row.update(n_det=0, matched=None, err_med=None, err_rms=None,
                       err_p95=None, err_max=None, id_ok=False,
                       note="검출선 없음")
            rows.append(row); continue
        gap, mlid, d = min(cand, key=lambda t: t[0])
        row["matched"] = mlid
        row["match_gap_px"] = round(gap, 2)
        row["id_ok"] = (mlid == lid)
        if row["id_ok"]:
            n_id_ok[axis] += 1
        if gap > MATCH_TOL_PX:
            row.update(n_det=0, err_med=None, err_rms=None, err_p95=None,
                       err_max=None,
                       note=f"미검출 — 가장 가까운 검출선이 {gap:.1f}px 떨어져 있음")
            rows.append(row); continue

        row["n_det"] = len(d)
        if ln["fixed"] == "alpha":          # V선: v 를 따라가며 u 오차
            o = np.argsort(gt[:, 1])
            ref = np.interp(d[:, 1], gt[o, 1], gt[o, 0])
            e = d[:, 0] - ref
        else:                                # H선: u 를 따라가며 v 오차
            o = np.argsort(gt[:, 0])
            ref = np.interp(d[:, 0], gt[o, 0], gt[o, 1])
            e = d[:, 1] - ref
        # 계통 편차와 무작위 오차를 갈라 놓는다. 둘은 성질도 대책도 다르다.
        #   계통(중앙값) — 좌표 규약·주점·기선 같은 것이 어긋난 것. 소프트웨어로 고친다.
        #   무작위(중앙값 둘레 표준편차) — 이것이 진짜 검출 정밀도 σ_u 다.
        # 화소 오차를 그대로 삼각측량에 넣어 **실제 깊이 차이**도 잰다.
        # 환산식(dZ = Z²/(f·b)·du)은 선형 근사이고, 무엇보다 "화소 0.3px"
        # 이라는 숫자만으로는 얼마나 나쁜지 感이 오지 않는다.
        if ln["fixed"] == "alpha":
            dz = _depth_diff_mm(d, gt, lid, cap, cp_raw, su, flipped)
            row.update(dz_med_mm=dz[0], dz_noise_mm=dz[1], dz_p95_mm=dz[2])
        row.update(err_med=float(np.median(e)),
                   err_noise=float(np.std(e - np.median(e))),
                   err_rms=float(np.sqrt(np.mean(e ** 2))),
                   err_p95=float(np.percentile(np.abs(e), 95)),
                   err_max=float(np.abs(e).max()), note=None)
        rows.append(row)
        if ln["fixed"] == "alpha":
            all_err.append(e)

    # 렌더가 안티에일리어싱 없이 이진으로 그려졌는지 확인한다. 그렇다면
    # 선 위치가 0.5px 격자에 갇히고, 그것이 서브픽셀 정밀도의 하한이 된다.
    gimg = (img[:, :, 1].astype(float)
            - 0.5 * (img[:, :, 0].astype(float) + img[:, :, 2].astype(float)))
    lit = gimg[gimg > gimg.max() * 0.15]
    binary = bool(len(lit) and (lit > gimg.max() * 0.9).mean() > 0.9)
    q_px = 0.5 / np.sqrt(12.0) * su if binary else None
    q_mm = (z_ref ** 2 / float(cp_raw["camera"]["f_px"]) / b * q_px * 1000.0
            if q_px else None)

    E = np.concatenate(all_err) if all_err else np.zeros(0)
    v_rows = [r for r in rows if r["fixed"] == "alpha"]
    ok = [r for r in v_rows if r["err_rms"] is not None]
    missed = [r for r in v_rows if r["err_rms"] is None]

    # ── 계열별(V 세로 / H 가로) 검출 성적 ──
    # 예전에는 V선만 집계했다. 가로선은 추적까지 해 놓고 성적을 아무 데도
    # 남기지 않아, "가로축이 검출되고 있는가" 를 조서에서 확인할 방법이
    # 없었다. 검출 정확도(화소)와 깊이 기여(이득 g)는 별개의 문제이므로
    # 둘을 같이 싣는다 — 가로선은 잘 찾아도 깊이를 못 줄 수 있고, 그
    # 이유는 검출이 아니라 기하에 있다.
    la = cap.get("line_angles", {})
    families = {}
    for pre, fx, nm, axis_nm in (("V", "alpha", "V선(세로)", "u"),
                                 ("H", "beta", "H선(가로)", "v")):
        fr = [r for r in rows if r["fixed"] == fx]
        if not fr:
            continue
        fok = [r for r in fr if r["err_rms"] is not None]
        e_all = [r["err_med"] for r in fok]
        n_all = [r["err_noise"] for r in fok]
        gains = [la[r["lid"]]["depth_gain"] for r in fr
                 if r["lid"] in la and "depth_gain" in la[r["lid"]]]
        gfin = [g for g in gains if np.isfinite(g) and g < 1e6]
        # 정답선 자체가 그 축으로 얼마나 움직이는가 = 그 선이 담고 있는
        # 깊이 신호의 크기. 검출 잡음보다 작으면 원리적으로 못 쓴다.
        swing = []
        for r in fr:
            ln = cast.get(r["lid"])
            if ln:
                a = np.array([p["uv"][0 if fx == "alpha" else 1]
                              for p in ln["points"]], float)
                swing.append(float(a.max() - a.min()))
        # 오차를 세 갈래로 나눈다. 셋은 원인도 대책도 다르다.
        #   계통편차   : 모든 선이 같은 방향으로 밀린 양. 좌표 규약·주점·
        #                기선이 어긋난 것이며 소프트웨어로 없앨 수 있다.
        #   선간편차   : 선마다 밀린 양이 서로 다른 산포. 발사각 α_i 모델이
        #                실제 DOE 와 다른 것이 주원인 — 출고 캘리브레이션으로
        #                실측 α_i 를 넣으면 준다.
        #   선내잡음   : 한 선을 따라가며 생기는 산포. 이것이 진짜 서브픽셀
        #                검출 정밀도이고, 렌더 양자화가 하한을 만든다.
        # 통합 σ 는 셋이 섞인 값이라, 어디를 고쳐야 하는지는 이 분해를
        # 봐야 알 수 있다.
        pooled = None
        if fok:
            ev = [r for r in fok]
            pooled = float(np.sqrt(np.mean(
                [r["err_noise"] ** 2 + (r["err_med"] - np.median(e_all)) ** 2
                 for r in ev])))
        families[pre] = {
            "이름": nm, "측정축": axis_nm,
            "선 수": len(fr), "검출": len(fok),
            "미검출": [r["lid"] for r in fr if r["err_rms"] is None],
            "계통편차_px": (round(float(np.median(e_all)), 4) if e_all else None),
            "선간편차_px": (round(float(np.std(e_all)), 4)
                        if len(e_all) > 1 else None),
            "선내잡음_px": (round(float(np.median(n_all)), 4) if n_all else None),
            "통합오차_px": (round(pooled, 4) if pooled is not None else None),
            "정답선_변동폭_px": (round(float(np.median(swing)), 4)
                            if swing else None),
            "깊이이득_중앙": (round(float(np.median(gfin)), 3) if gfin else None),
            "깊이가능": sum(1 for g in gains
                         if np.isfinite(g) and g <= _EQ7.MAX_DEPTH_GAIN),
        }

    # 화소 오차를 깊이 오차로 환산 — 이 숫자가 최종 정확도를 좌우한다.
    #   dZ = Z²/(f·b) · du       (센서 화소 기준)
    f_sensor = float(cp_raw["camera"]["f_px"])
    bias = float(np.median(E)) if len(E) else None
    noise = float(np.std(E - np.median(E))) if len(E) else None
    rms_sensor = (float(np.sqrt(np.mean(E ** 2))) * su) if len(E) else None
    to_mm = (lambda px: z_ref ** 2 / (f_sensor * b) * (px * su) * 1000.0)
    dz_mm = to_mm(float(np.sqrt(np.mean(E ** 2)))) if len(E) else None
    return {
        "image": os.path.basename(fp), "image_size": [w, h],
        "flipped": flipped,
        "scale_to_sensor": round(su, 4), "scale_to_sensor_v": round(sv, 4),
        "_detected": {k: np.asarray(v, float) for k, v in det.items()},
        "f_px_image": round(f_img, 1), "f_px_sensor": f_sensor,
        "z_ref_m": round(z_ref, 3),
        "n_lines_gt": len(cast), "n_lines_det": len(det),
        "rows": rows,
        "err_med_px": (round(float(np.median(E)), 4) if len(E) else None),
        "err_rms_px": (round(float(np.sqrt(np.mean(E ** 2))), 4) if len(E) else None),
        "err_p95_px": (round(float(np.percentile(np.abs(E), 95)), 3) if len(E) else None),
        "err_bias_px": (round(bias, 4) if bias is not None else None),
        "err_noise_px": (round(noise, 4) if noise is not None else None),
        "err_rms_sensor_px": (round(rms_sensor, 4) if rms_sensor is not None else None),
        "bias_sensor_px": (round(bias * su, 4) if bias is not None else None),
        "noise_sensor_px": (round(noise * su, 4) if noise is not None else None),
        "depth_err_mm": (round(dz_mm, 3) if dz_mm is not None else None),
        "depth_bias_mm": (round(to_mm(abs(bias)), 3) if bias is not None else None),
        "depth_noise_mm": (round(to_mm(noise), 3) if noise is not None else None),
        "mm_per_px_depth": round(z_ref ** 2 / (f_sensor * b) * su * 1000.0, 3),
        "n_v_matched": len(ok), "n_v_total": len(v_rows),
        "n_v_missed": len(missed),
        "missed_lines": [r["lid"] for r in missed],
        "id_ok": {k: (n_id_ok[k], n_id_tot[k]) for k in ("V", "H")},
        "sigma_u_design_px": CALIB.SIGMA_U_PX,
        "quantization_px": q_px, "quantization_mm": q_mm,
        "dz_med_mm": _agg(v_rows, "dz_med_mm"),
        "dz_noise_mm": _agg(v_rows, "dz_noise_mm"),
        "dz_p95_mm": _agg(v_rows, "dz_p95_mm"),
        "families": families,
        "max_depth_gain": _EQ7.MAX_DEPTH_GAIN,
    }


def _depth_of(point, cap):
    """raycast 점의 조사기 좌표 Z."""
    rt = cap["raw"]["rig_transform"]
    L = np.array(rt["laser_pos_world"], float)
    R = cap["R_cam"]
    return float((np.array(point["xyz_world"], float) - L) @ R[:, 2])


def _depth_span(cap, lo_q=2.0, hi_q=98.0):
    """
    장면의 깊이 구간 [m]. 선검출 예측 밴드가 이만큼을 덮어야 한다.

    한 장면에 깊이가 여럿이면(벽 뒤 + 기둥 앞) 시차가 선마다 달라진다.
    실장비는 이 구간을 도면이나 직전 촬영에서 얻거나, 사양의 작업거리
    (1.0~1.5m)를 그대로 쓴다. 여기서는 raycast 깊이 분포에서 잡는다.
    """
    zs = []
    for lid, ln in cap["cast"].items():
        if ln["fixed"] != "alpha":
            continue
        zs += [_depth_of(p, cap) for p in ln["points"][::300]]
    if not zs:
        return None
    zs = np.array(zs, float)
    return [float(np.percentile(zs, lo_q)), float(np.percentile(zs, hi_q))]


def _median_depth(cap):
    """장면 대표 거리 — 선검출 예측 밴드의 시차항에 쓰인다."""
    zs = []
    for lid, ln in cap["cast"].items():
        if ln["fixed"] != "alpha":
            continue
        zs.append(_depth_of(ln["points"][len(ln["points"]) // 2], cap))
    return float(np.median(zs)) if zs else 1.2


def _remove_laser(rgb, ridge_thresh=8.0, grow=2, passes=4, med_size=15):
    """
    사진에서 초록 레이저선만 지운다.

    검측 결과를 얹을 배경은 장면만 보이는 편이 낫다. 원본 격자가 남아
    있으면 검출점과 겹쳐 무엇이 결과인지 구분되지 않고, 이 내보내기의
    CAM.png 격자는 화면 등간격으로 그려져 있어 발사각을 따르는 검출점과
    애초에 겹치지도 않는다(간격 47.2~48.1px vs 실제 44~57px).

    초록 과잉분 G − (R+B)/2 만으로는 못 가른다. 이 렌더는 장면 자체가
    옅게 초록을 띠어(중앙값 6) 문턱을 어디에 두든 배경이 함께 잡히거나
    흐린 선이 남는다. 대신 **얇은 능선**인지를 본다 — 초록 과잉분에서
    그 지역 중앙값을 빼면 넓게 깔린 색조는 사라지고 폭 몇 화소짜리 선만
    남는다. 창(med_size)은 선폭보다 충분히 커야 한다.

    잡은 자리는 주변 성한 화소의 평균으로 메운다. 창을 조금씩 키우며
    몇 번 돌린다. G 만 눌러 놓으면 회색 줄이 남는다 — 선 자리의 R·B 도
    이미 레이저 반사로 들떠 있기 때문이다.
    """
    a = np.asarray(rgb, float).copy()
    g = a[:, :, 1] - 0.5 * (a[:, :, 0] + a[:, :, 2])
    try:
        from scipy.ndimage import median_filter
        ridge = g - median_filter(g, size=int(med_size), mode="nearest")
    except Exception:
        ridge = g - float(np.median(g))
    m = ridge > ridge_thresh
    if not m.any():
        return a
    try:
        from scipy.ndimage import binary_dilation, uniform_filter
    except Exception:
        rb = 0.5 * (a[:, :, 0] + a[:, :, 2])
        a[m, 1] = rb[m]
        return a

    if grow > 0:
        m = binary_dilation(m, iterations=int(grow))
    valid = (~m).astype(float)
    for k in range(passes):
        win = 5 + 4 * k
        w = uniform_filter(valid, size=win, mode="nearest")
        for c in range(3):
            ch = a[:, :, c] * valid
            f = uniform_filter(ch, size=win, mode="nearest")
            fill = np.divide(f, w, out=np.zeros_like(f), where=w > 1e-6)
            need = m & (w > 1e-6)
            a[need, c] = fill[need]
        filled = m & (w > 1e-6)
        valid[filled] = 1.0
        m = m & ~filled
        if not m.any():
            break
    return a


def _base_image(path, size, flipped=False, cx_img=None, cy_img=None,
                show_true_laser=False):
    """
    결과 이미지의 배경 — 장면 사진에서 레이저선을 지운 것.

    뒤집힌 캡처라도 이미지를 돌리지 않는다. 주점 기준 반전(2c−u)은 화소
    격자에 정확히 떨어지지 않아 리샘플링이 들어가고, 그만큼 배경이 검출점
    기준에서 밀린다(이 표본 1px = 센서 2px). 대신 그릴 화소 좌표를 배경
    규약으로 되돌린다(REPORT.save_segmentation 의 uv_transform).

    show_true_laser 가 참이면 CAST.png 의 실제 레이저를 옅게 얹는다.
    검출점이 레이저 위에 놓이는지 눈으로 확인할 때만 쓴다.
    """
    try:
        from PIL import Image
    except ImportError:
        return None

    def _load(name):
        fp = os.path.join(path, name)
        if not os.path.exists(fp):
            return None
        im = Image.open(fp).convert("RGB")
        return np.asarray(im.resize((size[0], size[1]), Image.BICUBIC), float)

    scene = _load("CAM.png")
    if scene is None:
        laser = _load("CAST.png")
        return None if laser is None else laser.astype(np.uint8)
    scene = _remove_laser(scene)
    # ── 남은 초록 기운까지 뺀 뒤 흑백으로 ──
    # 능선 제거는 폭 몇 화소짜리 선의 심만 지운다. 이 렌더의 레이저는
    # 그 둘레로 30px 남짓 퍼진 옅은 무리를 달고 있어, 중앙값 창(15px)이
    # 배경과 구분하지 못하고 그대로 남는다. 실제로 결과 이미지에 초록
    # 격자가 비쳐 검출점과 뒤섞였다.
    #
    # 두 단계로 없앤다.
    #   1) 남은 초록 과잉분 G − max(R,B) 를 G 에서 뺀다 → 색조가 사라진다
    #   2) 흑백으로 바꾼다 → 배경에 색이 아예 없어야 부재 색이 읽힌다
    # 장면의 결·모서리·명암은 그대로 남으므로 "어디를 쟀는지" 는 여전히
    # 보인다. 배경은 맥락을 주는 역할이지 그 자체가 결과가 아니다.
    ex = scene[:, :, 1] - np.maximum(scene[:, :, 0], scene[:, :, 2])
    scene[:, :, 1] -= np.clip(ex, 0.0, None)
    gray = (0.299 * scene[:, :, 0] + 0.587 * scene[:, :, 1]
            + 0.114 * scene[:, :, 2])
    scene = np.repeat(gray[:, :, None], 3, axis=2)

    if show_true_laser:
        laser = _load("CAST.png")
        if laser is not None:
            lg = laser[:, :, 1] - 0.5 * (laser[:, :, 0] + laser[:, :, 2])
            al = np.clip(lg / 200.0, 0.0, 1.0)[:, :, None] * 0.55
            scene = scene * (1 - al) + np.array([70.0, 245.0, 110.0]) * al
    return np.clip(scene, 0, 255).astype(np.uint8)


def _agg(rows, key):
    """선별 값의 대표치 — 중앙값."""
    v = [r[key] for r in rows if r.get(key) is not None]
    return round(float(np.median(v)), 3) if v else None


def _depth_diff_mm(det_uv, gt_uv, lid, cap, cp_raw, su, flipped):
    """
    같은 행에서 검출 화소와 정답 화소를 각각 삼각측량해 깊이를 비교한다.

    엑셀의 "깊이 환산" 열은 화소 오차에 dZ = Z²/(f·b) 를 곱한 값이라
    선형 근사다. 이것은 근사가 아니라 실제로 두 번 풀어 뺀 값이다.
    """
    f = float(cp_raw["camera"]["f_px"])
    b = float(cp_raw["baseline_m"])
    cx = float(cp_raw["camera"]["cx_px"])
    # 검측이 실제로 쓰는 발사각을 그대로 쓴다. 내보내기의 angle_deg 라벨은
    # 화소와 부호가 맞지 않는 경우가 있어(H선 역순, 벽체 u축 반전) 그것을
    # 쓰면 깊이가 통째로 틀린다 — 실제로 Z 가 1.55m 대신 0.19m 로 나왔다.
    info = cap["line_angles"].get(lid)
    if info is None:
        return None, None, None
    a_rad = float(info["angle_rad"])
    d = np.asarray(det_uv, float)
    g = np.asarray(gt_uv, float)
    o = np.argsort(g[:, 1])
    u_ref = np.interp(d[:, 1], g[o, 1], g[o, 0])

    # 센서 스케일 + 검측 규약으로
    ud = d[:, 0] * su
    ug = u_ref * su
    if flipped:
        ud, ug = 2 * cx - ud, 2 * cx - ug
    den_d = f * np.tan(a_rad) - (ud - cx)
    den_g = f * np.tan(a_rad) - (ug - cx)
    ok = (np.abs(den_d) > 1e-6) & (np.abs(den_g) > 1e-6)
    if not ok.any():
        return None, None, None
    zd = f * b / den_d[ok]
    zg = f * b / den_g[ok]
    good = np.isfinite(zd) & np.isfinite(zg) & (zd > 0) & (zg > 0)
    if not good.any():
        return None, None, None
    dz = (zd[good] - zg[good]) * 1000.0
    med = float(np.median(dz))
    return (round(med, 3), round(float(np.std(dz - med)), 3),
            round(float(np.percentile(np.abs(dz), 95)), 3))


def save_detection_overlay(path, cap, det, out_png, zoom=6, crop=140,
                           image_path=None):
    """
    선검출이 실제 레이저 위에 얹혔는지 눈으로 확인하는 그림.

    배경은 CAST.png(실제 레이저) 그대로, 그 위에 검출한 점을 자홍색으로
    찍는다. 오른쪽에 한 곳을 확대해 붙인다 — 원본 크기에서는 선폭 2px,
    오차 0.3px 라 겹침 여부가 눈에 보이지 않기 때문이다.

    검측 결과 그림(_세그멘테이션.png)의 배경에는 레이저를 지워 두었다.
    거기 남아 있던 CAM.png 격자는 화면 등간격으로 그려진 것이라 발사각을
    따르는 검출점과 애초에 겹치지 않는다. 겹침 확인은 이 그림으로 한다.
    """
    if not det or det.get("error"):
        return None
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return None
    fp = image_path or os.path.join(path, det.get("image", "CAST.png"))
    if not os.path.exists(fp):
        return None
    im = Image.open(fp).convert("RGB")
    W, H = im.size
    # 레이저를 초록으로 또렷하게
    a = np.asarray(im, float)
    g = a[:, :, 1] - 0.5 * (a[:, :, 0] + a[:, :, 2])
    lay = np.zeros_like(a)
    al = np.clip(g / max(g.max(), 1.0), 0, 1)[:, :, None]
    lay = np.array([40.0, 235.0, 90.0]) * al + 18.0
    im = Image.fromarray(np.clip(lay, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(im)
    pts = []
    for lid, arr in det.get("_detected", {}).items():
        for u, v in np.asarray(arr, float)[::3]:
            d.point((u, v), fill=(236, 72, 153))
            pts.append((u, v))
    if not pts:
        return None

    # 확대 삽입 — 선이 가장 조밀한 중앙부
    cxc, cyc = W // 2, H // 2
    box = (max(0, cxc - crop // 2), max(0, cyc - crop // 2),
           min(W, cxc + crop // 2), min(H, cyc + crop // 2))
    zi = im.crop(box).resize(((box[2] - box[0]) * zoom,
                              (box[3] - box[1]) * zoom), Image.NEAREST)
    canvas = Image.new("RGB", (W + zi.width + 24, max(H, zi.height)), (18, 18, 22))
    canvas.paste(im, (0, 0))
    canvas.paste(zi, (W + 24, 0))
    cd = ImageDraw.Draw(canvas)
    cd.rectangle(box, outline=(255, 255, 0), width=2)
    cd.rectangle([W + 24, 0, W + 24 + zi.width - 1, zi.height - 1],
                 outline=(255, 255, 0), width=2)
    font = REPORT._korean_font(max(16, H // 45))
    txt = (f"초록 = 실제 레이저(CAST)   자홍 = 선검출 결과   "
           f"우측 {zoom}배 확대")
    if font:
        cd.rectangle([W + 30, zi.height + 8, W + 30 + int(cd.textlength(txt, font=font)) + 12,
                      zi.height + 8 + font.size + 10], fill=(18, 18, 22))
        cd.text((W + 36, zi.height + 12), txt, fill=(240, 240, 240), font=font)
    dirn = os.path.dirname(os.path.abspath(out_png))
    if dirn:
        os.makedirs(dirn, exist_ok=True)
    canvas.save(out_png)
    return out_png


def verify_triangulation(cap, stride=None):
    """
    정답 화소를 삼각측량한 3D 를 내보내기의 xyz_world 와 맞대 본다.

    이것이 확인하는 것은 **검출이 아니라 계산**이다. 화소가 완벽하다고
    가정했을 때 f·주점·기선·자세·eq1 이 전부 맞물려 돌아가는지를 본다.
    여기서 오차가 나오면 그 아래 모든 숫자가 의미가 없다.

    선검출 정확도(검출 화소 vs 정답 화소)와는 다른 층이다.
      이 함수      정답 화소 → 3D   vs  진짜 3D      … 계산이 맞는가
      선검출 평가   검출 화소       vs  정답 화소     … 검출이 맞는가
    """
    # lines_pixels 를 만들 때 쓴 stride 를 그대로 써야 점 순서가 맞는다.
    # 다른 값을 쓰면 서로 다른 점을 짝지어 오차가 미터 단위로 나온다.
    stride = cap.get("stride", 1) if stride is None else stride
    R = cap["R_cam"]
    L = np.array(cap["raw"]["rig_transform"]["laser_pos_world"], float)
    lx, _, _ = PIPE.triangulate_lines(cap["lines_pixels"], cap["line_angles"],
                                      cap["camera_params"])
    if not lx:
        return None
    got, truth = [], []
    for lid, pts in lx.items():
        ln = cap["cast"].get(lid)
        if ln is None:
            continue
        src = ln["points"][::stride][:len(pts)]
        P = np.array([p["xyz_world"] for p in src], float)
        if len(P) == 0:
            continue
        Pl = (P - L) @ R
        m = min(len(Pl), len(pts))
        got.append(np.asarray(pts)[:m]); truth.append(Pl[:m])
    if not got:
        return None
    D = np.vstack(got); T = np.vstack(truth)
    e = np.linalg.norm(D - T, axis=1) * 1000.0
    ez = (D[:, 2] - T[:, 2]) * 1000.0
    return {"n_points": int(len(D)),
            "dist_med_mm": round(float(np.median(e)), 5),
            "dist_max_mm": round(float(e.max()), 5),
            "z_med_mm": round(float(np.median(ez)), 5),
            "z_sigma_mm": round(float(ez.std()), 5)}


def detected_lines_sensor(cap, det, families=("V", "H")):
    """
    선검출 결과를 **검측이 쓰는 좌표 규약** 으로 옮긴다.

    검출은 화면 캡처 이미지 위에서 이뤄지고, 검측은 센서 화소 기준이다.
    게다가 이 내보내기는 캡처에 따라 uv 가 180° 돌아 있다. 둘을 맞춰
    주지 않으면 삼각측량이 통째로 어긋난다.

      1) 화면 → 센서 배율 (가로·세로가 다르므로 따로)
      2) 필요하면 주점 기준 반전 (2c − u, 2c − v)

    이미지를 돌리지 않고 좌표만 되돌리는 이유는, 주점이 이미지 중심과
    다르면 회전 리샘플링이 화소 격자에 딱 떨어지지 않아 그만큼 계통
    오차가 남기 때문이다(실측 −0.96px = 깊이 18.5mm).
    """
    if not det or det.get("error"):
        return {}
    cp = cap["camera_params"]
    su, sv = det["scale_to_sensor"], det["scale_to_sensor_v"]
    flip = bool(cap["diag"]["uv 180° 뒤집힘"])
    cx, cy = cp["cx_px"], cp["cy_px"]
    out = {}
    for lid, d in (det.get("_detected") or {}).items():
        if lid[0] not in families:
            continue
        d = np.asarray(d, float)
        if len(d) < 5:
            continue
        uv = d * np.array([su, sv])
        if flip:
            uv = np.stack([2 * cx - uv[:, 0], 2 * cy - uv[:, 1]], axis=1)
        out[lid] = uv
    return out


def verify_depth(lines_uv, lines_xyz, cap):
    """
    삼각측량으로 나온 깊이를 raycast 참값과 직접 맞대 본다.

    왜 따로 재는가
    -------------
    화소 오차(px)와 각도 오차(°)만 보면 정작 궁금한 것이 빠진다 —
    **이 점의 거리가 몇 mm 틀렸나** 이다. 검측식(평면·축 적합)은 수만
    점을 평균하므로 각도는 좋게 나오지만, 평활도는 점별 깊이 오차가
    그대로 결과다. 그래서 점별 깊이를 따로 잰다.

    비교 방법
    --------
    검출점과 정답점은 같은 점이 아니다(선 위 표본 위치가 다르다). 같은
    선에서 스캔축 좌표가 같은 자리의 참값을 선형보간해 꺼내 비교한다.
    V선은 v 를 따라, H선은 u 를 따라 맞춘다.

    참값 3D 는 내보내기의 xyz_world 를 조사기 좌표계로 옮겨 쓴다.
        P_laser = (P_world − L) @ R
    """
    rt = cap["raw"].get("rig_transform") or {}
    if "laser_pos_world" not in rt:
        return None
    L = np.array(rt["laser_pos_world"], float)
    R = cap["R_cam"]
    cx = cap["camera_params"]["cx_px"]
    cy = cap["camera_params"]["cy_px"]
    flip = bool(cap["diag"]["uv 180° 뒤집힘"])
    su = cap["raw"]["sensor_size"][0] / float(
        cap["raw"].get("screenshot_size", cap["raw"]["sensor_size"])[0])
    sv = cap["raw"]["sensor_size"][1] / float(
        cap["raw"].get("screenshot_size", cap["raw"]["sensor_size"])[1])

    dz, dxyz, per_line = [], [], {}
    for lid, uv in lines_uv.items():
        ln = cap["cast"].get(lid)
        if ln is None or lid not in lines_xyz:
            continue
        g_uv = np.array([[p["uv"][0] * su, p["uv"][1] * sv]
                         for p in ln["points"]], float)
        if flip:
            g_uv = np.stack([2 * cx - g_uv[:, 0], 2 * cy - g_uv[:, 1]], axis=1)
        g_xyz = (np.array([p["xyz_world"] for p in ln["points"]], float)
                 - L) @ R
        ax = 1 if ln["fixed"] == "alpha" else 0     # V선은 v, H선은 u 로 맞춘다
        o = np.argsort(g_uv[:, ax])
        s_gt = g_uv[o, ax]
        uv = np.asarray(uv, float)
        got = np.asarray(lines_xyz[lid], float)
        n = min(len(uv), len(got))
        uv, got = uv[:n], got[:n]
        inr = (uv[:, ax] >= s_gt[0]) & (uv[:, ax] <= s_gt[-1])
        if inr.sum() < 5:
            continue
        ref = np.column_stack([np.interp(uv[inr, ax], s_gt, g_xyz[o, k])
                               for k in range(3)])
        e = got[inr] - ref
        dz.append(e[:, 2])
        dxyz.append(np.linalg.norm(e, axis=1))
        per_line[lid] = round(float(np.median(np.abs(e[:, 2]))) * 1000.0, 3)
    if not dz:
        return None
    Z = np.concatenate(dz) * 1000.0
    D = np.concatenate(dxyz) * 1000.0
    return {"n_points": int(len(Z)),
            "z_bias_mm": round(float(np.median(Z)), 3),
            "z_noise_mm": round(float(np.std(Z - np.median(Z))), 3),
            "z_rms_mm": round(float(np.sqrt(np.mean(Z ** 2))), 3),
            "z_p95_mm": round(float(np.percentile(np.abs(Z), 95)), 3),
            "dist_med_mm": round(float(np.median(D)), 3),
            "dist_p95_mm": round(float(np.percentile(D, 95)), 3),
            "per_line_mm": per_line}


def evaluate_end_to_end(path, cap, det, backend="geom", sigma_u_px=None):
    """
    검출 화소로 검측까지 돌려 정답 화소 결과와 맞대 본다.

    선검출 오차를 화소로만 보면 그것이 판정에 얼마나 옮겨 붙는지 알 수
    없다. 한 점의 깊이 오차는 크더라도 면적합이 수만 점을 평균하므로
    각도는 훨씬 정확해진다. 반대로 평활도는 점별 오차가 그대로 남는다.
    같은 장면을 두 번(정답 화소 / 검출 화소) 통과시켜 그 차이를 잰다.
    """
    if not det or det.get("error"):
        return None
    cp = dict(cap["camera_params"])
    su = det["scale_to_sensor"]
    flip = bool(cap["diag"]["uv 180° 뒤집힘"])
    cx, cy = cp["cx_px"], cp["cy_px"]

    lines_det = detected_lines_sensor(cap, det)
    if not lines_det:
        return None
    su_px = CALIB.SIGMA_U_PX if sigma_u_px is None else float(sigma_u_px)
    out = {}
    for tag, lp in (("gt", cap["lines_pixels"]), ("det", lines_det)):
        xyz, uv, ti = PIPE.triangulate_lines(lp, cap["line_angles"], cp)
        if not xyz:
            return None
        r = PIPE.inspect_image(uv, xyz, cp, cap["g_hat"],
                              seg_backend=backend, sigma_u_px=su_px,
                              line_gain=ti["line_gain"],
                              aux_lines_uv=ti.get("skipped_uv"))
        best = {}
        for reg in r["regions"]:
            if reg["status"] != "measured":
                continue
            c = reg["class"]
            if c not in best or reg["n_points"] > best[c]["n_points"]:
                best[c] = reg
        out[tag] = best

    rows = []
    for cls in sorted(set(out["gt"]) | set(out["det"])):
        a, b = out["gt"].get(cls), out["det"].get(cls)
        row = {"class": cls,
               "theta_gt": (round(a["theta_deg"], 4) if a else None),
               "theta_det": (round(b["theta_deg"], 4) if b else None),
               "n_gt": (a["n_points"] if a else 0),
               "n_det": (b["n_points"] if b else 0)}
        if a and b:
            row["dtheta_deg"] = round(abs(a["theta_deg"] - b["theta_deg"]), 4)
            fa = (a.get("flatness") or {}).get("max_gap_mm")
            fb = (b.get("flatness") or {}).get("max_gap_mm")
            if fa is not None and fb is not None:
                row["gap_gt_mm"], row["gap_det_mm"] = fa, fb
                row["dgap_mm"] = round(abs(fa - fb), 3)
        rows.append(row)
    dts = [r["dtheta_deg"] for r in rows if r.get("dtheta_deg") is not None]
    dgs = [r["dgap_mm"] for r in rows if r.get("dgap_mm") is not None]
    return {"rows": rows,
            "max_dtheta_deg": (round(max(dts), 4) if dts else None),
            "max_dgap_mm": (round(max(dgs), 3) if dgs else None)}


# =====================================================================
# 검측 실행
# =====================================================================
def inspect_folder(path, out_dir=None, backend="geom", stride=1, site=None,
                   sigma_u_px=None, eval_detection=True, pc_stride=20,
                   source="detected"):
    """
    내보내기 폴더 하나를 검측한다.

    흐름 — 실장비가 하는 일과 같은 순서다.
      [1] 선검출    CAST.png(레이캐스트 렌더) → 화소 (u,v)
      [2] 3D 복원   화소 → eq7 삼각측량 → (X,Y,Z) 조사기 좌표
                    raycast 참값과 깊이를 직접 대조한다
      [3] 검측      영역분할 → 부재별 수직도·수평도·평활도
      [4] 산출      3D 점군 그림 · 세그멘테이션 · 엑셀 조서

    source : "detected" | "gt"
        검측에 넣을 화소. 기본은 검출 화소다 — 실장비에는 정답 화소가
        없으므로 그래야 현장에서 나올 값이 조서에 실린다. "gt" 는
        "카메라가 완벽했다면" 의 상한을 보고 싶을 때만 쓴다.
    """
    name = os.path.basename(os.path.normpath(path))
    cap = load_folder(path, stride=stride)
    cp = cap["camera_params"]

    print("=" * 70)
    print(f"입력: {name}   ({cap['meta'].get('case')})")
    print("=" * 70)
    for k, v in cap["diag"].items():
        print(f"  {k:<24}{v}")

    tri = None
    try:
        tri = verify_triangulation(cap)
    except Exception:
        tri = None
    if tri:
        print(f"  {'삼각측량 자체 검증':<24}정답 화소 → 3D 오차 중앙 "
              f"{tri['dist_med_mm']:.4f}mm (최대 {tri['dist_max_mm']:.3f}mm)")

    out_dir = out_dir or os.path.join(path, "_검측결과")
    os.makedirs(out_dir, exist_ok=True)

    # ── [1단계] 선검출 — 렌더 이미지에서 화소를 찾는다 ──
    # 실장비에는 정답 화소가 없다. 이미지에서 선을 찾아내는 것이 첫 단계이고,
    # 그 결과가 뒤의 3D·검측을 전부 결정한다. 정답 화소는 여기서 "얼마나
    # 잘 찾았나" 를 재는 자로만 쓴다.
    det_eval = None
    if eval_detection:
        try:
            det_eval = evaluate_line_detection(path, cap)
        except Exception as e:
            det_eval = {"error": f"평가 실패: {e}"}
        if det_eval and not det_eval.get("error"):
            print(f"\n  선검출 대조 ({det_eval['image']}) — 정답 화소 기준")
            print(f"    V선 검출        {det_eval['n_v_matched']}/{det_eval['n_v_total']}"
                  + (f"   미검출 {det_eval['missed_lines']}"
                     if det_eval["missed_lines"] else ""))
            print(f"    계통 편차        {det_eval['err_bias_px']:+.3f} px  "
                  f"→ 깊이 {det_eval['depth_bias_mm']:.1f} mm")
            print(f"    무작위 오차 σ_u   {det_eval['err_noise_px']:.3f} px  "
                  f"→ 깊이 {det_eval['depth_noise_mm']:.1f} mm  "
                  f"(설계 가정 {det_eval['sigma_u_design_px']} px)")
            for pre, d in (det_eval.get("families") or {}).items():
                sw, gm = d["정답선_변동폭_px"], d["깊이이득_중앙"]
                print(f"    {d['이름']:<12}검출 {d['검출']}/{d['선 수']}"
                      f"   선내잡음 {d['선내잡음_px']} px"
                      f" / 선간편차 {d['선간편차_px']} px"
                      f" → 통합 {d['통합오차_px']} px")
                print(f"    {'':<12}깊이 신호(정답선 {d['측정축']} 변동폭) "
                      f"{sw if sw is not None else '-'} px"
                      f"   깊이가능 {d['깊이가능']}/{d['선 수']}"
                      + (f" (g≈{gm})" if gm is not None else " (g=∞ — 원리적 불가)"))

    # 불확실도에 **측정한** 선검출 오차를 넣는다.
    # 설계 가정 0.2px 를 그대로 쓰면, 실제 검출이 그보다 나쁠 때 평활도를
    # "합격" 으로 내주게 된다. 판정의 근거가 되는 σ 는 가정이 아니라
    # 이 촬영에서 실제로 잰 값이어야 한다.
    su = CALIB.SIGMA_U_PX if sigma_u_px is None else float(sigma_u_px)
    su_src = "지정값" if sigma_u_px is not None else "설계 가정"
    if sigma_u_px is None and det_eval and not det_eval.get("error"):
        m = det_eval.get("err_noise_px")
        if m:
            su = float(m) * det_eval["scale_to_sensor"]
            su_src = "이 촬영에서 실측"
            print(f"    → 불확실도 σ_u 에 실측 {su:.3f}px 사용 "
                  f"(설계 가정 {CALIB.SIGMA_U_PX}px)")

    # ── [2단계] 검출 화소 → 3D 좌표(깊이) ──
    # source="detected" 가 기본이다. 실장비가 실제로 하는 일이 이것이고,
    # 조서에 실리는 값도 그래야 현장에서 나올 값이 된다. 정답 화소로
    # 돌린 결과는 "카메라가 완벽했다면" 의 상한이라, 비교용으로만 쓴다.
    src = (source or "detected").lower()
    lines_in = None
    if src == "detected":
        lines_in = detected_lines_sensor(cap, det_eval)
        if not lines_in:
            print("  [경고] 검출 화소를 얻지 못해 정답 화소로 되돌린다.")
            src = "gt"
    if src != "detected":
        lines_in = cap["lines_pixels"]

    lines_xyz, lines_uv, tri_info = PIPE.triangulate_lines(
        lines_in, cap["line_angles"], cp)
    n3d = sum(len(v) for v in lines_xyz.values())
    nf = tri_info["n_by_family"]
    src_ko = "검출 화소" if src == "detected" else "정답 화소"
    print(f"\n  [2단계] 3D 복원 — 입력: {src_ko}")
    print(f"    {'삼각측량 성공 점':<22}{n3d}"
          f"   (V {nf.get('V', 0)} + H {nf.get('H', 0)})")
    for fam, d in tri_info["family"].items():
        gm = d["이득_중앙"]
        print(f"    {'  ' + d['이름']:<22}"
              f"{d['선 수']}개 중 깊이가능 {d['깊이가능']}개"
              + (f"   이득 g 중앙 {gm}" if gm is not None else "")
              + (f"   g=∞ {d['무한대']}개" if d["무한대"] else ""))
    if n3d == 0:
        print("  [중단] 삼각측량된 점이 없다.")
        return None

    # 이 3D 가 맞는지 raycast 참값과 **직접** 맞댄다. 화소 오차나 각도
    # 오차가 아니라, 점의 거리가 몇 mm 틀렸는지가 여기서 나온다.
    depth = None
    try:
        depth = verify_depth(lines_uv, lines_xyz, cap)
    except Exception as e:
        print(f"    [경고] 깊이 검증 실패: {e}")
    if depth:
        print(f"    {'깊이 오차 (vs raycast)':<22}"
              f"치우침 {depth['z_bias_mm']:+.2f} mm / "
              f"산포 {depth['z_noise_mm']:.2f} mm / "
              f"RMS {depth['z_rms_mm']:.2f} mm  ({depth['n_points']:,}점)")
        print(f"    {'3D 위치 오차':<22}"
              f"중앙 {depth['dist_med_mm']:.2f} mm / "
              f"95% {depth['dist_p95_mm']:.2f} mm")

    # ── [3단계] 영역분할 + 부재별 검측 ──
    print(f"\n  [3단계] 영역분할 → 수직도·수평도·평활도")
    res = PIPE.inspect_image(lines_uv, lines_xyz, cp, cap["g_hat"],
                             seg_backend=backend, sigma_u_px=su,
                             line_gain=tri_info["line_gain"],
                             aux_lines_uv=tri_info.get("skipped_uv"))
    res["triangulation"] = tri_info
    res["pixel_source"] = src
    res["depth_check"] = depth

    # 격자선 한 줄만 걸린 부재를 가림 그림자로 되살린다 (eq8).
    # 검측 좌표는 센서 기준·표준 규약, 이미지는 화면 기준·원본 규약이다.
    if det_eval and not det_eval.get("error"):
        try:
            from PIL import Image as _PIL
            _im = np.asarray(_PIL.open(os.path.join(path, det_eval["image"]))
                             .convert("RGB"))
            _su = det_eval["scale_to_sensor"]
            _sv = det_eval["scale_to_sensor_v"]
            _fl = bool(cap["diag"]["uv 180° 뒤집힘"])
            _cx, _cy = cp["cx_px"], cp["cy_px"]

            def _to_img(a, _cx=_cx, _cy=_cy, _fl=_fl, _su=_su, _sv=_sv):
                a = np.asarray(a, float)
                if _fl:
                    a = np.stack([2 * _cx - a[..., 0], 2 * _cy - a[..., 1]],
                                 axis=-1)
                return a / np.array([_su, _sv])

            _cpi = {"f_px": cp["f_px"] / _su,
                    "b_m": cap.get("b_raw", cp["b_m"]),
                    "cx_px": _cx / _su, "cy_px": _cy / _sv}
            _n = PIPE.resolve_single_plane_members(res, _im, _cpi,
                                                   cap["g_hat"],
                                                   to_image=_to_img)
            if _n:
                print(f"    가림 그림자로 {_n}개 부재의 옆 기울기 복원")
        except Exception as e:
            print(f"    [경고] 실루엣 복원 실패: {e}")
    print()
    print(PIPE.format_report(res))

    base = _base_image(path, cp["resolution"])
    # 배경은 내보내기 원본 규약, 검측 좌표는 표준 규약이다. 배경을 돌리는
    # 대신 그릴 좌표를 되돌린다 (리샘플링 없음).
    flip = bool(cap["diag"]["uv 180° 뒤집힘"])
    cxp, cyp = cp["cx_px"], cp["cy_px"]
    uv_tf = ((lambda a: np.stack([2 * cxp - np.asarray(a, float)[..., 0],
                                  2 * cyp - np.asarray(a, float)[..., 1]],
                                 axis=-1)) if flip else None)
    seg = REPORT.save_segmentation(os.path.join(out_dir, f"{name}_세그멘테이션.png"),
                                   res, base_image=base,
                                   shape=(cp["resolution"][1], cp["resolution"][0]),
                                   uv_transform=uv_tf)
    # ── 3D 점군 — 부재별로 어디에 점이 찍혔는가 ──
    # 세그멘테이션 이미지는 화소 위 그림이라 깊이가 안 보인다. 벽과 바닥이
    # 사진에서는 붙어 있어도 3D 에서는 직각으로 갈라지고, 그게 맞게 갈렸는지가
    # 곧 검측이 맞았는지다.
    pc3d = pc_csv = None
    try:
        pc3d = REPORT.save_pointcloud_3d(
            os.path.join(out_dir, f"{name}_3D점군.png"), res, cap["g_hat"],
            title=f"3D 점군 — {name}")
        pc_csv = REPORT.save_pointcloud_csv(
            os.path.join(out_dir, f"{name}_3D좌표.csv"), res,
            g_hat=cap["g_hat"], stride=max(1, int(pc_stride)))
    except Exception as e:
        print(f"  [경고] 3D 점군 산출 실패: {e}")
    meta = {"현장": site or "-", "입력 폴더": name,
            "케이스": cap["meta"].get("case"),
            "촬영 시각": cap["meta"].get("captured_at")}
    meta.update({k: str(v) for k, v in cap["diag"].items()})
    meta["불확실도 σ_u (px)"] = f"{su:.3f}  ({su_src})"
    caveats = [
        ("실촬영이 아니라 Isaac raycast 내보내기다. 다만 검측에 넣은 화소는 "
         "정답이 아니라 **렌더 이미지에서 선검출로 찾은 것**이므로, 화소→3D→"
         "검측의 전 구간이 실장비와 같은 경로를 지난다."
         if src == "detected" else
         "검측에 정답 화소(raycast)를 그대로 넣었다. 선검출 단계를 건너뛴 "
         "값이라 '카메라가 완벽했다면' 의 상한이며, 현장 실측값이 아니다."),
        f"불확실도(σ_n)는 선검출 오차 σ_u={su:.3f}px({su_src})로 계산했다.",
        "카메라 자세는 헤더가 아니라 (xyz_world, uv) 대응에서 Kabsch 로 "
        "복원했다. 헤더에는 롤이 없어 아래를 보는 촬영에서 자세를 세울 수 "
        "없고, 캡처마다 uv 축 방향도 다르다.",
    ]
    if backend == "geom":
        caveats.append("기하 전용 백엔드는 동바리/기둥/철근과 벽/거푸집/조적을 "
                       "구분하지 못한다. 부재 종류는 사람이 확인해야 한다.")
    if det_eval and not det_eval.get("error"):
        if det_eval["missed_lines"]:
            caveats.append(
                f"선검출이 {len(det_eval['missed_lines'])}개 선을 놓쳤다"
                f"({', '.join(det_eval['missed_lines'])}). 이 표본에서는 앞에 선"
                f" 부재 위에 떨어진 선들이며, 예측 밴드를 장면 대표거리 하나로"
                f" 잡는 한 다른 깊이의 면에 걸린 선은 놓치기 쉽다.")
        if abs(det_eval["err_bias_px"]) > 0.3:
            caveats.append(
                f"선검출에 {det_eval['err_bias_px']:+.2f}px 의 계통 편차가 있다"
                f"(깊이 {det_eval['depth_bias_mm']:.1f}mm). 무작위 오차가 아니라"
                f" 좌표 규약이 어긋난 것이므로 원인을 찾아 제거해야 한다.")
    ov = None
    if det_eval and not det_eval.get("error"):
        try:
            ov = save_detection_overlay(
                path, cap, det_eval,
                os.path.join(out_dir, f"{name}_선검출대조.png"))
        except Exception:
            ov = None
    e2e = None
    if det_eval and not det_eval.get("error"):
        try:
            e2e = evaluate_end_to_end(path, cap, det_eval, backend=backend,
                                      sigma_u_px=su)
        except Exception:
            e2e = None
        if e2e and e2e.get("max_dgap_mm") is not None:
            caveats.append(
                f"검출 화소로 검측까지 돌리면 각도는 최대 "
                f"{e2e['max_dtheta_deg']}° 차이(목표 ±0.5°의 "
                f"{0.5/max(e2e['max_dtheta_deg'],1e-9):.0f}분의 1)로 거의 "
                f"영향이 없으나, 평활도 자처짐은 최대 {e2e['max_dgap_mm']}mm "
                f"차이가 난다. 점별 화소 오차가 그대로 표면 요철로 보이기 "
                f"때문이다.")
    if depth:
        caveats.append(
            f"검출 화소로 푼 3D 를 raycast 참값과 직접 맞대면 깊이 치우침 "
            f"{depth['z_bias_mm']:+.2f}mm, 산포 {depth['z_noise_mm']:.2f}mm "
            f"({depth['n_points']:,}점)이다. 평활도는 점별 깊이 오차가 그대로 "
            f"결과이므로, 이 산포가 목표 ±2mm 를 넘으면 평활도는 측정불가다.")
    xl = XLS.save_excel(os.path.join(out_dir, f"{name}_품질검측조서.xlsx"), res,
                        meta=meta, seg_image_path=seg, extra_caveats=caveats,
                        detection=det_eval, end_to_end=e2e,
                        triangulation=tri, g_hat=cap["g_hat"],
                        pointcloud_image=pc3d, detection_image=ov,
                        depth_check=depth, pixel_source=src,
                        pc_stride=max(1, int(pc_stride)))
    print()
    if ov:
        print(f"  선검출 대조 이미지: {ov}")
    print(f"  세그멘테이션 이미지: {seg}")
    if pc3d:
        print(f"  3D 점군 이미지:     {pc3d}")
    if pc_csv:
        print(f"  3D 좌표 CSV:       {pc_csv}")
    print(f"  엑셀 조서:          {xl}")
    return {"result": res, "capture": cap, "seg": seg, "xlsx": xl,
            "pointcloud": pc3d, "pointcloud_csv": pc_csv,
            "name": name, "out_dir": out_dir}


def _truth_depth(cap, lines_xyz):
    """raycast xyz_world 를 조사기 좌표 Z 로 바꿔 복원 깊이와 비교."""
    rt = cap["raw"].get("rig_transform") or {}
    if "laser_pos_world" not in rt:
        return None
    return None      # 자세 부호 규약이 캡처마다 달라 여기서는 생략


def main():
    ap = argparse.ArgumentParser(description="Isaac raycast 내보내기 폴더 검측")
    ap.add_argument("path", help="내보내기 폴더 (또는 --all 과 함께 상위 폴더)")
    ap.add_argument("--all", action="store_true",
                    help="하위 폴더를 모두 처리한다")
    ap.add_argument("--out", default=None, help="산출물 폴더")
    ap.add_argument("--backend", default="geom", choices=["geom", "sam", "vlm"])
    ap.add_argument("--stride", type=int, default=1,
                    help="선 위 점을 N개마다 하나씩 (기본 1 = 전부)")
    ap.add_argument("--site", default=None)
    ap.add_argument("--sigma-u", type=float, default=None,
                    help="선검출 픽셀오차 가정 [px]. 기본은 프로파일 값")
    ap.add_argument("--no-detect-eval", action="store_true",
                    help="선검출 정확도 대조를 건너뛴다 (느릴 때)")
    ap.add_argument("--source", default="detected",
                    choices=["detected", "gt"],
                    help="검측에 넣을 화소. detected=이미지에서 검출한 것"
                         "(기본, 실장비와 같음), gt=raycast 정답 화소(상한)")
    ap.add_argument("--pc-stride", type=int, default=20,
                    help="3D 좌표 CSV·엑셀에 N개마다 한 점 (기본 20). "
                         "그림과 요약 통계는 항상 전체 점을 쓴다")
    ap.add_argument("--profile", default=None)
    a = ap.parse_args()
    if a.profile:
        CALIB.use_profile(a.profile)

    targets = ([os.path.join(a.path, d) for d in sorted(os.listdir(a.path))
                if os.path.isdir(os.path.join(a.path, d))
                and os.path.exists(os.path.join(a.path, d,
                                                "camera_params.json"))]
               if a.all else [a.path])
    if not targets:
        print("처리할 폴더가 없다."); return 2
    for t in targets:
        out = (os.path.join(a.out, os.path.basename(os.path.normpath(t)))
               if a.out else None)
        inspect_folder(t, out_dir=out, backend=a.backend, pc_stride=a.pc_stride,
                       source=a.source,
                       stride=a.stride, site=a.site, sigma_u_px=a.sigma_u,
                       eval_detection=not a.no_detect_eval)
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
