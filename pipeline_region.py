#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pipeline_region.py — 이미지 1장 → 영역별 품질검측
========================================================================
현장 사진 한 장에 벽·바닥·동바리·철근이 함께 들어와도, 영역을 나눠
각 부재에 맞는 검측식을 적용한다.

  1 shot = (rgb_off, rgb_on, IMU ĝ, calib)
   ├─[A ] 선검출        rgb_on  → {lid: [(u,v)…]}
   ├─[C ] 영역분할      rgb_off → label_map / point_labels
   ├─[eq1] 삼각측량             → (X,Y,Z) 조사기 좌표계
   ├─[eq5] 영역 할당            → 침식·깊이불연속 제거 + 의미x기하 융합
   └─ 영역별 검측
       벽·거푸집·조적  → eq2 평면(TLS) → eq3 수직도(중력) + eq4 평활도(면내)
       바닥·슬래브     → eq2 평면(TLS) → eq3 수평도(중력) + eq4 평활도(면내)
       동바리·기둥·철근 → eq2 축(PCA)  → eq3 축 수직도(중력)  [평활도 N/A]
       전 영역        → eq5 불확실도 → KCS 합격/기준초과/측정불가

기존 inspection.py 는 STATIONS 딕셔너리에 "이 스테이션은 수직도" 라고
적어두고 화면 전체를 단일 평면으로 적합했다. 이 모듈은 그 결정을 설정이
아니라 인지 결과에서 가져온다.

실행 (합성 씬 자체검증):
  python3 pipeline_region.py
========================================================================
"""
import numpy as np
import importlib.util as _ilu, os as _os


def _load(name):
    spec = _ilu.spec_from_file_location(
        name, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), f"{name}.py"))
    m = _ilu.module_from_spec(spec); spec.loader.exec_module(m)
    return m


_EQ1 = _load("eq1_triangulation")
_EQ2 = _load("eq2_plane_fit")
_EQ3 = _load("eq3_orientation")
_EQ4 = _load("eq4_flatness_line")
_EQ5 = _load("eq5_region_assign")
_EQ6 = _load("eq6_straightedge")
_SEG = _load("C_영역분할")

# 평활도 허용 기준 (PDF 1.2 표: 노출 콘크리트 3m당 7mm, 미장 1m당 10mm 등).
# 면적 기준이 아니라 구간 기준이므로 영역 크기와 함께 판정한다.
FLATNESS_TOL_MM = {"wall": 7.0, "plaster_wall": 10.0, "masonry": 10.0,
                   "formwork_wall": 7.0, "formwork_column": 7.0,
                   "floor": 10.0, "slab": 10.0, "ceiling": 3.0}


# =====================================================================
# 영역 1개 검측
# =====================================================================
def _defects_in_image(fd, camera_params):
    """
    검출된 요철 덩어리를 화면 좌표로 옮긴다.

    eq4 는 요철을 **면내 좌표(u,v)** 로 돌려준다. 평활도 계산에는 그것이
    맞지만, 조서를 읽는 사람은 "벽 어디가 튀어나왔나" 를 그림에서 보고
    싶어 한다. 면내 좌표를 3D 로 되돌린 뒤 카메라로 투영한다.

        P = origin + u·e1 + v·e2        (w 는 mm 단위라 투영에 영향이 없다)
        u_px = f·(X − b)/Z + c_x,  v_px = f·Y/Z + c_y

    투영식에 −b 가 들어가는 것은 카메라가 조사기에서 X 로 b 만큼 떨어져
    있기 때문이다(eq1 과 같은 규약).
    """
    clusters = fd.get("verified_clusters") or []
    if not clusters or fd.get("basis") is None:
        return []
    e1, e2, _ = fd["basis"]
    origin = np.asarray(fd["origin"], float)
    # cand_points_uv 가 클러스터 point_idx 의 기준 배열이다.
    cand_uv = np.asarray(fd.get("cand_points_uv")
                         if fd.get("cand_points_uv") is not None
                         else fd.get("defect_points_uv"))
    f = float(camera_params["f_px"]); b = float(camera_params["b_m"])
    cx = float(camera_params["cx_px"]); cy = float(camera_params["cy_px"])

    out = []
    for c in clusters:
        idx = np.asarray(c["point_idx"], dtype=int)
        if len(cand_uv) and idx.max() < len(cand_uv):
            uv = cand_uv[idx]
        else:
            # 기준 배열을 못 얻으면 덩어리 중심 하나만 표시한다.
            # 잘못된 배열을 인덱싱해 엉뚱한 위치를 그리는 것보다 낫다.
            uv = np.array([c["center_xy"]], dtype=float)
        P = origin + np.outer(uv[:, 0], e1) + np.outer(uv[:, 1], e2)
        Z = P[:, 2]
        ok = Z > 1e-6
        if not ok.any():
            continue
        P = P[ok]; Z = Z[ok]
        up = f * (P[:, 0] - b) / Z + cx
        vp = f * P[:, 1] / Z + cy
        out.append({
            "center_px": [round(float(up.mean()), 1), round(float(vp.mean()), 1)],
            "bbox_px": [round(float(up.min()), 1), round(float(vp.min()), 1),
                        round(float(up.max()), 1), round(float(vp.max()), 1)],
            "depth_mm": round(float(c["depth_mm"]), 2),
            "extent_mm": round(float(c["extent_mm"]), 1),
            "n_points": int(c["n_points"]),
            "z_m": round(float(np.median(Z)), 3),
        })
    out.sort(key=lambda d: -abs(d["depth_mm"]))
    return out


def measure_region(points_3d, cls, g_hat, camera_params,
                   flatness_threshold_mm=1.5, sigma_u_px=0.2,
                   target_sigma_mm=2.0, member_length_m=None):
    """
    한 영역의 3D 점군에 클래스에 맞는 검측식을 적용한다.

    Returns
    -------
    dict — kind, theta_deg, judge, flatness, uncertainty, n_points, status
        status : "measured" | "rejected"
    """
    pts = np.asarray(points_3d, dtype=float)
    kind = _EQ5.MEASURE_KIND.get(cls)
    kcs_cls = _EQ5.KCS_CLASS.get(cls, cls)
    out = {"class": cls, "kind": kind, "n_points": int(len(pts)),
           "status": "rejected", "reject_reason": None,
           "theta_deg": None, "judge": None,
           "flatness": None, "uncertainty": None}

    if kind is None:
        out["reject_reason"] = f"검측 대상 클래스 아님 ({cls})"
        return out
    if len(pts) < 12:
        out["reject_reason"] = f"점 부족 ({len(pts)} < 12)"
        return out

    # ── 선형 부재: 축 적합 ──
    if kind == "axis_vertical":
        # 얇은 부재는 마스크 실루엣에서 반드시 오염되므로 robust 적합을 쓴다
        ax = _EQ2.fit_axis_ransac(pts)
        out["axis"] = {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                       for k, v in ax.items() if k != "reject_reason"}
        if not ax["is_valid"]:
            out["reject_reason"] = ax["reject_reason"]
            return out
        theta = _EQ3.measure_axis_verticality(ax["direction"], g_hat)
        out["theta_deg"] = round(theta, 4)
        # 격자가 닿은 구간만 보이므로 부재 전체 길이는 알 수 없다.
        # 측정 구간을 부재 길이로 대신 쓰면 mm 판정의 근거가 틀어지므로,
        # 전체 길이는 None 으로 두고 각도로 판정한다
        # (member_length_m 은 도면에서 알 때 호출부가 넘긴다).
        out["judge"] = _EQ3.judge_kcs(theta, kcs_cls,
                                      member_length_m=member_length_m,
                                      measured_span_m=ax["length_m"])

        # 세장비에서 오는 각도 불확실도를 판정에 반영한다.
        # 부재가 짧게만 보이면 원통 단면이 주축을 끌어당겨 각도가 흔들린다.
        # 측정값만 보고 합격을 내주면 그 흔들림이 판정에 그대로 숨는다.
        unc = ax.get("angle_uncertainty_deg")
        j = out["judge"]
        j["cross_section_resolved"] = ax.get("cross_section_resolved", True)
        if ax.get("note"):
            # 단면이 잡히지 않았다는 사실은 판정만큼 중요하다. 지름을
            # 모르면 부재 종류(동바리/기둥/철근)를 고를 수 없고, 그에 따라
            # 적용할 KCS 허용치도 고를 수 없다. 조서 비고에 남긴다.
            j["note"] = ((j.get("note") + " / ") if j.get("note") else "") \
                + ax["note"]
        if unc is not None:
            j["angle_uncertainty_deg"] = unc
            j["slenderness"] = ax.get("slenderness")
            tol = j.get("allow_deg", 0.5)
            if j.get("is_pass") and abs(theta) + unc > tol:
                j["is_pass"] = None
                j["judgement"] = "판정보류(노출길이)"
                j["note"] = (
                    f"측정 {theta:.3f}° 는 허용 {tol}° 이내이나, 부재가 "
                    f"{ax['length_m']*1000:.0f}mm 만 보여(세장비 "
                    f"{ax['slenderness'] if ax.get('slenderness') else '미확인'}) "
                    f"각도 불확실도가 ±{unc:.2f}° 다. "
                    f"합쳐서 {abs(theta)+unc:.2f}° 로 허용치를 넘을 수 있어 "
                    f"합격 판정을 내리지 않는다. 부재를 세로로 더 길게 담아 "
                    f"재촬영할 것")

        out["uncertainty"] = _EQ5.region_uncertainty(
            pts, camera_params, normal=None, sigma_u_px=sigma_u_px,
            target_sigma_mm=target_sigma_mm)
        # 평활도는 원통면이라 성립하지 않음
        out["flatness"] = {"applicable": False,
                           "reason": "선형 부재(원통면) — 평활도 정의 없음"}
        out["status"] = "measured"
        return out

    # ── 면 부재: 방향 무관 TLS 평면 적합 ──
    plane, inliers = _EQ2.fit_plane_tls_ransac(pts, threshold=0.01)
    normal = np.array(plane[:3], dtype=float)
    out["plane"] = [round(float(x), 6) for x in plane]
    out["plane_inliers"] = int(inliers.sum())

    theta = _EQ3.measure_from_gravity(normal, g_hat, kind)
    out["theta_deg"] = round(theta, 4)
    out["judge"] = _EQ3.judge_kcs(theta, kcs_cls, member_length_m=None)

    unc = _EQ5.region_uncertainty(pts, camera_params, normal=normal,
                                  sigma_u_px=sigma_u_px,
                                  target_sigma_mm=target_sigma_mm)
    out["uncertainty"] = unc

    # ── 평활도 ──
    # 두 가지를 함께 낸다. 쓰임이 다르다.
    #   eq4 : 요철의 위치·개수·깊이를 찾는다 (어디가 문제인지)
    #   eq6 : KCS 가 규정한 직선자 처짐량으로 판정한다 (합격인지)
    # 전역 평면 잔차(eq4)로는 시방 판정을 할 수 없다. 넓은 면이 완만히
    # 휘면 잔차는 크지만 3m 자에는 안 걸리고, 좁고 급한 굴곡은 그 반대다.
    fd = _EQ4.detect_defects_region(pts, plane=plane,
                                    threshold_mm=flatness_threshold_mm)
    flat = {"applicable": True,
            "defect_max_dev_mm": round(fd["overall_max_dev_mm"], 3),
            "rms_dev_mm": round(fd["rms_dev_mm"], 3),
            "raw_max_dev_mm": round(fd["raw_max_dev_mm"], 3),
            "defect_clusters": len(fd["verified_clusters"]),
            "defect_count": fd["defect_count"],
            "reject_reason": fd.get("reject_reason")}
    flat["defects"] = _defects_in_image(fd, camera_params)

    kcs = _EQ6.judge_kcs_flatness(
        pts, cls, plane=plane,
        sigma_normal_mm=(None if unc["flatness_measurable"]
                         else unc["sigma_normal_mm"]),
        target_sigma_mm=target_sigma_mm)
    flat["kcs"] = kcs
    flat["judgement"] = kcs["judgement"]
    flat["is_pass"] = kcs["is_pass"]
    if kcs["checks"]:
        c0 = max(kcs["checks"], key=lambda x: x.get("ratio") or 0.0)
        flat["straightedge_length_m"] = c0["length_m"]
        flat["max_gap_mm"] = c0["max_gap_mm"]
        flat["upper_estimate_mm"] = c0["upper_estimate_mm"]
        flat["tolerance_mm"] = c0["tolerance_mm"]
    if not unc["flatness_measurable"]:
        flat["note"] = (f"법선방향 불확실도 σ_n={unc['sigma_normal_mm']}mm > "
                        f"목표 {target_sigma_mm}mm "
                        f"(Z={unc['z_mean_m']}m, 입사각 {unc['incidence_deg']}°)")
    elif kcs.get("note"):
        flat["note"] = kcs["note"]
    out["flatness"] = flat
    out["status"] = "measured"
    return out


def split_incoherent_region(xyz, cls, g_hat, min_points=12,
                            outlier_frac=0.08, plane_threshold_m=0.015):
    """
    한 영역 안에 성질이 다른 면이 섞여 있으면 기하로 되쪼갠다.

    【왜 필요한가】
      세그멘테이션이 두 부재를 **같은 라벨로 병합**하면(예: 바닥을 벽으로
      오분류) 두 면의 점이 한 영역에 들어온다. 그러면 영역 평면적합이 두
      면에 걸쳐지고, 점이 적은 쪽 부재는 outlier 로 밀려 통째로 사라진다.
      실측에서 바닥 480점이 벽 4121점에 병합되자 바닥 검측 결과가 아예
      나오지 않았다. 경계가 몇 px 어긋나는 것과는 차원이 다른 실패다.

      eq5 의 라벨 융합은 이 경우를 못 고친다. 융합은 "이 영역이 무엇인가"를
      바로잡을 뿐, "이 영역이 사실 둘"이라는 것은 판단하지 않기 때문이다.

    【판정 기준】
      대표 모델(면이면 평면, 선형이면 축)에 대한 outlier 비율이
      outlier_frac 을 넘고 그 수가 min_points 이상이면 섞였다고 본다.
      깨끗한 단일 면은 outlier 가 노이즈뿐이라 이 문턱에 닿지 않는다.

    Returns
    -------
    list of (class_name, index_array) — 쪼갤 필요가 없으면 원본 하나만
    """
    pts = np.asarray(xyz, dtype=float)
    n = len(pts)
    keep_all = [(cls, np.arange(n))]
    if n < 2 * min_points:
        return keep_all

    # 형상 자체가 불분명하면 그것이 곧 "섞였다"는 증거다.
    # 대표 모델 적합이 실패했는데 그대로 넘어가면, 병합된 영역이 통째로
    # 기각되어 두 부재를 다 잃는다(실측: 벽 4134점 + 동바리 288점이 한
    # 라벨로 묶이자 축 적합이 무효가 되고 영역 전체가 기각됨).
    ev = _EQ5.geometric_evidence(pts, g_hat)
    need_split = (ev["shape"] not in ("plane_vertical", "plane_horizontal",
                                      "linear_vertical")
                  or float(ev.get("confidence", 0.0)) < 0.5)

    if not need_split:
        kind = _EQ5.MEASURE_KIND.get(cls)
        if kind in ("plane_vertical", "plane_horizontal"):
            try:
                _, inl = _EQ2.fit_plane_tls_ransac(pts,
                                                   threshold=plane_threshold_m)
                n_out = int((~inl).sum())
            except Exception:
                return keep_all
        elif kind == "axis_vertical":
            ax = _EQ2.fit_axis_pca(pts)
            if not ax["is_valid"]:
                n_out = n                      # 적합 실패 → 섞인 것으로 본다
            else:
                c, d = ax["centroid"], ax["direction"]
                rel = pts - c
                radial = np.linalg.norm(rel - np.outer(rel @ d, d), axis=1)
                n_out = int((radial > 3.0 * max(ax["radial_rms_mm"], 1.0)
                             / 1000.0).sum())
        else:
            return keep_all

        if n_out < min_points or n_out / n < outlier_frac:
            return keep_all

    # 섞였다 → 기하 전용 백엔드로 이 영역만 다시 나눈다
    try:
        seg = _SEG.segment(None, backend="geom",
                           table={"xyz": pts,
                                  "uv": np.zeros((n, 2)),
                                  "lid": np.array(["R"] * n, dtype=object),
                                  "seq": np.arange(n)},
                           g_hat=g_hat, min_plane_points=min_points,
                           min_linear_points=min_points)
    except Exception:
        return keep_all

    labels, names = seg["point_labels"], seg["class_names"]
    subs = []
    for cid in np.unique(labels):
        sub_cls = names.get(int(cid), "background")
        if sub_cls in _EQ5.IGNORE_CLASSES:
            continue
        idx = np.where(labels == cid)[0]
        if len(idx) >= min_points:
            subs.append((sub_cls, idx))
    if len(subs) < 2:
        return keep_all
    return subs


# =====================================================================
# 한 장 전체 검측
# =====================================================================
def inspect_image(lines_pixels, lines_xyz, camera_params, g_hat,
                  rgb_off=None, seg_backend="gt", seg_kwargs=None,
                  erode_default_px=3, erode_thin_px=1,
                  min_region_points=12, sigma_u_px=0.2,
                  target_sigma_mm=2.0, flatness_threshold_mm=1.5,
                  split_incoherent=True):
    """
    선검출 결과 + 삼각측량 결과 + 세그멘테이션으로 영역별 검측을 수행한다.

    Parameters
    ----------
    lines_pixels : {lid: [(u,v), ...]}   A_선검출 출력 (ON 프레임)
    lines_xyz    : {lid: [(X,Y,Z), ...]} eq1 삼각측량 결과 (같은 순서)
    camera_params: dict — f_px, b_m, cx_px, cy_px
    g_hat        : (3,) 조사기 좌표계 중력 단위벡터 (eq3.gravity_in_laser_frame)
    rgb_off      : (H,W,3) 레이저 OFF 프레임 — 세그멘테이션 입력
    seg_backend  : "gt" | "geom" | "sam" | "vlm"
    seg_kwargs   : 백엔드 인자 (gt → label_map, id_to_semantic)

    Returns
    -------
    dict — regions, summary, segmentation, assign_stats
    """
    seg_kwargs = dict(seg_kwargs or {})
    table = _EQ5.build_point_table(lines_pixels, lines_xyz)
    if len(table["xyz"]) == 0:
        return {"regions": [], "summary": {"n_regions": 0},
                "error": "삼각측량된 격자점이 없습니다."}

    # ── [C] 영역 분할 ──
    if seg_backend == "geom":
        seg_kwargs.setdefault("table", table)
        seg_kwargs.setdefault("g_hat", g_hat)
        seg_kwargs.setdefault("camera_params", camera_params)
    seg = _SEG.segment(rgb_off, backend=seg_backend, **seg_kwargs)

    # ── [eq5] 점 → 영역 ──
    if seg["label_map"] is not None:
        regions, stats = _EQ5.assign_points_to_regions(
            table, seg["label_map"], seg["class_names"],
            erode_default_px=erode_default_px, erode_thin_px=erode_thin_px,
            min_points=min_region_points)
    else:
        regions, stats = _regions_from_point_labels(
            table, seg["point_labels"], seg["class_names"],
            min_points=min_region_points)

    # ── 영역별 검측 ──
    results = []
    n_split = 0
    n_linear_rescued = 0
    for reg in regions:
        pts_all = table["xyz"][reg["idx"]]
        ev0 = _EQ5.geometric_evidence(pts_all, g_hat)
        fu0 = _EQ5.fuse_label(reg["class"], ev0)

        # ── 1) 병합된 영역이면 먼저 되쪼갠다 ──
        # 세그멘테이션이 두 부재를 한 라벨로 묶었을 수 있다. 이 단계를
        # 선형 정제보다 **먼저** 해야 한다. 순서를 바꾸면, 라벨이 선형인
        # 영역에서 축 바깥 점이 통째로 버려져 같이 묶여 있던 다른 부재가
        # 재분할에 도달하지 못한다(실측: 바닥 471점이 동바리 라벨에 묶이자
        # 축 정제가 이를 버려 바닥 검측이 통째로 사라짐).
        parts = ([(fu0["final_class"], np.arange(len(pts_all)))]
                 if not split_incoherent
                 else split_incoherent_region(pts_all, fu0["final_class"],
                                              g_hat, min_points=min_region_points))
        if len(parts) > 1:
            n_split += 1

        for part_cls, sub_idx in parts:
            pts = pts_all[sub_idx]
            # 이 조각이 원래 table 의 몇 번 점인지 끝까지 들고 간다.
            # 세그멘테이션 결과 이미지를 그리려면 각 점의 화소 좌표가
            # 필요한데, 아래 선형 정제가 점을 걸러내므로 인덱스를 같이
            # 걸러야 짝이 맞는다.
            keep = np.asarray(reg["idx"])[np.asarray(sub_idx)]

            # ── 2) 조각별 선형 부재 정제 ──
            # 동바리·철근처럼 가는 부재는 마스크가 몇 px 만 밖으로 밀려도
            # 뒤쪽 면의 점이 딸려 들어오고, 부재가 가늘어 그 비중이 크다.
            # 그러면 형상 판별이 선형에서 평면으로 뒤집히고, "면↔선형은
            # 기하 우선" 규칙이 올바른 의미 라벨을 버려 부재가 사라진다
            # (실측: 마스크 +8px 팽창에서 동바리 288점에 벽 33점이 섞여 소실).
            rescued = False
            if part_cls in _EQ5.LINEAR_VERTICAL_CLASSES:
                ax0 = _EQ2.fit_axis_ransac(pts)
                cand = pts[ax0["inlier_mask"]] if ax0["is_valid"] else None
                if cand is not None and len(cand) >= min_region_points:
                    # 정제 결과가 **실제로 1D 부재인지** 확인해야 한다.
                    # 축 적합은 평면을 얇게 저민 조각에도 성공한다. 바닥
                    # 조각에 축을 맞추면 축이 면 안에 누워 수직도가 89° 로
                    # 나온다(실측: 라벨 오분류 시 동바리 오차 88.79°).
                    # 원통은 3번째 주축에도 두께가 남지만 판 조각은 납작해
                    # geometric_evidence 의 선형 판정으로 가려낼 수 있다.
                    if _EQ5.geometric_evidence(cand, g_hat)["shape"] == "linear_vertical":
                        if ax0.get("inlier_frac", 1.0) < 0.999:
                            n_linear_rescued += 1
                        pts = cand
                        keep = keep[ax0["inlier_mask"]]
                        rescued = True

            ev = _EQ5.geometric_evidence(pts, g_hat)
            if rescued:
                fu = {"semantic_class": part_cls, "geom_shape": ev["shape"],
                      "geom_confidence": round(float(ev.get("confidence", 0.0)), 3),
                      "final_class": part_cls, "source": "semantic",
                      "agreed": ev["shape"] == "linear_vertical",
                      "note": ("선형 부재 — robust 축 적합으로 오염점 제거 후 "
                               "의미 라벨 유지")}
            elif len(parts) == 1:
                fu = fu0
            else:
                fu = _EQ5.fuse_label(part_cls, ev)
            final_cls = fu["final_class"]

            r = measure_region(pts, final_cls, g_hat, camera_params,
                               flatness_threshold_mm=flatness_threshold_mm,
                               sigma_u_px=sigma_u_px,
                               target_sigma_mm=target_sigma_mm)
            r["region_id"] = int(reg["class_id"])
            r["label_fusion"] = {k: v for k, v in fu.items() if k != "note"}
            r["label_fusion_note"] = fu["note"]
            r["geom_shape"] = ev["shape"]
            r["from_split"] = bool(len(parts) > 1)
            # 결과 이미지·엑셀에서 쓰는 화소 좌표 (검측에는 쓰지 않는다)
            r["point_uv"] = table["uv"][keep]
            r["point_idx"] = keep
            results.append(r)

    measured = [r for r in results if r["status"] == "measured"]
    summary = {
        "n_regions": len(results),
        "n_measured": len(measured),
        "n_rejected": len(results) - len(measured),
        "classes": sorted({r["class"] for r in results}),
        "label_corrections": sum(1 for r in results
                                 if r["label_fusion"]["source"] == "geometric"),
        "regions_split": n_split,
        "linear_members_rescued": n_linear_rescued,
        "flatness_unmeasurable": sum(
            1 for r in measured
            if (r["flatness"] or {}).get("judgement") == "측정불가"),
    }
    return {"regions": results, "summary": summary,
            "segmentation": {"backend": seg["backend"], "meta": seg["meta"]},
            "assign_stats": stats}


def _regions_from_point_labels(table, point_labels, class_names, min_points=12):
    """geom 백엔드처럼 점 단위 라벨을 주는 경우의 영역 구성."""
    labels = np.asarray(point_labels)
    disc = _EQ5.mark_depth_discontinuity(table)
    stats = {"total": len(labels), "out_of_image": 0, "ignored_class": 0,
             "eroded_away": 0, "discontinuity": int(disc.sum()), "assigned": 0}
    regions = []
    for cid in np.unique(labels):
        cls = class_names.get(int(cid), "background")
        keep = (labels == cid) & ~disc
        if cls in _EQ5.IGNORE_CLASSES:
            stats["ignored_class"] += int((labels == cid).sum())
            continue
        if keep.sum() < min_points:
            continue
        idx = np.where(keep)[0]
        stats["assigned"] += len(idx)
        regions.append({"class": cls, "class_id": int(cid), "idx": idx,
                        "n_points": int(len(idx))})
    return regions, stats


# =====================================================================
# 촬영 결과 → 검측 (Isaac 비의존)
# =====================================================================
def triangulate_lines(lines_pixels, line_angles, camera_params):
    """
    검출된 선의 픽셀점을 eq1 로 3D 로 되돌린다.

    V선만 처리하는 이유
    ------------------
    기선이 X축 방향이라 시차는 u 좌표에만 나타나고, 깊이를 풀려면 그 점의
    발사각 α 를 알아야 한다.
      · V선 : α 가 선마다 고정 → 선을 따라 조밀 샘플링해도 전부 풀린다
      · H선 : β 만 고정이고 α 는 점마다 다르다. v 좌표는 (v-c_y)/f = tanβ
              라는 이미 아는 사실만 되풀이하므로 깊이 정보가 없다.
    따라서 H선은 V선과의 교점에서만 α 를 회복할 수 있고, 그 교점은 이미
    V선 샘플에 포함된다. V선 조밀 샘플링이 곧 완전한 정보다.

    Returns
    -------
    lines_xyz : {lid: [(X,Y,Z), ...]}
    lines_uv  : {lid: [(u,v), ...]}   삼각측량에 성공한 점만 (순서 일치)
    """
    f = float(camera_params["f_px"]); b = float(camera_params["b_m"])
    cx = float(camera_params["cx_px"]); cy = float(camera_params["cy_px"])
    lines_xyz, lines_uv, skipped = {}, {}, []

    for lid, pts in lines_pixels.items():
        info = line_angles.get(lid)
        if info is None:
            continue
        if info.get("fixed") != "alpha":
            skipped.append(lid)
            continue
        alpha = float(info["angle_rad"])
        xyz, uv = [], []
        for p in pts:
            u, v = float(p[0]), float(p[1])
            try:
                X, Y, Z = _EQ1.triangulate_point(u, v, alpha, 0.0, f, b, cx, cy)
            except ValueError:
                continue                      # 시차 음수 = 캘리브레이션 이상
            if not np.isfinite(Z) or Z <= 0:
                continue
            xyz.append([X, Y, Z]); uv.append([u, v])
        if len(xyz) >= 5:
            lines_xyz[lid] = xyz
            lines_uv[lid] = uv
    return lines_xyz, lines_uv, skipped


def inspect_capture(lines_pixels, line_angles, camera_params, R_world_cam,
                    label_map=None, id_to_semantic=None, rgb_off=None,
                    g_hat=None, backend=None, **kw):
    """
    한 번의 촬영 결과를 받아 영역별 검측까지 수행한다.

    inspection.py(Isaac)와 오프라인 검증 양쪽에서 같은 코드를 쓰기 위해
    Isaac 의존성을 두지 않는다.

    Parameters
    ----------
    R_world_cam : (3,3) or None
        카메라 자세행렬. 주면 여기서 중력을 유도한다(g_hat 미지정 시).
    label_map, id_to_semantic : Isaac 시맨틱 어노테이터 출력
        있으면 backend='gt', 없으면 backend='geom' 으로 자동 폴백.
    """
    if g_hat is None:
        if R_world_cam is None:
            raise ValueError("g_hat 또는 R_world_cam 중 하나는 필요합니다 "
                             "(중력 기준 없이는 수직·수평도를 정의할 수 없음).")
        g_hat = _EQ3.gravity_from_camera_rotation(R_world_cam)

    lines_xyz, lines_uv, skipped = triangulate_lines(
        lines_pixels, line_angles, camera_params)
    if not lines_xyz:
        return {"regions": [], "summary": {"n_regions": 0},
                "error": "삼각측량 가능한 V선이 없습니다.",
                "skipped_lines": skipped}

    if backend is None:
        backend = "gt" if label_map is not None else "geom"
    seg_kwargs = ({"label_map": label_map, "id_to_semantic": id_to_semantic}
                  if backend == "gt" else {})

    res = inspect_image(lines_uv, lines_xyz, camera_params, g_hat,
                        rgb_off=rgb_off, seg_backend=backend,
                        seg_kwargs=seg_kwargs, **kw)
    res["gravity_laser_frame"] = [round(float(x), 6) for x in np.asarray(g_hat)]
    res["skipped_lines"] = skipped
    res["n_triangulated"] = int(sum(len(v) for v in lines_xyz.values()))
    return res


# =====================================================================
# 보고
# =====================================================================
def format_report(result):
    """영역별 검측 결과를 사람이 읽는 표로 만든다."""
    lines = []
    s = result["summary"]
    seg = result.get("segmentation", {})
    lines.append(f"세그멘테이션 backend={seg.get('backend')}  "
                 f"영역 {s['n_regions']}개 (검측 {s['n_measured']} / "
                 f"기각 {s['n_rejected']})  라벨교정 {s['label_corrections']}건")
    st = result.get("assign_stats", {})
    if st:
        lines.append(f"  점 배분: 전체 {st['total']} → 할당 {st['assigned']} "
                     f"(화면밖 {st['out_of_image']}, 비검측 {st['ignored_class']}, "
                     f"침식 {st['eroded_away']}, 깊이불연속 {st['discontinuity']})")
    lines.append("")
    hdr = (f"  {'클래스':<14}{'검측':<10}{'각도(°)':>9}{'판정':>10}"
           f"{'자처짐(mm)':>12}{'평활판정':>10}{'σ_n(mm)':>9}{'점수':>7}")
    lines.append(hdr)
    lines.append("  " + "-" * (len(hdr) - 2))
    kind_ko = {"plane_vertical": "수직도", "plane_horizontal": "수평도",
               "axis_vertical": "축수직도"}
    for r in result["regions"]:
        if r["status"] != "measured":
            lines.append(f"  {r['class']:<14}{'기각':<10}"
                         f"{'-':>9}{'-':>10}{'-':>12}{'-':>10}{'-':>9}"
                         f"{r['n_points']:>7}   ← {r['reject_reason']}")
            continue
        j = r["judge"] or {}
        verdict = "합격" if j.get("is_pass") else "기준초과"
        f = r["flatness"] or {}
        fmax = (f"{f.get('max_gap_mm', 0.0):.2f}" if f.get("applicable")
                else "N/A")
        fjud = f.get("judgement", "N/A") if f.get("applicable") else "N/A"
        sn = r["uncertainty"]["sigma_normal_mm"]
        lines.append(f"  {r['class']:<14}{kind_ko.get(r['kind'], r['kind']):<10}"
                     f"{r['theta_deg']:>9.4f}{verdict:>10}"
                     f"{fmax:>12}{fjud:>10}{sn:>9.2f}{r['n_points']:>7}")
        if r["label_fusion"]["source"] == "geometric":
            lines.append(f"      ↳ 라벨 교정: {r['label_fusion_note']}")
        if f.get("judgement") == "측정불가":
            lines.append(f"      ↳ {f['note']}")
    return "\n".join(lines)


# ============ 자체 검증 ============
if __name__ == "__main__":
    _SYN = _load("synth_scene")
    print("=" * 78)
    print("영역별 품질검측 파이프라인 — 합성 씬 자체검증")
    print("=" * 78)
    scene = _SYN.build_scene()
    print(_SYN.describe(scene))
    print()

    for backend in ("gt", "geom"):
        kw = ({"label_map": scene["label_map"],
               "id_to_semantic": scene["id_to_semantic"]}
              if backend == "gt" else {})
        res = inspect_image(scene["lines_pixels"], scene["lines_xyz"],
                            scene["camera_params"], scene["g_hat"],
                            rgb_off=scene["rgb_off"],
                            seg_backend=backend, seg_kwargs=kw)
        print(f"── backend = {backend} " + "─" * 55)
        print(format_report(res))
        print()
        print(_SYN.score(scene, res))
        print()
