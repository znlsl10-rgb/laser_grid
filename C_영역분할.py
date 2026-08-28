#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
C_영역분할.py — [품질검측 알고리즘 C] 현장 이미지 영역 분할
========================================================================
입력 이미지를 벽·바닥·동바리·철근·거푸집 등으로 나눠, 영역별로 서로 다른
검측식(수직도/수평도/평활도)을 적용할 수 있게 한다.

【중요 — 레이저 OFF 프레임을 쓴다】
  레이저 ON 이미지는 초록 격자선이 화면을 덮어, 세그멘테이션 모델이
  선을 물체 경계로 오인한다. 그런데 하드웨어는 차영상 모드(PDF 2.2)를
  위해 이미 ON/OFF 두 프레임을 연속 촬영한다.
    · 세그멘테이션 → OFF 프레임 (레이저 없음, 깨끗함)
    · 선검출       → ON  프레임 (기존과 동일)
  두 프레임 간격이 수십 µs 라 마스크가 픽셀 단위로 그대로 정합된다.
  추가 하드웨어 비용 없이 문제가 사라진다.

【백엔드】
  "gt"   : Isaac Sim Semantics 어노테이터 정답 마스크.
           시뮬 검증의 기준선(baseline). 세그멘테이션 오차 0인 상태에서
           검측식·선검출만의 오차를 분리 측정할 때 쓴다.
  "geom" : 3D 점군만으로 다중 평면 RANSAC + 선형 클러스터 분리.
           네트워크·모델 없이 동작하는 폴백. 벽/바닥은 잘 나누지만
           "동바리 vs 기둥", "벽 vs 거푸집"은 구분하지 못한다.
  "sam"  : GroundingDINO/OWLv2 + SAM2  (Phase 3)
  "vlm"  : SAM2 자동마스크 + VLM 라벨링 (Phase 3)

【인터페이스】
  segment(rgb_off, table=None, camera_params=None, g_hat=None,
          backend="gt", **kw)
    → {"backend", "label_map", "class_names", "point_labels", "meta"}

  label_map    : (H,W) int  픽셀별 클래스 id      (gt/sam/vlm)
  point_labels : (N,)  int  격자점별 클래스 id     (geom)
  둘 중 하나만 채워지며, eq5_region_assign 이 양쪽 모두를 받는다.
========================================================================
"""
import numpy as np

try:
    from sklearn.cluster import DBSCAN
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

import importlib.util as _ilu, os as _os

def _load(name):
    spec = _ilu.spec_from_file_location(
        name, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), f"{name}.py"))
    m = _ilu.module_from_spec(spec); spec.loader.exec_module(m)
    return m

_EQ5 = _load("eq5_region_assign")


# =====================================================================
# Isaac Semantics 클래스 → 검측 클래스 사전
# =====================================================================
# 1-1_build_inspection_lab_realistic.py 가 prim 에 부여하는 semantic label 을
# eq5_region_assign 의 클래스 이름으로 옮긴다. 씬에 라벨을 추가하면
# 여기에도 한 줄 추가하면 된다.
ISAAC_SEMANTIC_LUT = {
    "wall": "wall",             "wall_back": "wall",
    "formwork_wall": "formwork_wall",
    "formwork_column": "formwork_column",
    "masonry": "masonry",       "plaster_wall": "plaster_wall",
    "floor": "floor",           "floor_top": "floor",
    "slab": "slab",             "ceiling": "ceiling",
    "panel": "wall",            # 평활도 검증 패널은 수직면으로 취급
    "shoring": "shoring",       # 동바리
    "column": "column",
    "rebar": "rebar",           # 철근
    "clutter": "clutter",       "debris": "clutter",
    "pallet": "clutter",        "scaffold": "clutter",
    "equipment": "equipment",
    "BACKGROUND": "background", "UNLABELLED": "background",
}


def map_isaac_labels(id_to_semantic):
    """
    Isaac 어노테이터의 idToLabels 를 검측 클래스 사전으로 변환한다.

    Parameters
    ----------
    id_to_semantic : {id(int|str): "class:wall"} 또는 {id: {"class": "wall"}}
        Replicator semantic_segmentation 어노테이터가 주는 형태 두 가지를
        모두 받는다.

    Returns
    -------
    class_names : {int id: 검측 클래스 이름}
    unknown     : list of str — LUT 에 없어 background 로 떨어진 라벨
    """
    class_names, unknown = {}, []
    for k, v in id_to_semantic.items():
        if isinstance(v, dict):
            raw = v.get("class", v.get("semantic", ""))
        else:
            raw = str(v)
        raw = raw.split(":")[-1].strip()
        cls = ISAAC_SEMANTIC_LUT.get(raw) or ISAAC_SEMANTIC_LUT.get(raw.lower())
        if cls is None:
            cls = "background"
            if raw:
                unknown.append(raw)
        class_names[int(k)] = cls
    return class_names, sorted(set(unknown))


# =====================================================================
# 백엔드 1: Isaac Sim 정답 마스크
# =====================================================================
def _backend_gt(rgb_off, label_map=None, id_to_semantic=None, **kw):
    """
    Isaac Semantics 어노테이터가 준 라벨맵을 그대로 쓴다.

    세그멘테이션 오차가 0인 상태를 만들어, 최종 측정오차 중 검측식과
    선검출이 기여하는 몫만 분리해 볼 수 있다(오차 분해의 기준선).
    """
    if label_map is None:
        raise ValueError("backend='gt' 에는 label_map 이 필요합니다 "
                         "(Isaac semantic_segmentation 어노테이터 출력).")
    lm = np.asarray(label_map)
    if lm.ndim == 3:      # (H,W,1) 또는 (H,W,4) uint8 인코딩 대응
        lm = lm[..., 0]
    if id_to_semantic:
        class_names, unknown = map_isaac_labels(id_to_semantic)
    else:
        class_names, unknown = {int(i): "background" for i in np.unique(lm)}, []
    return {"label_map": lm.astype(np.int32), "class_names": class_names,
            "point_labels": None,
            "meta": {"unknown_labels": unknown,
                     "n_classes": len(set(class_names.values()))}}


# =====================================================================
# 백엔드 2: 기하 전용 (다중 평면 RANSAC + 선형 클러스터)
# =====================================================================
def _backend_geom(rgb_off, table=None, g_hat=None, camera_params=None,
                  plane_threshold_m=0.01, min_plane_points=60,
                  max_planes=4, cluster_eps_m=0.08, min_linear_points=15,
                  merge_gap_m=0.25, occluder_margin_m=0.10, **kw):
    """
    3D 점군만으로 영역을 나눈다 — 모델·네트워크 불필요한 폴백.

    순서
    ----
    1. 순차 RANSAC 으로 큰 평면부터 뽑아내고 inlier 를 제거한다.
    2. 각 평면 inlier 를 DBSCAN 으로 공간 분리하고(같은 평면 위의 떨어진
       두 벽을 한 영역으로 묶지 않기 위함), 그중 가림 그림자로 갈라진
       조각만 다시 합친다(_merge_occlusion_split).
    3. 남은 점을 DBSCAN 으로 묶고, 선형성이 높은 덩어리를 선형 부재로 본다.
    4. 각 덩어리의 기하 증거로 wall / floor / shoring 라벨을 붙인다.

    한계 (반드시 인지할 것)
    ----
    · "동바리 vs 기둥 vs 철근", "벽 vs 거푸집 vs 조적"은 구분하지 못한다.
      허용오차(KCS)가 달라지므로 이 백엔드만으로는 정확한 판정이 어렵다.
    · 벽에 바싹 붙은 얇은 동바리는 벽 평면의 inlier 로 흡수될 수 있다.
    """
    if table is None or len(table["xyz"]) == 0:
        raise ValueError("backend='geom' 에는 3D 점 테이블(table)이 필요합니다.")
    if g_hat is None:
        g_hat = _EQ5._EQ3.G_UPRIGHT

    pts = np.asarray(table["xyz"], dtype=float)
    N = len(pts)
    labels = np.full(N, -1, dtype=np.int32)
    remaining = np.arange(N)
    deferred = []            # 평면으로 확정되지 않아 선형 단계로 넘길 점
    class_names, next_id = {}, 0

    # ── 0. 선형 부재 선추출 ──
    # 얇은 부재는 점이 적어 평면 단계에서 쉽게 흡수된다. 실제로 동바리
    # 431점에 인접한 바닥 85점이 섞이자 덩어리 전체가 평면으로 판정되어
    # 동바리가 사라졌다. 그래서 평면을 뽑기 전에 먼저 걷어낸다.
    #
    # 벽의 가느다란 조각을 잘못 집지 않도록 두 가지를 함께 요구한다.
    #   · robust 축 적합이 유효하고 축 주변 점 비율이 충분할 것
    #   · 그 점들이 실제로 1D 부재로 판별될 것 (판/원통 두께비 검사)
    for grp in _spatial_groups(pts, cluster_eps_m, min_linear_points):
        gidx = np.arange(N)[grp]
        try:
            # 혼합 덩어리에서 얇은 부재를 끄집어내는 것이 목적이므로
            # inlier 비율 하한을 낮춘다. 실제로 동바리 471점이 바닥 491점과
            # 한 덩어리를 이루면 비율이 0.49 로, 절반을 요구하면 놓친다.
            # 대신 아래 형상 판별(판/원통 두께비)로 벽 조각을 걸러낸다.
            ax = _EQ5._EQ2.fit_axis_ransac(pts[gidx], min_inlier_frac=0.15)
        except Exception:
            continue
        if not ax["is_valid"]:
            continue
        sub = gidx[ax["inlier_mask"]]
        if len(sub) < min_linear_points:
            continue
        if _EQ5.geometric_evidence(pts[sub], g_hat)["shape"] != "linear_vertical":
            continue
        labels[sub] = next_id
        # 기하는 동바리/기둥/철근을 구분하지 못한다 → 가장 흔한 shoring
        class_names[next_id] = "shoring"
        next_id += 1
    if next_id:
        remaining = np.where(labels < 0)[0]

    # ── 1~2. 순차 RANSAC 평면 추출 ──
    for _ in range(max_planes):
        if len(remaining) < min_plane_points:
            break
        sub = pts[remaining]
        try:
            # 방향 무관 TLS RANSAC 필수.
            # fit_plane_ransac(Z=aX+bY+c)은 광축과 나란한 면(비스듬히
            # 들어온 바닥)을 표현하지 못해 법선이 수 도(°) 틀어진다.
            plane, inlier_mask = _EQ5._EQ2.fit_plane_tls_ransac(
                sub, threshold=plane_threshold_m)
        except Exception:
            break
        if inlier_mask.sum() < min_plane_points:
            break

        inl_global = remaining[inlier_mask]
        consumed = []
        groups = _merge_occlusion_split(
            pts, inl_global,
            _spatial_groups(pts[inl_global], cluster_eps_m, min_plane_points),
            plane, merge_gap_m)
        for grp in groups:
            gidx = inl_global[grp]
            ev = _EQ5.geometric_evidence(pts[gidx], g_hat)
            cls = {"plane_vertical": "wall",
                   "plane_horizontal": "floor"}.get(ev["shape"])
            if cls is None:
                # 평면으로 확정되지 않은 덩어리(얇은 원통 등)는 라벨을 붙이지
                # 않고 선형 단계로 넘긴다. 평면 후보에서는 빼야 다음 회차가
                # 같은 점을 다시 집어 무한히 맴돌지 않는다.
                deferred.append(gidx)
                consumed.append(gidx)
                continue
            labels[gidx] = next_id
            class_names[next_id] = cls
            next_id += 1
            consumed.append(gidx)

        if not consumed:
            # 이번 회차 inlier 가 어떤 그룹도 이루지 못함(산발적) → 통째로 보류
            deferred.append(inl_global)
            consumed.append(inl_global)
        drop = np.concatenate(consumed)
        remaining = np.setdiff1d(remaining, drop, assume_unique=False)

    # ── 3~4. 잔여 점 + 보류 점에서 선형 부재 (선추출에서 놓친 것) ──
    leftover = (np.union1d(remaining, np.concatenate(deferred))
                if deferred else remaining)
    n_single_line = 0
    for grp in _spatial_groups(pts[leftover], cluster_eps_m, min_linear_points):
        gidx = leftover[grp]
        ev = _EQ5.geometric_evidence(pts[gidx], g_hat)
        if ev["shape"] == "linear_vertical":
            labels[gidx] = next_id
            # 기하는 동바리/기둥/철근을 구분하지 못한다 → 가장 흔한 shoring 으로
            # 두되, meta 에 구분 불가임을 남긴다.
            class_names[next_id] = "shoring"
            next_id += 1
        elif (ev["shape"] == "degenerate_line"
              and ev.get("theta_deg") is not None
              and ev["theta_deg"] < 30.0
              and _stands_in_front(pts, gidx, g_hat, occluder_margin_m)):
            # 격자선이 한 줄만 걸린 부재. 단면이 잡히지 않아 형상만으로는
            # 벽에 그은 선 한 줄과 구분되지 않는다. 그러나 그 선이 주변
            # 면보다 뚜렷하게 **앞에** 서 있으면 가림물, 곧 부재다.
            #
            # 연직에서 30° 안이라는 조건을 함께 건다. 이것이 없으면 면
            # 경계의 짧은 조각이 부재로 잡혀 "축 수직도 89.8°, 기준초과"
            # 같은 결과가 나온다(실제 내보내기에서 44점짜리 조각이 그랬다).
            # linear_vertical 판정과 같은 문턱이다.
            #
            # 축 방향은 오히려 이 경우가 더 깨끗하다. 원통 단면이 주축을
            # 끌어당기는 일이 없기 때문이다. 대신 지름을 알 수 없으므로
            # 부재 종류(동바리/기둥/철근)와 세장비 판정은 성립하지 않는다.
            labels[gidx] = next_id
            class_names[next_id] = "shoring"
            next_id += 1
            n_single_line += 1

    bg_id = next_id
    class_names[bg_id] = "background"
    labels[labels < 0] = bg_id

    return {"label_map": None, "class_names": class_names,
            "point_labels": labels,
            "meta": {"n_regions": next_id,
                     "unassigned": int((labels == bg_id).sum()),
                     "single_line_members": n_single_line,
                     "caveat": (
                         "기하 전용 백엔드는 동바리/기둥/철근과 벽/거푸집/조적을 "
                         "구분하지 못함"
                         + (f" / 격자선이 한 줄만 걸린 부재 {n_single_line}개 — "
                            f"단면 미확인, 축 방향만 유효"
                            if n_single_line else ""))}}


def _stands_in_front(pts, gidx, g_hat, margin_m):
    """
    이 점 무리가 주변 면보다 앞에 서 있는가 (가림물인가).

    격자선 한 줄짜리 무리는 형상만으로는 부재인지 면의 일부인지 알 수
    없다. 가릴 수 있는 것은 깊이뿐이다. 무리의 화면상 좌우 이웃과 깊이를
    비교해, 무리가 뚜렷하게 가까우면 앞에 선 부재로 본다.

    이웃을 화면이 아니라 3D 로 고른다. 이 단계에는 화소 좌표가 없고,
    같은 시선 방향에서 더 먼 점을 찾으면 되므로 시선 각도로 이웃을
    정의하는 편이 정확하다.
    """
    sub = pts[gidx]
    if len(sub) < 5:
        return False
    z = float(np.median(sub[:, 2]))
    # 무리의 시선 방향(단위벡터) 중심
    d0 = sub / np.linalg.norm(sub, axis=1, keepdims=True)
    c = d0.mean(axis=0); c /= np.linalg.norm(c)

    others = np.setdiff1d(np.arange(len(pts)), gidx, assume_unique=False)
    if len(others) < 20:
        return False
    o = pts[others]
    do = o / np.linalg.norm(o, axis=1, keepdims=True)
    ang = np.degrees(np.arccos(np.clip(do @ c, -1, 1)))
    # 시야각 15° 안의 이웃 — 같은 장면 안이면서 무리 밖
    near = o[ang < 15.0]
    if len(near) < 20:
        return False
    z_bg = float(np.median(near[:, 2]))
    return (z_bg - z) > margin_m


def _merge_occlusion_split(all_points, member_idx, groups, plane, merge_gap_m,
                           front_margin_m=0.03):
    """
    같은 평면에서 갈라진 조각 중 **가림 때문에** 갈라진 것만 다시 합친다.

    왜 필요한가
    ----------
    DBSCAN eps 는 점 밀도에 맞춰 자동으로 좁아진다. 격자를 조밀하게 만들면
    eps 도 함께 좁아지므로, 앞에 선 부재가 벽에 드리운 빈 띠가 갑자기
    "서로 다른 두 벽" 으로 보이게 된다. 실제로 V선을 20 → 40 개로 늘리자
    벽 6,844점이 4,188 + 2,667 로 쪼개졌고, 직선자 프로파일이 1.15m 에서
    0.69m 로 줄어 평활도가 3.9mm 에서 2.6mm 로 낮게 나왔다. 각도는 두
    조각 모두 같은 법선을 주므로 멀쩡했고, 그래서 평활도만 조용히 틀렸다.

    무엇으로 구분하나 — 두 가지를 순서대로 본다.
      1. 좁은 틈      두 조각의 최근접 거리가 merge_gap_m 미만이면 합친다.
                     Ø48.6mm 동바리의 그림자는 10cm 안쪽이다.
      2. 가림물 확인   틈이 넓어도, 그 사이에 **평면보다 앞에 있는 점**이
                     있으면 가림물이 서 있다는 뜻이므로 합친다. 실제
                     내보내기에서 기둥 3개가 벽을 4토막 낸 경우가 이쪽이다
                     (기둥 간격 0.5m 이상이라 1번으로는 안 걸린다).

    개구부·서로 다른 벽은 둘 다에 걸리지 않는다. 사이가 넓고, 그 사이에
    앞에 선 것도 없기 때문이다.
    """
    if len(groups) < 2 or merge_gap_m <= 0:
        return groups
    try:
        from scipy.spatial import cKDTree
    except Exception:
        return groups

    P = np.asarray(all_points, float)
    member = np.asarray(member_idx)
    pts = P[member]
    parent = list(range(len(groups)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i

    # 평면 법선을 센서(원점) 쪽으로 맞춘다. 그래야 부호 있는 거리의
    # 양수가 "앞에 있다" 를 뜻한다.
    n = np.array(plane[:3], float); d = float(plane[3])
    nn = np.linalg.norm(n)
    if nn < 1e-12:
        return groups
    n /= nn; d /= nn
    if d < 0:                      # 원점의 부호 있는 거리 = d
        n, d = -n, -d
    outside = np.setdiff1d(np.arange(len(P)), member, assume_unique=False)
    tree_out = cKDTree(P[outside]) if len(outside) else None

    trees = [cKDTree(pts[g]) for g in groups]
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            dist, idx = trees[j].query(pts[groups[i]], k=1)
            k = int(np.argmin(dist))
            gap = float(dist[k])
            if gap < merge_gap_m:
                merge = True
            elif tree_out is None:
                merge = False
            else:
                # 두 조각의 최근접 쌍을 잇는 선분 위에 가림물이 있는가
                a = pts[groups[i]][k]
                b = pts[groups[j]][int(idx[k])]
                t = np.linspace(0.15, 0.85, 7)[:, None]
                seg = a * (1 - t) + b * t
                near = tree_out.query_ball_point(seg, r=max(gap * 0.5, 0.05))
                cand = sorted({q for lst in near for q in lst})
                merge = False
                if cand:
                    sd = P[outside][cand] @ n + d      # 양수 = 센서 쪽
                    merge = bool((sd > front_margin_m).any())
            if merge:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[ri] = rj
    merged = {}
    for i, g in enumerate(groups):
        merged.setdefault(find(i), []).append(g)
    return [np.concatenate(v) if len(v) > 1 else v[0] for v in merged.values()]


def _spatial_groups(points, eps_m, min_points, min_samples=4):
    """
    DBSCAN 으로 공간 분리. sklearn 없으면 전체를 한 덩어리로 반환.

    eps 는 점 밀도에 맞춰 자동 확대한다. 격자점 간격은 측정거리·격자
    해상도·면의 기울기에 따라 크게 달라지므로(1m에서 ~5cm, 3m에서
    ~15cm), eps 를 상수로 두면 먼 영역이 통째로 노이즈 처리된다.
    → eps = max(지정값, 3 × 최근접이웃거리 중앙값)
    """
    n = len(points)
    if n < min_points:
        return []
    if not HAS_SKLEARN:
        return [np.arange(n)]

    eps = float(eps_m)
    try:
        from scipy.spatial import cKDTree
        # 자기 자신 제외한 최근접 이웃 거리
        d, _ = cKDTree(points).query(points, k=2)
        nn_med = float(np.median(d[:, 1]))
        if np.isfinite(nn_med) and nn_med > 0:
            eps = max(eps, 3.0 * nn_med)
    except Exception:
        pass

    db = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    out = []
    for lbl in sorted(set(db.labels_)):
        if lbl == -1:
            continue
        idx = np.where(db.labels_ == lbl)[0]
        if len(idx) >= min_points:
            out.append(idx)
    return out


# =====================================================================
# 백엔드 3·4: 학습 모델 (Phase 3)
# =====================================================================
def _backend_sam(rgb_off, text_prompts=None, **kw):
    """
    [Phase 3] GroundingDINO/OWLv2 + SAM2.

    구현 개요
    --------
    1. 오픈보캐뷸러리 검출기에 건설 도메인 텍스트 프롬프트를 준다
       (예: "concrete wall", "concrete floor slab",
             "steel pipe shoring post", "reinforcing steel bar",
             "formwork panel")
    2. 검출 박스를 SAM2 에 프롬프트로 넣어 인스턴스 마스크를 얻는다
    3. 프롬프트 문자열 → ISAAC_SEMANTIC_LUT 와 같은 클래스 이름으로 매핑
    4. label_map 으로 합성해 반환

    로컬 GPU 추론이므로 API 키가 필요 없고, 현장 배포에 가장 가깝다.
    """
    raise NotImplementedError(
        "backend='sam' 은 Phase 3 구현 대상입니다. "
        "현재는 'gt'(시뮬 검증) 또는 'geom'(폴백)을 사용하세요.")


def _backend_vlm(rgb_off, **kw):
    """
    [Phase 3] SAM2 자동 마스크 + VLM 라벨링.

    구현 개요
    --------
    1. SAM2 automatic mask generator 로 클래스 불명 마스크를 다수 생성
    2. 각 마스크의 크롭을 모아 VLM 에 한 번에 질의해 건설 도메인 라벨 부여
       (기하가 구분 못 하는 "동바리 vs 기둥", "벽 vs 거푸집"에 강함)
    3. 라벨을 label_map 으로 합성

    이미지당 1회 호출로 비용을 억제한다. 결과는 eq5_region_assign.fuse_label()
    에서 기하 증거와 교차검증되므로, VLM 이 틀려도 벽/바닥 혼동은 교정된다.
    """
    raise NotImplementedError(
        "backend='vlm' 은 Phase 3 구현 대상입니다. "
        "현재는 'gt'(시뮬 검증) 또는 'geom'(폴백)을 사용하세요.")


# =====================================================================
# 디스패처
# =====================================================================
_BACKENDS = {"gt": _backend_gt, "geom": _backend_geom,
             "sam": _backend_sam, "vlm": _backend_vlm}


def segment(rgb_off, backend="gt", **kw):
    """
    현장 이미지를 검측 영역으로 분할한다.

    Parameters
    ----------
    rgb_off : (H,W,3) | None
        **레이저 OFF 프레임**. 격자선이 없는 깨끗한 이미지를 넣어야 한다.
        backend='geom' 은 점군만 쓰므로 None 이어도 된다.
    backend : {"gt", "geom", "sam", "vlm"}
    **kw    : 백엔드별 인자
        gt   : label_map, id_to_semantic
        geom : table, g_hat, camera_params, plane_threshold_m, ...
        sam  : text_prompts

    Returns
    -------
    dict — backend, label_map, class_names, point_labels, meta
    """
    if backend not in _BACKENDS:
        raise ValueError(f"알 수 없는 backend='{backend}'. "
                         f"가능: {sorted(_BACKENDS)}")
    out = _BACKENDS[backend](rgb_off, **kw)
    out["backend"] = backend
    return out


# ============ 자체 검증 ============
if __name__ == "__main__":
    print("[알고리즘 C] 영역 분할 검증")
    rng = np.random.default_rng(11)
    g = _EQ5._EQ3.G_UPRIGHT

    # 벽(Z=1.2 정면) + 바닥(Y=0.8 아래, 비스듬) + 동바리(수직 파이프) 합성
    wall = np.column_stack([rng.uniform(-.6, .6, 500), rng.uniform(-.5, .5, 500),
                            np.full(500, 1.2)]) + rng.normal(0, 5e-4, (500, 3))
    floor = np.column_stack([rng.uniform(-.6, .6, 400), np.full(400, 0.8),
                             rng.uniform(0.5, 1.15, 400)]) \
        + rng.normal(0, 5e-4, (400, 3))
    t = rng.uniform(-0.45, 0.45, 150); ph = rng.uniform(-1, 1, 150)
    e1, e2 = _EQ5._EQ2.plane_tangent_basis(g)
    post = (np.outer(t, g) + 0.0243 * (np.cos(ph)[:, None] * e1
                                       + np.sin(ph)[:, None] * e2)
            + np.array([0.35, 0.1, 0.85])) + rng.normal(0, 8e-4, (150, 3))

    xyz = np.vstack([wall, floor, post])
    truth = np.r_[np.zeros(500, int), np.ones(400, int), np.full(150, 2)]
    table = {"xyz": xyz, "uv": np.zeros((len(xyz), 2)),
             "lid": np.array(["V0"] * len(xyz), dtype=object),
             "seq": np.arange(len(xyz))}

    res = segment(None, backend="geom", table=table, g_hat=g)
    cn, pl = res["class_names"], res["point_labels"]
    print(f"  backend={res['backend']}  영역 {res['meta']['n_regions']}개  "
          f"미할당 {res['meta']['unassigned']}점")

    name_of = {0: "wall", 1: "floor", 2: "shoring"}
    for tid in (0, 1, 2):
        got = pl[truth == tid]
        vals, cnt = np.unique(got, return_counts=True)
        top = vals[np.argmax(cnt)]
        cls = cn.get(int(top), "?")
        acc = cnt.max() / len(got)
        mark = "PASS" if cls == name_of[tid] else "FAIL"
        print(f"  참값 {name_of[tid]:8s} → 최다 라벨 '{cls}' "
              f"({acc*100:5.1f}% 일치)  {mark}")

    print(f"  주의: {res['meta']['caveat']}")

    # gt 백엔드 인터페이스
    lm = np.zeros((40, 40), np.int32); lm[:20] = 1; lm[20:] = 2
    gt = segment(None, backend="gt", label_map=lm,
                 id_to_semantic={0: "class:BACKGROUND", 1: "class:wall",
                                 2: "class:floor_top"})
    print(f"  gt 백엔드 class_names={gt['class_names']}")

    for bk in ("sam", "vlm"):
        try:
            segment(None, backend=bk)
        except NotImplementedError as e:
            print(f"  {bk} 백엔드: {str(e).splitlines()[0]}")
