"""
[식 ⑤] 영역 할당 — 세그멘테이션 마스크 × 레이저 격자점 결합
==================================================================
세그멘테이션(C_영역분할)이 준 픽셀 라벨맵과 삼각측량(eq1)이 준 3D 점을
결합해, 검측 단위인 "영역(region)"을 만든다.

【이 모듈이 푸는 문제】
  1. 경계 오염
     레이저 선이 벽에서 바닥으로 넘어가는 지점의 점은 두 면 어디에도
     속하지 않는데, 그대로 평면적합에 들어가면 법선을 끌어당긴다.
     → 마스크 침식(erode) + 선 방향 깊이 불연속 검출로 제거.

  2. 세그멘테이션 오류
     VLM/SAM 라벨은 완벽하지 않다. 그런데 우리는 3D 점군과 중력을
     이미 갖고 있으므로 **기하로 의미를 검증**할 수 있다.
     → 법선이 중력과 수직이면 VLM이 뭐라 했든 벽이고,
       얇고 곧게 뻗은 1D 클러스터면 동바리다.
     반대로 "동바리 vs 기둥 vs 철근"처럼 기하가 구분하지 못하는 것은
     의미 라벨이 이긴다. 이 비대칭이 융합 규칙의 핵심이다.

  3. 측정 가능성
     같은 사진 안에서도 정면인 벽과 경사진 바닥은 정밀도가 다르다.
     σ_Z = σ_u·Z²/(f·b) (PDF 5.2) 에 더해, 면이 시선에 비스듬할수록
     깊이 오차가 법선 방향으로 1/cos(입사각) 만큼 증폭된다.
     → 영역별 불확실도를 산출해 "측정불가" 게이트를 건다.

【좌표계】 조사기 좌표계 (X 우, Y 하, Z 전방) — eq1 규약과 동일
==================================================================
"""
import numpy as np

try:
    from scipy import ndimage as _ndi
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

import importlib.util as _ilu, os as _os

def _load(name):
    spec = _ilu.spec_from_file_location(
        name, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), f"{name}.py"))
    m = _ilu.module_from_spec(spec); spec.loader.exec_module(m)
    return m

_EQ2 = _load("eq2_plane_fit")
_EQ3 = _load("eq3_orientation")


# =====================================================================
# 클래스 체계
# =====================================================================
# 검측 방식별 분류. 세그멘테이션 백엔드(C_영역분할)는 이 이름을 뱉어야 한다.
PLANE_VERTICAL_CLASSES = {"wall", "formwork_wall", "formwork_column", "masonry",
                          "plaster_wall"}
PLANE_HORIZONTAL_CLASSES = {"floor", "slab", "ceiling"}
LINEAR_VERTICAL_CLASSES = {"shoring", "column", "rebar"}   # 동바리·기둥·철근
IGNORE_CLASSES = {"background", "clutter", "equipment", "unknown"}


MEASURE_KIND = {}
for _c in PLANE_VERTICAL_CLASSES:   MEASURE_KIND[_c] = "plane_vertical"
for _c in PLANE_HORIZONTAL_CLASSES: MEASURE_KIND[_c] = "plane_horizontal"
for _c in LINEAR_VERTICAL_CLASSES:  MEASURE_KIND[_c] = "axis_vertical"

# eq3.KCS_SPEC 조회용 (세부 클래스 → 시방 분류)
KCS_CLASS = {"wall": "wall", "plaster_wall": "wall", "masonry": "masonry",
             "formwork_wall": "formwork", "formwork_column": "formwork",
             "floor": "floor", "slab": "slab", "ceiling": "slab",
             "shoring": "shoring", "column": "column", "rebar": "rebar"}


# =====================================================================
# 1. 점 테이블
# =====================================================================
def build_point_table(lines_pixels, lines_xyz):
    """
    선 단위 딕셔너리를 평평한 점 테이블로 편다.

    선 내 순번(seq)을 보존해야 깊이 불연속 검출에서 "선을 따라 인접한
    점"을 알 수 있다.

    Parameters
    ----------
    lines_pixels : {lid: [(u,v), ...]}   A_선검출 출력 (선 방향 정렬)
    lines_xyz    : {lid: [(X,Y,Z), ...]} 같은 순서의 eq1 삼각측량 결과

    Returns
    -------
    dict — uv (N,2), xyz (N,3), lid (N,), seq (N,)
    """
    uv, xyz, lid, seq = [], [], [], []
    for key in lines_pixels:
        if key not in lines_xyz:
            continue
        p2 = np.asarray(lines_pixels[key], dtype=float)
        p3 = np.asarray(lines_xyz[key], dtype=float)
        n = min(len(p2), len(p3))
        if n == 0:
            continue
        uv.append(p2[:n, :2]); xyz.append(p3[:n, :3])
        lid.extend([key] * n); seq.extend(range(n))
    if not uv:
        return {"uv": np.empty((0, 2)), "xyz": np.empty((0, 3)),
                "lid": np.empty(0, dtype=object), "seq": np.empty(0, dtype=int)}
    return {"uv": np.vstack(uv), "xyz": np.vstack(xyz),
            "lid": np.array(lid, dtype=object), "seq": np.array(seq, dtype=int)}


# =====================================================================
# 2. 마스크 침식 — 경계 오염 제거
# =====================================================================
def erode_mask(mask, px):
    """
    이진 마스크를 px 만큼 침식한다.

    경계 근처 격자점은 (a) 마스크 자체의 부정확, (b) 레이저 선이 두 면에
    걸치며 생기는 서브픽셀 혼합, (c) 실루엣에서의 삼각측량 부정확이
    겹치므로 제외하는 편이 항상 낫다. 얇은 동바리에서는 침식이 과하면
    점이 다 날아가므로 호출부에서 폭에 맞춰 px 를 정한다.
    """
    if px <= 0:
        return mask.astype(bool)
    m = mask.astype(bool)
    if HAS_SCIPY:
        return _ndi.binary_erosion(m, iterations=int(px), border_value=0)
    # scipy 없을 때: 상하좌우 시프트 AND 를 px 회 반복 (4-이웃 침식)
    for _ in range(int(px)):
        e = m.copy()
        e[1:, :] &= m[:-1, :]; e[:-1, :] &= m[1:, :]
        e[:, 1:] &= m[:, :-1]; e[:, :-1] &= m[:, 1:]
        e[0, :] = e[-1, :] = e[:, 0] = e[:, -1] = False
        m = e
    return m


def erosion_px_for_class(cls, default_px=3, thin_px=1):
    """선형 부재는 폭이 좁으므로 침식을 약하게 준다."""
    return thin_px if cls in LINEAR_VERTICAL_CLASSES else default_px


# =====================================================================
# 3. 깊이 불연속 제거
# =====================================================================
def mark_depth_discontinuity(table, jump_ratio=0.05, min_jump_m=0.02):
    """
    선을 따라가며 Z가 급변하는 지점의 양옆 점을 표시한다.

    벽→바닥, 벽→동바리처럼 면이 바뀌는 곳에서 레이저 선의 깊이는
    계단처럼 끊긴다. 마스크 침식만으로는 마스크가 부정확할 때 남으므로
    깊이 자체로 한 번 더 거른다.

    임계는 상대(jump_ratio × 국소 Z)와 절대(min_jump_m) 중 큰 쪽을 쓴다.
    거리가 멀수록 σ_Z ∝ Z² 로 커지므로 상대 임계가 필요하고,
    가까울 때 노이즈로 오검출되지 않도록 절대 하한이 필요하다.

    Returns
    -------
    bad : (N,) bool — True면 불연속 인접점(제외 대상)
    """
    N = len(table["seq"])
    bad = np.zeros(N, dtype=bool)
    if N == 0:
        return bad
    Z = table["xyz"][:, 2]
    for key in np.unique(table["lid"]):
        idx = np.where(table["lid"] == key)[0]
        idx = idx[np.argsort(table["seq"][idx])]
        if len(idx) < 3:
            continue
        z = Z[idx]
        dz = np.abs(np.diff(z))
        loc = np.maximum(np.abs(z[:-1]), np.abs(z[1:]))
        thr = np.maximum(loc * jump_ratio, min_jump_m)
        jump = dz > thr
        bad[idx[:-1][jump]] = True
        bad[idx[1:][jump]] = True
    return bad


# =====================================================================
# 4. 점 → 영역 할당
# =====================================================================
def assign_points_to_regions(table, label_map, class_names,
                             erode_default_px=3, erode_thin_px=1,
                             drop_discontinuity=True, min_points=12):
    """
    라벨맵을 격자점 픽셀 좌표에서 샘플링해 영역별 점 인덱스를 만든다.

    Parameters
    ----------
    table : build_point_table() 결과
    label_map : (H, W) int — 픽셀별 클래스 id
    class_names : {id: "wall", ...}
    min_points : int — 이 미만이면 영역을 만들지 않는다

    Returns
    -------
    regions : list of dict — {class, class_id, idx, n_points}
    stats   : dict — 제외 사유별 점 수
    """
    H, W = label_map.shape[:2]
    N = len(table["seq"])
    stats = {"total": N, "out_of_image": 0, "ignored_class": 0,
             "eroded_away": 0, "discontinuity": 0, "assigned": 0}
    if N == 0:
        return [], stats

    u = np.rint(table["uv"][:, 0]).astype(int)
    v = np.rint(table["uv"][:, 1]).astype(int)
    inside = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    stats["out_of_image"] = int((~inside).sum())

    disc = mark_depth_discontinuity(table) if drop_discontinuity \
        else np.zeros(N, dtype=bool)
    stats["discontinuity"] = int((disc & inside).sum())

    regions = []
    for cid, cls in class_names.items():
        if cls in IGNORE_CLASSES:
            raw = np.zeros(N, dtype=bool)
            raw[inside] = (label_map[v[inside], u[inside]] == cid)
            stats["ignored_class"] += int(raw.sum())
            continue

        mask = (label_map == cid)
        if not mask.any():
            continue
        er = erode_mask(mask, erosion_px_for_class(cls, erode_default_px,
                                                   erode_thin_px))

        raw = np.zeros(N, dtype=bool)
        keep = np.zeros(N, dtype=bool)
        raw[inside] = mask[v[inside], u[inside]]
        keep[inside] = er[v[inside], u[inside]]
        stats["eroded_away"] += int((raw & ~keep).sum())

        keep &= ~disc
        if keep.sum() < min_points:
            continue
        idx = np.where(keep)[0]
        stats["assigned"] += len(idx)
        regions.append({"class": cls, "class_id": int(cid),
                        "idx": idx, "n_points": int(len(idx))})
    return regions, stats


# =====================================================================
# 5. 기하 증거 산출
# =====================================================================
def geometric_evidence(points_3d, g_hat,
                       linear_ratio=0.15, planar_ratio=0.15,
                       thin_extent_m=0.12, align_deg=30.0,
                       min_thickness_ratio=0.02,
                       min_planar_extent_m=0.004):
    """
    점군 자체가 말하는 형상·자세를 뽑는다 (의미 라벨과 독립).

    PCA 고윳값 λ1≥λ2≥λ3 로 형상을 판별한다.
      선형(1D) : λ2/λ1 이 작고(가늘고 길다), 횡방향 실제 크기도 작으며,
                 **세 번째 축에도 두께가 있을 것** (λ3/λ2 ≥ min_thickness_ratio)
      평면(2D) : λ3/λ2 가 작을 것 (납작하다)

    세 번째 조건이 없으면 가느다란 **판 조각**이 선형 부재로 오인된다.
    실제로 동바리에 가려 갈라진 벽 조각이 그렇게 오분류되어 바닥까지
    놓치는 연쇄 실패가 있었다. 원통(동바리)은 보이는 반쪽이 깊이 방향으로
    휘어 λ3 가 살아 있는 반면(실측 λ3/λ2 ≈ 0.144), 판 조각은 납작해
    λ3 가 노이즈 수준이다(실측 0.0004). 약 400배 차이라 분리가 확실하다.

    Returns
    -------
    dict — shape, normal, axis, theta_deg, confidence, eig, extent_m
    """
    pts = np.asarray(points_3d, dtype=float)
    out = {"shape": "unknown", "normal": None, "axis": None,
           "theta_deg": None, "confidence": 0.0,
           "n_points": int(len(pts))}
    if len(pts) < 6:
        return out

    c = pts.mean(axis=0)
    _, sv, vt = np.linalg.svd(pts - c, full_matrices=False)
    lam = (sv ** 2) / max(len(pts) - 1, 1)
    out["eig"] = [float(x) for x in lam]
    # 각 주축 방향 실제 크기 (표준편차 → 대략적 반폭)
    out["extent_m"] = [float(np.sqrt(max(x, 0.0))) for x in lam]

    r21 = lam[1] / lam[0] if lam[0] > 1e-18 else 1.0
    r32 = lam[2] / lam[1] if lam[1] > 1e-18 else 1.0
    g = _EQ3.normalize(g_hat)

    out["thickness_ratio"] = float(r32)

    # ── 퇴화 검사: 1차원 점집합에는 평면을 맞출 수 없다 ──
    # 격자선 한 줄만 걸린 부재(가는 기둥·동바리)는 점이 한 직선 위에
    # 놓인다. 그 집합을 지나는 평면은 무수히 많으므로 법선이 아무 값이나
    # 나오고, 그대로 두면 "입사각 89.9°, 수직도 0.0000°, 합격" 같은
    # 지어낸 결과가 조서에 실린다. 실제 내보내기에서 기둥 3개가 전부
    # 그렇게 나왔다.
    #
    # 두 번째 주축의 실제 크기로 판정한다. 비율이 아니라 미터로 봐야
    # 한다 — 아주 긴 선은 λ2/λ1 이 작아도 λ2 자체는 클 수 있고, 짧고
    # 통통한 조각은 그 반대다. 실측한 횡방향 퍼짐:
    #
    #   동바리 Ø48.6mm       14.25 mm     ← 원통 표면이 감기며 생긴 폭
    #   철근   Ø25.4mm        7.23 mm
    #   ─────────────── 문턱 4mm ───────────────
    #   벽 위 V선 한 줄        0.96 mm     ← σ_Z 노이즈뿐
    #   기둥 위 V선 한 줄      0.00 mm     ← raycast, 노이즈 없음
    #
    # 문턱은 측정 노이즈(σ_normal)보다 커야 뜻이 있다. 노이즈가 부재
    # 반경만 해지면 그 부재는 애초에 단면이 잡히지 않는다.
    ext2 = float(np.sqrt(max(lam[1], 0.0)))
    if ext2 < min_planar_extent_m:
        axis = vt[0] / np.linalg.norm(vt[0])
        th = _EQ3.measure_from_gravity(axis, g, "axis_vertical")
        out.update(shape="degenerate_line", axis=axis, theta_deg=th,
                   confidence=0.0,
                   note=(f"횡방향 퍼짐 {ext2*1000:.1f}mm — 격자선이 한 줄만 "
                         f"걸려 단면이 잡히지 않는다. 평면을 맞출 수 없고, "
                         f"선형 부재인지 면의 일부인지는 깊이 불연속으로만 "
                         f"가릴 수 있다"))
        return out
    if (r21 < linear_ratio
            and np.sqrt(max(lam[1], 0.0)) < thin_extent_m
            and r32 >= min_thickness_ratio):
        # 1D 선형 부재 (가늘고 길며, 3번째 축에도 두께가 있음 = 원통)
        axis = vt[0] / np.linalg.norm(vt[0])
        th = _EQ3.measure_from_gravity(axis, g, "axis_vertical")
        out.update(shape="linear_vertical" if th < align_deg else "linear_oblique",
                   axis=axis, theta_deg=th,
                   confidence=float(np.clip(1.0 - r21 / linear_ratio, 0, 1)))
        return out

    if r32 < planar_ratio:
        # 2D 평면
        n = vt[2] / np.linalg.norm(vt[2])
        th_v = _EQ3.measure_from_gravity(n, g, "plane_vertical")
        th_h = _EQ3.measure_from_gravity(n, g, "plane_horizontal")
        if th_v < align_deg:
            shape, th = "plane_vertical", th_v
        elif th_h < align_deg:
            shape, th = "plane_horizontal", th_h
        else:
            shape, th = "plane_oblique", min(th_v, th_h)
        out.update(shape=shape, normal=n, theta_deg=th,
                   confidence=float(np.clip(1.0 - r32 / planar_ratio, 0, 1)))
    return out


# =====================================================================
# 6. 의미 × 기하 융합
# =====================================================================
def fuse_label(semantic_class, evidence, min_geom_conf=0.5):
    """
    세그멘테이션 라벨과 기하 증거를 합쳐 최종 클래스를 정한다.

    【비대칭 규칙 — 이 모듈의 핵심】
      · 벽 ↔ 바닥 : **기하가 이긴다.**
        중력 대비 자세는 점군이 직접 말해주고, 이 구분을 틀리면
        수직도/수평도 중 아예 엉뚱한 식이 적용되기 때문이다.
      · 면 ↔ 선형 : **기하가 이긴다.** (평면적합 vs 축적합 선택 문제)
      · 동바리 vs 기둥 vs 철근, 벽 vs 거푸집 vs 조적 :
        **의미가 이긴다.** 기하는 이 셋을 구분할 수단이 없고,
        허용오차(KCS)만 달라지므로 라벨을 신뢰한다.
      · 기하 신뢰도가 낮으면(min_geom_conf 미만) 의미를 유지하되
        low_confidence 플래그를 세워 보고한다.

    Returns
    -------
    dict — final_class, source, agreed, confidence, note
    """
    sem = semantic_class
    shape = evidence.get("shape", "unknown")
    conf = float(evidence.get("confidence", 0.0))

    sem_kind = MEASURE_KIND.get(sem)
    geom_kind = {"plane_vertical": "plane_vertical",
                 "plane_horizontal": "plane_horizontal",
                 "linear_vertical": "axis_vertical"}.get(shape)

    out = {"semantic_class": sem, "geom_shape": shape,
           "geom_confidence": round(conf, 3),
           "final_class": sem, "source": "semantic",
           "agreed": sem_kind == geom_kind, "note": None}

    if geom_kind is None or conf < min_geom_conf:
        # 기하가 판단을 못 함 → 의미 유지
        if sem_kind is None:
            out["note"] = "의미·기하 모두 검측 대상 아님"
        elif geom_kind is None:
            out["note"] = f"기하 형상 불명({shape}) — 의미 라벨 유지, 신뢰도 낮음"
            out["low_confidence"] = True
        else:
            out["note"] = f"기하 신뢰도 부족({conf:.2f}) — 의미 라벨 유지"
            out["low_confidence"] = True
        return out

    if sem_kind == geom_kind:
        out["note"] = "의미·기하 일치"
        return out

    # 불일치 → 기하가 이기는 축인지 판단
    if sem_kind is None:
        # 의미는 배경/잡동사니라 했으나 기하는 뚜렷한 면/선형
        out["note"] = f"의미={sem}(비검측)이나 기하={shape} — 검토 필요, 검측 제외 유지"
        out["low_confidence"] = True
        return out

    # 세부 클래스는 유지하되 검측 방식만 기하에 맞춘다.
    # (예: 의미가 formwork_wall 인데 기하가 수평면 → floor 계열로 교정)
    fallback = {"plane_vertical": "wall", "plane_horizontal": "floor",
                "axis_vertical": "shoring"}[geom_kind]
    same_family = [c for c in _family_of(geom_kind) if c == sem]
    out["final_class"] = sem if same_family else fallback
    out["source"] = "geometric"
    out["note"] = (f"의미={sem}({sem_kind}) vs 기하={shape}({geom_kind}) 불일치 "
                   f"→ 기하 채택, 클래스={out['final_class']}")
    return out


def _family_of(kind):
    if kind == "plane_vertical":   return PLANE_VERTICAL_CLASSES
    if kind == "plane_horizontal": return PLANE_HORIZONTAL_CLASSES
    if kind == "axis_vertical":    return LINEAR_VERTICAL_CLASSES
    return set()


# =====================================================================
# 7. 영역별 불확실도 — "측정불가" 게이트
# =====================================================================
def region_uncertainty(points_3d, camera_params, normal=None,
                       sigma_u_px=0.2, target_sigma_mm=2.0):
    """
    영역의 측정 불확실도를 산출한다 (PDF 5.2 σ_Z 식 확장).

      σ_Z = σ_u · Z² / (f · b)                         … 깊이 방향
      σ_n = σ_Z / |cos φ|,  φ = 시선과 면 법선의 사잇각  … 법선 방향

    같은 사진 안에서도 정면인 벽은 φ≈0 이라 σ_n≈σ_Z 이지만, 비스듬히
    들어온 바닥은 φ가 커져 σ_n 이 크게 증폭된다. 평활도(±2mm)는 법선
    방향 오차가 곧 측정치이므로 이 증폭을 반영해야 판정이 정직해진다.

    Returns
    -------
    dict — z_mean_m, sigma_z_mm, incidence_deg, sigma_normal_mm,
           flatness_measurable(bool)
    """
    pts = np.asarray(points_3d, dtype=float)
    f = float(camera_params.get("f_px", 2318.8))
    b = float(camera_params.get("b_m", 0.150))
    Z = pts[:, 2]
    z_mean = float(np.mean(Z))
    sigma_z_mm = float(sigma_u_px * z_mean ** 2 / (f * b) * 1000.0)

    out = {"z_mean_m": round(z_mean, 4),
           "z_range_m": [round(float(Z.min()), 4), round(float(Z.max()), 4)],
           "sigma_z_mm": round(sigma_z_mm, 3),
           "sigma_u_px": sigma_u_px}

    if normal is not None:
        n = _EQ3.normalize(normal)
        # 영역 중심을 향한 시선 방향
        los = _EQ3.normalize(pts.mean(axis=0))
        cos_phi = abs(float(np.dot(los, n)))
        cos_phi = max(cos_phi, 1e-3)          # 완전 스침 방지
        out["incidence_deg"] = round(float(np.degrees(np.arccos(
            min(cos_phi, 1.0)))), 2)
        out["sigma_normal_mm"] = round(sigma_z_mm / cos_phi, 3)
    else:
        out["incidence_deg"] = None
        out["sigma_normal_mm"] = round(sigma_z_mm, 3)

    out["flatness_measurable"] = bool(out["sigma_normal_mm"] <= target_sigma_mm)
    out["target_sigma_mm"] = target_sigma_mm
    return out


# ============ 자체 검증 ============
if __name__ == "__main__":
    print("[식 ⑤] 영역 할당 검증")
    rng = np.random.default_rng(7)
    g = np.array([0.0, 1.0, 0.0])          # 장비 정립 → 중력 +Y

    # 1) 기하 증거: 수직벽 / 수평바닥 / 동바리
    wall = np.column_stack([rng.uniform(-.5, .5, 400), rng.uniform(-.4, .4, 400),
                            np.full(400, 1.2)]) + rng.normal(0, 5e-4, (400, 3))
    # 바닥: 장비보다 0.8m 아래에 있는 수평면을 비스듬히(grazing) 내려다봄
    #       → 법선은 중력과 평행(수평면), 대신 입사각이 커서 σ_n 이 증폭된다
    floor = np.column_stack([rng.uniform(-.5, .5, 400), np.full(400, 0.8),
                             rng.uniform(0.6, 2.5, 400)]) \
        + rng.normal(0, 5e-4, (400, 3))
    t = rng.uniform(-1.0, 1.0, 200)
    ph = rng.uniform(-1, 1, 200)
    e1, e2 = _EQ2.plane_tangent_basis(g)
    post = (np.outer(t, g) + 0.0243 * (np.cos(ph)[:, None] * e1
                                       + np.sin(ph)[:, None] * e2)
            + np.array([0.2, 0, 1.0])) + rng.normal(0, 8e-4, (200, 3))

    for name, P, expect in [("벽", wall, "plane_vertical"),
                            ("바닥", floor, "plane_horizontal"),
                            ("동바리", post, "linear_vertical")]:
        ev = geometric_evidence(P, g)
        mark = "PASS" if ev["shape"] == expect else "FAIL"
        print(f"  [1] {name:4s} 기하판별 → {ev['shape']:17s} "
              f"(기대 {expect:17s}) conf {ev['confidence']:.2f}  {mark}")

    # 2) 융합: 세그멘테이션이 바닥을 벽이라 오분류한 경우
    ev_floor = geometric_evidence(floor, g)
    fu = fuse_label("wall", ev_floor)
    print(f"  [2] 오분류 교정  의미=wall + 기하={ev_floor['shape']} "
          f"→ 최종={fu['final_class']} ({fu['source']})")
    print(f"      {fu['note']}")

    # 3) 융합: 기하가 구분 못 하는 동바리 vs 기둥 → 의미 유지
    ev_post = geometric_evidence(post, g)
    fu2 = fuse_label("shoring", ev_post)
    print(f"  [3] 의미 우선   의미=shoring + 기하={ev_post['shape']} "
          f"→ 최종={fu2['final_class']} ({fu2['source']}) 일치={fu2['agreed']}")

    # 4) 깊이 불연속: 벽(1.2m) → 바닥(2.0m) 로 넘어가는 선
    tbl = {"uv": np.zeros((40, 2)), "seq": np.arange(40),
           "lid": np.array(["V0"] * 40, dtype=object),
           "xyz": np.column_stack([np.zeros(40), np.zeros(40),
                                   np.r_[np.full(20, 1.2), np.full(20, 2.0)]])}
    bad = mark_depth_discontinuity(tbl)
    print(f"  [4] 깊이 불연속  제거점 {bad.sum()}개 "
          f"(위치 {np.where(bad)[0].tolist()}) "
          f"{'PASS' if bad.sum() == 2 and bad[19] and bad[20] else 'FAIL'}")

    # 5) 불확실도: 정면 벽 vs 경사 바닥
    cp = {"f_px": 2318.8, "b_m": 0.150}
    n_w = geometric_evidence(wall, g)["normal"]
    n_f = geometric_evidence(floor, g)["normal"]
    uw = region_uncertainty(wall, cp, n_w)
    uf = region_uncertainty(floor, cp, n_f)
    print(f"  [5] 불확실도  벽   Z={uw['z_mean_m']}m σ_Z={uw['sigma_z_mm']}mm "
          f"입사각={uw['incidence_deg']}° σ_n={uw['sigma_normal_mm']}mm "
          f"평활도측정 {'가능' if uw['flatness_measurable'] else '불가'}")
    print(f"              바닥 Z={uf['z_mean_m']}m σ_Z={uf['sigma_z_mm']}mm "
          f"입사각={uf['incidence_deg']}° σ_n={uf['sigma_normal_mm']}mm "
          f"평활도측정 {'가능' if uf['flatness_measurable'] else '불가'}")

    # 6) 마스크 침식
    m = np.zeros((60, 60), bool); m[20:40, 20:40] = True
    print(f"  [6] 마스크 침식  {m.sum()} → {erode_mask(m, 3).sum()} px "
          f"(경계 3px 제거, scipy={HAS_SCIPY})")
