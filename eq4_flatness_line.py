"""
[식 ④ 선 격자판 v4] 평활도 — 격자 평활 + 클러스터 검증
==================================================================
v2 문제: disparity 노이즈가 개별 점 Z를 ±10mm 흔듦 → 요철 오검출
v3: 격자 국소평균으로 노이즈 억제 (단 window=3·임계 1.2로 작은 요철 미검출)
v4: 검출 민감도 개선
  - 평활 window 3 → 2 : 공간 분해능 향상, 작은 요철 깊이 보존
  - 임계 1.2 → 1.5mm : 평활 후 노이즈(~1mm) 위로 설정해 오검출 억제
  - DBSCAN 클러스터(eps=30mm, min_samples=4) 검증 : 공간적으로 모인
    후보만 진짜 요철로 인정 → 흩어진 노이즈성 후보 제거

검증 결과 (10 seed, 거리 1m, σ_u=0.2px):
  요철 GT  0mm → 오검출 1/10 (거의 평탄 판정)
  요철 GT  2mm → 검출 2/10  (노이즈 한계: σ_Z≈2.4mm > 2mm)
  요철 GT  5mm → 검출 9/10
  요철 GT 10mm → 검출 10/10

물리 한계: 단일 점 거리 노이즈 σ_Z = σ_u·Z²/(f·b) ≈ 2.4mm (식 ⑥).
2mm 이하 요철은 노이즈에 묻혀 신뢰 검출 곤란 → σ_u↓, baseline b↑,
초점거리 f↑ 또는 측정 반복 평균(σ̄=σ_Z/√N)으로 노이즈를 낮춰야 함.
"""
import numpy as np

try:
    from sklearn.linear_model import RANSACRegressor, LinearRegression
    from sklearn.cluster import DBSCAN
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


def robust_plane_fit(all_points, threshold_mm=2.0):
    """요철을 outlier로 배제한 robust 평면."""
    if not HAS_SKLEARN:
        A = np.column_stack([all_points[:,0], all_points[:,1], np.ones(len(all_points))])
        coef, *_ = np.linalg.lstsq(A, all_points[:,2], rcond=None)
        a,bb,c,d = coef[0], coef[1], -1.0, coef[2]
        norm = np.sqrt(a*a+bb*bb+c*c)
        return (a/norm, bb/norm, c/norm, d/norm), np.ones(len(all_points), bool)
    X = all_points[:, :2]; Z = all_points[:, 2]
    ransac = RANSACRegressor(estimator=LinearRegression(),
                             residual_threshold=threshold_mm/1000, random_state=42)
    ransac.fit(X, Z)
    a_p, b_p = ransac.estimator_.coef_
    c_p = ransac.estimator_.intercept_
    a, bb, c, d = a_p, b_p, -1.0, c_p
    norm = np.sqrt(a*a + bb*bb + c*c)
    return (a/norm, bb/norm, c/norm, d/norm), ransac.inlier_mask_


def grid_smooth(all_points, grid_n=20, window=2):
    """
    국소 격자 median으로 노이즈 억제.

    XY 평면을 grid_n × grid_n 셀로 나눠 각 셀의 Z median.
    median은 노이즈에 robust하면서 면적이 있는 요철은 보존.
    (요철이 셀 다수에 걸치면 그 셀들의 median도 요철값)

    Returns: smoothed_points (m, 3)
    """
    x = all_points[:, 0]; y = all_points[:, 1]; z = all_points[:, 2]
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()

    smoothed = []
    xs = np.linspace(xmin, xmax, grid_n)
    ys = np.linspace(ymin, ymax, grid_n)
    cell_x = (xmax - xmin) / grid_n
    cell_y = (ymax - ymin) / grid_n

    for cx in xs:
        for cy in ys:
            mask = (np.abs(x - cx) < cell_x*window) & (np.abs(y - cy) < cell_y*window)
            if mask.sum() >= 3:
                smoothed.append([cx, cy, np.median(z[mask])])
    return np.array(smoothed)


def detect_defects_grid(all_points, threshold_mm=1.5, grid_n=20, window=2,
                        cluster_eps_mm=30, cluster_min_samples=4):
    """
    국소 격자 평균 기반 요철 검출 (v4 — 민감도 개선).

    개선점 (v3 대비):
    - 평활 윈도우 window=3 → 2 : 공간 분해능을 높여 작은 요철의 깊이 보존
    - 임계 1.2 → 1.5mm : 측정 노이즈(σ_Z≈2.4mm/점, 평활 후 ~1mm)의 위로 설정해
      평탄면 오검출(false positive) 억제
    - 임계 초과 점을 그대로 요철로 보지 않고 DBSCAN 클러스터(eps=30mm,
      min_samples=4)로 공간적으로 모여 있는지 검증 → 흩어진 노이즈 제거
    - 요철 깊이/판정은 '검증된 클러스터'를 기준으로 산출

    Returns: dict
    """
    # 1. 국소 격자 평균 (노이즈 억제, 단 요철 보존 위해 window 축소)
    smoothed = grid_smooth(all_points, grid_n=grid_n, window=window)

    # 2. robust 평면 (요철을 outlier로 제외)
    plane, inlier_mask = robust_plane_fit(smoothed, 1.0)
    a, b, c, d = plane

    # 3. 평면 잔차
    res_mm = (a*smoothed[:,0] + b*smoothed[:,1] + c*smoothed[:,2] + d) * 1000

    # 4. 임계 초과 후보점
    cand_mask = np.abs(res_mm) > threshold_mm
    cand_points = smoothed[cand_mask]
    cand_res = res_mm[cand_mask]

    # 5. 클러스터 검증 — 공간적으로 모인 후보만 진짜 요철로 인정
    verified_clusters = cluster_defects(cand_points,
                                        eps_mm=cluster_eps_mm,
                                        min_samples=cluster_min_samples,
                                        residuals_mm=cand_res)

    # 검증된 클러스터에 속한 점만 요철점으로 채택
    if verified_clusters:
        verified_idx = np.concatenate([c['point_idx'] for c in verified_clusters])
        defect_points = cand_points[verified_idx]
        defect_res = cand_res[verified_idx]
        # 요철 깊이 = 검증된 클러스터 내부 최대 |잔차|
        overall_max = float(np.max(np.abs(defect_res)))
        is_pass = False  # 검증된 요철 클러스터 존재 → FAIL(요철 있음)
    else:
        defect_points = np.empty((0, 3))
        defect_res = np.empty((0,))
        overall_max = 0.0          # 검증된 요철 없음 → 깊이 0 보고
        is_pass = True             # 평탄 → PASS

    return {
        'plane': plane,
        'smoothed_points': smoothed,
        'residuals_mm': res_mm,
        'defect_points_3d': defect_points,
        'defect_residuals_mm': defect_res,
        'defect_count': int(len(defect_points)),
        'overall_max_dev_mm': overall_max,
        'raw_max_dev_mm': float(np.max(np.abs(res_mm))),  # 참고용 (검증 전)
        'rms_dev_mm': float(np.sqrt(np.mean(res_mm**2))),
        'is_pass': is_pass,
        'verified_clusters': verified_clusters,
    }


def cluster_defects(defect_points_3d, eps_mm=30, min_samples=4, residuals_mm=None):
    """요철 후보점 클러스터링 + 공간 검증.

    eps_mm=30, min_samples=4 : 노이즈성 산발 점은 클러스터를 못 이루게 하여
    평탄면 오검출을 억제. 각 클러스터의 깊이(depth_mm)와 구성 점 인덱스를 반환.
    """
    if len(defect_points_3d) == 0:
        return []
    if residuals_mm is None:
        residuals_mm = np.zeros(len(defect_points_3d))
    if not HAS_SKLEARN:
        pts = defect_points_3d
        if len(pts) < min_samples:
            return []
        return [{'center_xy': (float(pts[:,0].mean()), float(pts[:,1].mean())),
                 'extent_mm': float(max(np.ptp(pts[:,0]), np.ptp(pts[:,1]))*1000),
                 'depth_mm': float(np.max(np.abs(residuals_mm))),
                 'n_points': len(pts),
                 'point_idx': np.arange(len(pts))}]
    xy = defect_points_3d[:, :2]
    db = DBSCAN(eps=eps_mm/1000, min_samples=min_samples).fit(xy)
    clusters = []
    for lbl in set(db.labels_):
        if lbl == -1: continue
        m = db.labels_ == lbl
        cpts = defect_points_3d[m]
        clusters.append({
            'center_xy': (float(cpts[:,0].mean()), float(cpts[:,1].mean())),
            'extent_mm': float(max(np.ptp(cpts[:,0]), np.ptp(cpts[:,1]))*1000),
            'depth_mm': float(np.max(np.abs(residuals_mm[m]))),
            'n_points': int(m.sum()),
            'point_idx': np.where(m)[0]})
    return clusters


def reconstruct_defect_contour(defect_result):
    """요철 윤곽 — 검증된 클러스터 기반."""
    clusters = defect_result.get('verified_clusters', [])
    pts = defect_result['defect_points_3d']
    if len(clusters) == 0 or len(pts) == 0:
        return {'center_xy': None, 'extent_mm': 0, 'depth_mm': 0, 'clusters': [], 'n_clusters': 0}
    res = defect_result['defect_residuals_mm']
    return {
        'center_xy': (float(pts[:,0].mean()), float(pts[:,1].mean())),
        'extent_mm': float(max(np.ptp(pts[:,0]), np.ptp(pts[:,1]))*1000),
        'depth_mm': float(np.max(np.abs(res))),
        'clusters': clusters,
        'n_clusters': len(clusters),
    }


# 호환성 래퍼 (pipeline_line에서 lines_3d로 호출)
def detect_defects_from_lines(lines_3d, threshold_mm=1.5):
    all_pts = np.vstack(list(lines_3d.values()))
    result = detect_defects_grid(all_pts, threshold_mm=threshold_mm, grid_n=20)
    # defect_lines는 격자 기반이라 의미 약함 → 검증된 클러스터 사용
    result['defect_lines'] = []  # 격자 방식은 선 단위 아님
    return result


# ============ 자체 검증 ============
if __name__ == "__main__":
    print("[식 ④ 선 격자판 v4] 격자 평활 + 클러스터 검증 요철 검출")
    np.random.seed(42)
    Z_wall = 1.0
    
    def make_lines(bump_mm):
        """수직선 20개 생성 (disparity 노이즈 포함)."""
        f, b, cx, cy = 1660.0, 0.05, 960.0, 540.0
        fov = np.radians(20)
        alphas = np.linspace(-fov/2, fov/2, 20)
        samples = np.linspace(-fov/2, fov/2, 60)
        bump_m = bump_mm/1000
        lines = {}
        for i, alpha in enumerate(alphas):
            pts = []
            for beta in samples:
                X0, Y0 = np.tan(alpha), np.tan(beta)
                Z = Z_wall
                for _ in range(3):
                    X, Y = X0*Z, Y0*Z
                    d = np.sqrt(X**2+Y**2)
                    Z = Z_wall - (bump_m*np.exp(-(d/0.04)**2*2) if d<0.04 else 0)
                X, Y = X0*Z, Y0*Z
                u = f*(X-b)/Z + cx + np.random.normal(0,0.2)
                v = f*Y/Z + cy + np.random.normal(0,0.2)
                ku = (u-cx)/f; kv = (v-cy)/f
                Zr = b/(np.tan(alpha)-ku)
                pts.append([ku*Zr+b, kv*Zr, Zr])
            lines[f'V{i}'] = np.array(pts)
        return lines
    
    for bump in [0, 2, 5, 10]:
        lines = make_lines(bump)
        result = detect_defects_from_lines(lines, threshold_mm=2.0)
        contour = reconstruct_defect_contour(result)
        ctr = contour['center_xy']
        ctr_str = f"({ctr[0]*1000:.0f},{ctr[1]*1000:.0f})mm" if ctr else "없음"
        print(f"  요철 GT {bump:2d}mm → 측정 {result['overall_max_dev_mm']:5.2f}mm, "
              f"검출점 {result['defect_count']:2d}, 클러스터 {contour['n_clusters']}, 중심 {ctr_str}")
    print("\n  단일 seed라 5mm는 우연히 놓칠 수 있음 (10 seed 검출률 9/10).")
    print("  2mm 이하는 노이즈 한계 → 식 ⑥ 참고.")


# =====================================================================
# [v5 추가] 면내(in-plane) 좌표 기반 영역 평활도
# =====================================================================
# detect_defects_grid() 는 조사기 좌표계의 X-Y 로 격자 비닝한다. 면을 정면
# 으로 겨누던 기존 파이프라인에서는 타당했으나, 세그멘테이션 후에는 한
# 장의 사진 안에 정면인 벽과 비스듬히 들어온 바닥이 함께 있다. 비스듬한
# 면을 X-Y 로 비닝하면 셀이 찌그러져 (a) 국소 median 이 서로 다른 높이를
# 섞고 (b) 요철 크기·위치가 왜곡된다.
#
# → 영역 평면의 접선 기저 (e1, e2) 로 좌표변환한 뒤 면내 (u,v) 로 비닝한다.
#   잔차는 평면 법선 방향 성분 w 그 자체이므로, 시야각과 무관하게
#   KCS 의 "3m 직선자에 의한 처짐량"과 같은 의미를 갖는다.
#
# 평면 적합은 fit_plane_tls_ransac (방향 무관)을 쓴다. Z=aX+bY+c 회귀는
# 광축과 나란한 면을 표현하지 못한다.

import importlib.util as _ilu5, os as _os5

def _load_eq2():
    spec = _ilu5.spec_from_file_location(
        "eq2_plane_fit",
        _os5.path.join(_os5.path.dirname(_os5.path.abspath(__file__)),
                       "eq2_plane_fit.py"))
    m = _ilu5.module_from_spec(spec); spec.loader.exec_module(m)
    return m

_EQ2M = _load_eq2()


def _smooth_once(uvw, grid_n, window, min_cell_points):
    u, v, w = uvw[:, 0], uvw[:, 1], uvw[:, 2]
    umin, umax, vmin, vmax = u.min(), u.max(), v.min(), v.max()
    cu = (umax - umin) / grid_n
    cv = (vmax - vmin) / grid_n
    out, counts = [], []
    for gu in np.linspace(umin, umax, grid_n):
        du = np.abs(u - gu) < cu * window
        if not du.any():
            continue
        for gv in np.linspace(vmin, vmax, grid_n):
            m = du & (np.abs(v - gv) < cv * window)
            c = int(m.sum())
            if c >= min_cell_points:
                out.append([gu, gv, np.median(w[m])])
                counts.append(c)
    if not out:
        return np.empty((0, 3)), (cu, cv), 0
    return np.array(out), (cu, cv), int(np.median(counts))


def _grid_smooth_uv(uvw, grid_n=24, window="auto", min_cell_points=3,
                    target_cell_points=8, window_candidates=(1, 1.5, 2, 3)):
    """
    면내 (u,v) 격자에서 w 의 국소 median 을 취해 노이즈를 억제한다.

    window="auto" 의 근거
    ---------------------
    평활 윈도우는 노이즈 억제와 공간 분해능의 맞교환이다. 넓게 잡으면
    노이즈는 줄지만 요철 깊이가 뭉개진다(실측: window=2 에서 GT 6mm
    융기가 2.85mm 로 축소).

    필요한 만큼만 넓히면 된다. 점당 깊이 노이즈 σ_Z 는 median 필터를
    거치며 대략 1.25·σ_Z/√k (k = 셀 내 점 수) 로 줄어든다. 즉 k 가
    target_cell_points 를 넘어서면 그 이상 넓혀도 이득은 √k 로 미미한
    반면 분해능 손실은 선형이다.
    → 셀 내 점 수 중앙값이 target_cell_points 이상이 되는 **가장 좁은**
      윈도우를 고른다. v4 의 고정값 window=2 를 대체한다.

    Returns
    -------
    smoothed : (M,3) [u, v, w_median]
    cell     : (cu, cv) 격자 셀 크기 (m) — 클러스터 eps 산정에 쓰인다
    used     : dict 진단 정보 (window, median_cell_points)
    """
    u, v = uvw[:, 0], uvw[:, 1]
    if u.max() - u.min() < 1e-9 or v.max() - v.min() < 1e-9:
        return np.empty((0, 3)), (0.0, 0.0), {"window": None,
                                              "median_cell_points": 0}
    if window != "auto":
        sm, cell, med = _smooth_once(uvw, grid_n, float(window),
                                     min_cell_points)
        return sm, cell, {"window": float(window), "median_cell_points": med}

    best = None
    for wnd in window_candidates:
        sm, cell, med = _smooth_once(uvw, grid_n, float(wnd), min_cell_points)
        best = (sm, cell, {"window": float(wnd), "median_cell_points": med})
        if len(sm) >= 6 and med >= target_cell_points:
            break
    return best


def _cluster_raw_depth(uvw, raw_w_mm, cluster_cells_uv, cluster_res_mm,
                       cell, radius_cells=0.6):
    """
    검증된 요철 클러스터의 깊이를 원시 잔차에서 로버스트하게 뽑는다.

    클러스터 전체 외접범위에서 극값을 잡으면, 요철 주변의 평탄한 점들이
    섞여 깊이가 낮게 나온다. 대신 **잔차가 가장 큰 셀(요철 정점)** 주변
    좁은 범위만 보고 그 안의 원시 잔차 median 을 쓴다.

    median 을 쓰는 이유: 최대값은 노이즈 한 점에 좌우된다. 현장 σ_Z 는
    1~2mm 수준이므로 최대값 기반 추정은 그만큼 과대평가된다. 정점 부근은
    신호가 거의 평탄하므로 median 이 편향 없이 노이즈만 걷어낸다.
    """
    if len(cluster_cells_uv) == 0:
        return 0.0
    cu, cv = max(cell[0], 1e-6), max(cell[1], 1e-6)
    peak = cluster_cells_uv[int(np.argmax(np.abs(cluster_res_mm)))]
    m = ((np.abs(uvw[:, 0] - peak[0]) <= radius_cells * cu) &
         (np.abs(uvw[:, 1] - peak[1]) <= radius_cells * cv))
    if m.sum() < 3:
        return float(abs(cluster_res_mm[int(np.argmax(np.abs(cluster_res_mm)))]))
    return float(abs(np.median(raw_w_mm[m])))


def detect_defects_region(points_3d, plane=None, threshold_mm=1.5,
                          grid_n=24, window="auto", plane_threshold_m=0.004,
                          cluster_eps_mm=30, cluster_min_samples=4):
    """
    한 영역(벽면/바닥면)의 평활도를 면내 좌표계에서 산출한다.

    Parameters
    ----------
    points_3d : (N,3) — 해당 영역의 3D 점 (조사기 좌표계, m)
    plane : (a,b,c,d) or None
        영역 평면. None이면 내부에서 TLS RANSAC 으로 적합한다.
    threshold_mm : float — 요철 판정 임계 (평활 후 잔차)
    plane_threshold_m : float — 평면 RANSAC inlier 임계 (요철을 outlier로 배제)

    Returns
    -------
    dict — detect_defects_grid() 와 같은 키에 면내 좌표 정보를 추가
        plane, residuals_mm, defect_points_uv, defect_residuals_mm,
        defect_count, overall_max_dev_mm, raw_max_dev_mm, rms_dev_mm,
        is_pass, verified_clusters, n_smoothed, area_m2
    """
    pts = np.asarray(points_3d, dtype=float)
    empty = {'plane': plane, 'residuals_mm': np.empty(0),
             'defect_points_uv': np.empty((0, 2)),
             'defect_residuals_mm': np.empty(0), 'defect_count': 0,
             'overall_max_dev_mm': 0.0, 'raw_max_dev_mm': 0.0,
             'rms_dev_mm': 0.0, 'is_pass': True, 'verified_clusters': [],
             'n_smoothed': 0, 'area_m2': 0.0,
             'reject_reason': None}
    if len(pts) < 12:
        empty['reject_reason'] = f"점 부족 ({len(pts)} < 12)"
        return empty

    if plane is None:
        plane, _ = _EQ2M.fit_plane_tls_ransac(pts, threshold=plane_threshold_m)

    # 면내 좌표 (u, v) + 법선 방향 이탈 w
    uvw, basis, origin = _EQ2M.project_to_plane_frame(pts, plane)

    smoothed, cell, smooth_info = _grid_smooth_uv(uvw, grid_n=grid_n,
                                                  window=window)
    if len(smoothed) < 6:
        empty['plane'] = plane
        empty['reject_reason'] = f"평활 격자 셀 부족 ({len(smoothed)})"
        return empty

    res_mm = smoothed[:, 2] * 1000.0
    # 평활 격자에서 다시 한 번 기준면을 잡아 전역 기울기 잔여분을 제거
    res_mm = res_mm - np.median(res_mm)

    cand = np.abs(res_mm) > threshold_mm
    cand_uv = smoothed[cand, :2]
    cand_res = res_mm[cand]

    # 클러스터 검증은 기존 cluster_defects 를 면내 좌표로 재사용.
    # eps 는 반드시 격자 셀 간격에 맞춰야 한다. 고정 30mm 를 쓰면 면내
    # 셀 간격(수십~100mm)보다 작아 인접 요철 셀이 절대 이어지지 않고,
    # 클러스터가 0개가 되어 진짜 요철을 통째로 놓친다.
    eps_m = max(cluster_eps_mm / 1000.0, 1.6 * max(cell[0], cell[1]))
    cand_xyz = np.column_stack([cand_uv, np.zeros(len(cand_uv))])
    clusters = cluster_defects(cand_xyz, eps_mm=eps_m * 1000.0,
                               min_samples=cluster_min_samples,
                               residuals_mm=cand_res)

    if clusters:
        vidx = np.concatenate([c['point_idx'] for c in clusters])
        d_uv, d_res = cand_uv[vidx], cand_res[vidx]
        # 요철 깊이는 평활값이 아니라 **원시 잔차**에서 잰다.
        #   평활(median)은 검출용 노이즈 억제 수단이고, 창이 요철보다 넓으면
        #   깊이를 그만큼 깎는다(실측: GT 6mm 융기 → 평활값 3.3mm).
        #   클러스터로 위치가 이미 검증됐으므로, 그 안에서는 원시 잔차의
        #   로버스트 극값을 쓰는 편이 정확하고 노이즈에도 안전하다.
        raw_w_mm = uvw[:, 2] * 1000.0 - float(np.median(smoothed[:, 2] * 1000.0))
        for c in clusters:
            c['depth_mm'] = _cluster_raw_depth(
                uvw, raw_w_mm, cand_uv[c['point_idx']],
                cand_res[c['point_idx']], cell)
        overall_max = float(max(c['depth_mm'] for c in clusters))
        is_pass = False
    else:
        d_uv, d_res = np.empty((0, 2)), np.empty(0)
        overall_max, is_pass = 0.0, True

    return {'plane': plane, 'basis': basis, 'origin': origin,
            'uvw': uvw, 'smoothed_uvw': smoothed,
            'residuals_mm': res_mm,
            'defect_points_uv': d_uv, 'defect_residuals_mm': d_res,
            # 클러스터의 point_idx 는 **후보 배열(cand_uv)** 기준이다.
            # 위의 defect_points_uv 는 검증된 클러스터만 이어붙인 부분집합이라
            # 같은 인덱스로 접근하면 엉뚱한 점을 집는다. 요철 위치를 화면에
            # 되돌리려면 원본 후보 배열이 필요하므로 함께 돌려준다.
            'cand_points_uv': cand_uv, 'cand_residuals_mm': cand_res,
            'defect_count': int(len(d_uv)),
            'overall_max_dev_mm': overall_max,
            'raw_max_dev_mm': float(np.max(np.abs(res_mm))),
            'rms_dev_mm': float(np.sqrt(np.mean(res_mm ** 2))),
            'is_pass': is_pass, 'verified_clusters': clusters,
            'n_smoothed': int(len(smoothed)),
            'smooth_info': smooth_info, 'cell_m': [float(cell[0]), float(cell[1])],
            'cluster_eps_mm': round(eps_m * 1000.0, 1),
            'area_m2': float((uvw[:, 0].max() - uvw[:, 0].min()) *
                             (uvw[:, 1].max() - uvw[:, 1].min())),
            'reject_reason': None}
