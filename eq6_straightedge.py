"""
[식 ⑥] 평활도 — KCS 직선자 판정
==================================================================
【왜 별도 식이 필요한가】
  eq4 는 적합 평면으로부터의 잔차로 요철을 찾는다. 요철의 위치·깊이를
  찾는 데는 맞지만, KCS 가 규정하는 판정값은 그것이 아니다.

    KCS 14 20 10 : "3m 직선자에 의한 처짐량"  3m당 7mm / 3m당 10mm
    KCS 41 46 00 : 미장 평탄도               1m당 10mm

  즉 현장 검사원은 **자를 표면에 얹고 자와 표면 사이의 틈**을 잰다.
  전역 평면 잔차와는 다음 두 가지가 다르다.

  1. 기준선이 전역 평면이 아니라 **국소 구간(3m 또는 1m)** 이다.
     넓은 면이 완만하게 휘어 있으면 전역 잔차는 크지만 3m 자에는
     걸리지 않는다. 반대로 좁고 급한 굴곡은 전역 잔차로는 작아 보여도
     자를 얹으면 크게 뜬다.
  2. 자는 표면에 **얹히는** 것이지 평균을 지나지 않는다. 볼록한 곳이
     있으면 자는 그 위에 걸터앉고 양옆에 틈이 생긴다.

  → 구간 [s0, s0+L] 안에서 자가 닿는 자리는 프로파일의 **상부 볼록껍질**
    이고, 처짐량은 그 껍질과 표면 사이 최대 간격이다. 이 정의는 볼록한
    돌출과 오목한 함몰을 모두 물리적으로 맞게 다룬다.

【부호 규약】
  면내 좌표계의 w 는 평면 법선 방향 성분이다. 법선의 부호는 적합
  과정에서 임의로 정해지므로, 여기서 **센서 쪽을 향하도록** 뒤집는다.
  그러면 w > 0 = 튀어나온 곳이 되어 "자가 그 위에 얹힌다"가 성립한다.

【좌표계】 조사기 좌표계 (X 우, Y 하, Z 전방) — eq1 규약
==================================================================
"""
import numpy as np
import importlib.util as _ilu, os as _os


def _load(name):
    spec = _ilu.spec_from_file_location(
        name, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), f"{name}.py"))
    m = _ilu.module_from_spec(spec); spec.loader.exec_module(m)
    return m


_EQ2 = _load("eq2_plane_fit")

# KCS 평활도 기준 (PDF 1.2 표).
#   length_m : 직선자 길이, tol_mm : 그 길이에서의 허용 처짐량
KCS_FLATNESS_SPEC = {
    # 노출 콘크리트(제물치장) 벽체 — KCS 14 20 10
    "wall":            [(3.0, 7.0)],
    "formwork_wall":   [(3.0, 7.0)],
    "formwork_column": [(3.0, 7.0)],
    # 미장 마감면 — KCS 41 46 00 (마감 두께 7mm 이상)
    "plaster_wall":    [(1.0, 10.0)],
    "masonry":         [(1.0, 10.0)],
    # 슬래브 바닥 — KCS 14 20 10 (마감 조건별 1m당 10 / 3m당 10 / 3m당 7)
    "floor":           [(1.0, 10.0), (3.0, 10.0)],
    "slab":            [(1.0, 10.0), (3.0, 10.0)],
    # 천장틀 — KCS 41 52 00 (3m당 ±3mm)
    "ceiling":         [(3.0, 3.0)],
}


# =====================================================================
# 면내 프로파일
# =====================================================================
def orient_normal_to_sensor(plane, points_3d):
    """
    평면 법선이 센서(원점) 쪽을 향하도록 부호를 맞춘다.

    TLS 적합은 법선 부호를 정해주지 않는다. 부호가 뒤집혀 있으면 돌출과
    함몰이 바뀌어, 자가 얹히는 자리가 반대로 계산된다.
    """
    a, b, c, d = plane
    n = np.array([a, b, c], dtype=float)
    centroid = np.asarray(points_3d, dtype=float).mean(axis=0)
    # 센서는 원점에 있고 면은 +Z 앞쪽에 있으므로, 센서를 향하는 법선은
    # 중심점과 반대 방향(내적 < 0)이다.
    if float(np.dot(n, centroid)) > 0:
        return (-a, -b, -c, -d)
    return (a, b, c, d)


def _profile_along(uvw, angle_rad, length_m, target_bin_points=8,
                   n_samples_per_edge=60, min_bin_points=3):
    """
    한 방향으로 자른 1D 프로파일을 만든다.

    자는 폭이 있는 강체이므로 선 하나가 아니라 주변 점을 모아 대표값을
    취한다. 3m 강철 자는 2cm 파장의 요동을 따라갈 수 없으므로, 이 평활은
    편의가 아니라 물리적으로 옳은 모형이다.

    【원형 창 이동 median 을 쓰는 이유】
      길쭉한 띠(band × bin)로 평균하면 두 가지가 동시에 틀어진다.
        · 창이 요철보다 넓으면 정점이 뭉개진다. 실측에서 자 방향 구간을
          10cm 로 잡자 σ=5cm 융기가 10mm→4.9mm 로 포화했다.
        · 창이 좁으면 창 안 점이 1~2개가 되어 median 이 노이즈를 못 줄인다.
          상부 볼록껍질은 위쪽 극값을 따라가므로, 평탄면에서도 노이즈만큼
          처짐이 잡힌다(실측 GT 0mm 에서 1.02mm).

      해상도와 평활을 분리하면 둘 다 해결된다.
        · 표본 위치는 촘촘하게 (자 하나에 n_samples_per_edge 개)
        · 대표값은 **밀도에서 역산한 반경 r 의 원형 창** median
          r = √(target / (π · 점밀도))  → 창 안 점 수가 항상 target 근처
      창이 등방적이라 어느 방향의 요철이든 같은 척도로 잡힌다.

    Returns
    -------
    s    : (M,) 자 방향 좌표 (m, 오름차순)
    w    : (M,) 그 위치의 대표 이탈 (m)
    info : dict — radius_m, sample_step_m, median_win_points, bin_noise_mm
    """
    e = np.array([np.cos(angle_rad), np.sin(angle_rad)])
    q = np.array([-np.sin(angle_rad), np.cos(angle_rad)])
    s_all = uvw[:, :2] @ e
    t_all = uvw[:, :2] @ q
    empty = (np.empty(0), np.empty(0),
             {"radius_m": 0.0, "sample_step_m": 0.0,
              "median_win_points": 0, "bin_noise_mm": None})
    N = len(s_all)
    if N < 12:
        return empty
    span = float(s_all.max() - s_all.min())
    extent_t = float(t_all.max() - t_all.min())
    if span < 1e-6 or extent_t < 1e-9:
        return empty

    # 평활 반경: 창 안 점 수가 target 이 되도록 면 밀도에서 역산
    density = N / max(span * extent_t, 1e-9)          # 점/m²
    radius = float(np.sqrt(target_bin_points / (np.pi * density)))
    radius = float(np.clip(radius, 0.005, max(span, extent_t) / 6.0))

    # 표본 위치: 자 하나에 n_samples_per_edge 개 (면이 자보다 짧으면 면 기준)
    step = max(min(length_m, span) / n_samples_per_edge, 1e-3)

    t_c = float(np.median(t_all))                      # 자를 면 중앙에 얹는다
    near_t = np.abs(t_all - t_c) <= radius
    if near_t.sum() < 8:
        near_t = np.abs(t_all - t_c) <= max(radius * 3.0, extent_t / 6.0)
    if near_t.sum() < 8:
        return empty
    s_n, t_n, w_n = s_all[near_t], t_all[near_t], uvw[near_t, 2]
    r2 = radius * radius

    ss, ww, cc, spread = [], [], [], []
    for s0 in np.arange(s_all.min(), s_all.max() + 1e-9, step):
        d2 = (s_n - s0) ** 2 + (t_n - t_c) ** 2
        m = d2 <= r2
        c = int(m.sum())
        if c < min_bin_points:
            continue
        val = float(np.median(w_n[m]))
        ss.append(float(s0)); ww.append(val); cc.append(c)
        if c >= 4:
            # 창 내부 산포 → 점당 측정 노이즈 추정 (MAD 기반, robust)
            spread.append(1.4826 * float(np.median(np.abs(w_n[m] - val))))
    if len(ss) < 3:
        return empty

    med_c = int(np.median(cc)) if cc else 0
    # 창 median 의 노이즈 ≈ 1.25·σ_점 / √k
    bin_noise_mm = (float(np.median(spread)) * 1.25 / max(np.sqrt(med_c), 1.0)
                    * 1000.0) if spread and med_c else None
    return (np.asarray(ss), np.asarray(ww),
            {"radius_m": round(radius, 4), "sample_step_m": round(step, 4),
             "median_win_points": med_c,
             "bin_noise_mm": (round(bin_noise_mm, 3)
                              if bin_noise_mm is not None else None)})


def _upper_hull_gap(s, w):
    """
    구간 전체에 자를 얹었을 때의 최대 틈.

    자는 프로파일 위에 얹히므로 닿는 자리는 상부 볼록껍질이다.
    껍질과 표면의 최대 간격이 처짐량이 된다.

    Returns
    -------
    gap : float (m)   최대 틈
    at  : float (m)   그 위치의 s 좌표
    """
    n = len(s)
    if n < 3:
        return 0.0, float(s[0]) if n else 0.0

    # 상부 볼록껍질 (모노톤 체인). 위로 볼록해야 하므로 우회전만 남긴다.
    hull = []
    for i in range(n):
        while len(hull) >= 2:
            (x1, y1), (x2, y2) = hull[-2], hull[-1]
            # (x1,y1)->(x2,y2)->(s,w) 가 좌회전(반시계)이면 (x2,y2)는 껍질 아래
            if (x2 - x1) * (w[i] - y1) - (y2 - y1) * (s[i] - x1) >= 0:
                hull.pop()
            else:
                break
        hull.append((s[i], w[i]))

    hs = np.array([h[0] for h in hull])
    hw = np.array([h[1] for h in hull])
    hull_w = np.interp(s, hs, hw)
    gaps = hull_w - w
    k = int(np.argmax(gaps))
    return float(gaps[k]), float(s[k])


def straightedge_gap(points_3d, length_m=3.0, plane=None,
                     n_directions=4, target_bin_points=8,
                     n_samples_per_edge=60, min_bin_points=3, step_ratio=0.25):
    """
    길이 length_m 직선자를 여러 방향·위치로 얹어 최대 처짐량을 구한다.

    Parameters
    ----------
    length_m : float — 직선자 길이 (KCS 3m 또는 1m)
    n_directions : int — 자를 얹는 방향 수 (0°부터 180°를 균등 분할)
    target_bin_points : int — 평활 창 안 목표 점 수 (창 반경 자동 산출)
    step_ratio : float — 자를 미는 간격 (length_m 대비)

    Returns
    -------
    dict
        max_gap_mm    : 최대 처짐량 [mm]
        span_m        : 실제로 자를 얹을 수 있었던 최대 구간 길이
        sufficient    : 면이 length_m 보다 커서 규정대로 쟀는지
        n_windows     : 평가한 자 위치 수
        at_uv         : 최대 처짐이 난 면내 좌표
        direction_deg : 그때의 자 방향
        reject_reason : 측정 불가 사유
    """
    pts = np.asarray(points_3d, dtype=float)
    out = {"length_m": float(length_m), "max_gap_mm": 0.0, "span_m": 0.0,
           "sufficient": False, "n_windows": 0, "at_uv": None,
           "direction_deg": None, "reject_reason": None}
    if len(pts) < 12:
        out["reject_reason"] = f"점 부족 ({len(pts)} < 12)"
        return out

    if plane is None:
        plane, _ = _EQ2.fit_plane_tls_ransac(pts, threshold=0.01)
    plane = orient_normal_to_sensor(plane, pts)
    uvw, _, _ = _EQ2.project_to_plane_frame(pts, plane)

    best = (0.0, None, None)
    span_best = 0.0
    n_win = 0
    noise_floors, prof_info = [], None

    for k in range(n_directions):
        ang = np.pi * k / n_directions
        s, w, pinfo = _profile_along(uvw, ang, length_m,
                                     target_bin_points=target_bin_points,
                                     n_samples_per_edge=n_samples_per_edge,
                                     min_bin_points=min_bin_points)
        if len(s) < 3:
            continue
        if prof_info is None:
            prof_info = pinfo
        if pinfo.get("bin_noise_mm"):
            noise_floors.append(pinfo["bin_noise_mm"])
        span = float(s[-1] - s[0])
        span_best = max(span_best, span)

        if span < length_m:
            # 면이 자보다 짧다 — 규정 길이로는 못 잰다. 가진 구간 전체로
            # 재되 sufficient=False 로 남겨 판정에서 구분한다.
            gap, at = _upper_hull_gap(s, w)
            n_win += 1
            if gap > best[0]:
                best = (gap, at, np.degrees(ang))
            continue

        step = max(length_m * step_ratio, pinfo["sample_step_m"])
        s0 = s[0]
        while s0 + length_m <= s[-1] + 1e-9:
            m = (s >= s0) & (s <= s0 + length_m)
            if m.sum() >= 3:
                gap, at = _upper_hull_gap(s[m], w[m])
                n_win += 1
                if gap > best[0]:
                    best = (gap, at, np.degrees(ang))
            s0 += step

    out["max_gap_mm"] = round(best[0] * 1000.0, 3)
    out["span_m"] = round(span_best, 3)
    out["profile"] = prof_info
    # 노이즈 바닥: 요철이 전혀 없어도 측정 노이즈만으로 잡히는 처짐량.
    # 상부 볼록껍질이 위쪽 극값을 따라가므로 편향은 항상 양(+)이다.
    # 구간 median 노이즈 σ_b 와 구간 수 n 에 대해 극값 기대치 ≈ σ_b·√(2 ln n).
    if noise_floors:
        n_bins = max(int(np.ceil(min(length_m, max(span_best, 1e-6))
                                 / max(prof_info["sample_step_m"], 1e-6))), 3)
        out["noise_floor_mm"] = round(
            float(np.median(noise_floors)) * float(np.sqrt(2 * np.log(n_bins))),
            3)
        out["gap_above_noise_mm"] = round(
            max(0.0, out["max_gap_mm"] - out["noise_floor_mm"]), 3)
    out["sufficient"] = bool(span_best >= length_m)
    out["n_windows"] = n_win
    if best[1] is not None:
        out["at_uv"] = [round(float(best[1]), 4), None]
        out["direction_deg"] = round(float(best[2]), 1)
    if n_win == 0:
        out["reject_reason"] = "프로파일을 만들 수 없음 (점 분포 부족)"
    return out


def gap_with_uncertainty(points_3d, length_m, plane=None,
                         fine_points=4, coarse_points=16, **kw):
    """
    자 처짐량을 **두 평활 척도**로 재어 분해능 편향까지 함께 추정한다.

    【왜 두 번 재는가】
      요철 폭이 평활 반경보다 좁으면 자 처짐량은 반드시 낮게 나온다.
      이는 알고리즘 결함이 아니라 점 밀도의 물리적 한계다. 예를 들어
      1041점/m² 면에서 σ=5cm 융기의 중심부에는 점이 8개뿐이라, 어떤
      추정량도 5cm 규모로 평활될 수밖에 없다.

      문제는 이 과소평가가 **기준초과를 합격으로 내보내는** 방향이라는
      점이다(실측: GT 8mm 융기 → 자 처짐 5.6mm → 허용 7mm 대비 "합격").
      값이 얼마나 덜 잡혔는지 알 수 없으면 판정을 신뢰할 수 없다.

      → 평활을 촘촘히/성기게 두 번 걸어 결과가 얼마나 달라지는지 본다.
        평활에 둔감하면(차이 작음) 요철이 충분히 해상된 것이고,
        민감하면(차이 큼) 아직 덜 잡힌 것이므로 그 차이를 편향
        추정치로 쓴다. 척도를 바꿔가며 수렴을 보는 표준적인 방법이다.

      각 척도의 값에서 노이즈 바닥을 먼저 빼야 한다. 촘촘한 평활은
      노이즈도 더 많이 통과시키므로, 빼지 않으면 노이즈 증가분을
      요철 편향으로 잘못 읽는다.

    Returns
    -------
    dict — max_gap_mm(촘촘한 척도), noise_floor_mm, resolution_bias_mm,
           upper_estimate_mm(= max_gap + bias), fine/coarse 원본
    """
    rf = straightedge_gap(points_3d, length_m=length_m, plane=plane,
                          target_bin_points=fine_points, **kw)
    rc = straightedge_gap(points_3d, length_m=length_m, plane=plane,
                          target_bin_points=coarse_points, **kw)
    nf = float(rf.get("noise_floor_mm") or 0.0)
    nc = float(rc.get("noise_floor_mm") or 0.0)
    bias = max(0.0, (rf["max_gap_mm"] - nf) - (rc["max_gap_mm"] - nc))

    out = dict(rf)
    out["noise_floor_mm"] = round(nf, 3)
    out["resolution_bias_mm"] = round(bias, 3)
    out["upper_estimate_mm"] = round(rf["max_gap_mm"] + bias, 3)
    out["coarse_gap_mm"] = rc["max_gap_mm"]
    out["resolution_limited"] = bool(bias > max(0.3, 0.1 * rf["max_gap_mm"]))
    return out


def judge_kcs_flatness(points_3d, member_class, plane=None,
                       sigma_normal_mm=None, target_sigma_mm=2.0, **kw):
    """
    부재 클래스에 맞는 KCS 직선자 기준으로 평활도를 판정한다.

    판정은 세 갈래다.
      기준초과       : 자 처짐량 자체가 허용치를 넘음
      판정보류(분해능): 처짐량은 허용 이내이나, 분해능 편향을 더하면 넘을
                       수 있음. 과소평가가 합격으로 새어나가지 않게 막는다
      합격           : 편향을 더해도 허용 이내

    sigma_normal_mm(eq5.region_uncertainty)을 주면 측정 불확실도가 목표를
    넘는 경우 판정을 내리지 않고 "측정불가"로 보고한다. 판정값이 노이즈에
    묻히는데 합격/불합격을 말하는 것은 정직하지 않다.

    Returns
    -------
    dict — checks[], judgement, is_pass, note
    """
    specs = KCS_FLATNESS_SPEC.get(member_class)
    out = {"member_class": member_class, "checks": [],
           "judgement": None, "is_pass": None, "note": None}
    if not specs:
        out["judgement"] = "해당없음"
        out["note"] = f"{member_class} 은 직선자 평활도 기준이 정의되지 않음"
        return out

    if plane is None:
        plane, _ = _EQ2.fit_plane_tls_ransac(np.asarray(points_3d, float),
                                             threshold=0.01)

    worst, worst_upper = -1.0, -1.0
    any_insufficient, res_limited = False, False
    for length_m, tol_mm in specs:
        r = gap_with_uncertainty(points_3d, length_m, plane=plane, **kw)
        r["tolerance_mm"] = tol_mm
        r["is_pass"] = bool(r["max_gap_mm"] <= tol_mm)
        r["ratio"] = round(r["max_gap_mm"] / tol_mm, 3) if tol_mm else None
        any_insufficient |= (not r["sufficient"])
        res_limited |= r["resolution_limited"]
        out["checks"].append(r)
        if tol_mm:
            worst = max(worst, r["max_gap_mm"] / tol_mm)
            worst_upper = max(worst_upper, r["upper_estimate_mm"] / tol_mm)

    if sigma_normal_mm is not None and sigma_normal_mm > target_sigma_mm:
        out["judgement"] = "측정불가"
        out["note"] = (f"법선방향 불확실도 σ_n={sigma_normal_mm}mm > "
                       f"목표 {target_sigma_mm}mm — 판정 보류")
        return out

    notes = []
    if worst > 1.0:
        out["is_pass"], out["judgement"] = False, "기준초과"
    elif worst_upper > 1.0:
        out["is_pass"], out["judgement"] = None, "판정보류(분해능)"
        c = max(out["checks"], key=lambda x: x["upper_estimate_mm"])
        notes.append(
            f"자 처짐 {c['max_gap_mm']:.2f}mm 는 허용 {c['tolerance_mm']}mm "
            f"이내이나, 분해능 편향 +{c['resolution_bias_mm']:.2f}mm 를 더하면 "
            f"{c['upper_estimate_mm']:.2f}mm 로 넘어설 수 있다. 요철 폭이 "
            f"측정 분해능(평활 반경 {c['profile']['radius_m']*1000:.0f}mm)보다 "
            f"좁아 값이 하한이므로 합격 판정을 내리지 않는다. 더 가까이서 "
            f"촬영하거나 격자 선 수를 늘려 재측정할 것")
    else:
        out["is_pass"], out["judgement"] = True, "합격"

    if res_limited and out["judgement"] == "합격":
        notes.append("요철이 분해능 부근이라 처짐량은 하한값 "
                     "(편향을 더해도 허용 이내라 합격)")
    if any_insufficient:
        notes.append("면이 규정 직선자 길이보다 작아 가진 구간 전체로 측정함 "
                     "(참고값). 규정대로 판정하려면 더 넓은 면을 촬영할 것")
    out["note"] = " / ".join(notes) if notes else None
    return out


# ============ 자체 검증 ============
if __name__ == "__main__":
    print("[식 ⑥] KCS 직선자 평활도 검증")
    rng = np.random.default_rng(17)

    def make_wall(bump_mm=0.0, bow_mm=0.0, W=3.6, Hh=2.4, n=9000, noise_mm=0.3):
        """벽면 점군. bump=국소 융기(σ5cm), bow=면 전체의 완만한 휨."""
        x = rng.uniform(-W / 2, W / 2, n)
        y = rng.uniform(-Hh / 2, Hh / 2, n)
        w = (bump_mm / 1000) * np.exp(-((x ** 2 + y ** 2) / (2 * 0.05 ** 2)))
        w += (bow_mm / 1000) * np.cos(np.pi * x / W)     # 전면에 걸친 활 모양
        Z = 1.2 - w                                       # 튀어나오면 Z 감소
        return np.column_stack([x, y, Z]) + rng.normal(0, noise_mm / 1000, (n, 3))

    print("\n  (1) 국소 융기 — 자가 융기 위에 얹히므로 틈 ≈ 융기 높이")
    for gt in (0, 3, 6, 10):
        P = make_wall(bump_mm=gt)
        r = straightedge_gap(P, length_m=3.0)
        print(f"    융기 {gt:2d}mm → 3m 자 처짐 {r['max_gap_mm']:5.2f}mm "
              f"(구간 {r['span_m']:.2f}m, 자 위치 {r['n_windows']}회)")

    print("\n  (2) 전역 휨 vs 국소 굴곡 — 전역 평면 잔차와 자 판정의 차이")
    #   완만한 활: 전역 잔차는 크지만 3m 자에는 덜 걸린다
    P_bow = make_wall(bow_mm=12.0)
    P_bump = make_wall(bump_mm=6.0)
    eq4 = _load("eq4_flatness_line")
    for name, P in (("전역 휨 12mm", P_bow), ("국소 융기 6mm", P_bump)):
        pl, _ = _EQ2.fit_plane_tls_ransac(P, threshold=0.02)
        res = eq4.detect_defects_region(P, plane=pl, threshold_mm=1.5)
        sg3 = straightedge_gap(P, length_m=3.0, plane=pl)
        sg1 = straightedge_gap(P, length_m=1.0, plane=pl)
        print(f"    {name:14s} eq4 전역잔차 최대 {res['raw_max_dev_mm']:5.2f}mm  "
              f"| 3m 자 {sg3['max_gap_mm']:5.2f}mm  1m 자 {sg1['max_gap_mm']:5.2f}mm")
    print("    → 전역 휨은 잔차가 커도 자에는 덜 걸리고, 국소 굴곡은 그 반대다.")
    print("      KCS 판정은 자 기준이므로 eq4 잔차로 합격/불합격을 말할 수 없다.")

    print("\n  (3) KCS 판정 (노출 콘크리트 벽 3m당 7mm) + 분해능 교차검증")
    for gt in (0, 5, 8, 12):
        P = make_wall(bump_mm=gt)
        j = judge_kcs_flatness(P, "wall")
        c = j["checks"][0]
        print(f"    융기 {gt:2d}mm → 자 처짐 {c['max_gap_mm']:5.2f}mm "
              f"(+편향 {c['resolution_bias_mm']:4.2f} → 상한 "
              f"{c['upper_estimate_mm']:5.2f}mm) / 허용 "
              f"{c['tolerance_mm']}mm → {j['judgement']}")
        if j["note"]:
            print(f"        ↳ {j['note']}")

    print("\n  (4) 측정불가 게이트")
    j = judge_kcs_flatness(make_wall(bump_mm=6), "wall", sigma_normal_mm=4.6)
    print(f"    σ_n=4.6mm (목표 2.0mm) → {j['judgement']}: {j['note']}")

    print("\n  (5) 면이 자보다 작을 때")
    P_small = make_wall(bump_mm=6, W=1.5, Hh=1.2, n=3000)
    j = judge_kcs_flatness(P_small, "wall")
    c = j["checks"][0]
    print(f"    면 1.5m < 자 3m → 구간 {c['span_m']}m, 규정충족 {c['sufficient']}")
    print(f"    {j['note']}")
