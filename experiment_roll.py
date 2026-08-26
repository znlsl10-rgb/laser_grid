#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
experiment_roll.py — 격자 회전각(roll) 이 검측 정확도에 미치는 영향
========================================================================
"가로선(H선)도 인식되어야 수직·수평·평활도가 정확해진다" 는 물음에 대한
실측 답이다. 결론부터:

  · 가로선 **검출** 은 이미 되고 있다 (A_선검출 Step4). 검출률 21/21.
  · 그런데 격자를 굴리지 않으면 가로선은 **깊이를 못 준다**. 레이저 평면이
    기선(X축)을 품어 시차가 선과 나란해지기 때문이다 (eq7 의 이득 g=∞).
    정답 데이터에서도 가로선의 v 는 1500점에 걸쳐 0.0004px 밖에 안 움직인다.
  · DOE 를 광축 둘레로 γ 만큼 굴리면 가로선에 x 성분이 생겨 깊이가 풀린다.
        V선 g = 1/cos γ      H선 g = 1/sin γ
    γ=45° 에서 두 계열이 g=√2 로 같아진다.

여기서 재는 것
-------------
굴림은 **공짜가 아니다**. 깊이를 주는 선이 늘어나는 대신 점당 깊이잡음이
1/cos γ 배가 된다. 각도(수직도·수평도)는 많은 점을 평균하므로 표본이
늘면 이긴다. 반면 평활도는 직선자 아래 **최대** 틈이라 극값 통계다 —
표본이 늘고 점당 잡음이 커지면 극값이 위로 밀린다. 그래서 두 항목이
반대로 움직인다. 그 교환비를 숫자로 내는 것이 이 스크립트다.

실행
----
    python3 experiment_roll.py                 # 기본 스캔
    python3 experiment_roll.py --rolls 0,20,30,45 --seeds 8

주의 — 여기 숫자는 합성 씬(synth_scene)의 것이다. 실제 하드웨어 사양과
장면이 들어오면 같은 스크립트를 그 값으로 다시 돌려 고르면 된다.
========================================================================
"""
import argparse
import os as _os
import importlib.util as _ilu
import numpy as np


def _load(name):
    spec = _ilu.spec_from_file_location(
        name, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                            f"{name}.py"))
    m = _ilu.module_from_spec(spec); spec.loader.exec_module(m)
    return m


SYN = _load("synth_scene")
EXP = _load("experiment_segmentation")
_EQ7 = _load("eq7_laser_plane")
# synth_scene 이 실제로 참조하는 calibration 인스턴스를 써야 한다.
CALIB = SYN._CALIB

SEEDS = (2026, 7, 99, 404, 1234, 55, 808, 3141)
BASE_PROFILE = "diagonal"


def _apply(n_v, n_h, roll_deg):
    """diagonal 프로파일을 주어진 격자·굴림으로 임시 설정한다."""
    p = CALIB.SPEC_PROFILES[BASE_PROFILE]
    p["n_vertical"], p["n_horizontal"] = int(n_v), int(n_h)
    p["laser_roll_deg"] = float(roll_deg)
    CALIB.use_profile(BASE_PROFILE)
    SYN.CAMERA_PARAMS.clear()
    SYN.CAMERA_PARAMS.update(CALIB.CAMERA_PARAMS)
    SYN.GRID.update({"n_vertical": int(n_v), "n_horizontal": int(n_h),
                     "fov_deg": CALIB.FOV_DEG, "samples_per_line": 250})


def straightedge_truth(n_v, n_h, roll_deg, dense=150):
    """
    이 기하에서의 직선자 처짐 참값 [mm].

    참값은 표면만으로 정해지지 않는다. 자가 걸치는 반폭 D 에 따라
    gap(d) = A(1−d/D) − A·exp(−d²/2σ²) 의 최대값이 달라지므로, 격자
    배치가 바뀌면 참값도 따라 움직인다. 그래서 표에서 찾지 않고
    "잡음 0 · 조밀 표본" 의 수렴값을 그때그때 구한다.
    """
    _apply(dense, dense, roll_deg)
    sc = SYN.build_scene(seed=2026, sigma_u_px=0.0)
    r = EXP.run_once(sc, sc["label_map"], "gt", sigma_u_px=1e-6)
    _apply(n_v, n_h, roll_deg)
    return float(r["wall_gap_mm"])


def measure(n_v, n_h, roll_deg, seeds=SEEDS):
    """한 구성으로 씬을 여러 번 만들어 항목별 오차를 모은다."""
    truth = straightedge_truth(n_v, n_h, roll_deg)
    su = CALIB.SIGMA_U_PX
    acc = {k: [] for k in ("wall", "floor", "shoring", "gap", "band")}
    n_depth = n_total = None
    gains = {}
    for sd in seeds:
        sc = SYN.build_scene(seed=sd, sigma_u_px=su)
        if n_depth is None:
            la = sc["line_angles"]
            n_total = len(la)
            n_depth = sum(1 for a in la.values()
                          if np.isfinite(a["depth_gain"])
                          and a["depth_gain"] <= _EQ7.MAX_DEPTH_GAIN)
            for pre in "VH":
                g = [a["depth_gain"] for l, a in la.items() if l[0] == pre]
                fin = [x for x in g if np.isfinite(x)]
                gains[pre] = (round(float(np.median(fin)), 3) if fin
                              else float("inf"))
        r = EXP.run_once(sc, sc["label_map"], "gt", sigma_u_px=su)
        for k in ("wall", "floor", "shoring"):
            acc[k].append(r["errors_deg"].get(k, np.nan))
        gap = r.get("wall_gap_mm", np.nan)
        acc["gap"].append(gap - truth)
        acc["band"].append(r.get("wall_gap_upper_mm", np.nan) - gap)
    out = {k: float(np.nanmean(np.abs(v))) for k, v in acc.items()}
    out.update(n_depth=n_depth, n_total=n_total, truth=truth,
               gain_v=gains.get("V"), gain_h=gains.get("H"),
               shoring_max=float(np.nanmax(np.abs(acc["shoring"]))))
    return out


def main():
    ap = argparse.ArgumentParser(description="격자 회전각 스캔")
    ap.add_argument("--rolls", default="0,15,30,45",
                    help="쉼표로 구분한 회전각 [°]")
    ap.add_argument("--grid", default="30x30",
                    help="V선수xH선수 (예: 30x30)")
    ap.add_argument("--seeds", type=int, default=len(SEEDS))
    a = ap.parse_args()
    n_v, n_h = (int(x) for x in a.grid.lower().split("x"))
    seeds = SEEDS[:max(1, a.seeds)]
    keep = CALIB.ACTIVE_PROFILE
    keep_p = dict(CALIB.SPEC_PROFILES[BASE_PROFILE])

    # 기준선 — 굴리지 않은 개선안 사양 (가로선은 깊이에 못 씀)
    CALIB.use_profile("improved")
    SYN.CAMERA_PARAMS.clear()
    SYN.CAMERA_PARAMS.update(CALIB.CAMERA_PARAMS)
    SYN.GRID.update({"n_vertical": CALIB.N_VERTICAL,
                     "n_horizontal": CALIB.N_HORIZONTAL,
                     "fov_deg": CALIB.FOV_DEG, "samples_per_line": 250})
    su = CALIB.SIGMA_U_PX
    tr = SYN.GT_STRAIGHTEDGE_MM["improved"]
    acc = {k: [] for k in ("wall", "floor", "shoring", "gap", "band")}
    for sd in seeds:
        sc = SYN.build_scene(seed=sd, sigma_u_px=su)
        r = EXP.run_once(sc, sc["label_map"], "gt", sigma_u_px=su)
        for k in ("wall", "floor", "shoring"):
            acc[k].append(r["errors_deg"].get(k, np.nan))
        gap = r.get("wall_gap_mm", np.nan)
        acc["gap"].append(gap - tr)
        acc["band"].append(r.get("wall_gap_upper_mm", np.nan) - gap)
    base = {k: float(np.nanmean(np.abs(v))) for k, v in acc.items()}

    print("=" * 92)
    print(f"격자 회전각(roll) 스캔 — 합성 씬 {len(seeds)}회, 격자 {n_v}×{n_h}")
    print("=" * 92)
    hdr = (f"{'구성':<24}{'깊이선':>7}{'g(V)':>7}{'g(H)':>8}"
           f"{'벽°':>9}{'바닥°':>9}{'동바리°':>10}{'평활mm':>9}{'밴드mm':>9}")
    print(hdr)
    print("-" * 92)
    print(f"{'기준: improved roll 0°':<24}"
          f"{CALIB.SPEC_PROFILES['improved']['n_vertical']:>7}"
          f"{1.0:>7.2f}{'inf':>8}"
          f"{base['wall']:>9.4f}{base['floor']:>9.4f}"
          f"{base['shoring']:>10.4f}{base['gap']:>9.4f}{base['band']:>9.4f}")
    rows = []
    for rd in (float(x) for x in a.rolls.split(",")):
        m = measure(n_v, n_h, rd, seeds)
        rows.append((rd, m))
        gh = m["gain_h"]
        print(f"{'roll ' + format(rd, 'g') + '°':<24}{m['n_depth']:>7}"
              f"{m['gain_v']:>7.2f}"
              f"{(format(gh, '.2f') if np.isfinite(gh) else 'inf'):>8}"
              f"{m['wall']:>9.4f}{m['floor']:>9.4f}"
              f"{m['shoring']:>10.4f}{m['gap']:>9.4f}{m['band']:>9.4f}")
    print("-" * 92)
    print("읽는 법")
    print("  · 깊이선 = 이득 g ≤ %g 를 통과해 실제 삼각측량에 쓰인 선 수."
          % _EQ7.MAX_DEPTH_GAIN)
    print("    굴림이 작으면 가로선의 g 가 커서 문턱을 못 넘고, 그때는")
    print("    가로선을 검출해도 깊이에는 한 점도 못 보탠다.")
    print("  · 각도(벽·바닥·동바리)는 굴리면 좋아진다. 표본이 늘고, 무엇보다")
    print("    면을 한 방향 줄무늬가 아니라 격자로 덮어 적합이 안정된다.")
    print("  · 평활도는 반대로 나빠진다. 직선자 처짐은 '가장 큰 틈' 이라")
    print("    극값 통계이고, 점당 잡음이 1/cos γ 배로 커지면 극값이 위로")
    print("    밀린다. 각도를 살리려 굴리면 평활도를 그만큼 내주는 거래다.")
    print("  · 어느 쪽을 살릴지는 현장이 정한다. 수직도가 주 검측이면 굴리고,")
    print("    평활도가 주 검측이면 굴리지 말고 V선 수를 늘리는 편이 낫다.")

    CALIB.SPEC_PROFILES[BASE_PROFILE].update(keep_p)
    CALIB.use_profile(keep)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
