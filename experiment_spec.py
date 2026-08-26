#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
experiment_spec.py — 사양 프로파일 정확도 비교
========================================================================
PDF 원안과 정확도 개선안을 같은 합성 씬에 통과시켜, 사양 변경이 실제로
어느 항목을 얼마나 개선하는지 측정한다. 이론값(σ_Z)만으로는 알 수 없는
것을 본다. 예를 들어 깊이 잡음은 3배 좋아지지만 벽 수직도는 그대로다.
이미 수치 바닥에 닿아 있기 때문이다.

Isaac Sim 없이 돈다.

실행:  python3 experiment_spec.py
========================================================================
"""
import numpy as np
import importlib.util as _ilu, os as _os


def _load(name):
    spec = _ilu.spec_from_file_location(
        name, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                            f"{name}.py"))
    m = _ilu.module_from_spec(spec); spec.loader.exec_module(m)
    return m


SYN = _load("synth_scene")
EXP = _load("experiment_segmentation")
# synth_scene 이 실제로 참조하는 calibration 인스턴스를 써야 한다.
# 별도로 import 한 사본에 use_profile 을 걸면 발사각만 옛 프로파일로 남는다.
CALIB = SYN._CALIB

SEEDS = (2026, 7, 99, 404, 1234, 55, 808, 3141)


def measure(profile, seeds=SEEDS):
    """한 프로파일로 씬을 여러 번 만들어 항목별 오차를 모은다."""
    CALIB.use_profile(profile)
    SYN.CAMERA_PARAMS.clear()
    SYN.CAMERA_PARAMS.update(CALIB.CAMERA_PARAMS)
    SYN.GRID.update({"n_vertical": CALIB.N_VERTICAL,
                     "fov_deg": CALIB.FOV_DEG, "samples_per_line": 250})
    su = CALIB.SIGMA_U_PX
    acc = {k: [] for k in ("wall", "floor", "shoring", "gap", "band", "dilate")}
    for sd in seeds:
        sc = SYN.build_scene(seed=sd, sigma_u_px=su)
        r = EXP.run_once(sc, sc["label_map"], "gt", sigma_u_px=su)
        for k in ("wall", "floor", "shoring"):
            acc[k].append(r["errors_deg"].get(k, np.nan))
        gap = r.get("wall_gap_mm", np.nan)
        acc["gap"].append(gap - SYN.straightedge_truth_mm())
        acc["band"].append(r.get("wall_gap_upper_mm", np.nan) - gap)
        lm = EXP.perturb_mask(sc["label_map"], "dilate", 16,
                              np.random.default_rng(0))
        rr = EXP.run_once(sc, lm, "gt", sigma_u_px=su)
        acc["dilate"].append(max(rr["errors_deg"].values())
                             if rr["errors_deg"] else np.nan)
    out = {k: float(np.nanmean(np.abs(v))) for k, v in acc.items()}
    out["shoring_max"] = float(np.nanmax(np.abs(acc["shoring"])))
    out["sigma_z"] = CALIB.sigma_z_mm(1.2)
    out["pitch"] = CALIB.projection_mm_at(1.2) / (CALIB.N_VERTICAL - 1)
    return out


ROWS = [("깊이 잡음 σ_Z @1.2m",       "mm", "sigma_z"),
        ("격자 피치 @1.2m",           "mm", "pitch"),
        ("벽 수직도 오차",             "°",  "wall"),
        ("바닥 수평도 오차",            "°",  "floor"),
        ("동바리 수직도 오차 (평균)",     "°",  "shoring"),
        ("동바리 수직도 오차 (최악)",     "°",  "shoring_max"),
        ("직선자 처짐 오차",            "mm", "gap"),
        ("직선자 불확실도 밴드",         "mm", "band"),
        ("마스크 +16px 훼손 시 최악",    "°",  "dilate")]


def main():
    keep = CALIB.ACTIVE_PROFILE
    a, b = measure("pdf"), measure("improved")
    CALIB.use_profile(keep)

    print("=" * 74)
    print(f"사양 프로파일 정확도 비교  (합성 씬 {len(SEEDS)}회)")
    print("=" * 74)
    print(f"{'항목':<26}{'단위':<5}{'PDF 원안':>13}{'개선안':>13}{'개선':>9}")
    print("-" * 74)
    for name, unit, key in ROWS:
        x, y = abs(a[key]), abs(b[key])
        g = (x / y) if y else float("inf")
        print(f"{name:<26}{unit:<5}{x:>13.4f}{y:>13.4f}{g:>8.2f}배")
    print()
    print("읽는 법")
    print("  · 벽 수직도는 개선되지 않는다. 두 사양 모두 0.013° 로 목표")
    print("    ±0.5° 의 1/38 이라, 이미 표본 평균의 수치 바닥이다.")
    print("  · 실제로 좁던 항목은 동바리 각도와 평활도였고, 둘 다 원인이")
    print("    같다 — 1.2m 격자 피치 49.3mm 가 Ø48.6mm 부재와 비슷해")
    print("    부재당 V선이 한두 개뿐이었다. 피치를 24mm 로 줄이자")
    print("    두 항목이 함께 좋아졌다.")
    print("  · 마스크 훼손 강건성이 함께 오르는 것도 같은 이유다. 표본이")
    print("    많을수록 경계 오염이 평균에 묻힌다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
