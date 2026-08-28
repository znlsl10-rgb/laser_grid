#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/scene_perturb.py — 회귀 [8] 이 쓰는 씬 훼손 도구
========================================================================
세그멘테이션이 완벽하지 않을 때도 각도 판정이 버티는지 보려면, 마스크를
일부러 망가뜨려 봐야 한다. 예전에는 experiment_segmentation.py 라는
연구용 스크립트 안에 있었는데, 그 파일의 나머지(표 출력·스윕 CLI)는
아무도 쓰지 않아 지웠다. 검증에 필요한 두 함수만 여기로 옮겼다.
========================================================================
"""
import os as _os
import importlib.util as _ilu
import numpy as np


def _load(name):
    root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    spec = _ilu.spec_from_file_location(name, _os.path.join(root, f"{name}.py"))
    m = _ilu.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


_PIPE = _load("pipeline_region")
_EQ5 = _load("eq5_region_assign")

try:
    from scipy import ndimage as _ndi
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

TARGETS = {"wall": "wall_verticality_deg",
           "floor": "floor_horizontality_deg",
           "shoring": "shoring_verticality_deg"}


def perturb_mask(label_map, mode, amount, rng, bg_id=0):
    """
    정답 마스크를 현실적인 방식으로 훼손한다.

    mode
    ----
    "erode"/"dilate" : 클래스 경계를 k px 안팎으로 밀어 어긋나게 한다.
        VLM/SAM 마스크는 경계가 몇 px 씩 어긋나는 것이 보통이며, 그 점들이
        엉뚱한 영역의 평면적합에 들어가면 법선을 끌어당긴다.
    "mislabel"       : 클래스 id 를 확률 p 로 서로 바꾼다(라벨 혼동).
    "dropout"        : 한 클래스를 통째로 배경으로 지운다(부재 누락).
    """
    lm = label_map.copy()
    ids = [int(i) for i in np.unique(label_map) if int(i) != bg_id]
    if not ids or amount == 0:
        return lm

    if mode in ("erode", "dilate"):
        if not HAS_SCIPY:
            return lm
        k = int(abs(amount))
        if mode == "erode":
            out = np.full_like(lm, bg_id)
            for cid in ids:
                out[_ndi.binary_erosion(label_map == cid, iterations=k,
                                        border_value=0)] = cid
            return out
        # 팽창: 각 클래스를 차례로 넓히고 나중 클래스가 겹친 자리를 가져간다.
        # 원본 라벨을 보존해 버리면 경계가 배경 쪽으로만 밀려 실제로는
        # 아무 일도 일어나지 않는다(옆 부재의 점을 끌어들이지 못함).
        out = label_map.copy()
        for cid in ids:
            out[_ndi.binary_dilation(label_map == cid, iterations=k)] = cid
        return out

    if mode == "mislabel":
        # 화소 단위가 아니라 **덩어리 단위**로 바꾼다. VLM 은 한 영역의
        # 라벨을 통째로 틀리지, 화소를 무작위로 흩뿌리지 않는다.
        for cid in ids:
            if rng.random() < amount:
                other = [i for i in ids if i != cid]
                if other:
                    lm[label_map == cid] = int(rng.choice(other))
        return lm

    if mode == "dropout":
        for cid in ids:
            if rng.random() < amount:
                lm[label_map == cid] = bg_id
        return lm

    raise ValueError(f"알 수 없는 mode: {mode}")


# =====================================================================
# 1회 측정
# =====================================================================
def run_once(scene, label_map=None, backend="gt", sigma_u_px=0.2):
    """한 조건에서 영역별 검측을 수행하고 클래스별 오차를 뽑는다."""
    res = _PIPE.inspect_capture(
        scene["lines_pixels"], scene["line_angles"],
        scene["camera_params"], scene["R_world_cam"],
        label_map=(label_map if backend == "gt" else None),
        id_to_semantic=(scene["id_to_semantic"] if backend == "gt" else None),
        rgb_off=scene["rgb_off"], backend=backend, sigma_u_px=sigma_u_px)

    best = {}
    for r in res["regions"]:
        if r["status"] != "measured":
            continue
        c = r["class"]
        if c not in best or r["n_points"] > best[c]["n_points"]:
            best[c] = r

    out = {"n_regions": res["summary"]["n_regions"],
           "label_corrections": res["summary"]["label_corrections"],
           "assigned_points": res.get("assign_stats", {}).get("assigned", 0),
           "errors_deg": {}, "measured": {}, "missing": []}
    for cls, key in TARGETS.items():
        r = best.get(cls)
        if r is None:
            out["missing"].append(cls)
            continue
        out["measured"][cls] = round(r["theta_deg"], 4)
        out["errors_deg"][cls] = round(abs(r["theta_deg"] - scene["gt"][key]), 4)
    w = best.get("wall")
    f = (w or {}).get("flatness") or {}
    if f.get("applicable"):
        out["wall_gap_mm"] = f.get("max_gap_mm")
        out["wall_gap_upper_mm"] = f.get("upper_estimate_mm")
        out["wall_flatness_judgement"] = f.get("judgement")
    return out
