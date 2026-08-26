#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
demo_report.py — 산출물 형식 견본 생성
========================================================================
합성 씬(벽 + 바닥 + 동바리 + 철근)을 한 장 만들어 실제 산출물과 똑같은
형식으로 엑셀 조서와 세그멘테이션 이미지를 낸다. Isaac 도 촬영본도 없이
"결과가 어떤 모양으로 나오는지" 를 먼저 확인하기 위한 것이다.

실촬영 경로는 inspect_png.py 이며, 그쪽과 같은 report_excel·report 를
쓰므로 형식이 어긋날 일이 없다.

실행:  python3 demo_report.py [출력폴더]
========================================================================
"""
import sys, os, collections
import numpy as np
import importlib.util as _ilu


def _load(name):
    spec = _ilu.spec_from_file_location(
        name, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           f"{name}.py"))
    m = _ilu.module_from_spec(spec); spec.loader.exec_module(m)
    return m


SYN = _load("synth_scene")
PIPE = _load("pipeline_region")
REPORT = _load("report")
XLS = _load("report_excel")
CALIB = SYN._CALIB


def main(out_dir=None):
    out_dir = out_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      "_report_demo")
    os.makedirs(out_dir, exist_ok=True)

    scene = SYN.build_scene(with_rebar=True)
    res = PIPE.inspect_capture(
        scene["lines_pixels"], scene["line_angles"], scene["camera_params"],
        scene["R_world_cam"], label_map=scene["label_map"],
        id_to_semantic=scene["id_to_semantic"], rgb_off=scene["rgb_off"],
        backend="gt", sigma_u_px=CALIB.SIGMA_U_PX)
    print(PIPE.format_report(res))

    seg_png = REPORT.save_segmentation(
        os.path.join(out_dir, "세그멘테이션.png"), res,
        base_image=scene["rgb_off"])

    # 마스크 화소 수 — 격자점 수와 다르다. 점은 검측에 실제로 쓰인 표본이고
    # 화소는 그 부재가 화면에서 차지하는 면적이다. 둘을 나란히 두면 "면적은
    # 큰데 표본이 적은" 부재(가는 철근)가 눈에 띈다.
    cnt = collections.Counter(np.asarray(scene["label_map"]).ravel().tolist())
    label_px = {scene["id_to_semantic"][k].split(":")[-1]: v
                for k, v in cnt.items() if k in scene["id_to_semantic"]}

    gt = scene["gt"]
    meta = {"현장": "합성 검증 씬 (demo_report.py)",
            "입력 이미지": "synth_scene 생성",
            "측정 거리(m)": SYN.WALL_DIST_M,
            "장비 하향각(°)": SYN.DEVICE_PITCH_DEG,
            "정답 — 벽 수직도(°)": gt["wall_verticality_deg"],
            "정답 — 바닥 수평도(°)": gt["floor_horizontality_deg"],
            "정답 — 동바리 수직도(°)": gt["shoring_verticality_deg"],
            "정답 — 철근 수직도(°)": gt["rebar_verticality_deg"],
            "정답 — 벽 융기(mm)": gt["wall_bump_mm"]}
    pitch = CALIB.projection_mm_at(SYN.WALL_DIST_M) / (CALIB.N_VERTICAL - 1)
    caveats = [
        "합성 씬이다. 실촬영의 재질 반사·번짐·모션블러는 반영되지 않는다.",
        f"격자 피치가 {pitch:.1f}mm 다. 이보다 가는 부재(철근 Ø25.4mm)는 "
        f"V선이 한두 줄만 걸려 표본이 얕다. 사양 프로파일을 바꾸면 이 부재의 "
        f"측정 여부 자체가 달라진다.",
    ]
    # 부재별 3D 점군 — 어디에 점이 찍혔는지, 부재가 제대로 갈렸는지
    pc_png = REPORT.save_pointcloud_3d(
        os.path.join(out_dir, "3D점군.png"), res, scene["g_hat"],
        title="3D 점군 — 합성 검증 씬")
    pc_csv = REPORT.save_pointcloud_csv(
        os.path.join(out_dir, "3D좌표.csv"), res,
        g_hat=scene["g_hat"], stride=10)
    xlsx = XLS.save_excel(os.path.join(out_dir, "품질검측조서.xlsx"), res,
                          meta=meta, label_pixels=label_px,
                          seg_image_path=seg_png, extra_caveats=caveats,
                          g_hat=scene["g_hat"], pointcloud_image=pc_png,
                          pc_stride=10)
    REPORT.save_record(REPORT.build_record(res, meta),
                       os.path.join(out_dir, "조서.json"))
    print()
    print(f"세그멘테이션 이미지: {seg_png}")
    if pc_png:
        print(f"3D 점군 이미지:     {pc_png}")
    if pc_csv:
        print(f"3D 좌표 CSV:       {pc_csv}")
    print(f"엑셀 조서:          {xlsx}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else None))
