#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
inspection.py — [품질검측] 현장 촬영 + 선검출(A) + 삼각측량(B) + eq 검증
========================================================================
실제 품질검측에서 매번 실행하는 메인 파이프라인.

흐름:
  씬 열기 + 카메라/레이저 세팅
  → 스테이션별 촬영
      레이저 위치 계산 (laser_pos, cam_pos, R)
      → IMU 기울기 주입 + 카메라 자세 세팅 (R_tilted, M44)
      → Raycast → GT 픽셀 + z_cam_m 저장
      → 발광 메시 렌더 → rgb_laser 획득
      → [A] detect()  → lines_pixels
      → 선 필터링 (불량선 제거)
      → IMU 픽셀 보정
      → data.json / rgb_raw.png / rgb_laser.png / overlay.png 저장
  → 3_pipeline_eq_verify.py 로 eq 검증

실행:
  python3 inspection.py
========================================================================
"""
import os, sys, json
import numpy as np

# =====================================================================
# 설정값 (캘리브레이션)
# =====================================================================
# 캘리브레이션 상수는 calibration.py 가 단일 출처다.
# 이전에는 inspection.py 와 synth_scene.py 가 같은 값을 따로 들고 있어
# 한쪽만 고치면 조용히 어긋났다.
import importlib.util as __ilu, os as __os
__spec = __ilu.spec_from_file_location(
    "calibration", __os.path.join(__os.path.dirname(__os.path.abspath(__file__)),
                                  "calibration.py"))
CALIB = __ilu.module_from_spec(__spec); __spec.loader.exec_module(CALIB)

# 카메라: 8mm F2.0 저왜곡 / 센서 3.45µm / 2448×2048  (PDF 2.2)
#   f_px = 8mm / 3.45µm = 2318.8 px
CAMERA_PARAMS = dict(CALIB.CAMERA_PARAMS)
# 격자: 수직 20 + 수평 20 (400교점), DOE 발산각 42.61° (투사 936mm @1.2m)
GRID_PARAMS = dict(CALIB.GRID_PARAMS)

STATIONS = {
    "StationA_Wall":  {"target": "/World/StationA/WallBackFace",
                       "normal": [0., -1., 0.], "inspect": "verticality",
                       "standoff_m": 1.0},
    "StationA_Floor": {"target": "/World/StationA/FloorTop",
                       "normal": [0.,  0., 1.], "inspect": "horizontality",
                       "standoff_m": 1.0},
    "StationB":       {"target": "/World/StationB/Panel",
                       "normal": [0., -1., 0.], "inspect": "flatness",
                       "standoff_m": 1.0},
    # StationC: 벽+바닥+동바리+철근이 한 화면에 들어오는 혼합 장면.
    # inspect 를 "auto" 로 두면 검측 종류를 설정에서 정하지 않고
    # 세그멘테이션 결과에서 영역별로 결정한다.
    "StationC_Mixed": {"target": "/World/StationC/WallFace",
                       "normal": [0., -1., 0.], "inspect": "auto",
                       "standoff_m": 1.2,
                       "pitch_down_deg": 22.0,
                       "segmentation": True},
}

# 세그멘테이션 기반 영역별 검측 설정
SEGMENTATION = {
    "enabled":       True,      # inspect=="auto" 스테이션에서 사용
    "backend":       "gt",      # gt(Isaac Semantics) | geom | sam | vlm
    "erode_px":      3,         # 마스크 침식 (경계 오염 제거)
    "erode_thin_px": 1,         # 동바리·철근처럼 얇은 부재는 약하게
    "sigma_u_px":    0.2,       # 선검출 픽셀 오차 (불확실도 산정용)
    "target_sigma_mm": 2.0,     # 평활도 목표 정밀도 (PDF 1.1)
}
SCENE_USD  = "/home/develop/Desktop/laser_grid_test_4/inspection_lab_realistic.usda"
GT_JSON    = "/home/develop/Desktop/laser_grid_test_4/inspection_ground_truth_realistic.json"
OUTPUT_DIR = "/home/develop/Desktop/laser_grid_test_4/result/realistic_dataset"

# =====================================================================
# A, B 알고리즘 import
# =====================================================================
import importlib.util as _ilu, os as _os

def _load_algo(path, func_name):
    spec = _ilu.spec_from_file_location("_algo", path)
    mod  = _ilu.module_from_spec(spec); spec.loader.exec_module(mod)
    return getattr(mod, func_name)


_MODULE_CACHE = {}

def _load_module_cached(name):
    """같은 모듈을 두 번 실행하지 않도록 캐시해 로드한다."""
    if name not in _MODULE_CACHE:
        spec = _ilu.spec_from_file_location(
            name, _os.path.join(_HERE, f"{name}.py"))
        m = _ilu.module_from_spec(spec); spec.loader.exec_module(m)
        _MODULE_CACHE[name] = m
    return _MODULE_CACHE[name]

_HERE = _os.path.dirname(_os.path.abspath(__file__))
fn_detect = None  # 초기값, SimulationApp 이후 _init_algorithms()로 로드

def _init_algorithms():
    """SimulationApp 완전 초기화 후 A 알고리즘 로드. main()과 experiment 양쪽에서 호출."""
    global fn_detect
    if fn_detect is None:
        fn_detect = _load_algo(_os.path.join(_HERE, "A_선검출.py"), "detect")
    return fn_detect

# =====================================================================
# Isaac Sim import
# =====================================================================
_RUNNING_IN_GUI = False
try:
    import omni.usd
    if omni.usd.get_context().get_stage() is not None:
        _RUNNING_IN_GUI = True
except Exception:
    pass

if not _RUNNING_IN_GUI:
    from isaacsim import SimulationApp
    simulation_app = SimulationApp({"headless": False,
                                    "width": 2448, "height": 2048})

import omni.kit.app
try:
    import carb
    cs = carb.settings.get_settings()
    cs.set("/rtx/post/bloom/enabled",      False)
    cs.set("/rtx/post/lensFlares/enabled", False)
except Exception:
    pass

from pxr import Usd, UsdGeom, UsdPhysics, UsdShade, UsdLux, Gf, Sdf
from omni.isaac.core import World
from omni.isaac.core.utils.stage import open_stage, get_current_stage
from omni.isaac.sensor import Camera
from omni.physx import get_physx_scene_query_interface
try:
    import omni.replicator.core as rep  # noqa
except Exception:
    pass
from PIL import Image, ImageDraw  # SimulationApp 이후 안전하게 import


def LOG(msg): print(msg, flush=True)

# =====================================================================
# 유틸
# =====================================================================
def _norm(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def _wait(world, n):
    for _ in range(n):
        if _RUNNING_IN_GUI: omni.kit.app.get_app().update()
        else:               world.step(render=True)


def _world_center_of(stage, path):
    p = stage.GetPrimAtPath(path)
    if not p or not p.IsValid(): return None
    try:
        bb  = UsdGeom.Imageable(p).ComputeWorldBound(
                Usd.TimeCode.Default(), UsdGeom.Tokens.default_)
        box = bb.ComputeAlignedBox()
        mn, mx = box.GetMin(), box.GetMax()
        if mn[0] <= mx[0]:
            return np.array([(mn[i]+mx[i])/2. for i in range(3)])
    except Exception:
        pass
    try:
        m = UsdGeom.Xformable(p).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        t = m.ExtractTranslation()
        return np.array([t[0], t[1], t[2]])
    except Exception:
        return None


def _rotmat_to_quat(R):
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t+1.)*2; w=.25*s; x=(R[2,1]-R[1,2])/s; y=(R[0,2]-R[2,0])/s; z=(R[1,0]-R[0,1])/s
    elif R[0,0]>R[1,1] and R[0,0]>R[2,2]:
        s=np.sqrt(1.+R[0,0]-R[1,1]-R[2,2])*2; w=(R[2,1]-R[1,2])/s; x=.25*s; y=(R[0,1]+R[1,0])/s; z=(R[0,2]+R[2,0])/s
    elif R[1,1]>R[2,2]:
        s=np.sqrt(1.+R[1,1]-R[0,0]-R[2,2])*2; w=(R[0,2]-R[2,0])/s; x=(R[0,1]+R[1,0])/s; y=.25*s; z=(R[1,2]+R[2,1])/s
    else:
        s=np.sqrt(1.+R[2,2]-R[0,0]-R[1,1])*2; w=(R[1,0]-R[0,1])/s; x=(R[0,2]+R[2,0])/s; y=(R[1,2]+R[2,1])/s; z=.25*s
    return np.array([w,x,y,z])


def _device_tilt_from_R(R):
    f = R[:,2]; up = R[:,1]
    pitch = np.degrees(np.arcsin(np.clip(-f[2],-1,1)))
    roll  = np.degrees(np.arctan2(up[0],up[2])) if abs(up[2])>1e-9 else 0.
    return {"pitch_deg": float(pitch), "roll_deg": float(roll)}


def _simulate_imu(R_ideal, pitch_deg, roll_deg):
    p,r = np.radians(pitch_deg), np.radians(roll_deg)
    Rp = np.array([[1,0,0],[0,np.cos(p),-np.sin(p)],[0,np.sin(p),np.cos(p)]])
    Rr = np.array([[np.cos(r),0,np.sin(r)],[0,1,0],[-np.sin(r),0,np.cos(r)]])
    R_t = R_ideal @ (Rr @ Rp)
    g_l = R_t.T @ np.array([0.,0.,-1.])
    return R_t, {
        "measured_pitch_deg": float(np.degrees(np.arctan2(-g_l[1],-g_l[2]))),
        "measured_roll_deg":  float(np.degrees(np.arctan2( g_l[0],-g_l[2]))),
        "injected_pitch_deg": pitch_deg, "injected_roll_deg": roll_deg,
    }


def _correct_imu(lines_pixels, imu_data, cp):
    f,cx,cy = cp["f_px"],cp["cx_px"],cp["cy_px"]
    p = np.radians(imu_data["measured_pitch_deg"])
    r = np.radians(imu_data["measured_roll_deg"])
    Rp_i = np.array([[1,0,0],[0,np.cos(p),np.sin(p)],[0,-np.sin(p),np.cos(p)]])
    Rr_i = np.array([[np.cos(r),0,-np.sin(r)],[0,1,0],[np.sin(r),0,np.cos(r)]])
    K = np.array([[f,0,cx],[0,f,cy],[0,0,1]],float)
    H = K @ (Rp_i @ Rr_i) @ np.linalg.inv(K)
    out = {}
    for lid,pts in lines_pixels.items():
        if not pts: out[lid]=pts; continue
        arr=np.array(pts,float); uvh=np.hstack([arr,np.ones((len(arr),1))])
        c=(H@uvh.T).T; w=c[:,2:3]; w=np.where(np.abs(w)<1e-9,1e-9,w)
        out[lid]=(c[:,:2]/w).tolist()
    return out


def _filter_lines(lines_pixels, cp, min_pts=30, max_u_std=80., max_v_std=80.):
    W,H=cp["resolution"]; vV,vH,rej=[],[],{}
    for lid,pts in lines_pixels.items():
        arr=np.array(pts,float) if pts else np.zeros((0,2))
        if len(arr)<min_pts: rej[lid]=f"점수부족({len(arr)})"; continue
        ok=(arr[:,0]>=0)&(arr[:,0]<W)&(arr[:,1]>=0)&(arr[:,1]<H)
        if ok.mean()<0.5: rej[lid]="범위이탈"; continue
        if lid.startswith("V"):
            if np.std(arr[ok,0])>max_u_std: rej[lid]="V직선성불량"; continue
            vV.append(lid)
        elif lid.startswith("H"):
            if np.std(arr[ok,1])>max_v_std: rej[lid]="H직선성불량"; continue
            vH.append(lid)
    key=lambda l: int(l[1:]) if l[1:].isdigit() else 0
    return sorted(vV,key=key), sorted(vH,key=key), rej



def _set_lighting_intensity(stage, scale):
    """조명 강도를 scale 배로 조정 (0=끔, 1=원래)."""
    _BASE_INTENSITIES = {
        "/World/SunKey":     2500.0,
        "/World/AmbientFill": 400.0,
        "/World/SunFill":     900.0,
    }
    try:
        for path, base_val in _BASE_INTENSITIES.items():
            p = stage.GetPrimAtPath(path)
            if p and p.IsValid():
                p.GetAttribute("intensity").Set(float(base_val * scale))
    except Exception:
        pass

def _make_emissive_grid(stage, lines_world, normal):
    """
    산업용 520nm 레이저 선 렌더링.

    실제 Laserlands 20×20 DOE 520nm 레이저 특성:
      - 선폭: 벽면 1m 거리에서 약 0.5~0.8mm
        → 카메라(f=1593px, 1m) 기준 약 0.8~1.3px 로 맺힘
        → 3D 선폭: 0.0006m (0.6mm) 로 설정
      - 발광 강도: 90mW급 레이저, 벽면 반사 후 카메라 수광
        → emissiveScale 800~1200 (배경 조명 대비 압도적 밝기)
      - 색상: 순수 녹색 (R=0, G=1, B=0) — 520nm 단색광
        → emissiveColor (0, 1, 0), diffuseColor (0, 0, 0)
      - 레이저선은 자체 발광만, diffuse 반사 없음

    선폭이 좁아지면(0.6mm → ~1px):
      → 가우시안 피팅의 peak가 더 선명 → 서브픽셀 정밀도 향상
      → pixel_rmse 감소 → σZ 감소 기대
    """
    base="/World/LaserGrid"
    if stage.GetPrimAtPath(base).IsValid(): stage.RemovePrim(base)
    UsdGeom.Scope.Define(stage, base)
    mat = UsdShade.Material.Define(stage, base+"/GreenLaser")
    sh  = UsdShade.Shader.Define(stage,  base+"/GreenLaser/S")
    sh.CreateIdAttr("UsdPreviewSurface")

    # 순수 520nm 녹색 — R,B 성분 제거 (G-avg(R,B) 채널분리 효과 극대화)
    sh.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(0.0, 1.0, 0.0))
    # 산업용 레이저 밝기: 배경 조명(2500 lux 수준) 대비 압도적으로 밝게
    sh.CreateInput("emissiveScale", Sdf.ValueTypeNames.Float).Set(1000.0)
    # 레이저선 자체는 diffuse 반사 없음 (발광체)
    sh.CreateInput("diffuseColor",  Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(0.0, 0.0, 0.0))
    sh.CreateInput("roughness",     Sdf.ValueTypeNames.Float).Set(0.0)
    sh.CreateInput("metallic",      Sdf.ValueTypeNames.Float).Set(0.0)
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")

    nrm = np.asarray(normal, float)
    nrm /= max(np.linalg.norm(nrm), 1e-9)

    # 선폭: 0.6mm (실제 산업용 레이저 선폭)
    # 기존 5mm → 0.6mm: 카메라 상 ~8px → ~1px 로 좁아짐
    LASER_WIDTH_M = 0.0006

    n = 0
    for lid, pts in lines_world.items():
        if len(pts) < 2: continue
        # 벽면에서 살짝 띄워서 z-fighting 방지 (0.5mm)
        off = [P + nrm * 0.0005 for P in pts]
        cv  = UsdGeom.BasisCurves.Define(stage, f"{base}/{lid}")
        cv.CreateTypeAttr().Set("linear")
        cv.CreatePointsAttr(
            [Gf.Vec3f(float(p[0]), float(p[1]), float(p[2])) for p in off])
        cv.CreateCurveVertexCountsAttr([len(off)])
        cv.CreateWidthsAttr([LASER_WIDTH_M] * len(off))
        cv.SetWidthsInterpolation(UsdGeom.Tokens.vertex)
        UsdShade.MaterialBindingAPI(cv).Bind(mat)
        n += 1
    return n


def _setup_scene():
    """씬 열기 + 콜라이더 + 조명."""
    stage=get_current_stage()
    already=bool(stage and stage.GetPrimAtPath("/World/StationB").IsValid())
    if _RUNNING_IN_GUI and already:
        LOG("  GUI 현장 재사용")
    else:
        if not os.path.exists(SCENE_USD):
            raise FileNotFoundError(f"씬 없음: {SCENE_USD}")
        open_stage(SCENE_USD); stage=get_current_stage()

    try: world=World(stage_units_in_meters=1.)
    except Exception as e: LOG(f"  [경고] World: {e}"); world=None

    for p in stage.Traverse():
        if p.IsA(UsdGeom.Mesh):
            if not p.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI.Apply(p)

    _boost_lighting(stage)
    return stage, world


def _boost_lighting(stage):
    """현실적 조명: 비스듬한 방향광(요철 입체감) + 앰비언트 + 보조광 (원본 동일)."""
    try:
        results = []
        # 1) 방향성 광원 — 비스듬히 내리쬐어 결함 요철이 명암으로 드러남
        dpath = "/World/SunKey"
        if not stage.GetPrimAtPath(dpath).IsValid():
            dist = UsdLux.DistantLight.Define(stage, dpath)
            dist.CreateIntensityAttr(2500.0)
            dist.CreateColorAttr(Gf.Vec3f(1.0, 0.97, 0.92))
            dist.CreateAngleAttr(1.0)
            xf = UsdGeom.Xformable(dist.GetPrim())
            xf.ClearXformOpOrder()
            xf.AddRotateXYZOp().Set(Gf.Vec3f(-35.0, 0.0, 25.0))
            results.append("DistantLight")
        # 2) 앰비언트 dome
        path = "/World/AmbientFill"
        if not stage.GetPrimAtPath(path).IsValid():
            dome = UsdLux.DomeLight.Define(stage, path)
            dome.CreateIntensityAttr(400.0)
            dome.CreateColorAttr(Gf.Vec3f(0.85, 0.88, 0.95))
            results.append("DomeLight")
        # 3) 보조 방향광
        dpath2 = "/World/SunFill"
        if not stage.GetPrimAtPath(dpath2).IsValid():
            d2 = UsdLux.DistantLight.Define(stage, dpath2)
            d2.CreateIntensityAttr(900.0)
            d2.CreateColorAttr(Gf.Vec3f(0.9, 0.93, 1.0))
            d2.CreateAngleAttr(2.0)
            xf2 = UsdGeom.Xformable(d2.GetPrim())
            xf2.ClearXformOpOrder()
            xf2.AddRotateXYZOp().Set(Gf.Vec3f(-20.0, 0.0, -40.0))
            results.append("FillLight")
        LOG(f"  조명: {', '.join(results)} 설치")
    except Exception as e:
        LOG(f"  [경고] 조명: {e}")


def apply_sensor_response(rgb):
    """
    렌더된 RGB 를 프로파일의 센서 응답으로 바꾼다.

    개선 프로파일은 모노 센서 + 520nm 대역통과 필터다. Isaac 은 RGB 로
    렌더하므로, 그 조합이 실제로 무엇을 하는지 화소 단계에서 흉내 낸다.

      · 대역통과 필터  520nm 만 통과 → 렌더 이미지의 G 채널만 남는다.
        R·B 로 들어오는 배경광이 사라져 직사광 아래 SNR 이 올라간다.
      · 모노 센서      베이어 배열이 없으므로 G 를 전 화소에서 얻는다.
        컬러 센서였다면 G 화소가 절반뿐이고 디모자이크가 선 단면을
        뭉개, 서브픽셀 중심이 흔들린다.

    반환은 (H,W,3) 회색조로 맞춘다. 뒤쪽 선검출·세그멘테이션이 3채널을
    전제로 하기 때문이다. 세그멘테이션이 색을 잃는 것이 이 사양 변경의
    대가이며, 흑백 문맥 영상으로 돌려야 한다.

    PDF 원안 프로파일에서는 아무것도 하지 않고 그대로 돌려준다.
    """
    if rgb is None or CALIB.OPTICAL_FILTER_NM is None:
        return rgb
    g = np.asarray(rgb)[:, :, 1]
    return np.repeat(g[:, :, None], 3, axis=2)


def _setup_camera(stage):
    """
    검측 카메라 생성 — 실물 사양을 그대로 넣는다 (PDF 2.2).

    USD 카메라의 화각은 focal_length / horizontal_aperture 로 정해지므로,
    센서 실물 크기(2448·3.45µm = 8.4456mm)를 aperture 에 넣고 초점거리를
    8mm 로 두면 렌더된 이미지의 픽셀 좌표가 삼각측량식의 u, v 와 정확히
    같은 뜻을 갖는다.

        f_px = focal_length · resolution_x / horizontal_aperture = 2318.8

    이전에는 aperture 를 36mm 로 고정하고 초점거리를 51.15mm 로 역산했다.
    f_px 는 같았지만 f-stop·초점거리가 실물과 달라, 심도나 렌즈 효과를
    켜는 순간 사양과 다른 이미지가 나온다.
    """
    icp = CALIB.isaac_camera_params()
    W_px, H_px = icp["resolution"]

    UsdGeom.Xform.Define(stage, "/World/InspectionRig")
    cam = Camera(prim_path="/World/InspectionRig/InspectionCamera",
                 resolution=(W_px, H_px))
    cam.initialize()

    try:
        cp = stage.GetPrimAtPath("/World/InspectionRig/InspectionCamera")
        uc = UsdGeom.Camera(cp)
        uc.GetFocalLengthAttr().Set(icp["focal_length_mm"])
        uc.GetHorizontalApertureAttr().Set(icp["horizontal_aperture_mm"])
        uc.GetVerticalApertureAttr().Set(icp["vertical_aperture_mm"])
        uc.GetFStopAttr().Set(0.0)          # 0 = 핀홀. 심도 실험 시 2.0 으로
        uc.GetFocusDistanceAttr().Set(icp["focus_distance_m"])
        uc.GetClippingRangeAttr().Set(Gf.Vec2f(*icp["clipping_range_m"]))
    except Exception as e:
        LOG(f"  [경고] 카메라 파라미터: {e}")

    # ── 시맨틱 세그멘테이션 어노테이터 ──
    # Replicator 가 씬의 Semantics 라벨을 읽어 화소별 정답 마스크를 준다.
    # 사람이 라벨링할 필요 없이 세그멘테이션 정답을 얻는 경로이며,
    # 이것이 있어야 최종 오차를 검측식/선검출/세그멘테이션으로 분해할 수 있다.
    _attach_semantic_annotator(stage)
    return cam


_SEM_ANNOTATOR = None
_SEM_RENDER_PRODUCT = None


def _attach_semantic_annotator(stage):
    """semantic_segmentation 어노테이터를 검측 카메라에 붙인다."""
    global _SEM_ANNOTATOR, _SEM_RENDER_PRODUCT
    if _SEM_ANNOTATOR is not None:
        return _SEM_ANNOTATOR
    try:
        import omni.replicator.core as rep
        W_px, H_px = CAMERA_PARAMS["resolution"]
        _SEM_RENDER_PRODUCT = rep.create.render_product(
            "/World/InspectionRig/InspectionCamera", (W_px, H_px))
        _SEM_ANNOTATOR = rep.AnnotatorRegistry.get_annotator(
            "semantic_segmentation", init_params={"colorize": False})
        _SEM_ANNOTATOR.attach([_SEM_RENDER_PRODUCT])
        LOG("  시맨틱 어노테이터 부착 완료")
    except Exception as e:
        LOG(f"  [경고] 시맨틱 어노테이터 부착 실패: {e} "
            f"→ 세그멘테이션은 backend='geom' 폴백 사용")
        _SEM_ANNOTATOR = None
    return _SEM_ANNOTATOR


def _grab_semantic_mask():
    """
    현재 프레임의 정답 라벨맵과 id→라벨 사전을 가져온다.

    Returns
    -------
    (label_map (H,W) int, id_to_semantic {id: "class:wall"}) 또는 (None, None)
    """
    if _SEM_ANNOTATOR is None:
        return None, None
    try:
        d = _SEM_ANNOTATOR.get_data()
        lm = np.asarray(d["data"])
        if lm.ndim == 3:
            lm = lm[..., 0]
        info = d.get("info", {}) or {}
        id2l = info.get("idToLabels", {}) or {}
        return lm.astype(np.int32), id2l
    except Exception as e:
        LOG(f"  [경고] 시맨틱 마스크 획득 실패: {e}")
        return None, None


def _set_camera_xform(stage, R, cam_pos):
    """R 행렬로 카메라 자세 직접 세팅 (SetLookAt 미사용)."""
    cp=stage.GetPrimAtPath("/World/InspectionRig/InspectionCamera")
    M44=Gf.Matrix4d(
        float(R[0,0]),float(R[1,0]),float(R[2,0]),0.,
        float(R[0,1]),float(R[1,1]),float(R[2,1]),0.,
        float(R[0,2]),float(R[1,2]),float(R[2,2]),0.,
        float(cam_pos[0]),float(cam_pos[1]),float(cam_pos[2]),1.)
    xf=UsdGeom.Xformable(cp); xf.ClearXformOpOrder(); xf.AddTransformOp().Set(M44)


def _make_line_angles(n_v, n_h, fov_deg):
    """발사각 생성은 calibration 이 단일 출처다 (레이저 수렴각 포함)."""
    return CALIB.make_line_angles(n_v, n_h, fov_deg,
                                  GRID_PARAMS.get("laser_tilt_deg", 0.0))


# =====================================================================
# 핵심: 스테이션 촬영
# =====================================================================
def capture_station(stage, world, camera, line_angles,
                    station_name, cfg, gt_full,
                    baseline_m=None, standoff_m=None,
                    noise_sigma_px=0.0, output_dir=None,
                    use_diff_image=False):
    """
    한 스테이션(벽/바닥/패널)을 촬영하고 data.json + 이미지를 저장한다.

    Parameters
    ----------
    baseline_m    : float or None
        기선 거리(m). None이면 CAMERA_PARAMS["b_m"] 사용 (품질검측 기본값)
    standoff_m    : float or None
        측정 거리(m). None이면 cfg["standoff_m"] 사용 (품질검측 기본값)
    noise_sigma_px: float
        픽셀 Gaussian 노이즈 σ (실험용, 품질검측에서는 0.0)
    use_diff_image: bool
        [방안4] True면 차영상 모드. 레이저 ON 프레임과 OFF 프레임
        (조명만 켜진 배경)을 각각 캡처하여 A_선검출에 함께 전달한다.
        배경광이 상쇄되어 현장 직사광 환경에서 검출 강건성이 향상된다.

    흐름
    ----
    위치 계산 (laser_pos, cam_pos, R_laser)
    → IMU 기울기 주입 (R_tilted)
    → 카메라 xform 세팅 (R_tilted 직접)
    → Raycast → GT픽셀 + z_cam_m  [노이즈 주입]
    → rgb_raw 캡처
    → [차영상] 레이저 OFF 프레임 캡처 (조명 ON, 레이저 없음)
    → 발광 메시 렌더 → rgb_laser 캡처
    → [A] fn_detect(rgb_laser, laser_off=rgb_off) → lines_pixels
    → 선 필터링
    → IMU 픽셀 보정
    → data.json 저장
    → overlay.png 저장 (raycast 파랑 + 검출 초록)
    """
    cp = dict(CAMERA_PARAMS)
    # 선검출이 격자 예측에 쓰는 값. 없으면 A_선검출이 20선·42.61° 를
    # 가정하는데, 개선 프로파일은 V선이 40개라 예측이 통째로 어긋난다.
    cp.update({"n_v": GRID_PARAMS["n_vertical"],
               "n_h": GRID_PARAMS["n_horizontal"],
               "fov_h_deg": GRID_PARAMS["fov_deg"],
               "fov_v_deg": GRID_PARAMS["fov_deg"],
               "image_w": CAMERA_PARAMS["resolution"][0],
               "image_h": CAMERA_PARAMS["resolution"][1],
               "standoff_z": None})
    # 실험에서 baseline 오버라이드
    if baseline_m is not None:
        cp["b_m"] = float(baseline_m)
    f, b, cx, cy = cp["f_px"], cp["b_m"], cp["cx_px"], cp["cy_px"]

    # 실험에서 standoff 오버라이드
    _standoff = standoff_m if standoff_m is not None else cfg.get("standoff_m", 1.0)

    cp["standoff_z"] = float(_standoff)   # 시차항 −f·b/Z 에 쓰인다

    # ── 위치 계산 ──
    center = _world_center_of(stage, cfg["target"])
    if center is None:
        LOG(f"  [건너뜀] {station_name}: prim 없음"); return None
    center += np.asarray(cfg.get("center_offset_m",[0.,0.,0.]),float)
    normal = _norm(cfg.get("normal",[0.,-1.,0.]))
    up     = np.array([0.,0.,1.])
    if abs(np.dot(normal,up))>0.95: up=np.array([0.,1.,0.])

    laser_center = center + normal * _standoff
    view_l  = _norm(center-laser_center)

    # 혼합 장면(StationC)은 벽을 정면으로 겨누면 바닥이 화면에 안 들어온다.
    # pitch_down_deg 만큼 시선을 아래로 돌려 두 면을 한 프레임에 담는다.
    # 이때 각 면에 대한 정면 가정이 깨지므로, 검측은 반드시 중력(ĝ) 기준
    # 식(eq3 v2)으로 해야 한다.
    _pitch = float(cfg.get("pitch_down_deg", 0.0))
    if abs(_pitch) > 1e-6:
        _ax = _norm(np.cross(view_l, up))          # 시선 기준 우측 축
        _th = np.radians(_pitch)
        _K  = np.array([[0., -_ax[2], _ax[1]],
                        [_ax[2], 0., -_ax[0]],
                        [-_ax[1], _ax[0], 0.]])
        _R  = np.eye(3) + np.sin(_th)*_K + (1-np.cos(_th))*(_K@_K)  # 로드리게스
        view_l = _norm(_R @ view_l)
        LOG(f"  촬영 자세: 아래로 {_pitch}° 숙임 (벽+바닥 동시 촬영)")
    right_l = _norm(np.cross(view_l, up))   # SetLookAt과 동일한 right 방향
    up_l    = _norm(np.cross(view_l,right_l))
    R_laser  = np.column_stack([right_l,up_l,view_l])
    laser_pos= laser_center.copy()
    cam_pos  = laser_center + right_l*b
    R_ideal  = R_laser.copy()

    # IMU (현재는 이상적 자세로 촬영, 추후 활성화)
    imu_data = {"injected_pitch_deg":0., "injected_roll_deg":0.,
                "measured_pitch_deg":0., "measured_roll_deg":0.}

    # ── 카메라 xform 세팅: SetLookAt ──
    # Step1: lookat_up 결정 (view_l과 평행하면 특이점 방지)
    lookat_up = up.copy()
    if abs(np.dot(view_l, lookat_up)) > 0.9:
        lookat_up = np.array([0., 1., 0.])
        if abs(np.dot(view_l, lookat_up)) > 0.9:
            lookat_up = np.array([1., 0., 0.])

    # Step2: right_l을 SetLookAt과 완전히 일치하도록 재계산
    # SetLookAt 내부: right = cross(forward, up_hint) 후 정규화
    right_l = _norm(np.cross(view_l, lookat_up))
    up_l    = _norm(np.cross(view_l, right_l))
    R  = np.column_stack([right_l, up_l, view_l])
    Rt = R.T

    # Step3: cam_pos, tgt_pos를 재계산된 right_l로 확정
    cam_pos = laser_center + right_l * b
    tgt_pos = cam_pos + view_l   # 카메라 정면 (레이저와 평행)

    cam_prim = stage.GetPrimAtPath("/World/InspectionRig/InspectionCamera")
    M = Gf.Matrix4d().SetLookAt(
        Gf.Vec3d(float(cam_pos[0]), float(cam_pos[1]), float(cam_pos[2])),
        Gf.Vec3d(float(tgt_pos[0]), float(tgt_pos[1]), float(tgt_pos[2])),
        Gf.Vec3d(float(lookat_up[0]), float(lookat_up[1]), float(lookat_up[2]))
    ).GetInverse()
    xf = UsdGeom.Xformable(cam_prim)
    xf.ClearXformOpOrder()
    xf.AddTransformOp().Set(M)
    _wait(world, 15)

    # ── Raycast → GT 픽셀 + z_cam_m ──
    query   = get_physx_scene_query_interface()
    fov     = np.radians(GRID_PARAMS["fov_deg"])
    samples = np.linspace(-fov/2,fov/2,GRID_PARAMS["samples_per_line"])
    lines_pixels_raycast={};  ground_truth={};  lines_world={}
    n_hit=0

    for lid,info in line_angles.items():
        px_pts,gt_pts,w_pts=[],[],[]
        for s in samples:
            alpha = info["angle_rad"] if info["fixed"]=="alpha" else s
            beta  = info["angle_rad"] if info["fixed"]=="beta"  else s
            d_local=_norm([np.tan(alpha),np.tan(beta),1.])
            d_world=R_laser@d_local
            hit=query.raycast_closest(tuple(laser_pos),tuple(d_world),30.)
            if hit and hit["hit"]:
                P=np.array(hit["position"])
                p_cam=Rt@(P-cam_pos); Z=p_cam[2]
                if Z>1e-4:
                    u_i=f*p_cam[0]/Z+cx; v_i=f*p_cam[1]/Z+cy
                    # 실험용 노이즈 (품질검측에서는 noise_sigma_px=0.0)
                    if noise_sigma_px > 0:
                        u=u_i+float(np.random.normal(0,noise_sigma_px))
                        v=v_i+float(np.random.normal(0,noise_sigma_px))
                    else:
                        u,v=u_i,v_i
                    px_pts.append([float(u),float(v)])
                    gt_pts.append({"xyz":P.tolist(),
                                   "u_px":float(u),"v_px":float(v),
                                   "distance_m":float(np.linalg.norm(P-cam_pos)),
                                   "z_cam_m":float(Z)})
                    w_pts.append(P); n_hit+=1
        lines_pixels_raycast[lid]=px_pts
        ground_truth[lid]=gt_pts
        lines_world[lid]=w_pts

    if n_hit==0:
        LOG(f"  [건너뜀] {station_name}: raycast 적중 없음"); return None

    # ── rgb_raw 캡처 ──
    _wait(world,20)
    try:
        rgba=camera.get_rgba()
        rgb_raw=apply_sensor_response(rgba[:,:,:3]) if rgba is not None else None
    except: rgb_raw=None

    # ── [방안4] 레이저 OFF 프레임 (차영상용) ──
    # rgb_raw가 이미 "조명 ON + 레이저 없음" 상태이므로 그대로 OFF 프레임으로
    # 사용한다. 차영상 모드에서는 아래 레이저 렌더도 조명을 켠 채 수행하여
    # ON/OFF 두 프레임의 조명 조건을 일치시킨다.
    # rgb_off 는 차영상뿐 아니라 **세그멘테이션 입력**으로도 쓰이므로 항상 남긴다.
    # 레이저 ON 프레임은 초록 격자선이 화면을 덮어 세그멘테이션 모델이 선을
    # 물체 경계로 오인한다. 두 프레임은 수십 µs 간격이라 마스크가 픽셀 단위로
    # 그대로 정합되므로, 추가 촬영 없이 문제가 사라진다.
    rgb_off = rgb_raw

    # ── 발광 메시 렌더 → rgb_laser ──
    # 일반 모드: 조명을 끄고 레이저만 촬영 (암실 모사, 배경광 없음)
    # [방안4] 차영상 모드: 조명을 켠 채로 레이저 촬영.
    #   ON(조명+레이저) − OFF(조명) 차분 시 배경광이 상쇄되어야 하므로
    #   두 프레임의 조명 조건이 반드시 동일해야 한다.
    rgb_laser=None
    try:
        if use_diff_image:
            # 차영상: 조명 유지 (OFF 프레임과 동일 조명 조건)
            _wait(world, 5)
        else:
            # 일반: 조명 강도를 0으로 (암실 모사)
            _set_lighting_intensity(stage, 0.0)
            _wait(world, 5)

        ng=_make_emissive_grid(stage,lines_world,normal)
        LOG(f"  발광 레이저 {ng}선"); _wait(world,60)
        rgba2=camera.get_rgba()
        rgb_laser=apply_sensor_response(rgba2[:,:,:3]) if rgba2 is not None else None
        if stage.GetPrimAtPath("/World/LaserGrid").IsValid():
            stage.RemovePrim("/World/LaserGrid")

        # 조명 복원 (일반 모드에서 껐을 경우)
        if not use_diff_image:
            _set_lighting_intensity(stage, 1.0)
            _wait(world, 5)
    except Exception as e:
        LOG(f"  [경고] 렌더: {e}")
        _set_lighting_intensity(stage, 1.0)

    # ── [A] 선검출 ──
    if fn_detect is None:
        _init_algorithms()
    # [방안4] 차영상 모드면 OFF 프레임 함께 전달 (배경광 제거)
    # 다중면 장면(inspect=="auto")에서는 A_선검출의 단일 평면 가정을 끈다.
    # 격자 동시 최적화·간격 기반 이상선 판정은 "화면 전체가 같은 거리의 한
    # 평면"이라는 전제 위에 서 있어, 벽+바닥+동바리가 섞이면 보정이 아니라
    # 실제 기하의 훼손이 된다.
    _multi = (cfg.get("inspect") == "auto") or bool(cfg.get("segmentation"))
    if rgb_laser is not None:
        _kw = {}
        if use_diff_image and rgb_off is not None:
            _kw["laser_off_image"] = rgb_off
        if _multi:
            _kw["multi_surface"] = True
        try:
            lines_pixels_detected = fn_detect(rgb_laser, lines_pixels_raycast,
                                              line_angles, cp, **_kw)
        except TypeError:
            # 구버전 A_선검출 (신규 인자 미지원) 폴백
            if _multi:
                LOG("  [경고] A_선검출이 multi_surface 를 지원하지 않습니다. "
                    "다중면 장면에서 격자 동시 최적화가 기하를 훼손할 수 있습니다.")
            lines_pixels_detected = fn_detect(rgb_laser,
                                              lines_pixels_raycast,
                                              line_angles, cp)
    else:
        lines_pixels_detected = lines_pixels_raycast

    # ── 선 필터링 ──
    valid_V,valid_H,rejected=_filter_lines(lines_pixels_detected,cp)
    valid_ids=set(valid_V+valid_H)
    lines_filtered={k:v for k,v in lines_pixels_detected.items() if k in valid_ids}
    gt_filtered   ={k:v for k,v in ground_truth.items()          if k in valid_ids}
    if not valid_V:
        LOG(f"  [경고] 유효 V선 없음"); return None

    # ── IMU 픽셀 보정 ──
    lines_corrected=_correct_imu(lines_filtered,imu_data,cp)

    # ── 시맨틱 정답 마스크 (세그멘테이션 기준선) ──
    seg_label_map, seg_id_to_label = (None, None)
    if cfg.get("segmentation") or cfg.get("inspect") == "auto":
        seg_label_map, seg_id_to_label = _grab_semantic_mask()
        if seg_label_map is not None:
            LOG(f"  시맨틱 마스크 {seg_label_map.shape} "
                f"클래스 {sorted(set(str(v) for v in seg_id_to_label.values()))}")

    # ── GT 매핑 ──
    gt_station_key="StationA" if station_name.startswith("StationA") else station_name
    gt_station=gt_full.get("stations",{}).get(gt_station_key,{})
    inspect=cfg.get("inspect","flatness")
    surfaces=gt_station.get("surfaces",{})
    gt_tilt=None
    if   inspect=="verticality":   gt_tilt=surfaces.get("WallBack",{}).get("signal_tilt_deg")
    elif inspect=="horizontality": gt_tilt=surfaces.get("Floor",{}).get("signal_tilt_deg")
    sgt=(gt_station if inspect=="flatness"
         else {"role":inspect,"gt_tilt_deg":gt_tilt,"surface":surfaces})

    # ── data.json 저장 ──
    out={
        "station": station_name, "inspect": inspect, "scene_usd": SCENE_USD,
        "camera_world":{
            "position_m":cam_pos.tolist(),
            "laser_position_m":laser_pos.tolist(),
            "wall_center_m":center.tolist(),
            "quaternion_wxyz":_rotmat_to_quat(R).tolist(),
            "forward_dir":R[:,2].tolist(),
            "standoff_m":float(_standoff),
            "baseline_m":float(b),
            "normal":normal.tolist(),
        },
        "device_tilt":  _device_tilt_from_R(R),
        "imu":          imu_data,
        "camera_params": cp,
        "line_angles":   line_angles,
        "lines_pixels":  lines_corrected,       # IMU 보정 후 → eq 검증 입력
        "lines_pixels_raw": lines_pixels_raycast,  # raycast 참조
        "ground_truth":  gt_filtered,
        "scene_ground_truth": sgt,
        "segmentation": ({"available": True,
                          "id_to_semantic": {str(k): (v if isinstance(v, str)
                                                      else v.get("class", ""))
                                             for k, v in seg_id_to_label.items()},
                          "label_map_png": "semantic_label.png"}
                         if seg_label_map is not None else {"available": False}),
        "quality":{"rays_hit":n_hit,
                   "rays_total":len(line_angles)*GRID_PARAMS["samples_per_line"],
                   "valid_V":len(valid_V),"rejected":len(rejected)},
    }
    _out_base = output_dir if output_dir is not None else OUTPUT_DIR
    out_dir=os.path.join(_out_base,station_name)
    os.makedirs(out_dir,exist_ok=True)
    with open(os.path.join(out_dir,"data.json"),"w",encoding="utf-8") as fp:
        json.dump(out,fp,ensure_ascii=False,indent=2)

    # ── 이미지 저장 ──
    _save_images(out_dir, rgb_raw, rgb_laser,
                 lines_corrected, lines_pixels_raycast,
                 rgb_off=rgb_off)

    # 라벨맵은 16bit PNG 로 저장 (클래스 id 를 무손실 보존)
    if seg_label_map is not None:
        try:
            Image.fromarray(seg_label_map.astype(np.uint16)).save(
                os.path.join(out_dir, "semantic_label.png"))
        except Exception as e:
            LOG(f"  [경고] 라벨맵 저장: {e}")

    # ── inspect=="auto" 이면 영역별 검측을 바로 수행 ──
    if cfg.get("inspect") == "auto" and SEGMENTATION.get("enabled"):
        try:
            out["region_inspection"] = _run_region_inspection(
                lines_corrected, cp, R, seg_label_map, seg_id_to_label,
                rgb_off, out_dir)
        except Exception as e:
            LOG(f"  [경고] 영역별 검측 실패: {e}")

    LOG(f"  {station_name}: 적중 {n_hit}  유효V {len(valid_V)}  "
        f"기각 {len(rejected)}  → {out_dir}")
    return out


def _save_images(out_dir, rgb_raw, rgb_laser,
                 lines_detected, lines_raycast, rgb_off=None):
    """rgb_raw / rgb_laser / rgb_off / overlay 저장."""
    try:
        if rgb_raw is not None:
            Image.fromarray(rgb_raw).save(os.path.join(out_dir,"rgb_raw.png"))
        if rgb_laser is not None:
            Image.fromarray(rgb_laser).save(os.path.join(out_dir,"rgb_laser.png"))
        # [방안4] 차영상 OFF 프레임 저장 (있을 때만)
        if rgb_off is not None:
            Image.fromarray(rgb_off).save(os.path.join(out_dir,"rgb_off.png"))

        base=rgb_laser if rgb_laser is not None else rgb_raw
        if base is None: return
        im=Image.fromarray(base).convert("RGB")
        W,H=im.size; dr=ImageDraw.Draw(im)
        for lid,pts in lines_raycast.items():
            xy=[(float(p[0]),float(p[1])) for p in pts
                if -10<=p[0]<W+10 and -10<=p[1]<H+10]
            if len(xy)>=2: dr.line(xy,fill=(0,80,255),width=2)   # 파랑: raycast 참값
        for lid,pts in lines_detected.items():
            xy=[(float(p[0]),float(p[1])) for p in pts
                if -10<=p[0]<W+10 and -10<=p[1]<H+10]
            if len(xy)>=2: dr.line(xy,fill=(40,230,70),width=2)  # 초록: A 검출
        im.save(os.path.join(out_dir,"overlay.png"))
    except Exception as e:
        LOG(f"  [경고] 이미지저장: {e}")


# =====================================================================
# 영역별 검측 (세그멘테이션 기반)
# =====================================================================
def _run_region_inspection(lines_pixels, cp, R, label_map, id_to_semantic,
                           rgb_off, out_dir):
    """
    한 장의 촬영 결과를 영역별로 검측한다.

    실제 계산은 pipeline_region 에 있다(Isaac 비의존이라 오프라인에서도
    같은 코드로 검증된다). 여기서는 입력을 넘기고 결과를 저장만 한다.

    label_map 이 없으면(어노테이터 미부착 등) 기하 전용 백엔드로 폴백한다.
    """
    mod = _load_module_cached("pipeline_region")
    pr, fmt = mod.inspect_capture, mod.format_report

    backend = SEGMENTATION.get("backend", "gt")
    if label_map is None and backend == "gt":
        LOG("  [폴백] 시맨틱 마스크 없음 → backend='geom'")
        backend = "geom"

    line_angles = _make_line_angles(GRID_PARAMS["n_vertical"],
                                    GRID_PARAMS["n_horizontal"],
                                    GRID_PARAMS["fov_deg"])
    res = pr(lines_pixels, line_angles, cp, R,
             label_map=label_map, id_to_semantic=id_to_semantic,
             rgb_off=rgb_off, backend=backend,
             erode_default_px=SEGMENTATION.get("erode_px", 3),
             erode_thin_px=SEGMENTATION.get("erode_thin_px", 1),
             sigma_u_px=SEGMENTATION.get("sigma_u_px", 0.2),
             target_sigma_mm=SEGMENTATION.get("target_sigma_mm", 2.0))

    LOG("\n" + fmt(res))
    try:
        with open(os.path.join(out_dir, "region_inspection.json"),
                  "w", encoding="utf-8") as fp:
            json.dump(_jsonable(res), fp, ensure_ascii=False, indent=2)
    except Exception as e:
        LOG(f"  [경고] 영역검측 결과 저장: {e}")
    return _jsonable(res)


def _jsonable(o):
    """numpy 타입을 json 이 받는 형태로 바꾼다."""
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


# =====================================================================
# 메인
# =====================================================================
def main():
    _init_algorithms()
    LOG("="*60)
    LOG("품질검측 파이프라인 시작")
    LOG("="*60)

    gt_full={}
    if os.path.exists(GT_JSON):
        with open(GT_JSON,encoding="utf-8") as fp: gt_full=json.load(fp)
    else:
        LOG(f"[경고] GT 없음: {GT_JSON}")

    stage,world=_setup_scene()
    try: world.reset()
    except Exception: pass

    os.makedirs(OUTPUT_DIR,exist_ok=True)
    from PIL import Image as _PIL
    tex_path=os.path.join(OUTPUT_DIR,"_grid_tex.png")
    img=np.zeros((2048,2048,3),dtype=np.uint8)
    nv,nh=GRID_PARAMS["n_vertical"],GRID_PARAMS["n_horizontal"]
    for i in range(nv): x=int(2048*(i+.5)/nv); img[:,max(0,x-3):x+3]=255
    for j in range(nh): y=int(2048*(j+.5)/nh); img[max(0,y-3):y+3,:]=255
    _PIL.fromarray(img).save(tex_path)

    camera=_setup_camera(stage)
    line_angles=_make_line_angles(GRID_PARAMS["n_vertical"],
                                  GRID_PARAMS["n_horizontal"],
                                  GRID_PARAMS["fov_deg"])

    _wait(world,30)

    results={}
    for name,cfg in STATIONS.items():
        LOG(f"\n── {name} ──")
        r=capture_station(stage,world,camera,line_angles,name,cfg,gt_full)
        if r: results[name]={"hit":r["quality"]["rays_hit"]}

    with open(os.path.join(OUTPUT_DIR,"metadata.json"),"w",encoding="utf-8") as fp:
        json.dump({"scene":SCENE_USD,"camera_params":CAMERA_PARAMS,
                   "stations":list(STATIONS.keys()),"results":results},
                  fp,ensure_ascii=False,indent=2)

    LOG(f"\n완료 → {OUTPUT_DIR}")
    LOG("다음: python3 3_pipeline_eq_verify.py <data.json> [모드]")
    LOG("      영역별 검측 결과는 각 스테이션의 region_inspection.json 참고")


if __name__=="__main__":
    main()
