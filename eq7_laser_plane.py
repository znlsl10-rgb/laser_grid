#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
[식 ⑦] 레이저 평면 기하 — V선·H선을 하나의 식으로
========================================================================
eq1 의 삼각측량식은 V선(발사각 α 고정) 전용이다.

    Z = f·b / (f·tanα − (u − c_x))

이 식은 "레이저 평면이 Y축을 품는다" 는 가정이 박혀 있어 H선에는 쓸 수
없다. 그래서 지금까지 H선은 검출은 하면서도 삼각측량에서 버려졌다.

여기서는 같은 기하를 **평면 법선** 하나로 일반화한다. DOE 가 만드는 선은
V든 H든 전부 "레이저 원점을 지나는 평면" 이므로, 그 평면의 법선 n 만
알면 같은 식으로 풀린다.

  레이저 원점 O=0, 카메라 C=(b,0,0), 카메라 축은 레이저 축과 평행.
  화소 (u,v) 의 정규 좌표를 û=(u−c_x)/f, v̂=(v−c_y)/f 라 하면

      P = C + Z·(û, v̂, 1) = (b + Z·û,  Z·v̂,  Z)

  P 가 레이저 평면 위에 있다는 조건 n·P = 0 을 Z 에 대해 풀면

      ┌─────────────────────────────────────┐
      │   Z = − n_x · b / (n_x·û + n_y·v̂ + n_z)   │
      └─────────────────────────────────────┘

  V선(회전 없음)에서 n = (cosα, 0, −sinα) 를 넣으면 eq1 과 정확히 같다.

【깊이 이득 g — 이 선이 깊이를 얼마나 잘 주는가】
  화소를 선의 법선 방향으로 1px 밀었을 때 깊이가 얼마나 흔들리는가를
  이상적인 V선 대비 배수로 나타낸 값이다. 위 식을 미분하면

      ∇Z = (Z²/(f·b)) · (1, n_y/n_x)          (화소 u,v 에 대한 기울기)

  이고, 이미지에서 이 선의 법선 방향은 m̂ = (n_x, n_y)/‖(n_x,n_y)‖ 이므로

      ┌───────────────────────────────────┐
      │   g = √(n_x² + n_y²) / |n_x|           │
      │   σ_Z = g · σ_perp · Z² / (f·b)        │
      └───────────────────────────────────┘

  · g = 1  : 이상적인 V선 (n_y = 0). eq1 의 감도 그대로.
  · g = ∞  : n_x = 0. 평면이 기선(X축)을 품어 시차가 선과 나란해진다.
             회전 없는 H선이 바로 이 경우 — **깊이 정보가 원리적으로 없다**.
             정답 데이터에서도 H선의 v 는 1500점에 걸쳐 0.0004px 밖에
             안 움직인다(검출 잡음 0.26px 의 1/650). 검출을 아무리 잘해도
             깊이가 안 나오는 이유가 이것이다.

【H선을 살리는 법 — 격자를 광축으로 굴린다(roll γ)】
  DOE 를 광축(Z) 둘레로 γ 만큼 굴리면 H평면의 법선에 x 성분이 생긴다.

      V선 : g = 1 / cos γ          H선 : g = 1 / sin γ

  γ=0    → V 1.00 / H  ∞     (현 하드웨어. H선은 깊이에 못 쓴다)
  γ=6.18°→ V 1.01 / H 9.29   (수렴각만으로는 부족)
  γ=45°  → V 1.41 / H 1.41   (두 계열이 대칭. 깊이 표본이 2배가 된다)

  γ=45° 에서 점당 깊이잡음은 √2 배 나빠지지만 표본이 2배라 평면 적합의
  평균 정밀도는 본전이고, **수평 방향으로도 깊이 표본이 생긴다**. 평활도는
  가로로 놓은 직선자 아래의 요철을 보는 검측이라 이 방향 표본이 있어야
  한다. 수직도·수평도도 면을 한 방향 줄무늬가 아니라 격자로 덮게 되어
  적합이 안정된다. 이것이 "가로축도 인식되어야 검측이 정확해진다" 의
  기하학적 내용이다.

  주의 — γ 는 **광축 둘레 회전(roll)** 이지 수렴각(tilt, Y축 둘레)이 아니다.
  수렴각 δ 로도 x 성분이 조금 생기지만 g = √(sin²δ·sin²β + cos²β)/(sinδ·|sinβ|)
  이라 중앙 H선(β=0)에서 발산하고 가장자리에서도 16배가 넘는다. 깊이용으로는
  쓸 수 없다.
========================================================================
"""
import numpy as np

# 깊이 이득이 이 값을 넘으면 삼각측량에 쓰지 않는다.
# 3.0 은 "이상적인 V선의 3배까지는 감수한다" 는 뜻이다. γ=45°(g=1.41)는
# 통과하고, 수렴각만 준 H선(g=9.3)은 막힌다.
MAX_DEPTH_GAIN = 3.0


def _rot_y(rad):
    c, s = np.cos(rad), np.sin(rad)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _rot_z(rad):
    c, s = np.cos(rad), np.sin(rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def plane_normal(fixed, angle_rad, tilt_rad=0.0, roll_rad=0.0):
    """
    선 하나의 레이저 평면 법선 (레이저 좌표계, 단위벡터).

    Parameters
    ----------
    fixed : "alpha" | "beta"
        V선은 α 고정, H선은 β 고정.
    angle_rad : float
        그 선의 발사각.
    tilt_rad : float
        수렴각 δ — 레이저를 Y축 둘레로 돌린 값. 기선 시차를 상쇄해
        격자를 화면 안으로 넣는 용도다 (calibration.LASER_TILT_DEG).
    roll_rad : float
        격자 회전 γ — 광축(Z) 둘레. H선에 깊이를 주는 유일한 회전이다
        (calibration.LASER_ROLL_DEG).

    Notes
    -----
    V평면은 Y축과 방향 (sinα,0,cosα) 를 품으므로 법선은 (cosα, 0, −sinα).
    H평면은 X축과 방향 (0,sinβ,cosβ) 를 품으므로 법선은 (0, −cosβ, sinβ).
    회전은 수렴각 → 격자회전 순으로 적용한다(장비에서 DOE 를 얹은 뒤
    모듈 전체를 돌리는 순서와 같다).
    """
    a = float(angle_rad)
    if fixed == "alpha":
        n = np.array([np.cos(a), 0.0, -np.sin(a)])
    elif fixed == "beta":
        n = np.array([0.0, -np.cos(a), np.sin(a)])
    else:
        raise ValueError(f"fixed 는 'alpha' 또는 'beta' 여야 한다: {fixed!r}")
    if tilt_rad:
        n = _rot_y(float(tilt_rad)) @ n
    if roll_rad:
        n = _rot_z(float(roll_rad)) @ n
    nrm = float(np.linalg.norm(n))
    return n / nrm if nrm > 0 else n


def depth_gain(normal):
    """
    이 평면이 주는 깊이 잡음 배수 g = √(n_x²+n_y²)/|n_x|.

    이상적인 V선을 1 로 두는 무차원 값이다. n_x=0 이면 inf 를 돌려준다
    (깊이 정보 없음). 수치미분과 소수점 아래 12자리까지 일치함을 확인했다.
    """
    n = np.asarray(normal, dtype=float)
    nx = abs(float(n[0]))
    lat = float(np.hypot(n[0], n[1]))
    if nx < 1e-12:
        return float("inf")
    return lat / nx


def triangulate_plane(uv, normal, f, b, cx, cy, z_min=0.05, z_max=60.0):
    """
    한 선의 화소점들을 3D 로 되돌린다 (벡터화).

    Parameters
    ----------
    uv : (N,2) array_like
    normal : (3,) 레이저 평면 법선
    f, b, cx, cy : 카메라 파라미터 (f·c 는 px, b 는 m)
    z_min, z_max : 물리적으로 말이 되는 깊이 범위 [m].
        분모가 0 근처면 Z 가 발산하는데, 그 점은 시차가 풀리지 않은
        것이지 아주 먼 곳에 있는 것이 아니다. 걸러내지 않으면 평면
        적합이 통째로 끌려간다.

    Returns
    -------
    xyz : (M,3) ndarray
    keep : (N,) bool ndarray — 어느 입력점이 살아남았는지
    """
    P = np.asarray(uv, dtype=float).reshape(-1, 2)
    n = np.asarray(normal, dtype=float)
    uh = (P[:, 0] - cx) / f
    vh = (P[:, 1] - cy) / f
    D = n[0] * uh + n[1] * vh + n[2]
    with np.errstate(divide="ignore", invalid="ignore"):
        Z = -n[0] * b / D
    keep = np.isfinite(Z) & (Z >= z_min) & (Z <= z_max)
    Zk = Z[keep]
    xyz = np.column_stack([b + Zk * uh[keep], Zk * vh[keep], Zk])
    return xyz, keep


def predicted_uv(normal, z_m, f, b, cx, cy, along, n_samples=None,
                 img_w=None, img_h=None):
    """
    거리 z_m 의 정면 평면에 이 선이 맺히는 위치 — 추적 밴드의 중심선.

    n_x·û + n_y·v̂ + n_z + n_x·b/Z = 0 을 한쪽 좌표에 대해 푼다. 회전이
    없으면 V선은 u=f·tanα−f·b/Z+c_x 로, H선은 v=f·tanβ+c_y 로 정확히
    돌아간다(calibration.predicted_u 와 동일).

    along : "v" | "u"
        스캔축. "v" 면 각 행에서의 u 를, "u" 면 각 열에서의 v 를 돌려준다.
    """
    n = np.asarray(normal, dtype=float)
    K = n[2] + n[0] * b / float(z_m)
    if along == "v":
        if abs(n[0]) < 1e-12:
            return None
        vs = np.arange(int(img_h if n_samples is None else n_samples),
                       dtype=float)
        vh = (vs - cy) / f
        uh = -(n[1] * vh + K) / n[0]
        return np.column_stack([uh * f + cx, vs])
    if along == "u":
        if abs(n[1]) < 1e-12:
            return None
        us = np.arange(int(img_w if n_samples is None else n_samples),
                       dtype=float)
        uh = (us - cx) / f
        vh = -(n[0] * uh + K) / n[1]
        return np.column_stack([us, vh * f + cy])
    raise ValueError("along 은 'v' 또는 'u'")


def line_planes(line_angles, tilt_rad=0.0, roll_rad=0.0):
    """
    {lid: {"fixed","angle_rad"}} → {lid: {"normal", "gain", "usable"}}.

    line_angles 항목에 "normal" 이 이미 있으면 그대로 쓴다. 실장비
    캘리브레이션은 발사각이 아니라 평면 자체를 재는 편이 정확하므로
    (평면 하나에 3자유도, 각도 모델보다 가정이 적다) 그 경로를 열어둔다.
    """
    out = {}
    for lid, info in line_angles.items():
        n = info.get("normal")
        if n is None:
            n = plane_normal(info.get("fixed", "alpha"),
                             float(info["angle_rad"]),
                             tilt_rad=tilt_rad, roll_rad=roll_rad)
        n = np.asarray(n, dtype=float)
        g = depth_gain(n)
        out[lid] = {"normal": n, "gain": g,
                    "usable": bool(np.isfinite(g) and g <= MAX_DEPTH_GAIN),
                    "fixed": info.get("fixed")}
    return out


def family_summary(planes):
    """V/H 계열별 깊이 이득 요약 — 조서와 터미널 출력용."""
    out = {}
    for pre, name in (("V", "V선(세로)"), ("H", "H선(가로)")):
        g = [p["gain"] for lid, p in planes.items() if lid.startswith(pre)]
        if not g:
            continue
        fin = [x for x in g if np.isfinite(x)]
        out[pre] = {
            "이름": name, "선 수": len(g),
            "깊이가능": sum(1 for lid, p in planes.items()
                          if lid.startswith(pre) and p["usable"]),
            "이득_중앙": (round(float(np.median(fin)), 3) if fin else None),
            "이득_최대": (round(float(np.max(fin)), 3) if fin else None),
            "무한대": len(g) - len(fin)}
    return out


# ============ 자체 검증 ============
if __name__ == "__main__":
    print("[식 ⑦] 레이저 평면 일반화 검증")
    import importlib.util as _ilu, os
    _s = _ilu.spec_from_file_location(
        "eq1", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "eq1_triangulation.py"))
    _EQ1 = _ilu.module_from_spec(_s); _s.loader.exec_module(_EQ1)

    f, b, cx, cy = 1593.0, 0.150, 1224.0, 1024.0
    rng = np.random.default_rng(11)

    # 1) V선에서 eq1 과 완전히 일치하는가
    worst = 0.0
    for _ in range(2000):
        al = np.radians(rng.uniform(-30, 30))
        Z = rng.uniform(0.8, 6.0)
        Xl = Z * np.tan(al); Yl = Z * rng.uniform(-0.5, 0.5)
        u = f * (Xl - b) / Z + cx; v = f * Yl / Z + cy
        n = plane_normal("alpha", al)
        xyz, keep = triangulate_plane([[u, v]], n, f, b, cx, cy)
        X1, Y1, Z1 = _EQ1.triangulate_point(u, v, al, 0.0, f, b, cx, cy)
        worst = max(worst, float(np.max(np.abs(xyz[0] - [X1, Y1, Z1]))))
    print(f"  eq1 대조 (V선 2000점)     최대차 {worst*1e9:.3f} nm"
          f"   {'PASS' if worst < 1e-9 else 'FAIL'}")

    # 2) H선도 굴리면 풀리는가 — 정답 3D 를 만들어 되돌려 본다
    print("  H선 복원 오차 (정답 3D → 화소 → 삼각측량)")
    for gdeg in (0.0, 6.18, 20.0, 45.0):
        g = np.radians(gdeg)
        be = np.radians(20.0)
        n = plane_normal("beta", be, roll_rad=g)
        # 이 평면 위의 점을 만든다: 평면 위 임의 방향 두 개로 전개
        e1 = np.cross(n, [0, 0, 1.0]); e1 /= np.linalg.norm(e1)
        e2 = np.cross(n, e1)
        if e2[2] < 0:                      # 카메라 앞쪽(+Z)을 향하게 둔다
            e2 = -e2
        s = rng.uniform(-1, 1, 400); t = rng.uniform(0.5, 3.0, 400)
        P = s[:, None] * e1 + t[:, None] * e2
        P = P[P[:, 2] > 0.5]
        u = f * (P[:, 0] - b) / P[:, 2] + cx
        v = f * P[:, 1] / P[:, 2] + cy
        xyz, keep = triangulate_plane(np.column_stack([u, v]), n, f, b, cx, cy)
        err = (np.abs(xyz - P[keep]).max() * 1000.0
               if keep.any() else float("nan"))
        print(f"    roll {gdeg:5.2f}°   g = {depth_gain(n):7.3f}"
              f"   복원오차 {err:9.6f} mm   "
              f"{'사용가능' if depth_gain(n) <= MAX_DEPTH_GAIN else '사용불가'}")

    # 3) 이득이 실제 잡음 증폭과 맞는가 (몬테카를로)
    print("  이득 g 검증 — 화소에 σ=0.3px 를 주고 깊이 산포를 잰다")
    Zc = 2.7
    for label, fixed, ang, gd in (("V roll 0°", "alpha", 10.0, 0.0),
                                  ("V roll45°", "alpha", 10.0, 45.0),
                                  ("H roll45°", "beta", 20.0, 45.0),
                                  ("H roll20°", "beta", 20.0, 20.0)):
        n = plane_normal(fixed, np.radians(ang), roll_rad=np.radians(gd))
        m = np.array([n[0], n[1]]); m = m / np.linalg.norm(m)
        # 깊이 Zc 에 있는 이 평면 위의 한 점
        uh0 = 0.0 if abs(n[1]) < 1e-9 else 0.05
        K = n[2] + n[0] * b / Zc
        if abs(n[1]) > 1e-9:
            vh0 = -(n[0] * uh0 + K) / n[1]
        else:
            uh0 = -K / n[0]; vh0 = 0.02
        u0, v0 = cx + f * uh0, cy + f * vh0
        e = rng.normal(0, 0.3, 20000)
        uv = np.column_stack([u0 + e * m[0], v0 + e * m[1]])
        xyz, keep = triangulate_plane(uv, n, f, b, cx, cy)
        sig = float(np.std(xyz[:, 2])) * 1000.0
        pred = depth_gain(n) * 0.3 * Zc ** 2 / (f * b) * 1000.0
        print(f"    {label}   실측 σ_Z {sig:6.3f} mm   예측 {pred:6.3f} mm"
              f"   {'PASS' if abs(sig - pred) < 0.05 * pred else 'FAIL'}")
