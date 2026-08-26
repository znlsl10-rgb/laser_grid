"""
[식 ①] 거리 측정 — 능동 삼각측량
====================================
Z = f·b / [ f·tan(α) - (u - c_x) ]
X = (u - c_x)·Z/f + b
Y = (v - c_y)·Z/f

PDF 검증식 (오차 0 mm 확인 완료)
"""
import numpy as np


def triangulate_point(u, v, alpha, beta, f, b, cx, cy):
    """
    단일 격자점의 3D 좌표 산출.
    
    Parameters
    ----------
    u, v : float
        픽셀 좌표 (격자점 검출 결과)
    alpha, beta : float
        레이저 방사각 (rad). 시스템 캘리브레이션 값
    f : float
        카메라 초점거리 (pixel)
    b : float
        기선 baseline (m). 카메라가 +x로 b만큼 떨어짐
    cx, cy : float
        주점 (광축이 센서 통과하는 픽셀 좌표)
    
    Returns
    -------
    X, Y, Z : float (m)
        조사기 좌표계 기준 3D 좌표
    """
    # 시차 (disparity)
    d = f * np.tan(alpha) - (u - cx)
    
    if d <= 0:
        raise ValueError(
            f"Invalid disparity d={d:.3f}. "
            f"캘리브레이션 오류 또는 카메라가 조사기의 -x측에 있을 가능성"
        )
    
    # 거리 Z (PDF 핵심식)
    Z = f * b / d
    
    # 3D 좌표 (픽셀 기반, 조사기 좌표계)
    X = (u - cx) * Z / f + b
    Y = (v - cy) * Z / f
    
    return X, Y, Z


# ============ 자체 검증 (PDF 정합성) ============
if __name__ == "__main__":
    # PDF 검증 조건
    f, b, cx, cy = 1660.0, 0.05, 960.0, 540.0
    alpha, beta = np.radians(10.0), np.radians(5.0)
    Z_true = 1.0
    
    # 정방향: Z → 픽셀
    X_t = Z_true * np.tan(alpha)
    Y_t = Z_true * np.tan(beta)
    u = f * (X_t - b) / Z_true + cx
    v = f * Y_t / Z_true + cy
    
    # 역산
    X, Y, Z = triangulate_point(u, v, alpha, beta, f, b, cx, cy)
    
    print(f"[식 ①] 거리 측정 검증")
    print(f"  픽셀 입력: ({u:.4f}, {v:.4f})")
    print(f"  Z 측정값: {Z:.6f} m (참값 {Z_true} m)")
    print(f"  오차: {abs(Z - Z_true) * 1000:.6f} mm")
    print(f"  X 측정값: {X:.6f} m (참값 {X_t:.6f} m)")
    print(f"  Y 측정값: {Y:.6f} m (참값 {Y_t:.6f} m)")
    print(f"  결과: {'✓ PASS' if abs(Z - Z_true) < 1e-6 else '✗ FAIL'}")
