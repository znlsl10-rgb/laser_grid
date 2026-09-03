#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"왜 기선을 대각으로 두면 가로선도 깊이를 주는가" 설명 그림
==========================================================
그림은 전부 **실제 코드의 식** 으로 그린다 (eq7_laser_plane).
손으로 그린 개념도가 아니라, 파이프라인이 쓰는 것과 같은 평면식이다.

    레이저 평면      n·P = 0            (평면은 조사기 원점을 지난다)
    카메라 시선      P = b + Z·(û, v̂, 1)
    →  n·b + Z·(n_x û + n_y v̂ + n_z) = 0
    →  깊이          Z = − (n·b) / (n_x û + n_y v̂ + n_z)
    →  화면상 자취    n_x û + n_y v̂ + n_z = − (n·b)/Z

마지막 줄이 이 문서의 전부다. 깊이 Z 는 **오직 (n·b) 를 통해서만** 자취에
들어간다. n·b = 0 이면 자취가 Z 와 무관해지고, 두 깊이의 선이 화면에서
완전히 겹친다 — 어떤 알고리즘도 구분할 수 없다.

실행:  python3 docs/기선각도_그림.py
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
OUT = os.path.join(HERE, "그림")

import importlib.util as _ilu


def _load(name):
    spec = _ilu.spec_from_file_location(name, os.path.join(ROOT, f"{name}.py"))
    m = _ilu.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


EQ7 = _load("eq7_laser_plane")
_load("plot_points3d")._korean_font()

# 예제 촬영과 같은 사양 (samples/example_capture)
F_PX, B_M = 942.4, 0.150
W, H = 1224, 1024
CX, CY = (W - 1) / 2.0, (H - 1) / 2.0
Z_FAR, Z_NEAR = 1.50, 1.20          # 배경 벽 / 앞에 선 부재

INK = "#1a1a1a"
FAR = "#9aa4b2"
NEAR = "#0b6bcb"
BASE = "#d1495b"
OK = "#2a9d8f"
BAD = "#d1495b"


# ============ 식 그대로 ============
def trace_xy(n, b_vec, Z, span=520):
    """깊이 Z 의 정면 평면과 레이저 평면이 만나는 선의 화소 좌표."""
    n = np.asarray(n, float)
    K = float(n[2] + (n @ b_vec) / Z)      # n_x û + n_y v̂ + K = 0
    lat = float(np.hypot(n[0], n[1]))
    m = np.array([n[0], n[1]]) / lat       # 화면상 선의 법선
    t = np.array([-m[1], m[0]])            # 선 방향
    p0 = np.array([CX, CY]) - m * (K * F_PX / lat)
    s = np.linspace(-span, span, 2)
    return p0[0] + t[0] * s, p0[1] + t[1] * s


def gain(n, b_vec):
    nb = float(np.asarray(n, float) @ b_vec)
    lat = float(np.hypot(n[0], n[1]))
    return np.inf if abs(nb) < 1e-12 else np.linalg.norm(b_vec) * lat / abs(nb)


def baseline_vec(psi_deg):
    p = np.radians(psi_deg)
    return B_M * np.array([np.cos(p), np.sin(p), 0.0])


# ============ 그림 1 — 깊이가 변하면 화소는 기선 방향으로만 움직인다 ============
def _line_pts(n, b_vec, Z, span):
    """자취 위의 두 끝점과, 주점에서 가장 가까운 점."""
    n = np.asarray(n, float)
    lat = float(np.hypot(n[0], n[1]))
    K = float(n[2] + (n @ b_vec) / Z)
    m = np.array([n[0], n[1]]) / lat
    t = np.array([-m[1], m[0]])
    p0 = np.array([CX, CY]) - m * (K * F_PX / lat)
    return p0 - t * span, p0 + t * span, p0, m, t


def fig1():
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 6.4))
    cfgs = [(0.0, 0.0, "① 기선 수평 · 굴림 0°   (지금 예제)"),
            (0.0, 20.0, "② 기선 수평 · 굴림 20°"),
            (45.0, 0.0, "③ 기선 45° 대각 · 굴림 0°")]
    R = 640
    for ax, (psi, roll, title) in zip(axes, cfgs):
        b = baseline_vec(psi)
        ax.set_title(title, fontsize=12.5, pad=12, color=INK)

        for k in range(-3, 4):                     # 배경 격자 (문맥용)
            for fam in ("V", "H"):
                n = EQ7.plane_normal("alpha" if fam == "V" else "beta",
                                     np.radians(k * 7.0),
                                     roll_rad=np.radians(roll))
                a, c, _, _, _ = _line_pts(n, b, Z_FAR, 900)
                ax.plot([a[0], c[0]], [a[1], c[1]], color="#e6e9ed", lw=1.5,
                        zorder=1)

        info = []
        for fam, ang, along in (("V", -14.0, -300), ("H", 14.0, -330)):
            n = EQ7.plane_normal("alpha" if fam == "V" else "beta",
                                 np.radians(ang), roll_rad=np.radians(roll))
            g = gain(n, b)
            af, cf, pf, m, t = _line_pts(n, b, Z_FAR, 900)
            an, cn, pn, _, _ = _line_pts(n, b, Z_NEAR, 900)
            ax.plot([af[0], cf[0]], [af[1], cf[1]], color=FAR, lw=4.2,
                    solid_capstyle="round", zorder=2)
            ax.plot([an[0], cn[0]], [an[1], cn[1]], color=NEAR, lw=2.5,
                    ls=(0, (6, 4)), zorder=3)
            d_px = float(np.linalg.norm(pn - pf))
            q = pf + t * along
            if d_px >= 1.5:
                ax.add_patch(FancyArrowPatch(q, q + (pn - pf),
                                             arrowstyle="<->",
                                             mutation_scale=12, lw=2.0,
                                             color="#111", zorder=6))
                ax.text(*(q + (pn - pf) / 2 + m * 46), f"{d_px:.1f} px",
                        fontsize=11.5, color="#111", fontweight="bold",
                        ha="center", va="center", zorder=7,
                        bbox=dict(boxstyle="round,pad=0.25", fc="white",
                                  ec="none", alpha=0.92))
            else:
                ax.annotate("두 선이 완전히 겹친다 (이동 0.00 px)",
                            xy=tuple(q), xytext=(0.56, 0.26),
                            textcoords=ax.transAxes, fontsize=11,
                            color=BAD, fontweight="bold", ha="center",
                            zorder=7,
                            arrowprops=dict(arrowstyle="->", color=BAD,
                                            lw=2.0))
            info.append((("세로선" if fam == "V" else "가로선"), d_px, g))

        # 기선 방향 — 화면 오른쪽 위에 따로 둔다
        c0 = np.array([0.80, 0.90])                # axes 좌표
        d = np.array([np.cos(np.radians(psi)),
                      -np.sin(np.radians(psi))]) * 0.13
        ax.add_patch(FancyArrowPatch(c0 - d, c0 + d, arrowstyle="<->",
                                     mutation_scale=17, lw=2.6, color=BASE,
                                     transform=ax.transAxes, zorder=6))
        ax.text(0.80, 0.985, "기선 방향", color=BASE, fontsize=10.5,
                ha="center", va="top", fontweight="bold",
                transform=ax.transAxes, zorder=7)

        rows = "\n".join(
            f"{nm}   이동 {d:.1f} px      g = "
            + ("∞  (깊이 못 잼)" if not np.isfinite(gg) else f"{gg:.2f}")
            for nm, d, gg in info)
        col = BAD if any(not np.isfinite(gg) for _, _, gg in info) else OK
        ax.text(0.5, 0.035, rows, transform=ax.transAxes, fontsize=11.5,
                ha="center", va="bottom", color=col, fontweight="bold",
                linespacing=1.7, zorder=8,
                bbox=dict(boxstyle="round,pad=0.55", fc="#f7f8fa",
                          ec="#dfe3e8"))
        ax.set_xlim(CX - R, CX + R)
        ax.set_ylim(CY + R * 0.95, CY - R * 0.95)
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color("#c9ced6")

    fig.suptitle("깊이가 변하면 화소는 오직 기선 방향으로만 움직인다\n"
                 "선이 그 방향과 나란하면 자기 위를 미끄러질 뿐이라, "
                 "화면에는 아무 변화도 남지 않는다",
                 fontsize=14, y=0.995, color=INK, linespacing=1.6)
    fig.text(0.5, 0.862,
             f"굵은 회색 = 배경 {Z_FAR:.2f} m 에 맺힌 레이저 선          "
             f"파란 점선 = 같은 선이 앞의 부재 {Z_NEAR:.2f} m 에 맺혔을 때",
             ha="center", fontsize=11, color="#5b6472")
    fig.tight_layout(rect=[0, 0.01, 1, 0.838])
    p = os.path.join(OUT, "1_시차방향.png")
    fig.savefig(p, dpi=150, facecolor="white"); plt.close(fig)
    return p


# ============ 그림 2 — g = 1/|sin θ| ============
def fig2():
    th = np.linspace(0.5, 90, 400)
    fig, ax = plt.subplots(figsize=(9.4, 5.4))
    ax.plot(th, 1.0 / np.sin(np.radians(th)), color=NEAR, lw=2.8, zorder=3)
    ax.axhline(EQ7.MAX_DEPTH_GAIN, color=BAD, lw=1.4, ls="--", zorder=2)
    ax.text(23.0, EQ7.MAX_DEPTH_GAIN + 0.18,
            f"코드 상한 g = {EQ7.MAX_DEPTH_GAIN:.0f} — 이보다 크면 그 선은 버린다",
            color=BAD, fontsize=9.5, ha="left", va="bottom")
    lim = np.degrees(np.arcsin(1.0 / EQ7.MAX_DEPTH_GAIN))
    ax.axvspan(0, lim, color=BAD, alpha=0.07, zorder=1)
    ax.text(lim / 2 + 1.0, 6.6, f"{lim:.1f}° 미만\n쓸 수 없는 구간",
            color=BAD, fontsize=10, ha="center", va="center",
            fontweight="bold", zorder=4,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none",
                      alpha=0.85))
    # 점은 곡선 위에, 이름표는 겹치지 않는 자리에 두고 지시선으로 잇는다
    marks = [(5.0, '가로선 · 굴림 5°\n← "살짝" 굴리면 여기', BAD, (17.0, 11.6)),
             (20.0, "가로선 · 굴림 20°", "#c77700", (26.0, 4.9)),
             (45.0, "대각 기선 45°\n세로·가로 둘 다 여기", OK, (43.0, 6.4)),
             (70.0, "세로선 · 굴림 20°", OK, (64.0, 4.3)),
             (90.0, "세로선 · 기선 수평", OK, (86.0, 2.0))]
    for t, lab, c, xy in marks:
        gg = 1.0 / np.sin(np.radians(t))
        ax.plot([t], [gg], "o", color=c, ms=9, zorder=6, mec="white", mew=1.6)
        ax.annotate(lab, (t, gg), xytext=xy, textcoords="data", fontsize=9.5,
                    color=c, fontweight="bold", ha="center", va="bottom",
                    zorder=6,
                    arrowprops=dict(arrowstyle="-", color=c, lw=1.0,
                                    alpha=0.55,
                                    shrinkA=2, shrinkB=6))
    ax.set_xlim(0, 93); ax.set_ylim(0, 12.5)
    ax.set_xlabel("화면에서 레이저 선과 기선이 이루는 각  θ  [°]", fontsize=11)
    ax.set_ylabel("깊이 잡음 배수  g", fontsize=11)
    ax.set_title("g = 1 / |sin θ|\n"
                 "굴림이든 기선 각도든, 깊이 정밀도를 정하는 것은 둘 사이의 "
                 "각 하나뿐이다 (72개 조합에서 정확식과 일치 확인)",
                 fontsize=12.5, pad=14, linespacing=1.5)
    ax.grid(alpha=0.25)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    p = os.path.join(OUT, "2_이득곡선.png")
    fig.savefig(p, dpi=150, facecolor="white"); plt.close(fig)
    return p


# ============ 그림 3 — 장비 배치 (앞에서 본 그림) ============
def fig3():
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 5.2))
    for ax, (psi, title) in zip(axes, [(0.0, "지금 배치 — 카메라를 조사기 옆에"),
                                       (45.0, "제안 — 카메라를 비스듬히 위옆에")]):
        p = np.radians(psi)
        cam = np.array([np.cos(p), np.sin(p)]) * 0.95
        ax.plot([0], [0], "s", color="#2b8a3e", ms=17, zorder=4)
        ax.text(0, -0.16, "조사기(DOE)", ha="center", va="top", fontsize=10,
                color="#2b8a3e", fontweight="bold")
        ax.plot([cam[0]], [cam[1]], "s", color=NEAR, ms=17, zorder=4)
        ax.text(cam[0] + 0.06, cam[1] + 0.16, "카메라", ha="center",
                va="bottom", fontsize=10, color=NEAR, fontweight="bold")
        ax.add_patch(FancyArrowPatch((0, 0), tuple(cam), arrowstyle="<->",
                                     mutation_scale=15, lw=2.4, color=BASE,
                                     zorder=3))
        mid = cam / 2
        ax.text(mid[0] + (0.0 if psi == 0 else -0.36), mid[1] - 0.10,
                f"기선 b = {B_M*1000:.0f}mm", ha="center", fontsize=9.5,
                color=BASE, fontweight="bold")
        # 화면에 찍히는 격자 — 굴림 0 이므로 두 배치 모두 반듯하다
        gx = 2.05
        for k in np.linspace(-0.40, 0.40, 5):
            ax.plot([gx + k, gx + k], [-0.42, 0.42], color="#4caf50", lw=1.4)
            ax.plot([gx - 0.42, gx + 0.42], [k, k], color="#4caf50", lw=1.4)
        ax.text(gx, 0.52, "화면에 찍히는 격자", ha="center", va="bottom",
                fontsize=9.5, color="#2b8a3e")
        b = baseline_vec(psi)
        gV = gain(EQ7.plane_normal("alpha", 0.0), b)
        gH = gain(EQ7.plane_normal("beta", 0.0), b)
        good = np.isfinite(gH)
        s = (f"세로선  g = {gV:.2f}\n"
             + (f"가로선  g = {gH:.2f}" if good else "가로선  g = ∞  →  깊이 못 잼"))
        ax.text(gx, -0.55, s, ha="center", va="top", fontsize=11,
                color=(OK if good else BAD), fontweight="bold",

                bbox=dict(boxstyle="round,pad=0.45", fc="#f7f8fa",
                          ec="#dfe3e8"))
        ax.set_title(title, fontsize=12, pad=12)
        ax.set_xlim(-0.5, 2.75); ax.set_ylim(-1.02, 1.28)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color("#c9ced6")
    fig.suptitle("격자는 굴리지 않고 카메라 장착 위치만 바꾼다\n"
                 "격자는 화면에서 반듯한 채로, 가로선이 깊이를 준다",
                 fontsize=13, y=0.99, linespacing=1.5)
    fig.tight_layout(rect=[0, 0, 1, 0.85])
    p = os.path.join(OUT, "3_장비배치.png")
    fig.savefig(p, dpi=150, facecolor="white"); plt.close(fig)
    return p


def main():
    os.makedirs(OUT, exist_ok=True)
    for fn in (fig1, fig2, fig3):
        print("[생성]", fn())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
