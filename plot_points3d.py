#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_points3d.py — 삼각측량 3D 점군을 **부재 구분까지 반영해서** 그린다
========================================================================
plot_xyz_points_steelbar.py 의 방식(matplotlib 3D · set_axes_equal · 전체
점 산점도)을 그대로 가져오되, 색칠 기준을 바꾼다.

    레퍼런스 :  선 계열(V=alpha고정 / H=beta고정) → 파랑 / 빨강
    여기     :  **세그멘테이션 부재**(벽 / 바닥 / 동바리 1·2·3 …) → 부재색
                가로선 점은 같은 부재색의 옅은 톤 (측정점과 구분)

왜 부재 기준인가
---------------
선 계열 색칠은 "어느 레이저 선이 맞았나" 를 보여 준다. 그건 검출 단계의
그림이다. 검측은 그 다음 단계 — 점이 **어느 부재에 속하는가** 로 갈린 뒤
부재별로 수직도·평활도가 나온다. 판정을 눈으로 확인하려면 판정이 쓴
그 구분으로 색을 칠해야 한다. `--by line` 으로 레퍼런스와 같은 색칠도
그대로 낼 수 있게 열어 뒀다.

정확도에 대해
------------
· **솎지 않는다.** 기본 stride=1 로 검측에 들어간 점을 전부 찍는다.
  (엑셀 시트는 파일 크기 때문에 20개마다 하나만 싣지만, 그림은 다르다.)
· **3축 등척.** set_axes_equal 로 세 축의 실제 배율을 맞춘다. 안 맞추면
  벽이 기울어 보이고 동바리가 굽어 보인다 — 그림이 거짓말을 한다.
· **좌표계를 고른다.** `--frame gravity` (기본) 는 중력 정렬 좌표
  (가로·깊이·높이) 라 벽은 수직으로, 바닥은 수평으로 선다. `--frame
  camera` 는 삼각측량이 직접 낸 조사기 좌표 (X, Y, Z) 그대로다.
· **가로선 점은 따로 표시한다.** 거리를 면에서 빌려 온 유도값이라
  검측에는 안 들어갔다. 범례에 "(투영)" 으로 적고 옅게 찍는다.

사용법
------
    # 이미지 하나로 파이프라인을 돌려 전체 점으로 그린다 (가장 정확)
    python3 plot_points3d.py CAST.png --save 점군.png

    # 캡처 폴더 (camera_params.json / cast_pixels.json 이 있는 곳)
    python3 plot_points3d.py ./capture_dir --save 점군.png

    # 이미 낸 좌표 CSV 로 (파이프라인 재실행 없이, 솎인 상태 그대로)
    python3 plot_points3d.py 결과_3D좌표.csv --save 점군.png

    # 레퍼런스 스키마 (triangulate_v2.py 의 xyz_result.json) — 선 색칠만
    python3 plot_points3d.py xyz_result.json --save 점군.png

옵션
----
    --save PATH          화면 대신 파일로 (원격·헤드리스면 자동으로 이쪽)
    --by member|class|line   색칠 기준 (기본 member)
                             member = 부재 하나하나 / class = 종류별
                             line   = 레퍼런스와 같은 V·H 색칠
    --frame gravity|camera   좌표계 (기본 gravity)
    --views quad|iso     4분할(등각+평면도+정면도+측면도) / 등각 하나
    --stride N           N개마다 하나만 (기본 1 = 전부)
    --no-aux             가로선(투영) 점을 빼고 그린다
    --xrange MIN MAX     축 확대 (--yrange, --zrange 도 같음)
    --elev E --azim A    등각 시점 각도
========================================================================
"""
import os as _os
import sys as _sys
import json as _json
import importlib.util as _ilu
import numpy as np


def _load(name):
    spec = _ilu.spec_from_file_location(
        name, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                            f"{name}.py"))
    m = _ilu.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


REPORT = _load("report")

# 레퍼런스와 같은 V·H 색
LINE_COLOR = {"V": "tab:blue", "H": "tab:red"}


# ============ 환경 ============
def _korean_font():
    """
    한글 폰트를 찾아 matplotlib 에 물린다. 못 찾으면 False 를 돌려주고,
    호출부가 라벨을 영문으로 바꾼다 — 네모(□□□)로 찍히는 것보다 낫다.

    코랩은 `apt-get install fonts-nanum` 뒤 캐시를 다시 만들어야 잡힌다.
    """
    import matplotlib
    from matplotlib import font_manager as fm
    want = ("NanumGothic", "NanumBarunGothic", "Malgun Gothic",
            "AppleGothic", "Noto Sans CJK KR", "Noto Sans KR",
            "WenQuanYi Zen Hei", "Droid Sans Fallback")
    have = {f.name for f in fm.fontManager.ttflist}
    for w in want:
        if w in have:
            matplotlib.rcParams["font.family"] = w
            matplotlib.rcParams["axes.unicode_minus"] = False
            return True
    # 설치는 됐는데 캐시에 없을 수 있다 — 파일을 직접 등록해 본다.
    for pat in ("/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
                "/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf",
                "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"):
        if _os.path.exists(pat):
            try:
                fm.fontManager.addfont(pat)
                matplotlib.rcParams["font.family"] = \
                    fm.FontProperties(fname=pat).get_name()
                matplotlib.rcParams["axes.unicode_minus"] = False
                return True
            except Exception:
                pass
    return False


def set_axes_equal(ax, max_ratio=12.0):
    """
    3축 실제 배율을 맞춘다 — 안 맞추면 형상이 눌리거나 늘어난다.

    흔한 방법은 세 축을 **가장 큰 범위** 로 똑같이 늘리는 것인데, 그러면
    깊이가 0.5~4.0m 로 넓은 장면에서 벽이 큰 정육면체 한구석에 몰려
    깨알만 하게 그려진다. 대신 축 범위는 데이터 그대로 두고 **그리는
    상자의 비율** 을 데이터 범위에 맞춘다. 1m 가 어느 축에서나 같은
    길이로 그려지므로 등척은 그대로이고, 화면은 다 쓴다.

    한 축이 거의 납작하면(면 부재의 두께) 상자가 종잇장이 되므로 비율을
    max_ratio 로 묶는다.
    """
    lim = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    r = np.maximum(lim[:, 1] - lim[:, 0], 1e-9)
    floor = float(r.max()) / float(max_ratio)
    grow = np.maximum(floor - r, 0.0) / 2.0          # 얇은 축만 넓힌다
    c = lim.mean(axis=1)
    r = np.maximum(r, floor)
    for k, setter in enumerate((ax.set_xlim3d, ax.set_ylim3d, ax.set_zlim3d)):
        if grow[k] > 0:
            setter([c[k] - r[k] / 2.0, c[k] + r[k] / 2.0])
    ax.set_box_aspect(tuple(r / r.max()))
    # 눌린 축에 눈금을 6개씩 그대로 두면 라벨이 서로 겹쳐 읽을 수 없다.
    # 상자에서 차지하는 비율만큼만 눈금을 둔다.
    from matplotlib.ticker import MaxNLocator
    for k, axis in enumerate((ax.xaxis, ax.yaxis, ax.zaxis)):
        n = int(np.clip(round(6.0 * r[k] / r.max()), 2, 6))
        axis.set_major_locator(MaxNLocator(nbins=n, prune=None))


# ============ 색 ============
def _shade(rgb, k, n):
    """
    같은 종류의 부재가 여러 본이면 밝기로 가른다.

    동바리 3본을 전부 같은 주황으로 칠하면 3D 에서 어느 게 어느 것인지
    알 수 없고, 그러면 "2번 동바리가 기준초과" 라는 판정을 그림에서
    확인할 수 없다. 색상은 종류를 지키고 밝기만 흔든다.
    """
    a = np.asarray(rgb, float) / 255.0
    if n <= 1:
        return tuple(a)
    t = 0.55 + 0.45 * (k / max(1, n - 1))          # 0.55 ~ 1.00
    return tuple(np.clip(a * t + (1.0 - t) * 0.15, 0.0, 1.0))


# ============ 결과 → 그릴 묶음 ============
def _frame_matrix(g_hat, frame):
    if frame == "camera" or g_hat is None:
        return None, ("X (m)", "Y (m)", "Z (m, 깊이)"), ("X (m)", "Y (m)",
                                                        "Z (m, depth)")
    return (REPORT.world_frame(g_hat),
            ("가로 (m)", "깊이 (m)", "높이 (m)"),
            ("Lateral (m)", "Depth (m)", "Height (m)"))


def groups_from_result(result, g_hat=None, by="member", frame="gravity",
                       stride=1, with_aux=True):
    """
    검측 결과 → [{label, color, xyz, aux}] 로 편다.

    by="member"  부재 하나하나 (같은 종류는 밝기로 구분)
    by="class"   종류별로 뭉쳐서
    by="line"    레퍼런스와 같은 V·H 계열 (point_lid 가 있어야 한다)
    """
    R, _, _ = _frame_matrix(g_hat, frame)
    st = max(1, int(stride))
    regs = [r for r in result.get("regions", [])
            if r.get("point_xyz") is not None and len(r["point_xyz"])]

    def _tf(P):
        P = np.asarray(P, float).reshape(-1, 3)[::st]
        return P @ R if R is not None else P

    if by == "line":
        # 레퍼런스와 같은 색칠 — 다만 점은 검측에 들어간 그 점이다.
        fam = {"V": [], "H": []}
        for r in regs:
            lid = r.get("point_lid")
            P = np.asarray(r["point_xyz"], float)
            if lid is None:
                fam["V"].append(P)
                continue
            L = np.asarray(lid, dtype=object)
            for k in ("V", "H"):
                m = np.array([str(x).startswith(k) for x in L])
                if m.any():
                    fam[k].append(P[m])
            if with_aux:
                A = r.get("aux_point_xyz")
                if A is not None and len(A):
                    fam["H"].append(np.asarray(A, float))
        out = []
        for k, lst in fam.items():
            if not lst:
                continue
            P = np.vstack(lst)
            out.append({"label": f"{k}선 ({len(P):,}점)",
                        "label_en": f"{k} lines ({len(P):,})",
                        "color": LINE_COLOR[k], "xyz": _tf(P), "aux": False})
        return out

    # 부재 번호는 **엑셀 9번 시트·3D좌표 CSV 와 같은 번호**를 쓴다.
    # 그림에서 "shoring #3" 을 보고 시트에서 3번 행을 찾을 수 있어야
    # 판정과 그림이 이어진다. 종류별로 1,2,3 을 새로 매기면 어긋난다.
    all_regs = result.get("regions", [])
    num = {id(r): i + 1 for i, r in enumerate(all_regs)}
    groups, seen = [], {}
    for r in regs:
        cls = r.get("class")
        seen[cls] = seen.get(cls, 0) + 1
    idx = {}
    for r in regs:
        cls = r.get("class")
        base = REPORT.CLASS_COLOR.get(cls, (200, 200, 200))
        if by == "class":
            key = cls
            col = tuple(np.asarray(base, float) / 255.0)
        else:
            idx[cls] = idx.get(cls, 0)
            col = _shade(base, idx[cls], seen[cls])
            key = (f"{cls} #{num.get(id(r), idx[cls] + 1)}"
                   if seen[cls] > 1 else cls)
            idx[cls] += 1
        j = (r.get("judge") or {})
        tag = j.get("judgement") or ("합격" if j.get("is_pass") else None)
        if r.get("status") != "measured":
            tag = "기각"
        P = _tf(r["point_xyz"])
        lab = f"{key} ({len(P):,}점" + (f", {tag})" if tag else ")")
        groups.append({"label": lab, "label_en": f"{key} ({len(P):,})",
                       "color": col, "xyz": P, "aux": False})
        if not with_aux:
            continue
        A = r.get("aux_point_xyz")
        if A is not None and len(A):
            Aw = _tf(A)
            surf = (r.get("aux_surface") or "투영").split(" (")[0]
            groups.append({
                "label": f"{key} 가로선 ({len(Aw):,}점, {surf})",
                "label_en": f"{key} H-lines ({len(Aw):,}, projected)",
                "color": tuple(np.asarray(col, float) * 0.72),
                "xyz": Aw, "aux": True})
    if by == "class":
        # 같은 키를 하나로 합친다
        merged = {}
        for g in groups:
            k = (g["label"].split(" (")[0], g["aux"])
            if k in merged:
                merged[k]["xyz"] = np.vstack([merged[k]["xyz"], g["xyz"]])
            else:
                merged[k] = dict(g)
        groups = []
        for (nm, is_aux), g in merged.items():
            n = len(g["xyz"])
            g["label"] = f"{nm}{' 가로선' if is_aux else ''} ({n:,}점" \
                         + (", 투영)" if is_aux else ")")
            g["label_en"] = f"{nm}{' H' if is_aux else ''} ({n:,})"
            groups.append(g)
    return groups


def defects_from_result(result, g_hat=None, frame="gravity"):
    """
    검출된 요철(평활도 결함)의 3D 위치. 점군만 보면 "여기가 튀었다" 가
    안 보이므로 따로 표시한다 — PIL 판 그림이 하던 일을 이어받는다.
    """
    R, _, _ = _frame_matrix(g_hat, frame)
    P, mm, skipped = [], [], 0
    for r in result.get("regions", []):
        fl = r.get("flatness") or {}
        # 평활도를 못 잰 영역의 '요철' 은 잡음이다. 32.1mm 짜리 붉은 원을
        # 그려 놓으면 근거 없는 결함을 그림이 주장하는 꼴이 된다 —
        # 실측에서 그 덩어리는 벽 전체(3004.9mm)였다.
        if fl.get("judgement") in ("측정불가", None) and fl.get("defects"):
            skipped += len(fl["defects"])
            continue
        for d in (fl.get("defects") or []):
            c = d.get("center_xyz")
            if c is None:
                continue
            P.append(np.asarray(c, float))
            mm.append(float(d.get("depth_mm", 0.0)))
    if not P:
        return np.empty((0, 3)), np.empty(0), skipped
    A = np.vstack(P)
    return ((A @ R if R is not None else A), np.asarray(mm, float), skipped)


def groups_from_csv(path, by="member", frame="gravity", stride=1,
                    with_aux=True):
    """파이프라인이 낸 `…_3D좌표.csv` 로 그린다 (재실행 없이)."""
    import csv
    rows = list(csv.DictReader(open(path, encoding="utf-8-sig")))
    if not rows:
        raise ValueError(f"빈 CSV: {path}")
    grav = all(k in rows[0] for k in ("가로_m", "깊이_m", "높이_m")) \
        and rows[0]["가로_m"] not in (None, "")
    if frame == "gravity" and not grav:
        print("  [주의] CSV 에 중력 좌표가 없어 조사기 좌표로 그립니다")
        frame = "camera"
    cols = (("가로_m", "깊이_m", "높이_m") if frame == "gravity"
            else ("X_m", "Y_m", "Z_m"))
    st = max(1, int(stride))

    bag = {}
    for i, r in enumerate(rows):
        if i % st:
            continue
        is_aux = (r.get("구분") or "").startswith("가로선")
        if is_aux and not with_aux:
            continue
        cls = r.get("클래스")
        key = (cls if by == "class"
               else f"{cls} #{r.get('부재번호')}")
        if by == "line":
            key = (r.get("선ID") or "H")[0]
        bag.setdefault((key, cls, is_aux), []).append(
            [float(r[c]) for c in cols])

    seen = {}
    for (key, cls, _a) in bag:
        seen.setdefault(cls, set()).add(key)
    idx, out = {}, []
    for (key, cls, is_aux), pts in sorted(bag.items()):
        P = np.asarray(pts, float)
        if by == "line":
            col = LINE_COLOR.get(key, "tab:gray")
            out.append({"label": f"{key}선 ({len(P):,}점)",
                        "label_en": f"{key} lines ({len(P):,})",
                        "color": col, "xyz": P, "aux": is_aux})
            continue
        base = REPORT.CLASS_COLOR.get(cls, (200, 200, 200))
        n = len(seen.get(cls, {key}))
        if not is_aux:
            idx[cls] = idx.get(cls, 0)
            k = idx[cls]
            idx[cls] += 1
        else:
            k = max(0, idx.get(cls, 1) - 1)
        col = (tuple(np.asarray(base, float) / 255.0) if by == "class"
               else _shade(base, k, n))
        if is_aux:
            col = tuple(np.asarray(col, float) * 0.72)
        nm = key
        out.append({"label": f"{nm}{' 가로선' if is_aux else ''} "
                             f"({len(P):,}점{', 투영' if is_aux else ''})",
                    "label_en": f"{nm} ({len(P):,})",
                    "color": col, "xyz": P, "aux": is_aux})
    return out


def groups_from_xyz_json(path, stride=1):
    """
    레퍼런스 스키마 — {lid: {"fixed": "alpha"|"beta", "points": [{"xyz":…}]}}.

    이 파일에는 **부재 구분이 없다**. 선 계열 색칠만 가능하고, 세그멘테이션
    반영을 원하면 이미지나 캡처 폴더를 넣어 파이프라인을 태워야 한다.
    """
    data = _json.load(open(path, encoding="utf-8"))
    st = max(1, int(stride))
    fam = {"V": [], "H": []}
    for lid, info in data.items():
        pts = info.get("points", [])
        if not pts:
            continue
        P = np.asarray([p["xyz"] for p in pts], float)[::st]
        if not len(P):
            continue
        k = "V" if info.get("fixed") == "alpha" else "H"
        fam[k].append(P)
    out = []
    for k, lst in fam.items():
        if not lst:
            continue
        P = np.vstack(lst)
        out.append({"label": f"{k}선 ({len(P):,}점)",
                    "label_en": f"{k} lines ({len(P):,})",
                    "color": LINE_COLOR[k], "xyz": P, "aux": False})
    return out


# ============ 그리기 ============
def draw(groups, path=None, title=None, g_hat=None, frame="gravity",
         views="quad", elev=None, azim=None, ranges=None, point_size=2.0,
         figsize=None, dpi=150, defects=None):
    """
    묶음들을 3D 산점도로 그린다.

    views="quad" 면 등각 + 평면도 + 정면도 + 측면도 네 칸. 등각 하나만으로는
    "정말 수직인가" 를 눈으로 못 가린다 — 옆에서 봐야 보인다.
    views="iso" 면 레퍼런스처럼 큰 등각 하나.
    """
    import matplotlib
    if path:
        matplotlib.use("Agg")
    ko = _korean_font()
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D            # noqa: F401

    _, lab_ko, lab_en = _frame_matrix(g_hat, frame)
    labs = lab_ko if ko else lab_en
    groups = [g for g in groups if g.get("xyz") is not None and len(g["xyz"])]
    if not groups:
        raise ValueError("그릴 점이 없다")
    ALL = np.vstack([g["xyz"] for g in groups])

    D = np.empty((0, 3)); Dmm = np.empty(0); Dskip = 0
    if defects is not None:
        D, Dmm, Dskip = defects
        D = np.asarray(D, float).reshape(-1, 3)

    def _scatter3d(ax):
        for g in groups:
            P = g["xyz"]
            ax.scatter(P[:, 0], P[:, 1], P[:, 2], s=point_size,
                       c=[g["color"]], alpha=0.45 if g["aux"] else 0.75,
                       marker="." if g["aux"] else "o",
                       linewidths=0,
                       label=(g["label"] if ko else g["label_en"]))
        if len(D):
            # 범례는 아래에서 프록시로 따로 만든다 — markerscale 을 점에
            # 맞춰 키우면 요철 동그라미만 화면을 덮는다.
            ax.scatter(D[:, 0], D[:, 1], D[:, 2], s=90, facecolors="none",
                       edgecolors="crimson", linewidths=1.4)
        ax.set_xlabel(labs[0]); ax.set_ylabel(labs[1]); ax.set_zlabel(labs[2])
        if elev is not None or azim is not None:
            ax.view_init(elev=elev if elev is not None else 30,
                         azim=azim if azim is not None else -60)
        set_axes_equal(ax)
        if ranges:
            for k, fn in (("x", ax.set_xlim3d), ("y", ax.set_ylim3d),
                          ("z", ax.set_zlim3d)):
                if ranges.get(k):
                    fn(*ranges[k])

    if views == "iso":
        fig = plt.figure(figsize=figsize or (11, 9))
        ax = fig.add_subplot(111, projection="3d")
        _scatter3d(ax)
        h, l = ax.get_legend_handles_labels()
        if len(D):
            from matplotlib.lines import Line2D
            h.append(Line2D([], [], marker="o", linestyle="none",
                            markerfacecolor="none", markeredgecolor="crimson",
                            markeredgewidth=1.4, markersize=3))
            l.append((f"요철 {len(D)}곳" if ko else f"Defects ({len(D)})")
                     + (f" (평활도 측정불가 {Dskip}곳 제외)"
                        if ko and Dskip else ""))
        ax.legend(h, l, loc="upper right", fontsize=8, markerscale=4)
    else:
        fig = plt.figure(figsize=figsize or (15, 12))
        ax = fig.add_subplot(2, 2, 1, projection="3d")
        _scatter3d(ax)
        ax.set_title("등각" if ko else "Isometric", fontsize=11)
        # 직교 3면 — 등척을 지켜야 각도를 눈으로 잴 수 있다
        # 화면 축을 어떻게 잡을지 — 중력 좌표는 (가로, 깊이, 높이)
        combos = ((0, 1, "평면도 (위에서)", "Top view"),
                  (0, 2, "정면도 (앞에서)", "Front view"),
                  (1, 2, "측면도 (옆에서)", "Side view"))
        for k, (i, j, nm, nm_en) in enumerate(combos, start=2):
            a = fig.add_subplot(2, 2, k)
            for g in groups:
                P = g["xyz"]
                a.scatter(P[:, i], P[:, j], s=point_size,
                          c=[g["color"]], alpha=0.4 if g["aux"] else 0.7,
                          marker="." if g["aux"] else "o", linewidths=0)
            if len(D):
                a.scatter(D[:, i], D[:, j], s=90, facecolors="none",
                          edgecolors="crimson", linewidths=1.4)
                # 요철이 한 자리에 겹치면 라벨도 겹친다. 가까운 것끼리는
                # 위아래로 어긋나게 놓는다.
                order = np.argsort(D[:, i])
                # "가깝다" 는 절대값이 아니라 그 축 범위 기준이다 — 깊이만
                # 좁은 측면도에서 8cm 차이도 화면에서는 겹친다.
                near = 0.04 * max(float(np.ptp(ALL[:, i])), 1e-6)
                last, step = -1e9, 0
                for q in order:
                    step = step + 1 if (D[q, i] - last) < near else 0
                    last = D[q, i]
                    a.annotate(f"{Dmm[q]:.1f}mm", (D[q, i], D[q, j]),
                               textcoords="offset points",
                               xytext=(7, 5 + 11 * (step % 3)),
                               fontsize=7, color="crimson")
            a.set_xlabel(labs[i]); a.set_ylabel(labs[j])
            a.set_title(nm if ko else nm_en, fontsize=11)
            a.set_aspect("equal", adjustable="datalim")
            a.grid(True, lw=0.3, alpha=0.4)
            if ranges:
                rx = ranges.get("xyz"[i]); ry = ranges.get("xyz"[j])
                if rx:
                    a.set_xlim(*rx)
                if ry:
                    a.set_ylim(*ry)
        handles, labels = fig.axes[0].get_legend_handles_labels()
        if len(D):
            from matplotlib.lines import Line2D
            handles.append(Line2D([], [], marker="o", linestyle="none",
                                  markerfacecolor="none",
                                  markeredgecolor="crimson",
                                  markeredgewidth=1.4, markersize=3))
            labels.append((f"요철 {len(D)}곳" if ko
                           else f"Defects ({len(D)})")
                          + (f" (평활도 측정불가 {Dskip}곳 제외)"
                             if ko and Dskip else ""))
        fig.legend(handles, labels, loc="lower center",
                   ncol=min(4, max(1, len(labels))), fontsize=9,
                   markerscale=4, frameon=False,
                   bbox_to_anchor=(0.5, -0.01))
    if title:
        fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97) if views != "iso" else None)

    if path:
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return path
    plt.show()
    return None


def save_pointcloud_mpl(path, result, g_hat=None, by="member",
                        frame="gravity", views="quad", stride=1,
                        with_aux=True, title=None, **kw):
    """검측 결과 → matplotlib 3D 점군 파일. 실패하면 None (호출부가 폴백)."""
    try:
        # 기본은 전부 찍는다. 다만 점이 너무 많으면 그리는 데만 몇 분이
        # 걸리고 인쇄 해상도에서 어차피 겹쳐 보이므로, 15만 점을 넘을
        # 때만 자동으로 솎는다(요청 stride 가 있으면 그쪽을 존중한다).
        if int(stride) <= 1:
            def _n(r, k):
                # numpy 배열에 `or []` 를 쓰면 진리값 판정에서 터진다.
                v = r.get(k)
                return 0 if v is None else len(v)
            tot = sum(_n(r, "point_xyz") + _n(r, "aux_point_xyz")
                      for r in result.get("regions", []))
            if tot > 150_000:
                stride = int(np.ceil(tot / 150_000))
        groups = groups_from_result(result, g_hat=g_hat, by=by, frame=frame,
                                    stride=stride, with_aux=with_aux)
        return draw(groups, path=path, title=title, g_hat=g_hat, frame=frame,
                    views=views,
                    defects=defects_from_result(result, g_hat, frame), **kw)
    except Exception as e:
        # 조용히 삼키면 안 된다 — 폴백이 도는 것과 그림이 틀린 것을
        # 구분할 수 없어진다. 실제로 numpy 배열에 `or []` 를 쓴 버그가
        # 이 자리에서 통째로 묻혔다.
        print(f"  [경고] matplotlib 3D 점군 실패 ({type(e).__name__}: {e})"
              " — PIL 판으로 대체합니다")
        return None


# ============ CLI ============
def _run_pipeline_for(target, verbose=True):
    """이미지·캡처 폴더 → (result, g_hat). 전체 점을 쓰려고 다시 돌린다."""
    if _os.path.isdir(target):
        LC = _load("load_capture")
        r = LC.inspect_folder(target, pc_stride=1, source="detected")
        for v in (r.values() if isinstance(r, dict) else []):
            if isinstance(v, dict) and v.get("regions"):
                return v, np.asarray(v.get("g_hat", [0, 1, 0]), float)
        if isinstance(r, dict) and r.get("regions"):
            return r, np.asarray(r.get("g_hat", [0, 1, 0]), float)
        raise ValueError("검측 결과를 찾지 못했다")
    RP = _load("run_pipeline")
    d = _os.path.dirname(_os.path.abspath(target))

    def _pick(n):
        p = _os.path.join(d, n)
        return p if _os.path.exists(p) else None
    res = RP.run(image=target, params=_pick("camera_params.json"),
                 truth=_pick("cast_pixels.json"),
                 scene_image=_pick("CAM.png"), pc_stride=1, verbose=verbose)
    return res["result"], res["g_hat"]


def main(argv=None):
    a = list(_sys.argv[1:] if argv is None else argv)
    if not a:
        print(__doc__)
        return 1

    def _opt(name, n=1, cast=str):
        if name not in a:
            return None
        i = a.index(name)
        v = [cast(x) for x in a[i + 1:i + 1 + n]]
        del a[i:i + 1 + n]
        return v[0] if n == 1 else v

    save = _opt("--save")
    by = _opt("--by") or "member"
    frame = _opt("--frame") or "gravity"
    views = _opt("--views") or "quad"
    stride = int(_opt("--stride") or 1)
    elev = _opt("--elev", cast=float)
    azim = _opt("--azim", cast=float)
    ranges = {}
    for k in "xyz":
        v = _opt(f"--{k}range", 2, float)
        if v:
            ranges[k] = tuple(v)
    with_aux = True
    if "--no-aux" in a:
        with_aux = False
        a.remove("--no-aux")
    if by not in ("member", "class", "line"):
        print(f"--by 는 member|class|line (받은 값: {by})")
        return 2
    if not a:
        print("입력을 하나 주세요 (이미지 / 캡처폴더 / CSV / xyz_result.json)")
        return 2
    target = a[0]

    g_hat = None
    defects = None
    if target.lower().endswith(".csv"):
        groups = groups_from_csv(target, by=by, frame=frame, stride=stride,
                                 with_aux=with_aux)
        note = "CSV (엑셀 시트와 같은 솎임)"
    elif target.lower().endswith(".json"):
        if by != "line":
            print("  [주의] xyz_result.json 에는 부재 구분이 없습니다 "
                  "— 선(V/H) 색칠로 그립니다")
        groups = groups_from_xyz_json(target, stride=stride)
        by, frame = "line", "camera"
        note = "xyz_result.json (부재 구분 없음)"
    else:
        result, g_hat = _run_pipeline_for(target)
        groups = groups_from_result(result, g_hat=g_hat, by=by, frame=frame,
                                    stride=stride, with_aux=with_aux)
        defects = defects_from_result(result, g_hat, frame)
        note = "파이프라인 전체 점"

    total = sum(len(g["xyz"]) for g in groups)
    print(f"\n[점군] {note} — {len(groups)}묶음 / {total:,}점 "
          f"/ 좌표계 {frame} / 색칠 {by}")
    for g in groups:
        P = g["xyz"]
        print(f"  {g['label']:<34} "
              f"X {P[:,0].min():7.3f}~{P[:,0].max():7.3f} "
              f"Y {P[:,1].min():7.3f}~{P[:,1].max():7.3f} "
              f"Z {P[:,2].min():7.3f}~{P[:,2].max():7.3f} m")

    out = draw(groups, path=save, g_hat=g_hat, frame=frame, views=views,
               elev=elev, azim=azim, ranges=ranges or None, defects=defects,
               title=f"3D 점군 — {_os.path.basename(target)}")
    if out:
        print(f"[저장] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
