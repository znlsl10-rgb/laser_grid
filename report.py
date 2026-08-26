#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
report.py — 영역별 검측 결과 보고서
========================================================================
pipeline_region 의 결과를 현장에서 바로 쓸 수 있는 형태로 낸다.

  1. 검측 조서 (JSON)  — 부재별 측정값·판정·근거·불확실도
  2. 오버레이 이미지   — 어느 영역이 무엇으로 판정됐는지 사진 위에 표시
  3. 요약 표 (텍스트)  — 터미널·로그용

【판정 표기】
  합격 / 기준초과 는 시방 기준 대비 결과이고, 다음 둘은 **판정을 하지
  않은** 상태다. 이 구분을 흐리면 신뢰할 수 없는 값이 합격으로 새어나간다.
    측정불가        : 측정 불확실도가 목표(±2mm)를 넘어 값 자체를 못 믿는다
    판정보류(분해능) : 값은 허용 이내이나, 요철 폭이 분해능보다 좁아
                     실제로는 넘을 수 있다
========================================================================
"""
import os, json
import numpy as np
import importlib.util as _ilu


def _load(name):
    spec = _ilu.spec_from_file_location(
        name, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           f"{name}.py"))
    m = _ilu.module_from_spec(spec); spec.loader.exec_module(m)
    return m


_EQ5 = _load("eq5_region_assign")

# 보고서 표기용 한글 이름
CLASS_KO = {"wall": "벽체", "formwork_wall": "벽체 거푸집",
            "formwork_column": "기둥 거푸집", "masonry": "조적벽체",
            "plaster_wall": "미장 벽면", "floor": "바닥", "slab": "슬래브",
            "ceiling": "천장", "shoring": "동바리", "column": "기둥",
            "rebar": "철근"}
KIND_KO = {"plane_vertical": "수직도", "plane_horizontal": "수평도",
           "axis_vertical": "부재 수직도"}
# 오버레이 색 (R,G,B)
CLASS_COLOR = {"wall": (60, 130, 246), "formwork_wall": (99, 102, 241),
               "formwork_column": (139, 92, 246), "masonry": (14, 165, 233),
               "plaster_wall": (56, 189, 248), "floor": (34, 197, 94),
               "slab": (22, 163, 74), "ceiling": (132, 204, 22),
               "shoring": (249, 115, 22), "column": (234, 88, 12),
               "rebar": (239, 68, 68), "background": (110, 110, 110)}
# 요철 표시색 — 부재 색 어느 것과도 겹치지 않는 자홍
DEFECT_COLOR = (236, 72, 153)
VERDICT_COLOR = {"합격": (34, 197, 94), "기준초과": (239, 68, 68),
                 "측정불가": (148, 163, 184), "판정보류(분해능)": (234, 179, 8)}

# 오버레이 범례용 ASCII 대체 표기.
# PIL 기본 비트맵 폰트에는 한글이 없어 □ 로 찍힌다. 한글 TTF 를 찾으면
# 한글로, 못 찾으면 아래 표기로 그린다(조서 JSON·텍스트는 항상 한글).
CLASS_EN = {"wall": "Wall", "formwork_wall": "Formwork(Wall)",
            "formwork_column": "Formwork(Col)", "masonry": "Masonry",
            "plaster_wall": "Plaster", "floor": "Floor", "slab": "Slab",
            "ceiling": "Ceiling", "shoring": "Shoring", "column": "Column",
            "rebar": "Rebar"}
KIND_EN = {"plane_vertical": "verticality", "plane_horizontal": "horizontality",
           "axis_vertical": "axis-verticality"}
VERDICT_EN = {"합격": "PASS", "기준초과": "FAIL", "측정불가": "N/M",
              "판정보류(분해능)": "HOLD", "해당없음": "n/a"}

# 배포 환경(윈도우/리눅스 워크스테이션)에 흔한 한글 폰트 경로
_KOREAN_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    "/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/google-noto-cjk/NotoSansCJK-Regular.ttc",
    "C:/Windows/Fonts/malgun.ttf",
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
    "/Library/Fonts/AppleGothic.ttf",
    # 컨테이너·CI 에 흔한 범용 CJK 폰트. 한글 자형이 전용 폰트만은
    # 못하지만 글자가 □ 로 깨지는 것보다는 낫다.
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/truetype/fonts-japanese-gothic.ttf",
]
_FONT_CACHE = {}


def _glyph(font, ch):
    """글자 하나의 비트맵을 바이트열로. 자형 유무 판별에 쓴다."""
    m = font.getmask(ch)
    return bytes(m)


def _korean_font(size):
    """한글을 그릴 수 있는 TTF 를 찾는다. 없으면 None."""
    key = int(size)
    if key in _FONT_CACHE:
        return _FONT_CACHE[key]
    try:
        from PIL import ImageFont
    except ImportError:
        _FONT_CACHE[key] = None
        return None
    paths = list(_KOREAN_FONT_CANDIDATES)
    for env in ("LASER_GRID_FONT",):
        if os.environ.get(env):
            paths.insert(0, os.environ[env])
    for fp in paths:
        if not os.path.exists(fp):
            continue
        try:
            f = ImageFont.truetype(fp, key)
            # 한글이 실제로 들어있는지 확인. getbbox 만으로는 부족하다 —
            # 자형이 없는 폰트도 .notdef(□) 를 그리며 폭을 돌려준다.
            # 서로 다른 두 글자가 같은 비트맵이면 둘 다 □ 라는 뜻이다.
            if f.getbbox("벽")[2] > 0 and _glyph(f, "벽") != _glyph(f, "철"):
                _FONT_CACHE[key] = f
                return f
        except Exception:
            continue
    _FONT_CACHE[key] = None
    return None


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


# =====================================================================
# 1. 검측 조서
# =====================================================================
def build_record(result, meta=None):
    """
    측정 결과를 부재별 조서 형태로 정리한다.

    Parameters
    ----------
    result : pipeline_region.inspect_image / inspect_capture 결과
    meta   : dict — 촬영 정보 (현장명, 촬영시각, 장비 시리얼 등)

    Returns
    -------
    dict — meta, summary, members[], caveats[]
    """
    members, caveats = [], []
    for i, r in enumerate(result.get("regions", [])):
        cls = r.get("class")
        item = {"no": i + 1,
                "부재": CLASS_KO.get(cls, cls),
                "class": cls,
                "검측항목": KIND_KO.get(r.get("kind"), r.get("kind")),
                "점수": r.get("n_points"),
                "상태": r.get("status")}
        if r["status"] != "measured":
            item["사유"] = r.get("reject_reason")
            members.append(item)
            continue

        j = r.get("judge") or {}
        item["측정각도_deg"] = r.get("theta_deg")
        item["판정"] = "합격" if j.get("is_pass") else "기준초과"
        item["기준"] = ({"방식": "mm", "편차_mm": j.get("deviation_mm"),
                        "허용_mm": j.get("allow_mm"),
                        "부재길이_m": j.get("member_length_m"),
                        "권장_h/1000_mm": j.get("allow_mm_recommended")}
                       if j.get("basis") == "mm"
                       else {"방식": "각도", "허용_deg": j.get("allow_deg")})
        if j.get("measured_span_m") is not None:
            item["기준"]["측정구간_m"] = j.get("measured_span_m")
            item["기준"]["측정구간편차_mm"] = j.get("span_deviation_mm")
        if j.get("note"):
            item["기준"]["비고"] = j["note"]
            caveats.append(f"{item['부재']}(#{item['no']}) {j['note']}")

        u = r.get("uncertainty") or {}
        item["불확실도"] = {"측정거리_m": u.get("z_mean_m"),
                        "입사각_deg": u.get("incidence_deg"),
                        "깊이_sigma_mm": u.get("sigma_z_mm"),
                        "법선방향_sigma_mm": u.get("sigma_normal_mm")}

        f = r.get("flatness") or {}
        if f.get("applicable"):
            item["평활도"] = {
                "판정": f.get("judgement"),
                "직선자_m": f.get("straightedge_length_m"),
                "처짐량_mm": f.get("max_gap_mm"),
                "처짐량상한_mm": f.get("upper_estimate_mm"),
                "허용_mm": f.get("tolerance_mm"),
                "요철깊이_mm": f.get("defect_max_dev_mm"),
                "요철개수": f.get("defect_clusters"),
                "비고": f.get("note")}
            if f.get("judgement") in ("측정불가", "판정보류(분해능)"):
                caveats.append(
                    f"{item['부재']}(#{item['no']}) 평활도 {f['judgement']}: "
                    f"{f.get('note') or ''}")
        else:
            item["평활도"] = {"판정": "해당없음",
                          "사유": (f.get("reason")
                                 or "선형 부재는 평활도가 정의되지 않음")}

        lf = r.get("label_fusion") or {}
        item["라벨"] = {"세그멘테이션": lf.get("semantic_class"),
                      "기하형상": lf.get("geom_shape"),
                      "채택": lf.get("source"),
                      "일치": lf.get("agreed")}
        if lf.get("source") == "geometric":
            caveats.append(f"{item['부재']}(#{item['no']}) 라벨 교정: "
                           f"{r.get('label_fusion_note')}")
        if r.get("from_split"):
            caveats.append(f"{item['부재']}(#{item['no']}) 는 병합된 영역을 "
                           f"기하로 되쪼개 얻은 영역")
        members.append(item)

    s = result.get("summary", {})
    measured = [m for m in members if m["상태"] == "measured"]
    fails = [m for m in measured if m.get("판정") == "기준초과"]
    flat_fail = [m for m in measured
                 if (m.get("평활도") or {}).get("판정") == "기준초과"]
    pending = [m for m in measured
               if (m.get("평활도") or {}).get("판정")
               in ("측정불가", "판정보류(분해능)")]

    return {
        "meta": dict(meta or {}),
        "summary": {
            "영역수": s.get("n_regions"),
            "검측": s.get("n_measured"),
            "기각": s.get("n_rejected"),
            "라벨교정": s.get("label_corrections"),
            "영역재분할": s.get("regions_split"),
            "선형부재정제": s.get("linear_members_rescued"),
            "자세_기준초과": len(fails),
            "평활도_기준초과": len(flat_fail),
            "평활도_판정보류": len(pending),
            "세그멘테이션": (result.get("segmentation") or {}).get("backend"),
            "중력벡터": result.get("gravity_laser_frame"),
        },
        "members": _jsonable(members),
        "caveats": caveats,
    }


def save_record(record, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(_jsonable(record), fp, ensure_ascii=False, indent=2)
    return path


# =====================================================================
# 2. 요약 표
# =====================================================================
def format_record(record):
    """조서를 사람이 읽는 표로 만든다."""
    L = []
    s = record["summary"]
    m = record["meta"]
    if m:
        L.append("  " + "  ".join(f"{k}: {v}" for k, v in m.items()))
    L.append(f"  세그멘테이션 {s.get('세그멘테이션')}   "
             f"영역 {s.get('영역수')}개 (검측 {s.get('검측')} / "
             f"기각 {s.get('기각')})   "
             f"라벨교정 {s.get('라벨교정')}  재분할 {s.get('영역재분할')}  "
             f"선형정제 {s.get('선형부재정제')}")
    g = s.get("중력벡터")
    if g:
        L.append(f"  중력벡터 ĝ = ({g[0]:.3f}, {g[1]:.3f}, {g[2]:.3f})")
    L.append("")

    hdr = (f"  {'#':<3}{'부재':<12}{'검측항목':<11}{'측정':>9}{'판정':>9}"
           f"{'자처짐':>9}{'허용':>7}{'평활판정':>12}{'σn(mm)':>8}{'점수':>7}")
    L.append(hdr)
    L.append("  " + "─" * (len(hdr) - 2))
    for it in record["members"]:
        if it["상태"] != "measured":
            L.append(f"  {it['no']:<3}{it['부재']:<12}{'기각':<11}"
                     f"{'-':>9}{'-':>9}{'-':>9}{'-':>7}{'-':>12}{'-':>8}"
                     f"{it['점수']:>7}   ← {it.get('사유')}")
            continue
        f = it.get("평활도") or {}
        gap = f.get("처짐량_mm")
        tol = f.get("허용_mm")
        L.append(
            f"  {it['no']:<3}{it['부재']:<12}{it['검측항목']:<11}"
            f"{it['측정각도_deg']:>9.4f}{it['판정']:>9}"
            f"{(f'{gap:.2f}' if gap is not None else '-'):>9}"
            f"{(f'{tol:.0f}' if tol is not None else '-'):>7}"
            f"{f.get('판정', '-'):>12}"
            f"{it['불확실도']['법선방향_sigma_mm']:>8.2f}{it['점수']:>7}")
        b = it.get("기준") or {}
        if b.get("방식") == "mm":
            L.append(f"      ↳ 부재길이 {b.get('부재길이_m')}m → 편차 "
                     f"{b.get('편차_mm')}mm / 허용 {b.get('허용_mm')}mm"
                     + (f" (권장 h/1000 {b.get('권장_h/1000_mm')}mm)"
                        if b.get("권장_h/1000_mm") is not None else ""))
        elif b.get("측정구간_m") is not None:
            L.append(f"      ↳ 각도 판정 (허용 ±{b.get('허용_deg')}°). "
                     f"측정구간 {b['측정구간_m']}m 기준 편차 "
                     f"{b.get('측정구간편차_mm')}mm — 부재 전체 길이를 알면 "
                     f"mm 판정 가능")

    if record["caveats"]:
        L.append("")
        L.append("  주의 사항")
        for c in record["caveats"]:
            L.append(f"    · {c}")

    fails = s.get("자세_기준초과", 0) + s.get("평활도_기준초과", 0)
    pend = s.get("평활도_판정보류", 0)
    L.append("")
    if fails == 0 and pend == 0:
        L.append("  종합: 전 부재 합격")
    else:
        L.append(f"  종합: 기준초과 {fails}건, 판정보류 {pend}건 "
                 f"— 판정보류는 재측정 필요")
    return "\n".join(L)


# =====================================================================
# 3. 오버레이 이미지
# =====================================================================
def save_segmentation(path, result, base_image=None, shape=None,
                      point_px=None, dim=0.62, show_defects=True,
                      uv_transform=None):
    """
    세그멘테이션 결과 이미지 — 색깔별로 무엇을 무엇으로 구분했는지.

    격자점을 클래스 색으로 찍고 범례를 얹는다. 화소 단위 마스크가 아니라
    점을 찍는 이유는, 기하 전용 백엔드에는 애초에 화소 마스크가 없고
    (3D 점에만 라벨이 붙는다) 실제 검측에 들어간 것도 그 점들이기 때문이다.
    마스크를 그리면 "칠해졌지만 검측에는 안 쓰인 화소" 가 생겨 결과를
    실제보다 넓어 보이게 만든다.

    base_image 는 레이저 OFF 프레임을 권장한다. dim 으로 밝기를 조절한다.
    너무 어둡게 깔면 점은 잘 보이지만 어느 부재 위에 찍힌 것인지 알 수
    없어진다 — 결과를 확인하려면 원본이 함께 보여야 한다.

    show_defects 가 참이면 검출된 요철 덩어리를 원과 깊이 값으로 표시한다.
    "요철 2곳 검출" 이라는 숫자만으로는 어디를 다시 봐야 할지 알 수 없다.

    uv_transform 은 검측 좌표를 배경 이미지 좌표로 옮기는 함수다. 내보내기
    화소 규약이 코드와 다를 때(180° 돌아 있는 캡처) 배경을 리샘플링해
    돌리는 대신 그릴 좌표만 되돌린다. 리샘플링은 반전 중심이 화소 격자에
    딱 떨어지지 않으면 그만큼 어긋나기 때문이다.
    """
    def _tf(arr):
        arr = np.asarray(arr, float)
        return arr if uv_transform is None else np.asarray(uv_transform(arr),
                                                           float)
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return None

    if base_image is not None:
        arr = np.asarray(base_image)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, -1)
        img = (arr[:, :, :3].astype(np.float32) * dim).astype(np.uint8)
    elif shape is not None:
        img = np.full((shape[0], shape[1], 3), 24, np.uint8)
    else:
        return None

    im = Image.fromarray(img).convert("RGB")
    W, H = im.size
    d = ImageDraw.Draw(im)
    rad = point_px if point_px else max(2, int(round(W / 700.0)))

    counts = {}
    for r in result.get("regions", []):
        uv = r.get("point_uv")
        if uv is None:
            continue
        cls = r.get("class")
        col = CLASS_COLOR.get(cls, (200, 200, 200))
        if r.get("status") != "measured":
            col = tuple(int(c * 0.45) for c in col)     # 기각 영역은 어둡게
        counts[cls] = counts.get(cls, 0) + len(uv)
        for u, v in _tf(uv):
            d.ellipse([u - rad, v - rad, u + rad, v + rad], fill=col)

    # ── 요철 위치 ──
    fsize = max(14, int(round(H / 48.0)))
    font = _korean_font(fsize)
    ko = font is not None
    n_def = 0
    if show_defects:
        for r in result.get("regions", []):
            f = r.get("flatness") or {}
            for k, dd in enumerate(f.get("defects") or []):
                n_def += 1
                (x0, y0), (x1, y1) = _tf([dd["bbox_px"][:2],
                                          dd["bbox_px"][2:]])
                x0, x1 = min(x0, x1), max(x0, x1)
                y0, y1 = min(y0, y1), max(y0, y1)
                cxp, cyp = _tf([dd["center_px"]])[0]
                # 반지름은 덩어리 크기에 맞추되 너무 작아지지 않게 둔다
                rad = max(np.hypot(x1 - x0, y1 - y0) / 2.0, W / 90.0)
                col = DEFECT_COLOR
                for wdt, off in ((max(3, W // 500), 0), (max(2, W // 800), 6)):
                    d.ellipse([cxp - rad - off, cyp - rad - off,
                               cxp + rad + off, cyp + rad + off],
                              outline=col, width=wdt)
                tick = rad * 0.35
                d.line([cxp - tick, cyp, cxp + tick, cyp], fill=col, width=2)
                d.line([cxp, cyp - tick, cxp, cyp + tick], fill=col, width=2)
                txt = f"요철 {dd['depth_mm']:.1f}mm" if ko \
                    else f"defect {dd['depth_mm']:.1f}mm"
                tw = int(d.textlength(txt, font=font)) if ko else len(txt) * 6
                # 라벨이 화면 밖으로 나가지 않게 붙일 쪽을 고른다.
                # 요철이 가장자리에 있을 때가 오히려 흔하다(면 경계).
                tx = cxp + rad + 12
                if tx + tw + 10 > W:
                    tx = cxp - rad - 12 - tw
                tx = min(max(tx, 8), W - tw - 8)
                ty = min(max(cyp - fsize, 8), H - fsize - 8)
                if ko:
                    d.rectangle([tx - 6, ty - 4, tx + tw + 6, ty + fsize + 6],
                                fill=(18, 18, 22))
                    d.text((tx, ty), txt, fill=col, font=font)
                else:
                    d.text((tx, ty), txt, fill=col)

    # ── 범례 ──
    rows = [(CLASS_COLOR.get(c, (200,) * 3),
             f"{CLASS_KO.get(c, c) if ko else CLASS_EN.get(c, c)}  "
             f"({c}, {n:,}점)" if ko else
             f"{CLASS_EN.get(c, c)} ({c}, {n} pts)")
            for c, n in sorted(counts.items(), key=lambda kv: -kv[1])]
    if n_def:
        rows.append((DEFECT_COLOR,
                     f"요철 {n_def}곳 (원 표시)" if ko
                     else f"defects: {n_def} (circled)"))
    if rows:
        if ko:
            pad, sw, rh = fsize // 2, fsize, int(fsize * 1.5)
            lw = pad * 2 + sw + pad + max(
                int(d.textlength(t, font=font)) for _, t in rows)
            lh = pad * 2 + rh * len(rows)
            d.rectangle([10, 10, 10 + lw, 10 + lh], fill=(18, 18, 22))
            y = 10 + pad
            for col, txt in rows:
                d.rectangle([10 + pad, y + 3, 10 + pad + sw, y + rh - 6],
                            fill=col)
                d.text((10 + pad + sw + pad, y), txt, fill=(240, 240, 240),
                       font=font)
                y += rh
        else:
            scale = max(1, int(round(W / 700.0)))
            pad, sw, rh = 4, 12, 13
            lw = 8 + sw + 4 + max(len(t) for _, t in rows) * 6 + 8
            leg = Image.new("RGB", (int(lw), pad * 2 + rh * len(rows)),
                            (18, 18, 22))
            dl = ImageDraw.Draw(leg)
            y = pad
            for col, txt in rows:
                dl.rectangle([6, y + 2, 6 + sw, y + rh - 3], fill=col)
                dl.text((6 + sw + 4, y + 2), txt, fill=(240, 240, 240))
                y += rh
            leg = leg.resize((leg.width * scale, leg.height * scale),
                             Image.NEAREST)
            im.paste(leg, (10, 10))

    dirn = os.path.dirname(os.path.abspath(path))
    if dirn:
        os.makedirs(dirn, exist_ok=True)
    im.save(path)
    return path


def save_overlay(path, result, table_uv=None, label_map=None,
                 base_image=None, class_names=None):
    """
    검측 결과를 이미지 위에 그린다.

    label_map 이 있으면 영역을 클래스 색으로 반투명 칠하고, 격자점은
    판정 색(합격 초록 / 기준초과 빨강 / 보류 노랑)으로 찍는다.

    base_image 는 레이저 OFF 프레임을 권장한다. 격자선이 없어 영역 경계가
    잘 보인다.
    """
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return None
    if label_map is None and base_image is None:
        return None

    if base_image is not None:
        arr = np.asarray(base_image)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, -1)
        img = arr[:, :, :3].astype(np.float32)
    else:
        img = np.zeros((*label_map.shape[:2], 3), np.float32)

    # 영역 색칠
    if label_map is not None and class_names:
        tint = np.zeros_like(img)
        for cid, cls in class_names.items():
            m = (label_map == cid)
            if not m.any():
                continue
            tint[m] = CLASS_COLOR.get(cls, (110, 110, 110))
        img = img * 0.62 + tint * 0.38

    im = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).convert("RGB")
    W, H = im.size

    # 범례 — 2448px 이미지에 기본 폰트로 그리면 글씨가 보이지 않는다.
    # 작은 캔버스에 그린 뒤 정수배로 확대해 붙인다(폰트 파일 의존 없음).
    fsize = max(14, int(round(H / 55.0)))
    font = _korean_font(fsize)
    ko = font is not None

    rows = []
    for r in result.get("regions", []):
        if r["status"] != "measured":
            continue
        cls = r["class"]
        vd = "합격" if (r.get("judge") or {}).get("is_pass") else "기준초과"
        f = r.get("flatness") or {}
        if ko:
            txt = (f"{CLASS_KO.get(cls, cls)} {KIND_KO.get(r['kind'], '')} "
                   f"{r['theta_deg']:.3f}° {vd}")
            if f.get("applicable"):
                txt += (f"  |  평활 {f.get('max_gap_mm', 0):.1f}mm "
                        f"{f.get('judgement')}")
        else:
            txt = (f"{CLASS_EN.get(cls, cls)} {KIND_EN.get(r['kind'], '')} "
                   f"{r['theta_deg']:.3f}deg {VERDICT_EN.get(vd, vd)}")
            if f.get("applicable"):
                txt += (f" | flat {f.get('max_gap_mm', 0):.1f}mm "
                        f"{VERDICT_EN.get(f.get('judgement'), '?')}")
        rows.append((CLASS_COLOR.get(cls, (200, 200, 200)), txt))

    if rows and ko:
        # 한글 TTF 가 있으면 원본 해상도에 그대로 그린다
        d = ImageDraw.Draw(im)
        pad, sw, rh = fsize // 2, fsize, int(fsize * 1.5)
        lw = pad * 2 + sw + pad + max(
            int(d.textlength(t, font=font)) for _, t in rows)
        lh = pad * 2 + rh * len(rows)
        d.rectangle([10, 10, 10 + lw, 10 + lh], fill=(18, 18, 22))
        y = 10 + pad
        for col, txt in rows:
            d.rectangle([10 + pad, y + 3, 10 + pad + sw, y + rh - 6], fill=col)
            d.text((10 + pad + sw + pad, y), txt, fill=(240, 240, 240),
                   font=font)
            y += rh
    elif rows:
        # 폰트가 없으면 작은 캔버스에 ASCII 로 그린 뒤 정수배 확대
        scale = max(1, int(round(W / 700.0)))
        pad, sw, rh = 4, 12, 13
        lw = 8 + sw + 4 + max(len(t) for _, t in rows) * 6 + 8
        lh = pad * 2 + rh * len(rows)
        leg = Image.new("RGB", (int(lw), int(lh)), (18, 18, 22))
        d = ImageDraw.Draw(leg)
        y = pad
        for col, txt in rows:
            d.rectangle([6, y + 2, 6 + sw, y + rh - 3], fill=col)
            d.text((6 + sw + 4, y + 2), txt, fill=(240, 240, 240))
            y += rh
        leg = leg.resize((leg.width * scale, leg.height * scale),
                         Image.NEAREST)
        im.paste(leg, (10, 10))
    dr = ImageDraw.Draw(im)

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    im.save(path)
    return path


# ============ 자체 검증 ============
if __name__ == "__main__":
    _SYN = _load("synth_scene")
    _PIPE = _load("pipeline_region")

    print("=" * 78)
    print("영역별 검측 조서 — 합성 씬")
    print("=" * 78)
    scene = _SYN.build_scene()
    res = _PIPE.inspect_capture(
        scene["lines_pixels"], scene["line_angles"],
        scene["camera_params"], scene["R_world_cam"],
        label_map=scene["label_map"], id_to_semantic=scene["id_to_semantic"],
        rgb_off=scene["rgb_off"], backend="gt")

    rec = build_record(res, meta={"현장": "합성 검증 씬",
                                  "측정거리_m": 1.2, "기선_mm": 150})
    print(format_record(rec))

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "_report_demo")
    p1 = save_record(rec, os.path.join(out_dir, "inspection_record.json"))
    cn, _ = _load("C_영역분할").map_isaac_labels(scene["id_to_semantic"])
    p2 = save_overlay(os.path.join(out_dir, "overlay.png"), res,
                      label_map=scene["label_map"], class_names=cn,
                      base_image=scene["rgb_off"])
    print(f"\n  조서 → {p1}")
    print(f"  오버레이 → {p2}")
