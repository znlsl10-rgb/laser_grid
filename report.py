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
              "판정보류(분해능)": "HOLD",
              "판정보류(노출길이)": "HOLD",
              "판정보류(단면 미분해)": "HOLD", "해당없음": "n/a"}

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
        # judgement 가 최종 표기다. is_pass 만 보면 "판정을 하지 않음"
        # (is_pass=None) 이 "기준초과" 로 뒤바뀐다.
        item["판정"] = j.get("judgement") or ("합격" if j.get("is_pass")
                                            else "기준초과")
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

    counts, aux_counts, one_line = {}, {}, {}
    # 깊이를 못 주는 선(가로선)의 점을 **먼저** 깔고 그 위에 측정점을 얹는다.
    # 겹치는 자리에서는 측정점이 보여야 한다.
    for r in result.get("regions", []):
        auv = r.get("aux_point_uv")
        if auv is None or not len(auv):
            continue
        cls = r.get("class")
        col = CLASS_COLOR.get(cls, (200, 200, 200))
        # 같은 부재 색을 유지하되 어둡게 — 측정점과 구분되면서
        # 어느 부재에 걸린 가로선인지는 색으로 바로 읽힌다.
        col = tuple(int(c * 0.72) for c in col)
        aux_counts[cls] = aux_counts.get(cls, 0) + len(auv)
        ar = max(1, rad - 1)
        for u, v in _tf(auv):
            d.ellipse([u - ar, v - ar, u + ar, v + ar], fill=col)

    for r in result.get("regions", []):
        uv = r.get("point_uv")
        if uv is None:
            continue
        cls = r.get("class")
        col = CLASS_COLOR.get(cls, (200, 200, 200))
        if r.get("status") != "measured":
            col = tuple(int(c * 0.45) for c in col)     # 기각 영역은 어둡게
        counts[cls] = counts.get(cls, 0) + len(uv)
        if r.get("single_plane") or r.get("n_lines") == 1:
            one_line[cls] = one_line.get(cls, 0) + 1
        # 격자선 한 줄짜리 부재는 점이 가늘어 눈에 안 띈다. 굵게 찍어
        # "여기 부재가 있다" 는 것이 보이게 한다 — 판정은 보류지만
        # 검출은 됐다는 사실이 그림에서 읽혀야 한다.
        pr = rad + 1 if (r.get("single_plane") or r.get("n_lines") == 1) else rad
        for u, v in _tf(uv):
            d.ellipse([u - pr, v - pr, u + pr, v + pr], fill=col)

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
                # 반지름은 덩어리 크기에 맞추되 위아래로 묶어 둔다.
                # 위 한계가 없으면 넓게 퍼진 요철에서 원이 화면을 통째로
                # 덮고 십자 표시가 이미지를 가로질러, 어디가 요철인지
                # 오히려 안 보인다(실측: 반지름 1100px).
                rad = float(np.clip(np.hypot(x1 - x0, y1 - y0) / 2.0,
                                    W / 90.0, W / 12.0))
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
    rows = []
    for c, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        a = aux_counts.get(c, 0)
        nm = CLASS_KO.get(c, c) if ko else CLASS_EN.get(c, c)
        ol = one_line.get(c, 0)
        rows.append((CLASS_COLOR.get(c, (200,) * 3),
                     (f"{nm}  ({c}, 측정 {n:,}점"
                      + (f" + 가로 {a:,}점" if a else "")
                      + (f", {ol}본은 격자선 1줄 → 판정보류" if ol else "")
                      + ")") if ko
                     else f"{nm} ({c}, {n} pts"
                          + (f" +{a} aux" if a else "")
                          + (f", {ol} single-line" if ol else "") + ")"))
    if sum(aux_counts.values()):
        rows.append(((110, 110, 118),
                     "어두운 점 = 가로선 (깊이는 면에서 빌림, 검측 제외)" if ko
                     else "dim = horizontal lines (projected, not measured)"))
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
        _j = r.get("judge") or {}
        vd = _j.get("judgement") or ("합격" if _j.get("is_pass")
                                     else "기준초과")
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


# =====================================================================
# 3D 점군 — 부재별로 어디에 점이 찍혔는가
# =====================================================================
def world_frame(g_hat, forward=(0.0, 0.0, 1.0)):
    """
    조사기 좌표계를 중력 정렬 좌표계로 바꾸는 정규직교 기저.

    조사기 좌표(X 우, Y 하, Z 전방)를 그대로 그리면 장비를 숙인 각도만큼
    장면 전체가 기울어 보인다. 검측 대상은 벽·바닥이라 "수직/수평이 눈에
    보이는" 축이 아니면 그림을 읽을 수 없다.

    Returns
    -------
    (3,3) — 열이 [right, fwd, up]. P_world = P_laser @ R
    """
    up = -np.asarray(g_hat, float)
    up = up / np.linalg.norm(up)
    fwd = np.asarray(forward, float)
    fwd = fwd - up * float(fwd @ up)          # 수평면으로 정사영
    nf = np.linalg.norm(fwd)
    if nf < 1e-6:                              # 시선이 중력과 나란한 극단
        fwd = np.array([0.0, 0.0, 1.0]) - up * float(up[2])
        nf = np.linalg.norm(fwd)
    fwd /= nf
    right = np.cross(fwd, up)
    right /= np.linalg.norm(right)
    return np.column_stack([right, fwd, up])


def _iso_project(P, az_deg=38.0, el_deg=24.0):
    """등각 정사영. P 는 (N,3) [right, fwd, up]."""
    a, e = np.radians(az_deg), np.radians(el_deg)
    ca, sa, ce, se = np.cos(a), np.sin(a), np.cos(e), np.sin(e)
    # 화면 가로 = 수평 회전, 화면 세로 = 높이 + 깊이의 기울기 성분
    x = P[:, 0] * ca - P[:, 1] * sa
    y = P[:, 2] * ce - (P[:, 0] * sa + P[:, 1] * ca) * se
    depth = P[:, 0] * sa + P[:, 1] * ca       # 클수록 멀다 (먼저 그린다)
    return x, y, depth


def _panel_points(canvas, x, y, cols, box, pad, radius=1, sizes=None):
    """
    (x,y) 를 box 안에 등척(aspect 보존)으로 눌러 담아 canvas 에 찍는다.

    numpy 로 직접 찍는다. 3만 점을 PIL ellipse 로 하나씩 그리면 느리고,
    등척을 지키지 않으면 벽이 기울어 보여 그림이 거짓말을 한다.

    Returns
    -------
    (scale, ox, oy) — 같은 변환으로 다른 것(요철 표시)을 얹기 위해
    """
    x0, y0, x1, y1 = box
    w, h = (x1 - x0) - 2 * pad, (y1 - y0) - 2 * pad
    if len(x) == 0 or w <= 0 or h <= 0:
        return None
    xmin, xmax = float(np.min(x)), float(np.max(x))
    ymin, ymax = float(np.min(y)), float(np.max(y))
    sx = w / max(xmax - xmin, 1e-6)
    sy = h / max(ymax - ymin, 1e-6)
    s = min(sx, sy)                            # 등척
    ox = x0 + pad + (w - (xmax - xmin) * s) / 2.0 - xmin * s
    oy = y1 - pad - (h - (ymax - ymin) * s) / 2.0 + ymin * s
    px = np.rint(x * s + ox).astype(np.int64)
    py = np.rint(-y * s + oy).astype(np.int64)
    H, W = canvas.shape[:2]
    r = int(radius)
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if dx * dx + dy * dy > r * r + 1:
                continue
            qx, qy = px + dx, py + dy
            ok = (qx >= x0) & (qx < x1) & (qy >= y0) & (qy < y1) \
                & (qx >= 0) & (qx < W) & (qy >= 0) & (qy < H)
            canvas[qy[ok], qx[ok]] = cols[ok]
    return s, ox, oy


def save_pointcloud_3d(path, result, g_hat, size=(1900, 1500), title=None,
                       max_points_per_region=40000, show_defects=True):
    """
    삼각측량으로 얻은 3D 점을 **부재별 색으로** 그린다.

    왜 필요한가
    ----------
    검측 결과표는 "벽 수직도 0.03°" 같은 숫자만 준다. 그 숫자가 어느
    점들에서 나왔는지, 부재가 제대로 갈렸는지, 점이 실제로 벽·바닥·동바리
    모양으로 놓였는지는 3D 로 봐야 안다. 세그멘테이션 이미지는 화소 위에
    찍은 그림이라 깊이가 안 보인다 — 바닥과 벽이 이미지에서는 붙어 있어도
    3D 에서는 직각으로 갈라진다.

    좌표계
    ------
    중력 정렬 좌표로 바꿔 그린다(world_frame). 가로=수평 좌우, 세로=높이,
    깊이=장비에서 멀어지는 방향. 장비를 숙이고 찍었어도 벽은 수직으로,
    바닥은 수평으로 보인다.

    네 개 화면
    ---------
      등각    : 전체 배치를 한눈에
      평면도  : 위에서 내려다본 배치 — 부재 사이 거리·가림이 보인다
      정면도  : 카메라가 보는 방향 — 세그멘테이션 이미지와 대응된다
      측면도  : 옆에서 — 벽이 정말 수직인지, 바닥이 수평인지

    각 화면은 등척(같은 배율)이라 길이를 눈으로 비교할 수 있고, 축척
    막대를 함께 그린다.
    """
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return None

    regs = [r for r in result.get("regions", [])
            if r.get("point_xyz") is not None and len(r["point_xyz"])]
    if not regs:
        return None

    R = world_frame(g_hat)
    W, H = int(size[0]), int(size[1])
    canvas = np.full((H, W, 3), 16, np.uint8)

    # 부재별 점·색
    Ps, Cs, counts, aux_counts, one_line = [], [], {}, {}, {}
    for r in regs:
        p = np.asarray(r["point_xyz"], float)
        if len(p) > max_points_per_region:      # 그림만 성기게, 통계는 전부
            idx = np.linspace(0, len(p) - 1, max_points_per_region).astype(int)
            p = p[idx]
        cls = r.get("class")
        col = np.array(CLASS_COLOR.get(cls, (200, 200, 200)), np.uint8)
        if r.get("status") != "measured":
            col = (col.astype(float) * 0.45).astype(np.uint8)
        Ps.append(p @ R)
        Cs.append(np.repeat(col[None, :], len(p), axis=0))
        counts[cls] = counts.get(cls, 0) + int(len(r["point_xyz"]))
        if r.get("single_plane") or r.get("n_lines") == 1:
            one_line[cls] = one_line.get(cls, 0) + 1

        # 가로선 점 — 면에서 거리를 빌려 온 유도값이다. 같은 부재 색을
        # 옅게 써서 "어디에 걸렸는지" 는 보이되 측정점과 구분되게 한다.
        a = r.get("aux_point_xyz")
        if a is not None and len(a):
            a = np.asarray(a, float)
            if len(a) > max_points_per_region:
                idx = np.linspace(0, len(a) - 1,
                                  max_points_per_region).astype(int)
                a = a[idx]
            acol = (col.astype(float) * 0.72).clip(0, 255).astype(np.uint8)
            Ps.append(a @ R)
            Cs.append(np.repeat(acol[None, :], len(a), axis=0))
            aux_counts[cls] = aux_counts.get(cls, 0) + int(
                len(r["aux_point_xyz"]))
    P = np.vstack(Ps); C = np.vstack(Cs)

    # 요철 3D 좌표 (있으면)
    dpts, ddep = [], []
    if show_defects:
        for r in regs:
            fl = r.get("flatness") or {}
            for dd in (fl.get("defects") or []):
                c3 = dd.get("center_xyz")
                if c3 is not None:
                    dpts.append(np.asarray(c3, float) @ R)
                    ddep.append(float(dd.get("depth_mm", 0.0)))
    D = np.array(dpts) if dpts else np.zeros((0, 3))

    fs = max(15, H // 78)
    font = _korean_font(fs)
    ko = font is not None
    im = Image.fromarray(canvas)
    d0 = ImageDraw.Draw(im)

    top = int(fs * 2.6)                         # 제목 줄
    n_rows = (len(counts) + (1 if len(D) else 0)
              + (1 if sum(aux_counts.values()) else 0))
    legend_h = int(fs * 1.9) * n_rows + fs * 3
    body_h = H - top - legend_h
    pw, ph = W // 2, body_h // 2
    panels = [
        ("등각", 0, 0), ("평면도 (위에서)", 1, 0),
        ("정면도 (카메라 방향)", 0, 1), ("측면도 (옆에서)", 1, 1)]
    panels_en = ["Isometric", "Top view", "Front view", "Side view"]

    arr = np.asarray(im).copy()
    boxes = {}
    for k, (nm, gx, gy) in enumerate(panels):
        x0 = gx * pw + 4; y0 = top + gy * ph + 4
        x1 = x0 + pw - 8; y1 = y0 + ph - 8
        boxes[k] = (x0, y0, x1, y1)
        if k == 0:
            px, py, dep = _iso_project(P)
            o = np.argsort(-dep)                # 먼 점 먼저 (painter)
            xs, ys, cc = px[o], py[o], C[o]
            unit = "m"
        else:
            ax = {1: (0, 1), 2: (0, 2), 3: (1, 2)}[k]
            xs, ys, cc = P[:, ax[0]], P[:, ax[1]], C
            unit = "m"
        tf = _panel_points(arr, xs, ys, cc, (x0, y0, x1, y1),
                           pad=int(fs * 2.2), radius=1)
        boxes[k] = ((x0, y0, x1, y1), tf)

    im = Image.fromarray(arr)
    d = ImageDraw.Draw(im)

    def _txt(xy, s, fill=(235, 235, 235), f=None):
        if ko:
            d.text(xy, s, fill=fill, font=f or font)
        else:
            d.text(xy, s, fill=fill)

    def _len(s, f=None):
        return int(d.textlength(s, font=f or font)) if ko else len(s) * 6

    ttl = title or ("3D 점군 — 부재별" if ko else "3D point cloud by member")
    _txt((16, fs // 2), ttl)
    n_aux = sum(aux_counts.values())
    sub = ((f"중력 정렬 좌표 · 점 {len(P):,}개"
            + (f" (측정 {len(P)-n_aux:,} + 가로 {n_aux:,})" if n_aux else "")
            + " · 등척")
           if ko else f"gravity-aligned, {len(P)} pts, equal scale")
    _txt((16 + _len(ttl) + fs, fs // 2 + 2), sub, fill=(150, 155, 165))

    axis_ko = {0: ("가로", "높이"), 1: ("가로", "깊이"),
               2: ("가로", "높이"), 3: ("깊이", "높이")}
    axis_en = {0: ("right", "up"), 1: ("right", "fwd"),
               2: ("right", "up"), 3: ("fwd", "up")}
    for k, (nm, gx, gy) in enumerate(panels):
        (x0, y0, x1, y1), tf = boxes[k]
        d.rectangle([x0, y0, x1, y1], outline=(58, 62, 72))
        _txt((x0 + 10, y0 + 6), nm if ko else panels_en[k], fill=(200, 205, 215))
        ax = axis_ko[k] if ko else axis_en[k]
        _txt((x1 - _len(ax[0]) - 12, y1 - fs - 8), ax[0], fill=(120, 126, 138))
        _txt((x0 + 10, y0 + fs + 10), ax[1], fill=(120, 126, 138))
        if tf is None:
            continue
        s, ox, oy = tf
        # 축척 막대 — 1m 가 몇 화소인지. 없으면 크기를 알 수 없다.
        for L in (1.0, 0.5, 0.2, 0.1):
            if L * s < (x1 - x0) * 0.6:
                break
        bx, by = x0 + 14, y1 - 14
        d.line([bx, by, bx + L * s, by], fill=(190, 195, 205), width=3)
        for e in (bx, bx + L * s):
            d.line([e, by - 5, e, by + 5], fill=(190, 195, 205), width=2)
        lab = f"{L:g} m"
        _txt((bx + L * s / 2 - _len(lab) / 2, by - fs - 8), lab,
             fill=(190, 195, 205))
        # 요철 위치
        if len(D):
            if k == 0:
                dx, dy, _ = _iso_project(D)
            else:
                a2 = {1: (0, 1), 2: (0, 2), 3: (1, 2)}[k]
                dx, dy = D[:, a2[0]], D[:, a2[1]]
            for j in range(len(D)):
                cxp, cyp = dx[j] * s + ox, -dy[j] * s + oy
                if not (x0 < cxp < x1 and y0 < cyp < y1):
                    continue
                rr = max(9, W // 150)
                d.ellipse([cxp - rr, cyp - rr, cxp + rr, cyp + rr],
                          outline=DEFECT_COLOR, width=3)
                d.line([cxp - rr * .5, cyp, cxp + rr * .5, cyp],
                       fill=DEFECT_COLOR, width=2)
                d.line([cxp, cyp - rr * .5, cxp, cyp + rr * .5],
                       fill=DEFECT_COLOR, width=2)

    # ── 범례 ──
    y = H - legend_h + fs
    _txt((16, y - fs - 2), "부재 구분" if ko else "members",
         fill=(150, 155, 165))
    for cls, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        col = CLASS_COLOR.get(cls, (200, 200, 200))
        a = aux_counts.get(cls, 0)
        d.rectangle([16, y + 4, 16 + fs, y + fs], fill=col)
        nm = (CLASS_KO.get(cls, cls) if ko else CLASS_EN.get(cls, cls))
        ol = one_line.get(cls, 0)
        _txt((16 + fs * 2, y),
             (f"{nm}  ({cls}, 측정 {n:,}점"
              + (f" + 가로 {a:,}점" if a else "")
              + (f", {ol}본은 격자선 1줄 → 판정보류" if ol else "") + ")") if ko
             else f"{nm} ({cls}, {n} pts" + (f" +{a} aux" if a else "")
                  + (f", {ol} single-line" if ol else "") + ")")
        y += int(fs * 1.9)
    if sum(aux_counts.values()):
        _txt((16 + fs * 2, y),
             ("어두운 점 = 가로선 — 깊이는 면에서 빌린 값이라 검측에는 안 쓴다"
              if ko else "faint = horizontal lines (projected, not measured)"),
             fill=(150, 155, 165))
        y += int(fs * 1.9)
    if len(D):
        d.ellipse([16, y + 3, 16 + fs, y + fs + 1], outline=DEFECT_COLOR,
                  width=2)
        mx = max(ddep) if ddep else 0.0
        _txt((16 + fs * 2, y),
             f"요철 {len(D)}곳 (최대 {mx:.1f}mm)" if ko
             else f"defects: {len(D)} (max {mx:.1f}mm)", fill=DEFECT_COLOR)

    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    im.save(path)
    return path


def region_xyz_rows(result, g_hat=None, stride=1):
    """
    부재별 3D 좌표를 표 한 장으로 편다 (엑셀·CSV 공용).

    좌표를 두 벌 낸다.
      조사기 좌표 (X,Y,Z) — 삼각측량이 직접 낸 값. 검측식이 쓴 그대로다.
      중력 좌표 (가로,깊이,높이) — 사람이 읽을 수 있는 값.
    중력 좌표의 원점은 **장비 광학중심** 이다. 높이가 음수면 장비보다
    아래라는 뜻이고, 절대 표고가 아니다. 표고로 바꾸려면 장비 설치 높이를
    더해야 한다 — 그 값은 이 장비가 알 수 없다.
    g_hat 이 없으면 중력 좌표는 비운다.

    stride 로 솎아 낸다. 3만 점을 그대로 시트에 넣으면 파일이 무거워지고
    사람이 읽지도 않는다. 통계(요약 행)는 항상 **전체 점**으로 낸다.
    """
    R = world_frame(g_hat) if g_hat is not None else None
    rows = []
    for i, r in enumerate(result.get("regions", [])):
        p = r.get("point_xyz")
        if p is None or not len(p):
            continue
        P = np.asarray(p, float)
        lid = r.get("point_lid")
        Pw = P @ R if R is not None else None
        for j in range(0, len(P), max(1, int(stride))):
            row = {"부재번호": i + 1,
                   "클래스": CLASS_KO.get(r.get("class"), r.get("class")),
                   "판정대상": KIND_KO.get(r.get("kind"), r.get("kind")),
                   "구분": "측정",
                   "선ID": (str(lid[j]) if lid is not None and j < len(lid)
                            else None),
                   "X_m": round(float(P[j, 0]), 5),
                   "Y_m": round(float(P[j, 1]), 5),
                   "Z_m": round(float(P[j, 2]), 5)}
            if Pw is not None:
                row.update({"가로_m": round(float(Pw[j, 0]), 5),
                            "깊이_m": round(float(Pw[j, 1]), 5),
                            "높이_m": round(float(Pw[j, 2]), 5)})
            rows.append(row)

        # 가로선 점 — 거리를 면에서 빌려 온 유도값. 좌표는 실제 화소에서
        # 나온 것이라 위치는 맞지만, 그 면에 대한 잔차가 정의상 0 이므로
        # 검측에는 쓰지 않았다. 구분 열로 표시한다.
        A = r.get("aux_point_xyz")
        if A is None or not len(A):
            continue
        A = np.asarray(A, float)
        Aw = A @ R if R is not None else None
        for j in range(0, len(A), max(1, int(stride))):
            row = {"부재번호": i + 1,
                   "클래스": CLASS_KO.get(r.get("class"), r.get("class")),
                   "판정대상": KIND_KO.get(r.get("kind"), r.get("kind")),
                   "구분": "가로선(유도값)",
                   "선ID": None,
                   "X_m": round(float(A[j, 0]), 5),
                   "Y_m": round(float(A[j, 1]), 5),
                   "Z_m": round(float(A[j, 2]), 5)}
            if Aw is not None:
                row.update({"가로_m": round(float(Aw[j, 0]), 5),
                            "깊이_m": round(float(Aw[j, 1]), 5),
                            "높이_m": round(float(Aw[j, 2]), 5)})
            rows.append(row)
    return rows


def region_xyz_summary(result, g_hat=None):
    """부재별 3D 요약 — 점수, 중심, 크기(외접 상자), 깊이 범위."""
    R = world_frame(g_hat) if g_hat is not None else None
    out = []
    for i, r in enumerate(result.get("regions", [])):
        p = r.get("point_xyz")
        if p is None or not len(p):
            continue
        P = np.asarray(p, float)
        Pw = P @ R if R is not None else P
        lid = r.get("point_lid")
        fams = {}
        if lid is not None:
            for l in np.unique(np.asarray(lid, dtype=object)):
                fams[str(l)[0]] = fams.get(str(l)[0], 0) + int(
                    np.sum(np.asarray(lid, dtype=object) == l))
        rec = {
            "부재번호": i + 1,
            "클래스": CLASS_KO.get(r.get("class"), r.get("class")),
            "검측": KIND_KO.get(r.get("kind"), r.get("kind")),
            "상태": {"measured": "검측함",
                   "rejected": "기각"}.get(r.get("status"), r.get("status")),
            "점수": int(len(P)),
            "가로선점수": int(len(r["aux_point_xyz"])
                       if r.get("aux_point_xyz") is not None else 0),
            "선수": r.get("n_lines"),
            "구간제한": bool(r.get("extent_limited")),
            "선형부재": bool(r.get("kind") == "axis_vertical"),
            "선구성": (" / ".join(f"{k} {v:,}점" for k, v in sorted(fams.items()))
                    or None),
            "깊이이득_RMS": r.get("depth_gain_rms"),
            # 분할 폭은 상수가 아니라 잰 값이다 — 어디서 왔는지 함께 남긴다.
            "부재반폭_mm": (round(float(r["half_width_m"]) * 1000.0, 1)
                       if r.get("half_width_m") else None),
            "폭_근거": r.get("width_source"),
            "가로선점_배치면": r.get("aux_surface"),
            "중심_X_m": round(float(P[:, 0].mean()), 4),
            "중심_Y_m": round(float(P[:, 1].mean()), 4),
            "중심_Z_m": round(float(P[:, 2].mean()), 4),
            "거리범위_m": [round(float(P[:, 2].min()), 3),
                       round(float(P[:, 2].max()), 3)],
        }
        if R is not None:
            # numpy 2.0 에서 ndarray.ptp() 가 없어졌다. np.ptp 를 쓴다.
            rec.update({
                "가로폭_m": round(float(np.ptp(Pw[:, 0])), 4),
                "깊이폭_m": round(float(np.ptp(Pw[:, 1])), 4),
                "측정구간_m": round(float(np.ptp(Pw[:, 2])), 4),
                "높이범위_m": [round(float(Pw[:, 2].min()), 3),
                           round(float(Pw[:, 2].max()), 3)]})
            # 이 값이 부재 길이인지, 격자가 스친 구간일 뿐인지 함께 낸다.
            # 이름을 '높이' 로 두면 같은 규격 동바리가 거리 때문에 서로
            # 다른 높이로 읽힌다 (실측 1.9938 / 1.9317 / 1.8652 m).
            lb = bool(r.get("length_is_lower_bound"))
            ends = r.get("extent_ends") or {}
            cut = [k for k, v in ends.items() if v]
            rec["길이확정"] = ("하한" if lb else "확정")
            rec["길이근거"] = (
                ("부재 전체가 측정 범위 안 — 이 값이 길이다" if not lb
                 else ("격자가 " + "·".join(cut) + " 끝에서 잘림 — 부재는 이보다 길다"
                       if cut else "격자·화면 경계에 닿음 — 이 값은 하한이다")))
        out.append(rec)
    return out


def save_pointcloud_csv(path, result, g_hat=None, stride=1):
    """부재별 3D 좌표를 CSV 로. CAD·측량 소프트로 그대로 넘어간다."""
    import csv
    rows = region_xyz_rows(result, g_hat=g_hat, stride=stride)
    if not rows:
        return None
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as fp:
        wr = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    return path
