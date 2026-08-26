#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
laser_signal.py — 사진에서 레이저만 남기는 한 곳
========================================================================
왜 따로 두나
-----------
"레이저는 초록이다" 라는 가정이 A_선검출·eq8·load_capture 세 곳에
따로 박혀 있었다. 값은 전부 G − (R+B)/2 로 같았지만, 세 곳이 각자
쓰고 있어 한 곳을 고쳐도 나머지가 옛 가정을 그대로 썼다.

더 큰 문제는 가정 자체다. 빨간 레이저(635·650nm)는 흔하고, 그때
G − (R+B)/2 는 **음수**가 되어 신호가 통째로 0 으로 잘린다. 실제로
A_선검출 은 그 경우 "신호 없음 → 기하 예측값 반환" 으로 빠져, 검출을
한 적이 없는데도 그럴듯한 좌표를 돌려준다. 조서에는 아무 경고도 안
남는다 — 가장 나쁜 실패다.

그래서 채널을 **재서** 고른다.

어떻게 고르나
------------
채널 c 의 과잉분을 정의한다.

    X_c = I_c − mean(다른 두 채널)

레이저가 단색이면 그 채널에서만 X 가 크게 뜬다. 세 채널의 상위 분위수
(99.5%)를 비교해 가장 큰 것을 고른다. 어느 채널도 뚜렷하지 않으면
(백색 레이저, 흑백 카메라, 과포화) 밝기에서 배경을 뺀 값으로 떨어진다.

판별은 **한 번만** 하고 결과를 함께 돌려준다. 검출과 그림자 읽기가
서로 다른 채널을 고르면 좌표계가 어긋나기 때문이다.

ON/OFF 두 프레임이 있으면 차영상을 쓴다. 배경광은 두 프레임에 공통이라
차분에서 지워지고 레이저만 남는다 — 햇빛이 드는 현장에서 특히 크다.
========================================================================
"""
import numpy as np

CHANNELS = ("R", "G", "B")


def _excess(a, c):
    """채널 c 의 과잉분 — 나머지 두 채널의 평균을 뺀다."""
    other = [k for k in range(3) if k != c]
    return a[:, :, c] - 0.5 * (a[:, :, other[0]] + a[:, :, other[1]])


def pick_channel(rgb, min_contrast=6.0, min_ratio=1.8):
    """
    어느 채널이 레이저인가. (이름, 인덱스, 점수, 근거) 를 돌려준다.

    점수는 과잉분의 **상위 분위수 − 중앙값** 이다. 두 가지를 동시에
    피하려고 이렇게 잡는다.

      · 평균이나 최대값으로 보면 안 된다. 선이 차지하는 화소는 전체의
        몇 %뿐이라 평균은 배경에 묻히고, 최대값은 핫픽셀 하나에 끌려간다.
      · 분위수만 봐도 안 된다. 벽 자체가 초록을 띠면 G 과잉분이 화면
        전체에서 높아 아무 선이 없어도 초록이 뽑힌다. 중앙값을 빼면
        그 색조가 상쇄되고 **선이 얹은 만큼** 만 남는다.

    두 문턱을 다 넘어야 단색으로 인정한다.

      · min_contrast : 절대 대비. 이보다 작으면 그냥 잡음이다.
      · min_ratio    : 나머지 두 채널 대비 배수. 백색 레이저·흑백 카메라
                       에서는 세 채널이 고르게 뜨는데, 절대값만 보면
                       잡음 중 가장 큰 채널이 뽑혀 엉뚱한 신호를 만든다
                       (실측: 세 채널 32.3/32.0/32.1 인데 R 이 뽑혔다).

    둘 중 하나라도 못 넘으면 None — 호출부가 밝기 기반으로 떨어진다.
    """
    a = np.asarray(rgb, dtype=np.float32)
    if a.ndim == 2:
        return None, None, 0.0, "흑백 이미지 — 채널 분리 없음"
    if a.shape[2] < 3:
        return None, None, 0.0, "채널이 3개 미만"
    scores = []
    for c in range(3):
        x = _excess(a, c)
        scores.append(float(np.percentile(x, 99.5) - np.median(x)))
    best = int(np.argmax(scores))
    rest = max(scores[c] for c in range(3) if c != best)
    why = (" / ".join(f"{CHANNELS[c]} {scores[c]:+.1f}" for c in range(3))
           + f", 배수 {scores[best] / max(rest, 1e-6):.1f}")
    if scores[best] < float(min_contrast):
        return None, None, scores[best], f"대비 부족 ({why})"
    if scores[best] < float(min_ratio) * max(rest, 1e-6):
        return None, None, scores[best], f"단색 아님 ({why})"
    return CHANNELS[best], best, scores[best], why


def _mono(a):
    """
    단색 채널이 안 잡힐 때 — 밝기에서 배경을 뺀다.

    선이 배경보다 어두운 스캔본도 있으므로 밝은 쪽·어두운 쪽 중 대비가
    큰 쪽을 쓴다.
    """
    g = a.mean(axis=2) if a.ndim == 3 else np.asarray(a, np.float32)
    med = float(np.median(g))
    up = float(np.percentile(g, 99.5)) - med
    dn = med - float(np.percentile(g, 0.5))
    return (g - med) if up >= dn else (med - g)


def laser_signal(rgb, off=None, channel=None, clip=False, info=False):
    """
    레이저만 남긴 2D 실수 배열.

    Parameters
    ----------
    off : (H,W,3) or None
        레이저 OFF 프레임. 주면 차영상(ON−OFF)으로 배경광을 지운다.
    channel : "R"|"G"|"B"|int|None
        고정하고 싶을 때. None 이면 이미지에서 판별한다.
    clip : bool
        음수를 0 으로 자를지. 선검출은 True, 그림자 읽기는 False 가
        맞다 — 그림자는 신호가 **없는** 곳이라 음수 쪽도 정보다.
    info : bool
        True 면 (signal, dict) 를 돌려준다.

    Returns
    -------
    (H, W) float32   (info=True 면 (signal, 판별내역))
    """
    a = np.asarray(rgb, dtype=np.float32)
    if channel is None:
        name, idx, score, why = pick_channel(a)
    else:
        idx = (CHANNELS.index(channel) if isinstance(channel, str)
               else int(channel))
        name, score, why = CHANNELS[idx], float("nan"), "호출부 지정"

    if idx is None:
        sig = _mono(a)
        if off is not None:
            sig = sig - _mono(np.asarray(off, np.float32))
        mode = "밝기(단색 채널 없음)"
    else:
        sig = _excess(a, idx)
        if off is not None:
            sig = sig - _excess(np.asarray(off, np.float32), idx)
        mode = f"{name}채널 과잉분" + (" (ON−OFF 차영상)" if off is not None
                                  else "")
    if clip:
        sig = np.clip(sig, 0, None)
    if not info:
        return sig
    return sig, {"channel": name, "index": idx, "score": round(float(score), 2)
                 if np.isfinite(score) else None, "mode": mode, "why": why}


def describe(rgb, off=None):
    """조서·로그에 한 줄로 남길 판별 결과."""
    _s, d = laser_signal(rgb, off=off, info=True)
    return (f"레이저 신호: {d['mode']}"
            + (f"  (채널 대비 {d['why']})" if d.get("why") else ""))


# ============ 자체 검증 ============
if __name__ == "__main__":
    rng = np.random.default_rng(7)
    H, W = 200, 300
    print("=" * 66)
    print("레이저 채널 자동 판별")
    print("=" * 66)
    ok = True
    for name, ci in (("초록 520nm", 1), ("빨강 650nm", 0), ("파랑 450nm", 2)):
        bg = rng.uniform(30, 70, (H, W, 3)).astype(np.float32)
        img = bg.copy()
        img[:, ::20, ci] += 180.0                    # 세로 레이저선
        img[::20, :, ci] += 180.0
        img = np.clip(img, 0, 255)
        got, idx, score, why = pick_channel(img)
        good = idx == ci
        ok &= good
        print(f"  {name:<12} → {got} (점수 {score:6.1f})  "
              f"{'PASS' if good else 'FAIL'}   [{why}]")
        sig = laser_signal(img, clip=True)
        lit = float((sig > 0.4 * sig.max()).mean())
        print(f"      신호 화소 비율 {lit:.3f}  (격자 20px 간격이면 ~0.10)")

    # 단색이 아닌 경우 — 흰 레이저
    bg = rng.uniform(30, 70, (H, W, 3)).astype(np.float32)
    img = bg.copy()
    img[:, ::20, :] += 180.0
    got, idx, score, why = pick_channel(np.clip(img, 0, 255))
    print(f"  {'백색 레이저':<12} → {got} — {why}")
    ok &= idx is None
    s = laser_signal(np.clip(img, 0, 255), clip=True)
    print(f"      밝기 경로 신호 최대 {s.max():.1f}  "
          f"{'PASS' if s.max() > 50 else 'FAIL'}")
    ok &= s.max() > 50

    # ON/OFF 차영상 — 배경광이 세도 레이저만 남아야 한다
    sun = rng.uniform(120, 200, (H, W, 3)).astype(np.float32)
    on = np.clip(sun.copy(), 0, 255)
    on[:, ::20, 1] = np.clip(on[:, ::20, 1] + 120.0, 0, 255)
    d = laser_signal(on, off=np.clip(sun, 0, 255), channel="G", clip=True)
    peak = float(d[:, ::20].mean())
    base = float(np.delete(d, np.arange(0, W, 20), axis=1).mean())
    print(f"  차영상: 선 위 {peak:.1f} / 그 밖 {base:.1f}  "
          f"{'PASS' if peak > 20 * max(base, 0.1) else 'FAIL'}")
    ok &= peak > 20 * max(base, 0.1)
    print("=" * 66)
    print("전체 통과" if ok else "실패 있음")
    raise SystemExit(0 if ok else 1)
