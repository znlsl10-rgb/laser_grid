#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
colab_run.py — 코랩에서 이 파일 하나만 돌리면 끝
========================================================================
코랩 셀에 두 줄:

    !wget -q -O colab_run.py "https://raw.githubusercontent.com/znlsl10-rgb/laser_grid/main/colab_run.py?v=$RANDOM"
    %run colab_run.py

무엇을 하나
----------
  1. 저장소를 **매번 최신본으로** 맞춘다 (없으면 clone, 있으면 fetch+reset).
     코드가 고쳐지면 다음 실행에서 그대로 반영된다.
  2. 이 파일 자신도 최신본으로 갈아탄다. 저장소의 colab_run.py 가
     지금 도는 것과 다르면 **새 것으로 다시 실행** 한다 — 부트스트랩만
     낡은 채로 남는 일이 없다.
  3. 의존성과 한글 폰트를 깔고,
  4. 올려 둔 파일에서 입력을 **알아서 찾아** (이름이 달라도 내용으로 가른다),
  5. 파이프라인을 돌려 엑셀 조서 하나와 그림들을 낸다.

입력 (레이저 이미지만 필수)
--------------------------
  레이저 격자 이미지   필수    아무 png/jpg
  camera_params.json  선택    없으면 사양 프로파일 값
  cast_pixels.json    선택    있으면 선검출·깊이 오차를 대조
  IMU (json)          선택    없으면 장비가 똑바로 섰다고 가정
  장면 사진(레이저 OFF) 선택    없으면 레이저 이미지에서 선을 지워 배경으로

파일은 왼쪽 파일 탭에 끌어다 놓아도 되고 files.upload() 로 올려도 된다.
둘 다 찾는다.

직접 지정하고 싶으면
-------------------
    %run colab_run.py --image 내이미지.png --out /content/결과 --views iso

파이썬에서 부르려면
------------------
    import colab_run
    res = colab_run.main(image='CAST.png')      # dict 반환
========================================================================
"""
import os
import re
import sys
import json
import glob
import shutil
import argparse
import subprocess

REPO_URL = "https://github.com/znlsl10-rgb/laser_grid.git"
BRANCH = "main"
REPO_DIR = "/content/laser_grid" if os.path.isdir("/content") else \
    os.path.join(os.getcwd(), "laser_grid")
# 입력을 찾아볼 곳 — 파일 탭(/content), 업로드 위치(작업폴더), 저장소 안
SEARCH_DIRS = ["/content", os.getcwd(), REPO_DIR]

_IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def _sh(cmd, **kw):
    """조용히 실행하고 (성공?, 출력) 을 돌려준다."""
    p = subprocess.run(cmd, shell=isinstance(cmd, str),
                       capture_output=True, text=True, **kw)
    return p.returncode == 0, (p.stdout + p.stderr).strip()


def _in_colab():
    try:
        import google.colab                                  # noqa: F401
        return True
    except Exception:
        return False


# ============ 1. 저장소를 최신으로 ============
def sync_repo(repo_dir=REPO_DIR, url=REPO_URL, branch=BRANCH, verbose=True):
    """
    없으면 clone, 있으면 **강제로** 원격에 맞춘다.

    pull 이 아니라 fetch + reset --hard 를 쓴다. 코랩에서 파일을 만지작
    거리다 pull 이 충돌로 막히면 "최신인 줄 알았는데 옛날 코드" 라는
    가장 나쁜 상태가 되기 때문이다. 이 폴더는 작업본이 아니라 사본이다.
    """
    def say(*x):
        if verbose:
            print(*x)

    if not os.path.isdir(os.path.join(repo_dir, ".git")):
        if os.path.exists(repo_dir):
            shutil.rmtree(repo_dir, ignore_errors=True)
        say(f"[저장소] 새로 내려받는 중 … {url}")
        ok, out = _sh(["git", "clone", "-q", "--branch", branch, url,
                       repo_dir])
        if not ok:
            raise RuntimeError(f"clone 실패:\n{out}")
    else:
        ok, out = _sh(["git", "-C", repo_dir, "fetch", "-q", "origin", branch])
        if not ok:
            say(f"[저장소] fetch 실패 — 있는 사본으로 진행합니다\n{out}")
        else:
            _sh(["git", "-C", repo_dir, "reset", "-q", "--hard",
                 f"origin/{branch}"])
            _sh(["git", "-C", repo_dir, "clean", "-qfd"])
    _, head = _sh(["git", "-C", repo_dir, "log", "--oneline", "-1"])
    say(f"[저장소] {repo_dir}  @ {head}")
    return repo_dir


def _relaunch_if_stale(repo_dir):
    """
    저장소의 colab_run.py 가 지금 도는 것과 다르면 그 쪽으로 갈아탄다.

    부트스트랩 자체가 낡으면 "최신 코드로 돌렸다" 는 말이 거짓이 된다.
    한 번만 갈아타도록 환경변수로 막는다.
    """
    if os.environ.get("LASER_GRID_BOOTSTRAPPED") == "1":
        return False
    fresh = os.path.join(repo_dir, "colab_run.py")
    try:
        here = os.path.abspath(__file__)
    except NameError:                       # 셀에 붙여넣어 실행한 경우
        here = None
    if not os.path.exists(fresh) or (here and os.path.samefile(fresh, here)):
        return False
    try:
        new = open(fresh, encoding="utf-8").read()
        old = open(here, encoding="utf-8").read() if here else ""
    except Exception:
        return False
    if new == old:
        return False
    print("[부트스트랩] 저장소의 colab_run.py 가 더 최신입니다 — 그쪽으로 "
          "다시 실행합니다")
    os.environ["LASER_GRID_BOOTSTRAPPED"] = "1"
    g = {"__name__": "__main__", "__file__": fresh}
    exec(compile(new, fresh, "exec"), g)
    return g.get("res") or True


# ============ 2. 의존성·폰트 ============
def ensure_deps(repo_dir, verbose=True):
    missing = []
    for mod, pkg in (("numpy", "numpy"), ("PIL", "Pillow"),
                     ("openpyxl", "openpyxl"), ("matplotlib", "matplotlib"),
                     ("scipy", "scipy"), ("sklearn", "scikit-learn")):
        try:
            __import__(mod)
        except ImportError:
            missing.append(pkg)
    if missing:
        if verbose:
            print(f"[설치] {', '.join(missing)}")
        ok, msg = _sh([sys.executable, "-m", "pip", "install", "-q"]
                      + missing)
        if not ok and verbose:
            print(f"[설치] 실패 — 그래도 진행합니다\n{msg[:400]}")
    elif verbose:
        print("[설치] 필요한 패키지가 이미 있습니다")

    # 3D 그림의 한글 라벨 — 없으면 영문으로 떨어지므로 치명적이진 않다.
    if _in_colab() and not glob.glob(
            "/usr/share/fonts/truetype/nanum/Nanum*"):
        if verbose:
            print("[설치] 한글 폰트 (fonts-nanum)")
        _sh("apt-get -qq install -y fonts-nanum > /dev/null 2>&1")


# ============ 3. 입력 찾기 ============
def _looks_like_camera_params(d):
    return isinstance(d, dict) and (
        "camera" in d or "f_px" in d or "baseline_m" in d)


def _looks_like_truth(d):
    if not isinstance(d, dict) or not d:
        return False
    k = next(iter(d))
    return bool(re.fullmatch(r"[VH]\d+", str(k))) and isinstance(d[k], dict)


def _looks_like_imu(d):
    return isinstance(d, dict) and any(
        x in d for x in ("pitch_deg", "roll_deg", "gravity", "accel"))


def _laser_score(path, repo_dir=None):
    """
    레이저 이미지인가 — 레이저 신호가 강하게 뜨는 화소 비율.

    파일 이름에 기대지 않는다. CAST/CAM 같은 규약을 모르는 사람이 올린
    사진도 가려야 하고, 반대로 이름만 맞고 내용이 다른 경우도 있다.
    채널(초록·빨강·파랑)은 저장소의 laser_signal 이 이미지에서 판별한다.
    """
    try:
        from PIL import Image
        import numpy as np
        im = Image.open(path).convert("RGB")
        im.thumbnail((480, 480))
        a = np.asarray(im, np.float32)
        try:
            import laser_signal as LS
            s = LS.laser_signal(a)
        except Exception:                 # 저장소를 아직 못 붙였을 때
            s = a[:, :, 1] - 0.5 * (a[:, :, 0] + a[:, :, 2])
        hi = float(np.percentile(s, 99.5))
        med = float(np.median(s))
        if hi - med < 1e-6:
            return 0.0
        return float((s > med + 0.5 * (hi - med)).mean())
    except Exception:
        return -1.0


def find_inputs(dirs=None, image=None, verbose=True):
    """
    올려 둔 파일에서 입력을 가른다. 이름이 규약과 달라도 내용으로 본다.
    같은 후보가 여럿이면 레이저 신호가 가장 센 것을 레이저 이미지로,
    가장 약한 것을 장면 사진(레이저 OFF)으로 쓴다.
    """
    dirs = [d for d in (dirs or SEARCH_DIRS) if os.path.isdir(d)]
    seen, imgs, jsons = set(), [], []
    for d in dirs:
        for f in sorted(os.listdir(d)):
            p = os.path.join(d, f)
            if not os.path.isfile(p):
                continue
            rp = os.path.realpath(p)
            if rp in seen:
                continue
            seen.add(rp)
            low = f.lower()
            if low.endswith(_IMG_EXT):
                imgs.append(p)
            elif low.endswith(".json"):
                jsons.append(p)

    out = {"image": None, "params": None, "truth": None, "imu": None,
           "scene_image": None}

    # ── json 세 종류 가르기 ──
    for p in jsons:
        try:
            d = json.load(open(p, encoding="utf-8"))
        except Exception:
            continue
        if out["params"] is None and _looks_like_camera_params(d):
            out["params"] = p
        elif out["truth"] is None and _looks_like_truth(d):
            out["truth"] = p
        elif out["imu"] is None and _looks_like_imu(d):
            out["imu"] = p

    # ── 이미지 가르기 ──
    if image:
        cand = [image] if os.path.exists(image) else \
            [p for p in imgs if os.path.basename(p) == image]
        if not cand:
            raise FileNotFoundError(f"이미지를 못 찾았습니다: {image}")
        out["image"] = cand[0]
        rest = [p for p in imgs if os.path.realpath(p)
                != os.path.realpath(out["image"])]
    else:
        # 결과 폴더에 다시 들어간 산출물(_세그멘테이션 등)은 후보에서 뺀다
        drop = ("세그멘테이션", "3d점군", "점군", "선검출", "overlay", "_검측")
        imgs = [p for p in imgs
                if not any(k in os.path.basename(p).lower() for k in drop)]
        if not imgs:
            raise FileNotFoundError(
                "레이저 격자 이미지를 못 찾았습니다. 왼쪽 파일 탭에 끌어다 "
                "놓거나 files.upload() 로 올린 뒤 다시 실행하세요.")
        scored = sorted(((_laser_score(p), p) for p in imgs), reverse=True)
        out["image"] = scored[0][1]
        rest = [p for _s, p in scored[1:]]
        if verbose and len(scored) > 1:
            print("[입력] 레이저 신호 세기로 고름: " + ", ".join(
                f"{os.path.basename(p)} {s:.3f}" for s, p in scored[:4]))
    if rest:
        # 남은 이미지 중 신호가 가장 약한 것이 레이저 OFF 사진일 가능성
        s_rest = sorted((_laser_score(p), p) for p in rest)
        if s_rest and s_rest[0][0] >= 0:
            out["scene_image"] = s_rest[0][1]

    if verbose:
        print("[입력]")
        for k, ko in (("image", "레이저 이미지 (필수)"),
                      ("params", "카메라 사양"), ("truth", "정답값"),
                      ("imu", "IMU"), ("scene_image", "장면 사진(OFF)")):
            v = out[k]
            print(f"   {ko:<20} {os.path.basename(v) if v else '— 없음'}")
        if out["imu"] is None:
            print("   ↳ IMU 가 없어 **장비가 똑바로 서 있다고 가정** 합니다")
    return out


# ============ 4. 실행 ============
def main(image=None, out=None, params=None, truth=None, imu=None,
         scene_image=None, repo_dir=REPO_DIR, views="quad", pc_stride=20,
         show=True, verbose=True, sync=True):
    """
    최신 코드로 검측을 한 번 돌린다.

    Returns
    -------
    dict — run_pipeline.run 의 결과 (xlsx, result, images, g_hat …)
    """
    if sync:
        repo_dir = sync_repo(repo_dir, verbose=verbose)
        fresh = _relaunch_if_stale(repo_dir)
        if fresh:
            # 새 부트스트랩이 이미 다 돌렸다. 그 결과를 그대로 넘긴다 —
            # 여기서 None 을 돌려주면 `res` 가 비어 뒷 셀이 깨진다.
            return None if fresh is True else fresh
        ensure_deps(repo_dir, verbose=verbose)

    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)

    # find_inputs 가 laser_signal 을 쓰므로 저장소가 경로에 들어간 뒤다.
    found = find_inputs(image=image, verbose=verbose)
    image = os.path.abspath(image or found["image"])
    params = os.path.abspath(params) if params else found["params"]
    truth = os.path.abspath(truth) if truth else found["truth"]
    scene_image = (os.path.abspath(scene_image) if scene_image
                   else found["scene_image"])
    imu_arg = None
    imu_path = os.path.abspath(imu) if imu else found["imu"]
    if imu_path:
        imu_arg = json.load(open(imu_path, encoding="utf-8"))

    out = out or ("/content/결과" if os.path.isdir("/content")
                  else os.path.join(os.getcwd(), "결과"))
    os.makedirs(out, exist_ok=True)

    from run_pipeline import run
    res = run(image=image, params=params, truth=truth, imu=imu_arg,
              scene_image=scene_image, out=out, pc_stride=pc_stride,
              verbose=verbose)

    # 3D 점군을 등각 한 장으로도 낸다 (조서에는 4분할이 들어간다)
    try:
        import plot_points3d as P3
        iso = os.path.join(out, "3D점군_등각.png")
        if P3.save_pointcloud_mpl(iso, res["result"], res["g_hat"],
                                  by="member", views="iso", elev=22,
                                  azim=-58, title="3D 점군 (부재별)"):
            res.setdefault("images", {})["3D점군(등각)"] = iso
    except Exception as e:
        if verbose:
            print(f"  [경고] 등각 점군 실패: {e}")

    if verbose:
        print()
        print("=" * 70)
        print(f"조서   : {res['xlsx']}")
        for k, v in (res.get("images") or {}).items():
            print(f"{k:<7}: {v}")
        print("=" * 70)
    if show:
        _show(res)
    return res


def _show(res):
    """노트북이면 표와 그림을 바로 띄운다. 아니면 조용히 넘어간다."""
    try:
        from IPython.display import display, Image
    except Exception:
        return
    # 판정표 한 장만 미리 보여 준다. 나머지는 조서를 열어 보면 된다 —
    # 셀에서 시트를 여럿 들추면 열 이름이 하나만 달라져도 통째로 죽는다.
    try:
        import pandas as pd
        df = pd.read_excel(res["xlsx"], sheet_name="6.검측결과")
        print("\n── 6.검측결과 ──")
        display(df)
    except Exception as e:
        print(f"  [알림] 표 미리보기 생략 ({type(e).__name__}: {e})")
    for k in ("세그멘테이션", "3D점군", "3D점군(등각)", "선검출대조"):
        p = (res.get("images") or {}).get(k)
        if p and os.path.exists(p):
            print(f"\n── {k} ──")
            display(Image(p, width=980))


def _cli(argv=None):
    ap = argparse.ArgumentParser(
        description="레이저 그리드 품질검측 — 코랩 원클릭 실행")
    ap.add_argument("--image", help="레이저 격자 이미지 (없으면 자동 탐색)")
    ap.add_argument("--params", help="camera_params.json")
    ap.add_argument("--truth", help="cast_pixels.json")
    ap.add_argument("--imu", help="IMU json")
    ap.add_argument("--scene-image", dest="scene_image",
                    help="레이저 OFF 사진")
    ap.add_argument("--out", help="결과 폴더 (기본 /content/결과)")
    ap.add_argument("--views", default="quad", choices=("quad", "iso"))
    ap.add_argument("--pc-stride", dest="pc_stride", type=int, default=20,
                    help="3D 좌표 시트에 N개마다 한 점 (기본 20)")
    ap.add_argument("--no-sync", dest="sync", action="store_false",
                    help="저장소를 다시 맞추지 않는다")
    ap.add_argument("--no-show", dest="show", action="store_false")
    ap.add_argument("--quiet", dest="verbose", action="store_false")
    # 노트북이 넘기는 -f kernel.json 같은 인자는 무시한다
    a, _unknown = ap.parse_known_args(argv)
    return main(**vars(a))


if __name__ == "__main__":
    # 모듈 수준에 남겨 둔다. 코랩에서 `%run colab_run.py` 로 돌리면 이
    # 이름이 노트북 네임스페이스로 들어와, 다음 셀에서 res['xlsx'] 처럼
    # 바로 이어 쓸 수 있다.
    res = _cli()
