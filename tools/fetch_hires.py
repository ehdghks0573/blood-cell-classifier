"""고해상도 BloodMNIST 를 끈질기게 내려받는다.

    python tools/fetch_hires.py --size 224
    python tools/fetch_hires.py --size 224 --retries 200 --wait 60

`docs/results.md` 의 한계 목록 첫 줄이 "28px 저해상도"다. 224px 원본을 받으면
그 한계를 없앨 수 있는데, **Zenodo 가 자주 죽는다.** 실제로 이 프로젝트에서
두 번 막혔다.

medmnist 패키지의 내려받기를 쓰지 않는 이유는 세 가지다.

  1. **이어받기가 없다.** 2GB 를 받다 끊기면 처음부터다.
  2. **재시도가 없다.** 한 번 실패하면 끝이다. Zenodo 의 장애는 대개 일시적이라
     기다렸다 다시 걸면 되는데, 그걸 사람이 지켜보고 있어야 한다.
  3. **원인을 감춘다.** `except:` 로 전부 삼키고 "수동으로 받으세요"만 말한다.
     서버가 죽은 건지, 네트워크가 막힌 건지, 디스크가 찬 건지 알 수 없다.

주소와 MD5 는 medmnist 패키지에서 읽는다. 손으로 적어 두면 데이터셋이 갱신될 때
조용히 틀린 파일을 받게 된다.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

FLAG = "bloodmnist"
CHUNK = 1 << 20  # 1MB


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="고해상도 BloodMNIST 내려받기")
    p.add_argument("--size", type=int, default=224, choices=[64, 128, 224])
    p.add_argument("--out", default=str(ROOT / "data"))
    p.add_argument("--retries", type=int, default=60,
                   help="재시도 횟수. Zenodo 장애는 대개 몇 분~몇 시간이다")
    p.add_argument("--wait", type=int, default=60, help="재시도 간격(초)")
    p.add_argument("--timeout", type=int, default=60, help="연결 제한 시간(초)")
    return p.parse_args(argv)


def md5_of(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


def attempt(url: str, dest: Path, timeout: int) -> tuple[bool, str]:
    """한 번 시도한다. 이미 받은 만큼은 건너뛴다 (HTTP Range).

    돌려주는 값은 (완료했는가, 설명) 이다. 중간까지 받고 끊긴 것도
    실패가 아니라 **진전**이므로 다음 시도에서 이어 간다.
    """
    have = dest.stat().st_size if dest.exists() else 0
    req = urllib.request.Request(url, headers={"User-Agent": "blood-cell-classifier"})
    if have:
        req.add_header("Range", f"bytes={have}-")

    try:
        with urllib.request.urlopen(req, timeout=timeout) as res:
            # 206 이면 이어받기가 먹혔고, 200 이면 서버가 처음부터 다시 준다
            if have and res.status == 200:
                have = 0
                dest.unlink(missing_ok=True)

            total = res.headers.get("Content-Length")
            total = int(total) + have if total else None

            mode = "ab" if have else "wb"
            last = time.time()
            with dest.open(mode) as f:
                while True:
                    block = res.read(CHUNK)
                    if not block:
                        break
                    f.write(block)
                    have += len(block)
                    if time.time() - last > 5:
                        pct = f" ({have / total:.1%})" if total else ""
                        print(f"    {human(have)}{pct}", flush=True)
                        last = time.time()

        return True, f"{human(have)} 받음"

    except urllib.error.HTTPError as e:
        return False, f"HTTP {e.code} {e.reason}"
    except urllib.error.URLError as e:
        return False, f"연결 실패: {e.reason}"
    except TimeoutError:
        return False, "시간 초과"
    except OSError as e:
        return False, f"{type(e).__name__}: {e}"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    from medmnist import INFO

    info = INFO[FLAG]
    url = info[f"url_{args.size}"]
    want_md5 = info[f"MD5_{args.size}"]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dest = out / f"{FLAG}_{args.size}.npz"

    print(f"파일 : {dest.name}")
    print(f"주소 : {url}")
    print(f"MD5  : {want_md5}")

    if dest.exists():
        print(f"\n이미 있음 ({human(dest.stat().st_size)}). MD5 확인 중...")
        if md5_of(dest) == want_md5:
            print("MD5 일치. 받을 것이 없습니다.")
            return 0
        print("MD5 불일치 — 덜 받았거나 깨진 파일입니다. 이어서 받습니다.")

    for i in range(1, args.retries + 1):
        print(f"\n[{i}/{args.retries}] {time.strftime('%H:%M:%S')} 시도")
        done, why = attempt(url, dest, args.timeout)

        if done:
            print(f"  내려받기 끝 — {why}. MD5 확인 중...")
            got = md5_of(dest)
            if got == want_md5:
                print(f"\nMD5 일치. {dest}")
                print(f"이제 --size {args.size} 로 학습할 수 있습니다:")
                print(f"  python train.py --size {args.size} --input-size 224 "
                      f"--epochs 20 --name hires_{args.size}")
                return 0

            # 서버가 잘린 파일을 완전한 것처럼 줬을 수 있다. 지우고 다시.
            print(f"  MD5 불일치 (받은 것 {got}). 파일을 지우고 다시 받습니다.")
            dest.unlink(missing_ok=True)
        else:
            got = dest.stat().st_size if dest.exists() else 0
            print(f"  실패: {why}" + (f" · 지금까지 {human(got)}" if got else ""))

        if i < args.retries:
            print(f"  {args.wait}초 뒤 다시 시도합니다.")
            time.sleep(args.wait)

    print(f"\n{args.retries}번 시도했지만 받지 못했습니다.")
    print("Zenodo 장애일 가능성이 큽니다. 브라우저에서 직접 받아")
    print(f"{dest} 로 넣어도 됩니다 — MD5 만 맞으면 됩니다.")
    return 1


if __name__ == "__main__":
    from src.cli import guard

    sys.exit(guard(main))
