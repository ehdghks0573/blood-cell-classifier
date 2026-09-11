"""출력 스트림이 글자 하나 때문에 죽지 않게 만든다.

파이썬은 **콘솔**에 쓸 때는 UTF-8 을 쓰지만, 출력을 파일이나 파이프로 넘기면
시스템 인코딩(한국어 윈도우에서는 cp949)으로 바뀐다. cp949 에는 em-dash(—)가
없어서, 이 프로젝트의 한국어 출력은 리다이렉트하는 순간 죽었다.

    UnicodeEncodeError: 'cp949' codec can't encode character '\u2014'

손으로 돌릴 때는 멀쩡하고 **로그로 남기려 할 때만** 죽는다. 제일 필요할 때
못 쓰는 셈이라, 실제로 고해상도 데이터를 이어받다가 여기서 멈췄다.

인코딩을 UTF-8 로 바꾸는 쪽은 택하지 않았다. PowerShell 5.1 은 네이티브 명령의
출력을 `[Console]::OutputEncoding`(cp949)으로 해석하므로, UTF-8 로 내보내면
`python train.py > log.txt` 에서 이번엔 **한글 전체**가 깨진다. 이 프로젝트의
안내문은 전부 PowerShell 기준이다. 쓸 수 있는 글자를 살리고 못 쓰는 글자
하나를 '?' 로 버리는 편이 손해가 작다.

`PYTHONIOENCODING` 을 직접 준 사람의 선택은 건드리지 않는다.
"""

from __future__ import annotations

import os
import sys


def make_output_encodable() -> None:
    """stdout·stderr 를 "못 쓰는 글자는 '?' 로" 모드로 바꾼다.

    인코딩은 그대로 둔다. 바꾸는 것은 인코딩 실패를 어떻게 처리하느냐 뿐이다.
    여러 번 불러도 안전하다.
    """
    if os.environ.get("PYTHONIOENCODING"):
        return

    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue  # 테스트의 StringIO 처럼 바꿀 수 없는 스트림
        try:
            reconfigure(errors="replace")
        except (ValueError, OSError):
            pass  # 이미 닫혔거나 바꿀 수 없는 스트림. 여기서 죽을 이유는 없다
