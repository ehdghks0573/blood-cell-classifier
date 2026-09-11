"""출력이 글자 하나 때문에 죽지 않는가.

실제로 있었던 일이다. 고해상도 데이터를 이어받는 도중 `fetch_hires.py` 가
`UnicodeEncodeError` 로 멈췄다 — 내려받기 코드는 멀쩡했고, 상황을 설명하는
**문장에 em-dash 가 들어 있었을 뿐**이다.

콘솔로 직접 쓸 때는 재현되지 않는다. 출력을 파일이나 파이프로 넘길 때만
인코딩이 cp949 로 바뀌기 때문에, **로그로 남기려 할 때만** 죽었다.
그래서 테스트도 콘솔이 아니라 cp949 파일에 쓰는 상황으로 만든다.
"""

import io
import sys

import pytest

from src import cli, console


def cp949_stream(tmp_path):
    """리다이렉트된 출력을 흉내 낸다. 한글은 되고 em-dash 는 안 되는 인코딩."""
    return (tmp_path / "log.txt").open("w", encoding="cp949")


@pytest.fixture(autouse=True)
def _no_ambient_pythonioencoding(monkeypatch):
    """돌리는 사람의 환경변수가 결과를 바꾸지 않게 한다.

    `make_output_encodable` 은 PYTHONIOENCODING 이 있으면 손을 떼도록 **일부러**
    만들어져 있다(아래 마지막 테스트). 그래서 그 변수를 켜 둔 셸에서 이 파일을
    돌리면 아래 테스트 셋이 무더기로 깨지고, 코드가 고장 난 것처럼 보인다.
    실제로 그렇게 한 번 속았다. 검사할 상황을 테스트가 직접 만든다.
    """
    monkeypatch.delenv("PYTHONIOENCODING", raising=False)


# ── 무엇이 문제였는가 ──

def test_cp949_cannot_hold_an_em_dash(tmp_path):
    """고치기 전에 죽던 지점. 이 전제가 깨지면 나머지 테스트는 의미가 없다."""
    with cp949_stream(tmp_path) as f:
        with pytest.raises(UnicodeEncodeError):
            f.write("—")


# ── 고친 뒤 ──

def test_output_survives_a_character_the_encoding_lacks(tmp_path, monkeypatch):
    with cp949_stream(tmp_path) as f:
        monkeypatch.setattr(sys, "stdout", f)
        console.make_output_encodable()
        print("고해상도 — 이어받기")

    text = (tmp_path / "log.txt").read_text(encoding="cp949")
    assert "고해상도" in text          # 쓸 수 있는 글자는 그대로 남는다
    assert "이어받기" in text
    assert "—" not in text             # 못 쓰는 글자만 버린다


def test_guard_protects_every_tool(tmp_path, monkeypatch):
    """도구마다 따로 챙기지 않아도 되게, guard 가 먼저 걸어 준다."""
    def main(argv):
        print("모델 다섯 개가 같은 26장을 틀린다 — 원인은 데이터다")
        return 0

    with cp949_stream(tmp_path) as f:
        monkeypatch.setattr(sys, "stdout", f)
        assert cli.guard(main) == 0

    assert "26장" in (tmp_path / "log.txt").read_text(encoding="cp949")


# ── 건드리지 말아야 할 것 ──

def test_explicit_pythonioencoding_is_respected(tmp_path, monkeypatch):
    """사람이 직접 인코딩을 정했으면 그 선택이 이긴다."""
    monkeypatch.setenv("PYTHONIOENCODING", "utf-8")
    with cp949_stream(tmp_path) as f:
        monkeypatch.setattr(sys, "stdout", f)
        console.make_output_encodable()
        assert f.errors == "strict"


def test_streams_without_reconfigure_are_left_alone(monkeypatch):
    """StringIO 로 출력을 가로채는 테스트·도구가 있다. 거기서 죽으면 안 된다."""
    buf = io.StringIO()
    monkeypatch.setattr(sys, "stdout", buf)
    monkeypatch.delattr(io.StringIO, "reconfigure", raising=False)
    console.make_output_encodable()  # 예외가 나지 않으면 통과
    print("가로채도 된다")
    assert "가로채도" in buf.getvalue()


def test_calling_twice_is_safe(tmp_path, monkeypatch):
    with cp949_stream(tmp_path) as f:
        monkeypatch.setattr(sys, "stdout", f)
        console.make_output_encodable()
        console.make_output_encodable()
        assert f.errors == "replace"


def test_pythonioencoding_was_the_users_choice(tmp_path, monkeypatch):
    """직접 준 사람의 선택은 건드리지 않는다 — src/console.py 에 적힌 약속이다.

    UTF-8 로 내보내라고 스스로 지정한 사람에게 "못 쓰는 글자는 버린다"를
    덮어씌우면, 버릴 이유가 없는 글자까지 버리게 된다.
    """
    monkeypatch.setenv("PYTHONIOENCODING", "utf-8")

    with cp949_stream(tmp_path) as f:
        monkeypatch.setattr(sys, "stdout", f)
        console.make_output_encodable()
        assert f.errors == "strict"   # 손대지 않았다
