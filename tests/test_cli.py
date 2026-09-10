"""실패했을 때 사람이 읽을 수 있는 말이 나오는가 (Day 12).

성능 테스트만 있고 실패 경로 테스트가 없으면, 잘 돌 때만 확인한 셈이다.
정작 처음 쓰는 사람이 만나는 것은 실패 쪽이다.

여기서 못박는 것은 두 가지다 — **트레이스백이 새어 나가지 않을 것**,
그리고 **무엇을 해야 하는지 메시지에 있을 것**.
"""

import argparse

import pytest
import torch

from src import cli


# ── UserError 는 메시지만, 버그는 그대로 터진다 ──

def test_user_error_prints_message_without_traceback(capsys):
    def main(argv):
        raise cli.UserError("데이터가 없습니다. tools/fetch_data.py 를 실행하세요.")

    assert cli.guard(main) == 1
    out = capsys.readouterr().out
    assert "tools/fetch_data.py" in out
    assert "Traceback" not in out


def test_real_bugs_are_not_swallowed():
    """코드의 버그까지 삼키면 고칠 수가 없다."""
    def main(argv):
        return 1 / 0

    with pytest.raises(ZeroDivisionError):
        cli.guard(main)


def test_unrelated_runtime_error_is_not_swallowed():
    def main(argv):
        raise RuntimeError("shape 이 안 맞습니다")

    with pytest.raises(RuntimeError):
        cli.guard(main)


def test_success_code_passes_through():
    assert cli.guard(lambda argv: 0) == 0


def test_interrupt_is_not_an_error_report(capsys):
    def main(argv):
        raise KeyboardInterrupt

    assert cli.guard(main) == 130  # 셸 관례
    assert "Traceback" not in capsys.readouterr().out


# ── VRAM 부족 ──

def test_out_of_memory_suggests_what_to_change(capsys):
    def main(argv):
        raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")

    assert cli.guard(main) == 1
    out = capsys.readouterr().out
    assert "--batch-size" in out
    assert "--cpu" in out
    assert "Traceback" not in out


def test_typed_oom_is_caught_too(capsys):
    """torch 최신 버전은 전용 예외를 던진다."""
    def main(argv):
        raise torch.cuda.OutOfMemoryError("out of memory")

    assert cli.guard(main) == 1
    assert "--batch-size" in capsys.readouterr().out


# ── 잘못된 인자 ──

@pytest.mark.parametrize("bad", ["0", "-3"])
def test_positive_int_rejects_zero_and_negative(bad):
    with pytest.raises(argparse.ArgumentTypeError):
        cli.positive_int(bad)


def test_positive_int_rejects_words():
    with pytest.raises(argparse.ArgumentTypeError):
        cli.positive_int("스물")


def test_positive_int_accepts_one():
    assert cli.positive_int("1") == 1


# ── 체크포인트 ──

def test_missing_run_lists_what_exists(tmp_path, capsys):
    (tmp_path / "runs" / "있는실행").mkdir(parents=True)
    (tmp_path / "runs" / "있는실행" / "best.pt").write_bytes(b"x")

    with pytest.raises(cli.UserError) as e:
        cli.load_run("없는실행", tmp_path, torch.device("cpu"))

    assert "있는실행" in str(e.value)  # 뭘 쓸 수 있는지 알려 준다


def test_missing_run_with_no_runs_at_all_tells_you_to_train(tmp_path):
    (tmp_path / "runs").mkdir()
    with pytest.raises(cli.UserError) as e:
        cli.load_run("아무거나", tmp_path, torch.device("cpu"))
    assert "train.py" in str(e.value)


def test_corrupt_checkpoint_says_so(tmp_path):
    """학습이 도중에 끊기면 반쯤 쓰인 파일이 남는다."""
    d = tmp_path / "runs" / "깨진실행"
    d.mkdir(parents=True)
    (d / "best.pt").write_text("이건 체크포인트가 아니다", encoding="utf-8")

    with pytest.raises(cli.UserError) as e:
        cli.load_run("깨진실행", tmp_path, torch.device("cpu"))
    assert "다시 학습" in str(e.value)


def test_checkpoint_missing_required_key(tmp_path):
    d = tmp_path / "runs" / "이상한실행"
    d.mkdir(parents=True)
    torch.save({"model": {}}, d / "best.pt")  # arch, input_size 가 없다

    with pytest.raises(cli.UserError) as e:
        cli.load_run("이상한실행", tmp_path, torch.device("cpu"))
    assert "arch" in str(e.value)


# ── 장치 ──

def test_cpu_flag_wins_even_with_a_gpu():
    assert cli.pick_device(force_cpu=True).type == "cpu"


def test_device_choice_is_reported_not_hidden():
    text = cli.describe_device(torch.device("cpu"))
    assert "CPU" in text
    assert "몇 시간" in text  # 얼마나 느린지까지 말해 준다
