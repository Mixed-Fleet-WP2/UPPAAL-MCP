"""Tests for server/verifyta.py against real verifyta 5.0.0 output captured in fixtures/."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from server import verifyta as vt
from server.verifyta import Verdict

FIX = Path(__file__).parent / "fixtures"
ROOT = Path(__file__).parent.parent
VERIFYTA = os.environ.get("VERIFYTA", vt.DEFAULT_VERIFYTA)
live = pytest.mark.skipif(not Path(VERIFYTA).is_file(), reason="verifyta not installed")


def fixture(name: str) -> tuple[str, str, int | None]:
    rc_file = FIX / f"{name}.rc"
    rc = int(rc_file.read_text()) if rc_file.exists() else None
    return (FIX / f"{name}.stdout").read_text(), (FIX / f"{name}.stderr").read_text(), rc


def results_for(name: str, qfile: str, timed_out: bool = False) -> list[vt.QueryResult]:
    out, err, rc = fixture(name)
    queries = vt.parse_query_file((FIX / qfile).read_text())
    return vt.build_results(queries, out, vt.parse_diagnostics(err), timed_out, rc)


# --------------------------------------------------------------------------- #
# Query file parsing
# --------------------------------------------------------------------------- #


def test_parse_query_file_keeps_lines_and_comments() -> None:
    qs = vt.parse_query_file((FIX / "train_gate.q").read_text())
    assert [(q.index, q.line, q.formula) for q in qs] == [
        (1, 2, "A[] not deadlock"),
        (2, 5, "A[] (train.Cross imply gate.Down)"),
    ]
    assert qs[0].comment == "R1: the system never deadlocks."


def test_parse_query_file_strips_line_and_multiline_comments() -> None:
    text = "/* multi\n line */\nE<> p.A // trailing\n// only comment\n\nA[] q /* inline */ \n"
    qs = vt.parse_query_file(text)
    assert [(q.line, q.formula) for q in qs] == [(3, "E<> p.A"), (6, "A[] q")]
    assert qs[0].comment == "multi line"


def test_smc_queries_detected_but_symbolic_ones_pass() -> None:
    qs = vt.parse_query_file(
        "Pr[<=100](<> a)\nsimulate[<=10]{x}\nE[<=10;100](max: x)\nE[] a\nE<> a\nA<> b\na --> b\nA[] not deadlock\n"
    )
    assert [q.line for q in vt.find_smc_queries(qs)] == [1, 2, 3]


# --------------------------------------------------------------------------- #
# Verdict parsing from captured fixtures
# --------------------------------------------------------------------------- #


def test_satisfied_and_violated() -> None:
    res = results_for("train_gate", "train_gate.q")
    assert [r.verdict for r in res] == [Verdict.SATISFIED, Verdict.VIOLATED]
    assert res[0].stats["states_explored"] == 24
    assert res[1].trace_file and res[1].trace_file.endswith("tg1.xml")
    assert res[0].trace_file is None


def test_default_output_with_ansi_progress_codes() -> None:
    res = results_for("train_gate_default", "train_gate.q")
    assert [r.verdict for r in res] == [Verdict.SATISFIED, Verdict.VIOLATED]


def test_maybe_results() -> None:
    assert [r.verdict for r in results_for("maybe_A", "train_gate.q")] == [Verdict.SATISFIED, Verdict.MAYBE]
    assert [r.verdict for r in results_for("maybe_Z", "sat.q")] == [Verdict.MAYBE, Verdict.SATISFIED]


def test_aborted_query_is_error_with_reason() -> None:
    res = results_for("overflow", "train_gate.q")
    assert res[0].verdict is Verdict.ERROR
    assert "out of range" in res[0].detail
    assert res[1].verdict is Verdict.VIOLATED


def test_timeout_is_never_satisfied() -> None:
    res = results_for("timeout", "stress.q", timed_out=True)
    assert [r.verdict for r in res] == [Verdict.TIMEOUT, Verdict.TIMEOUT]
    assert "started" in res[0].detail and "Not started" in res[1].detail


def test_truncated_output_without_timeout_is_error_not_satisfied() -> None:
    # Process died (e.g. memory limit) after starting query 1.
    res = results_for("timeout", "stress.q", timed_out=False)
    assert [r.verdict for r in res] == [Verdict.ERROR, Verdict.ERROR]


def test_satisfied_requires_exact_verdict_line() -> None:
    queries = vt.parse_query_file("A[] a\n")
    out = "Verifying formula 1 at x.q:1\n -- Formula is satisfied?\n"
    assert vt.build_results(queries, out, [], False, 0)[0].verdict is Verdict.ERROR


def test_model_syntax_errors_parsed() -> None:
    _, err, rc = fixture("syntax_error")
    diags = vt.parse_diagnostics(err)
    assert rc == 1 and len(diags) == 3
    assert diags[0].source == "model"
    assert diags[0].xpath == "/nta/template[1]/transition[2]/label[1]"
    assert diags[0].line == 1
    assert "unexpected T_ID" in diags[0].message
    res = results_for("syntax_error", "train_gate.q")
    assert all(r.verdict is Verdict.ERROR for r in res)


def test_query_errors_attributed_to_query_line() -> None:
    res = results_for("bad_query", "bad_query.q")
    assert res[0].verdict is Verdict.ERROR and "no member named Nowhere" in res[0].detail
    assert res[1].verdict is Verdict.ERROR and "syntax error" in res[1].detail


# --------------------------------------------------------------------------- #
# Command construction
# --------------------------------------------------------------------------- #


def test_build_command_flags(tmp_path: Path) -> None:
    cmd = vt.build_command("/v", Path("m.xml"), Path("q.q"), "shortest", "dfs", tmp_path / "trace-")
    assert cmd == ["/v", "-q", "-s", "-u", "-o", "1", "-t", "1", "-X", str(tmp_path / "trace-"), "m.xml", "q.q"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"trace_type": "bogus"},
        {"search_order": "bogus"},
        {"search_order": "optimal", "trace_type": "some"},
    ],
)
def test_build_command_rejects_bad_options(kwargs: dict[str, str]) -> None:
    with pytest.raises(vt.VerifytaOptionError):
        vt.build_command("/v", Path("m.xml"), Path("q.q"), trace_prefix=Path("/t/trace-"), **kwargs)


# --------------------------------------------------------------------------- #
# Live runs against the real binary
# --------------------------------------------------------------------------- #


@live
def test_live_train_gate_with_trace(tmp_path: Path) -> None:
    run = vt.verify(
        FIX / "train_gate.xml", FIX / "train_gate.q", trace_type="some", trace_dir=tmp_path
    )
    assert run.status == "ok"
    assert [r.verdict for r in run.queries] == [Verdict.SATISFIED, Verdict.VIOLATED]
    assert run.queries[1].trace_file and Path(run.queries[1].trace_file).is_file()


@live
def test_live_inline_query() -> None:
    run = vt.verify(FIX / "train_gate.xml", query_text="E<> train.Cross")
    assert run.queries[0].verdict is Verdict.SATISFIED


@live
def test_live_timeout() -> None:
    run = vt.verify(FIX / "stress.xml", FIX / "stress.q", timeout_s=1)
    assert run.status == "timeout" and run.timed_out
    assert all(r.verdict is Verdict.TIMEOUT for r in run.queries)
    assert run.duration_s < 10


@live
def test_live_compile_error() -> None:
    run = vt.verify(FIX / "syntax_error.xml", FIX / "train_gate.q")
    assert run.status == "compile_error"
    assert all(r.verdict is Verdict.ERROR for r in run.queries)


@live
def test_live_memory_limit_is_not_satisfied() -> None:
    run = vt.verify(FIX / "stress.xml", FIX / "stress.q", timeout_s=30, memory_mb=64)
    assert all(r.verdict in (Verdict.ERROR, Verdict.TIMEOUT) for r in run.queries)


@live
def test_live_rejects_smc() -> None:
    with pytest.raises(vt.VerifytaOptionError, match="not supported"):
        vt.verify(FIX / "train_gate.xml", query_text="Pr[<=10](<> train.Cross)")
