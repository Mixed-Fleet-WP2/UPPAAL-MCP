"""Tests for server/trace.py against XML traces captured from verifyta 5.0.0."""

from __future__ import annotations

from pathlib import Path

import pytest

from server import trace as tr

FIX = Path(__file__).parent / "fixtures"


def test_train_gate_counterexample() -> None:
    t = tr.parse_trace(FIX / "train_gate_trace_q2.xtr.xml")
    assert t["processes"] == ["train", "gate"]
    assert t["clocks"] == ["train.x", "gate.y"]
    steps = t["steps"]
    assert [s["locations"] for s in steps] == [
        {"train": "Far", "gate": "Up"},
        {"train": "Near", "gate": "Lowering"},
        {"train": "Cross", "gate": "Lowering"},
    ]
    assert steps[0]["via"] is None
    s1 = steps[1]["via"]
    assert s1["action"] == "appr"
    assert [(e["process"], e["from"], e["to"], e["sync"]) for e in s1["edges"]] == [
        ("train", "Far", "Near", "appr!"),
        ("gate", "Up", "Lowering", "appr?"),
    ]
    assert steps[2]["via"]["edges"][0]["guard"] == "x >= 10"
    # Same constraints verifyta prints with -y, minus the implied gate.y - train.x <= 15.
    assert steps[2]["clock_constraints"] == ["train.x <= 5", "10 <= gate.y <= 15", "train.x - gate.y <= -10"]
    assert steps[0]["clock_constraints"] == ["train.x == gate.y"]
    assert steps[2]["variables"] == {"crossings": 0}
    assert tr.find_loop(steps) is None


def test_locals_arrays_and_changes() -> None:
    steps = tr.parse_trace(FIX / "locals_trace_safety.xtr.xml")["steps"]
    last = steps[-1]
    assert last["locations"] == {"p0": "B", "p1": "B"}
    assert last["variables"] == {"g[0]": 5, "g[1]": 5, "flag": 1, "p0.n": 1, "p1.n": 1}
    assert last["changed"] == {"g[1]": [0, 5], "p1.n": [0, 1]}
    assert last["via"]["action"] == "internal (tau)"


def test_liveness_counterexample_loop() -> None:
    steps = tr.parse_trace(FIX / "locals_trace_liveness.xtr.xml")["steps"]
    loop = tr.find_loop(steps)
    assert loop is not None and loop < len(steps) - 1
    assert steps[loop]["locations"] == steps[-1]["locations"]


def test_compact_step_and_time() -> None:
    steps = tr.parse_trace(FIX / "train_gate_trace_q2.xtr.xml")["steps"]
    c = tr.compact_step(steps[2])
    assert c["time"] == "[10, inf)"
    assert "clock_bounds" not in c and c["via"].startswith("train: Near -> Cross")


def test_reading_guide() -> None:
    assert "LAST step" in tr.reading_guide("A[] p", "violated")
    assert "liveness" in tr.reading_guide("A<> p", "violated")
    assert "liveness" in tr.reading_guide("p --> q", "violated")
    assert "Witness" in tr.reading_guide("E<> p", "satisfied")


@pytest.mark.parametrize("content", ["not xml", "<nta/>", "<trace/>"])
def test_bad_trace_files(tmp_path: Path, content: str) -> None:
    p = tmp_path / "t.xml"
    p.write_text(content)
    with pytest.raises(tr.TraceError):
        tr.parse_trace(p)


def test_missing_trace_file(tmp_path: Path) -> None:
    with pytest.raises(tr.TraceError):
        tr.parse_trace(tmp_path / "nope.xml")
