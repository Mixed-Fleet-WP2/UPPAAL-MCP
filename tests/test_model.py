"""Tests for server/model.py: faithful round trips, reading and edit operations."""

from __future__ import annotations

import glob
from pathlib import Path

import pytest

from server import model as mdl
from server import verifyta as vt
from tests.test_verifyta import FIX, VERIFYTA, live

# Frozen copy of the train-gate model with its deliberate bug (guard x >= 10).
TRAIN_GATE = FIX / "train_gate.xml"
DEMOS = sorted(glob.glob(str(Path(VERIFYTA).parent.parent / "demo/**/*.xml"), recursive=True))


def doc() -> mdl.Document:
    return mdl.Document.load(TRAIN_GATE)


def patch(ops: list[dict]) -> tuple[mdl.Document, list[mdl.Change]]:
    d = doc()
    return d, mdl.apply_operations(d, ops)


def changed_lines(d: mdl.Document) -> list[str]:
    import difflib

    return [
        line
        for line in difflib.unified_diff(d.raw.splitlines(), d.serialize().splitlines(), lineterm="", n=0)
        if line[:1] in "+-" and not line.startswith(("+++", "---"))
    ]


@pytest.mark.parametrize("path", [str(TRAIN_GATE), str(FIX / "locals.xml"), *DEMOS])
def test_round_trip_is_byte_identical(path: str) -> None:
    d = mdl.Document.load(Path(path))
    assert d.serialize() == d.raw


def test_rejects_non_uppaal_xml() -> None:
    with pytest.raises(mdl.ModelError):
        mdl.Document.from_text("<foo/>")
    with pytest.raises(mdl.ModelError):
        mdl.Document.from_text("<nta>")


def test_read_model() -> None:
    m = mdl.read_model(doc())
    assert "int[0,3] crossings" in m["global_declarations"]
    train, gate = m["templates"]
    assert train["name"] == "Train" and train["declarations"] == "clock x;"
    assert [(l["name"], l["invariant"], l["initial"]) for l in train["locations"]] == [
        ("Far", None, True),
        ("Near", "x <= 20", False),
        ("Cross", "x <= 5", False),
    ]
    e = train["edges"][1]
    assert (e["index"], e["source"], e["target"], e["guard"], e["sync"], e["update"]) == (1, "Near", "Cross", "x >= 10", None, "x = 0")
    assert train["edges"][0]["sync"] == "appr!"
    assert gate["locations"][1]["invariant"] == "y <= 15"
    assert "system train, gate;" in m["system"]
    assert TRAIN_GATE.read_text().splitlines()[e["file_line"] - 1].strip().startswith("<transition")


def test_set_guard_changes_one_line() -> None:
    d, ch = patch([{"op": "set_guard", "template": "Train", "from": "Near", "to": "Cross", "value": "x >= 15"}])
    assert changed_lines(d) == [
        '-\t\t\t<label kind="guard" x="210" y="50">x &gt;= 10</label>',
        '+\t\t\t<label kind="guard" x="210" y="50">x &gt;= 15</label>',
    ]
    assert ch[0].description == "guard of edge Train: Near -> Cross: 'x >= 10' -> 'x >= 15'"
    assert ch[0].flags[0]["severity"] == "may_change_behaviour"


def test_add_guard_label_in_canonical_position() -> None:
    d, _ = patch([{"op": "set_guard", "template": "Train", "edge": 0, "value": "crossings < 3"}])
    out = d.serialize()
    tr = out[out.index('<transition id="id3">') : out.index("</transition>", out.index('<transition id="id3">'))]
    assert tr.index('<target ref="id1"/>') < tr.index('kind="guard"') < tr.index('kind="synchronisation"')
    assert '\n\t\t\t<label kind="guard">crossings &lt; 3</label>\n' in tr


def test_guard_false_removes_behaviour() -> None:
    _, ch = patch([{"op": "set_guard", "template": "Train", "edge": 1, "value": " ( false ) "}])
    assert ch[0].flags[0]["severity"] == "removes_behaviour"


def test_remove_label_with_empty_value() -> None:
    d, _ = patch([{"op": "set_update", "template": "Train", "edge": 1, "value": ""}])
    assert changed_lines(d) == ['-\t\t\t<label kind="assignment" x="210" y="70">x = 0</label>']


def test_set_invariant_and_flags() -> None:
    d, ch = patch([{"op": "set_invariant", "template": "Gate", "location": "Lowering", "value": "y <= 10"}])
    assert len(changed_lines(d)) == 2
    assert "invariant of location Gate.Lowering: 'y <= 15' -> 'y <= 10'" == ch[0].description
    _, ch = patch([{"op": "set_invariant", "template": "Gate", "location": "Down", "value": "false"}])
    assert ch[0].flags[0]["severity"] == "removes_behaviour"
    d, _ = patch([{"op": "set_invariant", "template": "Gate", "location": "Down", "value": "y <= 50"}])
    assert changed_lines(d) == ['+\t\t\t<label kind="invariant">y &lt;= 50</label>']


def test_replace_in_declaration() -> None:
    d, ch = patch([{"op": "replace_in_declaration", "scope": "Train", "old": "clock x;", "new": "clock x; int n;"}])
    assert changed_lines(d) == ["-\t\t<declaration>clock x;</declaration>", "+\t\t<declaration>clock x; int n;</declaration>"]
    with pytest.raises(mdl.ModelError, match="exactly once"):
        patch([{"op": "replace_in_declaration", "scope": "global", "old": "nonexistent", "new": ""}])


def test_set_declaration_global_and_system() -> None:
    d, _ = patch(
        [
            {"op": "set_declaration", "scope": "global", "text": "chan appr, leave;\nint[0,3] crossings;"},
            {"op": "set_declaration", "scope": "system", "text": "system Train, Gate;"},
        ]
    )
    m = mdl.read_model(mdl.Document.from_text(d.serialize()))
    assert m["global_declarations"] == "chan appr, leave;\nint[0,3] crossings;"
    assert m["system"] == "system Train, Gate;"


def test_add_and_remove_edge() -> None:
    d, ch = patch([{"op": "add_edge", "template": "Gate", "from": "Lowering", "to": "Raising", "sync": "leave?", "update": "y = 0"}])
    out = d.serialize()
    assert '\t\t<transition id="id14">\n\t\t\t<source ref="id7"/>\n\t\t\t<target ref="id9"/>\n' in out
    assert '\t\t\t<label kind="synchronisation">leave?</label>\n\t\t\t<label kind="assignment">y = 0</label>\n\t\t</transition>\n\t</template>' in out
    assert ch[0].flags == []
    gate = mdl.read_model(mdl.Document.from_text(out))["templates"][1]
    assert gate["edges"][-1]["source"] == "Lowering" and gate["edges"][-1]["sync"] == "leave?"

    d, ch = patch([{"op": "remove_edge", "template": "Gate", "from": "Raising", "to": "Up"}])
    assert ch[0].flags[0]["severity"] == "removes_behaviour"
    assert len(mdl.read_model(mdl.Document.from_text(d.serialize()))["templates"][1]["edges"]) == 3
    assert all(line.startswith("-") for line in changed_lines(d))


def test_add_and_remove_location() -> None:
    d, _ = patch([{"op": "add_location", "template": "Train", "name": "Stop", "invariant": "x <= 3", "kind": "urgent"}])
    locs = mdl.read_model(mdl.Document.from_text(d.serialize()))["templates"][0]["locations"]
    assert (locs[-1]["name"], locs[-1]["invariant"], locs[-1]["urgent"]) == ("Stop", "x <= 3", True)
    assert all(line.startswith("+") for line in changed_lines(d))

    d, ch = patch([{"op": "remove_location", "template": "Train", "location": "Cross"}])
    assert [f["severity"] for f in ch[0].flags] == ["removes_behaviour"] * 3  # location + 2 incident edges
    train = mdl.read_model(mdl.Document.from_text(d.serialize()))["templates"][0]
    assert [l["name"] for l in train["locations"]] == ["Far", "Near"] and len(train["edges"]) == 1
    with pytest.raises(mdl.ModelError, match="initial"):
        patch([{"op": "remove_location", "template": "Train", "location": "Far"}])


@pytest.mark.parametrize(
    "op, msg",
    [
        ({"op": "set_guard", "template": "Nope", "edge": 0, "value": "x"}, "No template"),
        ({"op": "set_guard", "template": "Train", "edge": 9, "value": "x"}, "out of range"),
        ({"op": "set_guard", "template": "Train", "from": "Far", "to": "Cross", "value": "x"}, "No edge"),
        ({"op": "set_guard", "template": "Train", "value": "x"}, "Select the edge"),
        ({"op": "set_invariant", "template": "Train", "location": "Nowhere", "value": "x"}, "No location"),
        ({"op": "add_location", "template": "Train", "name": "Far"}, "already has"),
        ({"op": "frobnicate"}, "Unknown op"),
    ],
)
def test_operation_errors(op: dict, msg: str) -> None:
    with pytest.raises(mdl.ModelError, match=msg):
        patch([op])


def test_error_names_failing_operation() -> None:
    with pytest.raises(mdl.ModelError, match=r"Operation 1 \(set_guard\)"):
        patch([{"op": "set_guard", "template": "Train", "edge": 1, "value": "x >= 15"}, {"op": "set_guard", "template": "X", "edge": 0, "value": ""}])


@live
def test_patched_models_compile(tmp_path: Path) -> None:
    d, _ = patch(
        [
            {"op": "add_location", "template": "Gate", "name": "Stuck"},
            {"op": "add_edge", "template": "Gate", "from": "Down", "to": "Stuck", "guard": "y > 100"},
            {"op": "set_guard", "template": "Train", "edge": 0, "value": "crossings < 3"},
        ]
    )
    p = tmp_path / "m.xml"
    p.write_text(d.serialize())
    assert vt.compile_model(p).ok
