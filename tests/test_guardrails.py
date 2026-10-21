"""Phase 5 guardrails: approved queries are read-only, resources and prompts."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import anyio
import pytest
from mcp import Client
from mcp.shared.exceptions import MCPError

from server.main import mcp
from server.workspace import Workspace, WorkspaceError
from tests.test_server import call, ws  # noqa: F401  (fixture)
from tests.test_verifyta import live

pytestmark = live

REQ = "Whenever the train is in the crossing, the gate is down"


def digest(root: Path) -> dict[str, str]:
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((root / "queries").glob("*.q"))}


@pytest.mark.parametrize("name", ["../train_gate.q", "sub/x.q", "x.txt", "/abs.q", "..q", ".hidden.q", ""])
def test_writable_query_path_rejects(ws: Path, name: str) -> None:
    with pytest.raises(WorkspaceError):
        Workspace(ws).writable_query_path(name)


def test_writable_query_path_rejects_symlink_to_approved(ws: Path) -> None:
    (ws / "queries/proposed/evil.q").symlink_to(ws / "queries/train_gate.q")
    with pytest.raises(WorkspaceError, match="outside"):
        Workspace(ws).writable_query_path("evil.q")


def test_propose_query_writes_only_to_proposed(ws: Path) -> None:
    before = digest(ws)
    res = call("propose_query", {"model": "train_gate", "requirement": REQ, "uppaal_queries": ["A[] (train.Cross imply gate.Down)", "E<> train.Cross"]})
    assert res["status"] == "ok" and res["query_file"] == "queries/proposed/train_gate.q"
    assert res["already_approved"] == ["A[] (train.Cross imply gate.Down)"]
    text = (ws / res["query_file"]).read_text()
    assert f"/* REQ: {REQ} (part 1/2) */\nA[] (train.Cross imply gate.Down)\n" in text
    lines = text.splitlines()
    assert [lines[e["line"] - 1] for e in res["entries"]] == ["A[] (train.Cross imply gate.Down)", "E<> train.Cross"]
    assert digest(ws) == before

    # Appending keeps earlier entries and line numbers stay right.
    res2 = call("propose_query", {"model": "train_gate", "requirement": "gate can open", "uppaal_queries": ["E<> gate.Up"]})
    lines = (ws / res2["query_file"]).read_text().splitlines()
    assert lines[res2["entries"][0]["line"] - 1] == "E<> gate.Up"

    run = call("verify", {"model": "train_gate", "query_file": res["query_file"]})
    assert [q["verdict"] for q in run["queries"]] == ["violated", "satisfied", "satisfied"]
    assert run["queries"][0]["requirement"] == f"{REQ} (part 1/2)"
    assert run["exploratory"] is False


def test_propose_query_type_errors_write_nothing(ws: Path) -> None:
    res = call("propose_query", {"model": "train_gate", "requirement": "x", "uppaal_queries": ["E<> train.Cross", "A[] train.Nowhere"]})
    assert res["status"] == "errors" and res["errors"][0]["formula_index"] == 1
    assert not (ws / "queries/proposed/train_gate.q").exists()


@pytest.mark.parametrize(
    "args, msg",
    [
        ({"file": "../train_gate.q"}, "Invalid proposed query file name"),
        ({"file": "train_gate.txt"}, "Invalid proposed query file name"),
        ({"uppaal_queries": ["Pr[<=10](<> train.Cross)"]}, "not supported"),
        ({"uppaal_queries": ["E<> train.Cross /* hi */"]}, "comment"),
        ({"uppaal_queries": []}, "non-empty"),
        ({"requirement": " "}, "requirement"),
    ],
)
def test_propose_query_refusals(ws: Path, args: dict[str, Any], msg: str) -> None:
    base = {"model": "train_gate", "requirement": REQ, "uppaal_queries": ["E<> train.Cross"]}
    res = call("propose_query", {**base, **args})
    assert res["status"] == "error" and msg in res["message"]


def test_no_tool_modifies_approved_queries(ws: Path) -> None:
    before = digest(ws)
    attempts: list[tuple[str, dict[str, Any]]] = [
        ("propose_query", {"model": "train_gate", "requirement": "r", "uppaal_queries": ["E<> train.Cross"], "file": "../train_gate.q"}),
        ("propose_query", {"model": "train_gate", "requirement": "r", "uppaal_queries": ["E<> train.Cross"], "file": "/home/x.q"}),
        ("patch_model", {"model": "../queries/train_gate.q", "operations": [{"op": "set_declaration", "text": ""}], "reason": "x"}),
        ("patch_model", {"model": "queries/train_gate", "operations": [{"op": "set_declaration", "text": ""}], "reason": "x"}),
        ("verify", {"model": "train_gate", "query_file": "train_gate.q"}),
        ("verify", {"model": "train_gate"}),
        ("check_syntax", {"model": "train_gate", "query_file": "train_gate.q"}),
    ]
    for tool, args in attempts:
        call(tool, args)
    assert digest(ws) == before


def test_verify_query_text_is_exploratory(ws: Path) -> None:
    res = call("verify", {"model": "train_gate", "query_text": "E<> train.Cross"})
    assert res["exploratory"] is True and "do not verify an approved requirement" in res["note"]
    res = call("verify", {"model": "train_gate"})
    assert res["exploratory"] is False and res["queries"][0]["requirement"] == "R1: the system never deadlocks."


def test_resources(ws: Path) -> None:
    async def go() -> dict[str, Any]:
        async with Client(mcp) as c:
            static = {str(r.uri) for r in (await c.list_resources()).resources}
            templates = {t.uri_template for t in (await c.list_resource_templates()).resource_templates}
            index = (await c.read_resource("uppaal://queries")).contents[0].text
            qfile = (await c.read_resource("uppaal://queries/train_gate")).contents[0].text
            sheet = (await c.read_resource("uppaal://docs/query-language")).contents[0].text
            return {"static": static, "templates": templates, "index": index, "qfile": qfile, "sheet": sheet}

    r = anyio.run(go)
    assert {"uppaal://queries", "uppaal://docs/query-language"} <= r["static"]
    assert "uppaal://queries/{name}" in r["templates"]
    assert '"formula": "A[] (train.Cross imply gate.Down)"' in r["index"]
    assert r["qfile"] == (ws / "queries/train_gate.q").read_text()
    for op in ("A[]", "E<>", "A<>", "E[]", "-->", "deadlock"):
        assert op in r["sheet"]


@pytest.mark.parametrize("uri", ["uppaal://queries/nonexistent", "uppaal://queries/proposed%2Ftrain_gate"])
def test_resource_errors(ws: Path, uri: str) -> None:
    async def go() -> str:
        async with Client(mcp) as c:
            try:
                await c.read_resource(uri)
            except MCPError as e:
                return str(e)
        return "no error"

    assert "No approved query file" in anyio.run(go)


def test_prompt(ws: Path) -> None:
    async def go() -> tuple[set[str], str]:
        async with Client(mcp) as c:
            names = {p.name for p in (await c.list_prompts()).prompts}
            res = await c.get_prompt("verify_requirement", {"requirement": REQ, "model": "train_gate"})
            return names, res.messages[0].content.text

    names, text = anyio.run(go)
    assert "verify_requirement" in names
    assert REQ in text and "propose_query" in text and "FULL approved suite" in text and "UNKNOWN" in text
