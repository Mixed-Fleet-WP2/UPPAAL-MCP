"""End-to-end tests of the MCP tools through an in-process MCP client."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import anyio
import pytest
from mcp import Client

from server.main import mcp
from tests.test_verifyta import FIX, ROOT, live

pytestmark = live


@pytest.fixture
def ws(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """An isolated, git-initialised copy of the workspace."""
    root = tmp_path / "ws"
    (root / "queries/proposed").mkdir(parents=True)
    (root / "traces").mkdir()
    shutil.copytree(ROOT / "models", root / "models")
    shutil.copy(ROOT / "queries/train_gate.q", root / "queries/train_gate.q")
    shutil.copy(FIX / "syntax_error.xml", root / "models/broken.xml")
    shutil.copy(FIX / "stress.xml", root / "models/stress.xml")
    shutil.copy(FIX / "stress.q", root / "queries/stress.q")
    git = ["git", "-C", str(root), "-c", "user.name=t", "-c", "user.email=t@t"]
    subprocess.run([*git, "init", "-q"], check=True)
    subprocess.run([*git, "add", "-A"], check=True)
    subprocess.run([*git, "commit", "-qm", "init"], check=True)
    monkeypatch.setenv("UPPAAL_WORKSPACE", str(root))
    return root


def call(tool: str, args: dict[str, Any] | None = None) -> dict[str, Any]:
    async def go() -> dict[str, Any]:
        async with Client(mcp) as c:
            r = await c.call_tool(tool, args or {})
            assert not r.is_error, r.content
            return r.structured_content  # type: ignore[return-value]

    return anyio.run(go)


def test_tool_list() -> None:
    async def go() -> list[str]:
        async with Client(mcp) as c:
            return sorted(t.name for t in (await c.list_tools()).tools)

    assert {"check_syntax", "list_models", "verify"} <= set(anyio.run(go))


def test_list_models(ws: Path) -> None:
    res = call("list_models")
    by_name = {m["model"]: m for m in res["models"]}
    tg = by_name["train_gate"]
    assert tg["templates"] == ["Train", "Gate"]
    assert tg["approved_queries"] == "queries/train_gate.q" and tg["approved_query_count"] == 2
    assert tg["git"]["commit"] and tg["git"]["dirty"] is False


def test_check_syntax_ok(ws: Path) -> None:
    res = call("check_syntax", {"model": "train_gate", "query_file": "train_gate.q"})
    assert res["status"] == "ok" and res["model_ok"] and res["queries_ok"]


def test_check_syntax_model_errors_have_file_lines(ws: Path) -> None:
    res = call("check_syntax", {"model": "broken"})
    assert res["status"] == "errors" and not res["model_ok"]
    first = res["errors"][0]
    lines = (ws / "models/broken.xml").read_text().splitlines()
    assert "undefined_var" in lines[first["file_line"] - 1]
    assert first["element"] == "guard of edge Train: Near -> Cross"


def test_check_syntax_query_errors(ws: Path) -> None:
    res = call("check_syntax", {"model": "train_gate", "query_text": "A[] not deadlock\nA[] train.Nowhere"})
    assert res["status"] == "errors" and res["model_ok"] and res["queries_ok"] is False
    assert res["errors"][0]["query_file_line"] == 2


def test_verify_approved_suite_and_log(ws: Path) -> None:
    res = call("verify", {"model": "train_gate"})
    assert res["status"] == "ok"
    assert [q["verdict"] for q in res["queries"]] == ["satisfied", "violated"]
    assert res["queries"][1]["has_trace"]
    run_dir = ws / res["run_dir"]
    meta = json.loads((run_dir / "run.json").read_text())
    assert (ws / meta["queries"][1]["trace_file"]).is_file()
    log = [json.loads(line) for line in (ws / "runs.jsonl").read_text().splitlines()]
    assert [r["verdict"] for r in log] == ["satisfied", "violated"]
    assert all(r["run_id"] == res["run_id"] and r["model_git_commit"] for r in log)
    assert log[1]["query"] == "A[] (train.Cross imply gate.Down)"


def test_verify_timeout_is_unknown(ws: Path) -> None:
    res = call("verify", {"model": "stress", "timeout_s": 1, "trace_type": "none"})
    assert res["status"] == "timeout"
    assert {q["verdict"] for q in res["queries"]} == {"timeout"}


def test_verify_bad_option_returns_error(ws: Path) -> None:
    res = call("verify", {"model": "train_gate", "search_order": "optimal", "trace_type": "some"})
    assert res["status"] == "error" and "requires trace_type" in res["message"]


@pytest.mark.parametrize(
    "args",
    [
        {"model": "../../etc/passwd"},
        {"model": "/etc/passwd"},
        {"model": "train_gate", "query_file": "../models/x.q"},
        {"model": "train_gate", "query_file": "/etc/x.q"},
        {"model": "nonexistent"},
    ],
)
def test_paths_confined(ws: Path, args: dict[str, Any]) -> None:
    res = call("verify", args)
    assert res["status"] == "error"


def test_symlink_escape_refused(ws: Path, tmp_path: Path) -> None:
    outside = tmp_path / "outside.xml"
    shutil.copy(ws / "models/train_gate.xml", outside)
    (ws / "models/link.xml").symlink_to(outside)
    res = call("check_syntax", {"model": "link"})
    assert res["status"] == "error" and "outside" in res["message"]


def test_get_trace_counterexample(ws: Path) -> None:
    run = call("verify", {"model": "train_gate"})
    res = call("get_trace", {"run_id": run["run_id"]})  # only query 2 has a trace
    assert res["status"] == "ok" and res["kind"] == "counterexample"
    assert res["query"] == {"index": 2, "formula": "A[] (train.Cross imply gate.Down)", "verdict": "violated"}
    assert res["length"] == 3 and len(res["steps"]) == 3
    assert res["steps"][-1]["locations"] == {"train": "Cross", "gate": "Lowering"}
    assert (ws / res["trace_json"]).is_file() and (ws / res["raw_trace"]).is_file()


def test_get_trace_summary_and_paging(ws: Path) -> None:
    shutil.copy(FIX / "locals.xml", ws / "models/locals.xml")
    run = call("verify", {"model": "locals", "query_text": "A<> p0.B and p1.B"})
    res = call("get_trace", {"run_id": run["run_id"], "query_index": 1, "head": 2, "tail": 2})
    assert res["steps"] is None and res["length"] > 4
    assert [s["step"] for s in res["summary"]["first"]] == [0, 1]
    assert res["summary"]["omitted_steps"] == res["length"] - 4
    assert res["loop_back_to_step"] is not None
    page = call("get_trace", {"run_id": run["run_id"], "query_index": 1, "from_step": 3, "to_step": 5})
    assert [s["step"] for s in page["steps"]] == [3, 4, 5]


def test_get_trace_errors(ws: Path) -> None:
    run = call("verify", {"model": "train_gate"})
    assert "no trace" in call("get_trace", {"run_id": run["run_id"], "query_index": 1})["message"]
    assert call("get_trace", {"run_id": "nope"})["status"] == "error"
    assert call("get_trace", {"run_id": "../.."})["status"] == "error"
    assert call("get_trace", {"run_id": run["run_id"], "query_index": 2, "from_step": 5})["status"] == "error"
