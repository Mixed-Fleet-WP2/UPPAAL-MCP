"""UPPAAL model-checking MCP server (stdio).

This server is the only component that runs verifyta or modifies model files.
Verdicts always come from verifyta; the server never infers them.

Configuration (environment variables):
  VERIFYTA          path to the verifyta binary
  UPPAAL_WORKSPACE  workspace root (default: the repository containing this file)
  UPPAAL_MEMORY_MB  address-space limit for verifyta (default 4096)
"""

from __future__ import annotations

import functools
import json
import os
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, Literal, TypeVar

import anyio
from mcp.server.mcpserver import MCPServer

from server import model as mdl
from server import verifyta as vt
from server.workspace import Workspace, WorkspaceError, utc_now

INSTRUCTIONS = """\
Model checking with UPPAAL. verifyta is the source of truth: report verdicts exactly
as returned (satisfied | violated | maybe | timeout | error). Never claim a property
holds unless its verdict is "satisfied". "timeout", "maybe" and "error" mean UNKNOWN.
Queries in queries/ are approved and read-only; propose new ones in queries/proposed/.
"""

mcp = MCPServer(name="uppaal", instructions=INSTRUCTIONS)

T = TypeVar("T")


def _ws() -> Workspace:
    return Workspace.from_env()


def _memory_mb() -> int:
    try:
        return int(os.environ.get("UPPAAL_MEMORY_MB", vt.DEFAULT_MEMORY_MB))
    except ValueError:
        return vt.DEFAULT_MEMORY_MB


def _error(message: str, **extra: Any) -> dict[str, Any]:
    return {"status": "error", "message": message, **extra}


def safe_tool(fn: Callable[..., Awaitable[dict[str, Any]]]) -> Callable[..., Awaitable[dict[str, Any]]]:
    """Convert expected failures into ``{"status": "error", "message": ...}`` so the
    agent always gets an actionable message instead of an exception."""

    @functools.wraps(fn)
    async def wrapper(*args: Any, **kwargs: Any) -> dict[str, Any]:
        try:
            return await fn(*args, **kwargs)
        except (WorkspaceError, vt.VerifytaOptionError) as e:
            return _error(str(e))
        except Exception as e:  # noqa: BLE001 - last resort, still reported to the agent
            return _error(f"Internal server error ({type(e).__name__}): {e}")

    return wrapper


# --------------------------------------------------------------------------- #
# Tools
# --------------------------------------------------------------------------- #


@mcp.tool()
@safe_tool
async def list_models() -> dict[str, Any]:
    """List the UPPAAL models in the workspace.

    Returns ``{"status": "ok", "models": [...]}``; each entry has:
      - ``model``: name to pass to other tools (e.g. ``"train_gate"``)
      - ``path``: workspace-relative path (``models/<name>.xml``)
      - ``templates``: template names, or ``parse_error`` if the XML is malformed
      - ``approved_queries``: path of the approved suite ``queries/<name>.q`` or null
      - ``approved_query_count``: number of queries in that suite
      - ``proposed_queries``: files in ``queries/proposed/`` whose name starts with the model name
      - ``git``: ``{commit, blob, dirty}`` for the model file
    """
    ws = _ws()
    models = []
    for p in sorted(ws.models_dir.rglob("*.xml")):
        entry: dict[str, Any] = {"model": p.relative_to(ws.models_dir).with_suffix("").as_posix(), "path": ws.rel(p)}
        try:
            entry["templates"] = [t.findtext("name") for t in mdl.load(p).getroot().iterfind("template")]
        except Exception as e:  # noqa: BLE001
            entry["parse_error"] = str(e)
        q = ws.approved_queries_for(p)
        entry["approved_queries"] = ws.rel(q) if q.is_file() else None
        entry["approved_query_count"] = len(vt.parse_query_file(q.read_text())) if q.is_file() else 0
        entry["proposed_queries"] = (
            [ws.rel(f) for f in sorted(ws.proposed_dir.glob(f"{p.stem}*.q"))] if ws.proposed_dir.is_dir() else []
        )
        entry["git"] = ws.git_info(p)
        models.append(entry)
    return {"status": "ok", "models": models}


@mcp.tool()
@safe_tool
async def check_syntax(model: str, query_file: str | None = None, query_text: str | None = None) -> dict[str, Any]:
    """Check a model (and optionally queries) for syntax and type errors WITHOUT verifying.

    Args:
      model: model name or path inside ``models/`` (``"train_gate"``, ``"train_gate.xml"``
        or ``"models/train_gate.xml"``).
      query_file: optional ``.q`` file inside ``queries/`` (e.g. ``"train_gate.q"`` or
        ``"queries/proposed/r3.q"``) to type-check against the model.
      query_text: optional query text (one query per line) to type-check instead.

    Returns:
      ``{"status": "ok" | "errors" | "error", "model_ok": bool, "queries_ok": bool | null,
      "errors": [...]}``. Each error has ``message``, ``source`` ("model" or "query") and
      either ``file_line`` (line in the model file), ``element`` (e.g. "guard of edge
      Train: Near -> Cross"), ``text`` (the offending line) and ``xpath``, or
      ``query_file_line`` (1-based line in the query text/file). ``queries_ok`` is
      null when no queries were given or the model itself failed to compile.
      Runtime errors such as integer overflow are NOT detected here; ``verify``
      reports them as verdict "error".
    """
    ws = _ws()
    mpath = ws.model_path(model)
    if query_file is not None and query_text is not None:
        return _error("Provide at most one of query_file or query_text.")

    def work() -> dict[str, Any]:
        mres = vt.compile_model(mpath)
        result: dict[str, Any] = {
            "model": ws.rel(mpath),
            "model_ok": mres.ok,
            "queries_ok": None,
            "errors": mdl.locate_diagnostics(mpath, mres.diagnostics),
        }
        if mres.message:
            result["message"] = mres.message
        if mres.ok and (query_file is not None or query_text is not None):
            if query_text is not None:
                with vt.inline_query_file(query_text) as qpath:
                    qres = vt.check_queries(mpath, qpath)
            else:
                qres = vt.check_queries(mpath, ws.query_path(query_file))  # type: ignore[arg-type]
            result["queries_ok"] = qres.ok
            result["errors"] += mdl.locate_diagnostics(mpath, qres.diagnostics)
            if qres.message:
                result["message"] = qres.message
        ok = result["model_ok"] and result["queries_ok"] is not False
        return {"status": "ok" if ok else "errors", **result}

    return await anyio.to_thread.run_sync(work)


@mcp.tool()
@safe_tool
async def verify(
    model: str,
    query_file: str | None = None,
    query_text: str | None = None,
    trace_type: Literal["none", "some", "shortest", "fastest"] = "some",
    search_order: Literal["bfs", "dfs", "random_dfs", "optimal", "random_optimal"] = "bfs",
    timeout_s: float = vt.DEFAULT_TIMEOUT_S,
) -> dict[str, Any]:
    """Run verifyta on a model and return one verdict per query.

    Args:
      model: model name or path inside ``models/``.
      query_file: ``.q`` file inside ``queries/`` (approved) or ``queries/proposed/``.
      query_text: inline queries, one per line (symbolic only: A[], E<>, A<>, E[], -->).
        If neither query_file nor query_text is given, the model's full approved suite
        ``queries/<model>.q`` is run.
      trace_type: "some" (default), "shortest", "fastest" or "none". A trace is a
        counterexample for a violated A[]/A<> query or a witness for a satisfied E<>/E[] query.
      search_order: "bfs" (default), "dfs", "random_dfs"; "optimal" and "random_optimal"
        require trace_type "shortest" or "fastest".
      timeout_s: wall-clock limit for the whole run, 0 < timeout_s <= 600 (default 60).

    Returns:
      ``{"status", "run_id", "queries": [...], "duration_s", "message", "errors"}``.
      ``status``: "ok" (every query got a verdict), "timeout", "compile_error" or "error".
      Each query: ``index``, ``line``, ``formula``, ``verdict``, ``detail``, ``stats``,
      ``has_trace``. Verdicts:
        - "satisfied": verifyta proved the property.
        - "violated": verifyta disproved it; use get_trace(run_id, index) for the counterexample.
        - "maybe": inconclusive (approximate state representation) - treat as UNKNOWN.
        - "timeout": not decided within timeout_s - UNKNOWN, never satisfied.
        - "error": not decided (compile/runtime error, memory limit) - see ``detail``.
      Every run is logged to runs.jsonl and stored under traces/<run_id>/.
    """
    ws = _ws()
    mpath = ws.model_path(model)
    if query_file is not None and query_text is not None:
        return _error("Provide at most one of query_file or query_text.")
    if query_file is None and query_text is None:
        qpath: Path | None = ws.approved_queries_for(mpath)
        if not qpath.is_file():
            return _error(f"No approved query suite {ws.rel(qpath)}; pass query_file or query_text.")
        query_source = ws.rel(qpath)
    elif query_file is not None:
        qpath = ws.query_path(query_file)
        query_source = ws.rel(qpath)
    else:
        qpath = None
        query_source = "inline"

    run_id = ws.new_run_id()
    run_dir = ws.run_dir(run_id)
    options = {"trace_type": trace_type, "search_order": search_order, "timeout_s": timeout_s, "memory_mb": _memory_mb()}
    git = ws.git_info(mpath)

    def work() -> vt.RunResult:
        return vt.verify(
            mpath,
            query_file=qpath,
            query_text=query_text,
            trace_type=trace_type,
            search_order=search_order,
            timeout_s=timeout_s,
            memory_mb=options["memory_mb"],
            trace_dir=run_dir if trace_type != "none" else None,
        )

    run = await anyio.to_thread.run_sync(work)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "stdout.txt").write_text(run.stdout)
    (run_dir / "stderr.txt").write_text(run.stderr)

    queries = []
    for q in run.queries:
        queries.append(
            {
                "index": q.index,
                "line": q.line,
                "formula": q.formula,
                "verdict": q.verdict.value,
                "detail": q.detail,
                "stats": q.stats,
                "has_trace": q.trace_file is not None,
            }
        )
    meta = {
        "run_id": run_id,
        "timestamp": utc_now(),
        "model": ws.rel(mpath),
        "model_git": git,
        "query_source": query_source,
        "options": options,
        "status": run.status,
        "message": run.message,
        "duration_s": run.duration_s,
        "returncode": run.returncode,
        "command": [ws.rel(Path(c)) if Path(c).is_absolute() and Path(c).resolve().is_relative_to(ws.root) else c for c in run.command],
        "queries": [{**qd, "trace_file": ws.rel(Path(q.trace_file)) if q.trace_file else None} for qd, q in zip(queries, run.queries)],
    }
    (run_dir / "run.json").write_text(json.dumps(meta, indent=2))
    ws.append_runs(
        [
            {
                "run_id": run_id,
                "timestamp": meta["timestamp"],
                "model": meta["model"],
                "model_git_commit": git["commit"],
                "model_git_blob": git["blob"],
                "model_dirty": git["dirty"],
                "query_source": query_source,
                "query_index": q["index"],
                "query": q["formula"],
                "options": options,
                "verdict": q["verdict"],
                "duration_s": run.duration_s,
                "run_status": run.status,
            }
            for q in queries
        ]
    )
    return {
        "status": run.status,
        "run_id": run_id,
        "model": ws.rel(mpath),
        "query_source": query_source,
        "queries": queries,
        "duration_s": run.duration_s,
        "message": run.message,
        "errors": mdl.locate_diagnostics(mpath, run.diagnostics),
        "run_dir": ws.rel(run_dir),
    }


def main() -> None:
    mcp.run("stdio")


if __name__ == "__main__":
    main()
