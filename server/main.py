"""UPPAAL model-checking MCP server (stdio).

This server is the only component that runs verifyta or modifies model files.
Verdicts always come from verifyta; the server never infers them.

Configuration (environment variables):
  VERIFYTA          path to the verifyta binary
  UPPAAL_WORKSPACE  workspace root (default: the repository containing this file)
  UPPAAL_MEMORY_MB  address-space limit for verifyta (default 4096)
"""

from __future__ import annotations

import difflib
import functools
import json
import os
import tempfile
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, Literal, TypeVar

import anyio
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ResourceError, ResourceNotFoundError

from server import model as mdl
from server import trace as tr
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
      ``requirement``: the requirement text from the ``/* ... */`` comment above the
      query in its .q file (``REQ:`` prefix removed), or null.
      ``exploratory`` is true for query_text runs: such results are sanity checks, not
      verification of an approved requirement.
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
                "requirement": q.requirement or None,
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
                "requirement": q["requirement"],
                "options": options,
                "verdict": q["verdict"],
                "duration_s": run.duration_s,
                "run_status": run.status,
                "exploratory": query_source == "inline",
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
        "exploratory": query_source == "inline",
        **(
            {"note": "Exploratory run (query_text): these results do not verify an approved requirement."}
            if query_source == "inline"
            else {}
        ),
    }


@mcp.tool()
@safe_tool
async def get_trace(
    run_id: str,
    query_index: int | None = None,
    head: int = 5,
    tail: int = 5,
    from_step: int | None = None,
    to_step: int | None = None,
) -> dict[str, Any]:
    """Return the counterexample or witness trace of a query from a verify run.

    Args:
      run_id: the ``run_id`` returned by ``verify``.
      query_index: 1-based ``index`` of the query in that run. May be omitted when
        exactly one query in the run has a trace.
      head, tail: number of first/last steps in ``summary`` (default 5 each).
      from_step, to_step: optional inclusive 0-based step range to return in full
        in ``steps`` (use this to page through long traces).

    Returns:
      ``{"status": "ok", "run_id", "query": {index, formula, verdict}, "kind":
      "counterexample" | "witness", "how_to_read", "length", "processes",
      "loop_back_to_step", "summary": {"first": [...], "last": [...], "omitted_steps"},
      "steps": [...] | null, "trace_json", "raw_trace"}``.

      ``steps`` contains every step when the trace has at most head + tail steps or a
      range was requested; otherwise it is null and the full JSON is in ``trace_json``.
      Each step: ``step`` (0-based), ``via`` (the transition taken to reach it:
      ``action``, ``edges`` [{process, from, to, guard, sync, update}] and a
      ``description``; null for step 0), ``locations`` (process -> location),
      ``variables`` (name -> value; locals as ``process.var``, arrays flattened,
      booleans as 0/1), ``changed`` (var -> [old, new]), ``clock_constraints``
      (symbolic zone, e.g. ``"train.x <= 5"``, ``"train.x - gate.y <= -10"``) and
      ``time`` (global-time interval). Summary steps omit per-clock bounds.
      ``loop_back_to_step`` (liveness counterexamples): the earlier step that the
      last step is identical to, or null if the trace does not loop.
      Cite step numbers when explaining a trace.
    """
    ws = _ws()
    run_dir = ws.run_dir(run_id)
    meta_path = run_dir / "run.json"
    if not meta_path.is_file():
        return _error(f"Unknown run_id {run_id!r}.")
    meta = json.loads(meta_path.read_text())
    with_trace = [q for q in meta["queries"] if q.get("trace_file")]
    if query_index is None:
        if len(with_trace) != 1:
            return _error(
                "query_index is required: queries with traces in this run are "
                f"{[q['index'] for q in with_trace] or 'none'}."
            )
        q = with_trace[0]
    else:
        matches = [q for q in meta["queries"] if q["index"] == query_index]
        if not matches:
            return _error(f"Run {run_id} has no query with index {query_index}.")
        q = matches[0]
        if not q.get("trace_file"):
            hint = (
                " Re-run verify with trace_type 'some' or 'shortest'."
                if meta["options"].get("trace_type") == "none"
                else " verifyta produces traces only for violated A[]/A<>/--> and satisfied E<>/E[] queries."
            )
            return _error(f"Query {query_index} ({q['formula']}, {q['verdict']}) has no trace.{hint}")
    if head < 0 or tail < 0:
        return _error("head and tail must be >= 0.")

    raw = ws.resolve(q["trace_file"], under=run_dir)
    json_path = run_dir / f"trace-q{q['index']}.json"

    def work() -> dict[str, Any]:
        if json_path.is_file():
            return json.loads(json_path.read_text())
        parsed = tr.parse_trace(raw)
        json_path.write_text(json.dumps(parsed, indent=1))
        return parsed

    try:
        parsed = await anyio.to_thread.run_sync(work)
    except tr.TraceError as e:
        return _error(str(e))
    steps = parsed["steps"]
    n = len(steps)
    first = steps[:head]
    last = steps[max(head, n - tail):] if tail else []
    full: list[dict[str, Any]] | None = None
    if from_step is not None or to_step is not None:
        lo = 0 if from_step is None else from_step
        hi = n - 1 if to_step is None else to_step
        if not (0 <= lo <= hi < n):
            return _error(f"Step range must satisfy 0 <= from_step <= to_step < {n}.")
        full = steps[lo : hi + 1]
    elif n <= head + tail:
        full = steps
    return {
        "status": "ok",
        "run_id": run_id,
        "model": meta["model"],
        "query": {"index": q["index"], "formula": q["formula"], "verdict": q["verdict"]},
        "kind": "counterexample" if q["verdict"] == "violated" else "witness",
        "how_to_read": tr.reading_guide(q["formula"], q["verdict"]),
        "length": n,
        "processes": parsed["processes"],
        "loop_back_to_step": tr.find_loop(steps),
        "summary": {
            "first": [tr.compact_step(s) for s in first],
            "last": [tr.compact_step(s) for s in last],
            "omitted_steps": max(0, n - len(first) - len(last)),
        },
        "steps": full,
        "trace_json": ws.rel(json_path),
        "raw_trace": ws.rel(raw),
    }


@mcp.tool()
@safe_tool
async def read_model(model: str) -> dict[str, Any]:
    """Return the structure of a model as JSON.

    Args:
      model: model name or path inside ``models/``.

    Returns:
      ``{"status": "ok", "model", "git", "global_declarations", "templates": [...],
      "system", "embedded_queries"}``. Each template has ``name``, ``parameters``,
      ``declarations`` (local declarations text), ``locations`` and ``edges``.
        - location: ``id``, ``name``, ``invariant`` (null if none), ``urgent``,
          ``committed``, ``initial``, ``file_line``.
        - edge: ``index`` (0-based within the template; use it as ``"edge"`` in
          patch_model), ``id``, ``source``, ``target`` (location names), ``select``,
          ``guard``, ``sync``, ``update`` (null if absent), ``file_line``.
      ``embedded_queries`` are queries saved inside the .xml; they are NOT the
      approved suite (that is ``queries/<model>.q``).
    """
    ws = _ws()
    mpath = ws.model_path(model)
    doc = mdl.Document.load(mpath)
    return {"status": "ok", "model": ws.rel(mpath), "git": ws.git_info(mpath), **mdl.read_model(doc)}


@mcp.tool()
@safe_tool
async def patch_model(model: str, operations: list[dict[str, Any]], reason: str, dry_run: bool = False) -> dict[str, Any]:
    """Apply targeted edits to a model, syntax-check the result, and git-commit it.

    All operations are applied in order to the current file. If any operation
    fails, or the patched model does not compile, NOTHING is written. Untouched
    parts of the file stay byte-identical. The model must have no uncommitted
    changes (so the commit contains only this patch).

    Args:
      model: model name or path inside ``models/``.
      operations: list of objects, each with an ``op`` field:
        - {"op": "set_declaration", "scope": "global" | "system" | <template>, "text": "..."}
        - {"op": "replace_in_declaration", "scope": ..., "old": "...", "new": "..."}
          (``old`` must occur exactly once)
        - {"op": "set_guard" | "set_sync" | "set_update" | "set_select",
           "template": "Train", <edge selector>, "value": "x >= 15"}
          (value "" removes the label)
        - {"op": "set_invariant", "template": "Gate", "location": "Lowering", "value": "y <= 10"}
        - {"op": "add_edge", "template": ..., "from": "A", "to": "B",
           "guard"?: ..., "sync"?: ..., "update"?: ..., "select"?: ...}
        - {"op": "remove_edge", "template": ..., <edge selector>}
        - {"op": "add_location", "template": ..., "name": "Wait", "invariant"?: ...,
           "kind"?: "normal" | "urgent" | "committed"}
        - {"op": "remove_location", "template": ..., "location": "Wait"}
          (also removes its incident edges; the initial location cannot be removed)
        Edge selector: ``"edge": <index from read_model>``, or ``"edge_id": "id4"``,
        or ``"from": "Near", "to": "Cross"`` (must match exactly one edge).
        Locations are referred to by name (or id).
      reason: why this change is made (used in the commit message), e.g.
        "Fix R2: train must not enter the crossing before the gate is down".
      dry_run: if true, check and return the diff without writing or committing.

    Returns:
      ``{"status": "ok" | "syntax_error" | "error", "dry_run", "changes": [...],
      "behaviour_warnings": [{"op_index", "severity", "message"}],
      "removes_behaviour": bool, "diff", "commit", "errors", "next_step"}``.
      ``severity`` "removes_behaviour" (edge/location removed, guard or invariant set
      to false) means the patch deletes executions: such a patch can make a property
      pass by hiding the scenario instead of fixing it - say so to the user.
      "may_change_behaviour" marks changed guards, invariants, syncs and
      declarations. On "syntax_error", ``errors`` lists compiler errors with
      ``file_line`` in the patched model; the file is unchanged.
    """
    ws = _ws()
    mpath = ws.model_path(model)
    if not reason or not reason.strip():
        return _error("'reason' is required: describe why the model is changed.")
    if not dry_run:
        if not ws.is_git_repo():
            return _error(f"{ws.root} is not a git repository; patch_model commits every change.")
        if ws.git_info(mpath)["dirty"]:
            return _error(
                f"{ws.rel(mpath)} has uncommitted changes. Ask the user to commit or discard them first, "
                "so the patch commit contains only this change."
            )
    original = mpath.read_text(encoding="utf-8")
    doc = mdl.Document.from_text(original)
    try:
        changes = mdl.apply_operations(doc, operations)
    except mdl.ModelError as e:
        return _error(str(e) + " No changes were written.")
    patched = doc.serialize()
    warnings = [{"op_index": c.op_index, **f} for c in changes for f in c.flags]
    result: dict[str, Any] = {
        "model": ws.rel(mpath),
        "dry_run": dry_run,
        "changes": [c.description for c in changes],
        "behaviour_warnings": warnings,
        "removes_behaviour": any(w["severity"] == "removes_behaviour" for w in warnings),
    }
    if patched == original:
        return {"status": "ok", **result, "diff": "", "commit": None, "message": "The operations did not change the file."}

    def check() -> tuple[vt.SyntaxCheck, list[dict[str, Any]]]:
        with tempfile.TemporaryDirectory(prefix="uppaal-mcp-") as tmp:
            tmp_model = Path(tmp) / mpath.name
            tmp_model.write_text(patched, encoding="utf-8")
            res = vt.compile_model(tmp_model)
            return res, mdl.locate_diagnostics(tmp_model, res.diagnostics)

    syntax, errors = await anyio.to_thread.run_sync(check)
    rel = ws.rel(mpath)
    diff = "".join(
        difflib.unified_diff(
            original.splitlines(keepends=True), patched.splitlines(keepends=True), f"a/{rel}", f"b/{rel}"
        )
    )
    if not syntax.ok:
        return {
            "status": "syntax_error",
            **result,
            "diff": diff,
            "commit": None,
            "errors": errors,
            "message": (syntax.message or "The patched model does not compile.") + " No changes were written.",
        }
    if dry_run:
        return {"status": "ok", **result, "diff": diff, "commit": None, "next_step": "Call again with dry_run=false to apply."}

    tmp_path = mpath.with_name(f".{mpath.name}.tmp")
    tmp_path.write_text(patched, encoding="utf-8")
    os.replace(tmp_path, mpath)
    lines = [f"Patch {rel}: {reason.strip()}", ""]
    lines += [f"- {c}" for c in result["changes"]]
    if warnings:
        lines += ["", "Behaviour warnings:"] + [f"- [{w['severity']}] {w['message']}" for w in warnings]
    lines += ["", "Applied by uppaal-mcp patch_model; syntax checked with verifyta."]
    try:
        commit = ws.commit_file(mpath, "\n".join(lines))
    except WorkspaceError as e:
        mpath.write_text(original, encoding="utf-8")
        return _error(f"{e}. The model file was restored.")
    approved = ws.approved_queries_for(mpath)
    return {
        "status": "ok",
        **result,
        "diff": ws.git("show", "--format=", "--no-color", commit, "--", rel).stdout or diff,
        "commit": commit,
        "next_step": (
            f"Re-run the FULL approved suite: verify(model={mpath.stem!r})"
            if approved.is_file()
            else "No approved suite exists for this model; verify the relevant queries."
        ),
    }


@mcp.tool()
@safe_tool
async def propose_query(model: str, requirement: str, formulas: list[str], file: str | None = None) -> dict[str, Any]:
    """Record a formalized requirement as proposed queries in ``queries/proposed/``.

    You (the agent) translate the natural-language requirement into one or more
    UPPAAL formulas; this tool type-checks them against the model and appends them,
    each preceded by ``/* REQ: <requirement> */``, to a proposed query file. Approved
    queries in ``queries/`` are read-only: only the user approves a query by moving
    it into ``queries/<model>.q``.

    Args:
      model: model name or path inside ``models/``.
      requirement: the requirement in natural language, as stated by the user.
      formulas: one or more symbolic queries (A[], E<>, A<>, E[], -->), one line
        each. Use several formulas when the requirement has several parts, e.g. a
        safety part and a liveness part, and add an ``E<>`` sanity check when an
        ``A[] (p imply q)`` query could hold vacuously.
      file: optional file name inside ``queries/proposed/`` (default ``<model>.q``).

    Returns:
      ``{"status": "ok", "query_file", "entries": [{"formula", "line"}], ...}`` or
      ``{"status": "errors", "errors": [{"formula_index", "formula", "message"}]}``
      (nothing written). ``already_approved`` lists formulas that already exist in
      the approved suite. Next: show the requirement and the formulas to the user and
      wait for their approval BEFORE calling ``verify(model, query_file=<query_file>)``.
    """
    ws = _ws()
    mpath = ws.model_path(model)
    if not requirement or not requirement.strip():
        return _error("'requirement' (natural-language text) is required.")
    if not isinstance(formulas, list) or not formulas or not all(isinstance(f, str) and f.strip() for f in formulas):
        return _error("'formulas' must be a non-empty list of query strings.")
    cleaned = [" ".join(f.split()) for f in formulas]
    for i, f in enumerate(cleaned):
        if "/*" in f or "*/" in f or "//" in f:
            return _error(f"Formula {i} contains a comment; put explanations in 'requirement'.")
    target = ws.writable_query_path(file or f"{mpath.stem}.q")
    smc = vt.find_smc_queries(vt.parse_query_file("\n".join(cleaned)))
    if smc:
        return _error("Statistical/strategy queries are not supported: " + "; ".join(q.formula for q in smc))

    def check() -> vt.SyntaxCheck:
        with vt.inline_query_file("\n".join(cleaned)) as qpath:
            return vt.check_queries(mpath, qpath)

    res = await anyio.to_thread.run_sync(check)
    if not res.ok:
        errors = []
        for d in res.diagnostics:
            idx = (d.line - 1) if d.source == "query" and d.line else None
            errors.append(
                {
                    "formula_index": idx,
                    "formula": cleaned[idx] if idx is not None and 0 <= idx < len(cleaned) else None,
                    "message": d.message,
                    "source": d.source,
                }
            )
        return {
            "status": "errors",
            "errors": errors or [{"message": res.message or "Queries could not be checked."}],
            "message": "Nothing was written. Fix the formulas (or the model) and try again.",
        }

    req = " ".join(requirement.replace("*/", "* /").split())
    approved = ws.approved_queries_for(mpath)
    approved_formulas = (
        {" ".join(q.formula.split()) for q in vt.parse_query_file(approved.read_text())} if approved.is_file() else set()
    )
    existing = target.read_text(encoding="utf-8") if target.is_file() else ""
    prefix = "" if not existing or existing.endswith("\n\n") else ("\n" if existing.endswith("\n") else "\n\n")
    block = []
    for i, f in enumerate(cleaned):
        part = f" (part {i + 1}/{len(cleaned)})" if len(cleaned) > 1 else ""
        block += [f"/* REQ: {req}{part} */", f]
    start_line = len((existing + prefix).splitlines()) + 1
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as fh:
        fh.write(prefix + "\n".join(block) + "\n")
    return {
        "status": "ok",
        "query_file": ws.rel(target),
        "requirement": req,
        "entries": [{"formula": f, "line": start_line + 2 * i + 1} for i, f in enumerate(cleaned)],
        "already_approved": [f for f in cleaned if f in approved_formulas],
        "next_step": (
            "Show the requirement and formulas to the user and ask them to confirm the formalization. "
            f"After approval run verify(model={mpath.stem!r}, query_file={ws.rel(target)!r})."
        ),
    }


# --------------------------------------------------------------------------- #
# Resources
# --------------------------------------------------------------------------- #


@mcp.resource(
    "uppaal://queries",
    name="approved-queries-index",
    description="Index of approved query files (read-only) with their queries and requirements.",
    mime_type="application/json",
)
def approved_queries_index() -> str:
    ws = _ws()
    files = []
    for p in sorted(ws.queries_dir.glob("*.q")):
        files.append(
            {
                "uri": f"uppaal://queries/{p.stem}",
                "file": ws.rel(p),
                "model": p.stem,
                "queries": [
                    {"index": q.index, "line": q.line, "formula": q.formula, "requirement": vt.requirement_of(q.comment) or None}
                    for q in vt.parse_query_file(p.read_text(encoding="utf-8"))
                ],
            }
        )
    return json.dumps({"approved_query_files": files}, indent=2)


@mcp.resource(
    "uppaal://queries/{name}",
    name="approved-query-file",
    description="Content of an approved query file queries/<name>.q (read-only).",
    mime_type="text/plain",
)
def approved_query_file(name: str) -> str:
    ws = _ws()
    stem = name.removesuffix(".q")
    try:
        p = ws.resolve(f"queries/{stem}.q", under=ws.queries_dir)
    except WorkspaceError as e:
        raise ResourceError(str(e)) from None
    if p.parent != ws.queries_dir.resolve() or not p.is_file():
        raise ResourceNotFoundError(f"No approved query file queries/{stem}.q; see uppaal://queries")
    return p.read_text(encoding="utf-8")


@mcp.resource(
    "uppaal://docs/query-language",
    name="query-language-cheat-sheet",
    description="UPPAAL query language cheat sheet: A[], E<>, A<>, E[], -->, deadlock, patterns and pitfalls.",
    mime_type="text/markdown",
)
def query_language() -> str:
    return (Path(__file__).resolve().parent.parent / "docs/query-language.md").read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
# Prompts
# --------------------------------------------------------------------------- #


@mcp.prompt(
    name="verify_requirement",
    description="Formalize a natural-language requirement, verify it with UPPAAL, and diagnose violations.",
)
def verify_requirement(requirement: str, model: str = "") -> str:
    """The agent loop for checking one requirement."""
    target = f"model `{model}`" if model else "the relevant model (use list_models)"
    return f"""Check this requirement against {target} with UPPAAL:

> {requirement}

verifyta is the only source of truth. Never state a verdict you did not get from `verify`.

1. **Formalize.** Read `uppaal://docs/query-language` and call `read_model` to learn the
   process, location and variable names. Translate the requirement into one or more
   symbolic queries. Add an `E<>` sanity check when an implication could hold vacuously.
   Call `propose_query(model, requirement, formulas)`. Show the user the requirement,
   each formula and what it means in plain words, and WAIT for their approval.
2. **Check, then verify.** `check_syntax` (with the proposed query file), then
   `verify(model, query_file=<proposed file>)`.
3. **If violated:** `get_trace(run_id, query_index)`. Explain the scenario step by step in
   plain language, citing step numbers, locations, variable values and clock constraints.
4. **Diagnose** and say which it is:
   - design bug (the system really behaves wrongly),
   - modeling error (the model does not reflect the real system), or
   - mis-formalized requirement (the query does not say what the user meant).
5. **Fix.** For a design bug or modeling error, propose a minimal change with
   `patch_model(..., dry_run=true)`, show the diff, and apply it after approval. Point out
   every `behaviour_warnings` entry; a patch with `removes_behaviour: true` may only hide
   the problem. Then re-run the FULL approved suite: `verify(model)` with no query
   arguments, plus the proposed queries. Report every verdict.
   For a mis-formalized requirement, propose a corrected query (step 1) instead.
6. **Never** weaken, edit or delete approved queries in `queries/` to make them pass.
7. **On timeout** (or `maybe`/`error`): the result is UNKNOWN, never "satisfied". Try another
   `search_order` (e.g. "dfs"), a smaller model instance or tighter variable bounds, and
   explain the trade-off: DFS or random DFS may find a counterexample sooner, but proving
   the property still requires exploring the whole state space; a smaller instance is
   faster to check but proves the property only for that instance.
"""


def main() -> None:
    mcp.run("stdio")


if __name__ == "__main__":
    main()
