"""Thin, defensive wrapper around UPPAAL's ``verifyta`` (validated against 5.0.0).

Facts about verifyta 5.0.0 this module relies on (see docs/verifyta-5.0-help.txt
and tests/fixtures/ for captured output):

* ``-t 0|1|2``  trace type: some / shortest / fastest.
* ``-o 0..4``   search order: BFS / DFS / random DFS / optimal-first /
  random optimal DFS (3 and 4 require ``-t1`` or ``-t2``).
* ``-X prefix`` writes each trace as XML to ``<prefix><n>.xml`` where ``n`` counts
  *traces produced*, not queries. The path is announced on stdout as
  ``XMLTrace outputted to: <path>`` right after the query's verdict.
* ``-q`` suppresses the option summary, ``-s`` the progress indicator,
  ``-u`` adds per-query statistics.
* The exit code is 0 even when a property is violated; it is 1 when the model or
  query file fails to compile or a query aborts. Verdicts must therefore be
  parsed from stdout.
* Per-query output: ``Verifying formula <k> at <qfile>:<line>`` followed by one of
  ``-- Formula is satisfied.`` / ``-- Formula is NOT satisfied.`` /
  ``-- Formula MAY be satisfied.`` / ``-- Formula MAY NOT be satisfied.`` /
  ``-- Aborted.``
* Diagnostics on stderr: ``<model>:<xpath>:<line>: [error] <msg>``. When the xpath
  is empty the diagnostic refers to the query file, and ``<line>`` is a line of
  the query file.

Safety rule: a query is reported ``satisfied`` only when verifyta printed exactly
``Formula is satisfied.`` for it. Anything else (no verdict, timeout, crash,
memory exhaustion) maps to ``timeout`` or ``error``.
"""

from __future__ import annotations

import contextlib
import os
import re
import resource
import signal
import subprocess
import tempfile
import threading
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

DEFAULT_VERIFYTA = "/home/user/Desktop/uppaal-5.0.0-linux64/bin/verifyta"
DEFAULT_TIMEOUT_S = 60.0
MAX_TIMEOUT_S = 600.0
DEFAULT_MEMORY_MB = 4096

TRACE_TYPES: dict[str, str | None] = {"none": None, "some": "0", "shortest": "1", "fastest": "2"}
SEARCH_ORDERS: dict[str, str] = {
    "bfs": "0",
    "dfs": "1",
    "random_dfs": "2",
    "optimal": "3",
    "random_optimal": "4",
}


class Verdict(str, Enum):
    SATISFIED = "satisfied"
    VIOLATED = "violated"
    MAYBE = "maybe"
    TIMEOUT = "timeout"
    ERROR = "error"


@dataclass
class Diagnostic:
    """One error/warning reported by verifyta."""

    severity: str  # "error" | "warning"
    message: str
    source: str  # "model" | "query"
    xpath: str | None = None  # element in the model, for source == "model"
    line: int | None = None  # 1-based line in the model or query file
    raw: str = ""


@dataclass
class Query:
    index: int  # 1-based position in the query file
    line: int  # 1-based line in the query file
    formula: str
    comment: str = ""


@dataclass
class QueryResult:
    index: int
    line: int | None
    formula: str
    verdict: Verdict
    detail: str = ""
    stats: dict[str, Any] = field(default_factory=dict)
    trace_file: str | None = None
    requirement: str = ""


@dataclass
class RunResult:
    """Outcome of one verifyta invocation."""

    status: str  # "ok" | "compile_error" | "timeout" | "error"
    queries: list[QueryResult]
    diagnostics: list[Diagnostic]
    duration_s: float
    returncode: int | None
    timed_out: bool
    command: list[str]
    stdout: str
    stderr: str
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        for q in d["queries"]:
            q["verdict"] = q["verdict"].value if isinstance(q["verdict"], Verdict) else q["verdict"]
        return d


class VerifytaOptionError(ValueError):
    """Raised for invalid user-supplied options (reported back, never crashes)."""


# --------------------------------------------------------------------------- #
# Query files
# --------------------------------------------------------------------------- #

_SMC_RE = re.compile(
    r"^\s*(Pr\s*[\[(]|simulate\b|E\s*\[\s*(<=|#|\d)|control\s*:|"
    r"(save|load)Strategy\b|strategy\s+\w+\s*=)"
)


def parse_query_file(text: str) -> list[Query]:
    """Split a ``.q`` file into queries, one non-comment, non-blank line each.

    ``/* ... */`` block comments directly preceding a query are kept as its
    ``comment``; ``//`` line comments are stripped.
    """
    queries: list[Query] = []
    pending_comment: list[str] = []
    in_block = False
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw
        code = ""
        i = 0
        while i < len(line):
            if in_block:
                end = line.find("*/", i)
                if end < 0:
                    pending_comment.append(line[i:].strip())
                    i = len(line)
                else:
                    pending_comment.append(line[i:end].strip())
                    in_block = False
                    i = end + 2
            else:
                start = line.find("/*", i)
                slash = line.find("//", i)
                if slash >= 0 and (start < 0 or slash < start):
                    code += line[i:slash]
                    break
                if start < 0:
                    code += line[i:]
                    break
                code += line[i:start]
                in_block = True
                i = start + 2
        code = code.strip()
        if code:
            comment = " ".join(c for c in pending_comment if c).strip()
            queries.append(Query(index=len(queries) + 1, line=lineno, formula=code, comment=comment))
            pending_comment = []
    return queries


def requirement_of(comment: str) -> str:
    """The requirement text of a query's preceding comment (``REQ:`` prefix removed)."""
    c = comment.strip()
    return c[4:].strip() if c.upper().startswith("REQ:") else c


def find_smc_queries(queries: list[Query]) -> list[Query]:
    """Return queries that are statistical (SMC) or strategy queries, which this
    server does not support."""
    return [q for q in queries if _SMC_RE.match(q.formula)]


# --------------------------------------------------------------------------- #
# Output parsing
# --------------------------------------------------------------------------- #

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
_FORMULA_RE = re.compile(r"^Verifying formula (\d+) at (.*):(\d+)\s*$")
_VERDICTS: list[tuple[re.Pattern[str], Verdict]] = [
    (re.compile(r"^-- Formula is satisfied\.$"), Verdict.SATISFIED),
    (re.compile(r"^-- Formula is NOT satisfied\.$"), Verdict.VIOLATED),
    (re.compile(r"^-- Formula MAY be satisfied\.$"), Verdict.MAYBE),
    (re.compile(r"^-- Formula MAY NOT be satisfied\.$"), Verdict.MAYBE),
    (re.compile(r"^-- Aborted\.$"), Verdict.ERROR),
]
_TRACE_RE = re.compile(r"^XMLTrace outputted to: (.*)$")
_STAT_RE = re.compile(r"^-- (States stored|States explored|CPU user time used|Virtual memory used|Resident memory used) : (\d+) (\w+)")
_DIAG_RE = re.compile(r"^(?P<file>.*?):(?P<xpath>/[^:]*)?:(?P<line>\d+): \[(?P<sev>error|warning)\] (?P<msg>.*)$")


def strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


def parse_diagnostics(stderr: str) -> list[Diagnostic]:
    """Parse verifyta's ``file:xpath:line: [severity] message`` lines."""
    diags: list[Diagnostic] = []
    for raw in strip_ansi(stderr).splitlines():
        m = _DIAG_RE.match(raw.strip())
        if not m:
            continue
        xpath = m.group("xpath")
        diags.append(
            Diagnostic(
                severity=m.group("sev"),
                message=m.group("msg"),
                source="model" if xpath else "query",
                xpath=xpath,
                line=int(m.group("line")),
                raw=raw.strip(),
            )
        )
    return diags


_STAT_KEYS = {
    "States stored": "states_stored",
    "States explored": "states_explored",
    "CPU user time used": "cpu_ms",
    "Virtual memory used": "virtual_kib",
    "Resident memory used": "resident_kib",
}


def parse_stdout(stdout: str) -> list[dict[str, Any]]:
    """Parse per-query blocks from verifyta stdout.

    Returns one dict per ``Verifying formula`` header with keys ``index``,
    ``line``, ``verdict`` (``Verdict`` or ``None`` when no verdict line was seen),
    ``detail``, ``stats`` and ``trace_file``.
    """
    blocks: list[dict[str, Any]] = []
    cur: dict[str, Any] | None = None
    for raw in strip_ansi(stdout).splitlines():
        line = raw.strip()
        if not line:
            continue
        m = _FORMULA_RE.match(line)
        if m:
            cur = {
                "index": int(m.group(1)),
                "line": int(m.group(3)),
                "verdict": None,
                "detail": "",
                "stats": {},
                "trace_file": None,
            }
            blocks.append(cur)
            continue
        if cur is None:
            continue
        for rx, verdict in _VERDICTS:
            if rx.match(line):
                cur["verdict"] = verdict
                cur["detail"] = line.removeprefix("-- ")
                break
        else:
            sm = _STAT_RE.match(line)
            if sm:
                cur["stats"][_STAT_KEYS[sm.group(1)]] = int(sm.group(2))
                continue
            tm = _TRACE_RE.match(line)
            if tm:
                cur["trace_file"] = tm.group(1).strip()
    return blocks


def build_results(
    queries: list[Query],
    stdout: str,
    diagnostics: list[Diagnostic],
    timed_out: bool,
    returncode: int | None,
) -> list[QueryResult]:
    """Combine the query list with parsed output into one result per query.

    Rules (in order):
      * a parsed verdict line wins (it is verifyta's own statement);
      * no verdict and the run timed out -> ``timeout``;
      * no verdict otherwise -> ``error`` (with the reason, if known).
    """
    blocks = {b["index"]: b for b in parse_stdout(stdout)}
    query_diags = [d for d in diagnostics if d.source == "query" and d.severity == "error"]
    model_errors = [d for d in diagnostics if d.source == "model" and d.severity == "error"]
    results: list[QueryResult] = []
    for q in queries:
        b = blocks.get(q.index)
        if b and b["verdict"] is not None:
            detail = b["detail"]
            if b["verdict"] is Verdict.ERROR and model_errors:
                detail += " " + "; ".join(d.message for d in model_errors)
            results.append(
                QueryResult(q.index, q.line, q.formula, b["verdict"], detail, b["stats"], b["trace_file"])
            )
            continue
        if timed_out:
            started = b is not None
            detail = ("Verification started but did not finish" if started else "Not started") + " before the timeout."
            results.append(QueryResult(q.index, q.line, q.formula, Verdict.TIMEOUT, detail))
            continue
        own = [d.message for d in query_diags if d.line == q.line]
        if own:
            detail = "Query error: " + "; ".join(own)
        elif query_diags or model_errors:
            detail = "Not verified: the model or query file failed to compile."
        elif b is not None and returncode is not None and returncode < 0:
            detail = (
                f"verifyta was killed by signal {-returncode} during verification without a verdict; "
                "most likely the memory limit was exceeded."
            )
        elif b is not None:
            detail = f"verifyta stopped during verification without a verdict (exit code {returncode})."
        else:
            detail = f"verifyta produced no result for this query (exit code {returncode})."
        results.append(QueryResult(q.index, q.line, q.formula, Verdict.ERROR, detail))
    for q, r in zip(queries, results):
        r.requirement = requirement_of(q.comment)
    return results


# --------------------------------------------------------------------------- #
# Running
# --------------------------------------------------------------------------- #


def build_command(
    verifyta: str,
    model: Path,
    query_file: Path,
    trace_type: str = "none",
    search_order: str = "bfs",
    trace_prefix: Path | None = None,
) -> list[str]:
    """Build the verifyta argument list. Raises ``VerifytaOptionError`` on bad options."""
    if trace_type not in TRACE_TYPES:
        raise VerifytaOptionError(f"trace_type must be one of {sorted(TRACE_TYPES)}, got {trace_type!r}")
    if search_order not in SEARCH_ORDERS:
        raise VerifytaOptionError(f"search_order must be one of {sorted(SEARCH_ORDERS)}, got {search_order!r}")
    if search_order in ("optimal", "random_optimal") and trace_type not in ("shortest", "fastest"):
        raise VerifytaOptionError(f"search_order {search_order!r} requires trace_type 'shortest' or 'fastest'")
    cmd = [verifyta, "-q", "-s", "-u", "-o", SEARCH_ORDERS[search_order]]
    t = TRACE_TYPES[trace_type]
    if t is not None:
        if trace_prefix is None:
            raise VerifytaOptionError("trace_prefix is required when a trace is requested")
        cmd += ["-t", t, "-X", str(trace_prefix)]
    cmd += [str(model), str(query_file)]
    return cmd


def _set_memory_limit(pid: int, memory_mb: int) -> None:
    limit = memory_mb * 1024 * 1024
    try:
        resource.prlimit(pid, resource.RLIMIT_AS, (limit, limit))
    except (ProcessLookupError, PermissionError, OSError):
        pass  # process already gone, or limit cannot be applied


def _kill_group(proc: subprocess.Popen[str]) -> None:
    for sig, wait in ((signal.SIGTERM, 2.0), (signal.SIGKILL, 5.0)):
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            return
        try:
            proc.wait(timeout=wait)
            return
        except subprocess.TimeoutExpired:
            continue


def run_process(cmd: list[str], timeout_s: float, memory_mb: int, cwd: Path | None = None) -> tuple[str, str, int | None, bool, float]:
    """Run ``cmd`` (no shell) with a wall-clock timeout and address-space limit.

    Returns ``(stdout, stderr, returncode, timed_out, duration_s)``. On timeout the
    whole process group is terminated and partial output is returned.
    """
    start = time.monotonic()
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        stdin=subprocess.DEVNULL,
        text=True,
        cwd=cwd,
        start_new_session=True,
    )
    _set_memory_limit(proc.pid, memory_mb)
    timed_out = False
    try:
        out, err = proc.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        timed_out = True
        _kill_group(proc)
        out, err = proc.communicate()
    return out or "", err or "", proc.returncode, timed_out, time.monotonic() - start


@dataclass
class SyntaxCheck:
    ok: bool
    diagnostics: list[Diagnostic]
    message: str = ""


def compile_model(model: Path, timeout_s: float = 30.0, verifyta: str | None = None) -> SyntaxCheck:
    """Compile ``model`` without verifying anything (``UPPAAL_COMPILE_ONLY``).

    Detects syntax and type errors in declarations, guards, invariants, updates
    and the system declaration. Runtime errors (e.g. integer overflow) are only
    found by ``verify``.
    """
    verifyta = verifyta or os.environ.get("VERIFYTA", DEFAULT_VERIFYTA)
    env = {**os.environ, "UPPAAL_COMPILE_ONLY": "1"}
    start = time.monotonic()
    proc = subprocess.Popen(
        [verifyta, "-q", "-s", str(model)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        stdin=subprocess.DEVNULL,
        text=True,
        env=env,
        start_new_session=True,
    )
    try:
        _, err = proc.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        _kill_group(proc)
        return SyntaxCheck(False, [], f"Compilation did not finish within {timeout_s:g}s ({time.monotonic() - start:.1f}s).")
    diags = parse_diagnostics(err or "")
    ok = proc.returncode == 0 and not any(d.severity == "error" for d in diags)
    msg = "" if ok else (f"verifyta exited with code {proc.returncode}: {(err or '').strip()[:500]}" if not diags else "")
    return SyntaxCheck(ok, diags, msg)


def check_queries(model: Path, query_file: Path, timeout_s: float = 30.0, verifyta: str | None = None) -> SyntaxCheck:
    """Type-check the queries in ``query_file`` against ``model`` without verifying them.

    verifyta compiles all queries before verifying the first one, and exits with
    diagnostics if any query is invalid. We start a real run and stop it as soon
    as ``Verifying formula 1`` appears (or the process exits).
    """
    verifyta = verifyta or os.environ.get("VERIFYTA", DEFAULT_VERIFYTA)
    proc = subprocess.Popen(
        [verifyta, "-q", "-s", str(model), str(query_file)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        stdin=subprocess.DEVNULL,
        text=True,
        start_new_session=True,
    )
    timer = threading.Timer(timeout_s, _kill_group, args=(proc,))
    timer.start()
    started = False
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            if line.startswith("Verifying formula"):
                started = True
                break
    finally:
        timer.cancel()
        if proc.poll() is None:
            _kill_group(proc)
        _, err = proc.communicate()
    diags = parse_diagnostics(err or "")
    if started and not any(d.severity == "error" for d in diags):
        return SyntaxCheck(True, diags)
    if diags:
        return SyntaxCheck(False, diags)
    return SyntaxCheck(False, [], f"Could not determine query validity (exit code {proc.returncode}).")


@contextlib.contextmanager
def inline_query_file(query_text: str) -> Iterator[Path]:
    """Write ``query_text`` to a temporary ``.q`` file and yield its path."""
    with tempfile.TemporaryDirectory(prefix="uppaal-mcp-") as tmp:
        path = Path(tmp) / "inline.q"
        path.write_text(query_text.rstrip("\n") + "\n")
        yield path


def verify(
    model: Path,
    query_file: Path | None = None,
    query_text: str | None = None,
    *,
    trace_type: str = "none",
    search_order: str = "bfs",
    timeout_s: float = DEFAULT_TIMEOUT_S,
    memory_mb: int = DEFAULT_MEMORY_MB,
    trace_dir: Path | None = None,
    verifyta: str | None = None,
) -> RunResult:
    """Verify ``model`` against a query file or inline query text.

    Never raises for verification problems; they are encoded in the result.
    Raises ``VerifytaOptionError`` only for invalid arguments.
    """
    verifyta = verifyta or os.environ.get("VERIFYTA", DEFAULT_VERIFYTA)
    if (query_file is None) == (query_text is None):
        raise VerifytaOptionError("Provide exactly one of query_file or query_text")
    if not (0 < timeout_s <= MAX_TIMEOUT_S):
        raise VerifytaOptionError(f"timeout_s must be in (0, {MAX_TIMEOUT_S}], got {timeout_s}")
    if not Path(verifyta).is_file():
        raise VerifytaOptionError(f"verifyta not found at {verifyta}; set the VERIFYTA environment variable")
    if not model.is_file():
        raise VerifytaOptionError(f"Model file not found: {model}")

    with contextlib.ExitStack() as stack:
        if query_text is not None:
            query_file = stack.enter_context(inline_query_file(query_text))
        assert query_file is not None
        if not query_file.is_file():
            raise VerifytaOptionError(f"Query file not found: {query_file}")
        queries = parse_query_file(query_file.read_text())
        if not queries:
            raise VerifytaOptionError("No queries found (all lines are blank or comments)")
        smc = find_smc_queries(queries)
        if smc:
            raise VerifytaOptionError(
                "Statistical/strategy queries are not supported (symbolic queries only): "
                + "; ".join(f"line {q.line}: {q.formula}" for q in smc)
            )
        prefix = None
        if trace_type != "none":
            if trace_dir is None:
                raise VerifytaOptionError("trace_dir is required when trace_type is not 'none'")
            trace_dir.mkdir(parents=True, exist_ok=True)
            prefix = trace_dir / "trace-"
        cmd = build_command(verifyta, model, query_file, trace_type, search_order, prefix)
        stdout, stderr, rc, timed_out, duration = run_process(cmd, timeout_s, memory_mb)

    diagnostics = parse_diagnostics(stderr)
    results = build_results(queries, stdout, diagnostics, timed_out, rc)
    if timed_out:
        status, message = "timeout", f"verifyta exceeded the {timeout_s:g}s timeout and was killed."
    elif any(d.severity == "error" for d in diagnostics) and not parse_stdout(stdout):
        status, message = "compile_error", "The model or queries failed to compile; nothing was verified."
    elif rc != 0 or any(r.verdict is Verdict.ERROR for r in results):
        status, message = "error", f"verifyta reported errors (exit code {rc})."
    else:
        status, message = "ok", ""
    return RunResult(status, results, diagnostics, round(duration, 3), rc, timed_out, cmd, stdout, stderr, message)
