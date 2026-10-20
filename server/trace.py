"""Parse verifyta's XML symbolic traces (``-X``) into JSON-friendly steps.

Trace format (verifyta 5.0.0, see tests/fixtures/train_gate_trace_q2.xtr.xml):

* ``<system>`` lists clocks, variables and, per process, its locations and the
  edges used by the trace (``<edge id from to>`` with guard/sync/update).
* ``<node id location_vector dbm_instance variable_vector>`` is a symbolic state.
* ``<dbm_instance>`` holds ``<clockbound clock1 clock2 bound comp>`` entries,
  meaning ``clock1 - clock2 comp bound``. ``sys.#t(0)`` is the zero clock and
  ``sys.#time`` is global time.
* ``<transition from to edges>`` links nodes; ``trace/@initial_node`` is the start.

There is no explicit loop marker for liveness counterexamples; ``find_loop``
reports a loop only when the final state equals an earlier one exactly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lxml import etree

ZERO = "sys.#t(0)"
TIME = "sys.#time"
_PARSER = etree.XMLParser(resolve_entities=False, no_network=True, huge_tree=True)


class TraceError(ValueError):
    """The trace file is missing or cannot be interpreted."""


@dataclass(frozen=True, order=True)
class Bound:
    """DBM bound ``(value, strict)``; ``value`` may be ``math.inf``."""

    value: float
    strict: bool  # True for '<', False for '<='

    def __add__(self, other: "Bound") -> "Bound":
        if math.isinf(self.value) or math.isinf(other.value):
            return Bound(math.inf, True)
        return Bound(self.value + other.value, self.strict or other.strict)

    def tighter_than(self, other: "Bound") -> bool:
        return self.value < other.value or (self.value == other.value and self.strict and not other.strict)

    @property
    def finite(self) -> bool:
        return not math.isinf(self.value)


def _num(v: float) -> int | float:
    return int(v) if float(v).is_integer() else v


def _short(clock_id: str) -> str:
    return "time" if clock_id == TIME else clock_id.removeprefix("sys.")


def _zone(dbm: dict[tuple[str, str], Bound], clocks: list[str]) -> tuple[dict[str, Any], list[str], dict[str, Any] | None]:
    """Render a canonical DBM as per-clock bounds, a minimal list of constraint
    strings, and the global-time interval."""
    inf = Bound(math.inf, True)

    def d(i: str, j: str) -> Bound:
        return dbm.get((i, j), inf)

    def interval(c: str) -> dict[str, Any]:
        lo, hi = d(ZERO, c), d(c, ZERO)
        return {
            "min": _num(-lo.value) if lo.finite else 0,
            "min_strict": lo.strict if lo.finite else False,
            "max": _num(hi.value) if hi.finite else None,
            "max_strict": hi.strict if hi.finite else False,
        }

    def render_interval(name: str, iv: dict[str, Any]) -> str | None:
        lo, hi = iv["min"], iv["max"]
        lop = "<" if iv["min_strict"] else "<="
        hop = "<" if iv["max_strict"] else "<="
        if hi is not None and lo == hi and not iv["min_strict"] and not iv["max_strict"]:
            return f"{name} == {lo}"
        if hi is None:
            return None if (lo == 0 and not iv["min_strict"]) else f"{name} {'>' if iv['min_strict'] else '>='} {lo}"
        if lo == 0 and not iv["min_strict"]:
            return f"{name} {hop} {hi}"
        return f"{lo} {lop} {name} {hop} {hi}"

    bounds: dict[str, Any] = {}
    constraints: list[str] = []
    for c in clocks:
        iv = interval(c)
        bounds[_short(c)] = iv
        s = render_interval(_short(c), iv)
        if s:
            constraints.append(s)
    done: set[tuple[str, str]] = set()
    for i in clocks:
        for j in clocks:
            if i == j or (i, j) in done:
                continue
            dij, dji = d(i, j), d(j, i)
            keep_ij = dij.finite and dij.tighter_than(d(i, ZERO) + d(ZERO, j))
            keep_ji = dji.finite and dji.tighter_than(d(j, ZERO) + d(ZERO, i))
            ni, nj = _short(i), _short(j)
            if dij.finite and dji.finite and dij.value == -dji.value and not dij.strict and not dji.strict and (keep_ij or keep_ji):
                c = _num(dij.value)
                constraints.append(f"{ni} == {nj}" if c == 0 else f"{ni} - {nj} == {c}")
                done |= {(i, j), (j, i)}
                continue
            if keep_ij:
                constraints.append(f"{ni} - {nj} {'<' if dij.strict else '<='} {_num(dij.value)}")
            done.add((i, j))
    time = None
    if (TIME, ZERO) in dbm or (ZERO, TIME) in dbm:
        time = interval(TIME)
    return bounds, constraints, time


def _fmt_edge(e: dict[str, Any]) -> str:
    parts = [f"{e['process']}: {e['from']} -> {e['to']}"]
    if e["guard"]:
        parts.append(f"when {e['guard']}")
    if e["sync"]:
        parts.append(f"sync {e['sync']}")
    if e["update"]:
        parts.append(f"do {e['update']}")
    return ", ".join(parts)


def parse_trace(path: Path) -> dict[str, Any]:
    """Parse an XML trace into ``{"processes", "clocks", "variables", "steps"}``.

    Each step: ``step`` (0-based), ``locations`` (process -> location name),
    ``variables`` (name -> value; locals are ``process.name``, arrays are flattened
    as ``a[0]``), ``changed`` (name -> [old, new] vs. previous step),
    ``clock_constraints`` (minimal readable constraints), ``clock_bounds``,
    ``time`` (global-time interval) and ``via`` (the transition taken to reach this
    step; null for step 0).
    """
    if not path.is_file():
        raise TraceError(f"Trace file not found: {path}")
    try:
        root = etree.parse(str(path), _PARSER).getroot()
    except etree.XMLSyntaxError as e:
        raise TraceError(f"Malformed trace XML: {e}") from e
    if root.tag != "trace":
        raise TraceError(f"Not a verifyta XML trace (root element <{root.tag}>)")

    system = root.find("system")
    if system is None:
        raise TraceError("Trace has no <system> section")
    clocks = [c.get("id") for c in system.iter("clock") if c.get("id") not in (ZERO, TIME)]
    var_names = {v.get("id"): v.get("id").removeprefix("sys.") for v in system.iter("variable")}
    loc_names: dict[str, str] = {}
    processes: list[str] = []
    edges: dict[str, dict[str, Any]] = {}
    for p in system.iterfind("process"):
        pname = p.get("name") or p.get("id")
        processes.append(pname)
        for loc in p.iterfind("location"):
            loc_names[loc.get("id")] = loc.get("name") or loc.get("id").split(".", 1)[-1]
        for e in p.iterfind("edge"):
            guard = (e.findtext("guard") or "").strip()
            sync = (e.findtext("sync") or "").strip()
            update = (e.findtext("update") or "").strip()
            edges[e.get("id")] = {
                "process": pname,
                "from": loc_names.get(e.get("from"), e.get("from")),
                "to": loc_names.get(e.get("to"), e.get("to")),
                "guard": None if guard in ("", "1") else guard,
                "sync": None if sync in ("", "tau") else sync,
                "update": update or None,
            }

    locvecs = {lv.get("id"): lv.get("locations", "").split() for lv in root.iterfind("location_vector")}
    varvecs = {
        vv.get("id"): {var_names.get(s.get("variable"), s.get("variable")): _int(s.get("value")) for s in vv.iterfind("variable_state")}
        for vv in root.iterfind("variable_vector")
    }
    dbms: dict[str, dict[tuple[str, str], Bound]] = {}
    for dbm in root.iterfind("dbm_instance"):
        entries: dict[tuple[str, str], Bound] = {}
        for cb in dbm.iterfind("clockbound"):
            b = cb.get("bound")
            entries[(cb.get("clock1"), cb.get("clock2"))] = Bound(math.inf if b == "inf" else float(b), cb.get("comp") == "<")
        dbms[dbm.get("id")] = entries
    nodes = {n.get("id"): n for n in root.iterfind("node")}
    out_edges = {t.get("from"): t for t in root.iterfind("transition")}

    steps: list[dict[str, Any]] = []
    node_id = root.get("initial_node")
    via: dict[str, Any] | None = None
    seen: set[str] = set()
    prev_vars: dict[str, Any] = {}
    while node_id is not None:
        if node_id in seen:  # defensive: verifyta traces are linear
            break
        seen.add(node_id)
        node = nodes.get(node_id)
        if node is None:
            raise TraceError(f"Trace refers to unknown node {node_id}")
        locs: dict[str, str] = {}
        for lid in locvecs.get(node.get("location_vector"), []):
            proc = lid.split(".", 1)[0]
            locs[proc] = loc_names.get(lid, lid.split(".", 1)[-1])
        variables = varvecs.get((node.get("variable_vector") or "").strip(), {})
        bounds, constraints, time = _zone(dbms.get((node.get("dbm_instance") or "").strip(), {}), clocks)
        step: dict[str, Any] = {
            "step": len(steps),
            "locations": locs,
            "variables": variables,
            "changed": {k: [prev_vars.get(k), v] for k, v in variables.items() if steps and prev_vars.get(k) != v},
            "clock_constraints": constraints,
            "clock_bounds": bounds,
            "time": time,
            "via": via,
        }
        steps.append(step)
        prev_vars = variables
        t = out_edges.get(node_id)
        if t is None:
            break
        used = [edges[e] for e in t.get("edges", "").split() if e in edges]
        senders = [e for e in used if e["sync"] and e["sync"].endswith("!")]
        action = (senders[0]["sync"][:-1] if senders else (used[0]["sync"][:-1] if used and used[0]["sync"] else None)) if used else None
        via = {
            "action": action or "internal (tau)",
            "edges": used,
            "description": "; ".join(_fmt_edge(e) for e in used),
        }
        node_id = t.get("to")
    return {"processes": processes, "clocks": [_short(c) for c in clocks], "variables": list(var_names.values()), "steps": steps}


def _int(v: str | None) -> int | float | str | None:
    if v is None:
        return None
    try:
        return int(v)
    except ValueError:
        try:
            return float(v)
        except ValueError:
            return v


def find_loop(steps: list[dict[str, Any]]) -> int | None:
    """Index of an earlier step identical to the last one (same locations,
    variables and clock constraints), i.e. where an infinite counterexample
    loops back. ``None`` if no such step exists."""
    if len(steps) < 2:
        return None
    last = steps[-1]
    key = (last["locations"], last["variables"], last["clock_constraints"])
    for s in steps[:-1]:
        if (s["locations"], s["variables"], s["clock_constraints"]) == key:
            return s["step"]
    return None


def compact_step(step: dict[str, Any]) -> dict[str, Any]:
    """A step without the bulky ``clock_bounds`` and with ``via`` reduced to its
    description, for summaries."""
    return {
        "step": step["step"],
        "via": step["via"]["description"] if step["via"] else None,
        "action": step["via"]["action"] if step["via"] else None,
        "locations": step["locations"],
        "variables": step["variables"],
        "changed": step["changed"],
        "clock_constraints": step["clock_constraints"],
        "time": _render_time(step["time"]),
    }


def _render_time(t: dict[str, Any] | None) -> str | None:
    if t is None:
        return None
    lo = f"{'(' if t['min_strict'] else '['}{t['min']}"
    hi = f"{t['max']}{')' if t['max_strict'] else ']'}" if t["max"] is not None else "inf)"
    return f"{lo}, {hi}"


def reading_guide(formula: str, verdict: str) -> str:
    """How the agent should interpret the trace for this kind of query."""
    f = formula.strip()
    if verdict == "violated":
        if f.startswith("A[]"):
            return "Counterexample: the LAST step is a reachable state where the A[] property is false."
        if f.startswith("A<>") or "-->" in f:
            return (
                "Counterexample to a liveness property: an execution along which the required "
                "state is never reached. It either loops back (see 'loop_back_to_step') or ends "
                "in a state that cannot progress (deadlock or time-lock)."
            )
        if f.startswith("E[]"):
            return "E[] is violated without a witness path; a trace, if present, is diagnostic only."
        return "Counterexample trace for the violated property."
    if verdict == "satisfied":
        if f.startswith("E<>"):
            return "Witness: the LAST step is a reachable state satisfying the E<> property."
        if f.startswith("E[]"):
            return "Witness: an execution along which the property holds forever (loops back or ends in a maximal state)."
        return "Witness trace for the satisfied property."
    return "Trace for a query that was not decided; do not draw conclusions about the property from it."
