"""UPPAAL .xml model access via lxml: reading, targeted edits, diagnostics.

Edits are applied to the parsed tree and serialized back so that untouched parts
of the file are byte-identical (verified on all UPPAAL 5.0 demo models): the
prolog (XML declaration, DOCTYPE) is copied verbatim, and empty elements keep
their original ``<x></x>`` or ``<x/>`` form.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lxml import etree

from .verifyta import Diagnostic

_PARSER = etree.XMLParser(
    remove_blank_text=False, resolve_entities=False, no_network=True, strip_cdata=False, remove_comments=False
)

# Order in which UPPAAL writes transition labels.
_EDGE_LABELS = ["select", "guard", "synchronisation", "assignment"]
_EDGE_FIELDS = {"select": "select", "guard": "guard", "sync": "synchronisation", "update": "assignment"}


class ModelError(ValueError):
    """An edit operation could not be applied (bad selector, invalid argument)."""


def load(path: Path) -> etree._ElementTree:
    return etree.parse(str(path), _PARSER)


# --------------------------------------------------------------------------- #
# Document with faithful serialization
# --------------------------------------------------------------------------- #


@dataclass
class Document:
    raw: str
    root: etree._Element

    @classmethod
    def from_text(cls, raw: str) -> "Document":
        try:
            root = etree.fromstring(raw.encode("utf-8"), _PARSER)
        except etree.XMLSyntaxError as e:
            raise ModelError(f"Model is not well-formed XML: {e}") from e
        if root.tag != "nta":
            raise ModelError(f"Not an UPPAAL model: root element is <{root.tag}>, expected <nta>")
        _mark_explicit_empty(root, raw)
        return cls(raw, root)

    @classmethod
    def load(cls, path: Path) -> "Document":
        return cls.from_text(path.read_text(encoding="utf-8"))

    def serialize(self) -> str:
        start = self.raw.find("<nta")
        end = self.raw.rfind("</nta>")
        if start < 0:
            raise ModelError("Cannot locate <nta> in the original text")
        suffix = self.raw[end + len("</nta>") :] if end >= 0 else "\n"
        return self.raw[:start] + etree.tostring(self.root, encoding="unicode", with_tail=False) + suffix


def _mark_explicit_empty(root: etree._Element, raw: str) -> None:
    """lxml cannot tell ``<x></x>`` from ``<x/>``; recover the original form so a
    round trip does not rewrite untouched lines."""
    by_tag: dict[str, list[etree._Element]] = {}
    for el in root.iter():
        if isinstance(el.tag, str) and len(el) == 0 and el.text is None:
            by_tag.setdefault(el.tag, []).append(el)
    for tag, els in by_tag.items():
        forms = re.findall(r"<%s\b[^<>]*?(/>|></%s>)" % (re.escape(tag), re.escape(tag)), raw)
        if len(forms) == len(els):
            for el, form in zip(els, forms):
                if form != "/>":
                    el.text = ""


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #


def _label(el: etree._Element, kind: str) -> str | None:
    for lab in el.iterfind("label"):
        if lab.get("kind") == kind:
            return lab.text or ""
    return None


def _loc_name(loc: etree._Element) -> str:
    return (loc.findtext("name") or "").strip() or loc.get("id", "?")


def _loc_by_id(tpl: etree._Element) -> dict[str, etree._Element]:
    return {loc.get("id"): loc for loc in tpl.iterfind("location")}


def _edge_dict(tpl: etree._Element, index: int, tr: etree._Element) -> dict[str, Any]:
    locs = _loc_by_id(tpl)
    src = tr.find("source")
    dst = tr.find("target")
    sid = src.get("ref") if src is not None else None
    did = dst.get("ref") if dst is not None else None
    return {
        "index": index,
        "id": tr.get("id"),
        "source": _loc_name(locs[sid]) if sid in locs else sid,
        "target": _loc_name(locs[did]) if did in locs else did,
        "select": _label(tr, "select"),
        "guard": _label(tr, "guard"),
        "sync": _label(tr, "synchronisation"),
        "update": _label(tr, "assignment"),
        "file_line": tr.sourceline,
    }


def read_model(doc: Document) -> dict[str, Any]:
    """Structured view of the model (see the ``read_model`` tool docstring)."""
    root = doc.root
    templates = []
    for tpl in root.iterfind("template"):
        init = tpl.find("init")
        init_id = init.get("ref") if init is not None else None
        locations = []
        for loc in tpl.iterfind("location"):
            locations.append(
                {
                    "id": loc.get("id"),
                    "name": _loc_name(loc),
                    "invariant": _label(loc, "invariant"),
                    "urgent": loc.find("urgent") is not None,
                    "committed": loc.find("committed") is not None,
                    "initial": loc.get("id") == init_id,
                    "file_line": loc.sourceline,
                }
            )
        decl = tpl.find("declaration")
        templates.append(
            {
                "name": tpl.findtext("name"),
                "parameters": tpl.findtext("parameter"),
                "declarations": decl.text if decl is not None else None,
                "declarations_file_line": decl.sourceline if decl is not None else None,
                "locations": locations,
                "edges": [_edge_dict(tpl, i, tr) for i, tr in enumerate(tpl.iterfind("transition"))],
            }
        )
    gdecl = root.find("declaration")
    system = root.find("system")
    embedded = [q.findtext("formula") for q in root.iterfind("queries/query")]
    return {
        "global_declarations": gdecl.text if gdecl is not None else None,
        "global_declarations_file_line": gdecl.sourceline if gdecl is not None else None,
        "templates": templates,
        "system": system.text if system is not None else None,
        "system_file_line": system.sourceline if system is not None else None,
        "embedded_queries": [q for q in embedded if q and q.strip()],
    }


# --------------------------------------------------------------------------- #
# Selectors
# --------------------------------------------------------------------------- #


def _template(root: etree._Element, name: Any) -> etree._Element:
    if not isinstance(name, str) or not name:
        raise ModelError("'template' (template name) is required")
    for tpl in root.iterfind("template"):
        if tpl.findtext("name") == name:
            return tpl
    names = [t.findtext("name") for t in root.iterfind("template")]
    raise ModelError(f"No template named {name!r}; templates are {names}")


def _location(tpl: etree._Element, ref: Any) -> etree._Element:
    if not isinstance(ref, str) or not ref:
        raise ModelError("location (name or id) is required")
    locs = list(tpl.iterfind("location"))
    by_name = [loc for loc in locs if _loc_name(loc) == ref]
    if len(by_name) == 1:
        return by_name[0]
    by_id = [loc for loc in locs if loc.get("id") == ref]
    if by_id:
        return by_id[0]
    raise ModelError(
        f"No location {ref!r} in template {tpl.findtext('name')}; locations are {[_loc_name(loc) for loc in locs]}"
    )


def _edge(tpl: etree._Element, op: dict[str, Any]) -> tuple[int, etree._Element]:
    """Select one edge by ``edge`` (index from read_model), ``edge_id``, or
    ``from``/``to`` location names (must be unique)."""
    edges = list(tpl.iterfind("transition"))
    if "edge" in op:
        i = op["edge"]
        if not isinstance(i, int) or not (0 <= i < len(edges)):
            raise ModelError(f"edge index {i!r} out of range 0..{len(edges) - 1} in template {tpl.findtext('name')}")
        return i, edges[i]
    if "edge_id" in op:
        for i, e in enumerate(edges):
            if e.get("id") == op["edge_id"]:
                return i, e
        raise ModelError(f"No edge with id {op['edge_id']!r} in template {tpl.findtext('name')}")
    if "from" in op and "to" in op:
        src = _location(tpl, op["from"]).get("id")
        dst = _location(tpl, op["to"]).get("id")
        hits = [
            (i, e)
            for i, e in enumerate(edges)
            if e.find("source") is not None
            and e.find("source").get("ref") == src
            and e.find("target") is not None
            and e.find("target").get("ref") == dst
        ]
        if len(hits) == 1:
            return hits[0]
        if not hits:
            raise ModelError(f"No edge {op['from']} -> {op['to']} in template {tpl.findtext('name')}")
        raise ModelError(
            f"{len(hits)} edges {op['from']} -> {op['to']} in template {tpl.findtext('name')} "
            f"(indices {[i for i, _ in hits]}); select one with 'edge': <index>"
        )
    raise ModelError("Select the edge with 'edge' (index), 'edge_id', or 'from' and 'to'")


# --------------------------------------------------------------------------- #
# Whitespace-preserving tree edits
# --------------------------------------------------------------------------- #


def _indent_of(parent: etree._Element) -> tuple[str, str]:
    """(separator before a child, separator before the parent's closing tag)."""
    kids = list(parent)
    if len(kids) >= 2:
        return kids[0].tail or "\n", kids[-1].tail or "\n"
    if kids:
        end = kids[0].tail or "\n"
        return parent.text or end + "\t", end
    end = parent.tail or "\n"
    return end + "\t", end


def _insert(parent: etree._Element, el: etree._Element, index: int) -> None:
    sep, end = _indent_of(parent)
    kids = list(parent)
    if not kids:
        parent.text = sep
        el.tail = end
        parent.insert(0, el)
        return
    if index >= len(kids):
        kids[-1].tail = sep
        el.tail = end
    else:
        el.tail = sep
    parent.insert(index, el)


def _remove(el: etree._Element) -> None:
    parent = el.getparent()
    prev = el.getprevious()
    if prev is not None and el.getnext() is None:
        prev.tail = el.tail
    elif prev is None and el.getnext() is None:
        parent.text = None
    parent.remove(el)


def _child_indent(parent: etree._Element) -> str:
    sep, _ = _indent_of(parent)
    return sep + ("\t" if "\t" in sep else "  ")


def _new_element(parent: etree._Element, tag: str, attrib: dict[str, str], children: list[etree._Element]) -> etree._Element:
    el = etree.Element(tag, attrib)
    if children:
        inner = _child_indent(parent)
        el.text = inner
        for i, c in enumerate(children):
            c.tail = inner if i < len(children) - 1 else _indent_of(parent)[0]
            el.append(c)
    return el


def _text_el(tag: str, text: str, **attrib: str) -> etree._Element:
    el = etree.Element(tag, attrib)
    el.text = text
    return el


def _set_edge_label(tr: etree._Element, kind: str, value: str) -> str | None:
    """Set (or with ``value == ""`` remove) a transition label; returns the old value."""
    old_el = next((lab for lab in tr.iterfind("label") if lab.get("kind") == kind), None)
    old = (old_el.text or "") if old_el is not None else None
    if value.strip() == "":
        if old_el is not None:
            _remove(old_el)
        return old
    if old_el is not None:
        old_el.text = value
        return old
    # Insert in UPPAAL's canonical order, after source/target and earlier labels.
    rank = _EDGE_LABELS.index(kind) if kind in _EDGE_LABELS else len(_EDGE_LABELS)
    pos = 0
    for i, c in enumerate(tr):
        if c.tag in ("source", "target"):
            pos = i + 1
        elif c.tag == "label" and (_EDGE_LABELS.index(c.get("kind")) if c.get("kind") in _EDGE_LABELS else 99) < rank:
            pos = i + 1
    _insert(tr, _text_el("label", value, kind=kind), pos)
    return old


def _set_invariant(loc: etree._Element, value: str) -> str | None:
    old_el = next((lab for lab in loc.iterfind("label") if lab.get("kind") == "invariant"), None)
    old = (old_el.text or "") if old_el is not None else None
    if value.strip() == "":
        if old_el is not None:
            _remove(old_el)
        return old
    if old_el is not None:
        old_el.text = value
        return old
    pos = 1 if loc.find("name") is not None else 0
    _insert(loc, _text_el("label", value, kind="invariant"), pos)
    return old


def _next_id(root: etree._Element) -> str:
    nums = [int(m.group(1)) for el in root.iter() if isinstance(el.tag, str) for m in [re.fullmatch(r"id(\d+)", el.get("id") or "")] if m]
    return f"id{max(nums, default=-1) + 1}"


def _declaration_el(root: etree._Element, scope: Any, create: bool) -> etree._Element:
    if scope == "global":
        el = root.find("declaration")
        if el is None and create:
            el = _text_el("declaration", "")
            _insert(root, el, 0)
        if el is None:
            raise ModelError("The model has no global declarations")
        return el
    if scope == "system":
        el = root.find("system")
        if el is None:
            raise ModelError("The model has no system declaration")
        return el
    tpl = _template(root, scope)
    el = tpl.find("declaration")
    if el is None and create:
        el = _text_el("declaration", "")
        pos = 1 + (tpl.find("parameter") is not None)
        _insert(tpl, el, pos)
    if el is None:
        raise ModelError(f"Template {scope} has no local declarations")
    return el


# --------------------------------------------------------------------------- #
# Operations
# --------------------------------------------------------------------------- #

_FALSE = {"false", "0", "!true", "nottrue", "1==0", "0==1", "!1"}


def _is_false(expr: str) -> bool:
    s = re.sub(r"\s+", "", expr).lower()
    while s.startswith("(") and s.endswith(")"):
        s = s[1:-1]
    return s in _FALSE


@dataclass
class Change:
    op_index: int
    op: str
    description: str
    flags: list[dict[str, str]] = field(default_factory=list)


def _flag(severity: str, message: str) -> dict[str, str]:
    return {"severity": severity, "message": message}


def _q(v: str | None) -> str:
    return "(none)" if v is None or v == "" else repr(v)


def _str_arg(op: dict[str, Any], key: str, *, required: bool = True, default: str | None = None) -> str:
    v = op.get(key, default)
    if v is None:
        if required:
            raise ModelError(f"'{key}' is required for op {op.get('op')!r}")
        return ""
    if not isinstance(v, str):
        raise ModelError(f"'{key}' must be a string")
    return v


def apply_operation(root: etree._Element, i: int, op: dict[str, Any]) -> Change:
    kind = op.get("op")
    if kind in ("set_declaration", "replace_in_declaration"):
        scope = op.get("scope", "global")
        el = _declaration_el(root, scope, create=kind == "set_declaration")
        where = "system declaration" if scope == "system" else ("global declarations" if scope == "global" else f"declarations of {scope}")
        old_text = el.text or ""
        if kind == "set_declaration":
            el.text = _str_arg(op, "text")
            desc = f"replaced {where}"
        else:
            old, new = _str_arg(op, "old"), _str_arg(op, "new", default="")
            n = old_text.count(old) if old else 0
            if n != 1:
                raise ModelError(f"'old' must occur exactly once in {where}; found {n} occurrences of {old!r}")
            el.text = old_text.replace(old, new)
            desc = f"{where}: {old!r} -> {new!r}"
        flags = [] if el.text == old_text else [_flag("may_change_behaviour", f"Edited {where}; review the diff for changed constants, ranges or functions.")]
        return Change(i, kind, desc, flags)

    if kind in ("set_guard", "set_sync", "set_update", "set_select"):
        tpl = _template(root, op.get("template"))
        _, tr = _edge(tpl, op)
        fieldname = kind.removeprefix("set_")
        value = _str_arg(op, "value")
        desc_edge = describe_element(tr)
        old = _set_edge_label(tr, _EDGE_FIELDS[fieldname], value)
        desc = f"{fieldname} of {desc_edge}: {_q(old)} -> {_q(value)}"
        flags: list[dict[str, str]] = []
        if fieldname == "guard" and _is_false(value):
            flags.append(_flag("removes_behaviour", f"Guard of {desc_edge} set to false: the edge can never be taken."))
        elif fieldname == "guard" and (old or "") != value:
            flags.append(_flag("may_change_behaviour", f"Guard of {desc_edge} changed; a stronger guard disables executions."))
        elif fieldname == "sync" and (old or "") != value:
            flags.append(_flag("may_change_behaviour", f"Synchronisation of {desc_edge} changed."))
        return Change(i, kind, desc, flags)

    if kind == "set_invariant":
        tpl = _template(root, op.get("template"))
        loc = _location(tpl, op.get("location"))
        value = _str_arg(op, "value")
        old = _set_invariant(loc, value)
        where = describe_element(loc)
        flags = []
        if _is_false(value):
            flags.append(_flag("removes_behaviour", f"Invariant of {where} set to false: the location becomes unreachable."))
        elif (old or "") != value:
            flags.append(_flag("may_change_behaviour", f"Invariant of {where} changed; a stronger invariant forces earlier exits or blocks entry."))
        return Change(i, kind, f"invariant of {where}: {_q(old)} -> {_q(value)}", flags)

    if kind == "add_edge":
        tpl = _template(root, op.get("template"))
        src = _location(tpl, op.get("from"))
        dst = _location(tpl, op.get("to"))
        children = [etree.Element("source", ref=src.get("id")), etree.Element("target", ref=dst.get("id"))]
        for f in ("select", "guard", "sync", "update"):
            v = _str_arg(op, f, required=False)
            if v.strip():
                children.append(_text_el("label", v, kind=_EDGE_FIELDS[f]))
        tr = _new_element(tpl, "transition", {"id": _next_id(root)}, children)
        _insert(tpl, tr, len(tpl))
        return Change(i, kind, f"added {describe_element(tr)}" + "".join(
            f", {f} {op[f]!r}" for f in ("select", "guard", "sync", "update") if op.get(f)
        ))

    if kind == "remove_edge":
        tpl = _template(root, op.get("template"))
        _, tr = _edge(tpl, op)
        desc = describe_element(tr)
        labels = {k: _label(tr, v) for k, v in _EDGE_FIELDS.items()}
        _remove(tr)
        detail = ", ".join(f"{k} {v!r}" for k, v in labels.items() if v)
        return Change(i, kind, f"removed {desc}" + (f" ({detail})" if detail else ""), [_flag("removes_behaviour", f"Removed {desc}.")])

    if kind == "add_location":
        tpl = _template(root, op.get("template"))
        name = _str_arg(op, "name")
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            raise ModelError(f"Invalid location name {name!r}")
        if any(_loc_name(loc) == name for loc in tpl.iterfind("location")):
            raise ModelError(f"Template {tpl.findtext('name')} already has a location {name!r}")
        xs = [int(loc.get("x")) for loc in tpl.iterfind("location") if (loc.get("x") or "").lstrip("-").isdigit()]
        children = [_text_el("name", name)]
        inv = _str_arg(op, "invariant", required=False)
        if inv.strip():
            children.append(_text_el("label", inv, kind="invariant"))
        lkind = op.get("kind", "normal")
        if lkind in ("urgent", "committed"):
            children.append(etree.Element(lkind))
        elif lkind != "normal":
            raise ModelError("location 'kind' must be 'normal', 'urgent' or 'committed'")
        loc = _new_element(tpl, "location", {"id": _next_id(root), "x": str(max(xs, default=0) + 150), "y": "0"}, children)
        locs = list(tpl.iterfind("location"))
        _insert(tpl, loc, tpl.index(locs[-1]) + 1 if locs else len(tpl))
        flags = [_flag("may_change_behaviour", f"Committed location {name} restricts interleaving.")] if lkind == "committed" else []
        return Change(i, kind, f"added {lkind} location {tpl.findtext('name')}.{name}" + (f" with invariant {inv!r}" if inv else ""), flags)

    if kind == "remove_location":
        tpl = _template(root, op.get("template"))
        loc = _location(tpl, op.get("location"))
        lid = loc.get("id")
        init = tpl.find("init")
        if init is not None and init.get("ref") == lid:
            raise ModelError(f"Cannot remove the initial location {_loc_name(loc)}")
        where = describe_element(loc)
        incident = [
            tr
            for tr in tpl.iterfind("transition")
            if (tr.find("source") is not None and tr.find("source").get("ref") == lid)
            or (tr.find("target") is not None and tr.find("target").get("ref") == lid)
        ]
        flags = [_flag("removes_behaviour", f"Removed {where}.")]
        for tr in incident:
            flags.append(_flag("removes_behaviour", f"Removed incident {describe_element(tr)}."))
            _remove(tr)
        _remove(loc)
        return Change(i, kind, f"removed {where} and {len(incident)} incident edge(s)", flags)

    raise ModelError(
        f"Unknown op {kind!r}; supported: set_declaration, replace_in_declaration, set_guard, set_sync, "
        "set_update, set_select, set_invariant, add_edge, remove_edge, add_location, remove_location"
    )


def apply_operations(doc: Document, operations: list[dict[str, Any]]) -> list[Change]:
    """Apply all operations in order; raises ``ModelError`` (naming the failing
    operation) and leaves ``doc`` partially modified - callers discard it."""
    if not isinstance(operations, list) or not operations:
        raise ModelError("'operations' must be a non-empty list")
    changes = []
    for i, op in enumerate(operations):
        if not isinstance(op, dict):
            raise ModelError(f"Operation {i} must be an object with an 'op' field")
        try:
            changes.append(apply_operation(doc.root, i, op))
        except ModelError as e:
            raise ModelError(f"Operation {i} ({op.get('op')}): {e}") from None
    return changes


# --------------------------------------------------------------------------- #
# Diagnostics
# --------------------------------------------------------------------------- #


def describe_element(el: etree._Element) -> str:
    """Human description of a model element, e.g. ``guard of edge Train: Near -> Cross``."""
    tree = el.getroottree()

    def loc_name(template: etree._Element, ref: str | None) -> str:
        for loc in template.iterfind("location"):
            if loc.get("id") == ref:
                return _loc_name(loc)
        return ref or "?"

    if el.tag == "label":
        parent = el.getparent()
        kind = el.get("kind", "label")
        return f"{kind} of {describe_element(parent)}" if parent is not None else kind
    if el.tag == "transition":
        tpl = el.getparent()
        src = el.find("source")
        dst = el.find("target")
        return (
            f"edge {tpl.findtext('name')}: "
            f"{loc_name(tpl, src.get('ref') if src is not None else None)} -> "
            f"{loc_name(tpl, dst.get('ref') if dst is not None else None)}"
        )
    if el.tag == "location":
        return f"location {el.getparent().findtext('name')}.{_loc_name(el)}"
    if el.tag == "declaration":
        parent = el.getparent()
        return "global declarations" if parent.tag == "nta" else f"declarations of template {parent.findtext('name')}"
    if el.tag == "parameter":
        return f"parameters of template {el.getparent().findtext('name')}"
    if el.tag == "system":
        return "system declaration"
    if el.tag == "template":
        return f"template {el.findtext('name')}"
    return tree.getpath(el)


def locate_diagnostics(model_path: Path, diags: list[Diagnostic]) -> list[dict[str, Any]]:
    """Turn verifyta diagnostics into dicts with file line numbers.

    verifyta reports ``<xpath>:<line-within-element-text>``; we resolve the XPath
    in the model and add the element's source line, so ``file_line`` points at
    the offending line of ``model_path``.
    """
    try:
        tree = load(model_path)
    except etree.XMLSyntaxError as e:
        tree = None
        xml_error = str(e)
    out: list[dict[str, Any]] = []
    for d in diags:
        item: dict[str, Any] = {"severity": d.severity, "message": d.message, "source": d.source}
        if d.source == "model":
            item["xpath"] = d.xpath
            item["line_in_element"] = d.line
            if tree is not None and d.xpath:
                try:
                    hits = tree.xpath(d.xpath)
                except etree.XPathError:
                    hits = []
                if hits and isinstance(hits[0], etree._Element):
                    el = hits[0]
                    item["element"] = describe_element(el)
                    lines = (el.text or "").splitlines()
                    if d.line is not None and 0 < d.line <= len(lines):
                        item["text"] = lines[d.line - 1].strip()
                    if el.sourceline is not None and d.line is not None:
                        item["file_line"] = el.sourceline + d.line - 1
            elif tree is None:
                item["xml_error"] = xml_error
        else:
            item["query_file_line"] = d.line
        out.append(item)
    return out
