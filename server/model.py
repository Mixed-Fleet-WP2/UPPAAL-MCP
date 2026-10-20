"""UPPAAL .xml model access via lxml."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from lxml import etree

from .verifyta import Diagnostic

_PARSER = etree.XMLParser(remove_blank_text=False, resolve_entities=False, no_network=True)


def load(path: Path) -> etree._ElementTree:
    return etree.parse(str(path), _PARSER)


def describe_element(el: etree._Element) -> str:
    """Human description of a model element, e.g. ``guard of edge Train: Near -> Cross``."""
    tree = el.getroottree()

    def loc_name(template: etree._Element, ref: str | None) -> str:
        for loc in template.iterfind("location"):
            if loc.get("id") == ref:
                return loc.findtext("name") or ref or "?"
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
        return f"location {el.getparent().findtext('name')}.{el.findtext('name') or el.get('id')}"
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
