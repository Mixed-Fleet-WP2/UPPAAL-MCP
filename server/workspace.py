"""Workspace layout, path confinement, git metadata and the run log."""

from __future__ import annotations

import json
import os
import re
import subprocess
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class WorkspaceError(ValueError):
    """A request referred to a path or resource the server will not touch."""


@dataclass(frozen=True)
class Workspace:
    root: Path

    @classmethod
    def from_env(cls) -> "Workspace":
        root = os.environ.get("UPPAAL_WORKSPACE") or Path(__file__).resolve().parent.parent
        return cls(Path(root).resolve())

    @property
    def models_dir(self) -> Path:
        return self.root / "models"

    @property
    def queries_dir(self) -> Path:
        return self.root / "queries"

    @property
    def candidates_dir(self) -> Path:
        return self.queries_dir / "candidates"

    @property
    def traces_dir(self) -> Path:
        return self.root / "traces"

    @property
    def runs_log(self) -> Path:
        return self.root / "runs.jsonl"

    # ------------------------------------------------------------------ paths

    def resolve(self, path: str | Path, *, under: Path | None = None) -> Path:
        """Resolve ``path`` (relative to the workspace root) and require that it
        stays inside the workspace, or inside ``under`` if given. Symlinks are
        resolved first, so they cannot be used to escape."""
        if isinstance(path, str) and ("\x00" in path or not path.strip()):
            raise WorkspaceError(f"Invalid path: {path!r}")
        p = Path(path)
        p = (p if p.is_absolute() else self.root / p).resolve()
        base = (under or self.root).resolve()
        if not p.is_relative_to(base):
            raise WorkspaceError(f"Path {str(path)!r} resolves outside {self.rel(base) or 'the workspace'}/; refused.")
        return p

    def rel(self, path: Path) -> str:
        """Workspace-relative POSIX path for display."""
        try:
            return path.resolve().relative_to(self.root).as_posix()
        except ValueError:
            return str(path)

    def model_path(self, model: str) -> Path:
        """Accept ``train_gate``, ``train_gate.xml`` or ``models/train_gate.xml``."""
        name = model.strip()
        if "/" not in name:
            name = f"models/{name}"
        if not name.endswith(".xml"):
            name += ".xml"
        p = self.resolve(name, under=self.models_dir)
        if not p.is_file():
            raise WorkspaceError(f"Model not found: {self.rel(p)}. Use list_models to see available models.")
        return p

    def query_path(self, query_file: str) -> Path:
        """Resolve a ``.q`` file under ``queries/`` (confirmed or candidate requirements)."""
        name = query_file.strip()
        if "/" not in name:
            name = f"queries/{name}"
        if not name.endswith(".q"):
            raise WorkspaceError(f"Query files must have the .q extension: {query_file!r}")
        p = self.resolve(name, under=self.queries_dir)
        if not p.is_file():
            raise WorkspaceError(f"Query file not found: {self.rel(p)}")
        return p

    def writable_query_path(self, name: str) -> Path:
        """The ONLY way tools obtain a query file path to write. Confirmed requirements
        in ``queries/`` are read-only; writes are confined to ``queries/candidates/``."""
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]*\.q", name):
            raise WorkspaceError(
                f"Invalid candidate query file name {name!r}: use a plain file name like 'train_gate.q' (no directories)."
            )
        p = self.resolve(self.candidates_dir / name, under=self.candidates_dir)
        if p.exists() and not p.is_file():
            raise WorkspaceError(f"{self.rel(p)} is not a regular file")
        return p

    def confirmed_queries_for(self, model: Path) -> Path:
        """The confirmed requirements for ``models/<stem>.xml`` are ``queries/<stem>.q``."""
        return self.queries_dir / f"{model.stem}.q"

    # ------------------------------------------------------------------- git

    def git(self, *args: str, check: bool = False) -> subprocess.CompletedProcess[str]:
        return subprocess.run(["git", "-C", str(self.root), *args], capture_output=True, text=True, check=check)

    def git_info(self, path: Path) -> dict[str, Any]:
        """Last commit touching ``path``, its current blob hash and whether it has
        uncommitted changes. Values are ``None`` when git is unavailable."""
        rel = self.rel(path)
        info: dict[str, Any] = {"commit": None, "blob": None, "dirty": None}
        try:
            c = self.git("log", "-1", "--format=%H", "--", rel)
            info["commit"] = (c.stdout.strip() or None) if c.returncode == 0 else None
            b = self.git("hash-object", "--", rel)
            info["blob"] = (b.stdout.strip() or None) if b.returncode == 0 else None
            s = self.git("status", "--porcelain", "--", rel)
            info["dirty"] = bool(s.stdout.strip()) if s.returncode == 0 else None
        except FileNotFoundError:
            pass
        return info

    def is_git_repo(self) -> bool:
        try:
            return self.git("rev-parse", "--is-inside-work-tree").stdout.strip() == "true"
        except FileNotFoundError:
            return False

    def commit_file(self, path: Path, message: str) -> str:
        """Stage and commit only ``path``; returns the new commit hash. Uses the
        repository's git identity, falling back to ``uppaal-mcp`` if none is set."""
        rel = self.rel(path)
        ident: list[str] = []
        if not self.git("config", "user.email").stdout.strip():
            ident = ["-c", "user.name=uppaal-mcp", "-c", "user.email=uppaal-mcp@localhost"]
        self.git("add", "--", rel, check=True)
        r = subprocess.run(
            ["git", "-C", str(self.root), *ident, "commit", "--only", "-q", "-m", message, "--", rel],
            capture_output=True,
            text=True,
        )
        if r.returncode != 0:
            raise WorkspaceError(f"git commit failed: {(r.stderr or r.stdout).strip()}")
        return self.git("rev-parse", "HEAD", check=True).stdout.strip()

    # ------------------------------------------------------------------- runs

    @staticmethod
    def new_run_id() -> str:
        return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6]

    def run_dir(self, run_id: str) -> Path:
        if not run_id or "/" in run_id or run_id.startswith("."):
            raise WorkspaceError(f"Invalid run_id: {run_id!r}")
        return self.resolve(self.traces_dir / run_id, under=self.traces_dir)

    def append_runs(self, records: list[dict[str, Any]]) -> None:
        with self.runs_log.open("a", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, sort_keys=True) + "\n")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
