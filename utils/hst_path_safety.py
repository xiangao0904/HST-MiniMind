from __future__ import annotations

import os
from pathlib import Path


class PathSafetyError(ValueError):
    """Raised when an output path escapes the project boundary."""


def get_project_root() -> Path:
    root = Path(os.environ.get("PROJECT_ROOT", Path.cwd())).expanduser().resolve()
    if str(root) == "/":
        raise PathSafetyError("PROJECT_ROOT must not resolve to /")
    return root


def is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def ensure_within_project(path: str | Path, project_root: str | Path | None = None) -> Path:
    root = Path(project_root).expanduser().resolve() if project_root else get_project_root()
    target = Path(path).expanduser()
    if not target.is_absolute():
        target = root / target
    target = target.resolve()
    if not is_relative_to(target, root):
        raise PathSafetyError(f"path escapes project root: {target} not under {root}")
    return target


def ensure_run_output_dir(path: str | Path, project_root: str | Path | None = None) -> Path:
    root = Path(project_root).expanduser().resolve() if project_root else get_project_root()
    target = ensure_within_project(path, root)
    run_root = (root / "hst_runs").resolve()
    if not is_relative_to(target, run_root):
        raise PathSafetyError(f"output_dir must be under {run_root}, got {target}")
    return target


def safe_mkdir(path: str | Path, project_root: str | Path | None = None) -> Path:
    target = ensure_within_project(path, project_root)
    target.mkdir(parents=True, exist_ok=True)
    return target
