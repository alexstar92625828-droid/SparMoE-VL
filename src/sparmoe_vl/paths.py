"""Portable repository and workspace path discovery.

The experiment entry points are designed to run from a source checkout, an
editable installation, or a regular package installation.  Environment
variables provide explicit locations for cluster jobs and non-standard data
layouts; otherwise the nearest SparMoE-VL checkout is discovered from the
current directory or this module.
"""

from __future__ import annotations

import os
from pathlib import Path


REPOSITORY_ROOT_ENV = "SPARMOE_VL_ROOT"
WORKSPACE_ROOT_ENV = "SPARMOE_VL_WORKSPACE"


def _expanded_path(value: str) -> Path:
    return Path(value).expanduser().resolve()


def _repository_candidate(path: Path) -> Path | None:
    start = path if path.is_dir() else path.parent
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").is_file() and (
            candidate / "src" / "sparmoe_vl"
        ).is_dir():
            return candidate
    return None


def repository_root() -> Path:
    """Return the active source checkout used for configs, weights, and outputs."""

    configured = os.environ.get(REPOSITORY_ROOT_ENV)
    if configured:
        return _expanded_path(configured)

    for start in (Path.cwd().resolve(), Path(__file__).resolve()):
        candidate = _repository_candidate(start)
        if candidate is not None:
            return candidate

    # A wheel can be imported outside a source checkout.  In that case the
    # current directory is the least surprising writable default; users can
    # set SPARMOE_VL_ROOT or pass explicit CLI paths.
    return Path.cwd().resolve()


def workspace_root() -> Path:
    """Return the directory containing external models and datasets."""

    configured = os.environ.get(WORKSPACE_ROOT_ENV)
    if configured:
        return _expanded_path(configured)
    return repository_root().parent
