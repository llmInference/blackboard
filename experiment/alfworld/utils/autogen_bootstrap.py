"""Bootstrap helpers for importing local AutoGen sources without installation."""
from __future__ import annotations

import importlib.metadata
import sys
from pathlib import Path


def bootstrap_local_autogen() -> None:
    """Ensure local AutoGen source paths and package metadata fallbacks are available."""
    root = Path(__file__).resolve().parents[3]
    autogen_root = root / "autogen" / "python" / "packages"
    for relative in (
        "autogen-core/src",
        "autogen-agentchat/src",
        "autogen-ext/src",
        "pyautogen/src",
    ):
        path = autogen_root / relative
        if path.exists() and str(path) not in sys.path:
            sys.path.insert(0, str(path))
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    if getattr(importlib.metadata, "_alfworld_local_autogen_patch", False):
        return

    real_version = importlib.metadata.version
    local_packages = {"autogen_agentchat", "autogen_core", "autogen_ext", "pyautogen"}

    def _version(name: str) -> str:
        try:
            return real_version(name)
        except importlib.metadata.PackageNotFoundError:
            if name in local_packages:
                return "0.0.0+local"
            raise

    importlib.metadata.version = _version  # type: ignore[assignment]
    setattr(importlib.metadata, "_alfworld_local_autogen_patch", True)
