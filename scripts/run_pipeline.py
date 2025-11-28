"""
Compatibility shim for older workflows.

This file used to be `scripts/run_pipeline.py`. The script has been
renamed to `scripts/prepare_datasets.py`. This module is kept as a thin
shim for backwards compatibility: it will import and call the new
`prepare_datasets.main()` so existing scripts/automation that still run
`run_pipeline.py` will continue to work.
"""

from __future__ import annotations

from importlib import import_module
import sys


def main() -> None:
    # Import the new script module and delegate to its main function.
    mod = import_module("scripts.prepare_datasets")
    if hasattr(mod, "main"):
        mod.main()
    else:
        # Fallback: try module in scripts package root
        mod = import_module("prepare_datasets")
        mod.main()


if __name__ == "__main__":
    # Ensure the repository root is on sys.path so the import works when
    # running the script directly (python scripts/run_pipeline.py)
    from pathlib import Path

    repo_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo_root / "scripts"))
    sys.path.insert(0, str(repo_root))

    main()
