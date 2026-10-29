"""Execute an active notebook in a fresh kernel of this exact Python environment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile

from jupyter_client import AsyncKernelManager
from jupyter_client.kernelspec import KernelSpecManager
import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebook", type=Path)
    parser.add_argument("--output", type=Path, required=True,
                        help="Executed copy under data/notebooks; source is never overwritten")
    args = parser.parse_args()
    source = (ROOT / args.notebook).resolve()
    output = (ROOT / args.output).resolve()
    notebook_root = (ROOT / "notebooks").resolve()
    if not source.is_relative_to(notebook_root) or source.is_relative_to(notebook_root / "legacy"):
        parser.error("Choose an active notebook under notebooks/, excluding legacy/.")
    if not output.is_relative_to((ROOT / "data/notebooks").resolve()):
        parser.error("--output must be under data/notebooks/.")
    output.parent.mkdir(parents=True, exist_ok=True)
    notebook = nbformat.read(source, as_version=4)
    # An explicit temporary spec avoids accidentally executing a system Python.
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=".kernel-") as temporary:
        spec_root = Path(temporary)
        spec_dir = spec_root / "fremtpl2"
        spec_dir.mkdir()
        spec = {"argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
                "display_name": "freMTPL2 current interpreter", "language": "python"}
        (spec_dir / "kernel.json").write_text(json.dumps(spec), encoding="utf-8")
        manager = AsyncKernelManager(kernel_name="fremtpl2",
                                     connection_file=str(spec_root / "connection.json"),
                                     kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(spec_root)]))
        client = NotebookClient(notebook, km=manager, timeout=600,
                                startup_timeout=60,
                                resources={"metadata": {"path": str(ROOT)}}, allow_errors=False)
        client.execute(cleanup_kc=True)
        nbformat.write(notebook, output)
    print(f"Executed {source.name} with {sys.executable}: {output}")


if __name__ == "__main__":
    main()
