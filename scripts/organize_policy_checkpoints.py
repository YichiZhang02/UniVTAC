"""Move policy runs to canonical physical directories and index real checkpoints."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from policy.checkpoints import load_run


def organize(root, apply=False):
    root = Path(root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    sources = []
    plans = []
    seen = set()
    destinations = set()
    for config in sorted(root.rglob("train_config.yml")):
        folder = config.parent.resolve()
        if folder in seen:
            continue
        seen.add(folder)
        if not folder.is_relative_to(root):
            raise ValueError(f"Run is outside the results root: {folder}")
        checkpoints = list(folder.glob("checkpoint_*.pt"))
        if not checkpoints:
            continue
        if any(path.is_symlink() for path in checkpoints):
            raise ValueError(f"Step checkpoints must be regular files: {folder}")
        run = load_run(folder)
        timestamp = None
        for parent in (folder, folder.parent):
            match = re.search(r"(\d{8}-\d{6}(?:-\d+)?)", parent.name)
            if match:
                timestamp = match.group(1)
                break
        timestamp = timestamp or run["config"].get("run_id")
        if timestamp is None:
            raise ValueError(f"Cannot identify a timestamp for {folder}")
        destination = root / f"{timestamp}_{run['label']}"
        if destination in destinations:
            raise FileExistsError(f"Duplicate canonical name: {destination}")
        destinations.add(destination)
        if destination.is_symlink():
            if destination.resolve() != folder:
                raise FileExistsError(f"Canonical link points to another run: {destination}")
        elif destination.exists() and destination != folder:
            raise FileExistsError(f"Canonical directory already exists: {destination}")
        if folder.stat().st_dev != root.stat().st_dev:
            raise ValueError(f"Cannot rename across filesystems without copying: {folder}")
        sources.append(folder)
        plans.append({"name": destination.name, "directory": str(destination),
                      "checkpoint": str(destination / run["checkpoint"].name),
                      "train_log": str(destination / "train.log"),
                      "exit_code": str(destination / "exit_code"), "step": run["step"]})
    # Validate all destinations before renaming any files. Rename keeps the large
    # checkpoint payloads intact and uses no additional checkpoint storage.
    if apply:
        completed = []
        try:
            for folder, plan in zip(sources, plans):
                destination = Path(plan["directory"])
                old_link = destination.readlink() if destination.is_symlink() else None
                if old_link is not None:
                    destination.unlink()
                try:
                    if folder != destination:
                        folder.rename(destination)
                except OSError:
                    if old_link is not None:
                        destination.symlink_to(old_link, target_is_directory=True)
                    raise
                completed.append((folder, destination, old_link))
        except OSError:
            for folder, destination, old_link in reversed(completed):
                if folder != destination:
                    destination.rename(folder)
                if old_link is not None:
                    destination.symlink_to(old_link, target_is_directory=True)
            raise
        for plan in plans:
            folder = Path(plan["directory"])
            # Retire former convenience pointers, retaining any regular file.
            for name in ("last.pt", ".last.pt.tmp"):
                pointer = folder / name
                if pointer.is_symlink():
                    pointer.unlink()
        index = root / "checkpoints_index.json"
        temporary = root / ".checkpoints_index.json.tmp"
        temporary.write_text(json.dumps(plans, indent=2) + "\n")
        temporary.replace(index)
        for parent in {folder.parent for folder in sources}:
            if parent != root and parent.is_dir() and not any(parent.iterdir()):
                parent.rmdir()
    return plans


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "policy_results")
    parser.add_argument("--apply", action="store_true", help="Rename run directories, remove legacy links, and update the index")
    args = parser.parse_args()
    plans = organize(args.root, args.apply)
    print(json.dumps({"mode": "apply" if args.apply else "dry-run", "count": len(plans), "runs": plans}, indent=2))


if __name__ == "__main__":
    main()
