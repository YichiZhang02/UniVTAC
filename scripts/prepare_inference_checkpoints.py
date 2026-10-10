"""Bundle tokenizer/processor assets into existing unified policy runs, without rewriting .pt files."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from policy.checkpoints import load_run
from policy.inference_assets import asset_path, export_assets


def discover(root):
    if (root / "train_config.yml").is_file():
        configs = [root / "train_config.yml"]
    else:
        configs = sorted(path for path in root.rglob("train_config.yml")
                         if not any(part.startswith(".")
                                    for part in path.relative_to(root).parts[:-1]))
    return [load_run(path.parent) for path in configs
            if any(path.parent.glob("checkpoint_*.pt"))]


def validate_assets(run):
    from transformers import AutoConfig, AutoProcessor, AutoTokenizer
    if run["policy"] == "starvla_groot":
        path = asset_path(run, "qwen_path")
        AutoConfig.from_pretrained(path, local_files_only=True)
        AutoProcessor.from_pretrained(path, local_files_only=True)
    elif run["policy"] == "pi05":
        AutoTokenizer.from_pretrained(asset_path(run, "tokenizer_path"), local_files_only=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "policy_results",
                        help="One run directory or a root containing runs")
    parser.add_argument("--models-root", type=Path, default=ROOT / "resources/pretrained_models")
    parser.add_argument("--apply", action="store_true", help="Export resources and update train_config.yml")
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    if not root.is_dir():
        parser.error(f"Missing results root: {root}")
    runs = [run for run in discover(root) if run["policy"] in ("starvla_groot", "pi05")]
    # Check every destination before making changes; do not overwrite untracked assets.
    for run in runs:
        if run["config"].get("inference") is None and (run["directory"] / "inference_assets").exists():
            raise FileExistsError(run["directory"] / "inference_assets")
    plans = [{"directory": str(run["directory"]), "policy": run["policy"],
              "checkpoint_count": len(list(run["directory"].glob("checkpoint_*.pt"))),
              "status": "already_bundled" if run["config"].get("inference") else "prepare"}
             for run in runs]
    if args.apply:
        from transformers import AutoConfig, AutoProcessor, AutoTokenizer
        models = args.models_root.expanduser().resolve()
        resources = {}
        # Load metadata and processors only, never the original model weights.
        for policy in {run["policy"] for run in runs if not run["config"].get("inference")}:
            if policy == "starvla_groot":
                source = models / "Qwen3.5-2B"
                resources[policy] = dict(qwen=SimpleNamespace(
                    processor=AutoProcessor.from_pretrained(source, local_files_only=True),
                    model=SimpleNamespace(config=AutoConfig.from_pretrained(source, local_files_only=True))))
            else:
                resources[policy] = dict(tokenizer=AutoTokenizer.from_pretrained(
                    models / "pi05_base/paligemma-3b-pt-224-tokenizer", local_files_only=True))
        for run, plan in zip(runs, plans):
            if run["config"].get("inference"):
                validate_assets(run)
                continue
            folder = run["directory"]
            # Validate staged resources before publishing the directory or configuration.
            with tempfile.TemporaryDirectory(prefix=".inference-", dir=folder) as staging:
                metadata = export_assets(staging, run["policy"], **resources[run["policy"]])
                updated = {**run["config"], "inference": metadata}
                validate_assets({**run, "directory": Path(staging), "config": updated})
                config_file = folder / "train_config.yml"
                backup = folder / "train_config.yml.pre-inference.bak"
                if not backup.exists():
                    backup.write_bytes(config_file.read_bytes())
                temporary = Path(staging) / "train_config.yml"
                temporary.write_text(yaml.safe_dump(updated, sort_keys=False))
                (Path(staging) / "inference_assets").rename(folder / "inference_assets")
                temporary.replace(config_file)
            plan["status"] = "prepared"
    print(json.dumps({"mode": "apply" if args.apply else "dry-run", "count": len(plans),
                      "runs": plans}, indent=2))


if __name__ == "__main__":
    main()
