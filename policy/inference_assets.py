"""Export local inference resources shared by every checkpoint in a run."""
from pathlib import Path


def export_assets(directory, policy, *, qwen=None, tokenizer=None):
    """Export HF objects; deliberately exclude backbone weights and optimizer state."""
    directory = Path(directory)
    if policy == "starvla_groot":
        relative = "inference_assets/qwen"
        target = directory / relative
        target.mkdir(parents=True, exist_ok=True)
        qwen.processor.save_pretrained(target)
        qwen.model.config.save_pretrained(target)
        return {"format_version": 1, "qwen_path": relative}
    if policy == "pi05":
        relative = "inference_assets/tokenizer"
        target = directory / relative
        target.mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(target)
        return {"format_version": 1, "tokenizer_path": relative}
    return None


def asset_path(run, key):
    """Resolve a portable run-relative resource; never fall back for broken bundles."""
    metadata = run["config"].get("inference")
    if metadata is None:
        return None  # Backward compatibility for unconverted runs.
    if metadata.get("format_version") != 1:
        raise ValueError("Unsupported inference resource format")
    relative = Path(metadata[key])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Inference resource must be run-relative: {relative}")
    target = run["directory"] / relative
    if not target.is_dir():
        raise FileNotFoundError(f"Missing bundled inference resource: {target}")
    return target
