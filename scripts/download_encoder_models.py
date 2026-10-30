"""Download official smallest encoder checkpoints; persist success and access errors."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from encoder.pretrained_registry import ROOT, SOURCES, source_dir
from encoder.catalog import canonical_name


def main():
    from huggingface_hub import HfApi, snapshot_download
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--methods', nargs='+')
    p.add_argument('--workers', type=int, default=3)
    args = p.parse_args()
    report = ROOT / 'resources/pretrained_models/encoder_downloads.json'
    records = json.loads(report.read_text()) if report.exists() else {}

    def download(item):
        (mode, method), (size, repo, patterns) = item
        target = source_dir(mode, method)
        target.mkdir(parents=True, exist_ok=True)
        record = dict(mode=mode, method=method, size=size, repo=repo, directory=str(target))
        try:
            revision = HfApi().model_info(repo).sha
            snapshot_download(repo, revision=revision, local_dir=target, allow_patterns=patterns, max_workers=2)
            record.update(status='downloaded', revision=revision)
            (target / 'source.json').write_text(json.dumps(record, indent=2))
        except Exception as exc:
            record.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        return f'{mode}/{method}', record

    items = [item for item in SOURCES.items() if not args.methods or item[0][1] in args.methods or canonical_name(*item[0]) in args.methods]
    if any(mode == 'tactile' and method != 'AnyTouch2' for (mode, method), _ in items):
        from scripts.setup_encoder_backends import ensure_sparsh
        ensure_sparsh()
    with ThreadPoolExecutor(args.workers) as pool:
        for future in as_completed([pool.submit(download, item) for item in items]):
            key, record = future.result()
            records[key] = record
            report.write_text(json.dumps(records, indent=2))
            print(key, record['status'], record.get('error', '')[:300], flush=True)
    if any(records[f'{mode}/{method}']['status'] != 'downloaded' for (mode, method), _ in items):
        sys.exit(1)


if __name__ == '__main__':
    main()
