"""Simulator-independent evaluation limits, mirrored from UniVTAC1 TaskCfg."""
import json
from pathlib import Path

SETTINGS = json.loads(Path(__file__).with_name('eval_settings.json').read_text())


def task_overrides(task):
    return [f"task_overrides.step_lim={SETTINGS['task_step_limits'][task]}",
            f"task_overrides.reset_time_limit={SETTINGS['reset_time_limit_seconds']}"]


if __name__ == '__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('task',choices=SETTINGS['task_step_limits'])
    args=parser.parse_args()
    print('\n'.join(task_overrides(args.task)))
