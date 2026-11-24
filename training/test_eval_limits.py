"""Check task limits without importing Isaac Sim or starting a rollout."""
import ast
import unittest
from pathlib import Path

from training.eval_config import SETTINGS,task_overrides


def configured_limit(path):
    tree=ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node,ast.ClassDef) and node.name in ('TaskCfg','BaseTaskCfg'):
            for item in node.body:
                if isinstance(item,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='step_lim' for t in item.targets):
                    return ast.literal_eval(item.value)
    return None


class EvaluationLimitsTest(unittest.TestCase):
    def test_task_limits_and_reference(self):
        root=Path(__file__).resolve().parents[1]
        reference=Path('/mnt/yichi_data/UniVTAC1/resources/envs')
        for task,limit in SETTINGS['task_step_limits'].items():
            local=configured_limit(root/'envs'/f'{task}.py')
            if local is None:local=configured_limit(root/'envs/_base_task.py')
            self.assertEqual(local,limit,task)
            self.assertEqual(task_overrides(task),[f'task_overrides.step_lim={limit}',
                                                  'task_overrides.reset_time_limit=1200'])
            if reference.exists():
                expected=configured_limit(reference/f'{task}.py')
                if expected is None:expected=configured_limit(reference/'_base_task.py')
                self.assertEqual(limit,expected,task)
        self.assertEqual(SETTINGS['episodes_per_task'],50)


if __name__=='__main__':unittest.main()
