"""Check progress survives interruption without importing Isaac Sim."""
import ast
import json
from pathlib import Path
import tempfile
import time
import traceback
from types import SimpleNamespace
import unittest

from scripts.eval_results import write_result


class EvaluationProgressTest(unittest.TestCase):
    def test_completed_trials_saved_before_next_reset(self):
        source = Path(__file__).with_name('eval_policy.py')
        tree = ast.parse(source.read_text())
        function = next(node for node in tree.body
                        if isinstance(node, ast.FunctionDef) and node.name == 'eval_policy')
        # Execute the real evaluation loop with a fake simulator. SystemExit
        # models an interruption that bypasses its Python Exception handler.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / 'result.json'
            context = dict(time=time, json=json, traceback=traceback,
                           Literal=__import__('typing').Literal, write_result=write_result,
                           args_cli=SimpleNamespace(result_json=output), log=lambda message: None)
            exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), 'exec'), context)

            class Task:
                save_root = root / 'videos'
                cfg = SimpleNamespace(step_lim=1)
                step_count = 1
                take_action_cnt = 0
                eval_success = False
                instruction = 'test'

                def reset(self, seed, instructions):
                    if seed == 1000001:
                        raise SystemExit('simulator interrupted')

                def _get_observations(self):
                    return {}

                def clean_cache(self, result):
                    pass

            def predict(task, observation):
                task.eval_success = True

            policy = SimpleNamespace(reset=lambda: None, eval=predict)
            with self.assertRaises(SystemExit):
                context['eval_policy'](Task(), policy, False, 1000000, -1, 20, {'seen': ['test']})
            self.assertEqual(json.loads(output.read_text()), {
                'test_num': 1, 'succ_num': 1, 'error_num': 0, 'next_seed': 1000001})


if __name__ == '__main__':
    unittest.main()
