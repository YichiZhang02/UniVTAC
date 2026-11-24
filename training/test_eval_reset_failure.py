"""Exercise the actual rollout function without importing Isaac Sim."""
import ast,json,tempfile,time,traceback,types,unittest
from pathlib import Path
from typing import Literal
from training import eval_results
class ResetFailureTest(unittest.TestCase):
    def test_bad_attachment_is_recorded_once_then_simulator_restart_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            function=next(n for n in ast.parse(Path('scripts/eval_policy.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='eval_policy')
            file=Path(tmp)/'result.json'
            ns=dict(Literal=Literal,args_cli=types.SimpleNamespace(task_name='insert_tube',result_file=str(file)),eval_results=eval_results,time=time,json=json,traceback=traceback,log=lambda *a:None)
            exec(compile(ast.fix_missing_locations(ast.Module(body=[function],type_ignores=[])),'scripts/eval_policy.py','exec'),ns)
            calls=[]
            class Task:
                cfg=types.SimpleNamespace(step_lim=300);save_root=Path(tmp)/'run'
                def reset(self,**kwargs):
                    calls.append(kwargs['seed']);raise RuntimeError('Timeout while waiting for tactile marker attachments to stabilize')
                def clean_cache(self,**kwargs):raise RuntimeError('scene cleanup also failed')
            with self.assertRaisesRegex(RuntimeError,'restart the simulator'):
                ns['eval_policy'](Task(),object(),False,1000000,-1,50,{'seen':['insert tube']})
            result=json.loads(file.read_text());self.assertEqual(result['valid'],0)
            self.assertEqual(len(result['rollouts']),1);self.assertEqual(result['rollouts'][0]['status'],'error')
            self.assertEqual(calls,[1000000])
if __name__=='__main__':unittest.main()
