"""Run one resumable eight-task evaluation on ECS, with a GPU lease and status."""
import argparse,fcntl,json,os,signal,subprocess,time
from pathlib import Path
from training.eval_results import atomic
from training.eval_config import SETTINGS


def run_with_watchdog(argv, *, cwd, env, result_files, stall_timeout=1800,
                      poll_interval=30, stop_grace=10):
    """Reap the whole evaluation group if no rollout is saved for 30 minutes.

    The 1200-second reset limit remains unchanged. Logs are not heartbeats:
    a simulator can keep printing while making no valid rollout progress.
    """
    started=time.time()
    process=subprocess.Popen(argv,cwd=cwd,env=env,start_new_session=True)
    try:
        while True:
            try:return process.wait(timeout=poll_interval),None
            except subprocess.TimeoutExpired:pass
            progress=max([started]+[p.stat().st_mtime for p in result_files if p.exists()])
            if time.time()-progress>stall_timeout:
                returncode=124
                return returncode,'No saved rollout progress for '+str(stall_timeout)+' seconds'
    finally:
        # Include policy server and simulator descendants, even if their shell exited.
        try:os.killpg(process.pid,signal.SIGTERM)
        except ProcessLookupError:pass
        try:process.wait(timeout=stop_grace)
        except subprocess.TimeoutExpired:pass
        try:os.killpg(process.pid,signal.SIGKILL)
        except ProcessLookupError:pass
        process.wait()


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True)
    p.add_argument('--gpu',type=int,required=True);p.add_argument('--port',type=int,required=True)
    p.add_argument('--episodes',type=int,default=50);p.add_argument('--seed',type=int,default=1000000)
    p.add_argument('--tasks',nargs='+',choices=tuple(SETTINGS['task_step_limits']))
    a=p.parse_args();selected=a.tasks or list(SETTINGS['task_step_limits']);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    status=out/'status.json'
    lock=open(f'/tmp/univtac-plan-gpu-{a.gpu}.lock','w')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        atomic(status,dict(status='Deferred',gpu=a.gpu,reason='GPU lease busy'));return
    # Lease before starting a policy server; never take a GPU with existing users.
    memory=int(subprocess.check_output(['nvidia-smi',f'--id={a.gpu}','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
    if memory>512:
        atomic(status,dict(status='Deferred',gpu=a.gpu,memory_mib=memory));return
    atomic(status,dict(status='Running',pid=os.getpid(),gpu=a.gpu,started=time.time()))
    env=dict(os.environ,TRAIN_PYTHON='/mnt/yichi_data/miniconda3/envs/univtac5090/bin/python',
             SIM_PYTHON='/mnt/yichi_data/miniconda3/envs/univtac5090/bin/python',
             POLICY_GPU=str(a.gpu),SIM_GPU=str(a.gpu),POLICY_PORT=str(a.port),EVAL_ROOT=str(out),
             START_SEED=str(a.seed),MAX_SEED='-1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
    env['TASKS']=' '.join(selected)
    root=Path(__file__).resolve().parents[1]
    exit_code,error=run_with_watchdog(
        ['bash',str(root/'training/evaluate.sh'),a.checkpoint,str(a.episodes)],cwd=root,env=env,
        result_files=[out/f'{task}.result.json' for task in selected])
    tasks={}
    for task in selected:
        file=out/f'{task}.result.json'
        if file.exists():tasks[task]=json.loads(file.read_text())
    complete=exit_code==0 and len(tasks)==len(selected) and all(r['complete'] and r['valid']==a.episodes and r['context']['step_lim']==SETTINGS['task_step_limits'][t] and r['context']['checkpoint']==a.checkpoint for t,r in tasks.items())
    summary=dict(status='Completed' if complete else 'Failed',exit_code=exit_code,gpu=a.gpu,
                 checkpoint=a.checkpoint,episodes=a.episodes,seed=a.seed,finished=time.time(),tasks=tasks)
    if error:summary['error']=error
    if complete:summary['macro_success_rate']=sum(r['successes']/r['valid'] for r in tasks.values())/len(selected)
    atomic(status,summary)


if __name__=='__main__':main()
