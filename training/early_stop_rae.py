"""Early-stop live immutable DLC runs using completed-epoch validation metadata."""
import argparse
import fcntl
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import time


def decision(status,patience=10):
    epoch=status.get('completed_epoch',0);best=status.get('best_epoch')
    if best is None or epoch<1:return False,0
    stale=max(0,epoch-best)
    return stale>=patience,stale


def atomic(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2));temp.replace(path)


def stop_job(job_id):
    command=['dlc','stop','job',job_id,'--force','--quiet','--region','cn-wulanchabu',
             '--endpoint','pai-dlc.cn-wulanchabu.aliyuncs.com']
    env={k:v for k,v in os.environ.items() if k.lower() not in ('http_proxy','https_proxy','all_proxy')}
    result=subprocess.run(command,env=env,capture_output=True,text=True,timeout=55)
    if result.returncode or '[FAILED]' in result.stdout+result.stderr:
        raise RuntimeError(result.stdout+result.stderr)
    # Confirm the stop completed before recording a terminal state.
    for _ in range(12):
        result=subprocess.run(['dlc','get','job',job_id,'--region','cn-wulanchabu',
                               '--endpoint','pai-dlc.cn-wulanchabu.aliyuncs.com'],env=env,
                              capture_output=True,text=True,timeout=55)
        if result.returncode==0 and '"Status": "Stopped"' in result.stdout:return
        if result.returncode==0 and '"Status": "Succeeded"' in result.stdout:return
        time.sleep(5)
    raise RuntimeError('Stop requested but terminal DLC state not yet confirmed; retry next poll')


def main():
    p=argparse.ArgumentParser();p.add_argument('--experiment',required=True)
    p.add_argument('--patience',type=int,default=10);p.add_argument('--interval',type=int,default=30)
    args=p.parse_args()
    if args.patience<1 or args.interval<1:raise ValueError('Patience and poll interval must be positive')
    root=Path(args.experiment).resolve();jobs=json.loads((root/'jobs.json').read_text())
    lock=(root/'early_stopping.lock').open('a')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise RuntimeError('An early-stopping monitor already owns this experiment')
    path=root/'early_stopping.json'
    history=json.loads(path.read_text()).get('jobs',{}) if path.exists() else {}
    config=dict(pid=os.getpid(),patience=args.patience,poll_seconds=args.interval,
                criterion='strictly lower completed-epoch val MSE; equality is no improvement',
                jobs=history,status='Monitoring')
    print(json.dumps(dict(event='monitor_started',**{k:v for k,v in config.items() if k!='jobs'})),flush=True)
    while True:
        all_done=True
        for job in jobs:
            key=job['job_id'];out=Path(job['output'])
            if history.get(key,{}).get('terminal'):continue
            status_path=out/'status.json'
            if not status_path.exists():
                history[key]=dict(variant=job['variant'],size=job['size'],state='AwaitingFirstEpoch')
                all_done=False;continue
            try:
                status=json.loads(status_path.read_text())
                should_stop,stale=decision(status,args.patience)
                entry=dict(variant=job['variant'],size=job['size'],completed_epoch=status.get('completed_epoch'),
                           best_epoch=status.get('best_epoch'),best_val_mse=status.get('best_val_mse'),
                           stale_epochs=stale,state=status.get('status'),terminal=False)
                if status.get('status') in ('Completed','EarlyStopped'):
                    entry['terminal']=True
                elif should_stop:
                    # status.json is published only AFTER last.pt/best artifacts.
                    # Stopping here preserves a complete best and last checkpoint.
                    stop_job(key)
                    entry.update(state='EarlyStopped',terminal=True,stopped_at=datetime.now(timezone.utc).isoformat())
                    updated=dict(status,status='EarlyStopped',stop_reason=f'{stale} epochs without val MSE improvement',patience=args.patience)
                    atomic(status_path,updated)
                    atomic(out/'early_stop.json',entry)
                    print(json.dumps(dict(event='early_stopped',job_id=key,**entry)),flush=True)
                else:all_done=False
                history[key]=entry
            except Exception as exc:
                all_done=False
                history[key]=dict(history.get(key,{}),error=str(exc),last_error_at=datetime.now(timezone.utc).isoformat())
                print(json.dumps(dict(event='monitor_error',job_id=key,error=str(exc))),flush=True)
        config.update(jobs=history,checked_at=datetime.now(timezone.utc).isoformat(),status='Finished' if all_done else 'Monitoring')
        atomic(path,config)
        if all_done:return
        time.sleep(args.interval)


if __name__=='__main__':main()
