"""Read-only monitor for submitted fresh baselines; writes status and final-step evidence."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import time
ROOT=Path('/mnt/yichi_data/UniVTAC')
def main():
    p=argparse.ArgumentParser();p.add_argument('--interval',type=int,default=60);a=p.parse_args()
    env={k:v for k,v in os.environ.items() if k.lower() not in ('http_proxy','https_proxy','all_proxy')}
    while True:
        entries=[]
        for folder in sorted((ROOT/'runs').glob('univtac-clean-*')):
            log=folder/'submission.log'
            if not log.exists():continue
            ids=re.findall(r'\b(dlc[a-z0-9]{10,})\b',log.read_text())
            if not ids:continue
            job=ids[-1]
            cmd=['dlc','get','job',job,'--workspace_id','218916','--region','cn-wulanchabu','--endpoint','pai-dlc.cn-wulanchabu.aliyuncs.com']
            result=subprocess.run(cmd,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
            (folder/'dlc_status.log').write_text(result.stdout)
            status=re.findall(r'"Status":\s*"([^"]+)"',result.stdout)
            final=folder/'step_050000.pt'
            metrics=folder/'train.jsonl'
            last=json.loads(metrics.read_text().strip().splitlines()[-1]) if metrics.exists() and metrics.stat().st_size else None
            entries.append(dict(run=folder.name,job=job,status=status[-1] if status else 'unknown',last_metric=last,checkpoint_50000=final.exists()))
        target=ROOT/'runs/status.json';temp=target.with_suffix('.tmp');temp.write_text(json.dumps(entries,indent=2));temp.replace(target)
        print(json.dumps(entries),flush=True)
        if entries and all(x['status'] in ('Succeeded','Failed','Stopped') for x in entries):return
        time.sleep(a.interval)
if __name__=='__main__':main()
