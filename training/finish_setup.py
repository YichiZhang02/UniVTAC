"""Wait for verified downloads and local DDP checks; submit the two baselines once."""
import json
import os
import re
from pathlib import Path
import subprocess
import time
ROOT=Path('/mnt/yichi_data/UniVTAC');DOWNLOADS=ROOT.parent/'UniVTAC_downloads'
PYTHON=str(ROOT.parent/'miniconda3/envs/univtac/bin/python')
TASKS=('grasp_classify','insert_HDMI','insert_hole','insert_tube','lift_bottle','lift_can','pull_out_key','put_bottle_in_shelf')
def main():
    target=ROOT/'runs';target.mkdir(exist_ok=True)
    marker=target/'baseline_submissions_complete.json'
    if marker.exists():raise SystemExit('Submissions already recorded')
    retries=0
    while True:
        counts={t:len(list((DOWNLOADS/'data/isaac51'/t/'hdf5').glob('*.hdf5'))) for t in TASKS}
        tests={m:(DOWNLOADS/f'local-ddp-smoke-{m}/step_000001.pt').exists() for m in ('act','starvla')}
        print(json.dumps(dict(counts=counts,ddp_checkpoints=tests)),flush=True)
        for m in ('act','starvla'):
            log=DOWNLOADS/f'ddp-smoke-{m}.log'
            if log.exists() and any(x in log.read_text(errors='replace') for x in ('Traceback (most recent call last)','CUDA out of memory')):
                raise RuntimeError(f'{m} smoke failed; inspect {log}')
        if not all(n==100 for n in counts.values()) and not Path('/proc/3132128').exists():
            if retries>=2:raise RuntimeError('Dataset download incomplete after retries')
            retries+=1
            env={k:v for k,v in os.environ.items() if k.lower() not in ('http_proxy','https_proxy','all_proxy')}
            subprocess.run([str(Path(PYTHON).parent/'modelscope'),'download','--dataset','byml2024/UniVTAC','--local_dir',str(DOWNLOADS/'data'),'--include','isaac51/*','--max-workers','8'],env=env,check=True)
            continue
        if all(n==100 for n in counts.values()) and all(tests.values()):break
        time.sleep(30)
    subprocess.run([PYTHON,'-m','training.verify_downloads'],cwd=ROOT,check=True)
    results=[]
    for model in ('act','starvla'):
        prior=[f for f in sorted((ROOT/'runs').glob(f'univtac-clean-{model}-*/submission.log')) if re.search(r'\bdlc[a-z0-9]{10,}\b',f.read_text()) and '[FAILED]' not in f.read_text()]
        if prior:
            results.append(dict(model=model,submission=prior[-1].read_text()));continue
        result=subprocess.run(['/opt/conda/bin/python','-m','training.submit_dlc','--model',model,'--gpus','4'],cwd=ROOT,check=True,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        print(result.stdout,flush=True);results.append(dict(model=model,submission=result.stdout))
    marker.write_text(json.dumps(results,indent=2))
    subprocess.run(['/opt/conda/bin/python','-m','training.watch_runs'],cwd=ROOT,check=True)

if __name__=='__main__':main()
