import subprocess,sys
from pathlib import Path
model=sys.argv[1]
for tactile in (False,True):
 stem=model+('_tactile' if tactile else '_vision')
 args=[sys.executable,'/tmp/benchmark_starvla_e2e.py','--model',model,'--output',f'artifacts/starvla_e2e_bs16_rae1_pool/{stem}.json']
 if tactile:args.append('--tactile')
 with Path(f'artifacts/starvla_e2e_bs16_rae1_pool/{stem}.log').open('w') as f:
  result=subprocess.run(args,stdout=f,stderr=subprocess.STDOUT)
 print(stem,'exit',result.returncode,flush=True)
