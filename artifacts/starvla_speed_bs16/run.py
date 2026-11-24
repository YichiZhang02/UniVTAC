import os,subprocess,sys
from pathlib import Path
model=sys.argv[1]
for tactile in (False,True):
 stem=model+('_tactile' if tactile else '_vision')
 args=[sys.executable,'/tmp/benchmark_starvla_speed.py','--model',model,'--output',f'artifacts/starvla_speed_bs16/{stem}.json']
 if tactile:args.append('--tactile')
 with Path(f'artifacts/starvla_speed_bs16/{stem}.log').open('w') as f:
  result=subprocess.run(args,stdout=f,stderr=subprocess.STDOUT)
 print(stem,'exit',result.returncode,flush=True)
