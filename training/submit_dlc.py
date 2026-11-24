"""DLC launcher. Uses platform credentials; records the exact command and response."""
from training.tactile_vlm import add_tactile_arguments, apply_tactile_arguments
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import shutil
import hashlib
from datetime import datetime,timezone
ROOT=Path('/mnt/yichi_data/UniVTAC')
def main():
    p=argparse.ArgumentParser()
    add_tactile_arguments(p)
    p.add_argument('--model',choices=['act','starvla','starvla_pi','starvla_groot_v1'],required=True)
    p.add_argument('--gpus',type=int,default=4)
    p.add_argument('--resource-id',help='Override automatic A800-first, A100-fallback selection')
    p.add_argument('--config',help='JSON configuration to freeze into this source snapshot')
    p.add_argument('--output',help='Explicit new run directory')
    p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--source-root',help='Immutable experiment source template')
    p.add_argument('--steps',type=int,default=50000)
    p.add_argument('--tactile',action='store_true')
    p.add_argument('--encoder',choices=['univtac','rae'],default='univtac')
    p.add_argument('--rae-variant',choices=['rae-1in1','rae-4in1'],default='rae-4in1')
    p.add_argument('--encoder-checkpoint')
    p.add_argument('--rae-size',choices=['S','B'],default='S')
    p.add_argument('--tokens',choices=['full','pool3x3','learned3x3','cls'],default='full')
    a=p.parse_args()
    if a.gpus<1 or 64%a.gpus: raise ValueError('GPU count must divide global batch64')
    micro=min(16,64//a.gpus)
    name=f"univtac-clean-{a.model}-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}"
    folder=Path(a.output).resolve() if a.output else ROOT/'runs'/name
    folder.mkdir(parents=True,exist_ok=False)
    source=folder/'source'
    source.mkdir()
    source_root=Path(a.source_root).resolve() if a.source_root else ROOT
    for source_name in ('training','encoder','policy/ACT','policy/starvla_groot','policy/starvla_pi','policy/starvla_groot_v1','policy/fresh','instructions'):
        shutil.copytree(source_root/source_name,source/source_name,ignore=shutil.ignore_patterns('__pycache__','*.pyc','.git'))
    shutil.copy2(source_root/'policy/_base_policy.py',source/'policy/_base_policy.py')
    if a.config:
        config=json.loads(Path(a.config).read_text())
        (source/'training/config.json').write_text(json.dumps(config,indent=2)+'\n')
    manifest={str(f.relative_to(source)):hashlib.sha256(f.read_bytes()).hexdigest() for f in source.rglob('*') if f.is_file()}
    (folder/'source_sha256.json').write_text(json.dumps(manifest,indent=2))
    command=['bash',str(source/'training/launch.sh'),'--model',a.model,'--output',str(folder),'--steps',str(a.steps),'--micro-batch',str(micro)]
    if a.tactile:command+=['--tactile','--encoder',a.encoder,'--tokens',a.tokens]
    if a.tactile and a.encoder=='rae':command+=['--rae-variant',a.rae_variant,'--rae-size',a.rae_size]
    if a.encoder_checkpoint:command+=['--encoder-checkpoint',a.encoder_checkpoint]
    if a.freeze_tactile_encoder is not None:
        command+=['--freeze-tactile-encoder' if a.freeze_tactile_encoder else '--no-freeze-tactile-encoder']
    for key in ('tactile_fusion','tactile_wrapper','tactile_rope','vlm_attention_mask'):
        if getattr(a,key) is not None:command+=['--'+key.replace('_','-'),getattr(a,key)]
    from training.dlc_resources import snapshot,preferred,POOLS
    resource_id=a.resource_id or preferred(snapshot(),a.gpus) or POOLS[0][1]
    cmd=['dlc','submit','pytorchjob','--region','cn-wulanchabu','--endpoint','pai-dlc.cn-wulanchabu.aliyuncs.com',
         '--workspace_id','218916','--resource_id',resource_id,'--name',name,
         '--data_sources','d-lu3fmaylj6r0n6hkms:v1:/mnt/yichi_data/',
         '--workers','1','--worker_image','daimon-acr-registry-vpc.cn-wulanchabu.cr.aliyuncs.com/daimon/zhangyichi:univtac-train-1001',
         '--worker_cpu','32','--worker_memory','400Gi','--worker_shared_memory','128Gi',
         '--worker_gpu',str(a.gpus),'--vpc_id','vpc-0jlibv5lchq2ljpnl6p4i',
         '--switch_id','vsw-0jlkzeskp0wfxqbgged89','--security_group_id','sg-0jl5nh21jy9r5uts3pk6',
         '--driver','580.95.05-open','--extended_cidrs','10.11.65.0/24,10.11.64.0/24',
         '--command',f'GPUS={a.gpus} '+shlex.join(command)]
    (folder/'submission.json').write_text(json.dumps(cmd,indent=2))
    (folder/'prepared.json').write_text(json.dumps(dict(name=name,output=str(folder),model=a.model,gpus=a.gpus),indent=2))
    if a.prepare_only:
        print(json.dumps(dict(name=name,output=str(folder),status='Prepared')));return
    env={k:v for k,v in os.environ.items() if k.lower() not in ('http_proxy','https_proxy','all_proxy')}
    result=subprocess.run(cmd,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    (folder/'submission.log').write_text(result.stdout)
    print(result.stdout)
    if '[FAILED]' in result.stdout or result.returncode:raise SystemExit(1)
if __name__=='__main__':main()
