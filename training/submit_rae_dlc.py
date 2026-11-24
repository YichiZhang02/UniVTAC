"""Freeze and submit the four 100-epoch RAE jobs; ledger prevents duplicates."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
from training.dlc_resources import snapshot,preferred,POOLS,replace_argument

ROOT=Path(__file__).resolve().parents[1]
REGION=['--region','cn-wulanchabu','--endpoint','pai-dlc.cn-wulanchabu.aliyuncs.com']


def atomic(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2));temp.replace(path)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--experiment',required=True)
    p.add_argument('--submit',action='store_true')
    p.add_argument('--gpus',type=int,default=1)
    p.add_argument('--resource-id',help='Override A800-first, A100-fallback selection')
    p.add_argument('--micro-batch',type=int)
    p.add_argument('--precision',choices=['amp_bf16','bf16','fp32'],default='amp_bf16')
    p.add_argument('--cache-root',default=str(ROOT/'data/contact_depth_deform_radial_v1'))
    p.add_argument('--samples-per-epoch',type=int,default=80000)
    args=p.parse_args()
    if args.gpus<1:raise ValueError('GPU count must be positive')
    micro=args.micro_batch or 64//args.gpus
    if micro<1 or 64%(args.gpus*micro):raise ValueError('Invalid GPU count/micro batch for global batch64')
    folder=Path(args.experiment).resolve();folder.mkdir(parents=True,exist_ok=True)
    ledger=folder/'jobs.json'
    if ledger.exists():
        records=json.loads(ledger.read_text())
    else:
        cache_manifest=Path(args.cache_root)/'manifest.json'
        if not cache_manifest.exists() or not json.loads(cache_manifest.read_text()).get('complete'):
            raise RuntimeError('Contact radial cache is not complete; finish training.cache_tactile first')
        source=folder/'source';source.mkdir()
        for name in ('encoder','training'):
            shutil.copytree(ROOT/name,source/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        eval_source=source/'data/encoder_eval_v1';eval_source.parent.mkdir(parents=True)
        shutil.copytree(ROOT/'data/encoder_eval_v1',eval_source)
        hashes={str(f.relative_to(source)):hashlib.sha256(f.read_bytes()).hexdigest()
                for f in source.rglob('*') if f.is_file()}
        atomic(folder/'source_sha256.json',hashes)
        stamp=datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')
        records=[]
        for size in ('S','B'):
            for variant in ('rae-1in1','rae-4in1'):
                run=folder/f'vit{size.lower()}_{variant}';run.mkdir()
                argv=['/mnt/yichi_data/miniconda3/envs/univtac/bin/python','-u','-m','torch.distributed.run',
                      '--standalone',f'--nproc_per_node={args.gpus}','-m','training.pretrain_rae',
                      '--variant',variant,'--size',size,'--epochs','100','--output',str(run),
                      '--samples-per-epoch',str(args.samples_per_epoch),
                      '--eval-root',str(eval_source),'--micro-batch',str(micro),'--global-batch','64',
                      '--data-mode','cached','--cache-root',str(Path(args.cache_root).resolve()),'--precision',args.precision,
                      '--workers','4','--stride','2','--interpolation','tps_bilinear','--grid','28']
                command='cd '+shlex.quote(str(source))+' && OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 '+shlex.join(argv)+' > '+shlex.quote(str(run/'console.log'))+' 2>&1'
                name=f'univtac-rae-vit{size.lower()}-{variant}-{stamp}'
                submit=['dlc','submit','pytorchjob',*REGION,'--workspace_id','218916',
                        '--resource_id','quota3ya3nugiuc6','--name',name,
                        '--data_sources','d-lu3fmaylj6r0n6hkms:v1:/mnt/yichi_data/',
                        '--workers','1','--worker_image','daimon-acr-registry-vpc.cn-wulanchabu.cr.aliyuncs.com/daimon/zhangyichi:univtac-train-1001',
                        '--worker_cpu','32','--worker_memory','400Gi','--worker_shared_memory','128Gi',
                        '--worker_gpu',str(args.gpus),'--vpc_id','vpc-0jlibv5lchq2ljpnl6p4i',
                        '--switch_id','vsw-0jlkzeskp0wfxqbgged89','--security_group_id','sg-0jl5nh21jy9r5uts3pk6',
                        '--driver','580.95.05-open','--extended_cidrs','10.11.65.0/24,10.11.64.0/24',
                        '--command',command]
                record=dict(name=name,size=size,variant=variant,epochs=100,gpus=args.gpus,
                            samples_per_epoch=args.samples_per_epoch,precision=args.precision,micro_batch=micro,
                            input_mode='depth_deform_radial',output=str(run),status='Prepared',command=submit)
                atomic(run/'submission.json',record);records.append(record)
        atomic(ledger,records)
    if not args.submit:
        print(json.dumps([{k:v for k,v in r.items() if k!='command'} for r in records],indent=2));return
    # Frozen source integrity is checked before any external mutation.
    for name,sha in json.loads((folder/'source_sha256.json').read_text()).items():
        if hashlib.sha256((folder/'source'/name).read_bytes()).hexdigest()!=sha:
            raise RuntimeError(f'Snapshot changed: {name}')
    env={k:v for k,v in os.environ.items() if k.lower() not in ('http_proxy','https_proxy','all_proxy')}
    for record in records:
        if record.get('job_id'):
            print(f"Already submitted {record['name']}: {record['job_id']}",flush=True);continue
        if record['status']!='Prepared':
            raise RuntimeError(f"Submission state {record['status']}; reconcile {record['name']} before retrying")
        resource_id=args.resource_id or preferred(snapshot(env),record['gpus']) or POOLS[0][1]
        record['command']=replace_argument(record['command'],'--resource_id',resource_id)
        record['resource_id']=resource_id
        atomic(Path(record['output'])/'submission.json',record)
        record['status']='Submitting';atomic(ledger,records)
        try:
            result=subprocess.run(record['command'],env=env,capture_output=True,text=True,timeout=55)
            output=result.stdout+result.stderr
        except subprocess.TimeoutExpired:
            record['status']='Unknown';atomic(ledger,records)
            raise RuntimeError('Submission timed out; query the unique name before retrying')
        (Path(record['output'])/'submission.log').write_text(output)
        match=re.search(r'\bdlc[a-z0-9]{8,}\b',output)
        if result.returncode or '[FAILED]' in output or not match:
            record['status']='SubmissionFailed';atomic(ledger,records);print(output,flush=True)
            raise RuntimeError('Submission failed; inspect ledger/log and reconcile before retrying')
        record.update(job_id=match.group(),status='Submitted');atomic(ledger,records)
        print(json.dumps({k:v for k,v in record.items() if k!='command'}),flush=True)


if __name__=='__main__':main()
