"""Persistent DLC -> ECS staged StarVLA experiment driver (standard library only)."""
import argparse,csv,fcntl,hashlib,itertools,json,os,re,shlex,shutil,subprocess,sys,time
from pathlib import Path
from datetime import datetime
from training.eval_config import SETTINGS
from training.lr_schedule import DEFAULTS as LR_DEFAULTS
from training.dlc_resources import snapshot as resource_snapshot,capacity,preferred,POOLS,replace_argument

ROOT=Path(__file__).resolve().parents[1]
PYTHON='/mnt/yichi_data/miniconda3/envs/univtac/bin/python'
REGION=['--region','cn-wulanchabu','--endpoint','pai-dlc.cn-wulanchabu.aliyuncs.com','--workspace_id','218916']
TERMINAL={'Succeeded','Failed','Stopped','Terminated','Deleted'}
ENV={k:v for k,v in os.environ.items() if k.lower() not in ('http_proxy','https_proxy','all_proxy')}


def atomic(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2));temp.replace(path)


def signature(cfg):
    return hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()


def run(argv,timeout=55):
    result=subprocess.run(argv,cwd=ROOT,env=ENV,capture_output=True,text=True,timeout=timeout)
    return result.returncode,result.stdout+result.stderr


class Plan:
    def __init__(self,folder):
        self.folder=Path(folder).resolve();self.file=self.folder/'state.json'
        self.ssh=['ssh','-F',str(self.folder/'ssh_config'),'-o','BatchMode=yes','-o','ConnectTimeout=15','ecs-5090-fifa']
        if self.file.exists():self.state=json.loads(self.file.read_text())
        else:
            base=json.loads((self.folder/'configs/baseline_starvla.json').read_text());base.pop('model',None)
            self.state=dict(version=1,status='Running',created=time.time(),base=base,jobs={},stages={},evaluations={},encoders={},errors=[],max_training=2,max_evaluation=2)
            for r in json.loads((self.folder/'bootstrap_jobs.json').read_text()):
                cfg=json.loads((self.folder/'configs'/f"{r['id']}.json").read_text())
                self.state['jobs'][r['id']]=dict(r,stage='baseline',config=cfg,signature=signature(cfg),gpus=4)
            self.state['stages']['baseline']=list(self.state['jobs'])
            self.save()

    def save(self):
        self.state['checked_at']=time.time();atomic(self.file,self.state)
        atomic(ROOT/'runs/active_starvla_plan.json',dict(experiment=str(self.folder),status=self.state['status'],checked_at=time.time()))

    def event(self,event,**data):
        row=dict(time=time.time(),event=event,**data)
        with (self.folder/'events.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)

    def encoder_path(self,size,variant):return self.folder/'encoders'/f'vit{size.lower()}_{variant}'/'best.pt'

    def collect_encoders(self):
        ledger=ROOT/'runs/rae_cached_single64_100epoch_20261116/jobs.json'
        for job in json.loads(ledger.read_text()):
            key=f"{job['size']}/{job['variant']}"
            if key in self.state['encoders']:continue
            output=Path(job['output']);status_file=output/'status.json'
            if not status_file.exists():continue
            status=json.loads(status_file.read_text())
            if status['status'] not in ('Completed','EarlyStopped'):continue
            best=output/'best.pt'
            if not best.exists():continue
            target=self.encoder_path(job['size'],job['variant']);target.parent.mkdir(exist_ok=True)
            if not target.exists():
                temp=target.with_suffix('.tmp');shutil.copy2(best,temp);temp.replace(target)
            sha=hashlib.sha256(target.read_bytes()).hexdigest()
            self.state['encoders'][key]=dict(path=str(target),sha256=sha,source=str(best),best_epoch=status['best_epoch'],best_val_mse=status['best_val_mse'],job_id=job['job_id'])
            atomic(target.parent/'provenance.json',self.state['encoders'][key]);self.save();self.event('encoder_locked',key=key)

    def config(self,model='starvla',**updates):
        cfg=dict(self.state['base']);cfg.update(model=model,**updates)
        if cfg['tactile']:
            cfg['tactile_frames']=1 if cfg['rae_variant']=='rae-1in1' else 4
            cfg['encoder_checkpoint']=str(self.encoder_path(cfg['rae_size'],cfg['rae_variant']))
        return cfg

    def add(self,stage,cfg):
        sha=signature(cfg)
        for key,job in self.state['jobs'].items():
            if job['signature']==sha:return key
        key=f'{stage}_{len(self.state["jobs"]):02d}_{sha[:8]}'
        output=self.folder/'policies'/key
        self.state['jobs'][key]=dict(id=key,stage=stage,model=cfg['model'],config=cfg,signature=sha,output=str(output),status='Planned',gpus=4)
        if self.state.get('plan_revision')==2 and cfg.get('lr_schedule')=='cosine':
            self.state['jobs'][key]['source_template']=str(self.folder/'frozen_source_v2')
        atomic(self.folder/'configs'/f'{key}.json',cfg);self.save();return key

    def ensure_stage(self,name,configs):
        if name in self.state['stages']:return
        ids=[self.add(name,cfg) for cfg in configs]
        self.state['stages'][name]=ids;self.save();self.event('stage_created',stage=name,jobs=ids)

    def evaluation(self,key,seed=1000000,round_index=0,kind='selection'):
        ident=f'{key}_{kind}_{round_index}'
        if ident not in self.state['evaluations']:
            self.state['evaluations'][ident]=dict(id=ident,job=key,status='Planned',kind=kind,seed=seed,episodes=50,round=round_index,output=str(self.folder/'evaluations'/ident))
            self.save()
        return self.state['evaluations'][ident]

    def score(self,key):
        rows=[e for e in self.state['evaluations'].values() if e['job']==key and e['kind']=='selection' and e['status']=='Completed']
        if not rows:return None
        successes={};counts={}
        for row in rows:
            for task,r in row['result']['tasks'].items():
                successes[task]=successes.get(task,0)+r['successes'];counts[task]=counts.get(task,0)+r['valid']
        return sum(successes[k]/counts[k] for k in successes)/8

    def winner(self,stage):
        winners=self.state.setdefault('winners',{})
        override=self.state.get('selection_overrides',{}).get(stage)
        if override:
            key=override['job']
            if key not in self.state['stages'].get(stage,[]):
                raise ValueError('Selected configuration is not a stage candidate: '+stage)
            if winners.get(stage)!=key:
                winners[stage]=key;self.save()
                self.event('stage_configuration_selected',stage=stage,job=key,
                           reason=override['reason'],score=self.score(key))
            return key
        if stage in winners:return winners[stage]
        ids=self.state['stages'].get(stage)
        if not ids:return None
        if any(self.score(key) is None for key in ids):return None
        # Wait until all requested tie-break rounds have finished.
        if any(e['status']!='Completed' for e in self.state['evaluations'].values() if e['job'] in ids and e['kind']=='selection'):return None
        scores={key:self.score(key) for key in ids};best=max(scores.values());tied=[k for k,v in scores.items() if abs(v-best)<1e-12]
        if len(tied)==1:
            winners[stage]=tied[0];self.save();self.event('stage_winner',stage=stage,job=tied[0],score=best);return tied[0]
        rounds=max(e['round'] for e in self.state['evaluations'].values() if e['job'] in tied and e['kind']=='selection')
        if rounds>=3:
            self.state['stages'][stage+'_tie']=dict(status='NeedsAttention',candidates=tied,score=best)
            return None
        for key in tied:self.evaluation(key,1000000+(rounds+1)*1000,rounds+1)
        self.event('tie_extension',stage=stage,candidates=tied,round=rounds+1);return None

    def advance(self):
        if self.state.get('independent_vlm_selection'):
            # Explicitly enabled model routes replace the historical GR00T winner.
            # Only explicitly enabled model stages advance.
            if 'S/rae-1in1' not in self.state['encoders']:return
            for route in self.state.get('independent_vlm_a_models',[]):
                model={'pi':'starvla_pi','groot_v1':'starvla_groot_v1'}[route]
                baseline=next(j['config'] for j in self.state['jobs'].values()
                              if j['stage']=='baseline_v2' and j['model']==model)
                cfg=dict(baseline,**LR_DEFAULTS)
                cfg.update(tactile=True,encoder='rae',rae_size='S',rae_variant='rae-1in1',
                           tactile_frames=1,encoder_checkpoint=str(self.encoder_path('S','rae-1in1')),
                           tactile_tokens='pool3x3',freeze_tactile_encoder=False,
                           tactile_fusion='vlm',vlm_attention_mask='causal',action_future_tokens=0,
                           head_action_queries_only=model!='starvla_groot_v1')
                stage='vlm_A_'+route
                self.ensure_stage(stage,[dict(cfg,tactile_wrapper=w,tactile_rope=r)
                    for w,r in itertools.product(('none','vision','tactile'),('none','1d','3d'))])
                best=self.winner(stage)
                if best and route in self.state.get('independent_vlm_b_models',[]):
                    selected=self.state['jobs'][best]['config']
                    b_stage='vlm_B_'+route
                    self.ensure_stage(b_stage,[dict(selected,tactile_tokens=t,freeze_tactile_encoder=f)
                        for t,f in itertools.product(('pool3x3','learned3x3'),(False,True))])
                    b_best=self.winner(b_stage)
                    if b_best and route in self.state.get('independent_vlm_c_models',[]):
                        required={f'{s}/{v}' for s,v in itertools.product(('S','B'),('rae-1in1','rae-4in1'))}
                        if not required.issubset(self.state['encoders']):continue
                        selected=self.state['jobs'][b_best]['config']
                        c_stage='vlm_C_'+route
                        freeze_options=self.state.get('pi_c_freeze_grid',[selected['freeze_tactile_encoder']]) if route=='pi' else [selected['freeze_tactile_encoder']]
                        self.ensure_stage(c_stage,[dict(selected,rae_size=s,rae_variant=v,freeze_tactile_encoder=frozen,
                            tactile_frames=1 if v=='rae-1in1' else 4,
                            encoder_checkpoint=str(self.encoder_path(s,v)))
                            for frozen,s,v in itertools.product(freeze_options,('S','B'),('rae-1in1','rae-4in1'))])
                        c_best=self.winner(c_stage)
                        if c_best and route in self.state.get('independent_vlm_d_models',[]):
                            selected=self.state['jobs'][c_best]['config']
                            d_stage='vlm_D_'+route
                            self.ensure_stage(d_stage,[dict(selected,vlm_attention_mask=mask)
                                for mask in ('causal','modality','bidirectional')])
                            self.winner(d_stage)
            return
        if 'S/rae-1in1' in self.state['encoders']:
            self.ensure_stage('vlm_A',[self.config(tactile=True,encoder='rae',rae_size='S',rae_variant='rae-1in1',tactile_tokens='pool3x3',freeze_tactile_encoder=False,tactile_fusion='vlm',tactile_wrapper=w,tactile_rope=r,vlm_attention_mask='causal') for w,r in itertools.product(('none','vision','tactile'),('none','1d','3d'))])
        best=self.winner('vlm_A')
        if best:
            cfg=self.state['jobs'][best]['config']
            if self.state.get('plan_revision')==2:
                model='starvla_groot_v1'
                cfg=dict(cfg,**LR_DEFAULTS);cfg.update(model=model,head_action_queries_only=model!='starvla_groot_v1',action_future_tokens=0)
            self.ensure_stage('vlm_B',[dict(cfg,tactile_tokens=t,freeze_tactile_encoder=f) for t,f in itertools.product(('pool3x3','learned3x3'),(False,True))])
        best=self.winner('vlm_B')
        if best and len(self.state['encoders'])==4:
            cfg=self.state['jobs'][best]['config']
            self.ensure_stage('vlm_C',[dict(cfg,rae_size=s,rae_variant=v,tactile_frames=1 if v=='rae-1in1' else 4,encoder_checkpoint=str(self.encoder_path(s,v))) for s,v in itertools.product(('S','B'),('rae-1in1','rae-4in1'))])
        best=self.winner('vlm_C')
        if best:
            cfg=self.state['jobs'][best]['config'];self.ensure_stage('vlm_D',[dict(cfg,vlm_attention_mask=m) for m in ('causal','modality','bidirectional')])
        best=self.winner('vlm_D')
        if best:
            cfg=self.state['jobs'][best]['config']
            other='starvla_pi' if cfg['model']!='starvla_pi' else 'starvla_groot_v1'
            self.ensure_stage('vlm_E',[dict(cfg,model=other,head_action_queries_only=other!='starvla_groot_v1')])
            self.ensure_stage('head',[dict(cfg,model=m,head_action_queries_only=m!='starvla_groot_v1',tactile_fusion='head',tactile_wrapper='none',tactile_rope='none') for m in (('starvla_groot_v1','starvla_pi') if self.state.get('plan_revision')==2 else ('starvla','starvla_pi'))])
            primary=self.state['stages']['baseline_v2' if self.state.get('plan_revision')==2 else 'baseline']+[best]+self.state['stages']['vlm_E']+self.state['stages']['head']
            if all(self.score(key) is not None for key in primary):
                for key in primary:self.evaluation(key,2000000,0,'final')
                finals=[e for e in self.state['evaluations'].values() if e['kind']=='final']
                if len(finals)==6 and all(e['status']=='Completed' for e in finals):self.state['status']='Completed'

    def cloud_status(self,job):
        code,output=run(['dlc','get','job',job['job_id'],*REGION])
        (Path(job['output'])/'dlc_status.log').write_text(output)
        try:
            payload=json.loads(output[output.index('{'):])
            job['resource_id']=payload.get('ResourceId',job.get('resource_id'))
            job['cloud_timing']={k:payload.get(k) for k in ('GmtCreateTime','GmtRunningTime','GmtFinishTime','GmtFinishedTime')}
        except (ValueError,KeyError):pass
        found=re.findall(r'"Status":\s*"([^"]+)"',output)
        if code or not found:raise RuntimeError(f'Could not query DLC {job["job_id"]}')
        return found[-1]

    def poll_training(self):
        for key,job in self.state['jobs'].items():
            if not job.get('job_id') or job['status'] in ('TrainingSucceeded','TrainingFailed','Relocating','RelocationFailed'):continue
            status=self.cloud_status(job);job['cloud_status']=status
            metrics=Path(job['output'])/'train.jsonl'
            if metrics.exists():
                lines=metrics.read_text().splitlines()
                if lines:job['last_metric']=json.loads(lines[-1])
            if status=='Succeeded':
                checkpoint=Path(job['output'])/'step_050000.pt'
                if not checkpoint.exists() or job.get('last_metric',{}).get('step')!=50000:
                    job.update(status='TrainingFailed',error='DLC succeeded without complete 50k artifacts')
                else:
                    job.update(status='TrainingSucceeded',checkpoint=str(checkpoint));self.evaluation(key)
                    self.event('training_completed',job=key)
            elif status in TERMINAL:job.update(status='TrainingFailed',error=f'DLC {status}')
            else:job['status']='Submitted'
            self.save()

    def reconcile_unknown(self,job):
        code,output=run(['dlc','get','job','--display_name',job['name'],*REGION])
        ids=list(dict.fromkeys(re.findall(r'\bdlc[a-z0-9]{8,}\b',output)))
        if not code and len(ids)==1:job.update(job_id=ids[0],status='Submitted')
        else:job.update(status='SubmissionUnknown',error='Reconcile unique DLC name before retry; no duplicate submission')
        self.save()

    def stop_queued(self,job):
        code,text=run(['dlc','stop','job',job['job_id'],'--force','--quiet',*REGION[:4]])
        (Path(job['output'])/'transfer_stop.log').write_text(text)
        if code or '[FAILED]' in text:raise RuntimeError('Could not stop queued DLC job '+job['job_id'])

    def rebalance_resources(self):
        pools=resource_snapshot(ENV)
        self.state['resources']=dict(checked_at=time.time(),preference=[name for name,_ in POOLS],
            pools={key:dict(name=q['QuotaName'],details=q['QuotaDetails'],available_jobs=capacity(q)) for key,q in pools.items()})
        # Finish interrupted transfers before considering new ones. Never resubmit before confirmed stop.
        for key,job in self.state['jobs'].items():
            if job['status']!='Relocating':continue
            status=self.cloud_status(job);job['cloud_status']=status
            if status=='Stopped':
                if job.get('cloud_timing',{}).get('GmtRunningTime') or (Path(job['output'])/'train.jsonl').exists():
                    job.update(status='RelocationFailed',error='Source job started while stopping; inspect artifacts before restart');self.save();continue
                output=Path(job['output']);old=job['job_id'];move=job['relocation']
                job.setdefault('job_history',[]).append(dict(job_id=old,resource_id=job['resource_id'],status=status,time=time.time()))
                for filename in ('submission.json','submission.log'):
                    if (output/filename).exists():shutil.copy2(output/filename,output/(old+'.'+filename))
                cmd=json.loads((output/'submission.json').read_text())
                name=job['name']+'-a100-'+str(int(time.time()))
                cmd=replace_argument(replace_argument(cmd,'--resource_id',move['target']),'--name',name)
                atomic(output/'submission.json',cmd)
                job.update(name=name,status='Prepared',resource_id=move['target'],resource_pinned=True)
                job.pop('job_id');job.pop('cloud_status',None);job.pop('cloud_timing',None);job.pop('relocation')
                self.event('resource_transfer_ready',job=key,old_job_id=old,target=job['resource_id']);self.save()
            elif status in TERMINAL:
                job.update(status='RelocationFailed',error='Unexpected source terminal state '+status);self.save()
            elif status=='Running':
                job.update(status='Submitted');job.pop('relocation');self.save()
            elif status=='Queuing':
                self.stop_queued(job)
        target=POOLS[1][1]
        reserved=sum(1 for j in self.state['jobs'].values() if j['status']=='Relocating' or (j['status']=='Prepared' and j.get('resource_pinned')))
        slots=max(0,capacity(pools[target])-reserved)
        for key,job in self.state['jobs'].items():
            if slots<=0:break
            if job['status']!='Submitted' or job.get('resource_id')!=POOLS[0][1] or job.get('cloud_status')!='Queuing':continue
            if self.cloud_status(job)!='Queuing':continue
            if (Path(job['output'])/'train.jsonl').exists():continue
            job.update(status='Relocating',relocation=dict(target=target,started=time.time()))
            self.save();self.event('resource_transfer_stopping',job=key,old_job_id=job['job_id'],target=target)
            self.stop_queued(job);slots-=1
        self.save()

    def submit(self):
        active=sum(j['status'] in ('Submitted','Submitting','SubmissionUnknown','Relocating') for j in self.state['jobs'].values())
        for key,job in self.state['jobs'].items():
            if active>=self.state['max_training']:break
            if job['status'] not in ('Planned','Prepared'):continue
            cfg=job['config']
            if cfg['tactile'] and not Path(cfg['encoder_checkpoint']).exists():continue
            output=Path(job['output'])
            if job['status']=='Planned':
                if not (output/'prepared.json').exists():
                    code,text=run([PYTHON,'-m','training.submit_dlc','--model',job['model'],'--gpus','4','--config',str(self.folder/'configs'/f'{key}.json'),'--output',str(output),'--prepare-only','--source-root',str(job.get('source_template',self.folder/'frozen_source')),'--resource-id',POOLS[0][1]],timeout=300)
                    if code:job.update(status='PreparationFailed',error=text[-3000:]);self.save();continue
                meta=json.loads((output/'prepared.json').read_text());job.update(name=meta['name'],status='Prepared');self.save()
            for name,sha in json.loads((output/'source_sha256.json').read_text()).items():
                if hashlib.sha256((output/'source'/name).read_bytes()).hexdigest()!=sha:raise RuntimeError('Frozen source changed')
            if job.get('resume_source'):
                for name,sha in json.loads((output/'resume_source_sha256.json').read_text()).items():
                    if hashlib.sha256((Path(job['resume_source'])/name).read_bytes()).hexdigest()!=sha:raise RuntimeError('Resume source changed')
            if not job.get('resource_pinned'):
                pools=resource_snapshot(ENV)
                job['resource_id']=preferred(pools,job['gpus']) or POOLS[0][1]
                atomic(output/'submission.json',replace_argument(json.loads((output/'submission.json').read_text()),'--resource_id',job['resource_id']))
            job['status']='Submitting';self.save()
            try:
                code,text=run(json.loads((output/'submission.json').read_text()))
                (output/'submission.log').write_text(text)
                match=re.search(r'\bdlc[a-z0-9]{8,}\b',text)
                if code or '[FAILED]' in text or not match:job.update(status='SubmissionFailed',error=text[-3000:])
                else:job.update(job_id=match.group(),status='Submitted');active+=1;self.event('training_submitted',job=key,job_id=job['job_id'])
            except subprocess.TimeoutExpired:job['status']='SubmissionUnknown'
            self.save()

    def poll_evaluations(self):
        for e in self.state['evaluations'].values():
            if e['status'] not in ('Running','Launching','LaunchUnknown'):continue
            path=Path(e['output'])/'status.json'
            if not path.exists():continue
            result=json.loads(path.read_text());status=result['status']
            if status=='Completed':
                if len(result.get('tasks',{}))!=8 or any(r['valid']!=50 or not r['complete'] for r in result['tasks'].values()):raise RuntimeError('Incomplete evaluation marked completed')
                e.update(status='Completed',result=result);self.event('evaluation_completed',id=e['id'],macro=result['macro_success_rate'])
            elif status=='Failed':
                # Preserve per-rollout records; at most one retry for transient faults.
                attempts=e.get('attempts',0)
                e.update(status='Planned' if attempts<2 else 'Failed',error='See remote console/status')
            elif status=='Deferred':e['status']='Planned'
            elif status=='Running':
                e.update(status='Running',remote_pid=result.get('pid',e.get('remote_pid')))
                pid=e.get('remote_pid')
                if pid and time.time()-result.get('started',time.time())>120:
                    code,text=run(self.ssh+[f'ps -p {int(pid)} -o args='])
                    if code!=255 and (code or 'training.remote_evaluation' not in text or e['output'] not in text):
                        e.update(status='Failed',error='ECS actor disappeared; partial per-rollout files retained; inspect orphan children before retry')
            self.save()

    def poll_shards(self):
        for e in self.state['evaluations'].values():
            if e['status']!='Sharded':continue
            for task,shard in e['shards'].items():
                if shard['status']=='WaitingForRepair':
                    dependency=shard['repair_dependency']
                    target=self.state['evaluations'][dependency['evaluation']]['shards'][dependency['task']]
                    if target['status'] in ('Completed','Failed'):
                        shard.update(status='Planned');self.save()
                    else:continue
                if shard['status']=='Completed':continue
                file=Path(shard['output'])/'status.json'
                if not file.exists():continue
                result=json.loads(file.read_text());status=result['status']
                # A worker can finish between the status read and the remote PID check.
                # Its final Completed record must supersede a stale local Failed/Planned.
                if status=='Failed':
                    from training.eval_results import validated_complete
                    saved=Path(shard['output'])/f'{task}.result.json'
                    log=Path(shard['output'])/f'{task}.log'
                    if saved.exists() and log.exists():
                        r=json.loads(saved.read_text())
                        context=dict(task=task,checkpoint=e.get('checkpoint',self.state['jobs'][e['job']]['checkpoint']),
                                     requested=e['episodes'],start_seed=e['seed'],step_lim=SETTINGS['task_step_limits'][task])
                        if validated_complete(r,context) and f"Final Result: {r['successes']}/{r['valid']}" in log.read_text(errors='replace'):
                            # Keep the failed actor's status and exit code for audit.
                            shard.update(status='Completed',result=r,recovery=dict(
                                reason='All requested rollouts saved before abnormal worker shutdown',
                                worker_exit_code=result.get('exit_code'),recovered_at=time.time()))
                            shard.pop('error',None)
                            self.event('evaluation_shard_recovered',id=e['id'],task=task,valid=r['valid'],successes=r['successes'])
                            continue
                if status!='Completed' and shard['status'] not in ('Running','Launching','LaunchUnknown'):continue
                if status=='Completed':
                    r=result.get('tasks',{}).get(task,{})
                    if not (r.get('complete') and r.get('valid')==50 and r.get('context',{}).get('checkpoint')==e.get('checkpoint',self.state['jobs'][e['job']]['checkpoint'])):raise RuntimeError('Invalid evaluation shard')
                    shard.update(status='Completed',result=r);shard.pop('error',None)
                elif status in ('Failed','Deferred'):
                    shard.update(status='Planned' if status=='Deferred' or shard.get('attempts',0)<2 else 'Failed',error='See shard console/status')
                elif status=='Running' and time.time()-result.get('started',time.time())>120:
                    pid=result.get('pid')
                    if pid:
                        code,text=run(self.ssh+[f'ps -p {int(pid)} -o args='])
                        if code!=255 and (code or 'training.remote_evaluation' not in text or shard['output'] not in text):
                            shard.update(status='Failed',error='Shard actor disappeared; inspect children before retry')
            if all(shard['status']=='Completed' for shard in e['shards'].values()):
                tasks={t:shard['result'] for t,shard in e['shards'].items()}
                result=dict(status='Completed',checkpoint=e.get('checkpoint',self.state['jobs'][e['job']]['checkpoint']),checkpoint_step=e.get('checkpoint_step',50000),tasks=tasks,macro_success_rate=sum(r['successes']/r['valid'] for r in tasks.values())/8,finished=time.time())
                atomic(Path(e['output'])/'status.json',result);e.update(status='Completed',result=result)
                self.event('evaluation_completed',id=e['id'],macro=result['macro_success_rate'])
            self.save()

    def launch_evaluations(self):
        preflight=self.folder/'preflight/status.json'
        if not preflight.exists() or json.loads(preflight.read_text()).get('status')!='Completed':return
        active=[e for e in self.state['evaluations'].values() if e['status'] in ('Running','Launching','LaunchUnknown')]
        active += [shard for e in self.state['evaluations'].values() if e['status']=='Sharded' for shard in e['shards'].values() if shard['status'] in ('Running','Launching','LaunchUnknown')]
        if len(active)>=self.state['max_evaluation']:return
        code,output=run(self.ssh+['nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits'])
        if code:raise RuntimeError('ECS connection failed')
        occupied={e['gpu'] for e in active};free=[]
        for line in output.splitlines():
            fields=line.split(',')
            if len(fields)==2:
                gpu,memory=map(int,fields)
                if gpu in self.state.get('evaluation_gpus',list(range(8))) and gpu not in occupied and memory<512:free.append(gpu)
        evaluations=sorted(self.state['evaluations'].values(),key=lambda e:(e['job']!=self.state.get('evaluation_priority'),e['id']))
        priority=self.state.get('evaluation_priority')
        if priority and any(e['job']==priority and e['status']!='Completed' for e in evaluations):evaluations=[e for e in evaluations if e['job']==priority]
        for e in evaluations:
            if not free or len(active)>=self.state['max_evaluation']:break
            if e['status'] not in ('Planned','Sharded'):continue
            job=self.state['jobs'][e['job']]
            if job['status']!='TrainingSucceeded':continue
            source=Path(job['output'])/'source'
            for name in ('encoder/tactile.py','training/actions.py','training/data.py','policy/fresh/deploy_policy.py'):
                if hashlib.sha256((source/name).read_bytes()).digest()!=hashlib.sha256((ROOT/name).read_bytes()).digest():raise RuntimeError(f'Inference preprocessing changed since training: {name}')
            for name,sha in json.loads((Path(job['output'])/'source_sha256.json').read_text()).items():
                if hashlib.sha256((source/name).read_bytes()).hexdigest()!=sha:raise RuntimeError('Trained source changed before evaluation')
            if e['status']=='Planned':
                e.update(status='Sharded',shards={task:dict(status='Planned',output=str(Path(e['output'])/'shards'/task)) for task in SETTINGS['task_step_limits']})
                # Continue valid records from the earlier sequential evaluation.
                for task,shard in e['shards'].items():
                    prior=Path(e['output'])/f'{task}.result.json'
                    if prior.exists():
                        dest=Path(shard['output']);dest.mkdir(parents=True,exist_ok=True);shutil.copy2(prior,dest/prior.name)
                self.save()
            # NAS directories created locally must be writable by the ECS uid/gid.
            for shard in e['shards'].values():
                dest=Path(shard['output']);dest.mkdir(parents=True,exist_ok=True)
                for directory in (dest.parent,dest):
                    if os.geteuid()==0:os.chown(directory,1000,1000)
                    directory.chmod(0o2775)
                for file in dest.glob('*.result.json'):
                    if os.geteuid()==0:os.chown(file,1000,1000)
            for task,shard in e['shards'].items():
                if not free or len(active)>=self.state['max_evaluation']:break
                if shard['status']!='Planned':continue
                gpu=free.pop(0);shard.update(status='Launching',gpu=gpu,attempts=shard.get('attempts',0)+1);self.save()
                argv=['/mnt/yichi_data/miniconda3/envs/univtac5090/bin/univtac-sim','python','-u','-m','training.remote_evaluation',
                      '--checkpoint',e.get('checkpoint',job['checkpoint']),'--output',shard['output'],'--gpu',str(gpu),'--port',str(5557+gpu),
                      '--seed',str(e['seed']),'--episodes','50','--tasks',task]
                out=shlex.quote(shard['output'])
                command=f'set -e; cd {shlex.quote(str(ROOT))}; mkdir -p {out}; nohup '+shlex.join(argv)+f' > {out}/console.log 2>&1 < /dev/null & echo $!'
                try:
                    code,text=run(self.ssh+[command])
                    if code:shard.update(status='LaunchUnknown',error=text[-1500:])
                    else:shard.update(status='Running',remote_pid=int(text.strip().splitlines()[-1]));active.append(shard);self.event('evaluation_shard_launched',id=e['id'],task=task,gpu=gpu)
                except Exception as exc:shard.update(status='LaunchUnknown',error=str(exc))
                self.save()

    def report(self):
        rows=[]
        for key,job in self.state['jobs'].items():
            row=dict(id=key,stage=job['stage'],model=job['model'],training_status=job['status'],job_id=job.get('job_id'),
                     macro_success_rate=self.score(key),checkpoint=job.get('checkpoint'))
            row['cloud_status']=job.get('cloud_status')
            row['step']=job.get('last_metric',{}).get('step')
            timing=job.get('cloud_timing',{});started=timing.get('GmtRunningTime')
            ended=timing.get('GmtFinishTime') or timing.get('GmtFinishedTime')
            row['training_wall_seconds']=None
            if started:
                end=datetime.fromisoformat(ended).timestamp() if ended else time.time()
                row['training_wall_seconds']=end-datetime.fromisoformat(started).timestamp()
            completed=[e for e in self.state['evaluations'].values() if e['job']==key and e['kind']=='selection' and e['status']=='Completed']
            row['evaluation_rounds']=len(completed)
            counts={};successes={};rpc_seconds=0.;rpc_calls=0
            for e in completed:
                for task,r in e['result']['tasks'].items():
                    counts[task]=counts.get(task,0)+r['valid'];successes[task]=successes.get(task,0)+r['successes']
                    for rollout in r.get('rollouts',[]):
                        rpc_seconds+=rollout.get('inference_seconds',0.);rpc_calls+=rollout.get('inference_calls',0)
            row.update({task:successes[task]/counts[task] for task in counts})
            row['rpc_ms_per_chunk']=1000*rpc_seconds/rpc_calls if rpc_calls else None
            rows.append(row)
        atomic(self.folder/'results.json',rows)
        fields=['id','stage','model','training_status','cloud_status','step','job_id','macro_success_rate','checkpoint','training_wall_seconds','evaluation_rounds','rpc_ms_per_chunk']+list(json.loads((ROOT/'training/eval_settings.json').read_text())['task_step_limits'])
        temp=self.folder/'results.csv.tmp'
        with temp.open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
        temp.replace(self.folder/'results.csv')
        final=[]
        for e in self.state['evaluations'].values():
            if e['kind']=='final' and e['status']=='Completed':
                job=self.state['jobs'][e['job']]
                row=dict(id=e['job'],model=job['model'],checkpoint=job['checkpoint'],macro_success_rate=e['result']['macro_success_rate'])
                row.update({task:r['successes']/r['valid'] for task,r in e['result']['tasks'].items()})
                final.append(row)
        atomic(self.folder/'final_results.json',final)

    def poll_preflight(self):
        folder=self.folder/'preflight';status=folder/'status.json';exit_file=folder/'exit_code'
        if not exit_file.exists():return
        result_file=folder/'evaluation/lift_can.result.json'
        result=json.loads(result_file.read_text()) if result_file.exists() else None
        complete=exit_file.read_text().strip()=='0' and result and result.get('complete') and result.get('valid')==1 and result.get('context',{}).get('step_lim')==3
        atomic(status,dict(status='Completed' if complete else 'Failed',smoke_only=True,
                           result=result,checked_at=time.time(),error=None if complete else 'See preflight console and lift_can log; no complete valid rollout'))

    def tick(self):
        if self.state.get('status')=='Paused':return
        self.poll_preflight()
        for job in self.state['jobs'].values():
            if job['status'] in ('Submitting','SubmissionUnknown'):self.reconcile_unknown(job)
        for name,sha in json.loads((self.folder/'frozen_source_sha256.json').read_text()).items():
            if hashlib.sha256((self.folder/'frozen_source'/name).read_bytes()).hexdigest()!=sha:raise RuntimeError('Common experiment source changed')
        if self.state.get('plan_revision')==2:
            for name,sha in json.loads((self.folder/'frozen_source_v2_sha256.json').read_text()).items():
                if hashlib.sha256((self.folder/'frozen_source_v2'/name).read_bytes()).hexdigest()!=sha:raise RuntimeError('V2 experiment source changed')
        self.collect_encoders();self.poll_training();self.poll_evaluations();self.poll_shards();self.advance();self.rebalance_resources();self.submit();self.launch_evaluations();self.report()
        errors=[key for key,j in self.state['jobs'].items() if 'Failed' in j['status'] or j['status']=='SubmissionUnknown']
        errors += [e['id']+'/'+task for e in self.state['evaluations'].values() for task,shard in e.get('shards',{}).items() if shard['status'] in ('Failed','LaunchUnknown')]
        errors += [e['id'] for e in self.state['evaluations'].values() if e['status'] in ('Failed','LaunchUnknown')]
        preflight=self.folder/'preflight/status.json'
        if preflight.exists() and json.loads(preflight.read_text()).get('status')=='Failed':errors.append('ECS_preflight')
        errors += [k for k,v in self.state['stages'].items() if isinstance(v,dict) and v.get('status')=='NeedsAttention']
        self.state['attention']=errors
        self.save()


def main():
    p=argparse.ArgumentParser();p.add_argument('--experiment',required=True);p.add_argument('--interval',type=int,default=60);p.add_argument('--once',action='store_true')
    a=p.parse_args();folder=Path(a.experiment).resolve()
    lock=open(folder/'controller.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    lock.write(str(os.getpid()));lock.flush()
    plan=Plan(folder);atomic(folder/'controller.json',dict(pid=os.getpid(),interval=a.interval,started=time.time()))
    while True:
        try:plan.tick()
        except Exception as exc:
            plan.state['errors']=(plan.state.get('errors',[])+[dict(time=time.time(),error=str(exc))])[-50:]
            plan.save();plan.event('controller_error',error=str(exc))
        if a.once or plan.state['status'] in ('Completed','Paused'):return
        time.sleep(a.interval)


if __name__=='__main__':main()
