"""Atomic per-rollout records; simulator errors do not count as valid episodes."""
import json
from pathlib import Path


def atomic(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2));temp.replace(path)


def load(path, context):
    if path and Path(path).exists():
        value=json.loads(Path(path).read_text())
        if value['context']!=context:raise ValueError('Evaluation context changed; use a new result file')
        return value
    return dict(context=context,rollouts=[],valid=0,successes=0,complete=False)


def validated_complete(result, context):
    """Validate saved episodes before recovering a worker's shutdown failure."""
    if result.get('context')!=context or result.get('complete') is not True:return False
    rows=result.get('rollouts',[])
    if any(r.get('status') not in ('success','failed','error') for r in rows):return False
    seeds=[r.get('seed') for r in rows]
    if any(type(seed) is not int or seed<context['start_seed'] for seed in seeds):return False
    if len(seeds)!=len(set(seeds)):return False
    valid=sum(r['status']!='error' for r in rows)
    successes=sum(r['status']=='success' for r in rows)
    return valid==context['requested']==result.get('valid') and successes==result.get('successes')


def append(path, result, seed, status, **details):
    if status not in ('success','failed','error'):raise ValueError(status)
    if any(r['seed']==seed for r in result['rollouts']):raise ValueError('Duplicate evaluation seed')
    result['rollouts'].append(dict(seed=seed,status=status,**details))
    result['valid']=sum(r['status']!='error' for r in result['rollouts'])
    result['successes']=sum(r['status']=='success' for r in result['rollouts'])
    result['complete']=result['valid']>=result['context']['requested']
    if path:atomic(path,result)
