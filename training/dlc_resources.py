"""DLC resource admission: A800 first, then A100; account for queued requests."""
import json
import subprocess

POOLS=(('A800','quota3ya3nugiuc6'),('A100','quota4wxjihmxlih'))

def quantities(q):
    def number(value):
        value=str(value)
        for suffix,scale in [('Gi',1),('Mi',1/1024),('Ki',1/1024**2)]:
            if value.endswith(suffix):return float(value[:-2])*scale
        return float(value)
    return {k:number(q[k]) for k in ('GPU','CPU','Memory')}

def capacity(quota,gpus=4):
    if quota.get('Status')!='Ready':return 0
    d=quota['QuotaDetails'];limit=quantities(d['AllocatableQuota'])
    allocated=quantities(d['AllocatedQuota']);submitted=quantities(d['SelfSubmittedQuota'])
    # Include this quota's queued requests, so delayed cloud allocation cannot overbook it.
    used={k:max(allocated[k],submitted[k]) for k in limit}
    return max(0,int(min((limit[k]-used[k])/v for k,v in dict(GPU=gpus,CPU=32,Memory=400).items())))

def snapshot(env=None):
    r=subprocess.run(['aliyun','paistudio','ListQuotas','--region','cn-wulanchabu',
                      '--WorkspaceIds','218916','--Verbose','true','--PageSize','100'],
                     env=env,capture_output=True,text=True,timeout=45)
    if r.returncode:raise RuntimeError('DLC resource query failed: '+r.stderr[:300])
    data=json.loads(r.stdout)
    pools={q['QuotaId']:q for q in data['Quotas']}
    if any(key not in pools for _,key in POOLS):raise RuntimeError('Configured DLC pool missing from workspace')
    return {key:pools[key] for _,key in POOLS}

def preferred(pools,gpus=4):
    return next((key for _,key in POOLS if capacity(pools[key],gpus)>0),None)

def replace_argument(argv,flag,value):
    argv=list(argv);argv[argv.index(flag)+1]=str(value);return argv
