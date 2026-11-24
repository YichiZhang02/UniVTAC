import hashlib,json
from pathlib import Path
from training.data import TASKS
ROOT=Path('/mnt/yichi_data/UniVTAC_downloads')
def main():
    counts={t:len(list((ROOT/'data/isaac51'/t/'hdf5').glob('*.hdf5'))) for t in TASKS}
    contact=len(list((ROOT/'data/contact').glob('*/hdf5/*.hdf5')))
    assert all(n==100 for n in counts.values()),counts
    assert contact==638,contact
    assert not list((ROOT/'data').rglob('*.incomplete')),'Downloads still running'
    from safetensors import safe_open
    model=ROOT/'models/Qwen3.5-2B/model.safetensors-00001-of-00001.safetensors'
    with safe_open(str(model),framework='pt') as f:assert len(list(f.keys()))>0
    result=dict(isaac51=counts,contact=contact,qwen_bytes=model.stat().st_size)
    (ROOT/'verified.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':main()
