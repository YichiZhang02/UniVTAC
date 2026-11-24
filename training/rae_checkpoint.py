"""Portable best-epoch RAE bundles: full model, state dictionaries and source."""
import copy
import hashlib
import json
from pathlib import Path

import torch


def architecture(model):
    return dict(class_name='encoder.rae.RAE', kwargs=dict(variant=model.variant,size=model.size),
                parameter_dtype=str(next(model.parameters()).dtype),
                encoder=dict(type=f'ViT-{model.size}/16', dim=model.dim, depth=12, heads=model.dim//64, patch_size=16,
                             ff_dim=model.dim*4, dropout=0., position='learned_absolute_no_rope',
                             input_frames=model.frames, latent_frames=model.encoder.latent_frames,
                             compression='post_encoder_conv1d_4_to_1' if model.frames==4 else 'none'),
                decoder=dict(dim=256, depth=4, heads=8, ff_dim=1024, patch_size=16,
                             temporal_expansion='conv_transpose1d_1_to_4' if model.frames==4 else 'none'),
                module_tree=str(model),
                parameter_shapes={k:list(v.shape) for k,v in model.state_dict().items()})


def save_checkpoint(path,model,optimizer,*,epoch,step,metrics,spec,config):
    path=Path(path)
    payload=dict(format_version=2,encoder=model.encoder.state_dict(),model=model.state_dict(),
                 architecture=architecture(model),encoder_spec=spec,config=config,
                 optimizer=optimizer.state_dict(),epoch=epoch,step=step,val_metrics=metrics,
                 selection_metric='minimum fixed-val pixel MSE; test unused')
    temporary=path.with_suffix('.tmp')
    torch.save(payload,temporary);temporary.replace(path)


def save_best(output,model,optimizer,*,epoch,step,metrics,spec,config):
    output=Path(output)
    save_checkpoint(output/'best.pt',model,optimizer,epoch=epoch,step=step,metrics=metrics,spec=spec,config=config)
    # Complete encoder+decoder object, including weights. The adjacent source
    # snapshot makes the class importable independently of the live repository.
    full=copy.deepcopy(model).cpu()
    temporary=output/'best_model.tmp'
    torch.save(full,temporary);temporary.replace(output/'best_model.pt')
    description=architecture(full)
    description.update(epoch=epoch,step=step,val_metrics=metrics,encoder_spec=spec)
    (output/'architecture.json').write_text(json.dumps(description,indent=2))
    source=output/'model_source/encoder';source.mkdir(parents=True,exist_ok=True)
    original=Path(__file__).resolve().parents[1]/'encoder/rae.py'
    code=original.read_bytes();(source/'rae.py').write_bytes(code)
    (source/'__init__.py').write_text('')
    info=dict(epoch=epoch,step=step,**metrics,architecture='architecture.json',
              checkpoint='best.pt',full_model='best_model.pt',source='model_source/encoder/rae.py',
              source_sha256=hashlib.sha256(code).hexdigest())
    temporary=output/'best.json.tmp';temporary.write_text(json.dumps(info,indent=2));temporary.replace(output/'best.json')
