"""Render matching RAE checkpoints on fixed validation clips (CPU by default)."""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/univtac-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from encoder.rae import RAE
from training.encoder_data import CachedFrozenClips


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', default='runs/rae_cached_single64_100epoch_20261116')
    parser.add_argument('--eval-root', default='data/encoder_eval_v1')
    parser.add_argument('--output', default='artifacts/rae_visualization')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    torch.set_num_threads(4)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    ds = CachedFrozenClips(args.eval_root, 'val')
    indices = [0, len(ds)//2, len(ds)-1]
    x = torch.stack([ds[i] for i in indices])
    models = [('vits_rae-1in1', 'ViT-S / 1in1'), ('vits_rae-4in1', 'ViT-S / 4in1'),
              ('vitb_rae-1in1', 'ViT-B / 1in1'), ('vitb_rae-4in1', 'ViT-B / 4in1')]
    predictions, report = [], {'split': 'val', 'indices': indices,
                               'keys': [ds.keys[i] for i in indices], 'models': {}}
    for name, label in models:
        path = Path(args.run_root)/name/'best.pt'
        ck = torch.load(path, map_location='cpu', weights_only=False)
        spec = ck['encoder_spec']
        if spec['representation'] != json.loads((Path(args.eval_root)/'depth_deform_radial.json').read_text())['representation']:
            raise ValueError('Checkpoint representation differs from evaluation cache')
        model = RAE(variant=spec['variant'], size=spec['size'])
        model.load_state_dict(ck['model'], strict=True)
        model.to(args.device).eval()
        with torch.inference_mode():
            y = torch.cat([model(clip[None].to(args.device)).cpu() for clip in x])
        predictions.append(y)
        report['models'][name] = dict(checkpoint=str(path.resolve()), step=ck['step'],
                                     selected_clips_mse=(y-x).square().mean().item())
        print(label, report['models'][name], flush=True)
        del model, ck
    fig, axes = plt.subplots(3, 5, figsize=(13, 8))
    for r, index in enumerate(indices):
        for c, values in enumerate([x]+predictions):
            axes[r, c].imshow(values[r, -1].permute(1, 2, 0).numpy().clip(0, 1))
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
            if r == 0:
                axes[r, c].set_title('Input' if c == 0 else models[c-1][1])
            if c == 0:
                axes[r, c].set_ylabel(f'Val clip {index}, t=3')
    fig.suptitle('RAE validation reconstruction | R=depth, G=radial dx, B=radial dy\n'
                 'Same fixed samples for all models; images displayed in [0, 1]')
    fig.tight_layout()
    fig.savefig(out/'comparison.png', dpi=160)
    plt.close(fig)
    # Separate channels expose errors hidden by composite colors.
    for (name, label), pred in zip(models, predictions):
        fig, axes = plt.subplots(4, 9, figsize=(18, 8))
        for t in range(4):
            for channel, channel_name in enumerate(['Depth', 'Radial dx', 'Radial dy']):
                for kind, image in enumerate([x[0, t, channel], pred[0, t, channel],
                                              (pred-x)[0, t, channel].abs()]):
                    ax = axes[t, channel*3+kind]
                    error = kind == 2
                    ax.imshow(image.numpy(), cmap='magma' if error else 'viridis',
                              vmin=0, vmax=.1 if error else 1)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    if t == 0:
                        ax.set_title(f'{channel_name}\n'+['Input', 'Reconstruction', '|Error| (0-0.1)'][kind], fontsize=9)
                    if channel == kind == 0:
                        ax.set_ylabel(f't={t}')
        fig.suptitle(f'{label} | validation clip {indices[0]} | normalized channel values\n'
                     'Input/reconstruction: 0-1; absolute error: 0-0.1, brighter = larger error')
        fig.tight_layout()
        fig.savefig(out/f'{name}_channels.png', dpi=140)
        plt.close(fig)
    (out/'metadata.json').write_text(json.dumps(report, indent=2))
    print(f'Saved figures to {out.resolve()}', flush=True)


if __name__ == '__main__':
    main()
