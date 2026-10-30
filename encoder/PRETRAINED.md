# Encoder catalog and initialization

Each encoder has its own directory with `model.py`, `train.py`, and `config.yml`.
The public catalog is defined in [catalog.py](catalog.py). `pretrained_mode`
identifies the model family; initialization is selected separately.

| Encoders | Size | Initialization |
| --- | --- | --- |
| ResNet, RVAE, RAE, SD-VAE, MAE, DINOv2, I-JEPA, V-JEPA | S or B | scratch only |
| Sparsh_mae, Sparsh_DINOv2, Sparsh_IJEPA, Sparsh_VJEPA, Wan22_VAE, Anytouch2 | official only | scratch, official, adapted |

Existing DINOv3, SigLIP2 and V-JEPA2 backends remain available with their fixed
upstream architectures and `official` size. Sparsh MAE/IJEPA/VJEPA use their
upstream Small architecture; Sparsh DINOv2 and Anytouch2 use Base. The size tag
`official` selects that fixed architecture, independently of weight initialization.
Selecting S/B for external models is rejected.

- `scratch`: randomly initialize and train on UniVTAC contact data.
- `official`: export upstream weights for downstream use, without contact-data training.
- `adapted`: initialize from upstream weights and fine-tune on UniVTAC contact data.

Local models default to scratch/S. External models default to official/official.
Names identify the family automatically, so `--pretrained-mode` is optional in
`train_encoder.sh`. Historical names and upstream S/B tags remain readable in
saved checkpoints. The former local `VAE` is now `RVAE`; its old import path is a
compatibility shim and its parameter keys are unchanged. Existing result files
are preserved. Explicit legacy checkpoint paths remain usable.

## Local reconstruction models

RVAE preserves the former ViT + residual CNN VAE and its 49 latent tokens.
RAE removes the spatial residual CNN and variational bottleneck: image → ViT →
196×D deterministic patch tokens → Transformer decoder → image. D is 384 for S
and 768 for B. It trains with reconstruction MSE only.

SD-VAE uses Diffusers `AutoencoderKL`, following the
[official SD-VAE architecture configuration](https://huggingface.co/stabilityai/sd-vae-ft-mse/blob/main/config.json).
B uses channel widths 128/256/512/512; S halves them to 64/128/256/256. Both have
two residual layers per stage, bottleneck attention, four latent channels, and
8× spatial compression. The complete S/B autoencoders have approximately 21M/84M
parameters, comparable to the S/B ViT scale. Inputs are 224×224; the exported
encoder returns 784×4 latent means. Training uses pixel MSE + normalized KL.
No official weights are loaded for any local model, including SD-VAE.

```bash
bash train_encoder.sh --method RVAE --size S --init scratch
bash train_encoder.sh --method RAE --size B --init scratch
bash train_encoder.sh --method SD-VAE --size S --init scratch
```

## External training and export

Download upstream weights when using official or adapted initialization:

```bash
python scripts/download_encoder_models.py --methods Sparsh_mae Sparsh_DINOv2 Sparsh_IJEPA Sparsh_VJEPA Wan22_VAE Anytouch2
bash train_encoder.sh --method Sparsh_mae --size official --init scratch
bash train_encoder.sh --method Sparsh_DINOv2 --size official --init official
bash train_encoder.sh --method Sparsh_IJEPA --size official --init adapted
bash train_encoder.sh --method Wan22_VAE --size official --init adapted
```

The upstream asset cache keeps its existing paths under
`resources/pretrained_models/<tactile|vision>/<upstream-method>/<upstream-size>/`.
The downloader records exact revisions and access errors. Sparsh requires the
pinned source installed by `python scripts/setup_encoder_backends.py`.
Scratch initializes without upstream weight files. Wan's fixed architecture
configuration and Anytouch2's model configuration are bundled in their directories.

Every encoder training batch contains 64 anchors. `NUM_PROCESSES` selects DDP
for external training and divides that batch across ranks. Model defaults live
in each model's `config.yml`; trailing arguments override them, for example
`-- --epochs 1 --max-steps 2`. Nonlinear TPS preprocessing keeps its existing
`auto|cpu|cuda` options and the same representation encoding.

Sparsh uses the existing masked reconstruction, DINO distillation, and JEPA
prediction recipes. Wan reconstructs all five observations with reconstruction
+ KL; Anytouch2 uses masked teacher-feature and patch reconstruction. These are
UniVTAC adaptation objectives. Wan's existing compiled five-frame policy encoder
remains the default; `WAN_TRAINING_OPTIMIZATION=reference` disables it.

## Checkpoints and downstream policies

New runs use:

```text
encoder_results/<input>/<mode>/<method>/<size>/<scratch|official|adapted>/<run>/
```

`encoder.pth` stores the selected encoder and constructor metadata. `last.pth`
stores complete resumable training state. RVAE, RAE, SD-VAE and Wan also save
matching full reconstruction weights in `best.pth`. `config.json` records the
file split, initialization, and training parameters. Resume requires the same
model, initialization, input representation, data split and distributed world size.
Official exports have no training state.

```bash
ENCODER_METHOD=RAE ENCODER_SIZE=S ENCODER_PRETRAIN_INIT=scratch bash train_policy.sh
ENCODER_METHOD=Sparsh_mae ENCODER_SIZE=official ENCODER_PRETRAIN_INIT=adapted bash train_policy.sh
ENCODER_METHOD=Wan22_VAE ENCODER_SIZE=official ENCODER_INIT=official bash train_policy.sh
```

`ENCODER_PRETRAIN_INIT` filters checkpoint discovery by initialization.
Scratch/adapted selections require trained checkpoints and never fall back to
raw official weights. `ENCODER_CKPT` selects a specific checkpoint; its model and
initialization are validated. Offline policy restoration uses saved constructor
metadata and embedded encoder weights. RVAE discovery also searches legacy VAE
results. Old artifacts without explicit provenance retain their historical
interpretation; they are never relabeled as adapted automatically.

## Temporal and feature contracts

Temporal encoders other than Wan consume **four observations at stride two**:
`[t-6,t-4,t-2,t]`. Negative episode indices repeat the first observation. History
never crosses trajectory boundaries and resets at deployment reset. Deployment
collects history at every environment action, including actions taken from an
already predicted chunk.
Wan2.2-VAE consumes **five real observations at stride two**, in chronological
order `[t-8,t-6,t-4,t-2,t]`, without inserting a duplicate frame into the clip.

| Encoder | Input per sensor | Implementation |
| --- | --- | --- |
| Local UniVTAC models | `[B,3,H,W]` | Existing single-frame architecture |
| Sparsh MAE / DINOv2 / I-JEPA | `[B,4,3,H,W]` | Official 6-channel two-frame convolution inflated to 12 channels |
| Sparsh V-JEPA | `[B,4,3,H,W]` | Official video encoder with tubelet size 2 |
| AnyTouch2 | `[B,4,3,H,W]` | Official CLIP-based video encoder, GelSight Mini sensor token |
| DINOv3 / SigLIP2 | `[B,3,H,W]` | Official visual encoder |
| V-JEPA2 | `[B,4,3,H,W]` | Official video encoder, predictor omitted from policy |
| Wan2.2 VAE | `[B,5,3,H,W]` | Native 5-frame encoding, then internal 2×2 latent patchify: 98 tokens × 192 dimensions |

Sparsh's 6-channel models already consume two time points. To implement the common
four-frame interface without dropping observations, their input convolution is
initialized with `concat(W,W)/2`. Frames are channel-concatenated newest first,
following upstream ordering. Sparsh video clips are also reversed internally to
the upstream newest-first ordering; the common dataset/deployment API remains
oldest-first. This is an explicit adaptation of the official
weights, rather than an unchanged upstream four-frame model.

RGB inputs to tactile-pretrained models subtract the episode's first tactile image,
add `.5` for Sparsh or `130/255` for AnyTouch2, and clamp to `[0,1]`.
Background subtraction stays in float space after the shared image resizing.
Sparsh rotates UniVTAC's landscape sensor raster clockwise into its upstream
portrait convention before resizing to **224×224** for new encoder exports and
policy training/inference. MAE, DINOv2 and I-JEPA return 196 patch tokens per
sensor; four-frame V-JEPA returns 392 (two temporal slices of 14×14).
The encoder checkpoint records `image_size`; restoring an existing 320×240
checkpoint retains that resolution and its original 300/600-token contract.
Re-export the tactile initializer to use 224×224 for a new policy run.
Policy spatial pooling (`pool`/`pool3x3`, `pool5x5`, `pool7x7`) retains AnyTouch2's
CLS token and excludes its five sensor tokens. Sparsh has no separate CLS token;
its single register token is retained for DINOv2, I-JEPA and V-JEPA, while MAE
has no prefix token. Only spatial patches are pooled, separately for each latent
time slice. At `pool3x3`, per-sensor token counts are MAE 9, DINOv2/I-JEPA 10,
and V-JEPA/AnyTouch2 19. `full` and `cls` output contracts are unchanged.
Depth/deform and marker-only representations
are already encoded signals and do not receive RGB background subtraction.
AnyTouch2 uses its official CLIP normalization, Sparsh uses `[0,1]`, DINOv3 and
V-JEPA2 use ImageNet normalization, and SigLIP2/Wan use `[-1,1]`.
Wan feature latents use the checkpoint's mean/std scaling.
For Wan, native `[B,48,2,14,14]` latents are normalized before lossless spatial
patchify into `[B,98,192]`; both latent time positions are retained. Two sensors
therefore provide 196 tactile tokens, or two pooled tokens in `cls` mode. The
policy's trainable projection consumes 192-dimensional features. Wan adaptation
reconstructs all five real input frames using the original latent tensor, before
patchify. Historical four-frame checkpoints without `wan_patch_size` retain their
48-dimensional/392-token interface and repeated-first-frame adapter when restored;
new checkpoints save `num_frames=5` and `wan_patch_size=2`.

For external encoders, `TACTILE_TYPE=cls` means mean-pooled patch features;
`full` returns patch tokens without register/sensor tokens. AnyTouch2 features use
its learned 512-dimensional projection. ACT and VLA projections are trainable.
All encoder parameters used in inference are registered inside the policy and are
fine-tuned with it; no external encoder is wrapped in `no_grad` or frozen.
Wan's decoder is only registered in its pretraining objective, not in the policy.


## Verification

```bash
python -m unittest scripts.test_encoder_catalog scripts.test_external_pretraining scripts.test_wan_training_fast scripts.test_tactile_pooling
```

The catalog tests check the complete size/initialization matrix, RAE/RVAE/SD-VAE
token counts, reconstruction gradients, encoder restore and downstream full,
CLS and pooled token modes. External tests cover provenance separation, strict
scratch weight loading and representation parity. Wan tests retain value,
gradient, fallback and state-dict checks.
