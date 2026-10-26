# Encoder pretraining sources

## GPU TPS preprocessing

All encoder entry points default to `--tactile-preprocessing auto`. Nonlinear
and radial depth/deform inputs use the same batched GPU TPS implementation as
policy training when the model runs on CUDA. Other inputs and CPU runs keep
their existing preprocessing. Force the CPU reference with
`TACTILE_PREPROCESSING=cpu bash train_encoder.sh ...` or
`train_encoder.sh --tactile-preprocessing cpu ...`.

Data workers read depth and markers and solve the small TPS systems. Dense field
evaluation runs on each DDP rank's CUDA device before model forward, including
validation and Wan latent calibration. Local action-conditioned JEPA preserves
its current/future/action contract, and kNN probes use the same preprocessing.
Four-/five-frame clips no longer redundantly convert the current frame once
before reading the temporal clip. Input representation versions do not change.

External encoder metrics now include `data_wait_seconds`,
`data_wait_max_seconds` (the greatest wait across DDP ranks), and `batch_seconds`.
The existing `seconds` field measures the update after the loader yields,
including GPU preprocessing; use `batch_seconds` for complete-step throughput.

Already running DLC jobs use frozen sources. To switch at a completed epoch,
stop the old job and use `scripts/submit_encoder_dlc.py --resume-ledger <old.json>
--methods ... --preprocessing cuda --submit`. This freezes `last.pth` and carries
the model, optimizer, EMA teachers, best encoders, epoch and step into a new run.
Keep the same GPU count (`--gpus` / `--wan-gpus`), sample budget and input mode.

## Random versus official initialization

Explicit `--init scratch|official` on `train_encoder.sh` starts pretraining from
random weights or official weights, respectively. Omitting it retains the legacy
tactile export behavior. `-- --export-only` explicitly exports official weights.
Local `univtac` models support scratch only. The `tactile`/`vision` catalog field
identifies the architecture family; `initialization` records the weight origin.

```bash
CUDA_VISIBLE_DEVICES=0 NUM_PROCESSES=1 bash train_encoder.sh \
  --pretrained-mode tactile --method MAE --init scratch \
  --input depth_deform_nonlinear_radial

CUDA_VISIBLE_DEVICES=0,1,2,3 NUM_PROCESSES=4 bash train_encoder.sh \
  --pretrained-mode vision --method Wan2.2-VAE --init scratch \
  --input depth_deform_nonlinear_radial
```

New external runs are stored at
`encoder_results/<input>/<mode>/<method>/<size>/<scratch|official>/<run>/`.
They save `initialization` in `config.json` and checkpoints. DDP divides the global
batch of 64 across ranks, keeps trajectories together in the train/validation
split, and saves each rank's RNG state for epoch-boundary resume. Resume uses the
same world size, data root, sample count and seed. `last.pth` includes optimizer,
EMA teachers and pretraining heads; `encoder.pth` contains the best encoder only.
Wan additionally saves its matching reconstruction model in `best.pth`.

Sparsh uses the official backbones with local `univtac_sparsh_ssl_v1` objectives:
MAE masked 12-channel patch reconstruction; DINOv2 cross-view register and masked
patch distillation plus KoLeo (1024 prototypes); I-JEPA and V-JEPA prediction of
two non-overlapping 6x6 target blocks using an EMA teacher and upstream predictor.
Video targets span both temporal tube slices. These are four-frame UniVTAC
recipes, not exact upstream pretraining reproductions. No xformers dependency is
required. RGB pretraining applies the same episode-first-image background
subtraction as policy training. Scratch Wan initializes encoder and decoder from
configuration without loading weights; latent mean/std start at zero/one and are
recalibrated on training-split posterior modes after every epoch.

Policy selection uses a separate initialization selector:

```bash
ENCODER_PRETRAINED_MODE=tactile ENCODER_METHOD=MAE ENCODER_SIZE=S \
ENCODER_PRETRAIN_INIT=scratch TACTILE_MODE=encode \
TACTILE_INPUT_MODE=depth_deform_nonlinear_radial bash train_policy.sh
```

Use `ENCODER_PRETRAIN_INIT=official` for encoders initialized from official weights.
The latest checkpoint is selected within that initialization only. Explicit
`ENCODER_CKPT` is checked against the requested initialization. Scratch never falls
back to official weights. `ENCODER_INIT=scratch` is a shorthand for scratch
selection; the historical `ENCODER_INIT=official` still means raw official weights
without contact-data pretraining. Defaults remain scratch for local models and
official for external models. Policy run names and configs record the selection.
Historical external checkpoints without an initialization field are interpreted
as official; historical local checkpoints are interpreted as scratch.

`scripts/submit_encoder_dlc.py --submit` submits scratch pretraining for
MAE-S, DINOv2-B, I-JEPA-S and V-JEPA-S on one GPU each, and Wan2.2-VAE-B on four
GPUs (16 anchors per GPU), using the existing DLC mount,
image and quota. It freezes the encoder training sources and records job IDs and
commands under `encoder_results/dlc_pretraining_<timestamp>/`. Inspect with
`--ledger <submissions.json> --status`; repeating with the same ledger does not
resubmit attempted jobs. Defaults: 100,000 anchors, seed 42, radial input, at most
5 epochs for every model, early stopping after 2 unimproved epochs.
`external_pretraining.yml` contains the initial training settings.

Verification:

```bash
python -m unittest scripts.test_external_pretraining
python scripts/verify_external_pretraining.py --method MAE --init scratch
python scripts/verify_external_pretraining.py --method Wan2.2-VAE --init official
```

The GPU check verifies SSL backward, gradients through every frame, parameter
updates, encoder export/reload and downstream pooled-token projection.

An encoder is identified by **representation / pretrained mode / method / size /
initialization**. The three modes describe architecture provenance; `initialization`
separately records scratch or official weights. All modes produce tactile features
when used by a policy.

All UniVTAC encoder training uses **batch size 64**, including vision adaptation
and parameter searches. Every training batch contains exactly 64 anchors;
incomplete training batches are dropped. Validation may retain its final partial
batch. This does not change the policy training batch size. A non-64 encoder CLI
or configuration override is rejected; there is no automatic smaller-batch fallback.
For external encoder adaptation, `--multi-gpu` splits that same global batch
across all visible CUDA devices (two GPUs: 32 anchors per GPU). Validation losses
are weighted by each replica's sample count. Checkpoints retain the same keys
as single-GPU training and can be restored without DataParallel.
Wan adaptation recomputes residual-block activations during backward to reduce
memory. Each recomputation receives an isolated snapshot of the causal feature
cache; cache updates from later chunks are never replayed as earlier inputs.
This training-only mechanism preserves encoder parameter names and policy APIs.

The local action-conditioned method is displayed as **V-JEPA\*** to distinguish
it from official V-JEPA. Its CLI, checkpoint and directory identifier remains
`V-JEPA` for compatibility.

| Mode | Methods | Default downloaded sizes |
| --- | --- | --- |
| `univtac` | ResNet, VAE, MAE, DINOv2, I-JEPA, V-JEPA\* | S; B also supported |
| `tactile` | Sparsh MAE, DINOv2, I-JEPA, V-JEPA; AnyTouch2 | S / B / S / S / B |
| `vision` | DINOv3, SigLIP2, V-JEPA2, Wan2.2-VAE | S / B / L / B |

Within `tactile`, use `METHOD=MAE`, not `Sparsh MAE`: the mode already identifies
the official Sparsh source. V-JEPA2's smallest official release is Large.
Wan's `B` is a repository size tag for the TI2V-5B model's **VAE only**, not a
ViT-Base or the complete 5B diffusion model. Selecting an unavailable size raises
an error; there is no random-weight fallback.

## Download and setup

Activate the project's `univtac` environment and install `requirements_train.txt`.
Commands below assume its Python is active; alternatively set `PYTHON_BIN` for
the shell launchers.

```bash
python scripts/download_encoder_models.py
```

Official initialization weights are saved under:

```text
resources/pretrained_models/{tactile,vision}/{method}/{size}/
```

The downloader records repository revisions and errors in
`resources/pretrained_models/encoder_downloads.json`; each successful directory
also has `source.json`. It installs the pinned official Sparsh source under
`resources/third_party/sparsh`. Reinstall source independently with
`python scripts/setup_encoder_backends.py`.

Official sources:
[Sparsh](https://github.com/facebookresearch/sparsh),
[AnyTouch2](https://github.com/GeWu-Lab/AnyTouch2),
[DINOv3](https://github.com/facebookresearch/dinov3),
[SigLIP2](https://huggingface.co/google/siglip2-base-patch16-224),
[V-JEPA2](https://github.com/facebookresearch/vjepa2),
[Wan2.2](https://github.com/Wan-Video/Wan2.2).

AnyTouch2 and DINOv3 require upstream access approval and Hugging Face login.
After obtaining access, run `hf auth login` in the environment and retry:

```bash
python scripts/download_encoder_models.py --methods AnyTouch2 DINOv3
```

DINOv3 Small is also available from the ModelScope repository
`facebook/dinov3-vits16-pretrain-lvd1689m`. Its `config.json` and
`model.safetensors` already use Hugging Face format; no weight conversion is
needed. Download that snapshot into `resources/pretrained_models/vision/DINOv3/S`
and load it with the existing Transformers adapter. For environments with an
unreachable internal OSS endpoint, unset `INTRA_CLOUD_ACCELERATION_REGION` and
`MODELSCOPE_DOWNLOAD_INTRA_CLOUD_REGION` for the download process.
The local snapshot's `source.json` records the download platform and SHA256.

No token should be put in this repository or a command saved to the README.

## Checkpoint layout and compatibility

```text
encoder_results/<representation>/<univtac|tactile|vision>/<method>/<size>/<run>/
    encoder.pth
    config.json
    metrics.jsonl       # training runs
    last.pth            # complete training state when enabled
```

Checkpoint metadata includes `pretrained_mode`, `method`, `size`, `num_frames`,
`frame_stride`, RGB convention and representation version. External models also
persist constructor configuration and upstream repository/revision.

Existing standalone encoders were moved into the `univtac` subtree. Old standalone
V-JEPA runs were deleted, because the new architecture jointly encodes four frames.
Policy checkpoints were not deleted: they contain their encoder weights, and the
old V-JEPA single-frame constructor remains available for restoring them.
Missing source metadata in historical local checkpoints defaults to `univtac`.

The migration is reproducible and idempotent:

```bash
python scripts/migrate_encoder_layout.py          # inspect remaining legacy paths
python scripts/migrate_encoder_layout.py --apply  # migrate those paths
```

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
| UniVTAC except V-JEPA | `[B,3,H,W]` | Existing single-frame architecture |
| UniVTAC V-JEPA\* | `[B,4,3,H,W]` | Joint attention over 784 patches and CLS, learned temporal positions |
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

## Training entry points

The requested encoder controls are:

```bash
PRETRAINED_MODE="${PRETRAINED_MODE:-univtac}" # univtac | tactile | vision
METHOD="${ENCODER_METHOD:-ResNet}"
```

Examples:

```bash
# Local four-frame JEPA, trained from scratch.
PRETRAINED_MODE=univtac ENCODER_METHOD=V-JEPA INPUT_MODE=depth_deform_nonlinear_radial bash train_encoder.sh

# Export official Sparsh weights as a representation-specific policy initializer.
PRETRAINED_MODE=tactile ENCODER_METHOD=DINOv2 INPUT_MODE=marker_rgb bash train_encoder.sh

# Adapt a visual model on UniVTAC tactile data using the selected common parameters.
PRETRAINED_MODE=vision ENCODER_METHOD=SigLIP2 INPUT_MODE=depth_deform_nonlinear_radial bash train_encoder.sh
```

Sizes default to the smallest available official model. `--size` / `ENCODER_SIZE`
can select supported local sizes. Existing `--method`, `--input`, `--config`,
`--data-root` and trailing training overrides remain available.

Vision adaptation can resume from `last.pth` with `--resume`; keep the same data
root, input representation, sample budget and seed. Pass a larger `--epochs` to
extend a completed run. The optimizer, EMA teacher, heads and RNG states are
restored. Newly written encoder snapshots are published atomically so a concurrent
policy launcher does not read an incomplete archive.

The policy controls are:

```bash
ENCODER_PRETRAINED_MODE="${ENCODER_PRETRAINED_MODE:-univtac}"
ENCODER_METHOD="${ENCODER_METHOD:-VAE}"
```

```bash
POLICY=act ENCODER_PRETRAINED_MODE=tactile ENCODER_METHOD=MAE \
  TACTILE_INPUT_MODE=depth_deform_nonlinear_radial bash train_policy.sh

POLICY=starvla_groot ENCODER_PRETRAINED_MODE=vision ENCODER_METHOD=SigLIP2 \
  TACTILE_INPUT_MODE=depth_deform_nonlinear_radial bash train_policy.sh
```

The launcher finds `encoder.pth` inside the matching representation/mode/method/size
subtree. For `tactile`, when no exported checkpoint exists, rank zero exports the
official initializer automatically. For `vision`, first run adaptation on the
desired representation. `ENCODER_CKPT` selects a specific run explicitly.
Policy configurations contain complete encoder constructor metadata; offline
restoration does not need the original encoder checkpoint or upstream weight files.
The corresponding Python packages and pinned Sparsh source must remain installed.
N0VTLA retains its separate native tactile architecture and does not use this encoder registry.

## Vision adaptation and parameter selection

The `vision` group starts from official visual/video weights and adapts them on
UniVTAC **tactile representations**. DINOv3, SigLIP2 and V-JEPA2 use a local masked
teacher-feature prediction objective plus RGB patch-mean reconstruction; the teacher
is an EMA copy initialized from upstream weights. No text pairs are required.
These are explicitly UniVTAC adaptation objectives, not reproductions of the
upstream DINO, SigLIP image/text or V-JEPA2 pretraining recipes.
Wan uses its pretrained decoder for pixel reconstruction plus KL regularization.

```bash
python scripts/tune_vision_encoders.py --gpus 0 1
```

To dedicate a pair of GPUs to one Wan search, pass a comma-separated GPU group:

```bash
python scripts/tune_vision_encoders.py --methods Wan2.2-VAE --gpus 0,1
# Or launch adaptation directly; the global batch remains 64.
CUDA_VISIBLE_DEVICES=0,1 PRETRAINED_MODE=vision ENCODER_METHOD=Wan2.2-VAE \
  bash train_encoder.sh -- --multi-gpu
```
With a normal `--gpus 0 1` search, other models run in separate GPU queues first;
Wan then uses both GPUs together after those queues finish.

The search uses only `depth_deform_nonlinear_radial`, seed 42, the same file-level
80/20 split as local pretraining, 8,192 sampled anchors, three epochs, at most
64 training and 32 validation batches per epoch. It compares learning rates
`1e-5,3e-5,1e-4` at weight decay `.05`, then compares `.01` at the winning learning
rate. All trials now use batch size 64 for every encoder. The earlier 8/4/2 trials
are historical results and do not establish parameter choices at batch size 64.
Selection uses each model's validation objective; losses are not comparable across
different architectures/objectives. This is a bounded parameter-selection pilot,
not a downstream success-rate evaluation or evidence of full convergence.

Completed selections are saved to `encoder/vision_config.yml` and reused by all
input representations without retuning. Full adaptation defaults to 100,000 anchors,
five epochs and validation-loss early stopping after three unimproved epochs.
Search reports and per-trial logs are under `encoder_results/vision_tuning_*/`.
Each successful selection exports its winning pilot weights to a `*_selected` run.
Models with missing official weights are reported as blocked and receive no claimed
parameter selection. Until a model's search finishes, its configuration entry is
a starting point rather than a validated parameter choice.

### Local results, 2026-10-19

All nine official initializers are available locally. AnyTouch2 was copied from
the existing StarVTLA snapshot and its SHA256 matched the original Hugging Face
download metadata. DINOv3 Small was downloaded from ModelScope in native Hugging
Face format and its SHA256 matched the source LFS content address. Both passed
real-weight forward and gradient checks.

Completed batch-64 searches (2026-10-19, 13:25 UTC):

| Model | Learning rate | Weight decay | Training batch | Batch-64 result |
| --- | --- | --- | --- | --- |
| SigLIP2-B | 1e-4 | .01 | 64 | Selected; validation loss .02374044 |
| V-JEPA2-L | 1e-4 | .01 | 64 | Selected; validation loss .06980791 |
| Wan2.2-VAE | 3e-5 | .01 | 64 globally, 32/GPU | Selected; validation loss .00006156318 |
| DINOv3-S | 1e-4 | .01 | 64 | Selected; validation loss .06535716 |

The historical 8/4/2 trials remain in their original reports, before the fixed-64 requirement.
They do not replace the new batch-64 selections. The current completed reports are
`encoder_results/vision_tuning_20261019-093043/report.json` (SigLIP2 and V-JEPA2;
its single-GPU Wan trials failed with out-of-memory) and
`encoder_results/vision_tuning_20261019-110052/report.json` (DINOv3).
The successful dual-GPU Wan report is
`encoder_results/vision_tuning_20261019-112330/report.json`. All four current
trials completed successfully for each model. The combined report is
`encoder_results/vision_batch64_summary.json`. These per-model parameters are
shared across representations; Wan's configuration also enables `multi_gpu`.
Make both devices visible when launching Wan, for example
`CUDA_VISIBLE_DEVICES=0,1 PRETRAINED_MODE=vision ENCODER_METHOD=Wan2.2-VAE bash train_encoder.sh`.
Selected checkpoints are in the corresponding representation/mode/method/size
`*_selected` directories.
Parameter selection uses adaptation loss, not manipulation success rate.
The original nominal-batch-64 V-JEPA\* run used variable-sized file-chunk batches
and was stopped when the exact-64 requirement was introduced. Its historical run is
`encoder_results/depth_deform_nonlinear_radial/univtac/V-JEPA/S/20261019-085237-737859/`.
The sampler now fills batches across file chunks to exactly 64 anchors. A new
100,000-anchor V-JEPA\* run was launched on GPU 1; find the latest run's `train.log`
for current progress. The completed image/video searches used GPU 0. Wan now uses
GPUs 0 and 1 with residual activation recomputation, sharing GPU 1 with local
V-JEPA training; global training batches still contain exactly 64 anchors.

## Verification

```bash
python -m unittest scripts.test_pretrained_encoders scripts.test_depth_deform_pipeline scripts.test_policy_input_recording
```

The temporal tests cover causal indexing, first-frame padding, pretraining/policy/
deployment parity, history collection during queued actions, reset isolation,
dependence on earlier frames and historical V-JEPA constructor restoration.
Wan-specific checks cover five-frame history parity, invertible latent patchify,
all-five-frame gradients, reconstruction backward, old 48-D policy restoration,
sample-weighted multi-GPU losses and causal-cache recomputation gradient equality.
Two-GPU global-batch-64 optimizer and real-data training/validation/save checks
passed; a synthetic step peaked at approximately 65.3 GiB allocated per device.
Real-weight smoke checks additionally exercise gradients, ACT/VLA encoder adapter
round trips and a short policy-training/checkpoint/deployment cycle. Smoke and
pilot runs are not task-performance measurements.
