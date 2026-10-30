# Tactile encoders

Encoder identities now include `pretrained_mode=univtac|tactile|vision`.
See [PRETRAINED.md](PRETRAINED.md) for the model catalog, official downloads,
initialization rules and temporal input contracts, adaptation objectives, tuning results and policy examples.
Local models support scratch initialization only. External models use fixed `official` size and support `scratch`, `official`, or `adapted` initialization. The descriptions below cover the locally implemented `univtac` methods.

The eight local encoder methods are `ResNet`, `RVAE`, `RAE`, `SD-VAE`, `MAE`, `DINOv2`, `I-JEPA`, and `V-JEPA`. Each method has its own `model.py`, `train.py`, and S/B training configuration. ResNet-S is the CNN ResNet18; ResNet-B is Wide ResNet50-2. SD-VAE uses a convolutional KL autoencoder. The other methods use ViT-S/B with 16×16 patches on 224×224 inputs.

| Method | Training forward | ACT `cls` | ACT `full` |
| --- | --- | --- | --- |
| ResNet | CNN spatial map → pooled 512-dimensional vector → image decoder; same-input pixel MSE | 1 × 512 | 64 spatial tokens on ACT's 256×256 input, 512 (S) or 2048 (B) channels |
| RVAE | ViT 196 patches → learned residual 3×3 stride-2 convolution → 49 latent means/log variances → sampled 49 latents → learned residual 7×7-to-14×14 upsampling → 196 reconstructed image patches; pixel MSE + KL | ViT CLS, 1 × 384/768 | 49 latent means, 7×7 × 384/768 |
| RAE | ViT → 196 × D deterministic patch tokens → Transformer pixel decoder; reconstruction MSE, no residual CNN reduction or KL | ViT CLS, 1 × 384/768 | 196 × 384/768 |
| SD-VAE | SD-style residual convolutional encoder → 28×28×4 posterior → convolutional decoder; reconstruction MSE + KL | Mean latent, 1 × 4 | 784 × 4 |
| MAE | Mask 75% of 196 patches → encode 49 visible patches plus CLS → decoder reconstructs masked pixels | Unmasked CLS, 1 × 384/768 | Unmasked CLS + 196 patches, 197 × 384/768 |
| DINOv2 | Two global and one local student views, two global EMA teacher views; CLS distillation, masked-patch iBOT-style distillation, and KoLeo feature regularization | CLS, 1 × 384/768 | CLS + 196 patches, 197 × 384/768 |
| I-JEPA | Single image, two non-overlapping 6×6 target blocks; context encoder sees 124 visible patches and predicts the other 72 with an EMA teacher; the context embedding also receives a Gaussian SIGReg constraint | ViT CLS, 1 × 384/768 | 196 patch tokens, 14×14 × 384/768 |
| V-JEPA\* | Encode one current frame, predict the next-frame EMA target projected CLS latent conditioned on the next absolute 8D joint target; SIGReg regularizes the projected context latent | Projected ViT CLS, 1 × 384/768 | 196 projected patch tokens, 14×14 × 384/768 |

All nine input modes have three channels. `marker_only` renders the first 63 current marker positions as radius-2 white dots on a black canvas, scales coordinates from 320×240 to the requested image size (224×224 by default), and copies the same raster into R, G, and B (`R = G = B`). It returns a float32 tensor of shape `[3, H, W]` in `[0, 1]`; repeating the channels adds no new marker information. Encoder pretraining and policy data loading use the same representation. Existing one-channel `marker_only` encoder checkpoints require retraining or explicit weight conversion before use with this input. ResNet, RVAE, RAE, SD-VAE and MAE reconstruct the selected input mode; for sparse `marker_only` images, foreground pixels receive 50× additional weight in the reconstruction MSE to avoid a blank-image solution. DINOv2, I-JEPA, and V-JEPA use feature-learning objectives. `depth_deform_20`, `depth_deform_80`, `depth_deform_160`, and `depth_deform_240` use `R = clip((30 - camera_depth_mm) / 5, 0, 1)`, identical to `depth_deform_nonlinear`. Raw camera depth is resized before mapping; saved `press_depth` is not used by any of these modes. Linear G/B retain byte-space `128 + 127 * clip(dx_px / U, -1, 1)` (B uses dy), divided by 255 in model tensors; U is 20, 80, 160, or 240 px. Zero displacement is 128/255. The linear modes retain reference-coordinate linear/IDW interpolation. Linear encoding metadata is versioned `linear_camera_30mm_5mm_<U>px_v2`; old 28 mm / 3 mm linear checkpoints require retraining and are rejected by encoding checks. Training uses existing saved markers; inference uses all markers before visibility filtering, noise, shuffling or padding, with stable IDs. Unified evaluation writes per-action reference/current coordinates, IDs and visibility to `marker_traces/<seed>_<result>.hdf5`, including out-of-view positions. Visibility is diagnostic only. New training configs/checkpoints record `depth_deform_encoding`. Retrain encoders and policies for this changed representation; old checkpoint weights are not converted automatically. `depth_marker_to_deform.py` retains the historical conversion for reproducibility. The historical ResNet18 checkpoint format is still accepted by ACT.

The displacement field uses linear interpolation inside the marker convex hull. Outside it, the eight nearest markers contribute with normalized inverse-square-distance weights, so nearby markers contribute more than distant ones.

```bash
./train_encoder.sh --method RVAE --size S --input marker_rgb
ENCODER_METHOD=RVAE ENCODER_SIZE=S INPUT_MODE=marker_rgb TACTILE_TYPE=full ./train_policy.sh
```

Use `TACTILE_TYPE=cls` for one token, or `full` for all method-specific tokens. `train_encoder.sh` loads batch size, learning rate, epochs, and method options from that method's `config.yml`; overrides follow `--`, for example `-- --epochs 1 --max-steps 2`. One A100 GPU is used by default; set `CUDA_VISIBLE_DEVICES` to choose a card. The default training budget is 100,000 tactile frames, split 80/20 by HDF5 trajectory file with seed 42. The same file split is used across model settings; validation is run after each epoch. Reconstruction methods select the lowest-validation-loss encoder. Self-distillation methods can select by frozen-encoder kNN accuracy on held-out object shapes, using labels only for checkpoint selection and never for pretraining gradients. The selected encoder is `encoder.pth`, while `last.pth` retains the latest full training state. A run is saved under `encoder_results/<input_mode>/<pretrained_mode>/<method>/<size>/scratch/<timestamp>/`, containing checkpoints, `metrics.jsonl`, `config.json` with the exact file split, and `train.log`. All run IDs are timestamps. Historical ResNet18 pretraining uses the same directory layout, with its legacy checkpoint format and sample count recorded in `config.json`. `train_policy.sh` selects the latest matching encoder checkpoint unless `ENCODER_CKPT` is set. Historical ablation files are archived outside `encoder_results` in `archive/encoder_ablation/` and are not used for policy checkpoint selection.

The training configs use batch size 64 on one A100 per run, so two independent runs can occupy the two cards. A fixed 8,192-frame, 3-epoch pilot compared batches 64, 256, and 512 across all ten method/size pairs on `marker_rgb`. Batch 64 had the best validation result for every pair; larger batches gave only a small wall-clock speed benefit at equal epochs, and DINOv2-B at 512 ran out of memory. The model-specific epoch limits and early stopping are in each `config.yml`. Smoke runs verify execution and memory, not convergence or downstream policy quality.

I-JEPA adds LeWM/LeJEPA-style sketched isotropic Gaussian regularization (SIGReg) to a learned projection of the pooled context tokens. Each step compares random one-dimensional projections of the image batch against a standard normal distribution using an Epps–Pulley characteristic-function statistic. The existing masked latent prediction and EMA teacher remain in place. Its configuration controls the SIGReg weight, projection dimension, and number of random projections; validation logs include `val_sigreg`, projected feature standard deviation, and teacher feature standard deviation to detect collapse.

V-JEPA uses single-frame `[B,C,H,W]` inputs: the current frame at `t` predicts the EMA target at `t+1` (`future_horizon: 1`). All UniVTAC encoders use `num_frames: 1` and `frame_stride: 1`; four-frame V-JEPA checkpoints require retraining. HDF5 files contain `embodiment/joint` rather than a separate `/action` array, so the action condition is the absolute joint target at `t+1`, columns `:8`. Action normalization uses training trajectories only. Pretraining applies a BatchNorm projector to student and target CLS features, with latent MSE plus SIGReg (weight 0.09). The projector is part of the encoder output: policy `cls` returns the projected CLS, `full` returns 196 projected patch tokens, and `pool` retains the projected CLS and pools the projected patches to 3×3. Pooling follows the projector because its GELU makes the two operations noncommutative. The CLS-only pretraining path and its BatchNorm behavior are unchanged; policy token sequences are flattened to `[B*N,D]` for projector BatchNorm. This is a local action-conditioned JEPA objective, not the official video V-JEPA implementation.

For MAE, DINOv2, I-JEPA, and RVAE, `cls` mode returns the ViT's final normalized CLS token; V-JEPA additionally applies its pretrained projector. RVAE's 49 latent means and I-JEPA's patch tokens remain available in `full` mode. ResNet uses its pooled CNN/FC feature as before. RVAE and I-JEPA CLS tokens receive no direct CLS-specific pretraining loss; existing policy checkpoints trained with pooled features need retraining for the changed `cls` semantics.

For repeated training on the bundled contact data, `scripts/build_encoder_manifest.py` creates `encoder_results/.contact_lengths.json` with HDF5 frame counts. Training uses that index when its file list matches; rebuild it after changing the source HDF5 files. `scripts/run_encoder_grid.py` runs the 48 method/size/input smoke combinations, and `scripts/tune_encoder.py` compares learning rates and batches on the fixed file split.

`depth_deform_nonlinear` shares its depth channel with all four linear modes; its deform mapping and TPS interpolation are unchanged. Its R channel is `clip((30 - camera_depth_mm) / 5, 0, 1)` (30 mm → 0, 25 mm → 1). It always uses raw camera depth, including when `press_depth` is present, so training and testing share the requested 30 mm baseline. G/B encode dx/dy in original sensor pixels: `f(x) = 0.5 + 0.015*x` for `|x| <= 20`, and `f(x) = 0.5 + sign(x)*(0.5 - 0.2/(1+(|x|-20)/(40/3)))` outside. Thus -20/0/+20 map to 0.2/0.5/0.8, with continuous value and slope and rational tails approaching 0/1. Float32 precision can round very large displacements to the endpoints. A zero-smoothing, degree-1 thin-plate spline (TPS) at reference positions reconstructs dense displacement before encoding; fixed-grid weights are cached. Duplicate padding controls within 0.001 sensor pixels are merged by averaging displacement during TPS fitting; raw positions and IDs remain unchanged. `marker_full` takes precedence over filtered `marker`, including coordinates outside the image. Unified evaluation continues to save `marker_full`, stable `marker_id`, and `marker_in_view` every action in `marker_traces/<seed>_<result>.hdf5`, independent of raw policy-input recording. Visibility does not filter TPS inputs. New checkpoints/configs record version `nonlinear_30mm_5mm_tps_v1`; this mode requires its own encoder training.

Train a depth/deform RVAE-S variant with `./train_encoder.sh --method RVAE --size S --input depth_deform_nonlinear -- --early-stop-patience 0` (or select `depth_deform_nonlinear_radial`, `depth_deform_20`, `depth_deform_80`, `depth_deform_160`, or `depth_deform_240`). Each mode has its own checkpoint directory and encoding metadata. `encoder.pth` selects the lowest validation loss across all epochs; `last.pth` stores the final training state. The old `depth_deform` input mode is no longer accepted.

`python scripts/train_depth_deform_vaes.py` launches the six supported depth/deform RVAE-S modes with up to four concurrent runs, at most 5 epochs and validation-loss early stopping after 3 epochs without improvement. Select modes with `--inputs`; override `--epochs`, `--early-stop-patience` (0 disables early stopping), or `--gpus` as needed. Visualization outputs belong under `.cache/` so Git ignores them. `scripts/visualize_key_nonlinear_comparison.py` defaults to `.cache/key_nonlinear_comparison`; `scripts/visualize_linear_vs_tps.py` defaults to `.cache/linear_vs_tps`.

`depth_deform_nonlinear_radial` uses the same depth and full-marker TPS as
`depth_deform_nonlinear`. It maps vectors jointly: `r = hypot(dx, dy)`,
`A(r) = 0.015*r` for `r <= 20`, otherwise
`A(r) = 0.5 - 0.2/(1 + (r-20)/(40/3))`, and
`(G,B) = (0.5,0.5) + A(r)*(dx,dy)/r`. Zero maps to `(0.5,0.5)`.
Direction is preserved because both components share one positive scale.
The linear core is defined by vector length, not by each component separately.
Large amplitudes remain strongly compressed; there is no deadzone or hard cutoff.
The valid G/B domain is a disk of radius 0.5 around `(0.5,0.5)`; decoder outputs
outside that disk need explicit radial projection before physical inversion.
Version `nonlinear_radial_30mm_5mm_tps_v1` distinguishes this mode from original
nonlinear checkpoints. It requires its own encoder and policy training.

The six supported depth/deform modes are `depth_deform_20`, `depth_deform_80`,
`depth_deform_160`, `depth_deform_240`, `depth_deform_nonlinear`, and
`depth_deform_nonlinear_radial`. Retired three-stage code/checkpoints are archived
under `.cache/deprecated_depth_deform_20261015/`, outside active discovery.

Train the radial RVAE-S for exactly five epochs:

```bash
./train_encoder.sh --method RVAE --size S --input depth_deform_nonlinear_radial -- --epochs 5 --early-stop-patience 0
```

Use `TACTILE_INPUT_MODE=depth_deform_nonlinear_radial ENCODER_METHOD=RVAE
ENCODER_SIZE=S` for policy training. `encoder.pth` stores the best encoder;
new RVAE runs also save the corresponding complete encoder/decoder in `best.pth`.
`last.pth` retains the final complete training state.
