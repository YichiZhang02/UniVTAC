"""Measure complete StarVLA training steps on a fixed, synthetic RGB batch.

Includes VLM, action head, backward, clipping and AdamW; excludes data loading,
validation and checkpoint export. Run each variant in a fresh process.
"""
import argparse
import json
import os
from pathlib import Path
import random
import sys
import time

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--policy", choices=("starvla_groot", "starvla_pi"), required=True)
    p.add_argument("--variant", choices=("baseline", "optimized"), default="baseline")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--expert-width", type=int, help="Override the action expert width for comparisons")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--check-features", action="store_true")
    p.add_argument("--check-vision", action="store_true")
    p.add_argument("--compile-head", action="store_true")
    p.add_argument("--profile", action="store_true")
    p.add_argument("--tactile-config", type=Path, help="Saved train_config.yml supplying the tactile encoder and temporal contract")
    p.add_argument("--tf32", action="store_true")
    p.add_argument("--ddp-static", action="store_true", help="Use DDP static_graph instead of per-step unused-parameter traversal")
    args = p.parse_args()
    if args.steps < 1 or args.warmup < 1:
        p.error("steps and warmup must be positive")
    torch.set_num_threads(4)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    if world_size > 1:
        args.device = f"cuda:{os.environ['LOCAL_RANK']}"
        torch.distributed.init_process_group("nccl")
    torch.cuda.set_device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    torch.manual_seed(42)
    random.seed(42)
    from policy.vla_common.engine.configs import FeatureType, PolicyFeature
    from policy.starvla_groot.configuration_starvla_groot import StarvlaGrootConfig
    from policy.starvla_groot.modeling_starvla_groot import StarvlaGrootPolicy
    from policy.starvla_pi.configuration_starvla_pi import StarvlaPiConfig
    from policy.starvla_pi.modeling_starvla_pi import StarvlaPiPolicy
    cls, model_cls = ((StarvlaGrootConfig, StarvlaGrootPolicy) if args.policy == "starvla_groot"
                      else (StarvlaPiConfig, StarvlaPiPolicy))
    base = yaml.safe_load((ROOT / "policy" / args.policy / "base_config.yml").read_text())
    cams = ["observation.images.cam_high", "observation.images.cam_wrist"]
    base.update(base_vlm=str(ROOT / "resources/pretrained_models/Qwen3.5-2B"),
                training_batch_vision_attention=args.variant != "baseline",
                device=args.device, chunk_size=50, n_action_steps=30, tactile_mode="none",
                top_camera_keys=cams[:1], wrist_camera_keys=cams[1:],
                state_mode="absolute_joint", action_dim=8, state_dim=8,
                input_features={**{k: PolicyFeature(FeatureType.VISUAL, (3, 224, 224)) for k in cams},
                                "observation.state": PolicyFeature(FeatureType.STATE, (8,))},
                output_features={"action": PolicyFeature(FeatureType.ACTION, (8,))})
    if args.expert_width is not None:
        base["action_expert_hidden_size"] = args.expert_width
    tactile_meta = None
    if args.tactile_config:
        saved = yaml.safe_load(args.tactile_config.read_text())
        tactile_meta = saved["tactile_encoder_config"]
        keys = ["observation.images.tac_left", "observation.images.tac_right"]
        base.update(tactile_mode="encode", tactile_input_mode=tactile_meta["input_mode"],
                    tactile_keys=keys, tactile_num_frames=tactile_meta.get("num_frames", 1),
                    tactile_frame_offset=tactile_meta.get("frame_stride", 1),
                    tactile_encoder_pretrained_mode=tactile_meta["pretrained_mode"],
                    tactile_type=tactile_meta["tactile_type"],
                    tactile_insert_location=saved["routing"]["tactile_insert_location"],
                    tactile_encoder_path=saved["routing"]["encoder_ckpt"],
                    tactile_encoder_config=tactile_meta)
        base["input_features"].update({k: PolicyFeature(FeatureType.VISUAL, (3, 224, 224)) for k in keys})
    config = cls(**base)
    model = model_cls(config).train()
    if args.compile_head:
        if args.policy == "starvla_pi":
            model.action_head._layerwise_dit = torch.compile(model.action_head._layerwise_dit)
        else:
            model.action_head.model.forward = torch.compile(model.action_head.model.forward)
    if args.variant == "baseline":
        # Reproduce the original unused vocabulary projection and cache defaults.
        def generation_forward(**kwargs):
            if args.policy == "starvla_groot":
                kwargs.pop("use_cache", None)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                return model.qwen_vl.model(**kwargs)
        model.qwen_vl.forward = generation_forward
    params = list(model.get_optim_params())
    optim = torch.optim.AdamW(params, lr=config.optimizer_lr, betas=tuple(config.optimizer_betas),
                              eps=config.optimizer_eps, weight_decay=config.optimizer_weight_decay,
                              fused=args.variant != "baseline")
    b = args.batch_size
    batch = {**{k: torch.rand(b, 3, 224, 224, device=args.device) for k in cams},
             "observation.state": torch.randn(b, 8, device=args.device),
             "action": torch.randn(b, 50, 8, device=args.device),
             "is_pad": torch.zeros(b, 50, dtype=torch.bool, device=args.device),
             "task": ["Lift the can."] * b}
    if tactile_meta:
        frames = tactile_meta.get("num_frames", 1)
        shape = (b, frames, 3, 224, 224) if frames > 1 else (b, 3, 224, 224)
        batch.update({k: torch.rand(*shape, device=args.device) for k in keys})
    if args.check_vision:
        attentions = [block.attn for block in model.qwen_vl.model.model.visual.blocks]
        patched = [attention.forward for attention in attentions]
        for attention in attentions:
            attention.forward = attention._univtac_original_forward
        torch.manual_seed(123)
        reference_loss = model(batch, return_loss_dict=False)[0]
        reference_loss.backward()
        reference_gradients = {name: parameter.grad.clone() for name, parameter in model.named_parameters()
                               if parameter.grad is not None}
        def compare_gradients():
            comparisons = {}
            for prefix in ("qwen_vl", "qwen_vl.model.model.visual", "qwen_vl.model.model.language_model", "action_head"):
                squared_error, squared_reference, max_error, equal, count = 0.0, 0.0, 0.0, 0, 0
                for name, parameter in model.named_parameters():
                    if not name.startswith(prefix) or name not in reference_gradients:
                        continue
                    actual, expected = parameter.grad.float(), reference_gradients[name].float()
                    delta = actual - expected
                    squared_error += delta.square().sum().item()
                    squared_reference += expected.square().sum().item()
                    max_error = max(max_error, delta.abs().max().item())
                    equal += int(torch.equal(actual, expected))
                    count += 1
                comparisons[prefix] = dict(relative_l2=(squared_error / max(squared_reference, 1e-30))**0.5,
                                           max_abs=max_error, exact_tensors=equal, tensors=count)
            return comparisons
        # Control for non-deterministic CUDA backward: repeat the original path
        # with the exact same weights and random draws before changing SDPA.
        model.zero_grad(set_to_none=True)
        torch.manual_seed(123)
        control_loss = model(batch, return_loss_dict=False)[0]
        control_loss.backward()
        control_gradients = compare_gradients()
        for attention, forward in zip(attentions, patched):
            attention.forward = forward
        model.zero_grad(set_to_none=True)
        torch.manual_seed(123)
        loss = model(batch, return_loss_dict=False)[0]
        loss.backward()
        comparisons = compare_gradients()
        result = dict(policy=args.policy, batch_size=b, head_autocast="float32", tf32=False,
                      baseline_loss=reference_loss.item(), batched_loss=loss.item(),
                      loss_abs_difference=abs(loss.item() - reference_loss.item()), gradients=comparisons,
                      control_loss=control_loss.item(), control_gradients=control_gradients,
                      parameters={name: sum(p.numel() for p in module.parameters())
                                  for name, module in (("vlm", model.qwen_vl), ("action_head", model.action_head))})
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2), flush=True)
        return
    if args.check_features:
        images = torch.stack([batch[k] for k in cams], dim=1)
        inputs = model.qwen_vl.build_qwenvl_inputs(images, batch["task"])
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            expected = model.qwen_vl.model(**inputs, output_hidden_states=True, use_cache=False)
            actual = model.qwen_vl.model.model(**inputs, output_hidden_states=True, use_cache=False)
        for i, (x, y) in enumerate(zip(expected.hidden_states, actual.hidden_states)):
            print(f"layer={i} wrapper={x.dtype} core={y.dtype} max_diff={(x.float()-y.float()).abs().max().item()}", flush=True)
            torch.testing.assert_close(x, y, rtol=0, atol=0, check_dtype=True)
        print("Feature parity passed", flush=True)
        return
    core_model = model
    if world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[int(os.environ['LOCAL_RANK'])], broadcast_buffers=False,
            gradient_as_bucket_view=True, find_unused_parameters=not args.ddp_static,
            static_graph=args.ddp_static)
        args.output = args.output.with_name(args.output.stem + f".rank{rank}" + args.output.suffix)
    records, losses = [], []
    torch.cuda.reset_peak_memory_stats()
    for step in range(args.warmup + args.steps):
        torch.cuda.synchronize()
        start = time.perf_counter()
        events = [torch.cuda.Event(enable_timing=True) for _ in range(5)]
        optim.zero_grad(set_to_none=True)
        events[0].record()
        kwargs = {} if args.variant == "baseline" else {"return_loss_dict": False}
        loss = model(batch, **kwargs)[0]
        events[1].record()
        loss.backward()
        events[2].record()
        norm = torch.nn.utils.clip_grad_norm_(params, config.optimizer_grad_clip_norm)
        events[3].record()
        optim.step()
        events[4].record()
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) * 1000
        value, grad_norm = loss.item(), norm.item()
        if not (torch.isfinite(loss).item() and torch.isfinite(norm).item()):
            raise RuntimeError(f"Non-finite loss/gradient at step {step}")
        print(f"step={step} ms={elapsed:.1f} loss={value:.6f} grad_norm={grad_norm:.4f}", flush=True)
        losses.append(value)
        if step >= args.warmup:
            records.append(dict(step_ms=elapsed, forward_ms=events[0].elapsed_time(events[1]),
                                backward_ms=events[1].elapsed_time(events[2]),
                                clip_ms=events[2].elapsed_time(events[3]),
                                optimizer_ms=events[3].elapsed_time(events[4])))
    averages = {k: sum(row[k] for row in records) / len(records) for k in records[0]}
    result = dict(policy=args.policy, variant=args.variant, compile_head=args.compile_head,
                  head_autocast="float32", tf32=args.tf32, batch_size=b, views=2, chunk_size=50,
                  world_size=world_size, rank=rank, ddp_static=args.ddp_static,
                  tactile_config=str(args.tactile_config) if args.tactile_config else None,
                  action_expert_hidden_size=config.action_expert_hidden_size,
                  action_head_parameters=sum(p.numel() for p in core_model.action_head.parameters()),
                  gpu=torch.cuda.get_device_name(), torch=torch.__version__, warmup=args.warmup,
                  steps=args.steps, means=averages, samples_per_second=1000 * b / averages["step_ms"],
                  peak_memory_gib=torch.cuda.max_memory_allocated() / 2**30,
                  losses=losses, records=records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("records", "losses")}), flush=True)
    if args.profile:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                               torch.profiler.ProfilerActivity.CUDA]) as prof:
            optim.zero_grad(set_to_none=True)
            loss = model(batch, return_loss_dict=False)[0]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, config.optimizer_grad_clip_norm)
            optim.step()
            torch.cuda.synchronize()
        print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=25), flush=True)
        prof.export_chrome_trace(str(args.output.with_suffix(".trace.json")))
    if world_size > 1:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
