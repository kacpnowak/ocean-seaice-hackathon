#!/usr/bin/env python
"""Measure what a training step actually costs, for every size preset.

Nobody should guess `max_steps`.  This script builds each shipped preset from
the real configs, runs a real training step (forward, backward, optimiser) at
the cluster's batch size and precision, and reports parameters, step time, peak
memory and the wall clock that implies.  It also runs the shape checks that the
whole pipeline depends on -- in particular the surface-only component, which is
the configuration most likely to break.

    # everything, on one GH200
    srun --account=hclimrep --partition=booster --gres=gpu:1 --time=00:30:00 --pty \
        .venv/bin/python scripts/benchmark_step.py

    # just the 30-minute model, including real data loading
    .venv/bin/python scripts/benchmark_step.py --presets tiny --data

Without a GPU it still runs (slowly) on the CPU and says so; the memory column
is then meaningless and is left blank.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hydra import compose, initialize_config_dir  # noqa: E402
from hydra.utils import instantiate  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from tensordict.tensordict import TensorDict  # noqa: E402

from oceanarches.dataloaders.variables import N_LAT, N_LON  # noqa: E402

CONFIG_DIR = str(Path(__file__).resolve().parent.parent / "configs")
PRESETS = ["tiny", "small", "base", "large"]

PRECISIONS = {
    "bf16-mixed": torch.bfloat16,
    "16-mixed": torch.float16,
    "32-true": torch.float32,
}


# ---------------------------------------------------------------------------
def build_cfg(preset: str, dataloader: str, cluster: str, extra: list[str] | None = None):
    overrides = [f"module={preset}", f"dataloader={dataloader}", f"cluster={cluster}"]
    overrides += extra or []
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR, job_name="benchmark"):
        cfg = compose(config_name="config", overrides=overrides)
    OmegaConf.resolve(cfg)
    return cfg


def random_state(cfg, batch: int, device, generator=None) -> TensorDict:
    """A batch shaped exactly like what GlorysForecast hands the model."""
    fields = {
        "surface": torch.randn(
            batch, cfg.dataloader.n_surface_in, 1, N_LAT, N_LON, device=device, generator=generator
        )
    }
    if cfg.dataloader.n_level_in:
        fields["level"] = torch.randn(
            batch,
            cfg.dataloader.n_level_in,
            cfg.module.n_depths,
            N_LAT,
            N_LON,
            device=device,
            generator=generator,
        )
    return TensorDict(fields, batch_size=batch)


def target_like(pred: TensorDict) -> TensorDict:
    return TensorDict({k: torch.randn_like(v) for k, v in pred.items()}, batch_size=pred.shape[0])


def parameter_counts(embedder, backbone) -> tuple[int, int]:
    return (
        sum(p.numel() for p in embedder.parameters()),
        sum(p.numel() for p in backbone.parameters()),
    )


# ---------------------------------------------------------------------------
def benchmark(cfg, batch: int, steps: int, warmup: int, device, dtype) -> dict:
    """One preset: params, seconds per training step, peak memory."""
    embedder = instantiate(cfg.module.embedder).to(device)
    backbone = instantiate(cfg.module.backbone).to(device)
    n_embedder, n_backbone = parameter_counts(embedder, backbone)

    optimiser = torch.optim.AdamW(
        list(embedder.parameters()) + list(backbone.parameters()), lr=1e-4
    )
    state = random_state(cfg, batch, device)
    prev = random_state(cfg, batch, device)
    cond = torch.randn(batch, cfg.module.backbone.cond_dim, device=device)
    autocast = torch.autocast(device.type, dtype=dtype, enabled=(dtype != torch.float32))

    shapes: dict[str, tuple] = {}
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    target = None
    times: list[float] = []
    for step in range(warmup + steps):
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()

        with autocast:
            tokens = embedder.encode(state, prev)
            pred = embedder.decode(backbone(tokens, cond))
            if target is None:
                target = target_like(pred)
                shapes = {
                    "tokens": tuple(tokens.shape),
                    **{f"out.{k}": tuple(v.shape) for k, v in pred.items()},
                }
            loss = sum(((pred - target) ** 2).mean().values())
        loss.backward()
        optimiser.step()
        optimiser.zero_grad(set_to_none=True)

        if device.type == "cuda":
            torch.cuda.synchronize()
        if step >= warmup:
            times.append(time.perf_counter() - start)

    times.sort()
    return dict(
        params_embedder=n_embedder,
        params_backbone=n_backbone,
        params_total=n_embedder + n_backbone,
        step_seconds=times[len(times) // 2],  # median: robust to a stray page fault
        step_seconds_min=times[0],
        peak_gib=(
            torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else float("nan")
        ),
        shapes=shapes,
        in_surface=tuple(state["surface"].shape),
        in_level=tuple(state["level"].shape) if "level" in state.keys() else None,
    )


def benchmark_with_data(cfg, batch: int, steps: int, warmup: int, device, dtype) -> dict:
    """Same, but fed by the real dataloader -- the number that sets the wall clock.

    A 30-minute run that is waiting on netCDF is still a 30-minute run.
    """
    from geoarches.main_hydra import collate_fn

    dataset = instantiate(cfg.dataloader.dataset)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch,
        num_workers=cfg.cluster.cpus,
        shuffle=True,
        collate_fn=collate_fn,
        persistent_workers=cfg.cluster.cpus > 0,
        drop_last=True,
    )
    embedder = instantiate(cfg.module.embedder).to(device)
    backbone = instantiate(cfg.module.backbone).to(device)
    optimiser = torch.optim.AdamW(
        list(embedder.parameters()) + list(backbone.parameters()), lr=1e-4
    )
    autocast = torch.autocast(device.type, dtype=dtype, enabled=(dtype != torch.float32))

    times: list[float] = []
    done = 0
    start = None
    for epoch in range(100):
        for sample in loader:
            if device.type == "cuda":
                torch.cuda.synchronize()
            if start is not None and done > warmup:
                times.append(time.perf_counter() - start)
            start = time.perf_counter()

            state = sample["state"].to(device)
            prev = sample["prev_state"].to(device)
            cond = torch.randn(state.shape[0], cfg.module.backbone.cond_dim, device=device)
            with autocast:
                pred = embedder.decode(backbone(embedder.encode(state, prev), cond))
                loss = sum(((pred - sample["next_state"].to(device)) ** 2).mean().values())
            loss.backward()
            optimiser.step()
            optimiser.zero_grad(set_to_none=True)

            done += 1
            if done >= warmup + steps + 1:
                break
        if done >= warmup + steps + 1:
            break

    times.sort()
    median = times[len(times) // 2]
    return dict(
        n_samples=len(dataset),
        workers=cfg.cluster.cpus,
        step_seconds=median,
        samples_per_second=batch / median,
        steps_per_epoch=len(dataset) // batch,
    )


# ---------------------------------------------------------------------------
def shape_checks(device) -> None:
    """The three claims the rest of the project relies on, checked out loud."""
    from geoarches.backbones.archesweather import ArchesWeatherCondBackbone

    from oceanarches.backbones.ocean_embedder import OceanEncodeDecodeLayer

    print("\nshape checks (batch 2, emb_dim 32)")
    print("-" * 78)
    cases = [
        ("full    ", 7, 7, 4, 4, 13),
        ("ocean   ", 7, 3, 4, 4, 13),
        ("seaice  ", 7, 4, 4, 0, 13),
        ("full/12 ", 7, 7, 4, 4, 12),
    ]
    for name, s_in, s_out, l_in, l_out, n_depths in cases:
        embedder = OceanEncodeDecodeLayer(
            s_in, s_out, l_in, l_out, n_depths, emb_dim=32, out_emb_dim=64
        ).to(device)
        backbone = ArchesWeatherCondBackbone(
            tensor_size=(embedder.z_dim, 60, 120),
            emb_dim=32,
            cond_dim=16,
            num_heads=(2, 4, 4, 2),
            window_size=(1, 6, 10),
            depth_multiplier=1,
            use_skip=True,
            # On, which is only possible because every depth preset lands on 8.
            first_interaction_layer="linear",
            axis_attn=True,
            mlp_layer="swiglu",
        ).to(device)
        state = TensorDict(
            {
                "surface": torch.randn(2, s_in, 1, N_LAT, N_LON, device=device),
                "level": torch.randn(2, l_in, n_depths, N_LAT, N_LON, device=device),
            },
            batch_size=2,
        )
        with torch.no_grad():
            tokens = embedder.encode(state, state)
            raw = backbone(tokens, torch.randn(2, 16, device=device))
            out = embedder.decode(raw)
        surface = tuple(out["surface"].shape)
        level = tuple(out["level"].shape) if "level" in out.keys() else None
        # Not `assert`: this is the whole point of the round trip, and `python -O`
        # strips asserts, so the advertised shape check would validate nothing.
        # The old message was a bare tuple, which said neither what was expected
        # nor which preset failed.
        expected_surface = (2, s_out, 1, N_LAT, N_LON)
        expected_level = (2, l_out, n_depths, N_LAT, N_LON) if l_out else None
        if surface != expected_surface or level != expected_level:
            raise SystemExit(
                f"{name}: the encode -> backbone -> decode round trip did not return the "
                f"input grid.\n  surface {surface}, expected {expected_surface}"
                f"\n  level   {level}, expected {expected_level}"
            )
        print(
            f"  {name} encode {tuple(tokens.shape)} -> backbone {tuple(raw.shape)} "
            f"-> surface {surface} level {level}"
        )
    print("  all round trips return the input grid (180x360), no fake south pole")
    print("  LinVert and axial attention were ON: the latent depth is geoarches' 8")


# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--presets", nargs="+", default=PRESETS, choices=PRESETS)
    parser.add_argument("--dataloader", default="glorys")
    parser.add_argument("--cluster", default="jupiter_1gpu")
    parser.add_argument("--batch-size", type=int, default=None, help="default: cluster.batch_size")
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument(
        "--data",
        action="store_true",
        help="also time steps fed by the real dataloader (needs the prepared archive)",
    )
    parser.add_argument("--data-preset", default="tiny")
    parser.add_argument("--data-dataloader", default="glorys_tiny")
    parser.add_argument("--target-minutes", type=float, default=30.0)
    parser.add_argument("--skip-checks", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}", end="")
    if device.type == "cuda":
        name = torch.cuda.get_device_name(0)
        total = torch.cuda.get_device_properties(0).total_memory / 2**30
        print(f"  {name}, {total:.0f} GiB")
    else:
        print("  (no GPU: step times are not representative, memory is not measured)")
    torch.set_float32_matmul_precision("medium")

    results = {}
    for preset in args.presets:
        cfg = build_cfg(preset, args.dataloader, args.cluster)
        batch = args.batch_size or cfg.cluster.batch_size
        dtype = PRECISIONS[cfg.cluster.precision]
        results[preset] = benchmark(cfg, batch, args.steps, args.warmup, device, dtype)
        results[preset]["batch"] = batch
        results[preset]["max_steps"] = cfg.module.max_steps
        results[preset]["precision"] = cfg.cluster.precision
        print(f"  measured {preset}", flush=True)

    print(f"\n{args.dataloader}, cluster={args.cluster}")
    print("=" * 112)
    header = (
        f"{'preset':7s} {'batch':>5s} {'params':>10s} {'emb':>8s} {'s/step':>8s} "
        f"{'ms/sample':>10s} {'peak GiB':>9s} {'GiB/sample':>11s} "
        f"{'max_steps':>10s} {'wall clock':>12s}"
    )
    print(header)
    print("-" * 112)
    for preset, r in results.items():
        hours = r["step_seconds"] * r["max_steps"] / 3600
        wall = f"{hours * 60:.0f} min" if hours < 3 else f"{hours:.1f} h"
        print(
            f"{preset:7s} {r['batch']:5d} {r['params_total'] / 1e6:9.1f}M "
            f"{r['params_embedder'] / 1e6:7.2f}M {r['step_seconds']:8.3f} "
            f"{1000 * r['step_seconds'] / r['batch']:10.1f} {r['peak_gib']:9.2f} "
            f"{r['peak_gib'] / r['batch']:11.2f} "
            f"{r['max_steps']:10d} {wall:>12s}"
        )
    print("-" * 112)
    # Read the table down the ms/sample and GiB/sample columns, not down s/step
    # and peak GiB: every preset runs at the batch size that fits it, so the raw
    # per-step numbers compare different amounts of work.
    print("s/step and peak GiB are AT THIS ROW'S BATCH SIZE and are not comparable")
    print("across rows; ms/sample and GiB/sample are. Peak memory is very close to")
    print("linear in batch x emb_dim x depth_multiplier, which is why the presets")
    print("were given the batch sizes they were.")
    print("params = embedder + backbone; s/step = median forward+backward+optimiser,")
    print("compute only (no data loading); wall clock = s/step x the preset's max_steps.")

    for preset, r in results.items():
        print(f"\n{preset}: in surface {r['in_surface']} level {r['in_level']}")
        for key, shape in r["shapes"].items():
            print(f"    {key:12s} {shape}")

    if args.data:
        cfg = build_cfg(args.data_preset, args.data_dataloader, args.cluster)
        batch = args.batch_size or cfg.cluster.batch_size
        dtype = PRECISIONS[cfg.cluster.precision]
        print(
            f"\nend-to-end with the real dataloader: {args.data_preset} / {args.data_dataloader}"
        )
        print("-" * 78)
        d = benchmark_with_data(cfg, batch, args.steps * 4, args.warmup * 2, device, dtype)
        budget = int(args.target_minutes * 60 / d["step_seconds"])
        print(f"  dataset          {d['n_samples']} samples, {d['workers']} workers")
        print(
            f"  step             {d['step_seconds']:.3f} s  ({d['samples_per_second']:.1f} samples/s)"
        )
        print(
            f"  compute only     {results.get(args.data_preset, {}).get('step_seconds', float('nan')):.3f} s"
        )
        print(f"  epoch            {d['steps_per_epoch']} steps")
        print(
            f"  configured       max_steps={cfg.module.max_steps} "
            f"-> {cfg.module.max_steps * d['step_seconds'] / 60:.1f} min "
            f"({cfg.module.max_steps / d['steps_per_epoch']:.1f} epochs)"
        )
        print(f"  {args.target_minutes:.0f} min budget  max_steps={budget}")

    if not args.skip_checks:
        shape_checks(device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
