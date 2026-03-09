#!/usr/bin/env python3
"""Plot shell-profile comparisons for a rollout-trained PWOM checkpoint.

Physical interpretation:
- Each shell node represents one coarse `(species, altitude_shell)` distribution cell along the
  field line.
- These plots compare the predicted and true shell-state profiles after autoregressive rollout, so
  they show where the surrogate is accurate or biased in altitude space for each species.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Iterable

import matplotlib

# The plotting script is non-interactive by design because it writes figures to disk.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Allow running this script directly from the repository root.
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from pwom_surrogate.config import ModelConfig
from pwom_surrogate.fieldline_graph_data import (
    FieldlineRolloutDataset,
    FieldlineWindowNormalization,
    available_dynamic_diagnostic_feature_names,
    build_boundary_source_node_mask,
    build_transport_interfaces,
    build_fieldline_rollout_arrays,
    dynamic_diagnostic_valid_mask,
    extract_dynamic_diagnostic_channel,
    load_fieldline_graph_series,
)
from pwom_surrogate.fieldline_graph_model import FieldlineGraphForecaster
from train_fieldline_graph import (
    build_support_band_node_mask,
    select_device,
)
from train_fieldline_graph_rollout import (
    rollout_forward,
    stats_to_torch,
)


# Channel labels make saved figures readable without having to cross-reference the data schema.
DEFAULT_CHANNELS = ("population_fraction", "mean_vpar", "mean_vperp")
CHANNEL_LABELS = {
    "population_fraction": "Population Fraction",
    "mean_vpar": "Mean Vpar (m/s)",
    "mean_vperp": "Mean Vperp (m/s)",
    "std_vpar": "Std Vpar (m/s)",
    "std_vperp": "Std Vperp (m/s)",
    "population_weighted_mean_vpar": "Population-Weighted Mean Vpar",
    "population_weighted_mean_vperp": "Population-Weighted Mean Vperp",
    "population_weighted_std_vpar": "Population-Weighted Std Vpar",
    "population_weighted_std_vperp": "Population-Weighted Std Vperp",
    "log_total_weight": "log(1 + Total Weight)",
    "mean_altitude_km": "Mean Altitude (km)",
    "std_altitude_km": "Std Altitude (km)",
    "open_fraction": "Open Fraction",
    "mean_b": "Mean B",
    "mean_e": "Mean E",
    "mean_g": "Mean G",
}


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for shell-profile plotting."""
    parser = argparse.ArgumentParser(description="Plot shell-profile rollout diagnostics for a PWOM surrogate checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-dir", type=str, default="")
    parser.add_argument("--split", choices=["train", "valid"], default="valid")
    parser.add_argument("--horizon", type=int, default=0)
    parser.add_argument("--channels", type=str, nargs="*", default=list(DEFAULT_CHANNELS))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=["cpu", "mps", "cuda", "auto"], default="auto")
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--verbose-data", action="store_true")
    return parser.parse_args()


def _validate_channel_names(channel_names: Iterable[str]) -> tuple[str, ...]:
    """Validate requested channel names against stored and derived diagnostic features."""
    available = set(available_dynamic_diagnostic_feature_names())
    resolved_names: list[str] = []
    for channel_name in channel_names:
        if channel_name not in available:
            raise ValueError(f"Unknown channel='{channel_name}'. Available channels: {sorted(available)}")
        resolved_names.append(channel_name)
    return tuple(resolved_names)


def _reshape_shell_tensor(
    tensor: np.ndarray,
    num_species: int,
    num_altitude_shells: int,
) -> np.ndarray:
    """Reshape shell-node tensors into `[samples, horizon, species, shell, channel]`.

    Physical interpretation:
    - The graph layout is species-major over shell nodes, so a plain reshape recovers the explicit
      `(species, altitude_shell)` structure for plotting.
    - The optional global node is excluded because shell profiles should only show physical cells.
    """
    shell_nodes = num_species * num_altitude_shells
    shell_tensor = tensor[:, :, :shell_nodes, :]
    return shell_tensor.reshape(
        shell_tensor.shape[0],
        shell_tensor.shape[1],
        num_species,
        num_altitude_shells,
        shell_tensor.shape[-1],
    )


def _plot_channel_profiles(
    output_dir: str,
    channel_name: str,
    shell_centers_km: np.ndarray,
    species_ids: tuple[int, ...],
    pred_mean: np.ndarray,
    true_mean: np.ndarray,
    step_index: int,
    split: str,
) -> str:
    """Write one figure comparing predicted and true shell profiles for all species.

    Physical interpretation:
    - Each subplot is one ion species.
    - The two lines show how the surrogate reconstructs the altitude dependence of the selected
      shell statistic after `step_index + 1` autoregressive rollout steps.
    """
    num_species = len(species_ids)
    ncols = 4
    nrows = int(math.ceil(num_species / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4.5 * ncols, 3.2 * nrows), sharex=True)
    axes_array = np.asarray(axes).reshape(-1)

    for species_offset, species_id in enumerate(species_ids):
        axis = axes_array[species_offset]
        axis.plot(shell_centers_km, true_mean[species_offset], color="#1f4e79", linewidth=2.0, label="true")
        axis.plot(shell_centers_km, pred_mean[species_offset], color="#d55c00", linewidth=2.0, linestyle="--", label="pred")
        axis.set_title(f"Species {species_id}")
        axis.grid(alpha=0.25, linewidth=0.6)
        if species_offset % ncols == 0:
            axis.set_ylabel(CHANNEL_LABELS.get(channel_name, channel_name))
        if species_offset >= (num_species - ncols):
            axis.set_xlabel("Altitude Shell Center (km)")

    for axis in axes_array[num_species:]:
        axis.axis("off")

    handles, labels = axes_array[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.suptitle(f"{split} rollout step {step_index + 1}: {CHANNEL_LABELS.get(channel_name, channel_name)}")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))

    output_path = os.path.join(output_dir, f"{channel_name}_step_{step_index + 1}.png")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def main() -> None:
    """Load the rollout checkpoint, aggregate shell profiles, and write diagnostic figures."""
    args = parse_args()
    device = select_device(args.device)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    checkpoint_dir = os.path.dirname(os.path.abspath(args.checkpoint))
    normalization = FieldlineWindowNormalization.load_json(os.path.join(checkpoint_dir, "normalization.json"))

    data_cfg = checkpoint["data_cfg"]
    model_cfg = ModelConfig(**checkpoint["model_cfg"])
    train_cfg = checkpoint["train_cfg"]
    eval_horizon = args.horizon if args.horizon > 0 else int(train_cfg["horizon"])
    data_dir = args.data_dir if args.data_dir else data_cfg["data_dir"]
    species_ids = tuple(int(species_id) for species_id in checkpoint["species_ids"])
    shell_edges_km = np.asarray(checkpoint["shell_edges_km"], dtype=np.float32)
    shell_centers_km = 0.5 * (shell_edges_km[:-1] + shell_edges_km[1:])
    output_dir = (
        os.path.abspath(args.output_dir)
        if args.output_dir
        else os.path.join(checkpoint_dir, "viz_shell_profiles")
    )
    os.makedirs(output_dir, exist_ok=True)

    channel_names = tuple(args.channels)
    channel_names = _validate_channel_names(channel_names=channel_names)

    series = load_fieldline_graph_series(
        data_dir=data_dir,
        split=args.split,
        num_particles=data_cfg["num_particles"],
        num_altitude_shells=data_cfg["num_altitude_shells"],
        include_global_node=data_cfg["include_global_node"],
        include_species_coupling_edges=data_cfg["include_species_coupling_edges"],
        species_ids=species_ids,
        shell_edges_km_override=shell_edges_km,
        verbose=args.verbose_data,
    )
    arrays = build_fieldline_rollout_arrays(
        series=series,
        window_length=data_cfg["window_length"],
        horizon=eval_horizon,
    )
    dataset = FieldlineRolloutDataset(
        static_node_features=series.static_node_features,
        arrays=arrays,
        normalization=normalization,
        predict_delta_state=model_cfg.predict_delta_state,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = FieldlineGraphForecaster(cfg=model_cfg).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    static_node_features = torch.as_tensor(series.static_node_features, dtype=torch.float32, device=device)
    edge_index = torch.as_tensor(series.edge_index, dtype=torch.long, device=device)
    stats_torch = stats_to_torch(
        stats=normalization,
        device=device,
        num_nodes=series.layout.num_nodes,
        target_schema=model_cfg.target_schema,
        velocity_species_weight_per_node=checkpoint.get("velocity_species_weight_per_node"),
        boundary_source_node_mask=checkpoint.get(
            "boundary_source_node_mask",
            build_boundary_source_node_mask(series.layout),
        ),
        support_band_node_mask=checkpoint.get("support_band_node_mask", build_support_band_node_mask(
            layout=series.layout,
            shell_edges_km=series.shell_edges_km,
            lower_km=float(train_cfg.get("support_band_lower_km", 2900.0)),
            upper_km=float(train_cfg.get("support_band_upper_km", 4700.0)),
        )),
        secondary_support_band_node_mask=checkpoint.get("secondary_support_band_node_mask", build_support_band_node_mask(
            layout=series.layout,
            shell_edges_km=series.shell_edges_km,
            lower_km=float(train_cfg.get("secondary_support_band_lower_km", 6900.0)),
            upper_km=float(train_cfg.get("secondary_support_band_upper_km", 7800.0)),
        )),
        transport_interfaces=(
            build_transport_interfaces(
                layout=series.layout,
                shell_edges_km=series.shell_edges_km,
            )
            if bool(model_cfg.flux_transport_head)
            else None
        ),
        transport_delta_scale=np.asarray(
            checkpoint.get("transport_delta_scale", np.ones(2, dtype=np.float32)),
            dtype=np.float32,
        ),
        transport_shell_node_mask=np.asarray(
            checkpoint.get(
                "transport_shell_node_mask",
                np.concatenate(
                    (
                        np.ones(series.layout.num_shell_nodes, dtype=np.float32),
                        np.zeros(series.layout.num_nodes - series.layout.num_shell_nodes, dtype=np.float32),
                    )
                ),
            ),
            dtype=np.float32,
        ),
    )

    pred_batches: list[np.ndarray] = []
    true_batches: list[np.ndarray] = []
    iterator = tqdm(loader, desc=f"profile:{args.split}", dynamic_ncols=True)

    with torch.no_grad():
        for batch in iterator:
            history_dynamic = batch["history_dynamic"].to(device=device, dtype=torch.float32)
            history_global_context = batch["history_global_context"].to(device=device, dtype=torch.float32)
            history_boundary_context = batch["history_boundary_context"].to(device=device, dtype=torch.float32)
            target_dynamic_seq = batch["target_dynamic_seq"].to(device=device, dtype=torch.float32)
            target_dynamic_seq_physical = batch["target_dynamic_seq_physical"].to(device=device, dtype=torch.float32)
            target_global_context_seq = batch["target_global_context_seq"].to(device=device, dtype=torch.float32)
            target_boundary_context_seq = batch["target_boundary_context_seq"].to(device=device, dtype=torch.float32)

            outputs = rollout_forward(
                model=model,
                static_node_features=static_node_features,
                history_dynamic=history_dynamic,
                history_global_context=history_global_context,
                history_boundary_context=history_boundary_context,
                target_dynamic_seq=target_dynamic_seq,
                target_dynamic_seq_physical=target_dynamic_seq_physical,
                target_global_context_seq=target_global_context_seq,
                target_boundary_context_seq=target_boundary_context_seq,
                edge_index=edge_index,
                stats_torch=stats_torch,
                predict_delta_state=model_cfg.predict_delta_state,
                rollout_weight=float(train_cfg["rollout_weight"]),
                support_band_loss_weight=float(train_cfg.get("support_band_loss_weight", 0.0)),
                secondary_support_band_loss_weight=float(train_cfg.get("secondary_support_band_loss_weight", 0.0)),
                # Replay the localized support-band penalty used in training so plotted profile losses
                # are evaluated under the same physical objective as the saved rollout checkpoint.
                localized_support_band_mean_vpar_loss_weight=float(
                    train_cfg.get("localized_support_band_mean_vpar_loss_weight", 0.0)
                ),
                occupied_shell_threshold=float(train_cfg.get("occupied_shell_threshold", 1.0e-3)),
                diagnostic_mean_vpar_loss_weight=float(train_cfg.get("diagnostic_mean_vpar_loss_weight", 0.0)),
                diagnostic_std_vpar_loss_weight=float(train_cfg.get("diagnostic_std_vpar_loss_weight", 0.0)),
                diagnostic_mean_vpar_scale=float(checkpoint.get("diagnostic_mean_vpar_scale", 1.0)),
                diagnostic_std_vpar_scale=float(checkpoint.get("diagnostic_std_vpar_scale", 1.0)),
                teacher_forcing_prob=0.0,
                training=False,
                diagnostic_band_mask_mode=str(train_cfg.get("diagnostic_band_mask_mode", "global")),
            )

            pred_batches.append(outputs["pred_dynamic_seq_physical"].detach().cpu().numpy())
            true_batches.append(target_dynamic_seq_physical.detach().cpu().numpy())
            iterator.set_postfix(
                total=f"{float(outputs['total_loss'].detach().cpu()):.5f}",
                roll=f"{float(outputs['rollout_loss'].detach().cpu()):.5f}",
                one=f"{float(outputs['one_step_loss'].detach().cpu()):.5f}",
            )

    if not pred_batches:
        raise ValueError("No rollout samples were available for shell-profile plotting.")

    pred_dynamic_seq = np.concatenate(pred_batches, axis=0)
    true_dynamic_seq = np.concatenate(true_batches, axis=0)

    pred_shell = _reshape_shell_tensor(
        tensor=pred_dynamic_seq,
        num_species=len(species_ids),
        num_altitude_shells=int(data_cfg["num_altitude_shells"]),
    )
    true_shell = _reshape_shell_tensor(
        tensor=true_dynamic_seq,
        num_species=len(species_ids),
        num_altitude_shells=int(data_cfg["num_altitude_shells"]),
    )

    summary: dict[str, object] = {
        "checkpoint": os.path.abspath(args.checkpoint),
        "split": args.split,
        "num_samples": int(pred_shell.shape[0]),
        "eval_horizon": int(eval_horizon),
        "species_ids": list(species_ids),
        "channels": list(channel_names),
        "plots": {},
        "channel_step_species_rmse": {},
    }

    for channel_name in channel_names:
        channel_plots: list[str] = []
        channel_rmse: dict[str, dict[str, float]] = {}

        for step_index in range(eval_horizon):
            pred_profile = extract_dynamic_diagnostic_channel(
                dynamic_physical=pred_shell[:, step_index],
                feature_name=channel_name,
                undefined_fill_value=np.nan,
            )
            true_profile = extract_dynamic_diagnostic_channel(
                dynamic_physical=true_shell[:, step_index],
                feature_name=channel_name,
                undefined_fill_value=np.nan,
            )
            pred_mean = np.nanmean(pred_profile, axis=0)
            true_mean = np.nanmean(true_profile, axis=0)
            plot_path = _plot_channel_profiles(
                output_dir=output_dir,
                channel_name=channel_name,
                shell_centers_km=shell_centers_km,
                species_ids=species_ids,
                pred_mean=pred_mean,
                true_mean=true_mean,
                step_index=step_index,
                split=args.split,
            )
            channel_plots.append(os.path.abspath(plot_path))

            # RMSE by species is a compact way to summarize where the shell-profile bias is
            # concentrated without losing the rollout-step separation.
            pred_valid_mask = dynamic_diagnostic_valid_mask(
                dynamic_physical=pred_shell[:, step_index],
                feature_name=channel_name,
            )
            true_valid_mask = dynamic_diagnostic_valid_mask(
                dynamic_physical=true_shell[:, step_index],
                feature_name=channel_name,
            )
            pair_valid_mask = np.asarray(pred_valid_mask & true_valid_mask, dtype=bool)
            squared_error = np.square(pred_profile - true_profile)
            species_rmse_values: list[float | None] = []
            for species_offset in range(len(species_ids)):
                species_error = squared_error[:, species_offset, :]
                species_valid = pair_valid_mask[:, species_offset, :]
                if np.any(species_valid):
                    species_rmse_values.append(float(np.sqrt(np.nanmean(species_error[species_valid]))))
                else:
                    # `None` is used instead of zero so sparse undefined shells are not mistaken
                    # for a perfect prediction in the summary JSON.
                    species_rmse_values.append(None)
            channel_rmse[f"step_{step_index + 1}"] = {
                str(species_id): species_rmse_values[species_offset]
                for species_offset, species_id in enumerate(species_ids)
            }

        summary["plots"][channel_name] = channel_plots
        summary["channel_step_species_rmse"][channel_name] = channel_rmse

    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"[plot] shell-profile plots saved in {output_dir}", flush=True)
    print(f"[plot] summary saved to {summary_path}", flush=True)


if __name__ == "__main__":
    main()
