#!/usr/bin/env python3
"""Plot focused species-level parallel-velocity diagnostics for a rollout-trained PWOM checkpoint.

Physical interpretation:
- The goal is to inspect the shell-by-shell parallel drift for one species, not just the global
  rollout loss.
- Population fraction is plotted alongside `mean_vpar` because low-occupancy shells naturally make
  shell-mean velocity statistics less reliable.
- Both direct and population-weighted velocity-channel layouts are supported.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import matplotlib

# The script writes figures to disk and should never depend on an interactive display.
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
    build_boundary_source_node_mask,
    build_transport_interfaces,
    build_fieldline_rollout_arrays,
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


# Direct velocity channels represent shell moments as mean/std values.
DIRECT_SCHEMA_INDEX = {
    "population_fraction": 0,
    "mean_vpar": 4,
    "mean_vperp": 5,
    "std_vpar": 7,
    "std_vperp": 8,
}

# Population-weighted channels represent momentum-like quantities.
POPULATION_WEIGHTED_SCHEMA_INDEX = {
    "population_fraction": 0,
    "population_weighted_mean_vpar": 4,
    "population_weighted_mean_vperp": 5,
    "population_weighted_std_vpar": 7,
    "population_weighted_std_vperp": 8,
}


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the focused species diagnostic plots."""
    parser = argparse.ArgumentParser(description="Plot focused mean_vpar diagnostics for one PWOM species.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-dir", type=str, default="")
    parser.add_argument("--split", choices=["train", "valid"], default="valid")
    parser.add_argument("--horizon", type=int, default=0)
    parser.add_argument("--focus-species", type=int, default=2)
    parser.add_argument(
        "--velocity-schema",
        choices=["auto", "direct", "population_weighted"],
        default="auto",
        help="How to interpret the velocity channels stored in the checkpoint outputs.",
    )
    parser.add_argument("--occupied-shell-threshold", type=float, default=1.0e-3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=["cpu", "mps", "cuda", "auto"], default="auto")
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--verbose-data", action="store_true")
    return parser.parse_args()


def infer_velocity_schema(checkpoint: dict[str, object], requested_schema: str) -> str:
    """Infer whether the checkpoint stores direct or population-weighted velocity channels.

    Physical interpretation:
    - Some checkpoints emit direct shell means.
    - Others emit population-weighted moment channels.
    """
    if requested_schema != "auto":
        return requested_schema

    history = checkpoint.get("history")
    if isinstance(history, list) and history:
        first_record = history[0]
        if isinstance(first_record, dict):
            rollout_mse = first_record.get("valid_rollout_channel_mse")
            if isinstance(rollout_mse, dict):
                if "population_weighted_mean_vpar" in rollout_mse:
                    return "population_weighted"
                if "mean_vpar" in rollout_mse:
                    return "direct"

    # Direct is the safer fallback for the oldest stable branch, which is the main checkpoint this
    # script is meant to inspect.
    return "direct"


def reshape_shell_tensor(
    tensor: np.ndarray,
    num_species: int,
    num_altitude_shells: int,
) -> np.ndarray:
    """Reshape shell-node tensors into `[samples, horizon, species, shell, channel]`."""
    shell_nodes = num_species * num_altitude_shells
    shell_tensor = tensor[:, :, :shell_nodes, :]
    return shell_tensor.reshape(
        shell_tensor.shape[0],
        shell_tensor.shape[1],
        num_species,
        num_altitude_shells,
        shell_tensor.shape[-1],
    )


def extract_focus_channel(
    dynamic_physical: np.ndarray,
    feature_name: str,
    velocity_schema: str,
    occupied_shell_threshold: float,
) -> np.ndarray:
    """Extract one focus diagnostic channel under either supported velocity schema.

    Physical interpretation:
    - `population_fraction` is always stored directly.
    - `mean_vpar` is either stored directly or recovered from a population-weighted channel,
      depending on checkpoint schema.
    """
    dynamic_physical = np.asarray(dynamic_physical, dtype=np.float32)
    population_fraction = np.asarray(
        dynamic_physical[..., DIRECT_SCHEMA_INDEX["population_fraction"]],
        dtype=np.float32,
    )
    if feature_name == "population_fraction":
        return population_fraction

    if velocity_schema == "direct":
        if feature_name not in DIRECT_SCHEMA_INDEX:
            raise ValueError(f"Unsupported direct-schema feature='{feature_name}'.")
        return np.asarray(dynamic_physical[..., DIRECT_SCHEMA_INDEX[feature_name]], dtype=np.float32)

    if feature_name != "mean_vpar":
        raise ValueError("Population-weighted focus plots currently support only `mean_vpar` and `population_fraction`.")

    weighted_mean_vpar = np.asarray(
        dynamic_physical[..., POPULATION_WEIGHTED_SCHEMA_INDEX["population_weighted_mean_vpar"]],
        dtype=np.float32,
    )
    safe_population = np.maximum(population_fraction, float(occupied_shell_threshold)).astype(np.float32, copy=False)
    return (weighted_mean_vpar / safe_population).astype(np.float32, copy=False)


def true_occupied_mask(
    dynamic_physical: np.ndarray,
    occupied_shell_threshold: float,
) -> np.ndarray:
    """Return the true occupied-shell mask used for focused species diagnostics.

    Physical interpretation:
    - The target occupancy determines whether `mean_vpar` is a meaningful diagnostic at all.
    - Predicted collapse in an occupied target shell should still count as an error, not disappear
      from the metric.
    """
    dynamic_physical = np.asarray(dynamic_physical, dtype=np.float32)
    population_fraction = np.asarray(
        dynamic_physical[..., DIRECT_SCHEMA_INDEX["population_fraction"]],
        dtype=np.float32,
    )
    return np.asarray(population_fraction >= float(occupied_shell_threshold), dtype=bool)


def masked_nanmean(values: np.ndarray, valid_mask: np.ndarray, axis: int) -> np.ndarray:
    """Compute a mean with explicit masks while leaving unsupported cells as `NaN`."""
    masked_values = np.where(valid_mask, values, 0.0).astype(np.float32, copy=False)
    valid_count = np.sum(valid_mask.astype(np.float32, copy=False), axis=axis)
    summed = np.sum(masked_values, axis=axis)
    mean_value = np.divide(
        summed,
        np.maximum(valid_count, 1.0),
        dtype=np.float32,
    )
    return np.where(valid_count > 0.0, mean_value, np.nan).astype(np.float32, copy=False)


def plot_focus_profiles(
    output_dir: str,
    shell_centers_km: np.ndarray,
    species_id: int,
    mean_vpar_pred: np.ndarray,
    mean_vpar_true: np.ndarray,
    population_pred: np.ndarray,
    population_true: np.ndarray,
) -> str:
    """Plot mean_vpar and population-fraction profiles for one species across rollout steps."""
    horizon = mean_vpar_pred.shape[0]
    fig, axes = plt.subplots(
        nrows=2,
        ncols=horizon,
        figsize=(4.4 * horizon, 7.0),
        sharex="col",
    )
    if horizon == 1:
        axes = np.asarray(axes).reshape(2, 1)

    for step_index in range(horizon):
        axis_vpar = axes[0, step_index]
        axis_pop = axes[1, step_index]

        # The top row isolates the parallel drift profile, which is the main unresolved channel.
        axis_vpar.plot(shell_centers_km, mean_vpar_true[step_index], color="#184e77", linewidth=2.0, label="true")
        axis_vpar.plot(shell_centers_km, mean_vpar_pred[step_index], color="#d1495b", linewidth=2.0, linestyle="--", label="pred")
        axis_vpar.set_title(f"Step {step_index + 1}")
        axis_vpar.grid(alpha=0.25, linewidth=0.6)
        if step_index == 0:
            axis_vpar.set_ylabel("Mean Vpar (m/s)")

        # The bottom row shows where population support is low, which explains many velocity
        # outliers in sparse shells.
        axis_pop.plot(shell_centers_km, population_true[step_index], color="#184e77", linewidth=2.0, label="true")
        axis_pop.plot(shell_centers_km, population_pred[step_index], color="#d1495b", linewidth=2.0, linestyle="--", label="pred")
        axis_pop.grid(alpha=0.25, linewidth=0.6)
        axis_pop.set_xlabel("Altitude Shell Center (km)")
        if step_index == 0:
            axis_pop.set_ylabel("Population Fraction")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.suptitle(f"Species {species_id}: mean_vpar and population support by rollout step")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))

    output_path = os.path.join(output_dir, f"species_{species_id}_mean_vpar_profiles.png")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def plot_mean_vpar_error_heatmap(
    output_dir: str,
    shell_centers_km: np.ndarray,
    species_id: int,
    signed_error_by_step_shell: np.ndarray,
    valid_fraction_by_step_shell: np.ndarray,
) -> tuple[str, str]:
    """Plot rollout-step heatmaps for signed mean_vpar error and occupied-shell support."""
    fig, axis = plt.subplots(figsize=(1.0 + 0.7 * signed_error_by_step_shell.shape[1], 4.8))
    mesh = axis.imshow(
        signed_error_by_step_shell,
        aspect="auto",
        cmap="coolwarm",
        origin="lower",
    )
    axis.set_title(f"Species {species_id}: mean_vpar signed error")
    axis.set_xlabel("Altitude Shell Index")
    axis.set_ylabel("Rollout Step")
    axis.set_xticks(np.arange(shell_centers_km.shape[0]))
    axis.set_xticklabels([f"{value:.0f}" for value in shell_centers_km], rotation=45, ha="right")
    axis.set_yticks(np.arange(signed_error_by_step_shell.shape[0]))
    axis.set_yticklabels([str(step + 1) for step in range(signed_error_by_step_shell.shape[0])])
    colorbar = fig.colorbar(mesh, ax=axis)
    colorbar.set_label("Pred - True Mean Vpar (m/s)")
    fig.tight_layout()
    error_path = os.path.join(output_dir, f"species_{species_id}_mean_vpar_error_heatmap.png")
    fig.savefig(error_path, dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(1.0 + 0.7 * valid_fraction_by_step_shell.shape[1], 4.8))
    mesh = axis.imshow(
        valid_fraction_by_step_shell,
        aspect="auto",
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        origin="lower",
    )
    axis.set_title(f"Species {species_id}: occupied-shell valid fraction")
    axis.set_xlabel("Altitude Shell Index")
    axis.set_ylabel("Rollout Step")
    axis.set_xticks(np.arange(shell_centers_km.shape[0]))
    axis.set_xticklabels([f"{value:.0f}" for value in shell_centers_km], rotation=45, ha="right")
    axis.set_yticks(np.arange(valid_fraction_by_step_shell.shape[0]))
    axis.set_yticklabels([str(step + 1) for step in range(valid_fraction_by_step_shell.shape[0])])
    colorbar = fig.colorbar(mesh, ax=axis)
    colorbar.set_label("Valid Occupied Fraction")
    fig.tight_layout()
    valid_path = os.path.join(output_dir, f"species_{species_id}_occupied_fraction_heatmap.png")
    fig.savefig(valid_path, dpi=180)
    plt.close(fig)

    return error_path, valid_path


def write_shell_metrics_csv(
    output_dir: str,
    shell_centers_km: np.ndarray,
    species_id: int,
    pred_mean_vpar: np.ndarray,
    true_mean_vpar: np.ndarray,
    pred_population: np.ndarray,
    true_population: np.ndarray,
    valid_mask: np.ndarray,
) -> str:
    """Write per-step, per-shell metrics so the dominant error shells are easy to inspect.

    Physical interpretation:
    - Each CSV row describes one altitude shell for one rollout step.
    - The CSV keeps the analysis quantitative: it shows whether a large `mean_vpar` error comes
      from systematic bias, sparse support, or both.
    """
    output_path = os.path.join(output_dir, f"species_{species_id}_mean_vpar_shell_metrics.csv")
    with open(output_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "step",
                "shell_index",
                "shell_center_km",
                "valid_fraction",
                "mean_vpar_rmse",
                "mean_vpar_bias",
                "pred_mean_vpar_mean",
                "true_mean_vpar_mean",
                "pred_population_fraction_mean",
                "true_population_fraction_mean",
            ]
        )

        horizon = pred_mean_vpar.shape[1]
        num_shells = pred_mean_vpar.shape[2]
        for step_index in range(horizon):
            for shell_index in range(num_shells):
                shell_valid = valid_mask[:, step_index, shell_index]
                valid_fraction = float(np.mean(shell_valid))
                if np.any(shell_valid):
                    shell_error = pred_mean_vpar[:, step_index, shell_index] - true_mean_vpar[:, step_index, shell_index]
                    shell_error = shell_error[shell_valid]
                    mean_vpar_rmse = float(np.sqrt(np.mean(np.square(shell_error))))
                    mean_vpar_bias = float(np.mean(shell_error))
                    pred_mean_vpar_mean = float(np.mean(pred_mean_vpar[:, step_index, shell_index][shell_valid]))
                    true_mean_vpar_mean = float(np.mean(true_mean_vpar[:, step_index, shell_index][shell_valid]))
                    pred_population_fraction_mean = float(
                        np.mean(pred_population[:, step_index, shell_index][shell_valid])
                    )
                    true_population_fraction_mean = float(
                        np.mean(true_population[:, step_index, shell_index][shell_valid])
                    )
                else:
                    mean_vpar_rmse = float("nan")
                    mean_vpar_bias = float("nan")
                    pred_mean_vpar_mean = float("nan")
                    true_mean_vpar_mean = float("nan")
                    pred_population_fraction_mean = float("nan")
                    true_population_fraction_mean = float("nan")

                writer.writerow(
                    [
                        step_index + 1,
                        shell_index,
                        float(shell_centers_km[shell_index]),
                        valid_fraction,
                        mean_vpar_rmse,
                        mean_vpar_bias,
                        pred_mean_vpar_mean,
                        true_mean_vpar_mean,
                        pred_population_fraction_mean,
                        true_population_fraction_mean,
                    ]
                )

    return output_path


def main() -> None:
    """Load the checkpoint, run rollout evaluation, and write focused species diagnostics."""
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
    if args.focus_species not in species_ids:
        raise ValueError(f"focus_species={args.focus_species} not found in checkpoint species_ids={species_ids}")
    focus_species_offset = species_ids.index(args.focus_species)
    shell_edges_km = np.asarray(checkpoint["shell_edges_km"], dtype=np.float32)
    shell_centers_km = 0.5 * (shell_edges_km[:-1] + shell_edges_km[1:])
    velocity_schema = infer_velocity_schema(checkpoint=checkpoint, requested_schema=args.velocity_schema)
    output_dir = (
        os.path.abspath(args.output_dir)
        if args.output_dir
        else os.path.join(checkpoint_dir, f"viz_species_{args.focus_species}_mean_vpar")
    )
    os.makedirs(output_dir, exist_ok=True)

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
    iterator = tqdm(loader, desc=f"focus:{args.split}", dynamic_ncols=True)

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
                # The focused plot should replay the same localized rollout objective that produced the
                # checkpoint so the reported shell-band diagnostics stay consistent with training.
                localized_support_band_mean_vpar_loss_weight=float(
                    train_cfg.get("localized_support_band_mean_vpar_loss_weight", 0.0)
                ),
                occupied_shell_threshold=float(train_cfg.get("occupied_shell_threshold", args.occupied_shell_threshold)),
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
        raise ValueError("No rollout samples were available for focused diagnostics.")

    pred_dynamic_seq = np.concatenate(pred_batches, axis=0)
    true_dynamic_seq = np.concatenate(true_batches, axis=0)
    pred_shell = reshape_shell_tensor(
        tensor=pred_dynamic_seq,
        num_species=len(species_ids),
        num_altitude_shells=int(data_cfg["num_altitude_shells"]),
    )
    true_shell = reshape_shell_tensor(
        tensor=true_dynamic_seq,
        num_species=len(species_ids),
        num_altitude_shells=int(data_cfg["num_altitude_shells"]),
    )

    pred_focus = pred_shell[:, :, focus_species_offset]
    true_focus = true_shell[:, :, focus_species_offset]

    pred_mean_vpar = extract_focus_channel(
        dynamic_physical=pred_focus,
        feature_name="mean_vpar",
        velocity_schema=velocity_schema,
        occupied_shell_threshold=args.occupied_shell_threshold,
    )
    true_mean_vpar = extract_focus_channel(
        dynamic_physical=true_focus,
        feature_name="mean_vpar",
        velocity_schema=velocity_schema,
        occupied_shell_threshold=args.occupied_shell_threshold,
    )
    pred_population = extract_focus_channel(
        dynamic_physical=pred_focus,
        feature_name="population_fraction",
        velocity_schema=velocity_schema,
        occupied_shell_threshold=args.occupied_shell_threshold,
    )
    true_population = extract_focus_channel(
        dynamic_physical=true_focus,
        feature_name="population_fraction",
        velocity_schema=velocity_schema,
        occupied_shell_threshold=args.occupied_shell_threshold,
    )
    valid_mask = true_occupied_mask(
        dynamic_physical=true_focus,
        occupied_shell_threshold=args.occupied_shell_threshold,
    )
    pred_mean_vpar_plot = np.where(valid_mask, pred_mean_vpar, np.nan)
    true_mean_vpar_plot = np.where(valid_mask, true_mean_vpar, np.nan)
    pred_population_plot = np.where(valid_mask, pred_population, np.nan)
    true_population_plot = np.where(valid_mask, true_population, np.nan)

    # Plot averages over samples so the figures show the mean shell structure instead of one
    # arbitrary rollout example.
    profiles_path = plot_focus_profiles(
        output_dir=output_dir,
        shell_centers_km=shell_centers_km,
        species_id=args.focus_species,
        mean_vpar_pred=masked_nanmean(pred_mean_vpar_plot, valid_mask=valid_mask, axis=0),
        mean_vpar_true=masked_nanmean(true_mean_vpar_plot, valid_mask=valid_mask, axis=0),
        population_pred=masked_nanmean(pred_population_plot, valid_mask=valid_mask, axis=0),
        population_true=masked_nanmean(true_population_plot, valid_mask=valid_mask, axis=0),
    )

    signed_error = pred_mean_vpar - true_mean_vpar
    signed_error_by_step_shell = np.asarray(
        [
            [
                float(np.mean(signed_error[:, step_index, shell_index][valid_mask[:, step_index, shell_index]]))
                if np.any(valid_mask[:, step_index, shell_index])
                else np.nan
                for shell_index in range(signed_error.shape[-1])
            ]
            for step_index in range(signed_error.shape[1])
        ],
        dtype=np.float32,
    )
    valid_fraction_by_step_shell = np.asarray(
        [
            [
                float(np.mean(valid_mask[:, step_index, shell_index]))
                for shell_index in range(valid_mask.shape[-1])
            ]
            for step_index in range(valid_mask.shape[1])
        ],
        dtype=np.float32,
    )
    error_heatmap_path, valid_heatmap_path = plot_mean_vpar_error_heatmap(
        output_dir=output_dir,
        shell_centers_km=shell_centers_km,
        species_id=args.focus_species,
        signed_error_by_step_shell=signed_error_by_step_shell,
        valid_fraction_by_step_shell=valid_fraction_by_step_shell,
    )
    shell_metrics_csv_path = write_shell_metrics_csv(
        output_dir=output_dir,
        shell_centers_km=shell_centers_km,
        species_id=args.focus_species,
        pred_mean_vpar=pred_mean_vpar,
        true_mean_vpar=true_mean_vpar,
        pred_population=pred_population,
        true_population=true_population,
        valid_mask=valid_mask,
    )

    step_rmse = {}
    for step_index in range(true_mean_vpar.shape[1]):
        step_valid = valid_mask[:, step_index]
        if np.any(step_valid):
            step_rmse[f"step_{step_index + 1}"] = float(
                np.sqrt(np.mean(np.square(signed_error[:, step_index][step_valid])))
            )
        else:
            step_rmse[f"step_{step_index + 1}"] = None

    summary = {
        "checkpoint": os.path.abspath(args.checkpoint),
        "split": args.split,
        "focus_species": int(args.focus_species),
        "velocity_schema": velocity_schema,
        "occupied_shell_threshold": float(args.occupied_shell_threshold),
        "num_samples": int(pred_focus.shape[0]),
        "eval_horizon": int(eval_horizon),
        "plots": {
            "profiles": os.path.abspath(profiles_path),
            "mean_vpar_error_heatmap": os.path.abspath(error_heatmap_path),
            "occupied_fraction_heatmap": os.path.abspath(valid_heatmap_path),
            "shell_metrics_csv": os.path.abspath(shell_metrics_csv_path),
        },
        "step_mean_vpar_rmse": step_rmse,
        "step_valid_fraction": {
            f"step_{step_index + 1}": float(np.mean(valid_mask[:, step_index]))
            for step_index in range(valid_mask.shape[1])
        },
    }

    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"[plot] focused species diagnostics saved in {output_dir}", flush=True)
    print(f"[plot] summary saved to {summary_path}", flush=True)


if __name__ == "__main__":
    main()
