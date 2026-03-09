#!/usr/bin/env python3
"""Evaluate a rollout-fine-tuned PWOM field-line graph checkpoint.

Physical interpretation:
- Rollout evaluation measures whether the shell-distribution graph remains accurate when its own
  predicted shell states are fed back autoregressively.
- Metrics are reported for the first predicted step and for the full rollout horizon so local fit
  and temporal stability can be separated.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

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
    DYNAMIC_DIAGNOSTIC_FEATURE_NAMES,
    DYNAMIC_NODE_FEATURE_NAMES,
    FieldlineRolloutDataset,
    FieldlineWindowNormalization,
    build_boundary_source_node_mask,
    build_transport_cell_width_per_node,
    build_species_node_mask,
    build_transport_interfaces,
    build_fieldline_rollout_arrays,
    extract_dynamic_diagnostic_channel,
    load_fieldline_graph_series,
)
from pwom_surrogate.fieldline_graph_model import FieldlineGraphForecaster
from train_fieldline_graph import (
    build_support_band_node_mask,
    compute_effective_diagnostic_weight,
    select_device,
)
from train_fieldline_graph_rollout import (
    compute_band_diagnostic_metrics,
    rollout_forward,
    stats_to_torch,
)


def parse_args() -> argparse.Namespace:
    """Parse rollout evaluator CLI arguments."""
    parser = argparse.ArgumentParser(description="Evaluate a rollout-fine-tuned PWOM surrogate checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-dir", type=str, default="")
    parser.add_argument("--split", choices=["train", "valid"], default="valid")
    parser.add_argument("--horizon", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=["cpu", "mps", "cuda", "auto"], default="auto")
    parser.add_argument("--output-json", type=str, default="")
    parser.add_argument("--verbose-data", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Load a rollout checkpoint and report rollout metrics on one split."""
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
    shell_edges_km = checkpoint["shell_edges_km"]

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
        transport_cell_width_km=np.asarray(
            checkpoint.get(
                "transport_cell_width_km",
                build_transport_cell_width_per_node(
                    layout=series.layout,
                    shell_edges_km=series.shell_edges_km,
                ),
            ),
            dtype=np.float32,
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
        transport_interface_width_km=np.asarray(
            checkpoint.get(
                "transport_interface_width_km",
                build_transport_interfaces(
                    layout=series.layout,
                    shell_edges_km=series.shell_edges_km,
                ).interface_width_km,
            ),
            dtype=np.float32,
        ),
    )

    total_examples = 0
    total_loss = 0.0
    total_rollout_loss = 0.0
    total_rollout_tail_loss = 0.0
    total_one_step_loss = 0.0
    total_support_band_rollout_loss = 0.0
    total_support_band_rollout_tail_loss = 0.0
    total_support_band_one_step_loss = 0.0
    total_secondary_support_band_rollout_loss = 0.0
    total_secondary_support_band_rollout_tail_loss = 0.0
    total_secondary_support_band_one_step_loss = 0.0
    total_diagnostic_mean_vpar_rollout_loss = 0.0
    total_diagnostic_mean_vpar_rollout_tail_loss = 0.0
    total_diagnostic_mean_vpar_one_step_loss = 0.0
    total_diagnostic_mean_vpar_rollout_valid_fraction = 0.0
    total_diagnostic_mean_vpar_one_step_valid_fraction = 0.0
    total_localized_support_band_mean_vpar_rollout_loss = 0.0
    total_localized_support_band_mean_vpar_rollout_tail_loss = 0.0
    total_localized_support_band_mean_vpar_one_step_loss = 0.0
    total_localized_support_band_mean_vpar_rollout_valid_fraction = 0.0
    total_localized_support_band_mean_vpar_one_step_valid_fraction = 0.0
    total_diagnostic_std_vpar_rollout_loss = 0.0
    total_diagnostic_std_vpar_rollout_tail_loss = 0.0
    total_diagnostic_std_vpar_one_step_loss = 0.0
    total_diagnostic_std_vpar_rollout_valid_fraction = 0.0
    total_diagnostic_std_vpar_one_step_valid_fraction = 0.0
    total_rollout_raw_mse = 0.0
    total_one_step_raw_mse = 0.0
    rollout_channel_sq_error = torch.zeros(len(DYNAMIC_NODE_FEATURE_NAMES), dtype=torch.float64)
    one_step_channel_sq_error = torch.zeros(len(DYNAMIC_NODE_FEATURE_NAMES), dtype=torch.float64)
    rollout_channel_count = 0
    one_step_channel_count = 0
    pred_batches: list[np.ndarray] = []
    true_batches: list[np.ndarray] = []
    support_band_node_mask_np = np.asarray(
        checkpoint.get(
            "support_band_node_mask",
            build_support_band_node_mask(
                layout=series.layout,
                shell_edges_km=series.shell_edges_km,
                lower_km=float(train_cfg.get("support_band_lower_km", 2900.0)),
                upper_km=float(train_cfg.get("support_band_upper_km", 4700.0)),
            ),
        ),
        dtype=bool,
    )
    focus_species_support_band_node_mask_np = None
    selection_focus_species_id = train_cfg.get("selection_focus_species_id")
    if selection_focus_species_id is not None:
        # Evaluation mirrors the trainer's species-specific support-band mask so specialized
        # checkpoints are ranked and inspected against the same local physical failure mode.
        focus_species_node_mask = build_species_node_mask(
            layout=series.layout,
            species_id=int(selection_focus_species_id),
        )
        focus_species_support_band_node_mask_np = np.logical_and(
            support_band_node_mask_np,
            np.asarray(focus_species_node_mask, dtype=bool),
        )

    # Standalone rollout evaluation replays the same effective primitive-diagnostic weights that
    # were active at the checkpoint epoch so objective comparisons stay consistent with training.
    effective_mean_vpar_loss_weight = compute_effective_diagnostic_weight(
        base_weight=float(train_cfg.get("diagnostic_mean_vpar_loss_weight", 0.0)),
        epoch_index=int(checkpoint.get("epoch", 1)),
        warmup_epochs=int(train_cfg.get("diagnostic_loss_warmup_epochs", 0)),
        warmup_start_factor=float(train_cfg.get("diagnostic_loss_warmup_start_factor", 0.0)),
    )
    effective_std_vpar_loss_weight = compute_effective_diagnostic_weight(
        base_weight=float(train_cfg.get("diagnostic_std_vpar_loss_weight", 0.0)),
        epoch_index=int(checkpoint.get("epoch", 1)),
        warmup_epochs=int(train_cfg.get("diagnostic_loss_warmup_epochs", 0)),
        warmup_start_factor=float(train_cfg.get("diagnostic_loss_warmup_start_factor", 0.0)),
    )

    iterator = tqdm(loader, desc=f"roll-eval:{args.split}", dynamic_ncols=True)
    with torch.no_grad():
        for batch in iterator:
            history_dynamic = batch["history_dynamic"].to(device=device, dtype=torch.float32)
            history_global_context = batch["history_global_context"].to(device=device, dtype=torch.float32)
            history_boundary_context = batch["history_boundary_context"].to(device=device, dtype=torch.float32)
            target_dynamic_seq = batch["target_dynamic_seq"].to(device=device, dtype=torch.float32)
            target_dynamic_seq_physical = batch["target_dynamic_seq_physical"].to(
                device=device,
                dtype=torch.float32,
            )
            target_global_context_seq = batch["target_global_context_seq"].to(
                device=device,
                dtype=torch.float32,
            )
            target_boundary_context_seq = batch["target_boundary_context_seq"].to(
                device=device,
                dtype=torch.float32,
            )

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
                localized_support_band_mean_vpar_loss_weight=float(
                    train_cfg.get("localized_support_band_mean_vpar_loss_weight", 0.0)
                ),
                occupied_shell_threshold=float(train_cfg.get("occupied_shell_threshold", 1.0e-3)),
                diagnostic_mean_vpar_loss_weight=float(effective_mean_vpar_loss_weight),
                diagnostic_std_vpar_loss_weight=float(effective_std_vpar_loss_weight),
                diagnostic_mean_vpar_scale=float(checkpoint.get("diagnostic_mean_vpar_scale", 1.0)),
                diagnostic_std_vpar_scale=float(checkpoint.get("diagnostic_std_vpar_scale", 1.0)),
                teacher_forcing_prob=0.0,
                training=False,
                diagnostic_band_mask_mode=str(train_cfg.get("diagnostic_band_mask_mode", "global")),
            )

            batch_size = int(history_dynamic.shape[0])
            total_examples += batch_size
            total_loss += float(outputs["total_loss"].detach().cpu()) * batch_size
            total_rollout_loss += float(outputs["rollout_loss"].detach().cpu()) * batch_size
            total_rollout_tail_loss += float(outputs["rollout_tail_loss"].detach().cpu()) * batch_size
            total_one_step_loss += float(outputs["one_step_loss"].detach().cpu()) * batch_size
            total_support_band_rollout_loss += float(outputs["support_band_rollout_loss"].detach().cpu()) * batch_size
            total_support_band_rollout_tail_loss += float(outputs["support_band_rollout_tail_loss"].detach().cpu()) * batch_size
            total_support_band_one_step_loss += float(outputs["support_band_one_step_loss"].detach().cpu()) * batch_size
            total_secondary_support_band_rollout_loss += float(outputs["secondary_support_band_rollout_loss"].detach().cpu()) * batch_size
            total_secondary_support_band_rollout_tail_loss += float(outputs["secondary_support_band_rollout_tail_loss"].detach().cpu()) * batch_size
            total_secondary_support_band_one_step_loss += float(outputs["secondary_support_band_one_step_loss"].detach().cpu()) * batch_size
            total_diagnostic_mean_vpar_rollout_loss += float(outputs["diagnostic_mean_vpar_rollout_loss"].detach().cpu()) * batch_size
            total_diagnostic_mean_vpar_rollout_tail_loss += float(outputs["diagnostic_mean_vpar_rollout_tail_loss"].detach().cpu()) * batch_size
            total_diagnostic_mean_vpar_one_step_loss += float(outputs["diagnostic_mean_vpar_one_step_loss"].detach().cpu()) * batch_size
            total_diagnostic_mean_vpar_rollout_valid_fraction += float(outputs["diagnostic_mean_vpar_rollout_valid_fraction"].detach().cpu()) * batch_size
            total_diagnostic_mean_vpar_one_step_valid_fraction += float(outputs["diagnostic_mean_vpar_one_step_valid_fraction"].detach().cpu()) * batch_size
            total_localized_support_band_mean_vpar_rollout_loss += float(outputs["localized_support_band_mean_vpar_rollout_loss"].detach().cpu()) * batch_size
            total_localized_support_band_mean_vpar_rollout_tail_loss += float(outputs["localized_support_band_mean_vpar_rollout_tail_loss"].detach().cpu()) * batch_size
            total_localized_support_band_mean_vpar_one_step_loss += float(outputs["localized_support_band_mean_vpar_one_step_loss"].detach().cpu()) * batch_size
            total_localized_support_band_mean_vpar_rollout_valid_fraction += float(outputs["localized_support_band_mean_vpar_rollout_valid_fraction"].detach().cpu()) * batch_size
            total_localized_support_band_mean_vpar_one_step_valid_fraction += float(outputs["localized_support_band_mean_vpar_one_step_valid_fraction"].detach().cpu()) * batch_size
            total_diagnostic_std_vpar_rollout_loss += float(outputs["diagnostic_std_vpar_rollout_loss"].detach().cpu()) * batch_size
            total_diagnostic_std_vpar_rollout_tail_loss += float(outputs["diagnostic_std_vpar_rollout_tail_loss"].detach().cpu()) * batch_size
            total_diagnostic_std_vpar_one_step_loss += float(outputs["diagnostic_std_vpar_one_step_loss"].detach().cpu()) * batch_size
            total_diagnostic_std_vpar_rollout_valid_fraction += float(outputs["diagnostic_std_vpar_rollout_valid_fraction"].detach().cpu()) * batch_size
            total_diagnostic_std_vpar_one_step_valid_fraction += float(outputs["diagnostic_std_vpar_one_step_valid_fraction"].detach().cpu()) * batch_size
            total_rollout_raw_mse += float(outputs["rollout_raw_mse"].detach().cpu()) * batch_size
            total_one_step_raw_mse += float(outputs["one_step_raw_mse"].detach().cpu()) * batch_size

            pred_dynamic_seq = outputs["pred_dynamic_seq_physical"].detach().cpu()
            target_dynamic_seq_cpu = target_dynamic_seq_physical.detach().cpu()
            pred_batches.append(pred_dynamic_seq.numpy())
            true_batches.append(target_dynamic_seq_cpu.numpy())
            rollout_channel_sq_error += torch.sum(
                torch.square(pred_dynamic_seq - target_dynamic_seq_cpu),
                dim=(0, 1, 2),
                dtype=torch.float64,
            )
            one_step_channel_sq_error += torch.sum(
                torch.square(pred_dynamic_seq[:, 0] - target_dynamic_seq_cpu[:, 0]),
                dim=(0, 1),
                dtype=torch.float64,
            )
            rollout_channel_count += int(pred_dynamic_seq.shape[0] * pred_dynamic_seq.shape[1] * pred_dynamic_seq.shape[2])
            one_step_channel_count += int(pred_dynamic_seq.shape[0] * pred_dynamic_seq.shape[2])
            iterator.set_postfix(
                total=f"{float(outputs['total_loss'].detach().cpu()):.5f}",
                roll=f"{float(outputs['rollout_loss'].detach().cpu()):.5f}",
                one=f"{float(outputs['one_step_loss'].detach().cpu()):.5f}",
                band=f"{float(outputs['support_band_rollout_loss'].detach().cpu()):.5f}",
                upband=f"{float(outputs['secondary_support_band_rollout_loss'].detach().cpu()):.5f}",
                vpar=f"{float(outputs['diagnostic_mean_vpar_rollout_loss'].detach().cpu()):.5f}",
                lvpar=f"{float(outputs['localized_support_band_mean_vpar_rollout_loss'].detach().cpu()):.5f}",
                svpar=f"{float(outputs['diagnostic_std_vpar_rollout_loss'].detach().cpu()):.5f}",
            )

    if total_examples == 0:
        raise ValueError("Rollout evaluation dataset is empty.")

    pred_dynamic_seq_all = np.concatenate(pred_batches, axis=0)
    true_dynamic_seq_all = np.concatenate(true_batches, axis=0)
    diagnostic_rollout_channel_mse: dict[str, float] = {}
    diagnostic_one_step_channel_mse: dict[str, float] = {}
    for feature_name in DYNAMIC_DIAGNOSTIC_FEATURE_NAMES:
        # Diagnostics recover physical shell means from density-weighted targets so evaluation
        # remains interpretable after the representation change.
        pred_channel = extract_dynamic_diagnostic_channel(
            dynamic_physical=pred_dynamic_seq_all,
            feature_name=feature_name,
        )
        true_channel = extract_dynamic_diagnostic_channel(
            dynamic_physical=true_dynamic_seq_all,
            feature_name=feature_name,
        )
        diagnostic_rollout_channel_mse[feature_name] = float(np.mean(np.square(pred_channel - true_channel)))
        diagnostic_one_step_channel_mse[feature_name] = float(
            np.mean(np.square(pred_channel[:, 0] - true_channel[:, 0]))
        )
    # These band-local diagnostics reproduce the localized velocity-evaluation regime used for the
    # specialized rollout artifacts, so post-run inspection matches training-time checkpointing.
    (
        support_band_diagnostic_rollout_channel_mse,
        support_band_diagnostic_rollout_valid_fraction,
        support_band_diagnostic_one_step_channel_mse,
        support_band_diagnostic_one_step_valid_fraction,
    ) = compute_band_diagnostic_metrics(
        pred_dynamic_physical=pred_dynamic_seq_all,
        true_dynamic_physical=true_dynamic_seq_all,
        occupied_shell_threshold=float(train_cfg.get("occupied_shell_threshold", 1.0e-3)),
        target_schema=str(data_cfg.get("target_schema", "population_weighted")),
        node_mask=support_band_node_mask_np,
    )
    focus_species_support_band_diagnostic_rollout_channel_mse: dict[str, float] = {}
    focus_species_support_band_diagnostic_rollout_valid_fraction: dict[str, float] = {}
    focus_species_support_band_diagnostic_one_step_channel_mse: dict[str, float] = {}
    focus_species_support_band_diagnostic_one_step_valid_fraction: dict[str, float] = {}
    if focus_species_support_band_node_mask_np is not None:
        (
            focus_species_support_band_diagnostic_rollout_channel_mse,
            focus_species_support_band_diagnostic_rollout_valid_fraction,
            focus_species_support_band_diagnostic_one_step_channel_mse,
            focus_species_support_band_diagnostic_one_step_valid_fraction,
        ) = compute_band_diagnostic_metrics(
            pred_dynamic_physical=pred_dynamic_seq_all,
            true_dynamic_physical=true_dynamic_seq_all,
            occupied_shell_threshold=float(train_cfg.get("occupied_shell_threshold", 1.0e-3)),
            target_schema=str(data_cfg.get("target_schema", "population_weighted")),
            node_mask=focus_species_support_band_node_mask_np,
        )

    metrics = {
        "checkpoint": os.path.abspath(args.checkpoint),
        "split": args.split,
        "num_samples": total_examples,
        "window_length": int(data_cfg["window_length"]),
        "eval_horizon": int(eval_horizon),
        "num_altitude_shells": int(data_cfg["num_altitude_shells"]),
        "num_particles": None if data_cfg["num_particles"] is None else int(data_cfg["num_particles"]),
        "species_ids": list(species_ids),
        "total_loss": total_loss / total_examples,
        "rollout_loss": total_rollout_loss / total_examples,
        "rollout_tail_loss": total_rollout_tail_loss / total_examples,
        "one_step_loss": total_one_step_loss / total_examples,
        "support_band_rollout_loss": total_support_band_rollout_loss / total_examples,
        "support_band_rollout_tail_loss": total_support_band_rollout_tail_loss / total_examples,
        "support_band_one_step_loss": total_support_band_one_step_loss / total_examples,
        "secondary_support_band_rollout_loss": total_secondary_support_band_rollout_loss / total_examples,
        "secondary_support_band_rollout_tail_loss": total_secondary_support_band_rollout_tail_loss / total_examples,
        "secondary_support_band_one_step_loss": total_secondary_support_band_one_step_loss / total_examples,
        "diagnostic_mean_vpar_rollout_loss": total_diagnostic_mean_vpar_rollout_loss / total_examples,
        "diagnostic_mean_vpar_rollout_tail_loss": total_diagnostic_mean_vpar_rollout_tail_loss / total_examples,
        "diagnostic_mean_vpar_one_step_loss": total_diagnostic_mean_vpar_one_step_loss / total_examples,
        "diagnostic_mean_vpar_rollout_valid_fraction": total_diagnostic_mean_vpar_rollout_valid_fraction / total_examples,
        "diagnostic_mean_vpar_one_step_valid_fraction": total_diagnostic_mean_vpar_one_step_valid_fraction / total_examples,
        "localized_support_band_mean_vpar_rollout_loss": total_localized_support_band_mean_vpar_rollout_loss / total_examples,
        "localized_support_band_mean_vpar_rollout_tail_loss": total_localized_support_band_mean_vpar_rollout_tail_loss / total_examples,
        "localized_support_band_mean_vpar_one_step_loss": total_localized_support_band_mean_vpar_one_step_loss / total_examples,
        "localized_support_band_mean_vpar_rollout_valid_fraction": total_localized_support_band_mean_vpar_rollout_valid_fraction / total_examples,
        "localized_support_band_mean_vpar_one_step_valid_fraction": total_localized_support_band_mean_vpar_one_step_valid_fraction / total_examples,
        "diagnostic_std_vpar_rollout_loss": total_diagnostic_std_vpar_rollout_loss / total_examples,
        "diagnostic_std_vpar_rollout_tail_loss": total_diagnostic_std_vpar_rollout_tail_loss / total_examples,
        "diagnostic_std_vpar_one_step_loss": total_diagnostic_std_vpar_one_step_loss / total_examples,
        "diagnostic_std_vpar_rollout_valid_fraction": total_diagnostic_std_vpar_rollout_valid_fraction / total_examples,
        "diagnostic_std_vpar_one_step_valid_fraction": total_diagnostic_std_vpar_one_step_valid_fraction / total_examples,
        "effective_diagnostic_mean_vpar_loss_weight": float(effective_mean_vpar_loss_weight),
        "effective_diagnostic_std_vpar_loss_weight": float(effective_std_vpar_loss_weight),
        "rollout_raw_mse": total_rollout_raw_mse / total_examples,
        "one_step_raw_mse": total_one_step_raw_mse / total_examples,
        "rollout_channel_mse": {
            name: float(value / max(rollout_channel_count, 1))
            for name, value in zip(DYNAMIC_NODE_FEATURE_NAMES, rollout_channel_sq_error.tolist())
        },
        "one_step_channel_mse": {
            name: float(value / max(one_step_channel_count, 1))
            for name, value in zip(DYNAMIC_NODE_FEATURE_NAMES, one_step_channel_sq_error.tolist())
        },
        "diagnostic_rollout_channel_mse": diagnostic_rollout_channel_mse,
        "diagnostic_one_step_channel_mse": diagnostic_one_step_channel_mse,
        "support_band_diagnostic_rollout_channel_mse": support_band_diagnostic_rollout_channel_mse,
        "support_band_diagnostic_rollout_valid_fraction": support_band_diagnostic_rollout_valid_fraction,
        "support_band_diagnostic_one_step_channel_mse": support_band_diagnostic_one_step_channel_mse,
        "support_band_diagnostic_one_step_valid_fraction": support_band_diagnostic_one_step_valid_fraction,
        "selection_focus_species_id": (
            None if selection_focus_species_id is None else int(selection_focus_species_id)
        ),
        "focus_species_support_band_diagnostic_rollout_channel_mse": (
            focus_species_support_band_diagnostic_rollout_channel_mse
        ),
        "focus_species_support_band_diagnostic_rollout_valid_fraction": (
            focus_species_support_band_diagnostic_rollout_valid_fraction
        ),
        "focus_species_support_band_diagnostic_one_step_channel_mse": (
            focus_species_support_band_diagnostic_one_step_channel_mse
        ),
        "focus_species_support_band_diagnostic_one_step_valid_fraction": (
            focus_species_support_band_diagnostic_one_step_valid_fraction
        ),
    }

    print("[rollout-eval] evaluation metrics", flush=True)
    print(metrics, flush=True)

    if args.output_json:
        output_path = os.path.abspath(args.output_json)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2)
        print(f"[rollout-eval] saved {output_path}", flush=True)


if __name__ == "__main__":
    main()
