#!/usr/bin/env python3
"""Fine-tune the PWOM field-line graph model for short autoregressive rollouts.

Physical interpretation:
- The one-step checkpoint already learns the local next-step shell dynamics.
- This rollout stage asks the same model to remain stable when its own predicted shell state is fed
  back into the next timestep, which is the first real surrogate requirement beyond one-step fit.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Allow running this script directly from the repository root.
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from pwom_surrogate.config import ModelConfig, TrainConfig
from pwom_surrogate.fieldline_graph_data import (
    TARGET_SCHEMA_POPULATION_WEIGHTED,
    build_boundary_source_node_mask,
    build_transport_cell_width_per_node,
    build_species_node_mask,
    build_transport_interfaces,
    DYNAMIC_DIAGNOSTIC_FEATURE_NAMES,
    DYNAMIC_NODE_FEATURE_NAMES,
    FieldlineRolloutDataset,
    FieldlineWindowNormalization,
    build_velocity_species_weight_metadata,
    build_fieldline_rollout_arrays,
    dynamic_diagnostic_valid_mask,
    extract_dynamic_diagnostic_channel,
    load_fieldline_graph_series,
)
from pwom_surrogate.fieldline_graph_model import FieldlineGraphForecaster
from train_fieldline_graph import (
    _stats_to_torch,
    apply_dynamic_constraints_model_space,
    build_support_band_node_mask,
    compute_diagnostic_scale,
    compute_effective_diagnostic_weight,
    inverse_transform_dynamic_torch,
    masked_mean_vpar_diagnostic_loss,
    masked_std_vpar_diagnostic_loss,
    reconstruct_prediction_raw,
    select_device,
    set_seed,
    support_band_loss,
    weighted_normalized_dynamic_mse,
)


def parse_args() -> argparse.Namespace:
    """Parse rollout fine-tuning CLI arguments."""
    parser = argparse.ArgumentParser(description="Fine-tune the PWOM surrogate model for short rollouts.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="runs/fieldline_rollout_h3")
    parser.add_argument("--data-dir", type=str, default="")
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--rollout-weight", type=float, default=0.2)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--teacher-forcing-prob", type=float, default=0.5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--early-stopping-patience", type=int, default=20)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-4)
    parser.add_argument(
        "--velocity-species-balance-power",
        type=float,
        default=None,
        help="Optional override for the one-step checkpoint's species-aware velocity weighting power.",
    )
    parser.add_argument(
        "--velocity-species-balance-max-weight",
        type=float,
        default=None,
        help="Optional override for the one-step checkpoint's maximum species-aware velocity weight.",
    )
    parser.add_argument(
        "--diagnostic-mean-vpar-loss-weight",
        type=float,
        default=None,
        help="Optional override for the auxiliary occupied-shell mean_vpar diagnostic loss weight.",
    )
    parser.add_argument(
        "--diagnostic-std-vpar-loss-weight",
        type=float,
        default=None,
        help="Optional override for the auxiliary occupied-shell std_vpar diagnostic loss weight.",
    )
    parser.add_argument(
        "--diagnostic-loss-warmup-epochs",
        type=int,
        default=None,
        help="Optional override for the primitive-diagnostic loss warmup length.",
    )
    parser.add_argument(
        "--diagnostic-loss-warmup-start-factor",
        type=float,
        default=None,
        help="Optional override for the primitive-diagnostic loss warmup start factor.",
    )
    parser.add_argument(
        "--support-band-loss-weight",
        type=float,
        default=None,
        help="Optional override for the focused support-collapse loss weight.",
    )
    parser.add_argument(
        "--support-band-lower-km",
        type=float,
        default=None,
        help="Optional override for the lower altitude bound of the support-collapse band.",
    )
    parser.add_argument(
        "--support-band-upper-km",
        type=float,
        default=None,
        help="Optional override for the upper altitude bound of the support-collapse band.",
    )
    parser.add_argument(
        "--secondary-support-band-loss-weight",
        type=float,
        default=None,
        help="Optional override for the weaker residual upper-band support loss weight.",
    )
    parser.add_argument(
        "--secondary-support-band-lower-km",
        type=float,
        default=None,
        help="Optional override for the lower altitude bound of the residual upper support band.",
    )
    parser.add_argument(
        "--secondary-support-band-upper-km",
        type=float,
        default=None,
        help="Optional override for the upper altitude bound of the residual upper support band.",
    )
    parser.add_argument(
        "--localized-support-band-mean-vpar-loss-weight",
        type=float,
        default=0.0,
        help="Small occupied-shell mean_vpar corrective loss applied only in the primary support band.",
    )
    parser.add_argument("--device", choices=["cpu", "mps", "cuda", "auto"], default="auto")
    parser.add_argument(
        "--diagnostic-band-mask-mode",
        choices=["global", "support_band", "secondary_support_band"],
        default=None,
        help="Optional override for localizing primitive velocity diagnostics to one altitude band.",
    )
    parser.add_argument(
        "--selection-metric",
        choices=[
            "valid_total_loss",
            "composite_velocity",
            "physical_mean_vpar_rollout_mse",
            "physical_std_vpar_rollout_mse",
            "physical_band_mean_vpar_rollout_mse",
            "physical_band_std_vpar_rollout_mse",
            "physical_focus_band_mean_vpar_rollout_mse",
        ],
        default="valid_total_loss",
        help="Checkpoint selection metric used for best.pt and early stopping.",
    )
    parser.add_argument(
        "--selection-focus-species-id",
        type=int,
        default=None,
        help="Optional species id for species-specific support-band checkpoint selection.",
    )
    parser.add_argument(
        "--selection-focus-rollout-loss-cap",
        type=float,
        default=None,
        help="Optional rollout-loss cap required before saving the species-focus support-band checkpoint.",
    )
    parser.add_argument(
        "--selection-rollout-loss-weight",
        type=float,
        default=1.0,
        help="Rollout-loss weight in the composite checkpoint-selection score.",
    )
    parser.add_argument(
        "--selection-mean-vpar-weight",
        type=float,
        default=0.02,
        help="Occupied-shell mean_vpar diagnostic weight in the composite checkpoint-selection score.",
    )
    parser.add_argument(
        "--selection-std-vpar-weight",
        type=float,
        default=0.01,
        help="Occupied-shell std_vpar diagnostic weight in the composite checkpoint-selection score.",
    )
    parser.add_argument(
        "--save-epoch-checkpoints",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Save checkpoint_epoch_<n>.pt files so diagnostic-best epochs can be recovered explicitly.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose-data", action="store_true")
    return parser.parse_args()


def stats_to_torch(
    stats: FieldlineWindowNormalization,
    device: torch.device,
    num_nodes: int,
    target_schema: str = TARGET_SCHEMA_POPULATION_WEIGHTED,
    velocity_species_weight_per_node: np.ndarray | None = None,
    boundary_source_node_mask: np.ndarray | None = None,
    support_band_node_mask: np.ndarray | None = None,
    secondary_support_band_node_mask: np.ndarray | None = None,
    transport_interfaces=None,
    transport_delta_scale: np.ndarray | None = None,
    transport_shell_node_mask: np.ndarray | None = None,
    transport_cell_width_km: np.ndarray | None = None,
    transport_interface_width_km: np.ndarray | None = None,
) -> dict[str, torch.Tensor]:
    """Move normalization tensors to the active device for rollout reconstruction.

    Physical interpretation:
    - Rollout must use the same normalization, shell-interface metadata, and conservative masks as
      one-step training; otherwise operator-split transport/source updates are evaluated against a
      different state contract than the checkpoint was trained with.
    """
    return _stats_to_torch(
        stats=stats,
        device=device,
        num_nodes=num_nodes,
        target_schema=target_schema,
        velocity_species_weight_per_node=velocity_species_weight_per_node,
        boundary_source_node_mask=boundary_source_node_mask,
        support_band_node_mask=support_band_node_mask,
        secondary_support_band_node_mask=secondary_support_band_node_mask,
        transport_interfaces=transport_interfaces,
        transport_delta_scale=transport_delta_scale,
        transport_shell_node_mask=transport_shell_node_mask,
        transport_cell_width_km=transport_cell_width_km,
        transport_interface_width_km=transport_interface_width_km,
    )


def compose_next_dynamic(
    pred_dynamic: torch.Tensor,
    true_dynamic: torch.Tensor,
    stats_torch: dict[str, torch.Tensor],
    teacher_forcing_prob: float,
    training: bool,
) -> torch.Tensor:
    """Build the next rollout history state from predicted state and true exogenous context.

    Physical interpretation:
    - State channels evolve autoregressively.
    - Exogenous/context channels are supplied from the true future sequence because they represent
      forcing or topology, not endogenous plasma dynamics.
    """
    state_index = stats_torch["state_channel_index"]
    context_index = stats_torch["context_channel_index"]
    next_dynamic = pred_dynamic.clone()

    if int(context_index.numel()) > 0:
        next_dynamic[..., context_index] = true_dynamic[..., context_index]

    if training and teacher_forcing_prob > 0.0 and int(state_index.numel()) > 0:
        teacher_mask = (
            torch.rand(
                (pred_dynamic.shape[0], 1, 1),
                device=pred_dynamic.device,
                dtype=pred_dynamic.dtype,
            )
            < teacher_forcing_prob
        )
        next_dynamic_state = torch.where(
            teacher_mask,
            true_dynamic[..., state_index],
            next_dynamic[..., state_index],
        )
        next_dynamic[..., state_index] = next_dynamic_state

    return next_dynamic


def build_rollout_input_features(
    static_node_features: torch.Tensor,
    history_dynamic: torch.Tensor,
    stats_torch: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Construct normalized graph inputs from the current rollout history.

    Physical interpretation:
    - The model always sees a short shell-state history, not just the last frame.
    - This keeps the rollout state tied to the recent temporal context that can encode unresolved
      transport lag, shell refilling, and species-coupled response.
    """
    batch_size, window_length, num_nodes, dynamic_dim = history_dynamic.shape
    history_flat = history_dynamic.permute(0, 2, 1, 3).reshape(
        batch_size,
        num_nodes,
        window_length * dynamic_dim,
    )
    static_expanded = static_node_features.unsqueeze(0).expand(batch_size, -1, -1)
    input_features = torch.cat((static_expanded, history_flat), dim=-1)
    return (input_features - stats_torch["input_mean"]) / stats_torch["input_std"]


def build_rollout_global_context_features(
    history_global_context: torch.Tensor,
    stats_torch: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Construct normalized line-global PWOM context history for rollout conditioning.

    Physical interpretation:
    - Latitude, timestep, and regime flags are exogenous line controls, so they should reach the
      model through their own encoder instead of being mixed into shell-state channels.
    """
    batch_size, window_length, context_dim = history_global_context.shape
    if int(stats_torch["context_mean"].numel()) == 0 or context_dim == 0:
        # Old checkpoints do not use the separate global-context path, so rollout should feed an
        # empty conditioning tensor that the model safely ignores.
        return history_global_context.new_zeros((batch_size, 0))
    history_flat = history_global_context.reshape(batch_size, window_length * context_dim)
    return (history_flat - stats_torch["context_mean"]) / stats_torch["context_std"]


def build_rollout_boundary_context_features(
    current_boundary_context: torch.Tensor,
    stats_torch: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Normalize the current lower-boundary control tensor for the boundary injection head.

    Physical interpretation:
    - Boundary drivers are exogenous controls applied at the lowest shell.
    - The rollout branch should feed the true current boundary controls to the dedicated source
      head instead of pretending they are autoregressive shell-state channels.
    """
    if int(stats_torch["boundary_context_mean"].numel()) == 0 or current_boundary_context.shape[-1] == 0:
        return current_boundary_context.new_zeros(current_boundary_context.shape[:-1] + (0,))
    return (
        current_boundary_context - stats_torch["boundary_context_mean"]
    ) / stats_torch["boundary_context_std"]


def resolve_diagnostic_node_mask(
    stats_torch: dict[str, torch.Tensor],
    diagnostic_band_mask_mode: str,
) -> torch.Tensor | None:
    """Resolve one optional node mask for localized primitive supervision.

    Physical interpretation:
    - `global` keeps the primitive auxiliary term active on every occupied shell.
    - `support_band` restricts it to the primary mid-altitude failure regime.
    - `secondary_support_band` reserves the same mechanism for the residual upper-altitude band.
    """
    normalized_mode = str(diagnostic_band_mask_mode or "global").strip().lower()
    if normalized_mode in {"", "global"}:
        return None
    if normalized_mode == "support_band":
        return stats_torch["support_band_node_mask"]
    if normalized_mode == "secondary_support_band":
        return stats_torch["secondary_support_band_node_mask"]
    raise ValueError(f"Unsupported diagnostic_band_mask_mode='{diagnostic_band_mask_mode}'.")


def compute_masked_diagnostic_channel_mse(
    pred_dynamic_physical: np.ndarray,
    true_dynamic_physical: np.ndarray,
    feature_name: str,
    occupied_shell_threshold: float,
    target_schema: str,
    node_mask: np.ndarray | None = None,
) -> tuple[float, float]:
    """Compute one physical diagnostic MSE inside an occupied-shell regime mask.

    Physical interpretation:
    - Primitive velocity diagnostics are only meaningful where shell occupancy is non-negligible.
    - The node mask then localizes that diagnostic to the altitude band we are explicitly trying
      to stabilize.
    """
    pred_channel = extract_dynamic_diagnostic_channel(
        dynamic_physical=pred_dynamic_physical,
        feature_name=feature_name,
        target_schema=target_schema,
    )
    true_channel = extract_dynamic_diagnostic_channel(
        dynamic_physical=true_dynamic_physical,
        feature_name=feature_name,
        target_schema=target_schema,
    )
    valid_mask = dynamic_diagnostic_valid_mask(
        dynamic_physical=true_dynamic_physical,
        feature_name=feature_name,
        target_schema=target_schema,
        population_epsilon=float(occupied_shell_threshold),
    )
    if node_mask is not None:
        expanded_node_mask = np.asarray(node_mask, dtype=bool).reshape(
            (1,) * max(valid_mask.ndim - 1, 0) + (len(node_mask),)
        )
        valid_mask = np.logical_and(valid_mask, expanded_node_mask)
    if not np.any(valid_mask):
        return float("inf"), 0.0
    sq_error = np.square(pred_channel - true_channel, dtype=np.float64)
    return float(np.mean(sq_error[valid_mask])), float(np.mean(valid_mask.astype(np.float32)))


def compute_band_diagnostic_metrics(
    pred_dynamic_physical: np.ndarray,
    true_dynamic_physical: np.ndarray,
    occupied_shell_threshold: float,
    target_schema: str,
    node_mask: np.ndarray,
) -> tuple[dict[str, float], dict[str, float], dict[str, float], dict[str, float]]:
    """Summarize support-band primitive diagnostics for rollout and first-step comparisons.

    Physical interpretation:
    - These metrics isolate the altitude band where shell support collapse and `mean_vpar` drift
      are coupled, so checkpoint selection can target the actual physical failure mode.
    """
    rollout_metrics: dict[str, float] = {}
    rollout_valid_fraction: dict[str, float] = {}
    one_step_metrics: dict[str, float] = {}
    one_step_valid_fraction: dict[str, float] = {}
    for feature_name in ("mean_vpar", "std_vpar"):
        rollout_mse, rollout_fraction = compute_masked_diagnostic_channel_mse(
            pred_dynamic_physical=pred_dynamic_physical,
            true_dynamic_physical=true_dynamic_physical,
            feature_name=feature_name,
            occupied_shell_threshold=occupied_shell_threshold,
            target_schema=target_schema,
            node_mask=node_mask,
        )
        one_step_mse, one_step_fraction = compute_masked_diagnostic_channel_mse(
            pred_dynamic_physical=pred_dynamic_physical[:, 0],
            true_dynamic_physical=true_dynamic_physical[:, 0],
            feature_name=feature_name,
            occupied_shell_threshold=occupied_shell_threshold,
            target_schema=target_schema,
            node_mask=node_mask,
        )
        rollout_metrics[feature_name] = rollout_mse
        rollout_valid_fraction[feature_name] = rollout_fraction
        one_step_metrics[feature_name] = one_step_mse
        one_step_valid_fraction[feature_name] = one_step_fraction
    return rollout_metrics, rollout_valid_fraction, one_step_metrics, one_step_valid_fraction


def rollout_forward(
    model: FieldlineGraphForecaster,
    static_node_features: torch.Tensor,
    history_dynamic: torch.Tensor,
    history_global_context: torch.Tensor,
    history_boundary_context: torch.Tensor,
    target_dynamic_seq: torch.Tensor,
    target_dynamic_seq_physical: torch.Tensor,
    target_global_context_seq: torch.Tensor,
    target_boundary_context_seq: torch.Tensor,
    edge_index: torch.Tensor,
    stats_torch: dict[str, torch.Tensor],
    predict_delta_state: bool,
    rollout_weight: float,
    support_band_loss_weight: float,
    secondary_support_band_loss_weight: float,
    localized_support_band_mean_vpar_loss_weight: float,
    occupied_shell_threshold: float,
    diagnostic_mean_vpar_loss_weight: float,
    diagnostic_std_vpar_loss_weight: float,
    diagnostic_mean_vpar_scale: float,
    diagnostic_std_vpar_scale: float,
    teacher_forcing_prob: float,
    training: bool,
    diagnostic_band_mask_mode: str = "global",
) -> dict[str, torch.Tensor]:
    """Run one autoregressive rollout and return losses and predictions.

    Physical interpretation:
    - The model predicts one future shell state at a time and then advances the history window.
    - Optional teacher forcing mixes in true future shell states during training so early rollout
      errors do not immediately destroy the temporal context.
    """
    current_history = history_dynamic
    current_history_global_context = history_global_context
    current_history_boundary_context = history_boundary_context
    pred_dynamic_steps: list[torch.Tensor] = []

    horizon = target_dynamic_seq.shape[1]
    for step_index in range(horizon):
        input_features = build_rollout_input_features(
            static_node_features=static_node_features,
            history_dynamic=current_history,
            stats_torch=stats_torch,
        )
        global_context = build_rollout_global_context_features(
            history_global_context=current_history_global_context,
            stats_torch=stats_torch,
        )
        boundary_context = build_rollout_boundary_context_features(
            current_boundary_context=current_history_boundary_context[:, -1],
            stats_torch=stats_torch,
        )
        last_dynamic = current_history[:, -1]
        # The operator-split model advances conservative shell quantities in physical space, so the
        # rollout loop must reconstruct the previous physical shell state before every prediction.
        last_dynamic_physical = inverse_transform_dynamic_torch(
            dynamic_model=last_dynamic,
            stats_torch=stats_torch,
        )
        pred_norm = model(
            node_features=input_features,
            edge_index=edge_index,
            global_context=global_context,
            boundary_context=boundary_context,
            last_dynamic=last_dynamic,
            last_dynamic_physical=last_dynamic_physical,
            stats_torch=stats_torch,
            predict_delta_state=predict_delta_state,
        )
        pred_dynamic = reconstruct_prediction_raw(
            pred_norm=pred_norm,
            last_dynamic=last_dynamic,
            stats_torch=stats_torch,
            predict_delta_state=predict_delta_state,
        )
        pred_dynamic = apply_dynamic_constraints_model_space(
            pred_dynamic=pred_dynamic,
            stats_torch=stats_torch,
        )
        pred_dynamic_steps.append(pred_dynamic)
        next_dynamic = compose_next_dynamic(
            pred_dynamic=pred_dynamic,
            true_dynamic=target_dynamic_seq[:, step_index],
            stats_torch=stats_torch,
            teacher_forcing_prob=teacher_forcing_prob,
            training=training,
        )
        current_history = torch.cat((current_history[:, 1:], next_dynamic.unsqueeze(1)), dim=1)
        next_global_context = target_global_context_seq[:, step_index]
        next_boundary_context = target_boundary_context_seq[:, step_index]
        current_history_global_context = torch.cat(
            (current_history_global_context[:, 1:], next_global_context.unsqueeze(1)),
            dim=1,
        )
        current_history_boundary_context = torch.cat(
            (current_history_boundary_context[:, 1:], next_boundary_context.unsqueeze(1)),
            dim=1,
        )

    pred_dynamic_seq = torch.stack(pred_dynamic_steps, dim=1)

    # The blended objective anchors the first step strongly and then adds a smaller penalty on the
    # later rollout steps. This protects the good one-step solution while still training temporal
    # stability.
    state_index = stats_torch["state_channel_index"]
    one_step_loss = weighted_normalized_dynamic_mse(
        pred_dynamic=pred_dynamic_seq[:, 0],
        target_dynamic=target_dynamic_seq[:, 0],
        stats_torch=stats_torch,
        channel_index=state_index,
    )
    rollout_loss = weighted_normalized_dynamic_mse(
        pred_dynamic=pred_dynamic_seq,
        target_dynamic=target_dynamic_seq,
        stats_torch=stats_torch,
        channel_index=state_index,
    )
    if horizon > 1:
        rollout_tail_loss = weighted_normalized_dynamic_mse(
            pred_dynamic=pred_dynamic_seq[:, 1:],
            target_dynamic=target_dynamic_seq[:, 1:],
            stats_torch=stats_torch,
            channel_index=state_index,
        )
    else:
        rollout_tail_loss = torch.zeros((), dtype=pred_dynamic_seq.dtype, device=pred_dynamic_seq.device)
    support_band_one_step_loss = support_band_loss(
        pred_dynamic=pred_dynamic_seq[:, 0],
        target_dynamic=target_dynamic_seq[:, 0],
        stats_torch=stats_torch,
    )
    support_band_rollout_loss = support_band_loss(
        pred_dynamic=pred_dynamic_seq,
        target_dynamic=target_dynamic_seq,
        stats_torch=stats_torch,
    )
    if horizon > 1:
        support_band_rollout_tail_loss = support_band_loss(
            pred_dynamic=pred_dynamic_seq[:, 1:],
            target_dynamic=target_dynamic_seq[:, 1:],
            stats_torch=stats_torch,
        )
    else:
        support_band_rollout_tail_loss = torch.zeros(
            (),
            dtype=pred_dynamic_seq.dtype,
            device=pred_dynamic_seq.device,
        )
    secondary_support_band_one_step_loss = support_band_loss(
        pred_dynamic=pred_dynamic_seq[:, 0],
        target_dynamic=target_dynamic_seq[:, 0],
        stats_torch=stats_torch,
        mask_key="secondary_support_band_node_mask",
    )
    secondary_support_band_rollout_loss = support_band_loss(
        pred_dynamic=pred_dynamic_seq,
        target_dynamic=target_dynamic_seq,
        stats_torch=stats_torch,
        mask_key="secondary_support_band_node_mask",
    )
    if horizon > 1:
        secondary_support_band_rollout_tail_loss = support_band_loss(
            pred_dynamic=pred_dynamic_seq[:, 1:],
            target_dynamic=target_dynamic_seq[:, 1:],
            stats_torch=stats_torch,
            mask_key="secondary_support_band_node_mask",
        )
    else:
        secondary_support_band_rollout_tail_loss = torch.zeros(
            (),
            dtype=pred_dynamic_seq.dtype,
            device=pred_dynamic_seq.device,
        )
    # This mask isolates the known physical failure regime: occupied shells in the mid-altitude
    # support-collapse band where mean_vpar drift grows across rollout steps.
    localized_support_band_node_mask = stats_torch["support_band_node_mask"]
    pred_dynamic_seq_physical = inverse_transform_dynamic_torch(
        dynamic_model=pred_dynamic_seq,
        stats_torch=stats_torch,
    )
    diagnostic_node_mask = resolve_diagnostic_node_mask(
        stats_torch=stats_torch,
        diagnostic_band_mask_mode=diagnostic_band_mask_mode,
    )
    diagnostic_mean_vpar_one_step_loss, diagnostic_mean_vpar_one_step_valid_fraction = masked_mean_vpar_diagnostic_loss(
        pred_dynamic_physical=pred_dynamic_seq_physical[:, 0],
        target_dynamic_physical=target_dynamic_seq_physical[:, 0],
        occupied_shell_threshold=occupied_shell_threshold,
        diagnostic_mean_vpar_scale=diagnostic_mean_vpar_scale,
        node_mask=diagnostic_node_mask,
    )
    diagnostic_mean_vpar_rollout_loss, diagnostic_mean_vpar_rollout_valid_fraction = masked_mean_vpar_diagnostic_loss(
        pred_dynamic_physical=pred_dynamic_seq_physical,
        target_dynamic_physical=target_dynamic_seq_physical,
        occupied_shell_threshold=occupied_shell_threshold,
        diagnostic_mean_vpar_scale=diagnostic_mean_vpar_scale,
        node_mask=diagnostic_node_mask,
    )
    if horizon > 1:
        diagnostic_mean_vpar_rollout_tail_loss, _ = masked_mean_vpar_diagnostic_loss(
            pred_dynamic_physical=pred_dynamic_seq_physical[:, 1:],
            target_dynamic_physical=target_dynamic_seq_physical[:, 1:],
            occupied_shell_threshold=occupied_shell_threshold,
            diagnostic_mean_vpar_scale=diagnostic_mean_vpar_scale,
            node_mask=diagnostic_node_mask,
        )
    else:
        diagnostic_mean_vpar_rollout_tail_loss = torch.zeros(
            (),
            dtype=pred_dynamic_seq.dtype,
            device=pred_dynamic_seq.device,
        )
    localized_support_band_mean_vpar_one_step_loss, localized_support_band_mean_vpar_one_step_valid_fraction = masked_mean_vpar_diagnostic_loss(
        pred_dynamic_physical=pred_dynamic_seq_physical[:, 0],
        target_dynamic_physical=target_dynamic_seq_physical[:, 0],
        occupied_shell_threshold=occupied_shell_threshold,
        diagnostic_mean_vpar_scale=diagnostic_mean_vpar_scale,
        node_mask=localized_support_band_node_mask,
    )
    localized_support_band_mean_vpar_rollout_loss, localized_support_band_mean_vpar_rollout_valid_fraction = masked_mean_vpar_diagnostic_loss(
        pred_dynamic_physical=pred_dynamic_seq_physical,
        target_dynamic_physical=target_dynamic_seq_physical,
        occupied_shell_threshold=occupied_shell_threshold,
        diagnostic_mean_vpar_scale=diagnostic_mean_vpar_scale,
        node_mask=localized_support_band_node_mask,
    )
    if horizon > 1:
        localized_support_band_mean_vpar_rollout_tail_loss, _ = masked_mean_vpar_diagnostic_loss(
            pred_dynamic_physical=pred_dynamic_seq_physical[:, 1:],
            target_dynamic_physical=target_dynamic_seq_physical[:, 1:],
            occupied_shell_threshold=occupied_shell_threshold,
            diagnostic_mean_vpar_scale=diagnostic_mean_vpar_scale,
            node_mask=localized_support_band_node_mask,
        )
    else:
        localized_support_band_mean_vpar_rollout_tail_loss = torch.zeros(
            (),
            dtype=pred_dynamic_seq.dtype,
            device=pred_dynamic_seq.device,
        )
    diagnostic_std_vpar_one_step_loss, diagnostic_std_vpar_one_step_valid_fraction = masked_std_vpar_diagnostic_loss(
        pred_dynamic_physical=pred_dynamic_seq_physical[:, 0],
        target_dynamic_physical=target_dynamic_seq_physical[:, 0],
        occupied_shell_threshold=occupied_shell_threshold,
        diagnostic_std_vpar_scale=diagnostic_std_vpar_scale,
        node_mask=diagnostic_node_mask,
    )
    diagnostic_std_vpar_rollout_loss, diagnostic_std_vpar_rollout_valid_fraction = masked_std_vpar_diagnostic_loss(
        pred_dynamic_physical=pred_dynamic_seq_physical,
        target_dynamic_physical=target_dynamic_seq_physical,
        occupied_shell_threshold=occupied_shell_threshold,
        diagnostic_std_vpar_scale=diagnostic_std_vpar_scale,
        node_mask=diagnostic_node_mask,
    )
    if horizon > 1:
        diagnostic_std_vpar_rollout_tail_loss, _ = masked_std_vpar_diagnostic_loss(
            pred_dynamic_physical=pred_dynamic_seq_physical[:, 1:],
            target_dynamic_physical=target_dynamic_seq_physical[:, 1:],
            occupied_shell_threshold=occupied_shell_threshold,
            diagnostic_std_vpar_scale=diagnostic_std_vpar_scale,
            node_mask=diagnostic_node_mask,
        )
    else:
        diagnostic_std_vpar_rollout_tail_loss = torch.zeros(
            (),
            dtype=pred_dynamic_seq.dtype,
            device=pred_dynamic_seq.device,
        )
    total_loss = one_step_loss + (rollout_weight * rollout_tail_loss)
    if support_band_loss_weight > 0.0:
        # The focused support term uses the same curriculum structure as the main rollout loss so
        # the problematic band is anchored at the first step and softly penalized on later steps.
        support_total = support_band_one_step_loss + (rollout_weight * support_band_rollout_tail_loss)
        total_loss = total_loss + (float(support_band_loss_weight) * support_total)
    if secondary_support_band_loss_weight > 0.0:
        # The weaker upper-band term addresses the residual high-altitude support-collapse error
        # without broadening the primary band that already improved the mid-altitude regime.
        secondary_support_total = secondary_support_band_one_step_loss + (
            rollout_weight * secondary_support_band_rollout_tail_loss
        )
        total_loss = total_loss + (
            float(secondary_support_band_loss_weight) * secondary_support_total
        )
    if diagnostic_mean_vpar_loss_weight > 0.0:
        # The auxiliary term mirrors the main rollout curriculum: first-step drift must stay good,
        # and later-step parallel drift is penalized more softly through the rollout weight.
        diagnostic_total = diagnostic_mean_vpar_one_step_loss + (
            rollout_weight * diagnostic_mean_vpar_rollout_tail_loss
        )
        total_loss = total_loss + (float(diagnostic_mean_vpar_loss_weight) * diagnostic_total)
    if localized_support_band_mean_vpar_loss_weight > 0.0:
        # This corrective term is intentionally local: it nudges mean_vpar only inside the known
        # failure band, which is safer than pushing the same primitive loss over the whole line.
        localized_support_band_mean_vpar_total = localized_support_band_mean_vpar_one_step_loss + (
            rollout_weight * localized_support_band_mean_vpar_rollout_tail_loss
        )
        total_loss = total_loss + (
            float(localized_support_band_mean_vpar_loss_weight)
            * localized_support_band_mean_vpar_total
        )
    if diagnostic_std_vpar_loss_weight > 0.0:
        # The occupied-shell parallel-spread term follows the same curriculum so it sharpens
        # velocity-width rollout behavior without overwhelming the base transport loss.
        diagnostic_std_total = diagnostic_std_vpar_one_step_loss + (
            rollout_weight * diagnostic_std_vpar_rollout_tail_loss
        )
        total_loss = total_loss + (float(diagnostic_std_vpar_loss_weight) * diagnostic_std_total)
    rollout_raw_mse = torch.mean(
        torch.square(
            pred_dynamic_seq_physical.index_select(dim=-1, index=state_index)
            - target_dynamic_seq_physical.index_select(dim=-1, index=state_index)
        )
    )
    one_step_raw_mse = torch.mean(
        torch.square(
            pred_dynamic_seq_physical[:, 0].index_select(dim=-1, index=state_index)
            - target_dynamic_seq_physical[:, 0].index_select(dim=-1, index=state_index)
        )
    )

    return {
        "pred_dynamic_seq": pred_dynamic_seq,
        "pred_dynamic_seq_physical": pred_dynamic_seq_physical,
        "total_loss": total_loss,
        "rollout_loss": rollout_loss,
        "rollout_tail_loss": rollout_tail_loss,
        "one_step_loss": one_step_loss,
        "support_band_rollout_loss": support_band_rollout_loss,
        "support_band_rollout_tail_loss": support_band_rollout_tail_loss,
        "support_band_one_step_loss": support_band_one_step_loss,
        "secondary_support_band_rollout_loss": secondary_support_band_rollout_loss,
        "secondary_support_band_rollout_tail_loss": secondary_support_band_rollout_tail_loss,
        "secondary_support_band_one_step_loss": secondary_support_band_one_step_loss,
        "diagnostic_mean_vpar_rollout_loss": diagnostic_mean_vpar_rollout_loss,
        "diagnostic_mean_vpar_rollout_tail_loss": diagnostic_mean_vpar_rollout_tail_loss,
        "diagnostic_mean_vpar_one_step_loss": diagnostic_mean_vpar_one_step_loss,
        "diagnostic_mean_vpar_rollout_valid_fraction": diagnostic_mean_vpar_rollout_valid_fraction,
        "diagnostic_mean_vpar_one_step_valid_fraction": diagnostic_mean_vpar_one_step_valid_fraction,
        "localized_support_band_mean_vpar_rollout_loss": localized_support_band_mean_vpar_rollout_loss,
        "localized_support_band_mean_vpar_rollout_tail_loss": localized_support_band_mean_vpar_rollout_tail_loss,
        "localized_support_band_mean_vpar_one_step_loss": localized_support_band_mean_vpar_one_step_loss,
        "localized_support_band_mean_vpar_rollout_valid_fraction": localized_support_band_mean_vpar_rollout_valid_fraction,
        "localized_support_band_mean_vpar_one_step_valid_fraction": localized_support_band_mean_vpar_one_step_valid_fraction,
        "diagnostic_std_vpar_rollout_loss": diagnostic_std_vpar_rollout_loss,
        "diagnostic_std_vpar_rollout_tail_loss": diagnostic_std_vpar_rollout_tail_loss,
        "diagnostic_std_vpar_one_step_loss": diagnostic_std_vpar_one_step_loss,
        "diagnostic_std_vpar_rollout_valid_fraction": diagnostic_std_vpar_rollout_valid_fraction,
        "diagnostic_std_vpar_one_step_valid_fraction": diagnostic_std_vpar_one_step_valid_fraction,
        "rollout_raw_mse": rollout_raw_mse,
        "one_step_raw_mse": one_step_raw_mse,
    }


def run_rollout_epoch(
    model: FieldlineGraphForecaster,
    loader: DataLoader,
    static_node_features: torch.Tensor,
    edge_index: torch.Tensor,
    optimizer: torch.optim.Optimizer | None,
    stats_torch: dict[str, torch.Tensor],
    predict_delta_state: bool,
    rollout_weight: float,
    support_band_loss_weight: float,
    secondary_support_band_loss_weight: float,
    localized_support_band_mean_vpar_loss_weight: float,
    occupied_shell_threshold: float,
    diagnostic_mean_vpar_loss_weight: float,
    diagnostic_std_vpar_loss_weight: float,
    diagnostic_mean_vpar_scale: float,
    diagnostic_std_vpar_scale: float,
    diagnostic_band_mask_mode: str,
    teacher_forcing_prob: float,
    grad_clip_norm: float,
    device: torch.device,
    epoch_label: str,
    compute_physical_diagnostics: bool,
    focus_species_support_band_node_mask: np.ndarray | None = None,
) -> dict[str, float | dict[str, float]]:
    """Run one train or validation epoch for autoregressive rollout supervision."""
    is_train = optimizer is not None
    model.train(is_train)

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

    iterator = tqdm(loader, desc=epoch_label, dynamic_ncols=True, leave=False)
    for batch in iterator:
        history_dynamic = batch["history_dynamic"].to(device=device, dtype=torch.float32)
        history_global_context = batch["history_global_context"].to(device=device, dtype=torch.float32)
        history_boundary_context = batch["history_boundary_context"].to(device=device, dtype=torch.float32)
        target_dynamic_seq = batch["target_dynamic_seq"].to(device=device, dtype=torch.float32)
        target_dynamic_seq_physical = batch["target_dynamic_seq_physical"].to(device=device, dtype=torch.float32)
        target_global_context_seq = batch["target_global_context_seq"].to(device=device, dtype=torch.float32)
        target_boundary_context_seq = batch["target_boundary_context_seq"].to(device=device, dtype=torch.float32)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

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
            predict_delta_state=predict_delta_state,
            rollout_weight=rollout_weight,
            support_band_loss_weight=support_band_loss_weight,
            secondary_support_band_loss_weight=secondary_support_band_loss_weight,
            localized_support_band_mean_vpar_loss_weight=localized_support_band_mean_vpar_loss_weight,
            occupied_shell_threshold=occupied_shell_threshold,
            diagnostic_mean_vpar_loss_weight=diagnostic_mean_vpar_loss_weight,
            diagnostic_std_vpar_loss_weight=diagnostic_std_vpar_loss_weight,
            diagnostic_mean_vpar_scale=diagnostic_mean_vpar_scale,
            diagnostic_std_vpar_scale=diagnostic_std_vpar_scale,
            teacher_forcing_prob=teacher_forcing_prob,
            training=is_train,
            diagnostic_band_mask_mode=diagnostic_band_mask_mode,
        )

        if is_train:
            outputs["total_loss"].backward()
            # Gradient clipping is a purely numerical safeguard against exploding autoregressive
            # sensitivities during early rollout fine-tuning.
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

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
        if compute_physical_diagnostics:
            # Validation selection uses recovered physical diagnostics, so the epoch loop keeps the
            # full rollout predictions needed to recompute those diagnostics exactly.
            pred_batches.append(pred_dynamic_seq.numpy())
            true_batches.append(target_dynamic_seq_cpu.numpy())

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
        raise ValueError("Encountered an empty rollout DataLoader.")

    rollout_channel_mse = {
        name: float(value / max(rollout_channel_count, 1))
        for name, value in zip(DYNAMIC_NODE_FEATURE_NAMES, rollout_channel_sq_error.tolist())
    }
    one_step_channel_mse = {
        name: float(value / max(one_step_channel_count, 1))
        for name, value in zip(DYNAMIC_NODE_FEATURE_NAMES, one_step_channel_sq_error.tolist())
    }
    diagnostic_rollout_channel_mse: dict[str, float] = {}
    diagnostic_one_step_channel_mse: dict[str, float] = {}
    support_band_diagnostic_rollout_channel_mse: dict[str, float] = {}
    support_band_diagnostic_rollout_valid_fraction: dict[str, float] = {}
    support_band_diagnostic_one_step_channel_mse: dict[str, float] = {}
    support_band_diagnostic_one_step_valid_fraction: dict[str, float] = {}
    focus_species_support_band_diagnostic_rollout_channel_mse: dict[str, float] = {}
    focus_species_support_band_diagnostic_rollout_valid_fraction: dict[str, float] = {}
    focus_species_support_band_diagnostic_one_step_channel_mse: dict[str, float] = {}
    focus_species_support_band_diagnostic_one_step_valid_fraction: dict[str, float] = {}
    if compute_physical_diagnostics:
        pred_dynamic_seq_all = np.concatenate(pred_batches, axis=0)
        true_dynamic_seq_all = np.concatenate(true_batches, axis=0)
        for feature_name in DYNAMIC_DIAGNOSTIC_FEATURE_NAMES:
            # Recovered physical diagnostics keep checkpoint selection tied to the same primitive
            # quantities used in post-training analysis, instead of the normalized auxiliary loss.
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
        (
            support_band_diagnostic_rollout_channel_mse,
            support_band_diagnostic_rollout_valid_fraction,
            support_band_diagnostic_one_step_channel_mse,
            support_band_diagnostic_one_step_valid_fraction,
        ) = compute_band_diagnostic_metrics(
            pred_dynamic_physical=pred_dynamic_seq_all,
            true_dynamic_physical=true_dynamic_seq_all,
            occupied_shell_threshold=occupied_shell_threshold,
            target_schema=str(model.cfg.target_schema),
            node_mask=stats_torch["support_band_node_mask"].detach().cpu().numpy() > 0.0,
        )
        if focus_species_support_band_node_mask is not None:
            # This mask isolates the exact persistent failure mode: one species in the occupied
            # support-collapse band, rather than averaging that behavior away over all species.
            (
                focus_species_support_band_diagnostic_rollout_channel_mse,
                focus_species_support_band_diagnostic_rollout_valid_fraction,
                focus_species_support_band_diagnostic_one_step_channel_mse,
                focus_species_support_band_diagnostic_one_step_valid_fraction,
            ) = compute_band_diagnostic_metrics(
                pred_dynamic_physical=pred_dynamic_seq_all,
                true_dynamic_physical=true_dynamic_seq_all,
                occupied_shell_threshold=occupied_shell_threshold,
                target_schema=str(model.cfg.target_schema),
                node_mask=np.asarray(focus_species_support_band_node_mask, dtype=bool),
            )

    return {
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
        "rollout_raw_mse": total_rollout_raw_mse / total_examples,
        "one_step_raw_mse": total_one_step_raw_mse / total_examples,
        "rollout_channel_mse": rollout_channel_mse,
        "one_step_channel_mse": one_step_channel_mse,
        "diagnostic_rollout_channel_mse": diagnostic_rollout_channel_mse,
        "diagnostic_one_step_channel_mse": diagnostic_one_step_channel_mse,
        "support_band_diagnostic_rollout_channel_mse": support_band_diagnostic_rollout_channel_mse,
        "support_band_diagnostic_rollout_valid_fraction": support_band_diagnostic_rollout_valid_fraction,
        "support_band_diagnostic_one_step_channel_mse": support_band_diagnostic_one_step_channel_mse,
        "support_band_diagnostic_one_step_valid_fraction": support_band_diagnostic_one_step_valid_fraction,
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


def compute_selection_score(
    record: dict[str, float | dict[str, float]],
    selection_metric: str,
    selection_rollout_loss_weight: float,
    selection_mean_vpar_weight: float,
    selection_std_vpar_weight: float,
) -> float:
    """Compute one scalar rollout checkpoint-selection score from validation metrics.

    Physical interpretation:
    - `valid_total_loss` preserves the original behavior and matches the exact optimized objective.
    - `composite_velocity` still prioritizes rollout stability, but it keeps the occupied-shell
      parallel-drift and spread diagnostics in the loop when choosing the checkpoint for analysis
      and downstream use.
    - The physical diagnostic modes use recovered primitive-moment MSE directly, which matches the
      quantities inspected in the shell-profile diagnostics after training.
    - The band-local physical modes use the same recovered primitive moments, but only inside the
      altitude regime where rollout diagnostics showed the shell-support failure is concentrated.
    """
    normalized_metric = str(selection_metric).strip().lower()
    if normalized_metric == "valid_total_loss":
        return float(record["valid_total_loss"])
    if normalized_metric == "composite_velocity":
        return (
            float(selection_rollout_loss_weight) * float(record["valid_rollout_loss"])
            + float(selection_mean_vpar_weight) * float(record["valid_diagnostic_mean_vpar_rollout_loss"])
            + float(selection_std_vpar_weight) * float(record["valid_diagnostic_std_vpar_rollout_loss"])
        )
    if normalized_metric == "physical_mean_vpar_rollout_mse":
        return float(record["valid_diagnostic_rollout_channel_mse"]["mean_vpar"])
    if normalized_metric == "physical_std_vpar_rollout_mse":
        return float(record["valid_diagnostic_rollout_channel_mse"]["std_vpar"])
    if normalized_metric == "physical_band_mean_vpar_rollout_mse":
        return float(record["valid_support_band_diagnostic_rollout_channel_mse"]["mean_vpar"])
    if normalized_metric == "physical_band_std_vpar_rollout_mse":
        return float(record["valid_support_band_diagnostic_rollout_channel_mse"]["std_vpar"])
    if normalized_metric == "physical_focus_band_mean_vpar_rollout_mse":
        return float(record["valid_focus_species_support_band_diagnostic_rollout_channel_mse"]["mean_vpar"])
    raise ValueError(f"Unsupported selection_metric='{selection_metric}'.")


def main() -> None:
    """Load a one-step checkpoint, fine-tune it on rollout supervision, and save best artifacts."""
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)
    device = select_device(args.device)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    checkpoint_dir = os.path.dirname(os.path.abspath(args.checkpoint))
    normalization = FieldlineWindowNormalization.load_json(os.path.join(checkpoint_dir, "normalization.json"))
    data_cfg = checkpoint["data_cfg"]
    model_cfg = checkpoint["model_cfg"]
    data_dir = args.data_dir if args.data_dir else data_cfg["data_dir"]
    species_ids = tuple(int(species_id) for species_id in checkpoint["species_ids"])
    shell_edges_km = checkpoint["shell_edges_km"]
    parent_train_cfg = checkpoint.get("train_cfg", {})
    velocity_species_balance_power = (
        float(parent_train_cfg.get("velocity_species_balance_power", 0.0))
        if args.velocity_species_balance_power is None
        else float(args.velocity_species_balance_power)
    )
    velocity_species_balance_max_weight = (
        float(parent_train_cfg.get("velocity_species_balance_max_weight", 4.0))
        if args.velocity_species_balance_max_weight is None
        else float(args.velocity_species_balance_max_weight)
    )
    diagnostic_mean_vpar_loss_weight = (
        float(parent_train_cfg.get("diagnostic_mean_vpar_loss_weight", 0.0))
        if args.diagnostic_mean_vpar_loss_weight is None
        else float(args.diagnostic_mean_vpar_loss_weight)
    )
    diagnostic_std_vpar_loss_weight = (
        float(parent_train_cfg.get("diagnostic_std_vpar_loss_weight", 0.0))
        if args.diagnostic_std_vpar_loss_weight is None
        else float(args.diagnostic_std_vpar_loss_weight)
    )
    diagnostic_band_mask_mode = (
        str(parent_train_cfg.get("diagnostic_band_mask_mode", "global"))
        if args.diagnostic_band_mask_mode is None
        else str(args.diagnostic_band_mask_mode)
    )
    diagnostic_loss_warmup_epochs = (
        int(parent_train_cfg.get("diagnostic_loss_warmup_epochs", 0))
        if args.diagnostic_loss_warmup_epochs is None
        else int(args.diagnostic_loss_warmup_epochs)
    )
    diagnostic_loss_warmup_start_factor = (
        float(parent_train_cfg.get("diagnostic_loss_warmup_start_factor", 0.0))
        if args.diagnostic_loss_warmup_start_factor is None
        else float(args.diagnostic_loss_warmup_start_factor)
    )
    support_band_loss_weight = (
        float(parent_train_cfg.get("support_band_loss_weight", 0.0))
        if args.support_band_loss_weight is None
        else float(args.support_band_loss_weight)
    )
    support_band_lower_km = (
        float(parent_train_cfg.get("support_band_lower_km", 2900.0))
        if args.support_band_lower_km is None
        else float(args.support_band_lower_km)
    )
    support_band_upper_km = (
        float(parent_train_cfg.get("support_band_upper_km", 4700.0))
        if args.support_band_upper_km is None
        else float(args.support_band_upper_km)
    )
    secondary_support_band_loss_weight = (
        float(parent_train_cfg.get("secondary_support_band_loss_weight", 0.0))
        if args.secondary_support_band_loss_weight is None
        else float(args.secondary_support_band_loss_weight)
    )
    localized_support_band_mean_vpar_loss_weight = float(
        args.localized_support_band_mean_vpar_loss_weight
    )
    secondary_support_band_lower_km = (
        float(parent_train_cfg.get("secondary_support_band_lower_km", 6900.0))
        if args.secondary_support_band_lower_km is None
        else float(args.secondary_support_band_lower_km)
    )
    secondary_support_band_upper_km = (
        float(parent_train_cfg.get("secondary_support_band_upper_km", 7800.0))
        if args.secondary_support_band_upper_km is None
        else float(args.secondary_support_band_upper_km)
    )
    print("[rollout-train] stage 1/4: loading rollout graph series...", flush=True)
    train_series = load_fieldline_graph_series(
        data_dir=data_dir,
        split="train",
        num_particles=data_cfg["num_particles"],
        num_altitude_shells=data_cfg["num_altitude_shells"],
        include_global_node=data_cfg["include_global_node"],
        include_species_coupling_edges=data_cfg["include_species_coupling_edges"],
        species_ids=species_ids,
        shell_edges_km_override=shell_edges_km,
        verbose=args.verbose_data,
    )
    valid_series = load_fieldline_graph_series(
        data_dir=data_dir,
        split="valid",
        num_particles=data_cfg["num_particles"],
        num_altitude_shells=data_cfg["num_altitude_shells"],
        include_global_node=data_cfg["include_global_node"],
        include_species_coupling_edges=data_cfg["include_species_coupling_edges"],
        species_ids=species_ids,
        shell_edges_km_override=shell_edges_km,
        verbose=args.verbose_data,
    )
    velocity_species_metadata = build_velocity_species_weight_metadata(
        layout=train_series.layout,
        dynamic_node_features=train_series.dynamic_node_features,
        balance_power=velocity_species_balance_power,
        max_weight=velocity_species_balance_max_weight,
    )
    support_band_node_mask = build_support_band_node_mask(
        layout=train_series.layout,
        shell_edges_km=train_series.shell_edges_km,
        lower_km=support_band_lower_km,
        upper_km=support_band_upper_km,
    )
    secondary_support_band_node_mask = build_support_band_node_mask(
        layout=train_series.layout,
        shell_edges_km=train_series.shell_edges_km,
        lower_km=secondary_support_band_lower_km,
        upper_km=secondary_support_band_upper_km,
    )
    focus_species_support_band_node_mask = None
    if args.selection_focus_species_id is not None:
        # Species-focus selection should only look inside the same occupied support-collapse band
        # so the checkpoint score targets the exact local failure mode we want to reduce.
        focus_species_node_mask = build_species_node_mask(
            layout=train_series.layout,
            species_id=int(args.selection_focus_species_id),
        )
        focus_species_support_band_node_mask = np.logical_and(
            np.asarray(support_band_node_mask, dtype=bool),
            np.asarray(focus_species_node_mask, dtype=bool),
        )
        print(
            "[rollout-train] focus species support band "
            f"species={int(args.selection_focus_species_id)} "
            f"active_nodes={int(np.sum(focus_species_support_band_node_mask))}/{train_series.layout.num_nodes}",
            flush=True,
        )
    print(
        "[rollout-train] velocity species weights "
        f"power={velocity_species_balance_power:.3f} "
        f"max={velocity_species_balance_max_weight:.3f} "
        f"weights={velocity_species_metadata['velocity_species_weight_by_species'].tolist()}",
        flush=True,
    )
    print(
        "[rollout-train] support band "
        f"{support_band_lower_km:.1f}-{support_band_upper_km:.1f} km "
        f"active_nodes={int(np.sum(support_band_node_mask))}/{train_series.layout.num_nodes}",
        flush=True,
    )
    print(
        "[rollout-train] secondary support band "
        f"{secondary_support_band_lower_km:.1f}-{secondary_support_band_upper_km:.1f} km "
        f"active_nodes={int(np.sum(secondary_support_band_node_mask))}/{train_series.layout.num_nodes}",
        flush=True,
    )

    print("[rollout-train] stage 2/4: building rollout windows...", flush=True)
    train_arrays = build_fieldline_rollout_arrays(
        series=train_series,
        window_length=data_cfg["window_length"],
        horizon=args.horizon,
    )
    valid_arrays = build_fieldline_rollout_arrays(
        series=valid_series,
        window_length=data_cfg["window_length"],
        horizon=args.horizon,
    )
    # Primitive diagnostic scales are recomputed from the rollout train split when the parent
    # one-step checkpoint predates them, which keeps rollout fine-tuning backward compatible.
    diagnostic_mean_vpar_scale = float(
        checkpoint.get(
            "diagnostic_mean_vpar_scale",
            compute_diagnostic_scale(
                target_dynamic_physical=train_arrays.target_dynamic_seq_physical,
                feature_name="mean_vpar",
                occupied_shell_threshold=float(parent_train_cfg.get("occupied_shell_threshold", 1.0e-3)),
                target_schema=str(data_cfg.get("target_schema", "population_weighted")),
            ),
        )
    )
    diagnostic_std_vpar_scale = float(
        checkpoint.get(
            "diagnostic_std_vpar_scale",
            compute_diagnostic_scale(
                target_dynamic_physical=train_arrays.target_dynamic_seq_physical,
                feature_name="std_vpar",
                occupied_shell_threshold=float(parent_train_cfg.get("occupied_shell_threshold", 1.0e-3)),
                target_schema=str(data_cfg.get("target_schema", "population_weighted")),
            ),
        )
    )
    train_dataset = FieldlineRolloutDataset(
        static_node_features=train_series.static_node_features,
        arrays=train_arrays,
        normalization=normalization,
        predict_delta_state=model_cfg["predict_delta_state"],
    )
    valid_dataset = FieldlineRolloutDataset(
        static_node_features=valid_series.static_node_features,
        arrays=valid_arrays,
        normalization=normalization,
        predict_delta_state=model_cfg["predict_delta_state"],
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    train_cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        horizon=args.horizon,
        rollout_weight=args.rollout_weight,
        teacher_forcing_prob=args.teacher_forcing_prob,
        grad_clip_norm=args.grad_clip_norm,
        num_workers=args.num_workers,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        velocity_species_balance_power=velocity_species_balance_power,
        velocity_species_balance_max_weight=velocity_species_balance_max_weight,
        occupied_shell_threshold=float(parent_train_cfg.get("occupied_shell_threshold", 1.0e-3)),
        support_band_loss_weight=support_band_loss_weight,
        support_band_lower_km=support_band_lower_km,
        support_band_upper_km=support_band_upper_km,
        secondary_support_band_loss_weight=secondary_support_band_loss_weight,
        secondary_support_band_lower_km=secondary_support_band_lower_km,
        secondary_support_band_upper_km=secondary_support_band_upper_km,
        localized_support_band_mean_vpar_loss_weight=localized_support_band_mean_vpar_loss_weight,
        diagnostic_mean_vpar_loss_weight=diagnostic_mean_vpar_loss_weight,
        diagnostic_std_vpar_loss_weight=diagnostic_std_vpar_loss_weight,
        diagnostic_band_mask_mode=diagnostic_band_mask_mode,
        diagnostic_loss_warmup_epochs=diagnostic_loss_warmup_epochs,
        diagnostic_loss_warmup_start_factor=diagnostic_loss_warmup_start_factor,
        selection_metric=args.selection_metric,
        selection_focus_species_id=args.selection_focus_species_id,
        selection_focus_rollout_loss_cap=args.selection_focus_rollout_loss_cap,
        selection_rollout_loss_weight=args.selection_rollout_loss_weight,
        selection_mean_vpar_weight=args.selection_mean_vpar_weight,
        selection_std_vpar_weight=args.selection_std_vpar_weight,
        save_epoch_checkpoints=bool(args.save_epoch_checkpoints),
        seed=args.seed,
    )
    with open(os.path.join(args.output_dir, "config.json"), "w", encoding="utf-8") as handle:
        json.dump(
            {
                "parent_checkpoint": os.path.abspath(args.checkpoint),
                "data_cfg": data_cfg,
                "model_cfg": model_cfg,
                "train_cfg": asdict(train_cfg),
                "species_ids": list(species_ids),
                "shell_edges_km": torch.as_tensor(shell_edges_km).tolist(),
                "velocity_species_mean_population_fraction": (
                    velocity_species_metadata["species_mean_population_fraction"].tolist()
                ),
                "velocity_species_weight_by_species": (
                    velocity_species_metadata["velocity_species_weight_by_species"].tolist()
                ),
                "velocity_species_weight_per_node": (
                    velocity_species_metadata["velocity_species_weight_per_node"].tolist()
                ),
                "support_band_node_mask": support_band_node_mask.tolist(),
                "secondary_support_band_node_mask": secondary_support_band_node_mask.tolist(),
                "focus_species_support_band_node_mask": (
                    None
                    if focus_species_support_band_node_mask is None
                    else focus_species_support_band_node_mask.astype(np.float32).tolist()
                ),
                "diagnostic_mean_vpar_scale": diagnostic_mean_vpar_scale,
                "diagnostic_std_vpar_scale": diagnostic_std_vpar_scale,
            },
            handle,
            indent=2,
        )
    normalization.save_json(os.path.join(args.output_dir, "normalization.json"))

    print("[rollout-train] stage 3/4: restoring one-step checkpoint...", flush=True)
    model = FieldlineGraphForecaster(cfg=ModelConfig(**model_cfg)).to(device)
    model.load_state_dict(checkpoint["model_state"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)

    static_node_features = torch.as_tensor(
        train_series.static_node_features,
        dtype=torch.float32,
        device=device,
    )
    edge_index = torch.as_tensor(train_series.edge_index, dtype=torch.long, device=device)
    stats_torch = stats_to_torch(
        stats=normalization,
        device=device,
        num_nodes=train_series.layout.num_nodes,
        target_schema=model_cfg["target_schema"],
        velocity_species_weight_per_node=velocity_species_metadata["velocity_species_weight_per_node"],
        boundary_source_node_mask=checkpoint.get(
            "boundary_source_node_mask",
            build_boundary_source_node_mask(train_series.layout),
        ),
        support_band_node_mask=support_band_node_mask,
        secondary_support_band_node_mask=secondary_support_band_node_mask,
        transport_interfaces=(
            build_transport_interfaces(
                layout=train_series.layout,
                shell_edges_km=train_series.shell_edges_km,
            )
            if bool(model_cfg.get("flux_transport_head", False))
            else None
        ),
        transport_cell_width_km=np.asarray(
            checkpoint.get(
                "transport_cell_width_km",
                build_transport_cell_width_per_node(
                    layout=train_series.layout,
                    shell_edges_km=train_series.shell_edges_km,
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
                        np.ones(train_series.layout.num_shell_nodes, dtype=np.float32),
                        np.zeros(train_series.layout.num_nodes - train_series.layout.num_shell_nodes, dtype=np.float32),
                    )
                ),
            ),
            dtype=np.float32,
        ),
        transport_interface_width_km=np.asarray(
            checkpoint.get(
                "transport_interface_width_km",
                build_transport_interfaces(
                    layout=train_series.layout,
                    shell_edges_km=train_series.shell_edges_km,
                ).interface_width_km,
            ),
            dtype=np.float32,
        ),
    )

    history: list[dict[str, float | dict[str, float]]] = []
    best_valid_total_loss = float("inf")
    best_selection_score = float("inf")
    best_mean_vpar_rollout_mse = float("inf")
    best_std_vpar_rollout_mse = float("inf")
    best_band_mean_vpar_rollout_mse = float("inf")
    best_band_std_vpar_rollout_mse = float("inf")
    best_focus_species_band_mean_vpar_rollout_mse = float("inf")
    best_focus_species_band_mean_vpar_constrained_rollout_mse = float("inf")
    best_epoch = 0
    best_total_epoch = 0
    best_mean_vpar_epoch = 0
    best_std_vpar_epoch = 0
    best_band_mean_vpar_epoch = 0
    best_band_std_vpar_epoch = 0
    best_focus_species_band_mean_vpar_epoch = 0
    best_focus_species_band_mean_vpar_constrained_epoch = 0
    epochs_without_improvement = 0

    print("[rollout-train] stage 4/4: rollout fine-tuning...", flush=True)
    for epoch in range(1, train_cfg.epochs + 1):
        # Primitive occupied-shell supervision is warmed in gradually so rollout first stabilizes
        # the coarse shell dynamics before small velocity-diagnostic corrections are applied.
        effective_mean_vpar_loss_weight = compute_effective_diagnostic_weight(
            base_weight=train_cfg.diagnostic_mean_vpar_loss_weight,
            epoch_index=epoch,
            warmup_epochs=train_cfg.diagnostic_loss_warmup_epochs,
            warmup_start_factor=train_cfg.diagnostic_loss_warmup_start_factor,
        )
        effective_std_vpar_loss_weight = compute_effective_diagnostic_weight(
            base_weight=train_cfg.diagnostic_std_vpar_loss_weight,
            epoch_index=epoch,
            warmup_epochs=train_cfg.diagnostic_loss_warmup_epochs,
            warmup_start_factor=train_cfg.diagnostic_loss_warmup_start_factor,
        )
        train_metrics = run_rollout_epoch(
            model=model,
            loader=train_loader,
            static_node_features=static_node_features,
            edge_index=edge_index,
            optimizer=optimizer,
            stats_torch=stats_torch,
            predict_delta_state=model_cfg["predict_delta_state"],
            rollout_weight=train_cfg.rollout_weight,
            support_band_loss_weight=train_cfg.support_band_loss_weight,
            secondary_support_band_loss_weight=train_cfg.secondary_support_band_loss_weight,
            localized_support_band_mean_vpar_loss_weight=train_cfg.localized_support_band_mean_vpar_loss_weight,
            occupied_shell_threshold=train_cfg.occupied_shell_threshold,
            diagnostic_mean_vpar_loss_weight=effective_mean_vpar_loss_weight,
            diagnostic_std_vpar_loss_weight=effective_std_vpar_loss_weight,
            diagnostic_mean_vpar_scale=diagnostic_mean_vpar_scale,
            diagnostic_std_vpar_scale=diagnostic_std_vpar_scale,
            teacher_forcing_prob=train_cfg.teacher_forcing_prob,
            grad_clip_norm=train_cfg.grad_clip_norm,
            device=device,
            epoch_label=f"roll-train {epoch}/{train_cfg.epochs}",
            compute_physical_diagnostics=False,
            diagnostic_band_mask_mode=train_cfg.diagnostic_band_mask_mode,
            focus_species_support_band_node_mask=focus_species_support_band_node_mask,
        )
        with torch.no_grad():
            valid_metrics = run_rollout_epoch(
                model=model,
                loader=valid_loader,
                static_node_features=static_node_features,
                edge_index=edge_index,
                optimizer=None,
                stats_torch=stats_torch,
                predict_delta_state=model_cfg["predict_delta_state"],
                rollout_weight=train_cfg.rollout_weight,
                support_band_loss_weight=train_cfg.support_band_loss_weight,
                secondary_support_band_loss_weight=train_cfg.secondary_support_band_loss_weight,
                localized_support_band_mean_vpar_loss_weight=train_cfg.localized_support_band_mean_vpar_loss_weight,
                occupied_shell_threshold=train_cfg.occupied_shell_threshold,
                diagnostic_mean_vpar_loss_weight=effective_mean_vpar_loss_weight,
                diagnostic_std_vpar_loss_weight=effective_std_vpar_loss_weight,
                diagnostic_mean_vpar_scale=diagnostic_mean_vpar_scale,
                diagnostic_std_vpar_scale=diagnostic_std_vpar_scale,
                teacher_forcing_prob=0.0,
                grad_clip_norm=train_cfg.grad_clip_norm,
                device=device,
                epoch_label=f"roll-valid {epoch}/{train_cfg.epochs}",
                compute_physical_diagnostics=True,
                diagnostic_band_mask_mode=train_cfg.diagnostic_band_mask_mode,
                focus_species_support_band_node_mask=focus_species_support_band_node_mask,
            )

        record = {
            "epoch": epoch,
            "train_total_loss": float(train_metrics["total_loss"]),
            "train_rollout_loss": float(train_metrics["rollout_loss"]),
            "train_rollout_tail_loss": float(train_metrics["rollout_tail_loss"]),
            "train_one_step_loss": float(train_metrics["one_step_loss"]),
            "train_support_band_rollout_loss": float(train_metrics["support_band_rollout_loss"]),
            "train_support_band_one_step_loss": float(train_metrics["support_band_one_step_loss"]),
            "train_secondary_support_band_rollout_loss": float(train_metrics["secondary_support_band_rollout_loss"]),
            "train_secondary_support_band_one_step_loss": float(train_metrics["secondary_support_band_one_step_loss"]),
            "train_diagnostic_mean_vpar_rollout_loss": float(train_metrics["diagnostic_mean_vpar_rollout_loss"]),
            "train_diagnostic_mean_vpar_one_step_loss": float(train_metrics["diagnostic_mean_vpar_one_step_loss"]),
            "train_localized_support_band_mean_vpar_rollout_loss": float(train_metrics["localized_support_band_mean_vpar_rollout_loss"]),
            "train_localized_support_band_mean_vpar_one_step_loss": float(train_metrics["localized_support_band_mean_vpar_one_step_loss"]),
            "train_diagnostic_std_vpar_rollout_loss": float(train_metrics["diagnostic_std_vpar_rollout_loss"]),
            "train_diagnostic_std_vpar_one_step_loss": float(train_metrics["diagnostic_std_vpar_one_step_loss"]),
            "train_rollout_raw_mse": float(train_metrics["rollout_raw_mse"]),
            "valid_total_loss": float(valid_metrics["total_loss"]),
            "valid_rollout_loss": float(valid_metrics["rollout_loss"]),
            "valid_rollout_tail_loss": float(valid_metrics["rollout_tail_loss"]),
            "valid_one_step_loss": float(valid_metrics["one_step_loss"]),
            "valid_support_band_rollout_loss": float(valid_metrics["support_band_rollout_loss"]),
            "valid_support_band_one_step_loss": float(valid_metrics["support_band_one_step_loss"]),
            "valid_secondary_support_band_rollout_loss": float(valid_metrics["secondary_support_band_rollout_loss"]),
            "valid_secondary_support_band_one_step_loss": float(valid_metrics["secondary_support_band_one_step_loss"]),
            "valid_diagnostic_mean_vpar_rollout_loss": float(valid_metrics["diagnostic_mean_vpar_rollout_loss"]),
            "valid_diagnostic_mean_vpar_one_step_loss": float(valid_metrics["diagnostic_mean_vpar_one_step_loss"]),
            "valid_localized_support_band_mean_vpar_rollout_loss": float(valid_metrics["localized_support_band_mean_vpar_rollout_loss"]),
            "valid_localized_support_band_mean_vpar_one_step_loss": float(valid_metrics["localized_support_band_mean_vpar_one_step_loss"]),
            "valid_diagnostic_std_vpar_rollout_loss": float(valid_metrics["diagnostic_std_vpar_rollout_loss"]),
            "valid_diagnostic_std_vpar_one_step_loss": float(valid_metrics["diagnostic_std_vpar_one_step_loss"]),
            "valid_rollout_raw_mse": float(valid_metrics["rollout_raw_mse"]),
            "valid_one_step_raw_mse": float(valid_metrics["one_step_raw_mse"]),
            "valid_rollout_channel_mse": valid_metrics["rollout_channel_mse"],
            "valid_one_step_channel_mse": valid_metrics["one_step_channel_mse"],
            "valid_diagnostic_rollout_channel_mse": valid_metrics["diagnostic_rollout_channel_mse"],
            "valid_diagnostic_one_step_channel_mse": valid_metrics["diagnostic_one_step_channel_mse"],
            # These localized diagnostics isolate the occupied-shell altitude band where the model
            # actually drifts, which is the regime we want to rank specialized checkpoints on.
            "valid_support_band_diagnostic_rollout_channel_mse": (
                valid_metrics["support_band_diagnostic_rollout_channel_mse"]
            ),
            "valid_support_band_diagnostic_rollout_valid_fraction": (
                valid_metrics["support_band_diagnostic_rollout_valid_fraction"]
            ),
            "valid_support_band_diagnostic_one_step_channel_mse": (
                valid_metrics["support_band_diagnostic_one_step_channel_mse"]
            ),
            "valid_support_band_diagnostic_one_step_valid_fraction": (
                valid_metrics["support_band_diagnostic_one_step_valid_fraction"]
            ),
            "valid_focus_species_support_band_diagnostic_rollout_channel_mse": (
                valid_metrics["focus_species_support_band_diagnostic_rollout_channel_mse"]
            ),
            "valid_focus_species_support_band_diagnostic_rollout_valid_fraction": (
                valid_metrics["focus_species_support_band_diagnostic_rollout_valid_fraction"]
            ),
            "valid_focus_species_support_band_diagnostic_one_step_channel_mse": (
                valid_metrics["focus_species_support_band_diagnostic_one_step_channel_mse"]
            ),
            "valid_focus_species_support_band_diagnostic_one_step_valid_fraction": (
                valid_metrics["focus_species_support_band_diagnostic_one_step_valid_fraction"]
            ),
            "effective_diagnostic_mean_vpar_loss_weight": float(effective_mean_vpar_loss_weight),
            "effective_diagnostic_std_vpar_loss_weight": float(effective_std_vpar_loss_weight),
        }
        record["selection_score"] = compute_selection_score(
            record=record,
            selection_metric=train_cfg.selection_metric,
            selection_rollout_loss_weight=train_cfg.selection_rollout_loss_weight,
            selection_mean_vpar_weight=train_cfg.selection_mean_vpar_weight,
            selection_std_vpar_weight=train_cfg.selection_std_vpar_weight,
        )
        history.append(record)

        print(
            f"[rollout-train] epoch={epoch} train_total={record['train_total_loss']:.6e} "
            f"valid_total={record['valid_total_loss']:.6e} "
            f"valid_roll={record['valid_rollout_loss']:.6e} "
            f"valid_one={record['valid_one_step_loss']:.6e}",
            f"valid_band={record['valid_support_band_rollout_loss']:.6e} ",
            f"valid_upband={record['valid_secondary_support_band_rollout_loss']:.6e} ",
            f"valid_vpar={record['valid_diagnostic_mean_vpar_rollout_loss']:.6e} ",
            f"valid_lvpar={record['valid_localized_support_band_mean_vpar_rollout_loss']:.6e} ",
            f"valid_svpar={record['valid_diagnostic_std_vpar_rollout_loss']:.6e} ",
            f"select={record['selection_score']:.6e} ",
            flush=True,
        )

        checkpoint_payload = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "data_cfg": data_cfg,
            "model_cfg": model_cfg,
            "train_cfg": asdict(train_cfg),
            "species_ids": list(species_ids),
            "shell_edges_km": torch.as_tensor(shell_edges_km).numpy(),
            "velocity_species_mean_population_fraction": (
                velocity_species_metadata["species_mean_population_fraction"].astype(np.float32)
            ),
            "velocity_species_weight_by_species": (
                velocity_species_metadata["velocity_species_weight_by_species"].astype(np.float32)
            ),
            "velocity_species_weight_per_node": (
                velocity_species_metadata["velocity_species_weight_per_node"].astype(np.float32)
            ),
            "support_band_node_mask": support_band_node_mask.astype(np.float32),
            "secondary_support_band_node_mask": secondary_support_band_node_mask.astype(np.float32),
            "diagnostic_mean_vpar_scale": float(diagnostic_mean_vpar_scale),
            "diagnostic_std_vpar_scale": float(diagnostic_std_vpar_scale),
            "parent_checkpoint": os.path.abspath(args.checkpoint),
            "best_epoch": best_epoch,
            "best_valid_total_loss": best_valid_total_loss,
            "best_total_epoch": best_total_epoch,
            "best_selection_score": best_selection_score,
            "best_mean_vpar_rollout_mse": best_mean_vpar_rollout_mse,
            "best_std_vpar_rollout_mse": best_std_vpar_rollout_mse,
            "best_band_mean_vpar_rollout_mse": best_band_mean_vpar_rollout_mse,
            "best_band_std_vpar_rollout_mse": best_band_std_vpar_rollout_mse,
            "best_focus_species_band_mean_vpar_rollout_mse": best_focus_species_band_mean_vpar_rollout_mse,
            "best_focus_species_band_mean_vpar_constrained_rollout_mse": (
                best_focus_species_band_mean_vpar_constrained_rollout_mse
            ),
            "best_mean_vpar_epoch": best_mean_vpar_epoch,
            "best_std_vpar_epoch": best_std_vpar_epoch,
            "best_band_mean_vpar_epoch": best_band_mean_vpar_epoch,
            "best_band_std_vpar_epoch": best_band_std_vpar_epoch,
            "best_focus_species_band_mean_vpar_epoch": best_focus_species_band_mean_vpar_epoch,
            "best_focus_species_band_mean_vpar_constrained_epoch": (
                best_focus_species_band_mean_vpar_constrained_epoch
            ),
            "selection_metric": train_cfg.selection_metric,
            "history": history,
        }
        torch.save(checkpoint_payload, os.path.join(args.output_dir, "last.pt"))
        if train_cfg.save_epoch_checkpoints:
            # Per-epoch checkpoints make it possible to recover diagnostic-best epochs even when
            # they are not the winner under the primary selection score.
            torch.save(checkpoint_payload, os.path.join(args.output_dir, f"checkpoint_epoch_{epoch}.pt"))

        if record["valid_total_loss"] < (best_valid_total_loss - train_cfg.early_stopping_min_delta):
            best_valid_total_loss = record["valid_total_loss"]
            best_total_epoch = epoch
            checkpoint_payload["best_valid_total_loss"] = best_valid_total_loss
            checkpoint_payload["best_total_epoch"] = best_total_epoch
            torch.save(checkpoint_payload, os.path.join(args.output_dir, "best_total.pt"))

        current_mean_vpar_rollout_mse = float(record["valid_diagnostic_rollout_channel_mse"]["mean_vpar"])
        if current_mean_vpar_rollout_mse < best_mean_vpar_rollout_mse:
            # The dedicated primitive checkpoint keeps the best recovered parallel-drift solution
            # even when the total rollout objective prefers a different compromise.
            best_mean_vpar_rollout_mse = current_mean_vpar_rollout_mse
            best_mean_vpar_epoch = epoch
            checkpoint_payload["best_mean_vpar_rollout_mse"] = best_mean_vpar_rollout_mse
            checkpoint_payload["best_mean_vpar_epoch"] = best_mean_vpar_epoch
            torch.save(checkpoint_payload, os.path.join(args.output_dir, "best_mean_vpar.pt"))

        current_std_vpar_rollout_mse = float(record["valid_diagnostic_rollout_channel_mse"]["std_vpar"])
        if current_std_vpar_rollout_mse < best_std_vpar_rollout_mse:
            # The dedicated spread checkpoint preserves the best parallel-width solution for later
            # analysis without forcing that choice into the main deployment artifact.
            best_std_vpar_rollout_mse = current_std_vpar_rollout_mse
            best_std_vpar_epoch = epoch
            checkpoint_payload["best_std_vpar_rollout_mse"] = best_std_vpar_rollout_mse
            checkpoint_payload["best_std_vpar_epoch"] = best_std_vpar_epoch
            torch.save(checkpoint_payload, os.path.join(args.output_dir, "best_std_vpar.pt"))

        current_band_mean_vpar_rollout_mse = float(
            record["valid_support_band_diagnostic_rollout_channel_mse"]["mean_vpar"]
        )
        if current_band_mean_vpar_rollout_mse < best_band_mean_vpar_rollout_mse:
            # This artifact targets the occupied-shell mid-altitude regime where the current model
            # loses parallel-drift support during rollout, which is different from the global best.
            best_band_mean_vpar_rollout_mse = current_band_mean_vpar_rollout_mse
            best_band_mean_vpar_epoch = epoch
            checkpoint_payload["best_band_mean_vpar_rollout_mse"] = best_band_mean_vpar_rollout_mse
            checkpoint_payload["best_band_mean_vpar_epoch"] = best_band_mean_vpar_epoch
            torch.save(checkpoint_payload, os.path.join(args.output_dir, "best_band_mean_vpar.pt"))

        current_band_std_vpar_rollout_mse = float(
            record["valid_support_band_diagnostic_rollout_channel_mse"]["std_vpar"]
        )
        if current_band_std_vpar_rollout_mse < best_band_std_vpar_rollout_mse:
            # This artifact keeps the best parallel-width solution in the same failure band so the
            # tradeoff against the global rollout objective is explicit instead of hidden.
            best_band_std_vpar_rollout_mse = current_band_std_vpar_rollout_mse
            best_band_std_vpar_epoch = epoch
            checkpoint_payload["best_band_std_vpar_rollout_mse"] = best_band_std_vpar_rollout_mse
            checkpoint_payload["best_band_std_vpar_epoch"] = best_band_std_vpar_epoch
            torch.save(checkpoint_payload, os.path.join(args.output_dir, "best_band_std_vpar.pt"))

        if focus_species_support_band_node_mask is not None:
            current_focus_species_band_mean_vpar_rollout_mse = float(
                record["valid_focus_species_support_band_diagnostic_rollout_channel_mse"]["mean_vpar"]
            )
            if current_focus_species_band_mean_vpar_rollout_mse < best_focus_species_band_mean_vpar_rollout_mse:
                # This artifact targets the exact remaining failure mode: one species, inside the
                # occupied support-collapse band, ranked on recovered physical mean_vpar.
                best_focus_species_band_mean_vpar_rollout_mse = current_focus_species_band_mean_vpar_rollout_mse
                best_focus_species_band_mean_vpar_epoch = epoch
                checkpoint_payload["best_focus_species_band_mean_vpar_rollout_mse"] = (
                    best_focus_species_band_mean_vpar_rollout_mse
                )
                checkpoint_payload["best_focus_species_band_mean_vpar_epoch"] = (
                    best_focus_species_band_mean_vpar_epoch
                )
                torch.save(
                    checkpoint_payload,
                    os.path.join(args.output_dir, "best_species_focus_band_mean_vpar.pt"),
                )
            rollout_loss_cap = train_cfg.selection_focus_rollout_loss_cap
            if (
                rollout_loss_cap is not None
                and float(record["valid_rollout_loss"]) <= float(rollout_loss_cap)
                and current_focus_species_band_mean_vpar_rollout_mse
                < best_focus_species_band_mean_vpar_constrained_rollout_mse
            ):
                # This artifact enforces a hard global rollout-quality constraint before ranking
                # on the species-specific band metric, so local gains cannot come from a broken
                # autoregressive trajectory.
                best_focus_species_band_mean_vpar_constrained_rollout_mse = (
                    current_focus_species_band_mean_vpar_rollout_mse
                )
                best_focus_species_band_mean_vpar_constrained_epoch = epoch
                checkpoint_payload["best_focus_species_band_mean_vpar_constrained_rollout_mse"] = (
                    best_focus_species_band_mean_vpar_constrained_rollout_mse
                )
                checkpoint_payload["best_focus_species_band_mean_vpar_constrained_epoch"] = (
                    best_focus_species_band_mean_vpar_constrained_epoch
                )
                checkpoint_payload["selection_focus_rollout_loss_cap"] = float(rollout_loss_cap)
                torch.save(
                    checkpoint_payload,
                    os.path.join(
                        args.output_dir,
                        "best_species_focus_band_mean_vpar_constrained.pt",
                    ),
                )

        if record["selection_score"] < (best_selection_score - train_cfg.early_stopping_min_delta):
            best_selection_score = float(record["selection_score"])
            best_epoch = epoch
            epochs_without_improvement = 0
            checkpoint_payload["best_epoch"] = best_epoch
            checkpoint_payload["best_selection_score"] = best_selection_score
            checkpoint_payload["selection_metric"] = train_cfg.selection_metric
            torch.save(checkpoint_payload, os.path.join(args.output_dir, "best.pt"))
        else:
            epochs_without_improvement += 1

        with open(os.path.join(args.output_dir, "history.json"), "w", encoding="utf-8") as handle:
            json.dump(history, handle, indent=2)

        if epochs_without_improvement >= train_cfg.early_stopping_patience:
            print(
                f"[rollout-train] early stopping at epoch={epoch} best_epoch={best_epoch} "
                f"best_selection_score={best_selection_score:.6e} "
                f"selection_metric={train_cfg.selection_metric} "
                f"best_mean_vpar_epoch={best_mean_vpar_epoch} "
                f"best_std_vpar_epoch={best_std_vpar_epoch} "
                f"best_band_mean_vpar_epoch={best_band_mean_vpar_epoch} "
                f"best_band_std_vpar_epoch={best_band_std_vpar_epoch} "
                f"best_focus_species_band_mean_vpar_epoch={best_focus_species_band_mean_vpar_epoch} "
                f"best_focus_species_band_mean_vpar_constrained_epoch={best_focus_species_band_mean_vpar_constrained_epoch}",
                flush=True,
            )
            break

    print(
        f"[rollout-train] rollout fine-tuning complete. best_epoch={best_epoch} "
        f"best_selection_score={best_selection_score:.6e} "
        f"selection_metric={train_cfg.selection_metric} "
        f"best_mean_vpar_epoch={best_mean_vpar_epoch} "
        f"best_std_vpar_epoch={best_std_vpar_epoch} "
        f"best_band_mean_vpar_epoch={best_band_mean_vpar_epoch} "
        f"best_band_std_vpar_epoch={best_band_std_vpar_epoch} "
        f"best_focus_species_band_mean_vpar_epoch={best_focus_species_band_mean_vpar_epoch} "
        f"best_focus_species_band_mean_vpar_constrained_epoch={best_focus_species_band_mean_vpar_constrained_epoch}",
        flush=True,
    )
    print(f"[rollout-train] artifacts saved in {os.path.abspath(args.output_dir)}", flush=True)


if __name__ == "__main__":
    main()
