#!/usr/bin/env python3
"""Train the one-step field-line graph surrogate for PWOM.

Physical interpretation:
- Each training sample uses a short history of shell-distribution states to predict the next
  shell-distribution state on the same field-line graph.
- This stage verifies that coarse shell graph dynamics are learnable at one-step horizon before
  long autoregressive rollout training.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Allow running this script directly from the repository root.
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from pwom_surrogate.config import DataConfig, ModelConfig, TrainConfig
from pwom_surrogate.fieldline_graph_data import (
    BOUNDARY_RESPONSE_CHANNEL_INDICES,
    TARGET_SCHEMA_NAMES,
    TARGET_SCHEMA_POPULATION_WEIGHTED,
    TARGET_SCHEMA_CLOSURE_TRANSPORT,
    SCHEMA_ASINH_FEATURE_SCALES,
    DYNAMIC_CONTEXT_CHANNEL_INDICES,
    DYNAMIC_FORCING_CHANNEL_INDICES,
    DYNAMIC_FEATURE_INDEX,
    DYNAMIC_CLOSURE_CHANNEL_INDICES,
    DYNAMIC_MOMENTUM_CHANNEL_INDICES,
    FieldlineWindowDataset,
    DYNAMIC_NONNEGATIVE_CHANNEL_INDICES,
    DYNAMIC_OCCUPANCY_SHAPE_CHANNEL_INDICES,
    DYNAMIC_STATE_CHANNEL_INDICES,
    DYNAMIC_UNIT_INTERVAL_CHANNEL_INDICES,
    NONNEGATIVE_VELOCITY_ASINH_CHANNEL_INDICES,
    OCCUPANCY_WEIGHTED_CHANNEL_INDICES,
    POPULATION_FRACTION_CHANNEL_INDEX,
    SIGNED_VELOCITY_ASINH_CHANNEL_INDICES,
    TRANSPORT_CONSERVATIVE_CHANNEL_INDICES,
    VELOCITY_ASINH_ABS_MAX,
    build_boundary_source_node_mask,
    build_transport_cell_width_per_node,
    build_transport_interfaces,
    dynamic_feature_names_for_schema,
    build_velocity_species_weight_metadata,
    build_fieldline_window_arrays,
    compute_window_normalization,
    dynamic_diagnostic_valid_mask,
    extract_dynamic_diagnostic_channel,
    load_fieldline_graph_series,
    validate_target_schema,
)
from pwom_surrogate.fieldline_graph_model import FieldlineGraphForecaster


def parse_args() -> argparse.Namespace:
    """Parse training CLI arguments."""
    parser = argparse.ArgumentParser(description="Train the PWOM surrogate one-step field-line graph model.")
    parser.add_argument("--data-dir", type=str, default="../pwom-surrogate-rebuild/dataset")
    parser.add_argument("--output-dir", type=str, default="runs/fieldline_onestep")
    parser.add_argument("--num-particles", type=int, default=512)
    parser.add_argument("--num-altitude-shells", type=int, default=16)
    parser.add_argument(
        "--shell-edges-json",
        type=str,
        default="",
        help="Optional JSON file containing explicit altitude shell edges in kilometers.",
    )
    parser.add_argument("--window-length", type=int, default=4)
    parser.add_argument("--species-ids", type=int, nargs="*", default=None)
    parser.add_argument(
        "--target-schema",
        choices=TARGET_SCHEMA_NAMES,
        default=TARGET_SCHEMA_POPULATION_WEIGHTED,
        help="Physical target parameterization used for one-step supervision.",
    )
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--message-passing-layers", type=int, default=4)
    parser.add_argument("--mlp-layers", type=int, default=2)
    parser.add_argument(
        "--split-decoder-heads",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use dedicated decoder heads for velocity moments instead of one shared output head.",
    )
    parser.add_argument(
        "--factorized-decoder-heads",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use separate decoder groups for density, geometry, topology, momentum, and forcing channels.",
    )
    parser.add_argument(
        "--conditional-momentum-head",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Predict occupancy/shape first and predict momentum channels conditioned on that intermediate state.",
    )
    parser.add_argument(
        "--uncertainty-aware-closure-head",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Predict per-channel closure log-variance for uncertainty-aware closure regression.",
    )
    parser.add_argument(
        "--flux-transport-head",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Predict same-species shell-interface fluxes and use their divergence for the conservative shell pair.",
    )
    parser.add_argument(
        "--operator-split-head",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Decompose the one-step update into explicit transport, boundary, and source branches.",
    )
    parser.add_argument(
        "--redistribution-transport-head",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Predict shellwise up/down/stay redistribution fractions plus carried velocities for the conservative pair.",
    )
    parser.add_argument(
        "--bounded-outflow-fraction-transport",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Predict bounded support transfer fractions at interfaces instead of raw support fluxes.",
    )
    parser.add_argument(
        "--simplex-support-transport",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Predict shellwise up/down/stay support fractions and carried velocities instead of raw interface fluxes.",
    )
    parser.add_argument(
        "--max-support-outflow-fraction",
        type=float,
        default=0.5,
        help="Maximum fraction of shell support that can be exported in one explicit transport step.",
    )
    parser.add_argument(
        "--flux-hidden-dim",
        type=int,
        default=128,
        help="Hidden width of the shell-interface flux decoder branch.",
    )
    parser.add_argument(
        "--global-context-film",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use FiLM-style global conditioning for line-level PWOM controls.",
    )
    parser.add_argument(
        "--boundary-injection-head",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use a dedicated lower-boundary source head acting only on the lowest shell nodes.",
    )
    parser.add_argument(
        "--boundary-hidden-dim",
        type=int,
        default=128,
        help="Hidden width of the lower-boundary source branch.",
    )
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--early-stopping-patience", type=int, default=20)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-4)
    parser.add_argument(
        "--velocity-species-balance-power",
        type=float,
        default=0.0,
        help="Inverse-population power used to upweight underrepresented species in velocity channels only.",
    )
    parser.add_argument(
        "--velocity-species-balance-max-weight",
        type=float,
        default=4.0,
        help="Maximum per-species velocity-loss weight before weights are renormalized to mean one.",
    )
    parser.add_argument(
        "--occupied-shell-threshold",
        type=float,
        default=1.0e-3,
        help="Minimum population fraction required before momentum losses are enforced on a shell.",
    )
    parser.add_argument(
        "--momentum-loss-weight",
        type=float,
        default=1.0,
        help="Relative weight applied to the conditional momentum loss term.",
    )
    parser.add_argument(
        "--closure-loss-weight",
        type=float,
        default=1.0,
        help="Relative weight applied to closure-channel heteroscedastic regression when enabled.",
    )
    parser.add_argument(
        "--flux-loss-weight",
        type=float,
        default=0.0,
        help="Relative weight applied to the conservative transport consistency term.",
    )
    parser.add_argument(
        "--conservative-state-loss-weight",
        type=float,
        default=0.25,
        help="Direct state-loss weight on the conservative shell pair when the flux head is enabled.",
    )
    parser.add_argument(
        "--non-conservative-state-loss-weight",
        type=float,
        default=1.0,
        help="Direct state-loss weight on all non-conservative shell channels when the flux head is enabled.",
    )
    parser.add_argument(
        "--residual-conservative-weight",
        type=float,
        default=0.0,
        help="Relative weight applied to the conservative residual-source regularizer.",
    )
    parser.add_argument(
        "--source-conservative-weight",
        type=float,
        default=0.0,
        help="Relative weight applied to the source-branch conservative regularizer in operator-split mode.",
    )
    parser.add_argument(
        "--support-band-source-suppression-weight",
        type=float,
        default=1.0,
        help="Relative weight on keeping operator-split source conservative updates smaller than transport inside the primary support band.",
    )
    parser.add_argument(
        "--support-outflow-saturation-weight",
        type=float,
        default=0.05,
        help="Mild penalty on redistribution runs that saturate the bounded one-step outflow cap across shells.",
    )
    parser.add_argument(
        "--boundary-source-loss-weight",
        type=float,
        default=0.0,
        help="Extra loss weight applied only on lowest-shell boundary-response channels.",
    )
    parser.add_argument(
        "--support-band-loss-weight",
        type=float,
        default=0.0,
        help="Extra loss weight for population-support channels in the problematic altitude band.",
    )
    parser.add_argument(
        "--support-band-lower-km",
        type=float,
        default=2900.0,
        help="Lower altitude bound of the support-collapse band to emphasize.",
    )
    parser.add_argument(
        "--support-band-upper-km",
        type=float,
        default=4700.0,
        help="Upper altitude bound of the support-collapse band to emphasize.",
    )
    parser.add_argument(
        "--secondary-support-band-loss-weight",
        type=float,
        default=0.0,
        help="Extra loss weight for the residual upper-altitude support-collapse band.",
    )
    parser.add_argument(
        "--secondary-support-band-lower-km",
        type=float,
        default=6900.0,
        help="Lower altitude bound of the residual upper support-collapse band.",
    )
    parser.add_argument(
        "--secondary-support-band-upper-km",
        type=float,
        default=7800.0,
        help="Upper altitude bound of the residual upper support-collapse band.",
    )
    parser.add_argument(
        "--diagnostic-mean-vpar-loss-weight",
        type=float,
        default=0.0,
        help="Low-weight auxiliary loss on recovered occupied-shell mean_vpar diagnostics.",
    )
    parser.add_argument(
        "--diagnostic-mean-vperp-loss-weight",
        type=float,
        default=0.0,
        help="Low-weight auxiliary loss on recovered occupied-shell mean_vperp diagnostics.",
    )
    parser.add_argument(
        "--diagnostic-std-vpar-loss-weight",
        type=float,
        default=0.0,
        help="Low-weight auxiliary loss on recovered occupied-shell std_vpar diagnostics.",
    )
    parser.add_argument(
        "--diagnostic-loss-warmup-epochs",
        type=int,
        default=0,
        help="Linearly ramp occupied-shell primitive diagnostic losses over this many epochs.",
    )
    parser.add_argument(
        "--diagnostic-loss-warmup-start-factor",
        type=float,
        default=0.0,
        help="Initial fraction of each diagnostic primitive-loss weight before warmup reaches full strength.",
    )
    parser.add_argument(
        "--occupancy-gate-center",
        type=float,
        default=1.0e-3,
        help="Population-fraction gate center used by the conditional momentum head.",
    )
    parser.add_argument(
        "--occupancy-gate-width",
        type=float,
        default=5.0e-4,
        help="Population-fraction gate width used by the conditional momentum head.",
    )
    parser.add_argument("--device", choices=["cpu", "mps", "cuda", "auto"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose-data", action="store_true")
    return parser.parse_args()


def select_device(choice: str) -> torch.device:
    """Select a torch device without forcing accelerator-specific dependencies."""
    if choice == "cuda":
        return torch.device("cuda")
    if choice == "mps":
        return torch.device("mps")
    if choice == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    """Seed NumPy, Python, and torch for reproducible graph training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_shell_edges_override(shell_edges_json: str) -> np.ndarray | None:
    """Load one explicit shell-edge array from JSON when overriding default bins.

    Physical interpretation:
    - A custom shell layout lets us concentrate graph resolution in the altitude band where rollout
      diagnostics show the surrogate losing support and parallel-drift fidelity.
    - The override is expressed as fixed shell edges in kilometers so train and valid use exactly
      the same field-line discretization.
    """
    normalized_path = str(shell_edges_json or "").strip()
    if not normalized_path:
        return None
    with open(normalized_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict):
        if "shell_edges_km" not in payload:
            raise ValueError(f"Shell-edge JSON {normalized_path} is missing 'shell_edges_km'.")
        payload = payload["shell_edges_km"]
    edges = np.asarray(payload, dtype=np.float32)
    if edges.ndim != 1 or edges.size < 2:
        raise ValueError(f"Shell-edge JSON {normalized_path} must define a 1D edge array.")
    if not np.all(np.isfinite(edges)):
        raise ValueError(f"Shell-edge JSON {normalized_path} contains non-finite values.")
    if not np.all(np.diff(edges) > 0.0):
        raise ValueError(f"Shell-edge JSON {normalized_path} must be strictly increasing.")
    return edges


def _stats_to_torch(
    stats,
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
    """Move normalization tensors to the active training device."""
    target_schema = validate_target_schema(target_schema)
    schema_feature_names = dynamic_feature_names_for_schema(target_schema)
    schema_feature_index = {
        feature_name: feature_offset for feature_offset, feature_name in enumerate(schema_feature_names)
    }
    schema_asinh_feature_scales = SCHEMA_ASINH_FEATURE_SCALES[target_schema]
    asinh_feature_names = tuple(schema_asinh_feature_scales.keys())
    asinh_indices = [int(schema_feature_index[feature_name]) for feature_name in asinh_feature_names]
    asinh_scales = [float(schema_asinh_feature_scales[feature_name]) for feature_name in asinh_feature_names]
    # The closure schema reuses the momentum slots, so signed and nonnegative moment constraints
    # still apply to the same channel positions even though their physical interpretation changes.
    signed_velocity_asinh_indices = (
        [int(schema_feature_index["parallel_momentum_density_proxy"])]
        if target_schema == TARGET_SCHEMA_CLOSURE_TRANSPORT
        else list(SIGNED_VELOCITY_ASINH_CHANNEL_INDICES)
    )
    nonnegative_velocity_asinh_indices = (
        [
            int(schema_feature_index["perpendicular_momentum_proxy"]),
            int(schema_feature_index["parallel_energy_proxy"]),
            int(schema_feature_index["perpendicular_energy_proxy"]),
        ]
        if target_schema == TARGET_SCHEMA_CLOSURE_TRANSPORT
        else list(NONNEGATIVE_VELOCITY_ASINH_CHANNEL_INDICES)
    )
    if velocity_species_weight_per_node is None:
        # If missing, default to neutral per-node velocity weights.
        velocity_species_weight_per_node = np.ones(num_nodes, dtype=np.float32)
    if boundary_source_node_mask is None:
        # If missing, disable lower-boundary source loss.
        boundary_source_node_mask = np.zeros(num_nodes, dtype=np.float32)
    if support_band_node_mask is None:
        # If missing, disable the auxiliary support-band loss.
        support_band_node_mask = np.zeros(num_nodes, dtype=np.float32)
    if secondary_support_band_node_mask is None:
        # If missing, disable the secondary support-band loss.
        secondary_support_band_node_mask = np.zeros(num_nodes, dtype=np.float32)
    if transport_delta_scale is None:
        # If missing, use unit scaling for conservative transport channels.
        transport_delta_scale = np.ones(len(TRANSPORT_CONSERVATIVE_CHANNEL_INDICES), dtype=np.float32)
    if transport_shell_node_mask is None and transport_interfaces is not None:
        # The global node is not part of the conservative shell stack and should not receive
        # transport or conservative residual updates from the flux head.
        transport_shell_node_mask = np.ones(num_nodes, dtype=np.float32)
    if transport_shell_node_mask is None:
        transport_shell_node_mask = np.zeros(num_nodes, dtype=np.float32)
    if transport_cell_width_km is None:
        # If missing, use unit shell widths.
        transport_cell_width_km = np.ones(num_nodes, dtype=np.float32)
    if transport_interface_width_km is None:
        # If missing, use unit interface widths.
        interface_count = 0 if transport_interfaces is None else int(transport_interfaces.left_node_index.shape[0])
        transport_interface_width_km = np.ones(interface_count, dtype=np.float32)
    non_transport_conservative_channel_indices = [
        feature_index
        for feature_index in range(int(stats.dynamic_mean.shape[0]))
        if feature_index not in set(TRANSPORT_CONSERVATIVE_CHANNEL_INDICES)
    ]
    return {
        # Rollout reconstruction normalizes rebuilt history windows with the same input statistics
        # used during one-step training, so these tensors must be shared through the common stats
        # dictionary instead of maintaining a second inconsistent copy in the rollout script.
        "input_mean": torch.as_tensor(stats.input_mean, dtype=torch.float32, device=device),
        "input_std": torch.as_tensor(stats.input_std, dtype=torch.float32, device=device),
        "context_mean": torch.as_tensor(stats.context_mean, dtype=torch.float32, device=device),
        "context_std": torch.as_tensor(stats.context_std, dtype=torch.float32, device=device),
        "boundary_context_mean": torch.as_tensor(
            stats.boundary_context_mean,
            dtype=torch.float32,
            device=device,
        ),
        "boundary_context_std": torch.as_tensor(
            stats.boundary_context_std,
            dtype=torch.float32,
            device=device,
        ),
        "dynamic_mean": torch.as_tensor(stats.dynamic_mean, dtype=torch.float32, device=device),
        "dynamic_std": torch.as_tensor(stats.dynamic_std, dtype=torch.float32, device=device),
        "delta_mean": torch.as_tensor(stats.delta_mean, dtype=torch.float32, device=device),
        "delta_std": torch.as_tensor(stats.delta_std, dtype=torch.float32, device=device),
        "population_channel_index": torch.as_tensor(
            [POPULATION_FRACTION_CHANNEL_INDEX],
            dtype=torch.long,
            device=device,
        ),
        # Rollout keeps endogenous shell state autoregressive while replacing only exogenous
        # controls with the true future values, so it still needs the original state/context split.
        "state_channel_index": torch.as_tensor(
            DYNAMIC_STATE_CHANNEL_INDICES,
            dtype=torch.long,
            device=device,
        ),
        "context_channel_index": torch.as_tensor(
            DYNAMIC_CONTEXT_CHANNEL_INDICES,
            dtype=torch.long,
            device=device,
        ),
        "occupancy_shape_channel_index": torch.as_tensor(
            DYNAMIC_OCCUPANCY_SHAPE_CHANNEL_INDICES,
            dtype=torch.long,
            device=device,
        ),
        "forcing_channel_index": torch.as_tensor(
            DYNAMIC_FORCING_CHANNEL_INDICES,
            dtype=torch.long,
            device=device,
        ),
        "momentum_channel_index": torch.as_tensor(
            DYNAMIC_MOMENTUM_CHANNEL_INDICES,
            dtype=torch.long,
            device=device,
        ),
        "support_band_channel_index": torch.as_tensor(
            [
                DYNAMIC_FEATURE_INDEX["population_fraction"],
                DYNAMIC_FEATURE_INDEX["log_total_weight"],
            ],
            dtype=torch.long,
            device=device,
        ),
        "occupancy_weighted_channel_index": torch.as_tensor(
            OCCUPANCY_WEIGHTED_CHANNEL_INDICES,
            dtype=torch.long,
            device=device,
        ),
        "asinh_channel_index": torch.as_tensor(
            asinh_indices,
            dtype=torch.long,
            device=device,
        ),
        "asinh_channel_scale": torch.as_tensor(
            asinh_scales,
            dtype=torch.float32,
            device=device,
        ),
        "nonnegative_channel_index": torch.as_tensor(
            DYNAMIC_NONNEGATIVE_CHANNEL_INDICES,
            dtype=torch.long,
            device=device,
        ),
        "unit_interval_channel_index": torch.as_tensor(
            DYNAMIC_UNIT_INTERVAL_CHANNEL_INDICES,
            dtype=torch.long,
            device=device,
        ),
        "signed_velocity_asinh_channel_index": torch.as_tensor(
            signed_velocity_asinh_indices,
            dtype=torch.long,
            device=device,
        ),
        "nonnegative_velocity_asinh_channel_index": torch.as_tensor(
            nonnegative_velocity_asinh_indices,
            dtype=torch.long,
            device=device,
        ),
        "velocity_species_weight_per_node": torch.as_tensor(
            velocity_species_weight_per_node,
            dtype=torch.float32,
            device=device,
        ),
        "boundary_source_node_mask": torch.as_tensor(
            boundary_source_node_mask,
            dtype=torch.float32,
            device=device,
        ),
        "boundary_response_channel_index": torch.as_tensor(
            BOUNDARY_RESPONSE_CHANNEL_INDICES,
            dtype=torch.long,
            device=device,
        ),
        "support_band_node_mask": torch.as_tensor(
            support_band_node_mask,
            dtype=torch.float32,
            device=device,
        ),
        "secondary_support_band_node_mask": torch.as_tensor(
            secondary_support_band_node_mask,
            dtype=torch.float32,
            device=device,
        ),
        "transport_conservative_channel_index": torch.as_tensor(
            TRANSPORT_CONSERVATIVE_CHANNEL_INDICES,
            dtype=torch.long,
            device=device,
        ),
        "non_transport_conservative_channel_index": torch.as_tensor(
            non_transport_conservative_channel_indices,
            dtype=torch.long,
            device=device,
        ),
        "transport_delta_scale": torch.as_tensor(
            transport_delta_scale,
            dtype=torch.float32,
            device=device,
        ),
        "transport_left_index": (
            torch.as_tensor(transport_interfaces.left_node_index, dtype=torch.long, device=device)
            if transport_interfaces is not None
            else None
        ),
        "transport_right_index": (
            torch.as_tensor(transport_interfaces.right_node_index, dtype=torch.long, device=device)
            if transport_interfaces is not None
            else None
        ),
        "transport_interface_features": (
            torch.as_tensor(transport_interfaces.interface_features, dtype=torch.float32, device=device)
            if transport_interfaces is not None
            else None
        ),
        "transport_interface_width_km": (
            torch.as_tensor(transport_interface_width_km, dtype=torch.float32, device=device)
            if transport_interfaces is not None
            else None
        ),
        "transport_shell_node_mask": (
            torch.as_tensor(transport_shell_node_mask, dtype=torch.float32, device=device)
            if transport_interfaces is not None
            else None
        ),
        "transport_cell_width_km": torch.as_tensor(
            transport_cell_width_km,
            dtype=torch.float32,
            device=device,
        ),
    }


def build_support_band_node_mask(
    layout,
    shell_edges_km: np.ndarray,
    lower_km: float,
    upper_km: float,
) -> np.ndarray:
    """Return a per-node mask for the altitude band where shell support collapses.

    Physical interpretation:
    - Diagnostics showed that rollout errors cluster in a mid/high-altitude band where predicted
      shell occupancy falls to zero while the true shell still has a small but nonzero population.
    - The same shell mask is repeated for every species stack because the failure is shared across
      species rather than isolated to one ion population.
    """
    shell_centers_km = 0.5 * (np.asarray(shell_edges_km[:-1]) + np.asarray(shell_edges_km[1:]))
    shell_mask = (
        (shell_centers_km >= float(lower_km))
        & (shell_centers_km <= float(upper_km))
    ).astype(np.float32)
    node_mask = np.zeros(layout.num_nodes, dtype=np.float32)
    for species_offset in range(layout.num_species):
        start = species_offset * layout.num_altitude_shells
        stop = start + layout.num_altitude_shells
        node_mask[start:stop] = shell_mask
    if layout.global_node_index is not None:
        node_mask[layout.global_node_index] = 0.0
    return node_mask


def compute_diagnostic_scale(
    target_dynamic_physical: np.ndarray,
    feature_name: str,
    occupied_shell_threshold: float,
    target_schema: str = TARGET_SCHEMA_POPULATION_WEIGHTED,
) -> float:
    """Estimate one dimensionless occupied-shell primitive-diagnostic scale.

    Physical interpretation:
    - The auxiliary primitive losses should stay dimensionless so their weights are interpretable.
    - The scale is estimated only where the requested diagnostic is physically defined, which
      prevents empty-shell velocity moments from distorting the normalization.
    """
    diagnostic_channel = extract_dynamic_diagnostic_channel(
        dynamic_physical=target_dynamic_physical,
        feature_name=feature_name,
        target_schema=target_schema,
        population_epsilon=occupied_shell_threshold,
    )
    valid_mask = dynamic_diagnostic_valid_mask(
        dynamic_physical=target_dynamic_physical,
        feature_name=feature_name,
        target_schema=target_schema,
        population_epsilon=occupied_shell_threshold,
    )
    if not np.any(valid_mask):
        return 1.0
    scale = float(np.std(diagnostic_channel[valid_mask]))
    return max(scale, 1.0)


def compute_mean_vpar_diagnostic_scale(
    target_dynamic_physical: np.ndarray,
    occupied_shell_threshold: float,
    target_schema: str = TARGET_SCHEMA_POPULATION_WEIGHTED,
) -> float:
    """Estimate the occupied-shell normalization scale for recovered mean_vpar."""
    return compute_diagnostic_scale(
        target_dynamic_physical=target_dynamic_physical,
        feature_name="mean_vpar",
        occupied_shell_threshold=occupied_shell_threshold,
        target_schema=target_schema,
    )


def compute_mean_vperp_diagnostic_scale(
    target_dynamic_physical: np.ndarray,
    occupied_shell_threshold: float,
    target_schema: str = TARGET_SCHEMA_POPULATION_WEIGHTED,
) -> float:
    """Estimate the occupied-shell normalization scale for recovered mean_vperp."""
    return compute_diagnostic_scale(
        target_dynamic_physical=target_dynamic_physical,
        feature_name="mean_vperp",
        occupied_shell_threshold=occupied_shell_threshold,
        target_schema=target_schema,
    )


def compute_std_vpar_diagnostic_scale(
    target_dynamic_physical: np.ndarray,
    occupied_shell_threshold: float,
    target_schema: str = TARGET_SCHEMA_POPULATION_WEIGHTED,
) -> float:
    """Estimate the occupied-shell normalization scale for recovered std_vpar."""
    return compute_diagnostic_scale(
        target_dynamic_physical=target_dynamic_physical,
        feature_name="std_vpar",
        occupied_shell_threshold=occupied_shell_threshold,
        target_schema=target_schema,
    )


def compute_transport_delta_scale(
    target_dynamic_physical: np.ndarray,
    last_dynamic_physical: np.ndarray,
) -> np.ndarray:
    """Estimate normalization scales for the conservative physical shell increments.

    Physical interpretation:
    - The flux-consistency term should compare transported support and transported parallel
      momentum on dimensionless scales.
    - The scale is estimated from train-split one-step physical increments so the flux head is
      judged relative to the variability that actually exists in PWOM shell updates.
    """
    conservative_delta = (
        np.asarray(target_dynamic_physical[..., TRANSPORT_CONSERVATIVE_CHANNEL_INDICES], dtype=np.float32)
        - np.asarray(last_dynamic_physical[..., TRANSPORT_CONSERVATIVE_CHANNEL_INDICES], dtype=np.float32)
    )
    scale = np.std(conservative_delta.reshape(-1, conservative_delta.shape[-1]), axis=0).astype(np.float32, copy=False)
    return np.maximum(scale, np.asarray([1.0e-6, 1.0], dtype=np.float32))


def reconstruct_prediction_raw(
    pred_norm: torch.Tensor,
    last_dynamic: torch.Tensor,
    stats_torch: dict[str, torch.Tensor],
    predict_delta_state: bool,
) -> torch.Tensor:
    """Map normalized model outputs back to model-space next-step shell-state units."""
    if predict_delta_state:
        pred_delta = pred_norm * stats_torch["delta_std"] + stats_torch["delta_mean"]
        # Residual reconstruction matches the physical expectation that shell states change
        # incrementally between neighboring PWOM snapshots.
        return last_dynamic + pred_delta
    return pred_norm * stats_torch["dynamic_std"] + stats_torch["dynamic_mean"]


def inverse_transform_dynamic_torch(
    dynamic_model: torch.Tensor,
    stats_torch: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Map model-space shell states back to physical units on the active torch device.

    Physical interpretation:
    - Velocity moments are stored in compressed `asinh` coordinates during learning.
    - Physical diagnostics must be computed after undoing that compression.
    """
    dynamic_physical = dynamic_model.clone()
    asinh_channel_index = stats_torch["asinh_channel_index"]
    if int(asinh_channel_index.numel()) == 0:
        return dynamic_physical

    asinh_channel_scale = stats_torch["asinh_channel_scale"]
    transformed = dynamic_physical.index_select(dim=-1, index=asinh_channel_index)
    dynamic_physical[..., asinh_channel_index] = torch.sinh(transformed) * asinh_channel_scale
    return dynamic_physical


def recovered_primitive_diagnostic_torch(
    dynamic_physical: torch.Tensor,
    feature_name: str,
    occupied_shell_threshold: float,
    target_schema: str = TARGET_SCHEMA_POPULATION_WEIGHTED,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Recover one occupied-shell primitive diagnostic from physical tensors.

    Physical interpretation:
    - Primitive velocity diagnostics are derived from population-weighted shell targets.
    - They are only meaningful where shell occupancy is large enough to represent a real
      species-conditioned distribution.
    """
    target_schema = validate_target_schema(target_schema)
    population_fraction = dynamic_physical[..., POPULATION_FRACTION_CHANNEL_INDEX]
    valid_mask = population_fraction >= float(occupied_shell_threshold)
    safe_population = torch.clamp(population_fraction, min=float(occupied_shell_threshold))

    if target_schema == TARGET_SCHEMA_CLOSURE_TRANSPORT:
        # Closure targets recover primitive means and spreads by dividing momentum/energy proxies
        # by shell occupancy, with a square root for energy-like spread channels.
        if feature_name == "mean_vpar":
            source_channel = dynamic_physical[..., DYNAMIC_CLOSURE_CHANNEL_INDICES[0]]
            recovered_channel = torch.where(
                valid_mask,
                source_channel / safe_population,
                torch.zeros_like(source_channel),
            )
        elif feature_name == "std_vpar":
            source_channel = dynamic_physical[..., DYNAMIC_CLOSURE_CHANNEL_INDICES[2]]
            recovered_channel = torch.where(
                valid_mask,
                torch.sqrt(torch.clamp(source_channel / safe_population, min=0.0)),
                torch.zeros_like(source_channel),
            )
        else:
            raise ValueError(f"Unsupported primitive diagnostic feature_name='{feature_name}'.")
        return recovered_channel, valid_mask

    if feature_name == "mean_vpar":
        source_channel = dynamic_physical[..., DYNAMIC_FEATURE_INDEX["population_weighted_mean_vpar"]]
    elif feature_name == "mean_vperp":
        source_channel = dynamic_physical[..., DYNAMIC_FEATURE_INDEX["population_weighted_mean_vperp"]]
    elif feature_name == "std_vpar":
        source_channel = dynamic_physical[..., DYNAMIC_FEATURE_INDEX["population_weighted_std_vpar"]]
    else:
        raise ValueError(f"Unsupported primitive diagnostic feature_name='{feature_name}'.")

    recovered_channel = torch.where(
        valid_mask,
        source_channel / safe_population,
        torch.zeros_like(source_channel),
    )
    return recovered_channel, valid_mask


def masked_mean_vpar_diagnostic_loss(
    pred_dynamic_physical: torch.Tensor,
    target_dynamic_physical: torch.Tensor,
    occupied_shell_threshold: float,
    diagnostic_mean_vpar_scale: float,
    node_mask: torch.Tensor | None = None,
    target_schema: str = TARGET_SCHEMA_POPULATION_WEIGHTED,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute a normalized auxiliary loss on occupied-shell recovered mean_vpar."""
    return masked_primitive_diagnostic_loss(
        pred_dynamic_physical=pred_dynamic_physical,
        target_dynamic_physical=target_dynamic_physical,
        feature_name="mean_vpar",
        occupied_shell_threshold=occupied_shell_threshold,
        diagnostic_scale=diagnostic_mean_vpar_scale,
        node_mask=node_mask,
        target_schema=target_schema,
    )


def masked_mean_vperp_diagnostic_loss(
    pred_dynamic_physical: torch.Tensor,
    target_dynamic_physical: torch.Tensor,
    occupied_shell_threshold: float,
    diagnostic_mean_vperp_scale: float,
    node_mask: torch.Tensor | None = None,
    target_schema: str = TARGET_SCHEMA_POPULATION_WEIGHTED,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute a normalized auxiliary loss on occupied-shell recovered mean_vperp."""
    return masked_primitive_diagnostic_loss(
        pred_dynamic_physical=pred_dynamic_physical,
        target_dynamic_physical=target_dynamic_physical,
        feature_name="mean_vperp",
        occupied_shell_threshold=occupied_shell_threshold,
        diagnostic_scale=diagnostic_mean_vperp_scale,
        node_mask=node_mask,
        target_schema=target_schema,
    )


def masked_std_vpar_diagnostic_loss(
    pred_dynamic_physical: torch.Tensor,
    target_dynamic_physical: torch.Tensor,
    occupied_shell_threshold: float,
    diagnostic_std_vpar_scale: float,
    node_mask: torch.Tensor | None = None,
    target_schema: str = TARGET_SCHEMA_POPULATION_WEIGHTED,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute a normalized auxiliary loss on occupied-shell recovered std_vpar."""
    return masked_primitive_diagnostic_loss(
        pred_dynamic_physical=pred_dynamic_physical,
        target_dynamic_physical=target_dynamic_physical,
        feature_name="std_vpar",
        occupied_shell_threshold=occupied_shell_threshold,
        diagnostic_scale=diagnostic_std_vpar_scale,
        node_mask=node_mask,
        target_schema=target_schema,
    )


def broadcast_optional_node_mask(
    node_mask: torch.Tensor | None,
    valid_mask: torch.Tensor,
) -> torch.Tensor | None:
    """Broadcast one per-node regime mask across batch/time axes.

    Physical interpretation:
    - The support-band mask lives on nodes only because altitude bands are properties of shell
      locations, not of individual rollout samples.
    - Primitive diagnostics can then be restricted to one altitude regime by expanding that
      node-wise mask across every batch and rollout step.
    """
    if node_mask is None:
        return None
    if node_mask.ndim != 1:
        raise ValueError("Expected one-dimensional node_mask indexed by graph node.")
    view_shape = [1] * max(valid_mask.ndim - 1, 0) + [int(node_mask.shape[0])]
    return node_mask.to(device=valid_mask.device, dtype=torch.bool).view(*view_shape)


def masked_primitive_diagnostic_loss(
    pred_dynamic_physical: torch.Tensor,
    target_dynamic_physical: torch.Tensor,
    feature_name: str,
    occupied_shell_threshold: float,
    diagnostic_scale: float,
    node_mask: torch.Tensor | None = None,
    target_schema: str = TARGET_SCHEMA_POPULATION_WEIGHTED,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute one normalized auxiliary loss on an occupied-shell primitive diagnostic.

    Physical interpretation:
    - This term nudges the model toward better primitive velocity recovery only where that
      primitive is physically defined.
    - The occupied-shell mask keeps undefined velocity diagnostics from forcing arbitrary values in
      empty shells.
    """
    pred_feature, _ = recovered_primitive_diagnostic_torch(
        dynamic_physical=pred_dynamic_physical,
        feature_name=feature_name,
        occupied_shell_threshold=occupied_shell_threshold,
        target_schema=target_schema,
    )
    target_feature, valid_mask = recovered_primitive_diagnostic_torch(
        dynamic_physical=target_dynamic_physical,
        feature_name=feature_name,
        occupied_shell_threshold=occupied_shell_threshold,
        target_schema=target_schema,
    )
    band_mask = broadcast_optional_node_mask(node_mask=node_mask, valid_mask=valid_mask)
    if band_mask is not None:
        # The regime mask localizes primitive supervision to the altitude shells where rollout
        # diagnostics show persistent kinetic drift instead of forcing a global compromise.
        valid_mask = torch.logical_and(valid_mask, band_mask)
    if not torch.any(valid_mask):
        zero_value = torch.zeros((), dtype=pred_dynamic_physical.dtype, device=pred_dynamic_physical.device)
        return zero_value, zero_value
    normalized_sq_error = torch.square(
        (pred_feature - target_feature) / max(float(diagnostic_scale), 1.0)
    )
    loss = torch.sum(normalized_sq_error * valid_mask.to(dtype=normalized_sq_error.dtype)) / torch.clamp(
        valid_mask.to(dtype=normalized_sq_error.dtype).sum(),
        min=1.0,
    )
    valid_fraction = valid_mask.to(dtype=normalized_sq_error.dtype).mean()
    return loss, valid_fraction


def compute_effective_diagnostic_weight(
    base_weight: float,
    epoch_index: int,
    warmup_epochs: int,
    warmup_start_factor: float,
) -> float:
    """Return one linearly warmed diagnostic-loss weight for the current epoch.

    Physical interpretation:
    - Primitive velocity diagnostics should only start influencing optimization once the coarse
      shell-state fit is already stable.
    - A short warmup prevents the auxiliary term from overpowering the base transport objective in
      the first few epochs.
    """
    if base_weight <= 0.0:
        return 0.0
    if warmup_epochs <= 0:
        return float(base_weight)
    clamped_start_factor = min(max(float(warmup_start_factor), 0.0), 1.0)
    progress = min(max(epoch_index - 1, 0), warmup_epochs) / max(warmup_epochs, 1)
    effective_factor = clamped_start_factor + ((1.0 - clamped_start_factor) * progress)
    return float(base_weight) * effective_factor


def apply_dynamic_constraints_model_space(
    pred_dynamic: torch.Tensor,
    stats_torch: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Apply simple physical bounds directly in the transformed model space.

    Physical interpretation:
    - Occupancy-like channels and positive-definite moments cannot go negative.
    - Probability-like channels remain inside `[0, 1]`.
    - Velocity moments are bounded in compressed `asinh` space so inverse transforms do not produce
      unphysical or numerically explosive shell speeds.
    - Hard bounds keep inverse transforms numerically stable under autoregressive use.
    """
    constrained = pred_dynamic.clone()
    nonnegative_index = stats_torch["nonnegative_channel_index"]
    unit_interval_index = stats_torch["unit_interval_channel_index"]
    signed_velocity_asinh_index = stats_torch["signed_velocity_asinh_channel_index"]
    nonnegative_velocity_asinh_index = stats_torch["nonnegative_velocity_asinh_channel_index"]

    if int(nonnegative_index.numel()) > 0:
        constrained[..., nonnegative_index] = torch.clamp(constrained[..., nonnegative_index], min=0.0)
    if int(unit_interval_index.numel()) > 0:
        constrained[..., unit_interval_index] = torch.clamp(
            constrained[..., unit_interval_index],
            min=0.0,
            max=1.0,
        )
    if int(signed_velocity_asinh_index.numel()) > 0:
        constrained[..., signed_velocity_asinh_index] = torch.clamp(
            constrained[..., signed_velocity_asinh_index],
            min=-VELOCITY_ASINH_ABS_MAX,
            max=VELOCITY_ASINH_ABS_MAX,
        )
    if int(nonnegative_velocity_asinh_index.numel()) > 0:
        constrained[..., nonnegative_velocity_asinh_index] = torch.clamp(
            constrained[..., nonnegative_velocity_asinh_index],
            min=0.0,
            max=VELOCITY_ASINH_ABS_MAX,
        )
    return constrained


def build_channel_loss_weights(
    target_dynamic: torch.Tensor,
    stats_torch: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Build per-channel loss weights for noisy velocity moments.

    Physical interpretation:
    - A shell with very little particle weight has noisy sample moments, especially for velocity.
    - Species-aware reweighting optionally compensates for ion populations that are consistently
      underrepresented across the training split.
    - The combined weighting keeps dense shells and dominant species from fully controlling the
      velocity objective while leaving non-velocity channels untouched.
    """
    channel_weights = torch.ones_like(target_dynamic)
    occupancy_weighted_channel_index = stats_torch["occupancy_weighted_channel_index"]
    if int(occupancy_weighted_channel_index.numel()) == 0:
        return channel_weights

    population_channel_index = stats_torch["population_channel_index"]
    population_fraction = torch.clamp(
        target_dynamic.index_select(dim=-1, index=population_channel_index).squeeze(-1),
        min=1.0e-6,
        max=1.0,
    )
    occupancy_weight = torch.clamp(torch.sqrt(population_fraction), min=5.0e-2, max=1.0)
    velocity_species_weight = stats_torch["velocity_species_weight_per_node"]
    node_weight_view_shape = [1] * target_dynamic.ndim
    node_weight_view_shape[-2] = int(velocity_species_weight.shape[0])
    node_weight_view_shape[-1] = 1
    velocity_species_weight = velocity_species_weight.view(*node_weight_view_shape)
    channel_weights[..., occupancy_weighted_channel_index] = (
        occupancy_weight.unsqueeze(-1) * velocity_species_weight
    )
    return channel_weights


def occupied_shell_mask_torch(
    dynamic_state: torch.Tensor,
    stats_torch: dict[str, torch.Tensor],
    occupied_shell_threshold: float,
) -> torch.Tensor:
    """Return a boolean shell mask for momentum losses and diagnostics.

    Physical interpretation:
    - A shell must contain enough particle weight before a species-conditioned velocity moment is
      a meaningful supervision target.
    - This keeps the momentum objective focused on shells where the DSMC statistics are defined.
    """
    population_channel_index = stats_torch["population_channel_index"]
    population_fraction = dynamic_state.index_select(dim=-1, index=population_channel_index).squeeze(-1)
    return population_fraction >= float(occupied_shell_threshold)


def weighted_normalized_dynamic_mse(
    pred_dynamic: torch.Tensor,
    target_dynamic: torch.Tensor,
    stats_torch: dict[str, torch.Tensor],
    channel_index: torch.Tensor | None = None,
    element_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute normalized shell-state MSE with occupancy weighting on noisy velocity moments."""
    original_feature_dim = pred_dynamic.shape[-1]
    dynamic_std = stats_torch["dynamic_std"]
    channel_weights = build_channel_loss_weights(target_dynamic=target_dynamic, stats_torch=stats_torch)

    if channel_index is not None:
        pred_dynamic = pred_dynamic.index_select(dim=-1, index=channel_index)
        target_dynamic = target_dynamic.index_select(dim=-1, index=channel_index)
        dynamic_std = dynamic_std.index_select(0, channel_index)
        channel_weights = channel_weights.index_select(dim=-1, index=channel_index)
        if element_weight is not None and element_weight.shape[-1] == original_feature_dim:
            element_weight = element_weight.index_select(dim=-1, index=channel_index)

    if element_weight is not None:
        if element_weight.ndim == pred_dynamic.ndim - 1:
            element_weight = element_weight.unsqueeze(-1)
        if element_weight.shape[-1] == 1 and pred_dynamic.shape[-1] != 1:
            element_weight = element_weight.expand_as(pred_dynamic)
        channel_weights = channel_weights * element_weight.to(dtype=channel_weights.dtype)

    sq_error = torch.square((pred_dynamic - target_dynamic) / dynamic_std)
    return torch.sum(channel_weights * sq_error) / torch.clamp(torch.sum(channel_weights), min=1.0)


def occupied_fraction_metric(
    target_dynamic: torch.Tensor,
    stats_torch: dict[str, torch.Tensor],
    occupied_shell_threshold: float,
) -> torch.Tensor:
    """Measure occupied-shell coverage directly from the target population channel.

    Physical interpretation:
    - Occupancy is a property of the target shell distribution, not of one particular loss branch.
    - Computing it centrally keeps diagnostics consistent across vanilla, flux, and conditional
      training modes.
    """
    occupied_shell_mask = occupied_shell_mask_torch(
        dynamic_state=target_dynamic,
        stats_torch=stats_torch,
        occupied_shell_threshold=occupied_shell_threshold,
    )
    return occupied_shell_mask.to(dtype=target_dynamic.dtype).mean()


def support_band_loss(
    pred_dynamic: torch.Tensor,
    target_dynamic: torch.Tensor,
    stats_torch: dict[str, torch.Tensor],
    mask_key: str = "support_band_node_mask",
) -> torch.Tensor:
    """Compute a focused population-support loss in the diagnosed failure band.

    Physical interpretation:
    - The failing regime is primarily a shell-support problem: population fraction and total shell
      weight collapse in a specific altitude band before the velocity diagnostics degrade.
    - This term adds extra supervision only on those density-like channels and only on those band
      nodes, instead of globally changing the architecture or every loss term.
    """
    support_band_channel_index = stats_torch["support_band_channel_index"]
    support_band_node_mask = stats_torch[mask_key]
    if not torch.any(support_band_node_mask > 0.0):
        return torch.zeros((), dtype=pred_dynamic.dtype, device=pred_dynamic.device)
    node_weight_shape = [1] * pred_dynamic.ndim
    node_weight_shape[-2] = int(support_band_node_mask.shape[0])
    node_weight = support_band_node_mask.view(*node_weight_shape)
    return weighted_normalized_dynamic_mse(
        pred_dynamic=pred_dynamic,
        target_dynamic=target_dynamic,
        stats_torch=stats_torch,
        channel_index=support_band_channel_index,
        element_weight=node_weight,
    )


def boundary_source_loss(
    pred_dynamic: torch.Tensor,
    target_dynamic: torch.Tensor,
    stats_torch: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Compute focused supervision on lowest-shell boundary-response channels only.

    Physical interpretation:
    - Boundary moments should mainly affect the base of the field line, not every shell.
    - This term encourages the new boundary branch to improve the physically relevant response
      channels where PWOM actually injects or replenishes particles.
    """
    boundary_source_node_mask = stats_torch["boundary_source_node_mask"]
    if not torch.any(boundary_source_node_mask > 0.0):
        return torch.zeros((), dtype=pred_dynamic.dtype, device=pred_dynamic.device)
    node_weight_shape = [1] * pred_dynamic.ndim
    node_weight_shape[-2] = int(boundary_source_node_mask.shape[0])
    node_weight = boundary_source_node_mask.view(*node_weight_shape)
    return weighted_normalized_dynamic_mse(
        pred_dynamic=pred_dynamic,
        target_dynamic=target_dynamic,
        stats_torch=stats_torch,
        channel_index=stats_torch["boundary_response_channel_index"],
        element_weight=node_weight,
    )


def transport_flux_consistency_loss(
    target_dynamic_physical: torch.Tensor,
    last_dynamic_physical: torch.Tensor,
    transport_delta_physical: torch.Tensor,
    stats_torch: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Compare the flux-derived conservative shell update against the true one-step increment.

    Physical interpretation:
    - This term teaches the interface branch to explain the support/parallel-drift change as
      transport between neighboring shells rather than leaving the entire burden to the residual
      node decoder.
    """
    conservative_channel_index = stats_torch["transport_conservative_channel_index"]
    target_delta_physical = (
        target_dynamic_physical.index_select(dim=-1, index=conservative_channel_index)
        - last_dynamic_physical.index_select(dim=-1, index=conservative_channel_index)
    )
    scale = stats_torch["transport_delta_scale"]
    shell_mask = stats_torch["transport_shell_node_mask"].to(dtype=transport_delta_physical.dtype)
    view_shape = [1] * (transport_delta_physical.ndim - 2) + [int(shell_mask.shape[0]), 1]
    shell_weight = shell_mask.view(*view_shape)
    normalized_sq_error = torch.square((transport_delta_physical - target_delta_physical) / scale)
    return torch.sum(shell_weight * normalized_sq_error) / torch.clamp(torch.sum(shell_weight), min=1.0)


def residual_conservative_loss(
    residual_conservative_physical: torch.Tensor,
    stats_torch: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Penalize large conservative residual sources left after the transport update.

    Physical interpretation:
    - The residual branch should capture local source terms, not reproduce the whole conservative
      update that ought to come from shell-to-shell transport.
    - A small L2 penalty keeps that decomposition honest without forcing the residual to zero.
    """
    scale = stats_torch["transport_delta_scale"]
    shell_mask = stats_torch["transport_shell_node_mask"].to(dtype=residual_conservative_physical.dtype)
    view_shape = [1] * (residual_conservative_physical.ndim - 2) + [int(shell_mask.shape[0]), 1]
    shell_weight = shell_mask.view(*view_shape)
    normalized_sq_value = torch.square(residual_conservative_physical / scale)
    return torch.sum(shell_weight * normalized_sq_value) / torch.clamp(torch.sum(shell_weight), min=1.0)


def support_band_source_suppression_loss(
    source_conservative_physical: torch.Tensor,
    transport_delta_physical: torch.Tensor,
    stats_torch: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Penalize support-band nodes where source conservative magnitude exceeds transport.

    Physical interpretation:
    - Redistribution transport should carry the dominant conservative update in the diagnosed
      collapse band rather than letting the local source branch absorb it.
    - The penalty activates only on the excess source magnitude, so it encourages transport
      without forcing any fixed minimum outward fraction.
    """
    support_band_node_mask = stats_torch["support_band_node_mask"].to(dtype=source_conservative_physical.dtype)
    if not torch.any(support_band_node_mask > 0.0):
        return torch.zeros((), dtype=source_conservative_physical.dtype, device=source_conservative_physical.device)
    scale = stats_torch["transport_delta_scale"]
    source_magnitude = torch.sum(torch.abs(source_conservative_physical / scale), dim=-1)
    transport_magnitude = torch.sum(torch.abs(transport_delta_physical / scale), dim=-1)
    excess_source = torch.relu(source_magnitude - transport_magnitude)
    view_shape = [1] * (source_conservative_physical.ndim - 2) + [int(support_band_node_mask.shape[0])]
    node_weight = support_band_node_mask.view(*view_shape)
    return torch.sum(node_weight * excess_source) / torch.clamp(torch.sum(node_weight), min=1.0)


def support_outflow_saturation_loss(
    support_outflow_fraction_per_node: torch.Tensor,
    stats_torch: dict[str, torch.Tensor],
    max_support_outflow_fraction: float,
    saturation_start: float = 0.9,
) -> torch.Tensor:
    """Penalize redistribution runs that sit on the configured outflow cap across shells.

    Physical interpretation:
    - Strong polar-wind outflow is physically plausible, but a solution that pushes every shell to
      the hard cap indicates the optimizer is using the bound itself as the transport policy.
    - The penalty only activates near the cap and stays zero in the interior of the allowed
      outflow range.
    """
    shell_mask = stats_torch["transport_shell_node_mask"].to(dtype=support_outflow_fraction_per_node.dtype)
    if not torch.any(shell_mask > 0.0):
        return torch.zeros((), dtype=support_outflow_fraction_per_node.dtype, device=support_outflow_fraction_per_node.device)
    normalized_outflow = support_outflow_fraction_per_node / max(float(max_support_outflow_fraction), 1.0e-6)
    normalized_excess = torch.relu(normalized_outflow - float(saturation_start)) / max(1.0 - float(saturation_start), 1.0e-6)
    penalty = torch.square(normalized_excess)
    return torch.sum(shell_mask * penalty) / torch.clamp(torch.sum(shell_mask), min=1.0)


def closure_transport_one_step_loss(
    pred_dynamic: torch.Tensor,
    target_dynamic: torch.Tensor,
    closure_logvar: torch.Tensor,
    stats_torch: dict[str, torch.Tensor],
    occupied_shell_threshold: float,
    closure_loss_weight: float,
) -> dict[str, torch.Tensor]:
    """Compute one-step loss for closure-transport targets with learned uncertainty.

    Physical interpretation:
    - Support/occupancy and forcing channels remain supervised with normalized MSE because they are
      directly observed shell-state quantities.
    - Closure channels are supervised with heteroscedastic NLL so sparse shells can express lower
      confidence instead of forcing one fixed regression scale everywhere.
    """
    occupancy_shape_channel_index = stats_torch["occupancy_shape_channel_index"]
    forcing_channel_index = stats_torch["forcing_channel_index"]
    closure_channel_index = stats_torch["momentum_channel_index"]
    occupied_shell_mask = occupied_shell_mask_torch(
        dynamic_state=target_dynamic,
        stats_torch=stats_torch,
        occupied_shell_threshold=occupied_shell_threshold,
    )

    occupancy_shape_loss = weighted_normalized_dynamic_mse(
        pred_dynamic=pred_dynamic,
        target_dynamic=target_dynamic,
        stats_torch=stats_torch,
        channel_index=occupancy_shape_channel_index,
    )
    forcing_loss = weighted_normalized_dynamic_mse(
        pred_dynamic=pred_dynamic,
        target_dynamic=target_dynamic,
        stats_torch=stats_torch,
        channel_index=forcing_channel_index,
    )

    closure_pred = pred_dynamic.index_select(dim=-1, index=closure_channel_index)
    closure_target = target_dynamic.index_select(dim=-1, index=closure_channel_index)
    closure_std = stats_torch["dynamic_std"].index_select(0, closure_channel_index)
    normalized_sq_error = torch.square((closure_pred - closure_target) / closure_std)
    occupied_weight = occupied_shell_mask.to(dtype=closure_pred.dtype).unsqueeze(-1)
    closure_nll = 0.5 * (
        torch.exp(-closure_logvar) * normalized_sq_error
        + closure_logvar
    )
    closure_nll = torch.sum(occupied_weight * closure_nll) / torch.clamp(torch.sum(occupied_weight), min=1.0)
    closure_mse = torch.sum(occupied_weight * normalized_sq_error) / torch.clamp(torch.sum(occupied_weight), min=1.0)
    total_loss = occupancy_shape_loss + forcing_loss + (float(closure_loss_weight) * closure_nll)
    return {
        "total_loss": total_loss,
        "occupancy_shape_loss": occupancy_shape_loss,
        "forcing_loss": forcing_loss,
        "momentum_loss": closure_nll,
        "closure_nll": closure_nll,
        "closure_mse": closure_mse,
        "occupied_fraction": occupied_shell_mask.to(dtype=pred_dynamic.dtype).mean(),
    }


def conditional_one_step_loss(
    pred_dynamic: torch.Tensor,
    target_dynamic: torch.Tensor,
    stats_torch: dict[str, torch.Tensor],
    occupied_shell_threshold: float,
    momentum_loss_weight: float,
) -> dict[str, torch.Tensor]:
    """Compute the conditional one-step objective for occupancy/shape and momentum.

    Physical interpretation:
    - Occupancy/shape and forcing are always meaningful and should be fit on every shell.
    - Momentum-like channels are only supervised where the target shell occupancy is large enough
      to define a physically interpretable shell distribution.
    """
    occupancy_shape_channel_index = stats_torch["occupancy_shape_channel_index"]
    forcing_channel_index = stats_torch["forcing_channel_index"]
    momentum_channel_index = stats_torch["momentum_channel_index"]
    occupied_shell_mask = occupied_shell_mask_torch(
        dynamic_state=target_dynamic,
        stats_torch=stats_torch,
        occupied_shell_threshold=occupied_shell_threshold,
    )
    occupancy_shape_loss = weighted_normalized_dynamic_mse(
        pred_dynamic=pred_dynamic,
        target_dynamic=target_dynamic,
        stats_torch=stats_torch,
        channel_index=occupancy_shape_channel_index,
    )
    forcing_loss = weighted_normalized_dynamic_mse(
        pred_dynamic=pred_dynamic,
        target_dynamic=target_dynamic,
        stats_torch=stats_torch,
        channel_index=forcing_channel_index,
    )
    momentum_loss = weighted_normalized_dynamic_mse(
        pred_dynamic=pred_dynamic,
        target_dynamic=target_dynamic,
        stats_torch=stats_torch,
        channel_index=momentum_channel_index,
        element_weight=occupied_shell_mask,
    )
    total_loss = occupancy_shape_loss + forcing_loss + (float(momentum_loss_weight) * momentum_loss)
    zero_value = torch.zeros((), dtype=pred_dynamic.dtype, device=pred_dynamic.device)
    return {
        "total_loss": total_loss,
        "occupancy_shape_loss": occupancy_shape_loss,
        "forcing_loss": forcing_loss,
        "momentum_loss": momentum_loss,
        "closure_nll": zero_value,
        "closure_mse": zero_value,
        "occupied_fraction": occupied_shell_mask.to(dtype=pred_dynamic.dtype).mean(),
    }


def flux_transport_one_step_loss(
    pred_dynamic: torch.Tensor,
    target_dynamic: torch.Tensor,
    stats_torch: dict[str, torch.Tensor],
    conservative_state_loss_weight: float,
    non_conservative_state_loss_weight: float,
) -> dict[str, torch.Tensor]:
    """Split one-step supervision between the transported pair and the remaining shell channels.

    Physical interpretation:
    - The flux head is meant to carry shell support and parallel-momentum transport between
      neighboring altitude cells, so the conservative pair should not dominate the direct decoder
      regression term.
    - Non-conservative channels still behave like local shell descriptors, so their direct
      next-state loss remains the stronger anchor for the base decoder.
    """
    conservative_channel_index = stats_torch["transport_conservative_channel_index"]
    non_conservative_channel_index = stats_torch["non_transport_conservative_channel_index"]
    conservative_state_loss = weighted_normalized_dynamic_mse(
        pred_dynamic=pred_dynamic,
        target_dynamic=target_dynamic,
        stats_torch=stats_torch,
        channel_index=conservative_channel_index,
    )
    non_conservative_state_loss = weighted_normalized_dynamic_mse(
        pred_dynamic=pred_dynamic,
        target_dynamic=target_dynamic,
        stats_torch=stats_torch,
        channel_index=non_conservative_channel_index,
    )
    total_loss = (
        (float(conservative_state_loss_weight) * conservative_state_loss)
        + (float(non_conservative_state_loss_weight) * non_conservative_state_loss)
    )
    zero_value = torch.zeros((), dtype=pred_dynamic.dtype, device=pred_dynamic.device)
    return {
        "total_loss": total_loss,
        "occupancy_shape_loss": zero_value,
        "forcing_loss": zero_value,
        "momentum_loss": zero_value,
        "closure_nll": zero_value,
        "closure_mse": zero_value,
        "occupied_fraction": zero_value,
        "conservative_state_loss": conservative_state_loss,
        "non_conservative_state_loss": non_conservative_state_loss,
    }


def run_epoch(
    model: FieldlineGraphForecaster,
    loader: DataLoader,
    edge_index: torch.Tensor,
    optimizer: torch.optim.Optimizer | None,
    stats_torch: dict[str, torch.Tensor],
    predict_delta_state: bool,
    grad_clip_norm: float,
    occupied_shell_threshold: float,
    momentum_loss_weight: float,
    closure_loss_weight: float,
    flux_loss_weight: float,
    conservative_state_loss_weight: float,
    non_conservative_state_loss_weight: float,
    residual_conservative_weight: float,
    source_conservative_weight: float,
    support_band_source_suppression_weight: float,
    support_outflow_saturation_weight: float,
    boundary_source_loss_weight: float,
    support_band_loss_weight: float,
    secondary_support_band_loss_weight: float,
    diagnostic_mean_vpar_loss_weight: float,
    diagnostic_mean_vperp_loss_weight: float,
    diagnostic_std_vpar_loss_weight: float,
    diagnostic_mean_vpar_scale: float,
    diagnostic_mean_vperp_scale: float,
    diagnostic_std_vpar_scale: float,
    target_schema: str,
    device: torch.device,
    epoch_label: str,
) -> dict[str, float]:
    """Run one train or validation epoch and return aggregate metrics."""
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_raw_mse = 0.0
    total_occupancy_shape_loss = 0.0
    total_forcing_loss = 0.0
    total_momentum_loss = 0.0
    total_closure_nll = 0.0
    total_closure_mse = 0.0
    total_flux_consistency_loss = 0.0
    total_conservative_state_loss = 0.0
    total_non_conservative_state_loss = 0.0
    total_residual_conservative_loss = 0.0
    total_source_conservative_loss = 0.0
    total_boundary_source_loss = 0.0
    total_support_band_loss = 0.0
    total_secondary_support_band_loss = 0.0
    total_support_band_source_suppression_loss = 0.0
    total_support_outflow_saturation_loss = 0.0
    total_support_flux_capped_fraction = 0.0
    total_support_stay_fraction_mean = 0.0
    total_support_outflow_fraction_mean = 0.0
    total_support_outflow_cap_saturation_mean = 0.0
    total_occupied_fraction = 0.0
    total_diagnostic_mean_vpar_loss = 0.0
    total_diagnostic_mean_vpar_valid_fraction = 0.0
    total_diagnostic_mean_vperp_loss = 0.0
    total_diagnostic_mean_vperp_valid_fraction = 0.0
    total_diagnostic_std_vpar_loss = 0.0
    total_diagnostic_std_vpar_valid_fraction = 0.0
    total_examples = 0
    iterator = tqdm(loader, desc=epoch_label, dynamic_ncols=True, leave=False)

    for batch in iterator:
        input_features = batch["input_features"].to(device=device, dtype=torch.float32)
        global_context = batch["global_context"].to(device=device, dtype=torch.float32)
        boundary_context = batch["boundary_context"].to(device=device, dtype=torch.float32)
        target_dynamic = batch["target_dynamic"].to(device=device, dtype=torch.float32)
        target_dynamic_physical = batch["target_dynamic_physical"].to(device=device, dtype=torch.float32)
        last_dynamic = batch["last_dynamic"].to(device=device, dtype=torch.float32)
        last_dynamic_physical = batch["last_dynamic_physical"].to(device=device, dtype=torch.float32)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        pred_norm, aux_outputs = model.forward_with_aux(
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
        if model.cfg.target_schema == TARGET_SCHEMA_CLOSURE_TRANSPORT and model.cfg.uncertainty_aware_closure_head:
            if "closure_logvar" not in aux_outputs:
                raise ValueError("uncertainty_aware_closure_head requires closure_logvar auxiliary output.")
            loss_terms = closure_transport_one_step_loss(
                pred_dynamic=pred_dynamic,
                target_dynamic=target_dynamic,
                closure_logvar=aux_outputs["closure_logvar"],
                stats_torch=stats_torch,
                occupied_shell_threshold=occupied_shell_threshold,
                closure_loss_weight=closure_loss_weight,
            )
        elif model.cfg.conditional_momentum_head:
            loss_terms = conditional_one_step_loss(
                pred_dynamic=pred_dynamic,
                target_dynamic=target_dynamic,
                stats_torch=stats_torch,
                occupied_shell_threshold=occupied_shell_threshold,
                momentum_loss_weight=momentum_loss_weight,
            )
        elif model.cfg.flux_transport_head:
            loss_terms = flux_transport_one_step_loss(
                pred_dynamic=pred_dynamic,
                target_dynamic=target_dynamic,
                stats_torch=stats_torch,
                conservative_state_loss_weight=conservative_state_loss_weight,
                non_conservative_state_loss_weight=non_conservative_state_loss_weight,
            )
        else:
            vanilla_loss = weighted_normalized_dynamic_mse(
                pred_dynamic=pred_dynamic,
                target_dynamic=target_dynamic,
                stats_torch=stats_torch,
            )
            zero_value = torch.zeros((), dtype=vanilla_loss.dtype, device=vanilla_loss.device)
            loss_terms = {
                "total_loss": vanilla_loss,
                "occupancy_shape_loss": zero_value,
                "forcing_loss": zero_value,
                "momentum_loss": zero_value,
                "closure_nll": zero_value,
                "closure_mse": zero_value,
                "occupied_fraction": zero_value,
            }
        loss_terms["occupied_fraction"] = occupied_fraction_metric(
            target_dynamic=target_dynamic,
            stats_torch=stats_torch,
            occupied_shell_threshold=occupied_shell_threshold,
        )
        if "conservative_state_loss" not in loss_terms:
            # Non-flux objectives do not separate transported and non-transported channels, so the
            # conservative/non-conservative reporting slots stay zero for schema compatibility.
            zero_value = torch.zeros((), dtype=loss_terms["total_loss"].dtype, device=loss_terms["total_loss"].device)
            loss_terms["conservative_state_loss"] = zero_value
            loss_terms["non_conservative_state_loss"] = zero_value
        loss = loss_terms["total_loss"]
        flux_consistency = torch.zeros((), dtype=loss.dtype, device=loss.device)
        residual_transport = torch.zeros((), dtype=loss.dtype, device=loss.device)
        source_conservative = torch.zeros((), dtype=loss.dtype, device=loss.device)
        support_band_source_suppression = torch.zeros((), dtype=loss.dtype, device=loss.device)
        support_outflow_saturation = torch.zeros((), dtype=loss.dtype, device=loss.device)
        if "transport_delta_physical" in aux_outputs:
            flux_consistency = transport_flux_consistency_loss(
                target_dynamic_physical=target_dynamic_physical,
                last_dynamic_physical=last_dynamic_physical,
                transport_delta_physical=aux_outputs["transport_delta_physical"],
                stats_torch=stats_torch,
            )
            residual_transport = residual_conservative_loss(
                residual_conservative_physical=aux_outputs["residual_conservative_physical"],
                stats_torch=stats_torch,
            )
            if flux_loss_weight > 0.0:
                # This term scores whether the interface branch explains the conservative shell
                # increment as actual transport between neighboring shells.
                loss = loss + (float(flux_loss_weight) * flux_consistency)
            if residual_conservative_weight > 0.0:
                # The residual conservative penalty discourages the node-wise residual branch from
                # absorbing transport that should be carried by shell interfaces.
                loss = loss + (float(residual_conservative_weight) * residual_transport)
        if "source_conservative_physical" in aux_outputs:
            source_conservative = residual_conservative_loss(
                residual_conservative_physical=aux_outputs["source_conservative_physical"],
                stats_torch=stats_torch,
            )
            if model.cfg.redistribution_transport_head and "transport_delta_physical" in aux_outputs:
                support_band_source_suppression = support_band_source_suppression_loss(
                    source_conservative_physical=aux_outputs["source_conservative_physical"],
                    transport_delta_physical=aux_outputs["transport_delta_physical"],
                    stats_torch=stats_torch,
                )
            if source_conservative_weight > 0.0:
                # In operator-split mode, the source branch should explain local creation and loss
                # terms, not re-learn conservative shell transport already assigned to interfaces.
                loss = loss + (float(source_conservative_weight) * source_conservative)
            if model.cfg.redistribution_transport_head and support_band_source_suppression_weight > 0.0:
                # The redistribution branch should dominate the conservative update in the support
                # collapse band; otherwise the model can fall back to an all-stay transport path.
                loss = loss + (
                    float(support_band_source_suppression_weight) * support_band_source_suppression
                )
        if model.cfg.redistribution_transport_head and "support_outflow_fraction_per_node" in aux_outputs:
            support_outflow_saturation = support_outflow_saturation_loss(
                support_outflow_fraction_per_node=aux_outputs["support_outflow_fraction_per_node"].squeeze(-1),
                stats_torch=stats_torch,
                max_support_outflow_fraction=float(model.cfg.max_support_outflow_fraction),
            )
            if support_outflow_saturation_weight > 0.0:
                loss = loss + (float(support_outflow_saturation_weight) * support_outflow_saturation)
        focused_boundary_loss = boundary_source_loss(
            pred_dynamic=pred_dynamic,
            target_dynamic=target_dynamic,
            stats_torch=stats_torch,
        )
        if boundary_source_loss_weight > 0.0:
            # This term applies only where PWOM lower-boundary moments enter the field line, so it
            # teaches the new boundary branch a localized source role instead of perturbing every
            # shell equally.
            loss = loss + (float(boundary_source_loss_weight) * focused_boundary_loss)
        support_focus_loss = support_band_loss(
            pred_dynamic=pred_dynamic,
            target_dynamic=target_dynamic,
            stats_torch=stats_torch,
        )
        secondary_support_focus_loss = support_band_loss(
            pred_dynamic=pred_dynamic,
            target_dynamic=target_dynamic,
            stats_torch=stats_torch,
            mask_key="secondary_support_band_node_mask",
        )
        if support_band_loss_weight > 0.0:
            # This term specifically resists the shell-support collapse diagnosed in the
            # mid/high-altitude band without globally retuning every channel loss.
            loss = loss + (float(support_band_loss_weight) * support_focus_loss)
        if secondary_support_band_loss_weight > 0.0:
            # The weaker upper-band term targets the residual high-altitude support collapse that
            # remained after the primary mid-band fix.
            loss = loss + (
                float(secondary_support_band_loss_weight) * secondary_support_focus_loss
            )
        pred_dynamic_physical = inverse_transform_dynamic_torch(
            dynamic_model=pred_dynamic,
            stats_torch=stats_torch,
        )
        diagnostic_mean_vpar_loss, diagnostic_mean_vpar_valid_fraction = masked_mean_vpar_diagnostic_loss(
            pred_dynamic_physical=pred_dynamic_physical,
            target_dynamic_physical=target_dynamic_physical,
            occupied_shell_threshold=occupied_shell_threshold,
            diagnostic_mean_vpar_scale=diagnostic_mean_vpar_scale,
            target_schema=target_schema,
        )
        diagnostic_mean_vperp_loss, diagnostic_mean_vperp_valid_fraction = masked_mean_vperp_diagnostic_loss(
            pred_dynamic_physical=pred_dynamic_physical,
            target_dynamic_physical=target_dynamic_physical,
            occupied_shell_threshold=occupied_shell_threshold,
            diagnostic_mean_vperp_scale=diagnostic_mean_vperp_scale,
            target_schema=target_schema,
        )
        diagnostic_std_vpar_loss, diagnostic_std_vpar_valid_fraction = masked_std_vpar_diagnostic_loss(
            pred_dynamic_physical=pred_dynamic_physical,
            target_dynamic_physical=target_dynamic_physical,
            occupied_shell_threshold=occupied_shell_threshold,
            diagnostic_std_vpar_scale=diagnostic_std_vpar_scale,
            target_schema=target_schema,
        )
        if diagnostic_mean_vpar_loss_weight > 0.0:
            # The auxiliary diagnostic term only nudges the optimizer toward better occupied-shell
            # parallel drift recovery; it should not replace the main state-space objective.
            loss = loss + (float(diagnostic_mean_vpar_loss_weight) * diagnostic_mean_vpar_loss)
        if diagnostic_mean_vperp_loss_weight > 0.0:
            # This occupied-shell perpendicular-drift term is kept separate so redistribution runs
            # can correct mean_vperp drift without globally changing the base shell-state loss.
            loss = loss + (float(diagnostic_mean_vperp_loss_weight) * diagnostic_mean_vperp_loss)
        if diagnostic_std_vpar_loss_weight > 0.0:
            # A matching occupied-shell spread term constrains parallel-velocity width only after
            # the shell support and bulk dynamics are already fit by the base objective.
            loss = loss + (float(diagnostic_std_vpar_loss_weight) * diagnostic_std_vpar_loss)
        raw_mse = torch.mean(torch.square(pred_dynamic_physical - target_dynamic_physical))

        if is_train:
            loss.backward()
            # Gradient clipping is a purely numerical safeguard against unstable early training.
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

        batch_size = int(input_features.shape[0])
        total_loss += float(loss.detach().cpu()) * batch_size
        total_raw_mse += float(raw_mse.detach().cpu()) * batch_size
        total_occupancy_shape_loss += float(loss_terms["occupancy_shape_loss"].detach().cpu()) * batch_size
        total_forcing_loss += float(loss_terms["forcing_loss"].detach().cpu()) * batch_size
        total_momentum_loss += float(loss_terms["momentum_loss"].detach().cpu()) * batch_size
        total_closure_nll += float(loss_terms["closure_nll"].detach().cpu()) * batch_size
        total_closure_mse += float(loss_terms["closure_mse"].detach().cpu()) * batch_size
        total_flux_consistency_loss += float(flux_consistency.detach().cpu()) * batch_size
        total_conservative_state_loss += float(loss_terms["conservative_state_loss"].detach().cpu()) * batch_size
        total_non_conservative_state_loss += float(loss_terms["non_conservative_state_loss"].detach().cpu()) * batch_size
        total_residual_conservative_loss += float(residual_transport.detach().cpu()) * batch_size
        total_source_conservative_loss += float(source_conservative.detach().cpu()) * batch_size
        total_boundary_source_loss += float(focused_boundary_loss.detach().cpu()) * batch_size
        total_support_band_loss += float(support_focus_loss.detach().cpu()) * batch_size
        total_secondary_support_band_loss += float(secondary_support_focus_loss.detach().cpu()) * batch_size
        total_support_band_source_suppression_loss += float(support_band_source_suppression.detach().cpu()) * batch_size
        total_support_outflow_saturation_loss += float(support_outflow_saturation.detach().cpu()) * batch_size
        total_support_flux_capped_fraction += float(
            aux_outputs.get(
                "support_flux_capped_fraction",
                torch.zeros((), dtype=loss.dtype, device=loss.device),
            ).detach().cpu()
        ) * batch_size
        total_support_stay_fraction_mean += float(
            aux_outputs.get(
                "support_stay_fraction_mean",
                torch.zeros((), dtype=loss.dtype, device=loss.device),
            ).detach().cpu()
        ) * batch_size
        total_support_outflow_fraction_mean += float(
            aux_outputs.get(
                "support_outflow_fraction_mean",
                torch.zeros((), dtype=loss.dtype, device=loss.device),
            ).detach().cpu()
        ) * batch_size
        total_support_outflow_cap_saturation_mean += float(
            aux_outputs.get(
                "support_outflow_cap_saturation_mean",
                torch.zeros((), dtype=loss.dtype, device=loss.device),
            ).detach().cpu()
        ) * batch_size
        total_occupied_fraction += float(loss_terms["occupied_fraction"].detach().cpu()) * batch_size
        total_diagnostic_mean_vpar_loss += float(diagnostic_mean_vpar_loss.detach().cpu()) * batch_size
        total_diagnostic_mean_vpar_valid_fraction += float(diagnostic_mean_vpar_valid_fraction.detach().cpu()) * batch_size
        total_diagnostic_mean_vperp_loss += float(diagnostic_mean_vperp_loss.detach().cpu()) * batch_size
        total_diagnostic_mean_vperp_valid_fraction += float(diagnostic_mean_vperp_valid_fraction.detach().cpu()) * batch_size
        total_diagnostic_std_vpar_loss += float(diagnostic_std_vpar_loss.detach().cpu()) * batch_size
        total_diagnostic_std_vpar_valid_fraction += float(diagnostic_std_vpar_valid_fraction.detach().cpu()) * batch_size
        total_examples += batch_size
        iterator.set_postfix(
            loss=f"{float(loss.detach().cpu()):.5f}",
            raw=f"{float(raw_mse.detach().cpu()):.5f}",
            mom=f"{float(loss_terms['momentum_loss'].detach().cpu()):.5f}",
            flux=f"{float(flux_consistency.detach().cpu()):.5f}",
            cstate=f"{float(loss_terms['conservative_state_loss'].detach().cpu()):.5f}",
            ncstate=f"{float(loss_terms['non_conservative_state_loss'].detach().cpu()):.5f}",
            cres=f"{float(residual_transport.detach().cpu()):.5f}",
            csrc=f"{float(source_conservative.detach().cpu()):.5f}",
            bsrc=f"{float(focused_boundary_loss.detach().cpu()):.5f}",
            band=f"{float(support_focus_loss.detach().cpu()):.5f}",
            upband=f"{float(secondary_support_focus_loss.detach().cpu()):.5f}",
            tsrc=f"{float(support_band_source_suppression.detach().cpu()):.5f}",
            sat=f"{float(support_outflow_saturation.detach().cpu()):.5f}",
            scap=f"{float(aux_outputs.get('support_flux_capped_fraction', torch.zeros((), dtype=loss.dtype, device=loss.device)).detach().cpu()):.5f}",
            stay=f"{float(aux_outputs.get('support_stay_fraction_mean', torch.zeros((), dtype=loss.dtype, device=loss.device)).detach().cpu()):.5f}",
            out=f"{float(aux_outputs.get('support_outflow_fraction_mean', torch.zeros((), dtype=loss.dtype, device=loss.device)).detach().cpu()):.5f}",
            vpar=f"{float(diagnostic_mean_vpar_loss.detach().cpu()):.5f}",
            vperp=f"{float(diagnostic_mean_vperp_loss.detach().cpu()):.5f}",
            svpar=f"{float(diagnostic_std_vpar_loss.detach().cpu()):.5f}",
        )

    if total_examples == 0:
        raise ValueError("Encountered an empty DataLoader.")

    return {
        "loss": total_loss / total_examples,
        "raw_mse": total_raw_mse / total_examples,
        "occupancy_shape_loss": total_occupancy_shape_loss / total_examples,
        "forcing_loss": total_forcing_loss / total_examples,
        "momentum_loss": total_momentum_loss / total_examples,
        "closure_nll": total_closure_nll / total_examples,
        "closure_mse": total_closure_mse / total_examples,
        "flux_consistency_loss": total_flux_consistency_loss / total_examples,
        "conservative_state_loss": total_conservative_state_loss / total_examples,
        "non_conservative_state_loss": total_non_conservative_state_loss / total_examples,
        "residual_conservative_loss": total_residual_conservative_loss / total_examples,
        "source_conservative_loss": total_source_conservative_loss / total_examples,
        "boundary_source_loss": total_boundary_source_loss / total_examples,
        "support_band_loss": total_support_band_loss / total_examples,
        "secondary_support_band_loss": total_secondary_support_band_loss / total_examples,
        "support_band_source_suppression_loss": total_support_band_source_suppression_loss / total_examples,
        "support_outflow_saturation_loss": total_support_outflow_saturation_loss / total_examples,
        "support_flux_capped_fraction": total_support_flux_capped_fraction / total_examples,
        "support_stay_fraction_mean": total_support_stay_fraction_mean / total_examples,
        "support_outflow_fraction_mean": total_support_outflow_fraction_mean / total_examples,
        "support_outflow_cap_saturation_mean": total_support_outflow_cap_saturation_mean / total_examples,
        "occupied_fraction": total_occupied_fraction / total_examples,
        "diagnostic_mean_vpar_loss": total_diagnostic_mean_vpar_loss / total_examples,
        "diagnostic_mean_vpar_valid_fraction": total_diagnostic_mean_vpar_valid_fraction / total_examples,
        "diagnostic_mean_vperp_loss": total_diagnostic_mean_vperp_loss / total_examples,
        "diagnostic_mean_vperp_valid_fraction": total_diagnostic_mean_vperp_valid_fraction / total_examples,
        "diagnostic_std_vpar_loss": total_diagnostic_std_vpar_loss / total_examples,
        "diagnostic_std_vpar_valid_fraction": total_diagnostic_std_vpar_valid_fraction / total_examples,
    }


@dataclass
class PreparedTrainingData:
    """All static data artifacts prepared before model initialization."""

    train_series: Any
    valid_series: Any
    normalization: Any
    train_loader: DataLoader
    valid_loader: DataLoader
    velocity_species_metadata: dict[str, np.ndarray]
    support_band_node_mask: np.ndarray
    secondary_support_band_node_mask: np.ndarray
    boundary_source_node_mask: np.ndarray
    transport_interfaces: Any
    transport_cell_width_km: np.ndarray
    transport_shell_node_mask: np.ndarray
    diagnostic_mean_vpar_scale: float
    diagnostic_mean_vperp_scale: float
    diagnostic_std_vpar_scale: float
    transport_delta_scale: np.ndarray


@dataclass
class RuntimeContext:
    """Runtime objects needed for training and checkpointing."""

    model: FieldlineGraphForecaster
    optimizer: torch.optim.Optimizer
    edge_index: torch.Tensor
    stats_torch: dict[str, torch.Tensor]


@dataclass
class BestCheckpointTracker:
    """Best-epoch bookkeeping across checkpoint selection rules."""

    best_valid_loss: float = float("inf")
    best_epoch: int = 0
    best_flux_transport_score: float = float("inf")
    best_flux_epoch: int = 0
    best_boundary_source_score: float = float("inf")
    best_boundary_epoch: int = 0
    epochs_without_improvement: int = 0


def _validate_training_args(args: argparse.Namespace) -> None:
    """Validate CLI combinations and apply conservative default overrides."""
    if args.uncertainty_aware_closure_head and args.target_schema != TARGET_SCHEMA_CLOSURE_TRANSPORT:
        raise ValueError("uncertainty-aware closure head requires --target-schema closure_transport.")
    if args.flux_transport_head and args.target_schema != TARGET_SCHEMA_POPULATION_WEIGHTED:
        raise ValueError("flux_transport_head currently supports only --target-schema population_weighted.")
    if args.operator_split_head and not args.flux_transport_head:
        raise ValueError("operator_split_head requires --flux-transport-head.")
    if args.redistribution_transport_head and not args.flux_transport_head:
        raise ValueError("redistribution_transport_head requires --flux-transport-head.")
    if args.redistribution_transport_head and args.bounded_outflow_fraction_transport:
        raise ValueError("redistribution_transport_head and bounded_outflow_fraction_transport are mutually exclusive.")
    if args.redistribution_transport_head and args.simplex_support_transport:
        raise ValueError("redistribution_transport_head and simplex_support_transport are mutually exclusive.")
    if (
        args.redistribution_transport_head
        and args.diagnostic_mean_vpar_loss_weight <= 0.0
        and args.diagnostic_mean_vperp_loss_weight <= 0.0
        and args.diagnostic_std_vpar_loss_weight <= 0.0
    ):
        args.diagnostic_mean_vpar_loss_weight = 0.02
        args.diagnostic_mean_vperp_loss_weight = 0.02
        if args.diagnostic_loss_warmup_epochs <= 0:
            args.diagnostic_loss_warmup_epochs = 10


def _prepare_training_data(
    args: argparse.Namespace,
    effective_num_altitude_shells: int,
    shell_edges_km_override: np.ndarray | None,
) -> PreparedTrainingData:
    """Load field-line series, build windows, and create dataloaders/statistics."""
    print("[train] stage 1/4: loading field-line graph series...", flush=True)
    train_series = load_fieldline_graph_series(
        data_dir=args.data_dir,
        split="train",
        num_particles=args.num_particles,
        num_altitude_shells=effective_num_altitude_shells,
        include_global_node=True,
        include_species_coupling_edges=True,
        species_ids=None if args.species_ids is None else tuple(args.species_ids),
        shell_edges_km_override=shell_edges_km_override,
        verbose=args.verbose_data,
    )
    valid_series = load_fieldline_graph_series(
        data_dir=args.data_dir,
        split="valid",
        num_particles=args.num_particles,
        num_altitude_shells=effective_num_altitude_shells,
        include_global_node=True,
        include_species_coupling_edges=True,
        # Validation must share the same species stacks and shell edges as training so node
        # semantics remain identical across splits.
        species_ids=train_series.layout.species_ids,
        shell_edges_km_override=train_series.shell_edges_km,
        verbose=args.verbose_data,
    )
    velocity_species_metadata = build_velocity_species_weight_metadata(
        layout=train_series.layout,
        dynamic_node_features=train_series.dynamic_node_features,
        balance_power=args.velocity_species_balance_power,
        max_weight=args.velocity_species_balance_max_weight,
    )
    support_band_node_mask = build_support_band_node_mask(
        layout=train_series.layout,
        shell_edges_km=train_series.shell_edges_km,
        lower_km=args.support_band_lower_km,
        upper_km=args.support_band_upper_km,
    )
    secondary_support_band_node_mask = build_support_band_node_mask(
        layout=train_series.layout,
        shell_edges_km=train_series.shell_edges_km,
        lower_km=args.secondary_support_band_lower_km,
        upper_km=args.secondary_support_band_upper_km,
    )
    boundary_source_node_mask = build_boundary_source_node_mask(train_series.layout)
    transport_interfaces = build_transport_interfaces(
        layout=train_series.layout,
        shell_edges_km=train_series.shell_edges_km,
    )
    transport_cell_width_km = build_transport_cell_width_per_node(
        layout=train_series.layout,
        shell_edges_km=train_series.shell_edges_km,
    )
    transport_shell_node_mask = np.zeros(train_series.layout.num_nodes, dtype=np.float32)
    transport_shell_node_mask[: train_series.layout.num_shell_nodes] = 1.0

    print(
        "[train] shell layout "
        f"num_shells={effective_num_altitude_shells} "
        f"custom_edges={bool(shell_edges_km_override is not None)}",
        flush=True,
    )
    print(
        f"[train] target schema={args.target_schema} uncertainty_closure={bool(args.uncertainty_aware_closure_head)}",
        flush=True,
    )
    if shell_edges_km_override is not None:
        print(
            "[train] custom shell edges "
            f"source={os.path.abspath(args.shell_edges_json)} "
            f"edges_km={train_series.shell_edges_km.tolist()}",
            flush=True,
        )
    print(
        "[train] velocity species weights "
        f"power={args.velocity_species_balance_power:.3f} "
        f"max={args.velocity_species_balance_max_weight:.3f} "
        f"weights={velocity_species_metadata['velocity_species_weight_by_species'].tolist()}",
        flush=True,
    )
    print(
        "[train] support band "
        f"{args.support_band_lower_km:.1f}-{args.support_band_upper_km:.1f} km "
        f"active_nodes={int(np.sum(support_band_node_mask))}/{train_series.layout.num_nodes}",
        flush=True,
    )
    print(
        "[train] secondary support band "
        f"{args.secondary_support_band_lower_km:.1f}-{args.secondary_support_band_upper_km:.1f} km "
        f"active_nodes={int(np.sum(secondary_support_band_node_mask))}/{train_series.layout.num_nodes}",
        flush=True,
    )
    print(
        "[train] boundary source nodes "
        f"active_nodes={int(np.sum(boundary_source_node_mask))}/{train_series.layout.num_nodes}",
        flush=True,
    )

    print("[train] stage 2/4: building one-step graph windows...", flush=True)
    train_arrays = build_fieldline_window_arrays(
        series=train_series,
        window_length=args.window_length,
        target_schema=args.target_schema,
    )
    valid_arrays = build_fieldline_window_arrays(
        series=valid_series,
        window_length=args.window_length,
        target_schema=args.target_schema,
    )
    normalization = compute_window_normalization(arrays=train_arrays)
    normalization.save_json(os.path.join(args.output_dir, "normalization.json"))
    diagnostic_mean_vpar_scale = compute_mean_vpar_diagnostic_scale(
        target_dynamic_physical=train_arrays.target_dynamic_physical,
        occupied_shell_threshold=args.occupied_shell_threshold,
        target_schema=args.target_schema,
    )
    diagnostic_mean_vperp_scale = compute_mean_vperp_diagnostic_scale(
        target_dynamic_physical=train_arrays.target_dynamic_physical,
        occupied_shell_threshold=args.occupied_shell_threshold,
        target_schema=args.target_schema,
    )
    diagnostic_std_vpar_scale = compute_std_vpar_diagnostic_scale(
        target_dynamic_physical=train_arrays.target_dynamic_physical,
        occupied_shell_threshold=args.occupied_shell_threshold,
        target_schema=args.target_schema,
    )
    transport_delta_scale = compute_transport_delta_scale(
        target_dynamic_physical=train_arrays.target_dynamic_physical,
        last_dynamic_physical=train_arrays.last_dynamic_physical,
    )

    train_dataset = FieldlineWindowDataset(
        arrays=train_arrays,
        normalization=normalization,
        predict_delta_state=True,
    )
    valid_dataset = FieldlineWindowDataset(
        arrays=valid_arrays,
        normalization=normalization,
        predict_delta_state=True,
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
    return PreparedTrainingData(
        train_series=train_series,
        valid_series=valid_series,
        normalization=normalization,
        train_loader=train_loader,
        valid_loader=valid_loader,
        velocity_species_metadata=velocity_species_metadata,
        support_band_node_mask=support_band_node_mask,
        secondary_support_band_node_mask=secondary_support_band_node_mask,
        boundary_source_node_mask=boundary_source_node_mask,
        transport_interfaces=transport_interfaces,
        transport_cell_width_km=transport_cell_width_km,
        transport_shell_node_mask=transport_shell_node_mask,
        diagnostic_mean_vpar_scale=diagnostic_mean_vpar_scale,
        diagnostic_mean_vperp_scale=diagnostic_mean_vperp_scale,
        diagnostic_std_vpar_scale=diagnostic_std_vpar_scale,
        transport_delta_scale=transport_delta_scale,
    )


def _build_run_configs(
    args: argparse.Namespace,
    effective_num_altitude_shells: int,
    prepared: PreparedTrainingData,
) -> tuple[DataConfig, ModelConfig, TrainConfig]:
    """Build data/model/train config dataclasses from CLI + prepared metadata."""
    train_series = prepared.train_series
    data_cfg = DataConfig(
        data_dir=args.data_dir,
        num_particles=args.num_particles,
        num_altitude_shells=effective_num_altitude_shells,
        window_length=args.window_length,
        species_ids=train_series.layout.species_ids,
        target_schema=args.target_schema,
    )
    model_cfg = ModelConfig(
        node_input_dim=train_series.static_node_features.shape[-1] + (args.window_length * train_series.node_output_dim),
        global_context_dim=args.window_length * train_series.global_context_dim,
        node_output_dim=train_series.node_output_dim,
        hidden_dim=args.hidden_dim,
        message_passing_layers=args.message_passing_layers,
        mlp_layers=args.mlp_layers,
        predict_delta_state=True,
        # The conditional momentum head is a separate decoder mode, so split/factorized
        # modes are disabled automatically when it is enabled.
        split_decoder_heads=bool((args.split_decoder_heads or args.factorized_decoder_heads) and not args.conditional_momentum_head),
        factorized_decoder_heads=bool(args.factorized_decoder_heads and not args.conditional_momentum_head),
        conditional_momentum_head=bool(args.conditional_momentum_head),
        target_schema=args.target_schema,
        uncertainty_aware_closure_head=bool(args.uncertainty_aware_closure_head),
        occupancy_gate_center=float(args.occupancy_gate_center),
        occupancy_gate_width=float(args.occupancy_gate_width),
        flux_transport_head=bool(args.flux_transport_head),
        flux_hidden_dim=int(args.flux_hidden_dim),
        global_context_film=bool(args.global_context_film),
        boundary_injection_head=bool(args.boundary_injection_head),
        boundary_hidden_dim=int(args.boundary_hidden_dim),
        operator_split_head=bool(args.operator_split_head),
        redistribution_transport_head=bool(args.redistribution_transport_head),
        bounded_outflow_fraction_transport=bool(args.bounded_outflow_fraction_transport),
        simplex_support_transport=bool(args.simplex_support_transport),
        max_support_outflow_fraction=float(args.max_support_outflow_fraction),
    )
    train_cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip_norm=args.grad_clip_norm,
        num_workers=args.num_workers,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        velocity_species_balance_power=args.velocity_species_balance_power,
        velocity_species_balance_max_weight=args.velocity_species_balance_max_weight,
        occupied_shell_threshold=args.occupied_shell_threshold,
        momentum_loss_weight=args.momentum_loss_weight,
        closure_loss_weight=args.closure_loss_weight,
        flux_loss_weight=args.flux_loss_weight,
        conservative_state_loss_weight=args.conservative_state_loss_weight,
        non_conservative_state_loss_weight=args.non_conservative_state_loss_weight,
        residual_conservative_weight=args.residual_conservative_weight,
        source_conservative_weight=args.source_conservative_weight,
        support_band_source_suppression_weight=args.support_band_source_suppression_weight,
        support_outflow_saturation_weight=args.support_outflow_saturation_weight,
        boundary_source_loss_weight=args.boundary_source_loss_weight,
        support_band_loss_weight=args.support_band_loss_weight,
        support_band_lower_km=args.support_band_lower_km,
        support_band_upper_km=args.support_band_upper_km,
        secondary_support_band_loss_weight=args.secondary_support_band_loss_weight,
        secondary_support_band_lower_km=args.secondary_support_band_lower_km,
        secondary_support_band_upper_km=args.secondary_support_band_upper_km,
        diagnostic_mean_vpar_loss_weight=args.diagnostic_mean_vpar_loss_weight,
        diagnostic_mean_vperp_loss_weight=args.diagnostic_mean_vperp_loss_weight,
        diagnostic_std_vpar_loss_weight=args.diagnostic_std_vpar_loss_weight,
        diagnostic_loss_warmup_epochs=args.diagnostic_loss_warmup_epochs,
        diagnostic_loss_warmup_start_factor=args.diagnostic_loss_warmup_start_factor,
        seed=args.seed,
    )
    return data_cfg, model_cfg, train_cfg


def _write_training_config(
    args: argparse.Namespace,
    data_cfg: DataConfig,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    prepared: PreparedTrainingData,
) -> None:
    """Persist user-facing JSON config and key physical metadata for reproducibility."""
    train_series = prepared.train_series
    with open(os.path.join(args.output_dir, "config.json"), "w", encoding="utf-8") as handle:
        json.dump(
            {
                "data_cfg": asdict(data_cfg),
                "model_cfg": asdict(model_cfg),
                "train_cfg": asdict(train_cfg),
                "species_ids": list(train_series.layout.species_ids),
                "shell_edges_km": train_series.shell_edges_km.tolist(),
                "shell_edges_json": os.path.abspath(args.shell_edges_json) if args.shell_edges_json else "",
                "velocity_species_mean_population_fraction": (
                    prepared.velocity_species_metadata["species_mean_population_fraction"].tolist()
                ),
                "velocity_species_weight_by_species": (
                    prepared.velocity_species_metadata["velocity_species_weight_by_species"].tolist()
                ),
                "velocity_species_weight_per_node": (
                    prepared.velocity_species_metadata["velocity_species_weight_per_node"].tolist()
                ),
                "boundary_source_node_mask": prepared.boundary_source_node_mask.tolist(),
                "support_band_node_mask": prepared.support_band_node_mask.tolist(),
                "secondary_support_band_node_mask": prepared.secondary_support_band_node_mask.tolist(),
                "diagnostic_mean_vpar_scale": prepared.diagnostic_mean_vpar_scale,
                "diagnostic_mean_vperp_scale": prepared.diagnostic_mean_vperp_scale,
                "diagnostic_std_vpar_scale": prepared.diagnostic_std_vpar_scale,
                "transport_delta_scale": prepared.transport_delta_scale.tolist(),
                "transport_interface_altitude_km": prepared.transport_interfaces.interface_altitude_km.tolist(),
                "transport_interface_width_km": prepared.transport_interfaces.interface_width_km.tolist(),
                "transport_interface_features": prepared.transport_interfaces.interface_features.tolist(),
                "transport_shell_node_mask": prepared.transport_shell_node_mask.tolist(),
                "transport_cell_width_km": prepared.transport_cell_width_km.tolist(),
            },
            handle,
            indent=2,
        )


def _initialize_runtime(
    args: argparse.Namespace,
    device: torch.device,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    prepared: PreparedTrainingData,
) -> RuntimeContext:
    """Initialize model/optimizer and move normalization metadata to device tensors."""
    model = FieldlineGraphForecaster(cfg=model_cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
    edge_index = torch.as_tensor(prepared.train_series.edge_index, dtype=torch.long, device=device)
    stats_torch = _stats_to_torch(
        stats=prepared.normalization,
        device=device,
        num_nodes=prepared.train_series.layout.num_nodes,
        target_schema=args.target_schema,
        velocity_species_weight_per_node=prepared.velocity_species_metadata["velocity_species_weight_per_node"],
        boundary_source_node_mask=prepared.boundary_source_node_mask,
        support_band_node_mask=prepared.support_band_node_mask,
        secondary_support_band_node_mask=prepared.secondary_support_band_node_mask,
        transport_interfaces=prepared.transport_interfaces if args.flux_transport_head else None,
        transport_delta_scale=prepared.transport_delta_scale,
        transport_shell_node_mask=prepared.transport_shell_node_mask,
        transport_cell_width_km=prepared.transport_cell_width_km,
        transport_interface_width_km=(
            prepared.transport_interfaces.interface_width_km if args.flux_transport_head else None
        ),
    )
    return RuntimeContext(
        model=model,
        optimizer=optimizer,
        edge_index=edge_index,
        stats_torch=stats_torch,
    )


def _prefix_metrics(metrics: dict[str, float], prefix: str) -> dict[str, float]:
    """Attach a split prefix to all metric names."""
    return {f"{prefix}_{metric_name}": float(metric_value) for metric_name, metric_value in metrics.items()}


def _build_epoch_record(
    epoch: int,
    train_metrics: dict[str, float],
    valid_metrics: dict[str, float],
    train_cfg: TrainConfig,
    effective_mean_vpar_loss_weight: float,
    effective_mean_vperp_loss_weight: float,
    effective_std_vpar_loss_weight: float,
) -> dict[str, float]:
    """Flatten train/valid metrics and auxiliary weights into one persisted epoch record."""
    record = {
        "epoch": epoch,
        **_prefix_metrics(train_metrics, "train"),
        **_prefix_metrics(valid_metrics, "valid"),
        "effective_diagnostic_mean_vpar_loss_weight": float(effective_mean_vpar_loss_weight),
        "effective_diagnostic_mean_vperp_loss_weight": float(effective_mean_vperp_loss_weight),
        "effective_diagnostic_std_vpar_loss_weight": float(effective_std_vpar_loss_weight),
    }
    record["valid_flux_transport_score"] = float(
        record["valid_flux_consistency_loss"]
        + (train_cfg.residual_conservative_weight * record["valid_residual_conservative_loss"])
    )
    return record


def _print_epoch_summary(record: dict[str, float]) -> None:
    """Print a compact high-signal epoch summary line."""
    print(
        f"[train] epoch={int(record['epoch'])} train_loss={record['train_loss']:.6e} "
        f"valid_loss={record['valid_loss']:.6e} "
        f"valid_occ={record['valid_occupancy_shape_loss']:.6e} "
        f"valid_mom={record['valid_momentum_loss']:.6e} "
        f"valid_closure={record['valid_closure_nll']:.6e} "
        f"valid_flux={record['valid_flux_consistency_loss']:.6e} "
        f"valid_cstate={record['valid_conservative_state_loss']:.6e} "
        f"valid_ncstate={record['valid_non_conservative_state_loss']:.6e} "
        f"valid_cres={record['valid_residual_conservative_loss']:.6e} "
        f"valid_scsrc={record['valid_source_conservative_loss']:.6e} "
        f"valid_bsrc={record['valid_boundary_source_loss']:.6e} "
        f"valid_band={record['valid_support_band_loss']:.6e} "
        f"valid_upband={record['valid_secondary_support_band_loss']:.6e} "
        f"valid_tsrc={record['valid_support_band_source_suppression_loss']:.6e} "
        f"valid_sat={record['valid_support_outflow_saturation_loss']:.6e} "
        f"valid_scap={record['valid_support_flux_capped_fraction']:.6e} "
        f"valid_stay={record['valid_support_stay_fraction_mean']:.6e} "
        f"valid_out={record['valid_support_outflow_fraction_mean']:.6e} "
        f"valid_vpar={record['valid_diagnostic_mean_vpar_loss']:.6e} "
        f"valid_vperp={record['valid_diagnostic_mean_vperp_loss']:.6e} "
        f"valid_svpar={record['valid_diagnostic_std_vpar_loss']:.6e} "
        f"valid_raw_mse={record['valid_raw_mse']:.6e}",
        flush=True,
    )


def _build_checkpoint_payload(
    epoch: int,
    args: argparse.Namespace,
    data_cfg: DataConfig,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    runtime: RuntimeContext,
    prepared: PreparedTrainingData,
    history: list[dict[str, float]],
    tracker: BestCheckpointTracker,
) -> dict[str, Any]:
    """Assemble checkpoint payload shared by all saved checkpoint variants."""
    return {
        "epoch": epoch,
        "model_state": runtime.model.state_dict(),
        "optimizer_state": runtime.optimizer.state_dict(),
        "data_cfg": asdict(data_cfg),
        "model_cfg": asdict(model_cfg),
        "train_cfg": asdict(train_cfg),
        "species_ids": list(prepared.train_series.layout.species_ids),
        "shell_edges_km": prepared.train_series.shell_edges_km.astype(np.float32),
        "shell_edges_json": (os.path.abspath(args.shell_edges_json) if args.shell_edges_json else ""),
        "velocity_species_mean_population_fraction": (
            prepared.velocity_species_metadata["species_mean_population_fraction"].astype(np.float32)
        ),
        "velocity_species_weight_by_species": (
            prepared.velocity_species_metadata["velocity_species_weight_by_species"].astype(np.float32)
        ),
        "velocity_species_weight_per_node": (
            prepared.velocity_species_metadata["velocity_species_weight_per_node"].astype(np.float32)
        ),
        "boundary_source_node_mask": prepared.boundary_source_node_mask.astype(np.float32),
        "support_band_node_mask": prepared.support_band_node_mask.astype(np.float32),
        "secondary_support_band_node_mask": prepared.secondary_support_band_node_mask.astype(np.float32),
        "diagnostic_mean_vpar_scale": float(prepared.diagnostic_mean_vpar_scale),
        "diagnostic_mean_vperp_scale": float(prepared.diagnostic_mean_vperp_scale),
        "diagnostic_std_vpar_scale": float(prepared.diagnostic_std_vpar_scale),
        "transport_delta_scale": prepared.transport_delta_scale.astype(np.float32),
        "transport_interface_altitude_km": prepared.transport_interfaces.interface_altitude_km.astype(np.float32),
        "transport_interface_width_km": prepared.transport_interfaces.interface_width_km.astype(np.float32),
        "transport_interface_features": prepared.transport_interfaces.interface_features.astype(np.float32),
        "transport_shell_node_mask": prepared.transport_shell_node_mask.astype(np.float32),
        "transport_cell_width_km": prepared.transport_cell_width_km.astype(np.float32),
        "best_epoch": tracker.best_epoch,
        "best_valid_loss": tracker.best_valid_loss,
        "best_flux_epoch": tracker.best_flux_epoch,
        "best_flux_transport_score": tracker.best_flux_transport_score,
        "best_boundary_epoch": tracker.best_boundary_epoch,
        "best_boundary_source_score": tracker.best_boundary_source_score,
        "history": history,
    }


def _update_best_checkpoint_artifacts(
    args: argparse.Namespace,
    train_cfg: TrainConfig,
    record: dict[str, float],
    checkpoint: dict[str, Any],
    track_boundary_artifact: bool,
    tracker: BestCheckpointTracker,
) -> None:
    """Apply checkpoint-selection rules and save best-performing artifacts."""
    if record["valid_loss"] < (tracker.best_valid_loss - train_cfg.early_stopping_min_delta):
        tracker.best_valid_loss = record["valid_loss"]
        tracker.best_epoch = int(record["epoch"])
        tracker.epochs_without_improvement = 0
        checkpoint["best_epoch"] = tracker.best_epoch
        checkpoint["best_valid_loss"] = tracker.best_valid_loss
        torch.save(checkpoint, os.path.join(args.output_dir, "best.pt"))
    else:
        tracker.epochs_without_improvement += 1

    flux_transport_score = record["valid_flux_transport_score"]
    checkpoint["best_flux_transport_score"] = flux_transport_score
    best_flux_path = os.path.join(args.output_dir, "best_flux_transport.pt")
    if flux_transport_score < (tracker.best_flux_transport_score - train_cfg.early_stopping_min_delta):
        tracker.best_flux_transport_score = flux_transport_score
        tracker.best_flux_epoch = int(record["epoch"])
        checkpoint["best_flux_epoch"] = tracker.best_flux_epoch
        checkpoint["best_flux_transport_score"] = tracker.best_flux_transport_score
        torch.save(checkpoint, best_flux_path)

    if track_boundary_artifact:
        boundary_source_score = record["valid_boundary_source_loss"]
        checkpoint["best_boundary_source_score"] = boundary_source_score
        best_boundary_path = os.path.join(args.output_dir, "best_boundary_source.pt")
        if boundary_source_score < (tracker.best_boundary_source_score - train_cfg.early_stopping_min_delta):
            tracker.best_boundary_source_score = boundary_source_score
            tracker.best_boundary_epoch = int(record["epoch"])
            checkpoint["best_boundary_epoch"] = tracker.best_boundary_epoch
            checkpoint["best_boundary_source_score"] = tracker.best_boundary_source_score
            torch.save(checkpoint, best_boundary_path)


def _run_training_loop(
    args: argparse.Namespace,
    data_cfg: DataConfig,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    prepared: PreparedTrainingData,
    runtime: RuntimeContext,
    device: torch.device,
) -> tuple[list[dict[str, float]], BestCheckpointTracker, bool]:
    """Execute training/validation epochs and checkpoint selection."""
    history: list[dict[str, float]] = []
    tracker = BestCheckpointTracker()
    track_boundary_artifact = bool(model_cfg.boundary_injection_head or train_cfg.boundary_source_loss_weight > 0.0)

    for epoch in range(1, train_cfg.epochs + 1):
        # Primitive diagnostic supervision is warmed in gradually so shell-support and coarse
        # transport fit stabilize before occupied-shell velocity diagnostics start steering the run.
        effective_mean_vpar_loss_weight = compute_effective_diagnostic_weight(
            base_weight=train_cfg.diagnostic_mean_vpar_loss_weight,
            epoch_index=epoch,
            warmup_epochs=train_cfg.diagnostic_loss_warmup_epochs,
            warmup_start_factor=train_cfg.diagnostic_loss_warmup_start_factor,
        )
        effective_mean_vperp_loss_weight = compute_effective_diagnostic_weight(
            base_weight=train_cfg.diagnostic_mean_vperp_loss_weight,
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
        train_metrics = run_epoch(
            model=runtime.model,
            loader=prepared.train_loader,
            edge_index=runtime.edge_index,
            optimizer=runtime.optimizer,
            stats_torch=runtime.stats_torch,
            predict_delta_state=model_cfg.predict_delta_state,
            grad_clip_norm=train_cfg.grad_clip_norm,
            occupied_shell_threshold=train_cfg.occupied_shell_threshold,
            momentum_loss_weight=train_cfg.momentum_loss_weight,
            closure_loss_weight=train_cfg.closure_loss_weight,
            flux_loss_weight=train_cfg.flux_loss_weight,
            conservative_state_loss_weight=train_cfg.conservative_state_loss_weight,
            non_conservative_state_loss_weight=train_cfg.non_conservative_state_loss_weight,
            residual_conservative_weight=train_cfg.residual_conservative_weight,
            source_conservative_weight=train_cfg.source_conservative_weight,
            support_band_source_suppression_weight=train_cfg.support_band_source_suppression_weight,
            support_outflow_saturation_weight=train_cfg.support_outflow_saturation_weight,
            boundary_source_loss_weight=train_cfg.boundary_source_loss_weight,
            support_band_loss_weight=train_cfg.support_band_loss_weight,
            secondary_support_band_loss_weight=train_cfg.secondary_support_band_loss_weight,
            diagnostic_mean_vpar_loss_weight=effective_mean_vpar_loss_weight,
            diagnostic_mean_vperp_loss_weight=effective_mean_vperp_loss_weight,
            diagnostic_std_vpar_loss_weight=effective_std_vpar_loss_weight,
            diagnostic_mean_vpar_scale=prepared.diagnostic_mean_vpar_scale,
            diagnostic_mean_vperp_scale=prepared.diagnostic_mean_vperp_scale,
            diagnostic_std_vpar_scale=prepared.diagnostic_std_vpar_scale,
            target_schema=args.target_schema,
            device=device,
            epoch_label=f"train {epoch}/{train_cfg.epochs}",
        )
        with torch.no_grad():
            valid_metrics = run_epoch(
                model=runtime.model,
                loader=prepared.valid_loader,
                edge_index=runtime.edge_index,
                optimizer=None,
                stats_torch=runtime.stats_torch,
                predict_delta_state=model_cfg.predict_delta_state,
                grad_clip_norm=train_cfg.grad_clip_norm,
                occupied_shell_threshold=train_cfg.occupied_shell_threshold,
                momentum_loss_weight=train_cfg.momentum_loss_weight,
                closure_loss_weight=train_cfg.closure_loss_weight,
                flux_loss_weight=train_cfg.flux_loss_weight,
                conservative_state_loss_weight=train_cfg.conservative_state_loss_weight,
                non_conservative_state_loss_weight=train_cfg.non_conservative_state_loss_weight,
                residual_conservative_weight=train_cfg.residual_conservative_weight,
                source_conservative_weight=train_cfg.source_conservative_weight,
                support_band_source_suppression_weight=train_cfg.support_band_source_suppression_weight,
                support_outflow_saturation_weight=train_cfg.support_outflow_saturation_weight,
                boundary_source_loss_weight=train_cfg.boundary_source_loss_weight,
                support_band_loss_weight=train_cfg.support_band_loss_weight,
                secondary_support_band_loss_weight=train_cfg.secondary_support_band_loss_weight,
                diagnostic_mean_vpar_loss_weight=effective_mean_vpar_loss_weight,
                diagnostic_mean_vperp_loss_weight=effective_mean_vperp_loss_weight,
                diagnostic_std_vpar_loss_weight=effective_std_vpar_loss_weight,
                diagnostic_mean_vpar_scale=prepared.diagnostic_mean_vpar_scale,
                diagnostic_mean_vperp_scale=prepared.diagnostic_mean_vperp_scale,
                diagnostic_std_vpar_scale=prepared.diagnostic_std_vpar_scale,
                target_schema=args.target_schema,
                device=device,
                epoch_label=f"valid {epoch}/{train_cfg.epochs}",
            )

        record = _build_epoch_record(
            epoch=epoch,
            train_metrics=train_metrics,
            valid_metrics=valid_metrics,
            train_cfg=train_cfg,
            effective_mean_vpar_loss_weight=effective_mean_vpar_loss_weight,
            effective_mean_vperp_loss_weight=effective_mean_vperp_loss_weight,
            effective_std_vpar_loss_weight=effective_std_vpar_loss_weight,
        )
        history.append(record)
        _print_epoch_summary(record)

        checkpoint = _build_checkpoint_payload(
            epoch=epoch,
            args=args,
            data_cfg=data_cfg,
            model_cfg=model_cfg,
            train_cfg=train_cfg,
            runtime=runtime,
            prepared=prepared,
            history=history,
            tracker=tracker,
        )
        torch.save(checkpoint, os.path.join(args.output_dir, "last.pt"))
        _update_best_checkpoint_artifacts(
            args=args,
            train_cfg=train_cfg,
            record=record,
            checkpoint=checkpoint,
            track_boundary_artifact=track_boundary_artifact,
            tracker=tracker,
        )
        with open(os.path.join(args.output_dir, "history.json"), "w", encoding="utf-8") as handle:
            json.dump(history, handle, indent=2)

        if tracker.epochs_without_improvement >= train_cfg.early_stopping_patience:
            print(
                f"[train] early stopping at epoch={epoch} best_epoch={tracker.best_epoch} "
                f"best_valid_loss={tracker.best_valid_loss:.6e}",
                flush=True,
            )
            break

    return history, tracker, track_boundary_artifact


def main() -> None:
    """Load train/valid graph windows, fit the one-step forecaster, and save checkpoints."""
    args = parse_args()
    _validate_training_args(args)

    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)
    device = select_device(args.device)
    shell_edges_km_override = load_shell_edges_override(args.shell_edges_json)
    effective_num_altitude_shells = (
        int(shell_edges_km_override.size - 1)
        if shell_edges_km_override is not None
        else int(args.num_altitude_shells)
    )

    prepared = _prepare_training_data(
        args=args,
        effective_num_altitude_shells=effective_num_altitude_shells,
        shell_edges_km_override=shell_edges_km_override,
    )
    data_cfg, model_cfg, train_cfg = _build_run_configs(
        args=args,
        effective_num_altitude_shells=effective_num_altitude_shells,
        prepared=prepared,
    )
    _write_training_config(
        args=args,
        data_cfg=data_cfg,
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        prepared=prepared,
    )

    print("[train] stage 3/4: initializing model and optimizer...", flush=True)
    runtime = _initialize_runtime(
        args=args,
        device=device,
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        prepared=prepared,
    )

    print("[train] stage 4/4: training one-step field-line graph model...", flush=True)
    _, tracker, track_boundary_artifact = _run_training_loop(
        args=args,
        data_cfg=data_cfg,
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        prepared=prepared,
        runtime=runtime,
        device=device,
    )

    completion_message = (
        f"[train] training complete. best_epoch={tracker.best_epoch} best_valid_loss={tracker.best_valid_loss:.6e} "
        f"best_flux_epoch={tracker.best_flux_epoch} best_flux_transport_score={tracker.best_flux_transport_score:.6e}"
    )
    if track_boundary_artifact:
        completion_message += (
            f" best_boundary_epoch={tracker.best_boundary_epoch} "
            f"best_boundary_source_score={tracker.best_boundary_source_score:.6e}"
        )
    print(completion_message, flush=True)
    print(f"[train] artifacts saved in {os.path.abspath(args.output_dir)}", flush=True)


if __name__ == "__main__":
    main()
