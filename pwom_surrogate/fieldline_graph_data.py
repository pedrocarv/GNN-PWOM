"""Field-line graph data utilities for the PWOM surrogate.

Physical interpretation:
- A node represents one coarse `(species, altitude_shell)` distribution cell, not one persistent
  macro-particle.
- Static node features encode the shell identity and graph geometry.
- Dynamic node features encode the shell population and low-order moments at one simulation time.
- The optional global node carries field-line integrated context that can broadcast exogenous
  forcing to all shell nodes.
"""

from __future__ import annotations

import hashlib
import os
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tqdm import tqdm


# The binary header text starts with the per-row schema and may optionally append a snapshot-level
# metadata schema used for true PWOM driver inputs such as lower-boundary moments and line regime.
BINARY_EXPECTED_ROW_HEADER = "ParticleID,Species,Altitude,Vpar,Vperp,Weight,B,E,G,IsOpen"
SNAPSHOT_META_PREFIX = "|SnapshotMeta:"
# Backward-compatible alias for existing tests and utilities that still reference the old name.
BINARY_EXPECTED_HEADER = BINARY_EXPECTED_ROW_HEADER

# Packed binary rows store one macro-particle state and local forcing sample.
BINARY_ROW_DTYPE = np.dtype(
    [
        ("particle_id", "<i4"),
        ("species", "<i4"),
        ("altitude", "<f8"),
        ("vpar", "<f8"),
        ("vperp", "<f8"),
        ("weight", "<f8"),
        ("b", "<f8"),
        ("e", "<f8"),
        ("g", "<f8"),
        ("is_open", "u1"),
    ],
    align=False,
)

# Static fields identify where a node sits in the field-line graph.
STATIC_NODE_FEATURE_NAMES = (
    "shell_center_km",
    "shell_width_km",
    "species_index_norm",
    "is_global",
)

# Dynamic fields summarize the coarse shell distribution that evolves in time.
# Physical interpretation:
# - Velocity-like channels are stored as population-weighted shell moments rather than bare means.
# - This makes low-occupancy shells better conditioned because momentum-like quantities naturally
#   decay to zero as a species disappears from a shell.
DYNAMIC_NODE_FEATURE_NAMES = (
    "population_fraction",
    "log_total_weight",
    "open_fraction",
    "mean_altitude_km",
    "population_weighted_mean_vpar",
    "population_weighted_mean_vperp",
    "std_altitude_km",
    "population_weighted_std_vpar",
    "population_weighted_std_vperp",
    "mean_b",
    "mean_e",
    "mean_g",
)

# Explicit dynamic-feature indices are defined immediately after the feature names so later
# boundary/source channel groups can safely reference the physical shell-state layout.
DYNAMIC_FEATURE_INDEX = {
    feature_name: feature_index for feature_index, feature_name in enumerate(DYNAMIC_NODE_FEATURE_NAMES)
}

# Line-global PWOM controls are kept out of the node-state tensor so the model does not confuse
# them with shell-resolved plasma variables. Only the controls that actually vary across the
# present dataset stay in this context block; constant controls are omitted because they carry no
# predictive signal and only add nuisance dimensions to the global conditioning path.
GLOBAL_CONTEXT_FEATURE_NAMES = (
    "sm_lat",
    "is_cusp_or_aurora",
)

# Lower-boundary moments and WPI coefficients are external controls, not predicted shell state.
# They are passed through a separate boundary-context tensor so the model can inject them only at
# the physical lower boundary instead of treating them like evolving interior plasma variables.
BOUNDARY_CONTEXT_FEATURE_NAMES = (
    "boundary_density",
    "boundary_velocity",
    "boundary_temperature",
    "wpi_dperp",
    "wpi_dexp",
)
BOUNDARY_CONTEXT_FEATURE_INDEX = {
    feature_name: feature_index for feature_index, feature_name in enumerate(BOUNDARY_CONTEXT_FEATURE_NAMES)
}

# These channels are true lower-boundary inputs in PWOM, so they should only drive the lowest
# shell of each species rather than being broadcast through the whole field line.
BOUNDARY_SOURCE_FEATURE_NAMES = (
    "boundary_density",
    "boundary_velocity",
    "boundary_temperature",
    "wpi_dperp",
    "wpi_dexp",
)
BOUNDARY_SOURCE_CHANNEL_INDICES = tuple(
    BOUNDARY_CONTEXT_FEATURE_INDEX[feature_name] for feature_name in BOUNDARY_SOURCE_FEATURE_NAMES
)

# These are the shell-state channels most directly affected by lower-boundary replenishment and
# heating. They define the focused boundary loss used by the new boundary injection pathway.
BOUNDARY_RESPONSE_FEATURE_NAMES = (
    "population_fraction",
    "log_total_weight",
    "population_weighted_mean_vpar",
    "population_weighted_mean_vperp",
    "population_weighted_std_vpar",
    "population_weighted_std_vperp",
)
BOUNDARY_RESPONSE_CHANNEL_INDICES = tuple(
    DYNAMIC_FEATURE_INDEX[feature_name] for feature_name in BOUNDARY_RESPONSE_FEATURE_NAMES
)

# Target schemas let training compare different physical output parameterizations without changing the
# coarse field-line graph itself. The current default keeps the population-weighted moment targets
# used by the stable split-head branch. The closure schema replaces the primitive-like spread
# targets with transport-style energy proxies that stay well behaved as shell occupancy vanishes.
TARGET_SCHEMA_POPULATION_WEIGHTED = "population_weighted"
TARGET_SCHEMA_CLOSURE_TRANSPORT = "closure_transport"
TARGET_SCHEMA_NAMES = (
    TARGET_SCHEMA_POPULATION_WEIGHTED,
    TARGET_SCHEMA_CLOSURE_TRANSPORT,
)

# Closure transport channels occupy the same tensor slots as the current momentum-like channels so
# future training code can swap target parameterizations without changing graph widths or topology.
CLOSURE_DYNAMIC_NODE_FEATURE_NAMES = (
    "population_fraction",
    "log_total_weight",
    "open_fraction",
    "mean_altitude_km",
    "parallel_momentum_density_proxy",
    "perpendicular_momentum_proxy",
    "std_altitude_km",
    "parallel_energy_proxy",
    "perpendicular_energy_proxy",
    "mean_b",
    "mean_e",
    "mean_g",
)
CLOSURE_FEATURE_INDEX = {
    feature_name: feature_index for feature_index, feature_name in enumerate(CLOSURE_DYNAMIC_NODE_FEATURE_NAMES)
}

# User-facing diagnostics still need true shell means and standard deviations, even though the
# stored training targets are density-weighted moment proxies.
DYNAMIC_DERIVED_DIAGNOSTIC_FEATURE_NAMES = (
    "mean_vpar",
    "mean_vperp",
    "std_vpar",
    "std_vperp",
)
DYNAMIC_DERIVED_DIAGNOSTIC_SOURCE = {
    "mean_vpar": "population_weighted_mean_vpar",
    "mean_vperp": "population_weighted_mean_vperp",
    "std_vpar": "population_weighted_std_vpar",
    "std_vperp": "population_weighted_std_vperp",
}
DYNAMIC_DIAGNOSTIC_FEATURE_NAMES = DYNAMIC_NODE_FEATURE_NAMES + DYNAMIC_DERIVED_DIAGNOSTIC_FEATURE_NAMES

# Decoder/output groups keep physically related shell quantities on separate prediction paths.
# Physical interpretation:
# - density channels describe how much of a species is present in a shell
# - geometry channels describe where that mass sits along the field line
# - momentum channels describe velocity content conditioned on the shell population
# - topology/forcing channels describe exogenous environment rather than transported plasma state
DYNAMIC_OCCUPANCY_SHAPE_FEATURE_NAMES = (
    "population_fraction",
    "log_total_weight",
    "open_fraction",
    "mean_altitude_km",
    "std_altitude_km",
)
DYNAMIC_DENSITY_FEATURE_NAMES = (
    "population_fraction",
    "log_total_weight",
)
DYNAMIC_TOPOLOGY_FEATURE_NAMES = ("open_fraction",)
DYNAMIC_GEOMETRY_FEATURE_NAMES = (
    "mean_altitude_km",
    "std_altitude_km",
)
DYNAMIC_MOMENTUM_FEATURE_NAMES = (
    "population_weighted_mean_vpar",
    "population_weighted_mean_vperp",
    "population_weighted_std_vpar",
    "population_weighted_std_vperp",
)
DYNAMIC_FORCING_FEATURE_NAMES = (
    "mean_b",
    "mean_e",
    "mean_g",
)
DYNAMIC_CLOSURE_FEATURE_NAMES = (
    "parallel_momentum_density_proxy",
    "perpendicular_momentum_proxy",
    "parallel_energy_proxy",
    "perpendicular_energy_proxy",
)
DYNAMIC_OCCUPANCY_SHAPE_CHANNEL_INDICES = tuple(
    DYNAMIC_FEATURE_INDEX[feature_name] for feature_name in DYNAMIC_OCCUPANCY_SHAPE_FEATURE_NAMES
)
DYNAMIC_DENSITY_CHANNEL_INDICES = tuple(
    DYNAMIC_FEATURE_INDEX[feature_name] for feature_name in DYNAMIC_DENSITY_FEATURE_NAMES
)
DYNAMIC_TOPOLOGY_CHANNEL_INDICES = tuple(
    DYNAMIC_FEATURE_INDEX[feature_name] for feature_name in DYNAMIC_TOPOLOGY_FEATURE_NAMES
)
DYNAMIC_GEOMETRY_CHANNEL_INDICES = tuple(
    DYNAMIC_FEATURE_INDEX[feature_name] for feature_name in DYNAMIC_GEOMETRY_FEATURE_NAMES
)
DYNAMIC_MOMENTUM_CHANNEL_INDICES = tuple(
    DYNAMIC_FEATURE_INDEX[feature_name] for feature_name in DYNAMIC_MOMENTUM_FEATURE_NAMES
)
# Closure targets intentionally reuse the momentum channel slots so closure-aware models can
# share the same graph width and output layout as split-head decoding.
DYNAMIC_CLOSURE_CHANNEL_INDICES = DYNAMIC_MOMENTUM_CHANNEL_INDICES
DYNAMIC_FORCING_CHANNEL_INDICES = tuple(
    DYNAMIC_FEATURE_INDEX[feature_name] for feature_name in DYNAMIC_FORCING_FEATURE_NAMES
)

# These channels are treated as exogenous context during rollout.
# Physical interpretation:
# - `mean_b`, `mean_e`, and `mean_g` summarize the local field environment.
# - Boundary moments and WPI coefficients are imposed from the fluid/regime side of PWOM and should
#   therefore be supplied from the true future sequence during teacher-forced rollout evaluation.
# - The true line-wide PWOM controls live in `GLOBAL_CONTEXT_FEATURE_NAMES` instead of the node
#   state tensor, which keeps the rollout state focused on shell-resolved plasma evolution.
DYNAMIC_CONTEXT_FEATURE_NAMES = (
    "mean_b",
    "mean_e",
    "mean_g",
)

# These channels are the state variables the rollout model should evolve autoregressively.
# Physical interpretation:
# - They describe shell occupancy and low-order moment structure of the plasma distribution.
DYNAMIC_STATE_FEATURE_NAMES = tuple(
    feature_name
    for feature_name in DYNAMIC_NODE_FEATURE_NAMES
    if feature_name not in DYNAMIC_CONTEXT_FEATURE_NAMES
)

# Channel-index lookups keep rollout slicing explicit and auditable.
DYNAMIC_STATE_CHANNEL_INDICES = tuple(
    DYNAMIC_NODE_FEATURE_NAMES.index(feature_name) for feature_name in DYNAMIC_STATE_FEATURE_NAMES
)
DYNAMIC_CONTEXT_CHANNEL_INDICES = tuple(
    DYNAMIC_NODE_FEATURE_NAMES.index(feature_name) for feature_name in DYNAMIC_CONTEXT_FEATURE_NAMES
)

# These channels are constrained to be nonnegative because they are magnitudes, occupancies, or
# positive-definite shell statistics in the coarse graph representation.
DYNAMIC_NONNEGATIVE_FEATURE_NAMES = (
    "population_fraction",
    "log_total_weight",
    "mean_altitude_km",
    "population_weighted_mean_vperp",
    "std_altitude_km",
    "population_weighted_std_vpar",
    "population_weighted_std_vperp",
)
DYNAMIC_NONNEGATIVE_CHANNEL_INDICES = tuple(
    DYNAMIC_NODE_FEATURE_NAMES.index(feature_name) for feature_name in DYNAMIC_NONNEGATIVE_FEATURE_NAMES
)

# Probability-like channels must remain inside `[0, 1]` for physical interpretability.
DYNAMIC_UNIT_INTERVAL_FEATURE_NAMES = (
    "population_fraction",
    "open_fraction",
)
DYNAMIC_UNIT_INTERVAL_CHANNEL_INDICES = tuple(
    DYNAMIC_NODE_FEATURE_NAMES.index(feature_name) for feature_name in DYNAMIC_UNIT_INTERVAL_FEATURE_NAMES
)

# Velocity-moment channels are trained in a compressed `asinh` space because shell-level means can
# become noisy and very large in low-occupancy regions, especially for the signed parallel drift.
VELOCITY_ASINH_FEATURE_SCALES = {
    "population_weighted_mean_vpar": 1.0e5,
    "population_weighted_mean_vperp": 1.0e5,
    "population_weighted_std_vpar": 1.0e5,
    "population_weighted_std_vperp": 1.0e5,
}
SCHEMA_ASINH_FEATURE_SCALES = {
    TARGET_SCHEMA_POPULATION_WEIGHTED: VELOCITY_ASINH_FEATURE_SCALES,
    TARGET_SCHEMA_CLOSURE_TRANSPORT: {
        # Momentum-density proxies keep the signed/unsigned first-moment information but are
        # better conditioned than raw means when shell occupancy becomes small.
        "parallel_momentum_density_proxy": 1.0e5,
        "perpendicular_momentum_proxy": 1.0e5,
        # Energy proxies are quadratic in shell spread, so they need a larger compression scale to
        # prevent upper-tail shells from dominating optimization numerically.
        "parallel_energy_proxy": 1.0e10,
        "perpendicular_energy_proxy": 1.0e10,
    },
}
VELOCITY_ASINH_ABS_MAX = 6.0
VELOCITY_ASINH_CHANNEL_INDICES = tuple(
    DYNAMIC_FEATURE_INDEX[feature_name] for feature_name in VELOCITY_ASINH_FEATURE_SCALES
)
SIGNED_VELOCITY_ASINH_FEATURE_NAMES = ("population_weighted_mean_vpar",)
SIGNED_VELOCITY_ASINH_CHANNEL_INDICES = tuple(
    DYNAMIC_FEATURE_INDEX[feature_name] for feature_name in SIGNED_VELOCITY_ASINH_FEATURE_NAMES
)
NONNEGATIVE_VELOCITY_ASINH_FEATURE_NAMES = (
    "population_weighted_mean_vperp",
    "population_weighted_std_vpar",
    "population_weighted_std_vperp",
)
NONNEGATIVE_VELOCITY_ASINH_CHANNEL_INDICES = tuple(
    DYNAMIC_FEATURE_INDEX[feature_name] for feature_name in NONNEGATIVE_VELOCITY_ASINH_FEATURE_NAMES
)

# Occupancy weighting is only applied to velocity-like moments because those statistics are the
# noisiest when a shell contains very little particle weight.
OCCUPANCY_WEIGHTED_FEATURE_NAMES = (
    "population_weighted_mean_vpar",
    "population_weighted_mean_vperp",
    "population_weighted_std_vpar",
    "population_weighted_std_vperp",
)
OCCUPANCY_WEIGHTED_CHANNEL_INDICES = tuple(
    DYNAMIC_FEATURE_INDEX[feature_name] for feature_name in OCCUPANCY_WEIGHTED_FEATURE_NAMES
)
POPULATION_FRACTION_CHANNEL_INDEX = DYNAMIC_FEATURE_INDEX["population_fraction"]

# Conservative transport supervision is restricted to the pair of shell quantities whose
# finite-volume update has the clearest physical meaning in the current coarse graph:
# population support and population-weighted parallel drift.
TRANSPORT_CONSERVATIVE_FEATURE_NAMES = (
    "population_fraction",
    "population_weighted_mean_vpar",
)
TRANSPORT_CONSERVATIVE_CHANNEL_INDICES = tuple(
    DYNAMIC_FEATURE_INDEX[feature_name] for feature_name in TRANSPORT_CONSERVATIVE_FEATURE_NAMES
)

# Interface features keep transport supervision tied to shell geometry instead of treating
# neighboring shells as unstructured pairs in latent space.
TRANSPORT_INTERFACE_FEATURE_NAMES = (
    "interface_center_norm",
    "interface_width_norm",
    "species_index_norm",
)


def build_velocity_species_weight_metadata(
    layout: "FieldlineGraphLayout",
    dynamic_node_features: np.ndarray,
    balance_power: float,
    max_weight: float,
) -> dict[str, np.ndarray]:
    """Build optional per-species velocity-loss weights from train-split shell occupancy.

    Physical interpretation:
    - Velocity moments for low-population species are estimated from fewer macro-particles and are
      therefore harder to fit than the dominant species.
    - These weights compensate that imbalance only in the velocity-moment channels while keeping
      the global loss scale near unity.
    """
    shell_population_fraction = np.asarray(
        dynamic_node_features[:, : layout.num_shell_nodes, POPULATION_FRACTION_CHANNEL_INDEX],
        dtype=np.float32,
    )
    species_population_fraction = shell_population_fraction.reshape(
        shell_population_fraction.shape[0],
        layout.num_species,
        layout.num_altitude_shells,
    ).sum(axis=-1)
    species_mean_population_fraction = species_population_fraction.mean(axis=0).astype(np.float32, copy=False)

    if balance_power <= 0.0:
        # A zero power is the explicit "off" switch so existing loss behavior stays reproducible.
        species_weight_by_species = np.ones(layout.num_species, dtype=np.float32)
    else:
        effective_fraction = np.clip(species_mean_population_fraction, 1.0e-6, 1.0).astype(np.float32, copy=False)
        raw_species_weight = np.power(effective_fraction, -float(balance_power)).astype(np.float32, copy=False)
        # The cap prevents ultra-rare species from overwhelming the loss simply because they are rare.
        raw_species_weight = np.minimum(raw_species_weight, float(max_weight)).astype(np.float32, copy=False)
        species_weight_by_species = (
            raw_species_weight / max(float(np.mean(raw_species_weight)), 1.0e-6)
        ).astype(np.float32, copy=False)

    velocity_species_weight_per_node = np.ones(layout.num_nodes, dtype=np.float32)
    for species_offset, species_weight in enumerate(species_weight_by_species):
        shell_start = species_offset * layout.num_altitude_shells
        shell_stop = shell_start + layout.num_altitude_shells
        # Every shell node in one species stack shares the same balancing weight because the
        # imbalance we are correcting is species-level sampling support, not shell geometry.
        velocity_species_weight_per_node[shell_start:shell_stop] = float(species_weight)

    return {
        "species_mean_population_fraction": species_mean_population_fraction,
        "velocity_species_weight_by_species": species_weight_by_species,
        "velocity_species_weight_per_node": velocity_species_weight_per_node,
    }

# A cache version lets us invalidate stale graph tensors when the feature schema changes.
# Bump the cache version when feature semantics change so stale cached graphs are never reused.
FIELDLINE_GRAPH_CACHE_VERSION = 7

# Metadata cache avoids rescanning the large binary file for repeated runs.
_BINARY_META_CACHE: dict[str, list[dict[str, int | float | str]]] = {}


def validate_target_schema(target_schema: str) -> str:
    """Validate and normalize one target-schema name used by feature conversions.

    Physical interpretation:
    - The underlying shell graph stays the same, but different target schemas emphasize different
      physical parameterizations of shell transport.
    - Explicit validation keeps closure-aware training from silently mixing incompatible target
      spaces.
    """
    normalized_schema = str(target_schema)
    if normalized_schema not in TARGET_SCHEMA_NAMES:
        raise ValueError(
            f"Unsupported target_schema='{target_schema}'. "
            f"Expected one of {TARGET_SCHEMA_NAMES}."
        )
    return normalized_schema


def dynamic_feature_names_for_schema(target_schema: str = TARGET_SCHEMA_POPULATION_WEIGHTED) -> tuple[str, ...]:
    """Return the ordered feature names for one physical target schema.

    Physical interpretation:
    - The population-weighted schema supervises shell moments directly in a better-conditioned
      density-weighted space.
    - The closure schema supervises closure-like transport proxies that are closer to shell
      momentum and energy content than to primitive means.
    """
    target_schema = validate_target_schema(target_schema)
    if target_schema == TARGET_SCHEMA_CLOSURE_TRANSPORT:
        return CLOSURE_DYNAMIC_NODE_FEATURE_NAMES
    return DYNAMIC_NODE_FEATURE_NAMES


def build_target_dynamic_view(
    dynamic_physical: np.ndarray,
    target_schema: str = TARGET_SCHEMA_POPULATION_WEIGHTED,
    population_epsilon: float = 1.0e-6,
) -> np.ndarray:
    """Convert canonical shell-state tensors into one target schema for future training branches.

    Physical interpretation:
    - The canonical stored graph state is the current population-weighted moment representation.
    - Closure transport targets reuse the same tensor width but replace spread moments with energy
      proxies so the supervised quantities remain well behaved when shell support becomes small.
    """
    target_schema = validate_target_schema(target_schema)
    dynamic_physical = np.asarray(dynamic_physical, dtype=np.float32)
    if target_schema == TARGET_SCHEMA_POPULATION_WEIGHTED:
        return np.asarray(dynamic_physical, dtype=np.float32).copy()

    # Start from the canonical shell state so all shared context channels, including appended PWOM
    # driver metadata, stay identical across schemas unless explicitly replaced below.
    closure_dynamic = np.asarray(dynamic_physical, dtype=np.float32).copy()

    population_fraction = np.asarray(
        dynamic_physical[..., DYNAMIC_FEATURE_INDEX["population_fraction"]],
        dtype=np.float32,
    )
    safe_population = np.maximum(population_fraction, float(population_epsilon)).astype(np.float32, copy=False)

    weighted_std_vpar = np.asarray(
        dynamic_physical[..., DYNAMIC_FEATURE_INDEX["population_weighted_std_vpar"]],
        dtype=np.float32,
    )
    weighted_std_vperp = np.asarray(
        dynamic_physical[..., DYNAMIC_FEATURE_INDEX["population_weighted_std_vperp"]],
        dtype=np.float32,
    )
    # Energy-like proxies are reconstructed from the stored weighted spreads so the closure schema
    # can be introduced without changing the cached canonical graph series.
    closure_dynamic[..., 7] = np.where(
        population_fraction > float(population_epsilon),
        (weighted_std_vpar * weighted_std_vpar) / safe_population,
        0.0,
    ).astype(np.float32, copy=False)
    closure_dynamic[..., 8] = np.where(
        population_fraction > float(population_epsilon),
        (weighted_std_vperp * weighted_std_vperp) / safe_population,
        0.0,
    ).astype(np.float32, copy=False)
    return closure_dynamic


def transform_dynamic_features(
    dynamic_physical: np.ndarray,
    target_schema: str = TARGET_SCHEMA_POPULATION_WEIGHTED,
) -> np.ndarray:
    """Map raw shell-state features into the model space used for training.

    Physical interpretation:
    - Moment or closure channels are compressed with `asinh` so rare shells with extreme drifts or
      transport proxies do not dominate the regression loss.
    - Non-moment channels remain in their original physical units so occupancy, altitude, and
      forcing descriptors stay easy to interpret.
    """
    target_schema = validate_target_schema(target_schema)
    dynamic_model = np.asarray(dynamic_physical, dtype=np.float32).copy()
    feature_names = dynamic_feature_names_for_schema(target_schema)
    feature_index = {feature_name: feature_offset for feature_offset, feature_name in enumerate(feature_names)}
    for feature_name, scale in SCHEMA_ASINH_FEATURE_SCALES[target_schema].items():
        feature_index_value = feature_index[feature_name]
        dynamic_model[..., feature_index_value] = np.arcsinh(dynamic_model[..., feature_index_value] / scale).astype(
            np.float32,
            copy=False,
        )
    return dynamic_model


def inverse_transform_dynamic_features(
    dynamic_model: np.ndarray,
    target_schema: str = TARGET_SCHEMA_POPULATION_WEIGHTED,
) -> np.ndarray:
    """Map model-space shell features back to physical units for reporting.

    Physical interpretation:
    - Training occurs in a numerically compressed moment/closure space.
    - Diagnostics and plots should still be reported in the original PWOM shell units.
    """
    target_schema = validate_target_schema(target_schema)
    dynamic_physical = np.asarray(dynamic_model, dtype=np.float32).copy()
    feature_names = dynamic_feature_names_for_schema(target_schema)
    feature_index = {feature_name: feature_offset for feature_offset, feature_name in enumerate(feature_names)}
    for feature_name, scale in SCHEMA_ASINH_FEATURE_SCALES[target_schema].items():
        feature_index_value = feature_index[feature_name]
        dynamic_physical[..., feature_index_value] = (scale * np.sinh(dynamic_physical[..., feature_index_value])).astype(
            np.float32,
            copy=False,
        )
    return dynamic_physical


def available_dynamic_diagnostic_feature_names(
    target_schema: str = TARGET_SCHEMA_POPULATION_WEIGHTED,
) -> tuple[str, ...]:
    """Return stored and derived shell-state feature names available for diagnostics.

    Physical interpretation:
    - Stored feature names depend on the target schema used by the model branch under inspection.
    - Diagnostics can still ask for recovered shell means and standard deviations when those are
      physically meaningful to inspect, even if the trained target space uses closure proxies.
    """
    return dynamic_feature_names_for_schema(target_schema) + DYNAMIC_DERIVED_DIAGNOSTIC_FEATURE_NAMES


def occupied_shell_mask(
    dynamic_state: np.ndarray,
    population_threshold: float,
) -> np.ndarray:
    """Return a boolean mask for shells whose predicted/true occupancy is physically meaningful.

    Physical interpretation:
    - Velocity moments only have a clear interpretation when the shell contains enough particle
      weight to define a species-conditioned distribution.
    - This mask lets diagnostics and losses ignore empty or effectively empty shells rather than
      pretending their mean velocity should be zero.
    """
    dynamic_state = np.asarray(dynamic_state, dtype=np.float32)
    population_fraction = np.asarray(
        dynamic_state[..., POPULATION_FRACTION_CHANNEL_INDEX],
        dtype=np.float32,
    )
    return np.asarray(population_fraction >= float(population_threshold), dtype=bool)


def dynamic_diagnostic_valid_mask(
    dynamic_physical: np.ndarray,
    feature_name: str,
    target_schema: str = TARGET_SCHEMA_POPULATION_WEIGHTED,
    population_epsilon: float = 1.0e-6,
) -> np.ndarray:
    """Return where one diagnostic channel is physically defined.

    Physical interpretation:
    - Stored channels are always defined because they are the actual regression targets.
    - Recovered mean and spread diagnostics are only meaningful when shell occupancy is nonzero.
    """
    dynamic_physical = np.asarray(dynamic_physical, dtype=np.float32)
    target_schema = validate_target_schema(target_schema)
    if feature_name in dynamic_feature_names_for_schema(target_schema):
        return np.ones(dynamic_physical.shape[:-1], dtype=bool)
    if feature_name not in DYNAMIC_DERIVED_DIAGNOSTIC_SOURCE:
        raise ValueError(
            f"Unknown diagnostic feature='{feature_name}'. "
            f"Available features: {sorted(available_dynamic_diagnostic_feature_names(target_schema))}"
        )
    return occupied_shell_mask(
        dynamic_state=dynamic_physical,
        population_threshold=population_epsilon,
    )


def extract_dynamic_diagnostic_channel(
    dynamic_physical: np.ndarray,
    feature_name: str,
    target_schema: str = TARGET_SCHEMA_POPULATION_WEIGHTED,
    population_epsilon: float = 1.0e-6,
    undefined_fill_value: float = 0.0,
) -> np.ndarray:
    """Extract one stored or derived physical diagnostic channel from shell-state tensors.

    Physical interpretation:
    - When shell occupancy is non-negligible, stored moment or closure channels are converted back
      to the corresponding shell mean or spread by dividing by population fraction.
    - Empty shells return zero for those derived diagnostics because mean velocity is undefined
      when there is effectively no particle weight in the shell.
    """
    dynamic_physical = np.asarray(dynamic_physical, dtype=np.float32)
    target_schema = validate_target_schema(target_schema)
    feature_names = dynamic_feature_names_for_schema(target_schema)
    feature_index = {feature_name_value: feature_offset for feature_offset, feature_name_value in enumerate(feature_names)}
    if feature_name in feature_index:
        return np.asarray(dynamic_physical[..., feature_index[feature_name]], dtype=np.float32)

    if feature_name not in DYNAMIC_DERIVED_DIAGNOSTIC_SOURCE:
        raise ValueError(
            f"Unknown diagnostic feature='{feature_name}'. "
            f"Available features: {sorted(available_dynamic_diagnostic_feature_names(target_schema))}"
        )

    population_fraction = np.asarray(
        dynamic_physical[..., feature_index["population_fraction"]],
        dtype=np.float32,
    )
    safe_population = np.maximum(population_fraction, float(population_epsilon)).astype(np.float32, copy=False)
    valid_mask = dynamic_diagnostic_valid_mask(
        dynamic_physical=dynamic_physical,
        feature_name=feature_name,
        target_schema=target_schema,
        population_epsilon=population_epsilon,
    )

    if target_schema == TARGET_SCHEMA_CLOSURE_TRANSPORT:
        if feature_name == "mean_vpar":
            source_channel = np.asarray(
                dynamic_physical[..., feature_index["parallel_momentum_density_proxy"]],
                dtype=np.float32,
            )
            derived_channel = np.where(valid_mask, source_channel / safe_population, float(undefined_fill_value))
        elif feature_name == "mean_vperp":
            source_channel = np.asarray(
                dynamic_physical[..., feature_index["perpendicular_momentum_proxy"]],
                dtype=np.float32,
            )
            derived_channel = np.where(valid_mask, source_channel / safe_population, float(undefined_fill_value))
        elif feature_name == "std_vpar":
            source_channel = np.asarray(
                dynamic_physical[..., feature_index["parallel_energy_proxy"]],
                dtype=np.float32,
            )
            derived_channel = np.where(
                valid_mask,
                np.sqrt(np.maximum(source_channel / safe_population, 0.0)).astype(np.float32, copy=False),
                float(undefined_fill_value),
            )
        elif feature_name == "std_vperp":
            source_channel = np.asarray(
                dynamic_physical[..., feature_index["perpendicular_energy_proxy"]],
                dtype=np.float32,
            )
            derived_channel = np.where(
                valid_mask,
                np.sqrt(np.maximum(source_channel / safe_population, 0.0)).astype(np.float32, copy=False),
                float(undefined_fill_value),
            )
        else:
            raise ValueError(
                f"Unsupported derived diagnostic feature='{feature_name}' for target_schema='{target_schema}'."
            )
        return np.asarray(derived_channel, dtype=np.float32)

    source_feature_name = DYNAMIC_DERIVED_DIAGNOSTIC_SOURCE[feature_name]
    weighted_channel = np.asarray(
        dynamic_physical[..., feature_index[source_feature_name]],
        dtype=np.float32,
    )
    derived_channel = np.where(valid_mask, weighted_channel / safe_population, float(undefined_fill_value))
    return np.asarray(derived_channel, dtype=np.float32)


@dataclass(frozen=True)
class FieldlineWindowNormalization:
    """Normalization statistics for one-step field-line graph windows.

    Physical interpretation:
    - Input statistics normalize shell descriptors so geometry and state channels live on a
      numerically comparable scale during training.
    - Dynamic and delta statistics define the target space for absolute-state and residual
      next-step prediction respectively.
    """

    # Input statistics operate on concatenated static + history window node descriptors.
    input_mean: np.ndarray
    input_std: np.ndarray
    # Global-context statistics operate on line-wide PWOM controls kept outside the node state.
    context_mean: np.ndarray
    context_std: np.ndarray
    # Boundary-context statistics operate on the exogenous lower-boundary driver tensor.
    boundary_context_mean: np.ndarray
    boundary_context_std: np.ndarray
    # Dynamic statistics operate on the raw next-step shell state.
    dynamic_mean: np.ndarray
    dynamic_std: np.ndarray
    # Delta statistics operate on the shell increment between the last history frame and the target.
    delta_mean: np.ndarray
    delta_std: np.ndarray

    def normalize_input(self, value: np.ndarray) -> np.ndarray:
        """Normalize one input node-feature tensor or batch."""
        return ((value - self.input_mean) / self.input_std).astype(np.float32, copy=False)

    def normalize_context(self, value: np.ndarray) -> np.ndarray:
        """Normalize one global-context tensor or batch.

        Physical interpretation:
        - Line-global PWOM controls such as latitude and regime flags live on their own scale and
          should not be normalized with shell-state channels.
        """
        if self.context_mean.size == 0 or self.context_std.size == 0:
            # Old checkpoints predate the separate context path; returning a zero-width tensor keeps
            # them evaluable because the model will ignore the absent conditioning stream.
            return np.zeros(value.shape[:-1] + (0,), dtype=np.float32)
        return ((value - self.context_mean) / self.context_std).astype(np.float32, copy=False)

    def normalize_boundary_context(self, value: np.ndarray) -> np.ndarray:
        """Normalize one boundary-context tensor or batch.

        Physical interpretation:
        - Boundary drivers are external controls applied at the base of the field line.
        - They need their own normalization because they are neither shell-state channels nor
          line-global controls.
        """
        if self.boundary_context_mean.size == 0 or self.boundary_context_std.size == 0:
            return np.zeros(value.shape[:-1] + (0,), dtype=np.float32)
        return ((value - self.boundary_context_mean) / self.boundary_context_std).astype(np.float32, copy=False)

    def normalize_dynamic(self, value: np.ndarray) -> np.ndarray:
        """Normalize one raw dynamic target tensor."""
        return ((value - self.dynamic_mean) / self.dynamic_std).astype(np.float32, copy=False)

    def normalize_delta(self, value: np.ndarray) -> np.ndarray:
        """Normalize one raw dynamic-delta tensor."""
        return ((value - self.delta_mean) / self.delta_std).astype(np.float32, copy=False)

    def denormalize_dynamic(self, value: np.ndarray) -> np.ndarray:
        """Map normalized dynamic predictions back to raw shell-state units."""
        return (value * self.dynamic_std + self.dynamic_mean).astype(np.float32, copy=False)

    def denormalize_delta(self, value: np.ndarray) -> np.ndarray:
        """Map normalized residual predictions back to raw shell-increment units."""
        return (value * self.delta_std + self.delta_mean).astype(np.float32, copy=False)

    def save_json(self, path: str) -> None:
        """Persist normalization statistics for deterministic evaluation."""
        payload = {
            "input_mean": self.input_mean.tolist(),
            "input_std": self.input_std.tolist(),
            "context_mean": self.context_mean.tolist(),
            "context_std": self.context_std.tolist(),
            "boundary_context_mean": self.boundary_context_mean.tolist(),
            "boundary_context_std": self.boundary_context_std.tolist(),
            "dynamic_mean": self.dynamic_mean.tolist(),
            "dynamic_std": self.dynamic_std.tolist(),
            "delta_mean": self.delta_mean.tolist(),
            "delta_std": self.delta_std.tolist(),
        }
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    @staticmethod
    def load_json(path: str) -> "FieldlineWindowNormalization":
        """Load persisted normalization statistics."""
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        context_mean = np.asarray(payload.get("context_mean", []), dtype=np.float32)
        context_std = np.asarray(payload.get("context_std", []), dtype=np.float32)
        boundary_context_mean = np.asarray(payload.get("boundary_context_mean", []), dtype=np.float32)
        boundary_context_std = np.asarray(payload.get("boundary_context_std", []), dtype=np.float32)
        return FieldlineWindowNormalization(
            input_mean=np.asarray(payload["input_mean"], dtype=np.float32),
            input_std=np.asarray(payload["input_std"], dtype=np.float32),
            context_mean=context_mean,
            context_std=context_std,
            boundary_context_mean=boundary_context_mean,
            boundary_context_std=boundary_context_std,
            dynamic_mean=np.asarray(payload["dynamic_mean"], dtype=np.float32),
            dynamic_std=np.asarray(payload["dynamic_std"], dtype=np.float32),
            delta_mean=np.asarray(payload["delta_mean"], dtype=np.float32),
            delta_std=np.asarray(payload["delta_std"], dtype=np.float32),
        )


@dataclass(frozen=True)
class FieldlineGraphLayout:
    """Discrete layout of the graph state space."""

    # Species IDs define one node stack per ion population.
    species_ids: tuple[int, ...]
    # Altitude shells resolve transport along the field line.
    num_altitude_shells: int
    # The global node carries exogenous forcing and line-integrated context.
    include_global_node: bool = True

    @property
    def num_species(self) -> int:
        """Return the number of ion populations represented in the graph."""
        return len(self.species_ids)

    @property
    def num_shell_nodes(self) -> int:
        """Return the number of physical shell nodes excluding the optional global node."""
        return self.num_species * self.num_altitude_shells

    @property
    def num_nodes(self) -> int:
        """Return the total node count including the optional global context node."""
        return self.num_shell_nodes + (1 if self.include_global_node else 0)

    @property
    def global_node_index(self) -> int | None:
        """Return the global node index when the layout includes one."""
        return self.num_shell_nodes if self.include_global_node else None


@dataclass(frozen=True)
class FieldlineGraphSeries:
    """One time series of field-line graph states sampled from PWOM output."""

    # Graph topology is constant across all timesteps in the same coarse discretization.
    layout: FieldlineGraphLayout
    # Directed edge list used by the graph model.
    edge_index: np.ndarray
    # Static node descriptors define the shell geometry and node type.
    static_node_features: np.ndarray
    # Dynamic node descriptors define the time-varying shell population state.
    dynamic_node_features: np.ndarray
    # Line-global PWOM controls are kept separately from node state so they can condition the graph
    # through a dedicated encoder instead of being mistaken for shell-resolved plasma variables.
    global_context_features: np.ndarray
    # Lower-boundary controls are a separate exogenous tensor because PWOM imposes them externally
    # at the lowest shell instead of evolving them as part of the interior plasma state.
    boundary_context_features: np.ndarray
    # Timestamps let later trainers align rollout diagnostics with simulation time.
    model_times: np.ndarray
    # Shell boundaries fix the same coarse altitude discretization over the whole split.
    shell_edges_km: np.ndarray
    # Rank anchors define the deterministic per-snapshot sampling policy.
    rank_indices: np.ndarray
    # Split name is kept for diagnostics and cache provenance.
    split: str
    # Source path records which binary history produced the coarse graph sequence.
    source_path: str

    @property
    def static_feature_names(self) -> tuple[str, ...]:
        """Return the ordered names of the static node channels."""
        return STATIC_NODE_FEATURE_NAMES

    @property
    def dynamic_feature_names(self) -> tuple[str, ...]:
        """Return the ordered names of the dynamic node channels."""
        return DYNAMIC_NODE_FEATURE_NAMES

    @property
    def num_timesteps(self) -> int:
        """Return the number of graph snapshots in the series."""
        return int(self.dynamic_node_features.shape[0])

    @property
    def global_context_dim(self) -> int:
        """Return the width of the line-global PWOM control vector."""
        return int(self.global_context_features.shape[-1])

    @property
    def boundary_context_dim(self) -> int:
        """Return the width of the lower-boundary control vector per node."""
        return int(self.boundary_context_features.shape[-1])

    @property
    def lowest_shell_node_mask(self) -> np.ndarray:
        """Return the mask for the lowest shell of each species stack.

        Physical interpretation:
        - This marks the cells where lower-boundary replenishment should enter the surrogate.
        - Exposing it on the series object keeps the boundary pathway part of the public data
          contract instead of reconstructing it ad hoc inside each trainer.
        """
        return build_lowest_shell_node_mask(layout=self.layout)

    @property
    def boundary_source_node_mask(self) -> np.ndarray:
        """Return the active lower-boundary source mask for the current coarse graph."""
        return build_boundary_source_node_mask(layout=self.layout)

    @property
    def node_input_dim(self) -> int:
        """Return the full model input width per node."""
        return int(self.static_node_features.shape[-1] + self.dynamic_node_features.shape[-1])

    @property
    def node_output_dim(self) -> int:
        """Return the dynamic state width that a forecaster should predict."""
        return int(self.dynamic_node_features.shape[-1])

    def input_node_features_at(self, timestep_index: int) -> np.ndarray:
        """Return one `[num_nodes, input_dim]` node-feature tensor for model input."""
        return np.concatenate(
            (self.static_node_features, self.dynamic_node_features[timestep_index]),
            axis=-1,
        ).astype(np.float32, copy=False)

    def summary(self) -> dict[str, object]:
        """Return a compact summary for CLI previews and JSON diagnostics."""
        return {
            "split": self.split,
            "source_path": self.source_path,
            "num_timesteps": self.num_timesteps,
            "num_nodes": int(self.layout.num_nodes),
            "num_shell_nodes": int(self.layout.num_shell_nodes),
            "num_species": int(self.layout.num_species),
            "num_altitude_shells": int(self.layout.num_altitude_shells),
            "node_input_dim": self.node_input_dim,
            "node_output_dim": self.node_output_dim,
            "global_context_dim": self.global_context_dim,
            "boundary_context_dim": self.boundary_context_dim,
            "lowest_shell_nodes": int(np.sum(self.lowest_shell_node_mask)),
            "boundary_source_nodes": int(np.sum(self.boundary_source_node_mask)),
            "shell_altitude_min_km": float(self.shell_edges_km[0]),
            "shell_altitude_max_km": float(self.shell_edges_km[-1]),
            "species_ids": list(self.layout.species_ids),
        }


@dataclass(frozen=True)
class FieldlineTransportInterfaces:
    """Directed shell-interface metadata for conservative transport supervision.

    Physical interpretation:
    - Each interface sits between two neighboring shells of the same species.
    - The flux head predicts net per-step exchange across that interface, and the node update is
      the discrete divergence of those interface exchanges.
    """

    # Left and right node indices define the orientation of the shell interface.
    left_node_index: np.ndarray
    right_node_index: np.ndarray
    # Interface features encode where the exchange sits along the field line.
    interface_features: np.ndarray
    # Physical altitude of the shell interface in kilometers.
    interface_altitude_km: np.ndarray
    # Average shell width around the interface sets the local control-volume scale.
    interface_width_km: np.ndarray

    @property
    def num_interfaces(self) -> int:
        """Return the number of same-species shell interfaces in the layout."""
        return int(self.left_node_index.shape[0])


def build_transport_cell_width_per_node(
    layout: FieldlineGraphLayout,
    shell_edges_km: np.ndarray,
) -> np.ndarray:
    """Return the control-volume width assigned to each shell node.

    Physical interpretation:
    - A shell-average conservative state changes by the interface-flux divergence divided by the
      shell width.
    - The global node is not part of that 1-D finite-volume stack, so its transport width is zero.
    """
    shell_edges_km = np.asarray(shell_edges_km, dtype=np.float32)
    if shell_edges_km.ndim != 1 or shell_edges_km.size != layout.num_altitude_shells + 1:
        raise ValueError(
            "shell_edges_km must be a 1D edge array with num_altitude_shells + 1 entries."
        )
    shell_widths_km = np.diff(shell_edges_km).astype(np.float32)
    node_widths = np.zeros(layout.num_nodes, dtype=np.float32)
    for species_offset in range(layout.num_species):
        start = species_offset * layout.num_altitude_shells
        stop = start + layout.num_altitude_shells
        node_widths[start:stop] = shell_widths_km
    if layout.global_node_index is not None:
        node_widths[layout.global_node_index] = 0.0
    return node_widths


@dataclass(frozen=True)
class FieldlineWindowArrays:
    """Precomputed one-step supervision arrays built from one graph time series."""

    # Inputs concatenate static node features with a short history of dynamic shell states.
    input_features: np.ndarray
    # Separate history of line-global PWOM controls used by the global conditioning encoder.
    input_global_context: np.ndarray
    # Lowest-shell PWOM boundary controls are kept separate from shell state and reused by the
    # dedicated boundary-injection branch.
    input_boundary_context: np.ndarray
    # The last history frame is needed to reconstruct the next state from a predicted delta.
    last_dynamic: np.ndarray
    # Model-space next state at the next time level.
    target_dynamic: np.ndarray
    # Model-space target increment from the last history frame to the next state.
    target_delta: np.ndarray
    # Physical-space last history frame is kept for metrics in original PWOM units.
    last_dynamic_physical: np.ndarray
    # Physical-space next state is kept for diagnostics and evaluation.
    target_dynamic_physical: np.ndarray
    # Time metadata keeps training diagnostics physically anchored in simulation time.
    input_times: np.ndarray
    target_times: np.ndarray

    @property
    def num_samples(self) -> int:
        """Return the number of one-step supervision windows."""
        return int(self.input_features.shape[0])


class FieldlineWindowDataset:
    """Dataset of normalized one-step graph windows.

    Physical interpretation:
    - Each item uses a short history of shell states to predict the next field-line shell state.
    - The target is either the next state itself or the state increment, depending on whether the
      forecaster is trained in absolute or residual mode.
    """

    def __init__(
        self,
        arrays: FieldlineWindowArrays,
        normalization: FieldlineWindowNormalization,
        predict_delta_state: bool,
    ) -> None:
        self.arrays = arrays
        self.normalization = normalization
        self.predict_delta_state = predict_delta_state

    def __len__(self) -> int:
        """Return the number of graph windows."""
        return self.arrays.num_samples

    def __getitem__(self, index: int) -> dict[str, np.ndarray | np.float32]:
        """Return one normalized supervision example."""
        input_features = self.normalization.normalize_input(self.arrays.input_features[index])
        if self.predict_delta_state:
            target_norm = self.normalization.normalize_delta(self.arrays.target_delta[index])
        else:
            target_norm = self.normalization.normalize_dynamic(self.arrays.target_dynamic[index])

        # The trainer keeps raw targets and the last dynamic state so it can report metrics in
        # physical shell units instead of only normalized loss.
        return {
            "input_features": input_features,
            "global_context": self.normalization.normalize_context(
                self.arrays.input_global_context[index]
            ),
            "boundary_context": self.normalization.normalize_boundary_context(
                self.arrays.input_boundary_context[index]
            ),
            "target_norm": target_norm,
            "target_dynamic": self.arrays.target_dynamic[index].astype(np.float32, copy=False),
            "last_dynamic": self.arrays.last_dynamic[index].astype(np.float32, copy=False),
            "target_dynamic_physical": self.arrays.target_dynamic_physical[index].astype(np.float32, copy=False),
            "last_dynamic_physical": self.arrays.last_dynamic_physical[index].astype(np.float32, copy=False),
            "input_time": np.float32(self.arrays.input_times[index]),
            "target_time": np.float32(self.arrays.target_times[index]),
        }


@dataclass(frozen=True)
class FieldlineRolloutArrays:
    """Precomputed rollout supervision arrays built from one graph time series.

    Physical interpretation:
    - Each sample contains one short history window plus a short future sequence.
    - The model is asked to reproduce the future shell-distribution evolution autoregressively,
      which is the first direct test of temporal stability in the coarse field-line state.
    """

    # Initial history window used to seed the autoregressive rollout.
    history_dynamic: np.ndarray
    # Initial history of line-global PWOM controls used by the separate conditioning path.
    history_global_context: np.ndarray
    # Initial lower-boundary control history. The rollout branch uses the last observed frame to
    # seed the boundary source head before future exogenous controls are supplied.
    history_boundary_context: np.ndarray
    # Future shell states over the rollout horizon in the transformed model space.
    target_dynamic_seq: np.ndarray
    # Future shell increments relative to the immediately preceding shell state in model space.
    target_delta_seq: np.ndarray
    # Physical-space future shell states are kept for metrics and plots.
    target_dynamic_seq_physical: np.ndarray
    # True future line-global PWOM controls are supplied during rollout because they are exogenous.
    target_global_context_seq: np.ndarray
    # True future lower-boundary controls are supplied during rollout because they are externally
    # imposed inputs, not predicted shell-state channels.
    target_boundary_context_seq: np.ndarray
    # Time metadata keeps rollout diagnostics tied to physical simulation time.
    input_time: np.ndarray
    target_times: np.ndarray

    @property
    def num_samples(self) -> int:
        """Return the number of rollout windows."""
        return int(self.history_dynamic.shape[0])


class FieldlineRolloutDataset:
    """Dataset of rollout supervision windows for autoregressive training.

    Physical interpretation:
    - A rollout sample starts from a short observed shell-state history and asks the model to
      evolve the field-line shell state for several future timesteps.
    - Normalization is still based on one-step train statistics so rollout loss stays in the same
      state space as the one-step pretraining checkpoint.
    """

    def __init__(
        self,
        static_node_features: np.ndarray,
        arrays: FieldlineRolloutArrays,
        normalization: FieldlineWindowNormalization,
        predict_delta_state: bool,
    ) -> None:
        self.static_node_features = static_node_features.astype(np.float32, copy=False)
        self.arrays = arrays
        self.normalization = normalization
        self.predict_delta_state = predict_delta_state

    def __len__(self) -> int:
        """Return the number of rollout windows."""
        return self.arrays.num_samples

    def __getitem__(self, index: int) -> dict[str, np.ndarray | np.float32]:
        """Return one rollout supervision example with normalized history inputs."""
        history_dynamic = self.arrays.history_dynamic[index].astype(np.float32, copy=False)
        history_flat = np.transpose(history_dynamic, (1, 0, 2)).reshape(
            history_dynamic.shape[1],
            history_dynamic.shape[0] * history_dynamic.shape[2],
        )
        input_features = np.concatenate((self.static_node_features, history_flat), axis=-1)
        input_features = self.normalization.normalize_input(input_features)

        if self.predict_delta_state:
            target_norm_seq = self.normalization.normalize_delta(self.arrays.target_delta_seq[index])
        else:
            target_norm_seq = self.normalization.normalize_dynamic(self.arrays.target_dynamic_seq[index])

        return {
            "input_features": input_features,
            "history_dynamic": history_dynamic,
            "history_global_context": self.arrays.history_global_context[index].astype(np.float32, copy=False),
            "history_boundary_context": self.arrays.history_boundary_context[index].astype(np.float32, copy=False),
            "target_dynamic_seq": self.arrays.target_dynamic_seq[index].astype(np.float32, copy=False),
            "target_dynamic_seq_physical": self.arrays.target_dynamic_seq_physical[index].astype(np.float32, copy=False),
            "target_global_context_seq": self.arrays.target_global_context_seq[index].astype(np.float32, copy=False),
            "target_boundary_context_seq": self.arrays.target_boundary_context_seq[index].astype(np.float32, copy=False),
            "target_norm_seq": target_norm_seq.astype(np.float32, copy=False),
            "input_time": np.float32(self.arrays.input_time[index]),
            "target_times": self.arrays.target_times[index].astype(np.float32, copy=False),
        }


def node_index(layout: FieldlineGraphLayout, species_offset: int, shell_index: int) -> int:
    """Map `(species_offset, shell_index)` to one flat node index.

    Physical interpretation:
    - Flat indexing lets the graph model treat shell states as one contiguous node set.
    - The layout still preserves the tensor product structure of species and altitude.
    """
    if species_offset < 0 or species_offset >= layout.num_species:
        raise IndexError(f"species_offset={species_offset} is outside [0, {layout.num_species})")
    if shell_index < 0 or shell_index >= layout.num_altitude_shells:
        raise IndexError(f"shell_index={shell_index} is outside [0, {layout.num_altitude_shells})")
    return species_offset * layout.num_altitude_shells + shell_index


def build_lowest_shell_node_mask(layout: FieldlineGraphLayout) -> np.ndarray:
    """Mark the lowest shell of every species stack.

    Physical interpretation:
    - PWOM applies boundary moments at the lower boundary, not uniformly across the field line.
    - The lowest shell of each species is therefore the natural support for a boundary injection
      head and for any lower-boundary-focused diagnostics or losses.
    """
    node_mask = np.zeros(layout.num_nodes, dtype=np.float32)
    for species_offset in range(layout.num_species):
        node_mask[node_index(layout=layout, species_offset=species_offset, shell_index=0)] = 1.0
    if layout.global_node_index is not None:
        node_mask[layout.global_node_index] = 0.0
    return node_mask


def build_species_node_mask(layout: FieldlineGraphLayout, species_id: int) -> np.ndarray:
    """Mark every shell node belonging to one species stack.

    Physical interpretation:
    - Species-focused diagnostics should only compare shells that belong to the same ion population.
    - The optional global node is never part of a species-local mask because it does not represent a
      single species shell.
    """
    node_mask = np.zeros(layout.num_nodes, dtype=np.float32)
    normalized_species_id = int(species_id)
    for species_offset, layout_species_id in enumerate(layout.species_ids):
        if int(layout_species_id) != normalized_species_id:
            continue
        start_index = node_index(layout=layout, species_offset=species_offset, shell_index=0)
        stop_index = start_index + layout.num_altitude_shells
        node_mask[start_index:stop_index] = 1.0
    if layout.global_node_index is not None:
        node_mask[layout.global_node_index] = 0.0
    return node_mask


def build_boundary_source_node_mask(layout: FieldlineGraphLayout) -> np.ndarray:
    """Alias the lowest-shell mask as the active lower-boundary source region.

    Physical interpretation:
    - In this design, lower-boundary source terms are only allowed to act on the
      lowest shell of each species.
    - Keeping this helper separate makes it easy to widen the active boundary region later
      without changing every trainer/evaluator call site.
    """
    return build_lowest_shell_node_mask(layout=layout)


def build_transport_interfaces(
    layout: FieldlineGraphLayout,
    shell_edges_km: np.ndarray,
) -> FieldlineTransportInterfaces:
    """Build same-species shell-interface metadata for conservative transport supervision.

    Physical interpretation:
    - One interface couples each adjacent shell pair within one species stack.
    - The interface center and width tell the flux head where the exchange occurs, which matters
      because the rollout failure is localized in a specific altitude band rather than uniformly
      along the field line.
    """
    shell_edges_km = np.asarray(shell_edges_km, dtype=np.float32)
    if shell_edges_km.ndim != 1 or shell_edges_km.size != layout.num_altitude_shells + 1:
        raise ValueError(
            "shell_edges_km must be a 1D edge array with num_altitude_shells + 1 entries."
        )

    total_altitude_span = float(max(shell_edges_km[-1] - shell_edges_km[0], 1.0))
    left_node_index: list[int] = []
    right_node_index: list[int] = []
    interface_features: list[tuple[float, float, float]] = []
    interface_altitude_km: list[float] = []
    interface_width_km: list[float] = []

    for species_offset in range(layout.num_species):
        species_index_norm = float(species_offset) / max(layout.num_species - 1, 1)
        for shell_index in range(layout.num_altitude_shells - 1):
            left_shell_width = float(shell_edges_km[shell_index + 1] - shell_edges_km[shell_index])
            right_shell_width = float(shell_edges_km[shell_index + 2] - shell_edges_km[shell_index + 1])
            interface_altitude = float(shell_edges_km[shell_index + 1])
            local_interface_width = 0.5 * (left_shell_width + right_shell_width)

            left_node_index.append(
                node_index(layout=layout, species_offset=species_offset, shell_index=shell_index)
            )
            right_node_index.append(
                node_index(layout=layout, species_offset=species_offset, shell_index=shell_index + 1)
            )
            interface_features.append(
                (
                    (interface_altitude - float(shell_edges_km[0])) / total_altitude_span,
                    local_interface_width / total_altitude_span,
                    species_index_norm,
                )
            )
            interface_altitude_km.append(interface_altitude)
            interface_width_km.append(local_interface_width)

    if len(left_node_index) == 0:
        feature_array = np.zeros((0, len(TRANSPORT_INTERFACE_FEATURE_NAMES)), dtype=np.float32)
    else:
        feature_array = np.asarray(interface_features, dtype=np.float32)

    return FieldlineTransportInterfaces(
        left_node_index=np.asarray(left_node_index, dtype=np.int64),
        right_node_index=np.asarray(right_node_index, dtype=np.int64),
        interface_features=feature_array,
        interface_altitude_km=np.asarray(interface_altitude_km, dtype=np.float32),
        interface_width_km=np.asarray(interface_width_km, dtype=np.float32),
    )


def build_fieldline_edge_index(
    layout: FieldlineGraphLayout,
    include_species_coupling_edges: bool = True,
) -> np.ndarray:
    """Build the directed edge list for the field-line distribution graph.

    Physical interpretation:
    - Same-species shell neighbors exchange mass, momentum, and energy along the field line.
    - Same-shell cross-species edges expose local composition coupling and collision-like effects.
    - Global-node edges broadcast exogenous forcing to every shell and collect shell summaries back.
    """
    edges: list[tuple[int, int]] = []

    # Altitude-neighbor edges define the 1-D transport backbone within each species.
    for species_offset in range(layout.num_species):
        for shell_index in range(layout.num_altitude_shells):
            src = node_index(layout=layout, species_offset=species_offset, shell_index=shell_index)
            if shell_index > 0:
                edges.append((src, node_index(layout=layout, species_offset=species_offset, shell_index=shell_index - 1)))
            if shell_index + 1 < layout.num_altitude_shells:
                edges.append((src, node_index(layout=layout, species_offset=species_offset, shell_index=shell_index + 1)))

    # Cross-species same-shell edges let each species respond to the local composition state.
    if include_species_coupling_edges and layout.num_species > 1:
        for shell_index in range(layout.num_altitude_shells):
            for src_species in range(layout.num_species):
                for dst_species in range(layout.num_species):
                    if src_species == dst_species:
                        continue
                    src = node_index(layout=layout, species_offset=src_species, shell_index=shell_index)
                    dst = node_index(layout=layout, species_offset=dst_species, shell_index=shell_index)
                    edges.append((src, dst))

    # Global-node edges couple all shell nodes to exogenous line-level forcing.
    if layout.include_global_node and layout.global_node_index is not None:
        global_index = layout.global_node_index
        for shell_node in range(layout.num_shell_nodes):
            edges.append((shell_node, global_index))
            edges.append((global_index, shell_node))

    # A contiguous integer edge array is the format expected by most graph runtimes.
    return np.asarray(edges, dtype=np.int64).T if edges else np.zeros((2, 0), dtype=np.int64)


def describe_fieldline_graph(
    layout: FieldlineGraphLayout,
    include_species_coupling_edges: bool = True,
) -> dict[str, int]:
    """Return a topology summary for dry-run inspection and test assertions."""
    edge_index = build_fieldline_edge_index(
        layout=layout,
        include_species_coupling_edges=include_species_coupling_edges,
    )
    return {
        "num_species": layout.num_species,
        "num_altitude_shells": layout.num_altitude_shells,
        "num_shell_nodes": layout.num_shell_nodes,
        "num_nodes": layout.num_nodes,
        "num_edges": int(edge_index.shape[1]),
        "has_global_node": int(layout.include_global_node),
        "has_species_coupling_edges": int(include_species_coupling_edges),
    }


def _read_exact(handle, n_bytes: int) -> bytes:
    """Read exactly `n_bytes` from a binary file object."""
    data = handle.read(n_bytes)
    if len(data) != n_bytes:
        raise EOFError(f"Expected {n_bytes} bytes, got {len(data)}")
    return data


def _split_binary_header(header: str) -> tuple[str, list[str]]:
    """Split the repeated binary header into row schema and optional snapshot metadata names."""
    if SNAPSHOT_META_PREFIX not in header:
        return header, []
    row_header, meta_header = header.split(SNAPSHOT_META_PREFIX, maxsplit=1)
    meta_names = [name.strip() for name in meta_header.split(",") if name.strip()]
    return row_header, meta_names


def _looks_like_binary_history(path: Path) -> bool:
    """Check whether a file follows the PWOM binary particle-history snapshot format."""
    try:
        with path.open("rb") as handle:
            raw = handle.read(4)
            if len(raw) != 4:
                return False
            header_len = int(np.frombuffer(raw, dtype="<i4")[0])
            if header_len <= 0 or header_len > 4096:
                return False
            header = _read_exact(handle, header_len).decode("ascii", errors="strict")
            row_header, _ = _split_binary_header(header)
            return row_header.startswith("ParticleID,Species,Altitude,")
    except Exception:
        return False


def _resolve_binary_source(data_dir: str, split: str) -> tuple[Path, bool]:
    """Resolve binary source file for a split.

    Return:
    - `(path, False)` when an explicit split file exists and is binary.
    - `(path, True)` when one shared history file should be split by time.
    """
    split_file = Path(data_dir) / f"{split}.dat"
    if split_file.exists() and _looks_like_binary_history(split_file):
        return split_file, False

    candidates = sorted(Path(data_dir).glob("particle_history_iline*.dat"))
    if len(candidates) == 1 and _looks_like_binary_history(candidates[0]):
        return candidates[0], True

    raise FileNotFoundError(
        f"No PWOM binary history found for split='{split}' in data_dir='{data_dir}'."
    )


def _scan_binary_snapshot_meta(path: Path) -> list[dict[str, int | float | str]]:
    """Scan binary snapshot metadata without materializing per-particle rows."""
    cache_key = str(path.resolve())
    cached = _BINARY_META_CACHE.get(cache_key)
    if cached is not None:
        return cached

    row_size = BINARY_ROW_DTYPE.itemsize
    file_size = int(path.stat().st_size)
    metas: list[dict[str, int | float | str]] = []
    with path.open("rb") as handle:
        while True:
            header_len_raw = handle.read(4)
            if not header_len_raw:
                break
            if len(header_len_raw) != 4:
                break

            header_len = int(np.frombuffer(header_len_raw, dtype="<i4")[0])
            if header_len <= 0:
                raise ValueError(f"Invalid header_len={header_len} in {path}")

            header = _read_exact(handle, header_len).decode("ascii", errors="strict")
            row_header, meta_names = _split_binary_header(header)
            line_global = int(np.frombuffer(_read_exact(handle, 8), dtype="<i8")[0])
            model_time = float(np.frombuffer(_read_exact(handle, 8), dtype="<f8")[0])
            n_particle = int(np.frombuffer(_read_exact(handle, 4), dtype="<i4")[0])
            if n_particle < 0:
                raise ValueError(f"Invalid n_particle={n_particle} in {path}")
            snapshot_meta: dict[str, float] = {}
            if meta_names:
                n_snapshot_meta = int(np.frombuffer(_read_exact(handle, 4), dtype="<i4")[0])
                if n_snapshot_meta != len(meta_names):
                    raise ValueError(
                        f"Snapshot metadata count mismatch in {path}: expected "
                        f"{len(meta_names)}, got {n_snapshot_meta}"
                    )
                meta_values = np.frombuffer(
                    _read_exact(handle, 8 * n_snapshot_meta),
                    dtype="<f8",
                ).astype(np.float64, copy=True)
                snapshot_meta = {
                    meta_name: float(meta_values[i]) for i, meta_name in enumerate(meta_names)
                }

            data_offset = int(handle.tell())
            data_end = data_offset + (n_particle * row_size)
            if data_end > file_size:
                break

            handle.seek(n_particle * row_size, os.SEEK_CUR)
            metas.append(
                {
                    "header": header,
                    "row_header": row_header,
                    "line_global": line_global,
                    "model_time": model_time,
                    "n_particle": n_particle,
                    "data_offset": data_offset,
                    "snapshot_meta": snapshot_meta,
                }
            )

    _BINARY_META_CACHE[cache_key] = metas
    return metas


def _shared_binary_split_bounds(total_timesteps: int) -> tuple[int, int]:
    """Compute contiguous train/valid bounds for one shared binary history file."""
    if total_timesteps < 2:
        raise ValueError("Need at least 2 timesteps in shared binary history.")

    valid_steps_env = os.environ.get("PWOM_BINARY_VALID_TIMESTEPS", "").strip()
    valid_frac_env = os.environ.get("PWOM_BINARY_VALID_FRACTION", "").strip()

    if valid_steps_env:
        valid_steps = int(valid_steps_env)
    elif valid_frac_env:
        valid_steps = int(round(total_timesteps * float(valid_frac_env)))
    else:
        valid_steps = int(round(total_timesteps * 0.2))

    valid_steps = max(1, min(valid_steps, total_timesteps - 1))
    train_end = total_timesteps - valid_steps
    return 0, train_end


def _initial_rank_indices(block_size: int, num_particles: int | None) -> np.ndarray:
    """Create deterministic rank anchors for per-timestep fixed-size resampling."""
    if block_size <= 0:
        raise ValueError("Cannot initialize rank anchors from an empty timestep block.")
    if num_particles is not None and num_particles > 0:
        return np.linspace(0, block_size - 1, num=num_particles, dtype=np.int64)
    return np.arange(block_size, dtype=np.int64)


def _map_rank_indices(rank_indices: np.ndarray, block_size: int) -> np.ndarray:
    """Map reference rank anchors onto a timestep block with potentially different size."""
    if block_size <= 0:
        raise ValueError("Cannot map rank anchors to an empty timestep block.")
    if rank_indices.size == 0:
        raise ValueError("Rank anchors are empty.")
    if block_size == 1:
        return np.zeros(rank_indices.shape[0], dtype=np.int64)

    ref_max = int(np.max(rank_indices))
    if ref_max <= 0:
        return np.zeros(rank_indices.shape[0], dtype=np.int64)

    mapped = np.rint(
        rank_indices.astype(np.float64) * float(block_size - 1) / float(ref_max)
    ).astype(np.int64)
    return np.clip(mapped, 0, block_size - 1)


def _sample_snapshot_rows(
    path: Path,
    meta: dict[str, int | float | str],
    rank_indices: np.ndarray,
) -> np.ndarray:
    """Read one snapshot and return the deterministic sampled rows."""
    rows_mm = np.memmap(
        path,
        dtype=BINARY_ROW_DTYPE,
        mode="r",
        offset=int(meta["data_offset"]),
        shape=(int(meta["n_particle"]),),
    )
    mapped = _map_rank_indices(rank_indices=rank_indices, block_size=int(meta["n_particle"]))
    sampled_rows = np.asarray(rows_mm[mapped], dtype=BINARY_ROW_DTYPE)
    del rows_mm
    return sampled_rows


def _hash_int_array(values: np.ndarray) -> str:
    """Build a short deterministic hash for integer index arrays."""
    return hashlib.sha1(np.asarray(values, dtype=np.int64).tobytes()).hexdigest()[:12]


def _graph_cache_path(
    data_dir: str,
    source_path: Path,
    split: str,
    split_start: int,
    split_end: int,
    num_particles: int | None,
    num_altitude_shells: int,
    include_global_node: bool,
    include_species_coupling_edges: bool,
    species_ids: tuple[int, ...] | None,
    shell_edges_km_override: np.ndarray | None,
    rank_indices: np.ndarray,
) -> Path:
    """Construct a cache path for coarse graph-series tensors."""
    cache_dir = Path(data_dir) / ".pwom_graph_cache"
    source_stat = source_path.stat()
    source_sig = f"{source_path.name}_{source_stat.st_size}_{source_stat.st_mtime_ns}"
    species_sig = "auto" if species_ids is None else "-".join(str(int(s)) for s in species_ids)
    particle_sig = "all" if num_particles is None else str(int(num_particles))
    rank_sig = _hash_int_array(rank_indices)
    shell_sig = (
        "auto"
        if shell_edges_km_override is None
        else hashlib.sha1(np.asarray(shell_edges_km_override, dtype=np.float32).tobytes()).hexdigest()[:12]
    )
    name = (
        f"fieldline_v{FIELDLINE_GRAPH_CACHE_VERSION}_"
        f"{source_sig}_"
        f"{split}_{split_start}_{split_end}_"
        f"N{particle_sig}_"
        f"S{num_altitude_shells}_"
        f"G{int(include_global_node)}_"
        f"C{int(include_species_coupling_edges)}_"
        f"species-{species_sig}_"
        f"shell-{shell_sig}_"
        f"rank-{rank_sig}.npz"
    )
    return cache_dir / name


def _load_graph_cache(cache_path: Path) -> dict[str, np.ndarray] | None:
    """Load cached graph tensors when available."""
    if not cache_path.exists():
        return None
    try:
        with np.load(cache_path, allow_pickle=False) as payload:
            return {
                "edge_index": np.asarray(payload["edge_index"], dtype=np.int64),
                "static_node_features": np.asarray(payload["static_node_features"], dtype=np.float32),
                "dynamic_node_features": np.asarray(payload["dynamic_node_features"], dtype=np.float32),
                "global_context_features": np.asarray(payload["global_context_features"], dtype=np.float32),
                "boundary_context_features": np.asarray(payload["boundary_context_features"], dtype=np.float32),
                "model_times": np.asarray(payload["model_times"], dtype=np.float64),
                "shell_edges_km": np.asarray(payload["shell_edges_km"], dtype=np.float32),
                "species_ids": np.asarray(payload["species_ids"], dtype=np.int64),
                "rank_indices": np.asarray(payload["rank_indices"], dtype=np.int64),
            }
    except Exception:
        return None


def _save_graph_cache(
    cache_path: Path,
    edge_index: np.ndarray,
    static_node_features: np.ndarray,
    dynamic_node_features: np.ndarray,
    global_context_features: np.ndarray,
    boundary_context_features: np.ndarray,
    model_times: np.ndarray,
    shell_edges_km: np.ndarray,
    species_ids: tuple[int, ...],
    rank_indices: np.ndarray,
) -> None:
    """Persist coarse graph tensors so repeated previews do not rescan the binary history."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        cache_path,
        edge_index=edge_index.astype(np.int64, copy=False),
        static_node_features=static_node_features.astype(np.float32, copy=False),
        dynamic_node_features=dynamic_node_features.astype(np.float32, copy=False),
        global_context_features=global_context_features.astype(np.float32, copy=False),
        boundary_context_features=boundary_context_features.astype(np.float32, copy=False),
        model_times=model_times.astype(np.float64, copy=False),
        shell_edges_km=shell_edges_km.astype(np.float32, copy=False),
        species_ids=np.asarray(species_ids, dtype=np.int64),
        rank_indices=rank_indices.astype(np.int64, copy=False),
    )


def build_static_node_features(
    layout: FieldlineGraphLayout,
    shell_edges_km: np.ndarray,
) -> np.ndarray:
    """Build the static node descriptors for one field-line graph layout."""
    static = np.zeros((layout.num_nodes, len(STATIC_NODE_FEATURE_NAMES)), dtype=np.float32)

    shell_centers_km = 0.5 * (shell_edges_km[:-1] + shell_edges_km[1:])
    shell_widths_km = shell_edges_km[1:] - shell_edges_km[:-1]
    denom_species = float(max(layout.num_species - 1, 1))

    # Shell nodes are arranged in one species-major stack so each node gets a fixed geometric tag.
    for species_offset in range(layout.num_species):
        species_code = float(species_offset) / denom_species
        for shell_index in range(layout.num_altitude_shells):
            idx = node_index(layout=layout, species_offset=species_offset, shell_index=shell_index)
            static[idx, 0] = float(shell_centers_km[shell_index])
            static[idx, 1] = float(shell_widths_km[shell_index])
            static[idx, 2] = species_code
            static[idx, 3] = 0.0

    # The global node uses the line midpoint and full altitude span as its geometric descriptor.
    if layout.include_global_node and layout.global_node_index is not None:
        static[layout.global_node_index, 0] = float(0.5 * (shell_edges_km[0] + shell_edges_km[-1]))
        static[layout.global_node_index, 1] = float(shell_edges_km[-1] - shell_edges_km[0])
        static[layout.global_node_index, 2] = 0.0
        static[layout.global_node_index, 3] = 1.0

    return static


def _weighted_mean(values: np.ndarray, weights: np.ndarray, fallback: float) -> float:
    """Compute a weighted mean with a deterministic fallback for empty shells."""
    weight_sum = float(np.sum(weights))
    if weight_sum <= 0.0 or values.size == 0:
        return float(fallback)
    return float(np.sum(values * weights) / weight_sum)


def _weighted_std(values: np.ndarray, weights: np.ndarray, fallback_mean: float) -> float:
    """Compute a weighted standard deviation with a safe empty-shell fallback."""
    weight_sum = float(np.sum(weights))
    if weight_sum <= 0.0 or values.size == 0:
        return 0.0
    mean_value = float(np.sum(values * weights) / weight_sum)
    variance = float(np.sum(weights * np.square(values - mean_value)) / weight_sum)
    return float(np.sqrt(max(variance, 0.0)))


def _global_dynamic_summary(rows: np.ndarray) -> dict[str, float]:
    """Compute line-integrated dynamic summary values for one snapshot.

    Physical interpretation:
    - These are true global shell means and spreads before any density-weighted remapping.
    - The global graph node later stores the same values because its population fraction is one by
      construction.
    """
    if rows.size == 0:
        raise ValueError("Cannot summarize an empty snapshot.")

    altitude_km = rows["altitude"].astype(np.float64) / 100000.0
    vpar = rows["vpar"].astype(np.float64)
    vperp = rows["vperp"].astype(np.float64)
    weights = rows["weight"].astype(np.float64)
    b = rows["b"].astype(np.float64)
    e = rows["e"].astype(np.float64)
    g = rows["g"].astype(np.float64)
    is_open = rows["is_open"].astype(np.float64)

    total_weight = float(np.sum(weights))
    total_weight = max(total_weight, 0.0)
    global_mean_altitude = _weighted_mean(values=altitude_km, weights=weights, fallback=float(np.mean(altitude_km)))

    return {
        "total_weight": total_weight,
        "open_fraction": _weighted_mean(values=is_open, weights=weights, fallback=float(np.mean(is_open))),
        "mean_altitude_km": global_mean_altitude,
        "mean_vpar": _weighted_mean(values=vpar, weights=weights, fallback=float(np.mean(vpar))),
        "mean_vperp": _weighted_mean(values=vperp, weights=weights, fallback=float(np.mean(vperp))),
        "std_altitude_km": _weighted_std(values=altitude_km, weights=weights, fallback_mean=global_mean_altitude),
        "std_vpar": _weighted_std(values=vpar, weights=weights, fallback_mean=float(np.mean(vpar))),
        "std_vperp": _weighted_std(values=vperp, weights=weights, fallback_mean=float(np.mean(vperp))),
        "mean_b": _weighted_mean(values=b, weights=weights, fallback=float(np.mean(b))),
        "mean_e": _weighted_mean(values=e, weights=weights, fallback=float(np.mean(e))),
        "mean_g": _weighted_mean(values=g, weights=weights, fallback=float(np.mean(g))),
    }


def _snapshot_meta_values_by_prefix(
    snapshot_meta: dict[str, float] | None,
    prefix: str,
    num_species: int,
) -> np.ndarray:
    """Extract one species-ordered metadata block from the binary snapshot metadata dictionary.

    Physical interpretation:
    - PWOM writes boundary moments and WPI coefficients once per snapshot in species order.
    - Every shell node of one species should see the same boundary/regime driver values because
      those inputs are imposed externally by the fluid/WPI side of the coupled model.
    """
    if not snapshot_meta:
        return np.zeros(num_species, dtype=np.float32)

    values = [float(value) for key, value in snapshot_meta.items() if key.startswith(prefix)]
    if not values:
        return np.zeros(num_species, dtype=np.float32)

    block = np.zeros(num_species, dtype=np.float32)
    copy_count = min(num_species, len(values))
    block[:copy_count] = np.asarray(values[:copy_count], dtype=np.float32)
    return block


def _snapshot_driver_context(
    snapshot_meta: dict[str, float] | None,
    num_species: int,
) -> dict[str, np.ndarray | float]:
    """Convert one snapshot metadata dictionary into line-level and species-level PWOM drivers.

    Physical interpretation:
    - `sm_lat`, `dt_advance`, `is_cusp_or_aurora`, and `use_wpi` are line-global drivers.
    - Boundary moments and WPI coefficients are species-specific lower-boundary inputs and should
      enter the graph near the bottom shell instead of being copied into every shell node.
    """
    snapshot_meta = snapshot_meta or {}
    return {
        "sm_lat": float(snapshot_meta.get("SmLat", 0.0)),
        "dt_advance": float(snapshot_meta.get("DtAdvance", 0.0)),
        "is_cusp_or_aurora": float(snapshot_meta.get("IsCuspOrAurora", 0.0)),
        "use_wpi": float(snapshot_meta.get("UseWPI", 0.0)),
        "boundary_density": _snapshot_meta_values_by_prefix(snapshot_meta, "DensityBC_", num_species),
        "boundary_velocity": _snapshot_meta_values_by_prefix(snapshot_meta, "VelocityBC_", num_species),
        "boundary_temperature": _snapshot_meta_values_by_prefix(snapshot_meta, "TemperatureBC_", num_species),
        "wpi_dperp": _snapshot_meta_values_by_prefix(snapshot_meta, "Dperp_", num_species),
        "wpi_dexp": _snapshot_meta_values_by_prefix(snapshot_meta, "Dexp_", num_species),
    }


def aggregate_snapshot_global_context(
    snapshot_meta: dict[str, float] | None = None,
) -> np.ndarray:
    """Build the separate line-global PWOM control vector for one snapshot.

    Physical interpretation:
    - These controls apply to the whole field line and should condition the graph through a
      dedicated encoder, not by masquerading as shell-resolved state.
    - Only controls that vary in the present dataset are kept here. Constant metadata such as
      fixed timestep or always-on WPI flags are excluded because they cannot help forecasting.
    """
    snapshot_meta = snapshot_meta or {}
    return np.asarray(
        [
            float(snapshot_meta.get("SmLat", 0.0)),
            float(snapshot_meta.get("IsCuspOrAurora", 0.0)),
        ],
        dtype=np.float32,
    )


def aggregate_snapshot_dynamic_features(
    rows: np.ndarray,
    layout: FieldlineGraphLayout,
    shell_edges_km: np.ndarray,
    snapshot_meta: dict[str, float] | None = None,
) -> np.ndarray:
    """Aggregate one sampled PWOM snapshot into per-node dynamic features.

    Physical interpretation:
    - Shell occupancy is measured with population fraction and total weight.
    - Velocity channels are stored as population-weighted moments so sparse shells contribute
      proportionally less momentum-like signal instead of an unstable raw mean.
    """
    dynamic = np.zeros((layout.num_nodes, len(DYNAMIC_NODE_FEATURE_NAMES)), dtype=np.float32)

    altitude_km = rows["altitude"].astype(np.float64) / 100000.0
    species = rows["species"].astype(np.int64)
    weights = rows["weight"].astype(np.float64)
    global_summary = _global_dynamic_summary(rows=rows)
    global_total_weight = max(global_summary["total_weight"], 1e-12)
    shell_centers_km = 0.5 * (shell_edges_km[:-1] + shell_edges_km[1:])
    shell_indices = np.searchsorted(shell_edges_km, altitude_km, side="right") - 1
    shell_indices = np.clip(shell_indices, 0, layout.num_altitude_shells - 1)

    # One shell state summarizes the local distribution represented by sampled macro-particles.
    for species_offset, species_id in enumerate(layout.species_ids):
        species_mask = species == species_id
        for shell_index in range(layout.num_altitude_shells):
            node_idx = node_index(layout=layout, species_offset=species_offset, shell_index=shell_index)
            mask = species_mask & (shell_indices == shell_index)
            shell_rows = rows[mask]
            shell_weights = shell_rows["weight"].astype(np.float64) if shell_rows.size > 0 else np.zeros(0, dtype=np.float64)
            shell_altitude_km = shell_rows["altitude"].astype(np.float64) / 100000.0 if shell_rows.size > 0 else np.zeros(0, dtype=np.float64)

            total_weight = float(np.sum(shell_weights)) if shell_rows.size > 0 else 0.0
            population_fraction = float(total_weight / global_total_weight)
            shell_mean_vpar = float(
                _weighted_mean(
                    values=shell_rows["vpar"].astype(np.float64) if shell_rows.size > 0 else np.zeros(0, dtype=np.float64),
                    weights=shell_weights,
                    fallback=0.0,
                )
            )
            shell_mean_vperp = float(
                _weighted_mean(
                    values=shell_rows["vperp"].astype(np.float64) if shell_rows.size > 0 else np.zeros(0, dtype=np.float64),
                    weights=shell_weights,
                    fallback=0.0,
                )
            )
            shell_std_vpar = float(
                _weighted_std(
                    values=shell_rows["vpar"].astype(np.float64) if shell_rows.size > 0 else np.zeros(0, dtype=np.float64),
                    weights=shell_weights,
                    fallback_mean=0.0,
                )
            )
            shell_std_vperp = float(
                _weighted_std(
                    values=shell_rows["vperp"].astype(np.float64) if shell_rows.size > 0 else np.zeros(0, dtype=np.float64),
                    weights=shell_weights,
                    fallback_mean=0.0,
                )
            )
            dynamic[node_idx, 0] = population_fraction
            dynamic[node_idx, 1] = float(np.log1p(total_weight))
            dynamic[node_idx, 2] = float(
                _weighted_mean(
                    values=shell_rows["is_open"].astype(np.float64) if shell_rows.size > 0 else np.zeros(0, dtype=np.float64),
                    weights=shell_weights,
                    fallback=global_summary["open_fraction"],
                )
            )
            dynamic[node_idx, 3] = float(
                _weighted_mean(
                    values=shell_altitude_km,
                    weights=shell_weights,
                    fallback=float(shell_centers_km[shell_index]),
                )
            )
            # Population weighting turns fragile shell means into momentum-density-like targets.
            dynamic[node_idx, 4] = population_fraction * shell_mean_vpar
            dynamic[node_idx, 5] = population_fraction * shell_mean_vperp
            dynamic[node_idx, 6] = float(
                _weighted_std(
                    values=shell_altitude_km,
                    weights=shell_weights,
                    fallback_mean=float(shell_centers_km[shell_index]),
                )
            )
            dynamic[node_idx, 7] = population_fraction * shell_std_vpar
            dynamic[node_idx, 8] = population_fraction * shell_std_vperp
            dynamic[node_idx, 9] = float(
                _weighted_mean(
                    values=shell_rows["b"].astype(np.float64) if shell_rows.size > 0 else np.zeros(0, dtype=np.float64),
                    weights=shell_weights,
                    fallback=global_summary["mean_b"],
                )
            )
            dynamic[node_idx, 10] = float(
                _weighted_mean(
                    values=shell_rows["e"].astype(np.float64) if shell_rows.size > 0 else np.zeros(0, dtype=np.float64),
                    weights=shell_weights,
                    fallback=global_summary["mean_e"],
                )
            )
            dynamic[node_idx, 11] = float(
                _weighted_mean(
                    values=shell_rows["g"].astype(np.float64) if shell_rows.size > 0 else np.zeros(0, dtype=np.float64),
                    weights=shell_weights,
                    fallback=global_summary["mean_g"],
                )
            )
    # The global node summarizes the whole field-line snapshot and acts as broadcast context.
    if layout.include_global_node and layout.global_node_index is not None:
        idx = layout.global_node_index
        dynamic[idx, 0] = 1.0
        dynamic[idx, 1] = float(np.log1p(global_summary["total_weight"]))
        dynamic[idx, 2] = float(global_summary["open_fraction"])
        dynamic[idx, 3] = float(global_summary["mean_altitude_km"])
        # The global node has unit population fraction, so weighted and unweighted moments match.
        dynamic[idx, 4] = float(global_summary["mean_vpar"])
        dynamic[idx, 5] = float(global_summary["mean_vperp"])
        dynamic[idx, 6] = float(global_summary["std_altitude_km"])
        dynamic[idx, 7] = float(global_summary["std_vpar"])
        dynamic[idx, 8] = float(global_summary["std_vperp"])
        dynamic[idx, 9] = float(global_summary["mean_b"])
        dynamic[idx, 10] = float(global_summary["mean_e"])
        dynamic[idx, 11] = float(global_summary["mean_g"])

    return dynamic


def aggregate_snapshot_boundary_context(
    layout: FieldlineGraphLayout,
    snapshot_meta: dict[str, float] | None = None,
) -> np.ndarray:
    """Build the lower-boundary control tensor for one sampled PWOM snapshot.

    Physical interpretation:
    - Boundary moments and WPI coefficients are externally imposed controls.
    - Only the lowest shell of each species receives them directly; all other nodes must infer
      their influence through message passing from those boundary-adjacent cells.
    """
    boundary_context = np.zeros((layout.num_nodes, len(BOUNDARY_CONTEXT_FEATURE_NAMES)), dtype=np.float32)
    snapshot_drivers = _snapshot_driver_context(snapshot_meta=snapshot_meta, num_species=layout.num_species)
    for species_offset in range(layout.num_species):
        node_idx = node_index(layout=layout, species_offset=species_offset, shell_index=0)
        boundary_context[node_idx, BOUNDARY_CONTEXT_FEATURE_INDEX["boundary_density"]] = float(
            snapshot_drivers["boundary_density"][species_offset]
        )
        boundary_context[node_idx, BOUNDARY_CONTEXT_FEATURE_INDEX["boundary_velocity"]] = float(
            snapshot_drivers["boundary_velocity"][species_offset]
        )
        boundary_context[node_idx, BOUNDARY_CONTEXT_FEATURE_INDEX["boundary_temperature"]] = float(
            snapshot_drivers["boundary_temperature"][species_offset]
        )
        boundary_context[node_idx, BOUNDARY_CONTEXT_FEATURE_INDEX["wpi_dperp"]] = float(
            snapshot_drivers["wpi_dperp"][species_offset]
        )
        boundary_context[node_idx, BOUNDARY_CONTEXT_FEATURE_INDEX["wpi_dexp"]] = float(
            snapshot_drivers["wpi_dexp"][species_offset]
        )
    return boundary_context


def _infer_species_ids(
    sampled_snapshots: list[np.ndarray],
    explicit_species_ids: tuple[int, ...] | None,
) -> tuple[int, ...]:
    """Infer which species stacks should exist in the field-line graph."""
    if explicit_species_ids is not None:
        return tuple(int(species_id) for species_id in explicit_species_ids)

    species_values: set[int] = set()
    for rows in sampled_snapshots:
        species_values.update(int(species_id) for species_id in np.unique(rows["species"]))

    if not species_values:
        raise ValueError("No species IDs were found in the sampled binary snapshots.")
    return tuple(sorted(species_values))


def _build_shell_edges_km(
    sampled_snapshots: list[np.ndarray],
    num_altitude_shells: int,
) -> np.ndarray:
    """Infer one fixed altitude discretization from the sampled split."""
    altitude_min_km = float("inf")
    altitude_max_km = float("-inf")

    for rows in sampled_snapshots:
        if rows.size == 0:
            continue
        altitude_km = rows["altitude"].astype(np.float64) / 100000.0
        altitude_min_km = min(altitude_min_km, float(np.min(altitude_km)))
        altitude_max_km = max(altitude_max_km, float(np.max(altitude_km)))

    if not np.isfinite(altitude_min_km) or not np.isfinite(altitude_max_km):
        raise ValueError("Unable to infer altitude shell bounds from sampled snapshots.")

    # A tiny pad prevents the max-altitude sample from sitting on an exclusive top edge.
    if altitude_max_km <= altitude_min_km:
        altitude_max_km = altitude_min_km + 1.0
    pad_km = max((altitude_max_km - altitude_min_km) * 1.0e-6, 1.0e-6)
    return np.linspace(
        altitude_min_km,
        altitude_max_km + pad_km,
        num=num_altitude_shells + 1,
        dtype=np.float32,
    )


def load_fieldline_graph_series(
    data_dir: str,
    split: str,
    num_particles: int | None,
    num_altitude_shells: int,
    include_global_node: bool = True,
    include_species_coupling_edges: bool = True,
    species_ids: tuple[int, ...] | None = None,
    shell_edges_km_override: np.ndarray | None = None,
    verbose: bool = False,
) -> FieldlineGraphSeries:
    """Load one split as a coarse field-line graph time series.

    Why this design:
    - The binary particle file is too large to load fully into memory.
    - Sampling a fixed number of rank anchors per snapshot preserves distribution coverage while
      keeping preprocessing bounded and deterministic.
    """
    source_path, is_shared_history = _resolve_binary_source(data_dir=data_dir, split=split)
    metas = _scan_binary_snapshot_meta(source_path)
    if len(metas) < 2:
        raise ValueError(f"Binary history {source_path} has fewer than 2 timesteps.")

    split_start = 0
    split_end = len(metas)
    if is_shared_history:
        _, train_end = _shared_binary_split_bounds(total_timesteps=len(metas))
        if split == "train":
            split_start, split_end = 0, train_end
        elif split == "valid":
            split_start, split_end = train_end, len(metas)
        else:
            raise ValueError(f"Unsupported split='{split}' for shared binary history.")

    split_metas = metas[split_start:split_end]
    if len(split_metas) < 2:
        raise ValueError(f"Split '{split}' has too few timesteps for graph supervision.")

    first_block_size = int(split_metas[0]["n_particle"])
    rank_indices = _initial_rank_indices(block_size=first_block_size, num_particles=num_particles)
    cache_path = _graph_cache_path(
        data_dir=data_dir,
        source_path=source_path,
        split=split,
        split_start=split_start,
        split_end=split_end,
        num_particles=num_particles,
        num_altitude_shells=num_altitude_shells,
        include_global_node=include_global_node,
        include_species_coupling_edges=include_species_coupling_edges,
        species_ids=species_ids,
        shell_edges_km_override=shell_edges_km_override,
        rank_indices=rank_indices,
    )

    cached = _load_graph_cache(cache_path=cache_path)
    if cached is not None:
        layout = FieldlineGraphLayout(
            species_ids=tuple(int(species_id) for species_id in cached["species_ids"].tolist()),
            num_altitude_shells=num_altitude_shells,
            include_global_node=include_global_node,
        )
        return FieldlineGraphSeries(
            layout=layout,
            edge_index=cached["edge_index"],
            static_node_features=cached["static_node_features"],
            dynamic_node_features=cached["dynamic_node_features"],
            global_context_features=cached["global_context_features"],
            boundary_context_features=cached["boundary_context_features"],
            model_times=cached["model_times"],
            shell_edges_km=cached["shell_edges_km"],
            rank_indices=cached["rank_indices"],
            split=split,
            source_path=str(source_path),
        )

    sampled_snapshots: list[np.ndarray] = []
    model_times: list[float] = []
    iterator = tqdm(split_metas, desc=f"graph:{split}", unit="snap", dynamic_ncols=True) if verbose else split_metas

    # Each snapshot is sampled independently because PWOM re-creates macro-particles every timestep.
    for meta in iterator:
        if str(meta.get("row_header", meta["header"])) != BINARY_EXPECTED_ROW_HEADER:
            raise ValueError(f"Unexpected binary row header in {source_path}: {meta['header']}")
        sampled_snapshots.append(_sample_snapshot_rows(path=source_path, meta=meta, rank_indices=rank_indices))
        model_times.append(float(meta["model_time"]))

    inferred_species_ids = _infer_species_ids(
        sampled_snapshots=sampled_snapshots,
        explicit_species_ids=species_ids,
    )
    layout = FieldlineGraphLayout(
        species_ids=inferred_species_ids,
        num_altitude_shells=num_altitude_shells,
        include_global_node=include_global_node,
    )
    shell_edges_km = (
        np.asarray(shell_edges_km_override, dtype=np.float32).copy()
        if shell_edges_km_override is not None
        else _build_shell_edges_km(
            sampled_snapshots=sampled_snapshots,
            num_altitude_shells=num_altitude_shells,
        )
    )
    static_node_features = build_static_node_features(layout=layout, shell_edges_km=shell_edges_km)
    edge_index = build_fieldline_edge_index(
        layout=layout,
        include_species_coupling_edges=include_species_coupling_edges,
    )

    dynamic_blocks = [
        aggregate_snapshot_dynamic_features(
            rows=rows,
            layout=layout,
            shell_edges_km=shell_edges_km,
            snapshot_meta=split_metas[snapshot_index].get("snapshot_meta", {}),
        )
        for snapshot_index, rows in enumerate(sampled_snapshots)
    ]
    global_context_blocks = [
        aggregate_snapshot_global_context(
            snapshot_meta=split_metas[snapshot_index].get("snapshot_meta", {}),
        )
        for snapshot_index in range(len(sampled_snapshots))
    ]
    boundary_context_blocks = [
        aggregate_snapshot_boundary_context(
            layout=layout,
            snapshot_meta=split_metas[snapshot_index].get("snapshot_meta", {}),
        )
        for snapshot_index in range(len(sampled_snapshots))
    ]
    dynamic_node_features = np.stack(dynamic_blocks, axis=0).astype(np.float32, copy=False)
    global_context_features = np.stack(global_context_blocks, axis=0).astype(np.float32, copy=False)
    boundary_context_features = np.stack(boundary_context_blocks, axis=0).astype(np.float32, copy=False)
    model_times_array = np.asarray(model_times, dtype=np.float64)

    _save_graph_cache(
        cache_path=cache_path,
        edge_index=edge_index,
        static_node_features=static_node_features,
        dynamic_node_features=dynamic_node_features,
        global_context_features=global_context_features,
        boundary_context_features=boundary_context_features,
        model_times=model_times_array,
        shell_edges_km=shell_edges_km,
        species_ids=inferred_species_ids,
        rank_indices=rank_indices,
    )

    return FieldlineGraphSeries(
        layout=layout,
        edge_index=edge_index,
        static_node_features=static_node_features,
        dynamic_node_features=dynamic_node_features,
        global_context_features=global_context_features,
        boundary_context_features=boundary_context_features,
        model_times=model_times_array,
        shell_edges_km=shell_edges_km,
        rank_indices=rank_indices,
        split=split,
        source_path=str(source_path),
    )


def build_fieldline_window_arrays(
    series: FieldlineGraphSeries,
    window_length: int,
    target_schema: str = TARGET_SCHEMA_POPULATION_WEIGHTED,
) -> FieldlineWindowArrays:
    """Build one-step graph supervision windows from a graph time series.

    Physical interpretation:
    - A supervision sample contains a short history of shell states and the immediately following
      shell state.
    - Flattening the history into node channels keeps one-step training simple while still
      exposing short-memory transport context.
    """
    if window_length < 1:
        raise ValueError("window_length must be at least 1.")
    if series.num_timesteps <= window_length:
        raise ValueError(
            f"Need more timesteps than window_length. T={series.num_timesteps} window_length={window_length}."
        )

    target_schema = validate_target_schema(target_schema)
    static = series.static_node_features.astype(np.float32, copy=False)
    dynamic_physical = build_target_dynamic_view(
        dynamic_physical=series.dynamic_node_features.astype(np.float32, copy=False),
        target_schema=target_schema,
    )
    # The trainer consumes a transformed state space so stiff velocity moments remain numerically
    # comparable to the other shell statistics.
    dynamic = transform_dynamic_features(dynamic_physical=dynamic_physical, target_schema=target_schema)
    samples = series.num_timesteps - window_length
    num_nodes = static.shape[0]
    input_dim = static.shape[-1] + (window_length * dynamic.shape[-1])
    global_context_dim = series.global_context_features.shape[-1]
    boundary_context_dim = series.boundary_context_features.shape[-1]

    input_features = np.zeros((samples, num_nodes, input_dim), dtype=np.float32)
    input_global_context = np.zeros((samples, window_length * global_context_dim), dtype=np.float32)
    input_boundary_context = np.zeros((samples, num_nodes, boundary_context_dim), dtype=np.float32)
    last_dynamic = np.zeros((samples, num_nodes, dynamic.shape[-1]), dtype=np.float32)
    target_dynamic = np.zeros((samples, num_nodes, dynamic.shape[-1]), dtype=np.float32)
    target_delta = np.zeros((samples, num_nodes, dynamic.shape[-1]), dtype=np.float32)
    last_dynamic_physical = np.zeros((samples, num_nodes, dynamic.shape[-1]), dtype=np.float32)
    target_dynamic_physical = np.zeros((samples, num_nodes, dynamic.shape[-1]), dtype=np.float32)
    input_times = np.zeros(samples, dtype=np.float64)
    target_times = np.zeros(samples, dtype=np.float64)

    # Each training example uses a contiguous shell-state history to predict the next shell state.
    for sample_index in range(samples):
        history = dynamic[sample_index : sample_index + window_length]
        history_global_context = series.global_context_features[
            sample_index : sample_index + window_length
        ]
        history_boundary_context = series.boundary_context_features[
            sample_index : sample_index + window_length
        ]
        history_flat = np.transpose(history, (1, 0, 2)).reshape(num_nodes, window_length * dynamic.shape[-1])
        input_features[sample_index] = np.concatenate((static, history_flat), axis=-1)
        input_global_context[sample_index] = history_global_context.reshape(-1)
        input_boundary_context[sample_index] = history_boundary_context[-1]
        last_dynamic[sample_index] = history[-1]
        target_dynamic[sample_index] = dynamic[sample_index + window_length]
        target_delta[sample_index] = target_dynamic[sample_index] - last_dynamic[sample_index]
        last_dynamic_physical[sample_index] = dynamic_physical[sample_index + window_length - 1]
        target_dynamic_physical[sample_index] = dynamic_physical[sample_index + window_length]
        input_times[sample_index] = float(series.model_times[sample_index + window_length - 1])
        target_times[sample_index] = float(series.model_times[sample_index + window_length])

    return FieldlineWindowArrays(
        input_features=input_features,
        input_global_context=input_global_context,
        input_boundary_context=input_boundary_context,
        last_dynamic=last_dynamic,
        target_dynamic=target_dynamic,
        target_delta=target_delta,
        last_dynamic_physical=last_dynamic_physical,
        target_dynamic_physical=target_dynamic_physical,
        input_times=input_times,
        target_times=target_times,
    )


def build_fieldline_rollout_arrays(
    series: FieldlineGraphSeries,
    window_length: int,
    horizon: int,
) -> FieldlineRolloutArrays:
    """Build fixed-horizon autoregressive rollout windows from a graph time series.

    Physical interpretation:
    - One rollout sample starts from an observed history of shell states and provides the next
      `horizon` true shell states as supervision.
    - The target delta sequence measures each future step relative to the immediately previous
      shell state, which matches the residual evolution assumption used by the model.
    """
    if window_length < 1:
        raise ValueError("window_length must be at least 1.")
    if horizon < 1:
        raise ValueError("horizon must be at least 1.")
    if series.num_timesteps <= (window_length + horizon - 1):
        raise ValueError(
            "Need more timesteps than window_length + horizon - 1. "
            f"T={series.num_timesteps} window_length={window_length} horizon={horizon}."
        )

    dynamic_physical = series.dynamic_node_features.astype(np.float32, copy=False)
    # Rollout uses the same transformed shell state representation as one-step pretraining so the
    # checkpoint can be fine-tuned without a schema mismatch.
    dynamic = transform_dynamic_features(dynamic_physical=dynamic_physical)
    samples = series.num_timesteps - window_length - horizon + 1
    global_context_dim = series.global_context_features.shape[-1]
    boundary_context_dim = series.boundary_context_features.shape[-1]
    history_dynamic = np.zeros(
        (samples, window_length, dynamic.shape[1], dynamic.shape[2]),
        dtype=np.float32,
    )
    history_global_context = np.zeros((samples, window_length, global_context_dim), dtype=np.float32)
    history_boundary_context = np.zeros(
        (samples, window_length, dynamic.shape[1], boundary_context_dim),
        dtype=np.float32,
    )
    target_dynamic_seq = np.zeros(
        (samples, horizon, dynamic.shape[1], dynamic.shape[2]),
        dtype=np.float32,
    )
    target_delta_seq = np.zeros_like(target_dynamic_seq)
    target_dynamic_seq_physical = np.zeros_like(target_dynamic_seq)
    target_global_context_seq = np.zeros((samples, horizon, global_context_dim), dtype=np.float32)
    target_boundary_context_seq = np.zeros(
        (samples, horizon, dynamic.shape[1], boundary_context_dim),
        dtype=np.float32,
    )
    input_time = np.zeros(samples, dtype=np.float64)
    target_times = np.zeros((samples, horizon), dtype=np.float64)

    # Each sample uses contiguous timesteps because shell transport evolves causally in time.
    for sample_index in range(samples):
        history = dynamic[sample_index : sample_index + window_length]
        future = dynamic[sample_index + window_length : sample_index + window_length + horizon]
        history_context = series.global_context_features[sample_index : sample_index + window_length]
        history_boundary = series.boundary_context_features[sample_index : sample_index + window_length]
        future_context = series.global_context_features[
            sample_index + window_length : sample_index + window_length + horizon
        ]
        future_boundary = series.boundary_context_features[
            sample_index + window_length : sample_index + window_length + horizon
        ]
        history_dynamic[sample_index] = history
        history_global_context[sample_index] = history_context
        history_boundary_context[sample_index] = history_boundary
        target_dynamic_seq[sample_index] = future
        target_dynamic_seq_physical[sample_index] = dynamic_physical[
            sample_index + window_length : sample_index + window_length + horizon
        ]
        target_global_context_seq[sample_index] = future_context
        target_boundary_context_seq[sample_index] = future_boundary

        previous = history[-1]
        for horizon_index in range(horizon):
            current = future[horizon_index]
            target_delta_seq[sample_index, horizon_index] = current - previous
            previous = current

        input_time[sample_index] = float(series.model_times[sample_index + window_length - 1])
        target_times[sample_index] = series.model_times[
            sample_index + window_length : sample_index + window_length + horizon
        ]

    return FieldlineRolloutArrays(
        history_dynamic=history_dynamic,
        history_global_context=history_global_context,
        history_boundary_context=history_boundary_context,
        target_dynamic_seq=target_dynamic_seq,
        target_delta_seq=target_delta_seq,
        target_dynamic_seq_physical=target_dynamic_seq_physical,
        target_global_context_seq=target_global_context_seq,
        target_boundary_context_seq=target_boundary_context_seq,
        input_time=input_time,
        target_times=target_times,
    )


def compute_window_normalization(
    arrays: FieldlineWindowArrays,
    eps: float = 1.0e-6,
) -> FieldlineWindowNormalization:
    """Compute normalization statistics from training windows only.

    Physical interpretation:
    - Training-only normalization prevents leakage from the validation split.
    - Delta normalization supports residual next-step prediction where shell changes are usually
      smaller and better conditioned than absolute shell values.
    """
    input_mean = np.mean(arrays.input_features, axis=(0, 1)).astype(np.float32)
    input_std = np.std(arrays.input_features, axis=(0, 1)).astype(np.float32)
    context_mean = np.mean(arrays.input_global_context, axis=0).astype(np.float32)
    context_std = np.std(arrays.input_global_context, axis=0).astype(np.float32)
    boundary_context_mean = np.mean(arrays.input_boundary_context, axis=(0, 1)).astype(np.float32)
    boundary_context_std = np.std(arrays.input_boundary_context, axis=(0, 1)).astype(np.float32)
    dynamic_mean = np.mean(arrays.target_dynamic, axis=(0, 1)).astype(np.float32)
    dynamic_std = np.std(arrays.target_dynamic, axis=(0, 1)).astype(np.float32)
    delta_mean = np.mean(arrays.target_delta, axis=(0, 1)).astype(np.float32)
    delta_std = np.std(arrays.target_delta, axis=(0, 1)).astype(np.float32)

    # Zero-variance channels correspond to fixed geometry or inactive shells and should remain finite.
    input_std = np.maximum(input_std, eps)
    context_std = np.maximum(context_std, eps)
    boundary_context_std = np.maximum(boundary_context_std, eps)
    dynamic_std = np.maximum(dynamic_std, eps)
    delta_std = np.maximum(delta_std, eps)

    return FieldlineWindowNormalization(
        input_mean=input_mean,
        input_std=input_std,
        context_mean=context_mean,
        context_std=context_std,
        boundary_context_mean=boundary_context_mean,
        boundary_context_std=boundary_context_std,
        dynamic_mean=dynamic_mean,
        dynamic_std=dynamic_std,
        delta_mean=delta_mean,
        delta_std=delta_std,
    )
