"""Topology and data-path tests for the PWOM surrogate v2 scaffold.

Physical interpretation:
- These tests verify that coarse shell graphs can be built consistently from PWOM-style snapshot
  data.
- They do not claim any physical accuracy yet; they only validate the data contract.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from pwom_surrogate.config import ModelConfig
from pwom_surrogate.fieldline_graph_data import (
    BINARY_EXPECTED_HEADER,
    BINARY_ROW_DTYPE,
    build_boundary_source_node_mask,
    build_lowest_shell_node_mask,
    TARGET_SCHEMA_CLOSURE_TRANSPORT,
    DYNAMIC_CONTEXT_CHANNEL_INDICES,
    DYNAMIC_DIAGNOSTIC_FEATURE_NAMES,
    DYNAMIC_STATE_CHANNEL_INDICES,
    FieldlineGraphLayout,
    FieldlineRolloutDataset,
    POPULATION_FRACTION_CHANNEL_INDEX,
    TRANSPORT_CONSERVATIVE_CHANNEL_INDICES,
    aggregate_snapshot_dynamic_features,
    available_dynamic_diagnostic_feature_names,
    build_target_dynamic_view,
    build_transport_interfaces,
    build_velocity_species_weight_metadata,
    build_fieldline_edge_index,
    build_fieldline_rollout_arrays,
    build_fieldline_window_arrays,
    build_static_node_features,
    compute_window_normalization,
    describe_fieldline_graph,
    occupied_shell_mask,
    extract_dynamic_diagnostic_channel,
    FieldlineWindowDataset,
    inverse_transform_dynamic_features,
    load_fieldline_graph_series,
    transform_dynamic_features,
)
from pwom_surrogate.fieldline_graph_model import FieldlineGraphForecaster
from train_fieldline_graph import masked_mean_vpar_diagnostic_loss
from train_fieldline_graph import masked_mean_vperp_diagnostic_loss
from train_fieldline_graph import support_band_source_suppression_loss
from train_fieldline_graph import support_outflow_saturation_loss


def test_layout_counts_include_global_node() -> None:
    """The total node count should include the optional global forcing node."""
    layout = FieldlineGraphLayout(species_ids=(1, 6), num_altitude_shells=4, include_global_node=True)
    assert layout.num_species == 2
    assert layout.num_shell_nodes == 8
    assert layout.num_nodes == 9
    assert layout.global_node_index == 8


def test_edge_index_has_expected_shape() -> None:
    """The edge list should be a standard `[2, E]` integer array."""
    layout = FieldlineGraphLayout(species_ids=(1, 6), num_altitude_shells=3, include_global_node=True)
    edge_index = build_fieldline_edge_index(layout=layout, include_species_coupling_edges=True)
    assert edge_index.shape[0] == 2
    assert edge_index.shape[1] > 0


def test_transport_interfaces_match_same_species_shell_neighbors() -> None:
    """Transport interfaces should connect each same-species adjacent shell pair once."""
    layout = FieldlineGraphLayout(species_ids=(1, 6), num_altitude_shells=4, include_global_node=True)
    shell_edges_km = np.asarray([100.0, 300.0, 700.0, 1500.0, 3100.0], dtype=np.float32)
    interfaces = build_transport_interfaces(layout=layout, shell_edges_km=shell_edges_km)

    # Two species with three interfaces each should produce six conservative interfaces total.
    assert interfaces.num_interfaces == 6
    assert interfaces.interface_features.shape == (6, 3)
    # The global node does not participate in shell-to-shell transport interfaces.
    assert np.max(interfaces.right_node_index) < layout.num_shell_nodes


def test_boundary_and_lowest_shell_masks_mark_only_species_base_shells() -> None:
    """Boundary-source masks should activate only the lowest shell of each species stack."""
    layout = FieldlineGraphLayout(species_ids=(1, 6, 7), num_altitude_shells=4, include_global_node=True)
    lowest_shell_mask = build_lowest_shell_node_mask(layout=layout)
    boundary_source_mask = build_boundary_source_node_mask(layout=layout)

    assert int(np.sum(lowest_shell_mask)) == layout.num_species
    assert np.array_equal(lowest_shell_mask, boundary_source_mask)
    assert lowest_shell_mask[0] == 1.0
    assert lowest_shell_mask[4] == 1.0
    assert lowest_shell_mask[8] == 1.0
    assert lowest_shell_mask[layout.global_node_index] == 0.0


def test_topology_summary_reports_same_node_count() -> None:
    """The summary helper should agree with the layout arithmetic."""
    layout = FieldlineGraphLayout(species_ids=(1, 6, 7), num_altitude_shells=5, include_global_node=True)
    summary = describe_fieldline_graph(layout=layout, include_species_coupling_edges=True)
    assert summary["num_nodes"] == layout.num_nodes
    assert summary["num_shell_nodes"] == layout.num_shell_nodes


def test_static_and_dynamic_features_have_expected_shape() -> None:
    """One synthetic snapshot should map to `[num_nodes, feature_dim]` shell descriptors."""
    layout = FieldlineGraphLayout(species_ids=(1, 6), num_altitude_shells=4, include_global_node=True)
    shell_edges_km = np.linspace(100.0, 500.0, num=5, dtype=np.float32)

    rows = np.zeros(4, dtype=BINARY_ROW_DTYPE)
    rows["species"] = np.asarray([1, 1, 6, 6], dtype=np.int32)
    rows["altitude"] = np.asarray([1.1e7, 1.9e7, 2.8e7, 3.6e7], dtype=np.float64)
    rows["vpar"] = np.asarray([10.0, 12.0, 20.0, 22.0], dtype=np.float64)
    rows["vperp"] = np.asarray([2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    rows["weight"] = np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    rows["b"] = np.asarray([5.0, 5.5, 6.0, 6.5], dtype=np.float64)
    rows["e"] = np.asarray([7.0, 7.5, 8.0, 8.5], dtype=np.float64)
    rows["g"] = np.asarray([9.0, 9.5, 10.0, 10.5], dtype=np.float64)
    rows["is_open"] = np.asarray([1, 1, 0, 0], dtype=np.uint8)

    static = build_static_node_features(layout=layout, shell_edges_km=shell_edges_km)
    dynamic = aggregate_snapshot_dynamic_features(rows=rows, layout=layout, shell_edges_km=shell_edges_km)

    assert static.shape == (layout.num_nodes, 4)
    assert dynamic.shape == (layout.num_nodes, 12)
    assert float(dynamic[:, 0].sum()) >= 1.0
    assert len(DYNAMIC_STATE_CHANNEL_INDICES) + len(DYNAMIC_CONTEXT_CHANNEL_INDICES) == dynamic.shape[-1]


def test_velocity_transform_roundtrips() -> None:
    """Velocity-moment transforms should be invertible up to float tolerance."""
    dynamic_physical = np.asarray(
        [
            [0.1, 3.0, 0.4, 3000.0, -2.5e4, 1.8e4, 50.0, 1.1e4, 7.0e3, 2.0e4, 1.0e-9, 400.0],
            [0.2, 4.0, 0.7, 5000.0, 8.0e4, 4.6e4, 60.0, 3.0e4, 1.8e4, 2.2e4, 2.0e-9, 420.0],
        ],
        dtype=np.float32,
    )
    dynamic_model = transform_dynamic_features(dynamic_physical=dynamic_physical)
    dynamic_roundtrip = inverse_transform_dynamic_features(dynamic_model=dynamic_model)
    assert np.allclose(dynamic_roundtrip, dynamic_physical, rtol=1.0e-5, atol=1.0e-4)


def test_derived_velocity_diagnostics_recover_shell_means() -> None:
    """Density-weighted velocity channels should recover physical shell means for diagnostics."""
    dynamic_physical = np.asarray(
        [
            [0.2, 3.0, 0.4, 3000.0, -5.0e4, 3.0e4, 50.0, 2.0e4, 1.0e4, 2.0e4, 1.0e-9, 400.0],
            [0.0, 0.0, 0.0, 3500.0, 1.5e4, 1.0e4, 40.0, 5.0e3, 4.0e3, 1.8e4, 0.8e-9, 390.0],
        ],
        dtype=np.float32,
    )

    mean_vpar = extract_dynamic_diagnostic_channel(dynamic_physical=dynamic_physical, feature_name="mean_vpar")
    mean_vperp = extract_dynamic_diagnostic_channel(dynamic_physical=dynamic_physical, feature_name="mean_vperp")
    std_vpar = extract_dynamic_diagnostic_channel(dynamic_physical=dynamic_physical, feature_name="std_vpar")

    assert np.isclose(mean_vpar[0], -2.5e5)
    assert np.isclose(mean_vperp[0], 1.5e5)
    assert np.isclose(std_vpar[0], 1.0e5)
    assert mean_vpar[1] == 0.0


def test_closure_transport_schema_recovers_primitive_velocity_diagnostics() -> None:
    """Closure transport targets should still recover primitive shell diagnostics correctly."""
    dynamic_population_weighted = np.asarray(
        [
            [0.25, 3.0, 0.4, 3000.0, -5.0e4, 2.5e4, 50.0, 2.0e4, 1.5e4, 2.0e4, 1.0e-9, 400.0],
            [0.0, 0.0, 0.0, 3500.0, 1.5e4, 1.0e4, 40.0, 5.0e3, 4.0e3, 1.8e4, 0.8e-9, 390.0],
        ],
        dtype=np.float32,
    )
    dynamic_closure = build_target_dynamic_view(
        dynamic_physical=dynamic_population_weighted,
        target_schema=TARGET_SCHEMA_CLOSURE_TRANSPORT,
    )

    mean_vpar = extract_dynamic_diagnostic_channel(
        dynamic_physical=dynamic_closure,
        feature_name="mean_vpar",
        target_schema=TARGET_SCHEMA_CLOSURE_TRANSPORT,
    )
    std_vpar = extract_dynamic_diagnostic_channel(
        dynamic_physical=dynamic_closure,
        feature_name="std_vpar",
        target_schema=TARGET_SCHEMA_CLOSURE_TRANSPORT,
    )

    # The closure schema stores momentum and energy proxies, but the recovered primitive
    # diagnostics should match the original shell mean and spread where support is nonzero.
    assert np.isclose(mean_vpar[0], -2.0e5)
    assert np.isclose(std_vpar[0], 8.0e4)
    assert mean_vpar[1] == 0.0


def test_occupied_shell_mask_uses_population_threshold() -> None:
    """Occupied-shell masking should follow shell population support, not velocity magnitude."""
    dynamic_state = np.asarray(
        [
            [2.0e-3, 1.0, 0.5, 1000.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 0.0, 70.0],
            [5.0e-4, 1.0, 0.5, 1000.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 0.0, 70.0],
        ],
        dtype=np.float32,
    )
    valid_mask = occupied_shell_mask(dynamic_state=dynamic_state, population_threshold=1.0e-3)
    assert valid_mask.tolist() == [True, False]


def test_diagnostic_feature_registry_includes_derived_velocity_names() -> None:
    """Diagnostic feature listing should expose both stored and recovered shell moments."""
    available = available_dynamic_diagnostic_feature_names()
    assert available == DYNAMIC_DIAGNOSTIC_FEATURE_NAMES
    assert "population_weighted_mean_vpar" in available
    assert "mean_vpar" in available


def test_fieldline_model_supports_split_decoder_heads() -> None:
    """The graph forecaster should preserve output shape across decoder factorizations."""
    edge_index = torch.as_tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    node_features = torch.randn(3, 12, dtype=torch.float32)

    single_head_cfg = ModelConfig(
        node_input_dim=12,
        node_output_dim=12,
        hidden_dim=32,
        message_passing_layers=2,
        mlp_layers=2,
        predict_delta_state=True,
        split_decoder_heads=False,
    )
    split_head_cfg = ModelConfig(
        node_input_dim=12,
        node_output_dim=12,
        hidden_dim=32,
        message_passing_layers=2,
        mlp_layers=2,
        predict_delta_state=True,
        split_decoder_heads=True,
    )
    factorized_head_cfg = ModelConfig(
        node_input_dim=12,
        node_output_dim=12,
        hidden_dim=32,
        message_passing_layers=2,
        mlp_layers=2,
        predict_delta_state=True,
        split_decoder_heads=True,
        factorized_decoder_heads=True,
    )

    single_head_model = FieldlineGraphForecaster(cfg=single_head_cfg)
    split_head_model = FieldlineGraphForecaster(cfg=split_head_cfg)
    factorized_head_model = FieldlineGraphForecaster(cfg=factorized_head_cfg)

    single_head_output = single_head_model(node_features=node_features, edge_index=edge_index)
    split_head_output = split_head_model(node_features=node_features, edge_index=edge_index)
    factorized_head_output = factorized_head_model(node_features=node_features, edge_index=edge_index)

    assert single_head_output.shape == (3, 12)
    assert split_head_output.shape == (3, 12)
    assert factorized_head_output.shape == (3, 12)


def test_fieldline_model_supports_conditional_momentum_head() -> None:
    """Conditional momentum mode should preserve output shape with occupancy-aware decoding."""
    edge_index = torch.as_tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    node_features = torch.randn(3, 12, dtype=torch.float32)
    last_dynamic = torch.randn(3, 12, dtype=torch.float32)
    stats_torch = {
        # Unit statistics make the conditional decode path easy to validate without coupling the
        # test to any specific normalization values from a dataset.
        "dynamic_mean": torch.zeros(12, dtype=torch.float32),
        "dynamic_std": torch.ones(12, dtype=torch.float32),
        "delta_mean": torch.zeros(12, dtype=torch.float32),
        "delta_std": torch.ones(12, dtype=torch.float32),
    }

    conditional_cfg = ModelConfig(
        node_input_dim=12,
        node_output_dim=12,
        hidden_dim=32,
        message_passing_layers=2,
        mlp_layers=2,
        predict_delta_state=True,
        split_decoder_heads=False,
        conditional_momentum_head=True,
    )

    conditional_model = FieldlineGraphForecaster(cfg=conditional_cfg)
    conditional_output = conditional_model(
        node_features=node_features,
        edge_index=edge_index,
        last_dynamic=last_dynamic,
        stats_torch=stats_torch,
        predict_delta_state=True,
    )

    assert conditional_output.shape == (3, 12)


def test_fieldline_model_exposes_closure_uncertainty_auxiliary() -> None:
    """Closure-aware mode should expose per-channel log-variance without changing main output shape."""
    edge_index = torch.as_tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    node_features = torch.randn(3, 12, dtype=torch.float32)

    uncertainty_cfg = ModelConfig(
        node_input_dim=12,
        node_output_dim=12,
        hidden_dim=32,
        message_passing_layers=2,
        mlp_layers=2,
        predict_delta_state=True,
        split_decoder_heads=True,
        target_schema="closure_transport",
        uncertainty_aware_closure_head=True,
    )
    uncertainty_model = FieldlineGraphForecaster(cfg=uncertainty_cfg)
    output, aux = uncertainty_model.forward_with_aux(node_features=node_features, edge_index=edge_index)

    assert output.shape == (3, 12)
    assert "closure_logvar" in aux
    assert aux["closure_logvar"].shape == (3, 4)


def test_fieldline_model_exposes_flux_transport_auxiliaries() -> None:
    """Flux-head mode should expose interface fluxes and conservative transport updates."""
    edge_index = torch.as_tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    node_features = torch.randn(3, 12, dtype=torch.float32)
    last_dynamic = torch.randn(3, 12, dtype=torch.float32)
    last_dynamic_physical = torch.randn(3, 12, dtype=torch.float32)
    stats_torch = {
        "dynamic_mean": torch.zeros(12, dtype=torch.float32),
        "dynamic_std": torch.ones(12, dtype=torch.float32),
        "delta_mean": torch.zeros(12, dtype=torch.float32),
        "delta_std": torch.ones(12, dtype=torch.float32),
        "asinh_channel_index": torch.as_tensor([4], dtype=torch.long),
        "asinh_channel_scale": torch.as_tensor([1.0e5], dtype=torch.float32),
        "transport_left_index": torch.as_tensor([0, 1], dtype=torch.long),
        "transport_right_index": torch.as_tensor([1, 2], dtype=torch.long),
        "transport_interface_features": torch.zeros((2, 3), dtype=torch.float32),
        "transport_interface_width_km": torch.ones(2, dtype=torch.float32),
        "transport_shell_node_mask": torch.ones(3, dtype=torch.float32),
        "transport_cell_width_km": torch.ones(3, dtype=torch.float32),
        "transport_delta_scale": torch.ones(2, dtype=torch.float32),
    }

    flux_cfg = ModelConfig(
        node_input_dim=12,
        node_output_dim=12,
        hidden_dim=32,
        message_passing_layers=2,
        mlp_layers=2,
        predict_delta_state=True,
        split_decoder_heads=True,
        flux_transport_head=True,
    )
    flux_model = FieldlineGraphForecaster(cfg=flux_cfg)
    output, aux = flux_model.forward_with_aux(
        node_features=node_features,
        edge_index=edge_index,
        last_dynamic=last_dynamic,
        last_dynamic_physical=last_dynamic_physical,
        stats_torch=stats_torch,
        predict_delta_state=True,
    )

    assert output.shape == (3, 12)
    assert "transport_interface_flux_physical" in aux
    assert aux["transport_interface_flux_physical"].shape == (2, 2)
    assert aux["transport_delta_physical"].shape == (3, 2)
    assert aux["residual_conservative_physical"].shape == (3, 2)


def test_redistribution_transport_preserves_support_and_reports_fraction_diagnostics() -> None:
    """Redistribution transport should conserve nonnegative support and expose stay/outflow means."""
    cfg = ModelConfig(
        node_input_dim=12,
        node_output_dim=12,
        hidden_dim=16,
        message_passing_layers=1,
        mlp_layers=2,
        predict_delta_state=True,
        split_decoder_heads=True,
        flux_transport_head=True,
        operator_split_head=True,
        redistribution_transport_head=True,
    )
    model = FieldlineGraphForecaster(cfg=cfg)
    for module in model.transport_flux_decoder.net:
        if isinstance(module, torch.nn.Linear):
            module.weight.data.zero_()
            module.bias.data.zero_()

    node_latent = torch.zeros((3, cfg.hidden_dim), dtype=torch.float32)
    last_dynamic = torch.zeros((3, 12), dtype=torch.float32)
    last_dynamic_physical = torch.zeros((3, 12), dtype=torch.float32)
    support_channel_local_index = TRANSPORT_CONSERVATIVE_CHANNEL_INDICES.index(POPULATION_FRACTION_CHANNEL_INDEX)
    momentum_channel_index = TRANSPORT_CONSERVATIVE_CHANNEL_INDICES[1 - support_channel_local_index]
    last_dynamic_physical[:, POPULATION_FRACTION_CHANNEL_INDEX] = torch.as_tensor([0.6, 0.3, 0.1], dtype=torch.float32)
    last_dynamic_physical[:, momentum_channel_index] = torch.as_tensor([0.12, -0.03, 0.02], dtype=torch.float32)
    stats_torch = {
        "dynamic_mean": torch.zeros(12, dtype=torch.float32),
        "dynamic_std": torch.ones(12, dtype=torch.float32),
        "delta_mean": torch.zeros(12, dtype=torch.float32),
        "delta_std": torch.ones(12, dtype=torch.float32),
        "asinh_channel_index": torch.as_tensor([momentum_channel_index], dtype=torch.long),
        "asinh_channel_scale": torch.as_tensor([1.0e5], dtype=torch.float32),
        "transport_delta_scale": torch.ones(2, dtype=torch.float32),
    }

    conservative_pred_norm, aux = model._build_flux_transport_update(
        node_latent=node_latent,
        last_dynamic=last_dynamic,
        last_dynamic_physical=last_dynamic_physical,
        stats_torch=stats_torch,
        transport_left_index=torch.as_tensor([0, 1], dtype=torch.long),
        transport_right_index=torch.as_tensor([1, 2], dtype=torch.long),
        transport_interface_features=torch.zeros((2, 3), dtype=torch.float32),
        transport_shell_node_mask=torch.ones(3, dtype=torch.float32),
    )

    next_support = aux["next_conservative_physical"][:, support_channel_local_index]
    assert conservative_pred_norm.shape == (3, 2)
    assert torch.all(next_support >= 0.0)
    assert torch.isclose(torch.sum(next_support), torch.tensor(1.0, dtype=torch.float32), atol=1.0e-6)
    assert torch.allclose(
        aux["transport_interface_flux_physical"][:, 1],
        torch.zeros(2, dtype=torch.float32),
        atol=1.0e-6,
    )
    assert torch.isclose(aux["support_stay_fraction_mean"], torch.tensor(0.75, dtype=torch.float32), atol=1.0e-6)
    assert torch.isclose(aux["support_outflow_fraction_mean"], torch.tensor(0.25, dtype=torch.float32), atol=1.0e-6)


def test_redistribution_transport_caps_total_outflow_fraction_by_config() -> None:
    """Redistribution transport should never exceed the configured one-step outflow cap."""
    cfg = ModelConfig(
        node_input_dim=12,
        node_output_dim=12,
        hidden_dim=16,
        message_passing_layers=1,
        mlp_layers=2,
        predict_delta_state=True,
        split_decoder_heads=True,
        flux_transport_head=True,
        operator_split_head=True,
        redistribution_transport_head=True,
        max_support_outflow_fraction=0.5,
    )
    model = FieldlineGraphForecaster(cfg=cfg)
    linear_layers = [module for module in model.transport_flux_decoder.net if isinstance(module, torch.nn.Linear)]
    for module in linear_layers:
        module.weight.data.zero_()
        module.bias.data.zero_()
    linear_layers[-1].bias.data[2] = 100.0

    node_latent = torch.zeros((3, cfg.hidden_dim), dtype=torch.float32)
    last_dynamic = torch.zeros((3, 12), dtype=torch.float32)
    last_dynamic_physical = torch.zeros((3, 12), dtype=torch.float32)
    momentum_channel_index = TRANSPORT_CONSERVATIVE_CHANNEL_INDICES[1]
    last_dynamic_physical[:, POPULATION_FRACTION_CHANNEL_INDEX] = torch.as_tensor([0.6, 0.3, 0.1], dtype=torch.float32)
    last_dynamic_physical[:, momentum_channel_index] = torch.as_tensor([0.12, -0.03, 0.02], dtype=torch.float32)
    stats_torch = {
        "dynamic_mean": torch.zeros(12, dtype=torch.float32),
        "dynamic_std": torch.ones(12, dtype=torch.float32),
        "delta_mean": torch.zeros(12, dtype=torch.float32),
        "delta_std": torch.ones(12, dtype=torch.float32),
        "asinh_channel_index": torch.as_tensor([momentum_channel_index], dtype=torch.long),
        "asinh_channel_scale": torch.as_tensor([1.0e5], dtype=torch.float32),
        "transport_delta_scale": torch.ones(2, dtype=torch.float32),
    }

    _, aux = model._build_flux_transport_update(
        node_latent=node_latent,
        last_dynamic=last_dynamic,
        last_dynamic_physical=last_dynamic_physical,
        stats_torch=stats_torch,
        transport_left_index=torch.as_tensor([0, 1], dtype=torch.long),
        transport_right_index=torch.as_tensor([1, 2], dtype=torch.long),
        transport_interface_features=torch.zeros((2, 3), dtype=torch.float32),
        transport_shell_node_mask=torch.ones(3, dtype=torch.float32),
    )

    assert torch.isclose(aux["support_outflow_fraction_mean"], torch.tensor(0.5, dtype=torch.float32), atol=1.0e-6)
    assert torch.isclose(aux["support_stay_fraction_mean"], torch.tensor(0.5, dtype=torch.float32), atol=1.0e-6)


def test_support_band_source_suppression_penalizes_source_dominance_only_in_band() -> None:
    """Support-band suppression should ignore out-of-band nodes and only penalize excess source."""
    source_conservative = torch.as_tensor(
        [[4.0, 1.0], [0.5, 0.5], [5.0, 5.0]],
        dtype=torch.float32,
    )
    transport_conservative = torch.as_tensor(
        [[1.0, 1.0], [1.0, 1.0], [0.0, 0.0]],
        dtype=torch.float32,
    )
    stats_torch = {
        "support_band_node_mask": torch.as_tensor([1.0, 1.0, 0.0], dtype=torch.float32),
        "transport_delta_scale": torch.ones(2, dtype=torch.float32),
    }

    loss = support_band_source_suppression_loss(
        source_conservative_physical=source_conservative,
        transport_delta_physical=transport_conservative,
        stats_torch=stats_torch,
    )

    assert torch.isclose(loss, torch.tensor(1.5, dtype=torch.float32), atol=1.0e-6)


def test_support_outflow_saturation_loss_activates_only_near_cap() -> None:
    """Outflow saturation loss should stay zero below the threshold and activate near the cap."""
    support_outflow_fraction_per_node = torch.as_tensor([0.45, 0.50, 0.20], dtype=torch.float32)
    stats_torch = {
        "transport_shell_node_mask": torch.as_tensor([1.0, 1.0, 1.0], dtype=torch.float32),
    }

    loss = support_outflow_saturation_loss(
        support_outflow_fraction_per_node=support_outflow_fraction_per_node,
        stats_torch=stats_torch,
        max_support_outflow_fraction=0.5,
    )

    assert torch.isclose(loss, torch.tensor(1.0 / 3.0, dtype=torch.float32), atol=1.0e-6)


def test_fieldline_model_supports_boundary_injection_and_film_context() -> None:
    """Boundary/global-control branches should preserve output shape and expose boundary updates."""
    edge_index = torch.as_tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    node_features = torch.randn(3, 12, dtype=torch.float32)
    last_dynamic = torch.randn(3, 12, dtype=torch.float32)
    global_context = torch.randn(3, 2, dtype=torch.float32)
    stats_torch = {
        "boundary_source_node_mask": torch.as_tensor([1.0, 0.0, 0.0], dtype=torch.float32),
    }

    boundary_cfg = ModelConfig(
        node_input_dim=12,
        global_context_dim=2,
        node_output_dim=12,
        hidden_dim=32,
        message_passing_layers=2,
        mlp_layers=2,
        predict_delta_state=True,
        split_decoder_heads=True,
        global_context_film=True,
        boundary_injection_head=True,
    )
    boundary_model = FieldlineGraphForecaster(cfg=boundary_cfg)
    output, aux = boundary_model.forward_with_aux(
        node_features=node_features,
        edge_index=edge_index,
        global_context=global_context,
        last_dynamic=last_dynamic,
        stats_torch=stats_torch,
        predict_delta_state=True,
    )

    assert output.shape == (3, 12)
    assert "boundary_injection_update" in aux
    assert aux["boundary_injection_update"].shape == (3, 12)
    assert torch.allclose(aux["boundary_injection_update"][1:], torch.zeros_like(aux["boundary_injection_update"][1:]))


def test_velocity_species_weight_metadata_upweights_sparse_species() -> None:
    """Rare species should receive larger velocity-loss weights than dominant species."""
    layout = FieldlineGraphLayout(species_ids=(1, 2), num_altitude_shells=2, include_global_node=True)
    dynamic_node_features = np.zeros((2, layout.num_nodes, 12), dtype=np.float32)

    # Species 1 dominates shell occupancy while species 2 remains consistently sparse.
    dynamic_node_features[:, 0, 0] = 0.45
    dynamic_node_features[:, 1, 0] = 0.35
    dynamic_node_features[:, 2, 0] = 0.12
    dynamic_node_features[:, 3, 0] = 0.08
    dynamic_node_features[:, layout.global_node_index, 0] = 1.0

    metadata = build_velocity_species_weight_metadata(
        layout=layout,
        dynamic_node_features=dynamic_node_features,
        balance_power=0.5,
        max_weight=4.0,
    )

    assert metadata["velocity_species_weight_by_species"][1] > metadata["velocity_species_weight_by_species"][0]
    assert metadata["velocity_species_weight_per_node"][2] == metadata["velocity_species_weight_by_species"][1]
    assert metadata["velocity_species_weight_per_node"][layout.global_node_index] == 1.0


def test_masked_mean_vpar_diagnostic_loss_ignores_empty_shells() -> None:
    """The auxiliary mean_vpar loss should only score occupied shells."""
    pred_dynamic_physical = torch.tensor(
        [
            [0.20, 3.0, 0.5, 1000.0, 4.0e4, 2.0e4, 50.0, 1.0e4, 8.0e3, 1.0, 0.0, 1.0],
            [1.0e-5, 3.0, 0.5, 1000.0, 9.9e8, 2.0e4, 50.0, 1.0e4, 8.0e3, 1.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    target_dynamic_physical = torch.tensor(
        [
            [0.20, 3.0, 0.5, 1000.0, 2.0e4, 2.0e4, 50.0, 1.0e4, 8.0e3, 1.0, 0.0, 1.0],
            [1.0e-5, 3.0, 0.5, 1000.0, -9.9e8, 2.0e4, 50.0, 1.0e4, 8.0e3, 1.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    loss, valid_fraction = masked_mean_vpar_diagnostic_loss(
        pred_dynamic_physical=pred_dynamic_physical,
        target_dynamic_physical=target_dynamic_physical,
        occupied_shell_threshold=1.0e-3,
        diagnostic_mean_vpar_scale=1.0e5,
    )
    # Only the first shell is occupied, so the second row's huge mismatch must be ignored.
    assert torch.isclose(valid_fraction, torch.tensor(0.5))
    assert float(loss) > 0.0
    assert float(loss) < 10.0


def test_masked_mean_vperp_diagnostic_loss_ignores_empty_shells() -> None:
    """The auxiliary mean_vperp loss should only score occupied shells."""
    pred_dynamic_physical = torch.tensor(
        [
            [0.20, 3.0, 0.5, 1000.0, 4.0e4, 3.0e4, 50.0, 1.0e4, 8.0e3, 1.0, 0.0, 1.0],
            [1.0e-5, 3.0, 0.5, 1000.0, 4.0e4, 9.9e8, 50.0, 1.0e4, 8.0e3, 1.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    target_dynamic_physical = torch.tensor(
        [
            [0.20, 3.0, 0.5, 1000.0, 4.0e4, 1.0e4, 50.0, 1.0e4, 8.0e3, 1.0, 0.0, 1.0],
            [1.0e-5, 3.0, 0.5, 1000.0, 4.0e4, -9.9e8, 50.0, 1.0e4, 8.0e3, 1.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    loss, valid_fraction = masked_mean_vperp_diagnostic_loss(
        pred_dynamic_physical=pred_dynamic_physical,
        target_dynamic_physical=target_dynamic_physical,
        occupied_shell_threshold=1.0e-3,
        diagnostic_mean_vperp_scale=1.0e5,
    )
    assert torch.isclose(valid_fraction, torch.tensor(0.5))
    assert float(loss) > 0.0
    assert float(loss) < 10.0


def _write_test_snapshot(handle, model_time: float, rows: np.ndarray) -> None:
    """Write one tiny PWOM-style binary snapshot to a temporary file."""
    header_bytes = BINARY_EXPECTED_HEADER.encode("ascii")

    # The binary layout mirrors the production PWOM particle-history file.
    handle.write(np.asarray([len(header_bytes)], dtype="<i4").tobytes())
    handle.write(header_bytes)
    handle.write(np.asarray([1], dtype="<i8").tobytes())
    handle.write(np.asarray([model_time], dtype="<f8").tobytes())
    handle.write(np.asarray([rows.shape[0]], dtype="<i4").tobytes())
    rows.astype(BINARY_ROW_DTYPE, copy=False).tofile(handle)


def test_load_fieldline_graph_series_from_binary_split(tmp_path: Path) -> None:
    """A tiny PWOM-style binary split should load into a two-step graph series."""
    train_path = tmp_path / "train.dat"

    rows0 = np.zeros(4, dtype=BINARY_ROW_DTYPE)
    rows0["particle_id"] = np.asarray([0, 1, 2, 3], dtype=np.int32)
    rows0["species"] = np.asarray([1, 1, 6, 6], dtype=np.int32)
    rows0["altitude"] = np.asarray([1.0e7, 1.5e7, 2.0e7, 2.5e7], dtype=np.float64)
    rows0["vpar"] = np.asarray([100.0, 110.0, 120.0, 130.0], dtype=np.float64)
    rows0["vperp"] = np.asarray([10.0, 11.0, 12.0, 13.0], dtype=np.float64)
    rows0["weight"] = np.asarray([1.0, 1.5, 2.0, 2.5], dtype=np.float64)
    rows0["b"] = np.asarray([5.0, 5.1, 5.2, 5.3], dtype=np.float64)
    rows0["e"] = np.asarray([6.0, 6.1, 6.2, 6.3], dtype=np.float64)
    rows0["g"] = np.asarray([7.0, 7.1, 7.2, 7.3], dtype=np.float64)
    rows0["is_open"] = np.asarray([1, 1, 0, 0], dtype=np.uint8)

    rows1 = rows0.copy()
    rows1["altitude"] += 1.0e6
    rows1["vpar"] += 5.0

    with train_path.open("wb") as handle:
        _write_test_snapshot(handle=handle, model_time=0.0, rows=rows0)
        _write_test_snapshot(handle=handle, model_time=1.0, rows=rows1)

    series = load_fieldline_graph_series(
        data_dir=str(tmp_path),
        split="train",
        num_particles=4,
        num_altitude_shells=4,
        include_global_node=True,
        include_species_coupling_edges=True,
        species_ids=(1, 6),
        verbose=False,
    )

    assert series.num_timesteps == 2
    assert series.layout.num_species == 2
    assert series.node_input_dim == 16
    assert series.node_output_dim == 12
    assert series.edge_index.shape[0] == 2

    arrays = build_fieldline_window_arrays(series=series, window_length=1)
    stats = compute_window_normalization(arrays=arrays)
    dataset = FieldlineWindowDataset(arrays=arrays, normalization=stats, predict_delta_state=True)
    sample = dataset[0]

    assert arrays.num_samples == 1
    assert sample["input_features"].shape[-1] == 16
    assert sample["target_norm"].shape[-1] == 12
    assert sample["target_dynamic_physical"].shape[-1] == 12

    rollout_arrays = build_fieldline_rollout_arrays(series=series, window_length=1, horizon=1)
    rollout_dataset = FieldlineRolloutDataset(
        static_node_features=series.static_node_features,
        arrays=rollout_arrays,
        normalization=stats,
        predict_delta_state=True,
    )
    rollout_sample = rollout_dataset[0]

    assert rollout_arrays.num_samples == 1
    assert rollout_sample["history_dynamic"].shape[0] == 1
    assert rollout_sample["target_dynamic_seq"].shape[0] == 1
    assert rollout_sample["target_dynamic_seq_physical"].shape[0] == 1
