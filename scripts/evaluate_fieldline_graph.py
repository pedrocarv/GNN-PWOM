#!/usr/bin/env python3
"""Evaluate a trained one-step field-line graph surrogate for PWOM.

Physical interpretation:
- Evaluation uses the saved coarse shell layout and normalization statistics from training so the
  validation split is scored in the same physical state space.
- Metrics are reported both in normalized training space and in raw shell-state units.
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
    TARGET_SCHEMA_CLOSURE_TRANSPORT,
    available_dynamic_diagnostic_feature_names,
    FieldlineWindowDataset,
    FieldlineWindowNormalization,
    build_boundary_source_node_mask,
    build_transport_cell_width_per_node,
    build_transport_interfaces,
    build_fieldline_window_arrays,
    dynamic_diagnostic_valid_mask,
    extract_dynamic_diagnostic_channel,
    load_fieldline_graph_series,
)
from pwom_surrogate.fieldline_graph_model import FieldlineGraphForecaster
from train_fieldline_graph import (
    _stats_to_torch,
    apply_dynamic_constraints_model_space,
    build_support_band_node_mask,
    closure_transport_one_step_loss,
    compute_effective_diagnostic_weight,
    conditional_one_step_loss,
    boundary_source_loss,
    flux_transport_one_step_loss,
    inverse_transform_dynamic_torch,
    masked_mean_vpar_diagnostic_loss,
    masked_mean_vperp_diagnostic_loss,
    masked_std_vpar_diagnostic_loss,
    occupied_fraction_metric,
    residual_conservative_loss,
    reconstruct_prediction_raw,
    select_device,
    support_band_loss,
    support_band_source_suppression_loss,
    support_outflow_saturation_loss,
    transport_flux_consistency_loss,
    weighted_normalized_dynamic_mse,
)


def parse_args() -> argparse.Namespace:
    """Parse evaluator CLI arguments."""
    parser = argparse.ArgumentParser(description="Evaluate a trained PWOM surrogate one-step checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-dir", type=str, default="../pwom-surrogate-rebuild/dataset")
    parser.add_argument("--split", choices=["train", "valid"], default="valid")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=["cpu", "mps", "cuda", "auto"], default="auto")
    parser.add_argument("--output-json", type=str, default="")
    parser.add_argument("--verbose-data", action="store_true")
    return parser.parse_args()

def main() -> None:
    """Load a checkpoint and report one-step validation metrics."""
    args = parse_args()
    device = select_device(args.device)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    checkpoint_dir = os.path.dirname(os.path.abspath(args.checkpoint))
    normalization = FieldlineWindowNormalization.load_json(os.path.join(checkpoint_dir, "normalization.json"))

    model_cfg = ModelConfig(**checkpoint["model_cfg"])
    train_cfg = checkpoint["train_cfg"]
    species_ids = tuple(int(species_id) for species_id in checkpoint["species_ids"])
    shell_edges_km = checkpoint["shell_edges_km"]
    data_cfg = checkpoint["data_cfg"]

    series = load_fieldline_graph_series(
        data_dir=args.data_dir,
        split=args.split,
        num_particles=data_cfg["num_particles"],
        num_altitude_shells=data_cfg["num_altitude_shells"],
        include_global_node=data_cfg["include_global_node"],
        include_species_coupling_edges=data_cfg["include_species_coupling_edges"],
        species_ids=species_ids,
        shell_edges_km_override=shell_edges_km,
        verbose=args.verbose_data,
    )
    target_schema = data_cfg.get("target_schema", model_cfg.target_schema)
    arrays = build_fieldline_window_arrays(
        series=series,
        window_length=data_cfg["window_length"],
        target_schema=target_schema,
    )
    dataset = FieldlineWindowDataset(
        arrays=arrays,
        normalization=normalization,
        predict_delta_state=model_cfg.predict_delta_state,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = FieldlineGraphForecaster(cfg=model_cfg).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    edge_index = torch.as_tensor(series.edge_index, dtype=torch.long, device=device)
    transport_interfaces = build_transport_interfaces(
        layout=series.layout,
        shell_edges_km=series.shell_edges_km,
    )
    transport_cell_width_km = checkpoint.get("transport_cell_width_km")
    if transport_cell_width_km is None:
        transport_cell_width_km = build_transport_cell_width_per_node(
            layout=series.layout,
            shell_edges_km=series.shell_edges_km,
        )
    transport_shell_node_mask = checkpoint.get("transport_shell_node_mask")
    if transport_shell_node_mask is None:
        transport_shell_node_mask = np.zeros(series.layout.num_nodes, dtype=np.float32)
        transport_shell_node_mask[: series.layout.num_shell_nodes] = 1.0
    stats_torch = _stats_to_torch(
        stats=normalization,
        device=device,
        num_nodes=series.layout.num_nodes,
        target_schema=target_schema,
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
        transport_interfaces=transport_interfaces if model_cfg.flux_transport_head else None,
        transport_delta_scale=checkpoint.get("transport_delta_scale"),
        transport_shell_node_mask=transport_shell_node_mask,
        transport_cell_width_km=np.asarray(transport_cell_width_km, dtype=np.float32),
        transport_interface_width_km=np.asarray(
            checkpoint.get(
                "transport_interface_width_km",
                transport_interfaces.interface_width_km,
            ),
            dtype=np.float32,
        ),
    )

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
    pred_batches: list[np.ndarray] = []
    true_batches: list[np.ndarray] = []
    iterator = tqdm(loader, desc=f"eval:{args.split}", dynamic_ncols=True)

    # Standalone checkpoint evaluation should replay the same effective diagnostic weights that
    # were active at the saved epoch so objective comparisons remain apples-to-apples.
    effective_mean_vpar_loss_weight = compute_effective_diagnostic_weight(
        base_weight=float(train_cfg.get("diagnostic_mean_vpar_loss_weight", 0.0)),
        epoch_index=int(checkpoint.get("epoch", 1)),
        warmup_epochs=int(train_cfg.get("diagnostic_loss_warmup_epochs", 0)),
        warmup_start_factor=float(train_cfg.get("diagnostic_loss_warmup_start_factor", 0.0)),
    )
    effective_mean_vperp_loss_weight = compute_effective_diagnostic_weight(
        base_weight=float(train_cfg.get("diagnostic_mean_vperp_loss_weight", 0.0)),
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

    with torch.no_grad():
        for batch in iterator:
            input_features = batch["input_features"].to(device=device, dtype=torch.float32)
            global_context = batch["global_context"].to(device=device, dtype=torch.float32)
            boundary_context = batch["boundary_context"].to(device=device, dtype=torch.float32)
            target_dynamic = batch["target_dynamic"].to(device=device, dtype=torch.float32)
            target_dynamic_physical = batch["target_dynamic_physical"].to(device=device, dtype=torch.float32)
            last_dynamic = batch["last_dynamic"].to(device=device, dtype=torch.float32)
            last_dynamic_physical = batch["last_dynamic_physical"].to(device=device, dtype=torch.float32)

            pred_norm, aux_outputs = model.forward_with_aux(
                node_features=input_features,
                edge_index=edge_index,
                global_context=global_context,
                boundary_context=boundary_context,
                last_dynamic=last_dynamic,
                last_dynamic_physical=last_dynamic_physical,
                stats_torch=stats_torch,
                predict_delta_state=model_cfg.predict_delta_state,
            )
            pred_dynamic = reconstruct_prediction_raw(
                pred_norm=pred_norm,
                last_dynamic=last_dynamic,
                stats_torch=stats_torch,
                predict_delta_state=model_cfg.predict_delta_state,
            )
            pred_dynamic = apply_dynamic_constraints_model_space(
                pred_dynamic=pred_dynamic,
                stats_torch=stats_torch,
            )
            if model_cfg.target_schema == TARGET_SCHEMA_CLOSURE_TRANSPORT and model_cfg.uncertainty_aware_closure_head:
                if "closure_logvar" not in aux_outputs:
                    raise ValueError("uncertainty-aware closure head requires closure_logvar auxiliary output.")
                loss_terms = closure_transport_one_step_loss(
                    pred_dynamic=pred_dynamic,
                    target_dynamic=target_dynamic,
                    closure_logvar=aux_outputs["closure_logvar"],
                    stats_torch=stats_torch,
                    occupied_shell_threshold=float(train_cfg.get("occupied_shell_threshold", 1.0e-3)),
                    closure_loss_weight=float(train_cfg.get("closure_loss_weight", 1.0)),
                )
            elif model_cfg.conditional_momentum_head:
                loss_terms = conditional_one_step_loss(
                    pred_dynamic=pred_dynamic,
                    target_dynamic=target_dynamic,
                    stats_torch=stats_torch,
                    occupied_shell_threshold=float(train_cfg.get("occupied_shell_threshold", 1.0e-3)),
                    momentum_loss_weight=float(train_cfg.get("momentum_loss_weight", 1.0)),
                )
            elif model_cfg.flux_transport_head:
                loss_terms = flux_transport_one_step_loss(
                    pred_dynamic=pred_dynamic,
                    target_dynamic=target_dynamic,
                    stats_torch=stats_torch,
                    conservative_state_loss_weight=float(train_cfg.get("conservative_state_loss_weight", 0.25)),
                    non_conservative_state_loss_weight=float(train_cfg.get("non_conservative_state_loss_weight", 1.0)),
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
                occupied_shell_threshold=float(train_cfg.get("occupied_shell_threshold", 1.0e-3)),
            )
            if "conservative_state_loss" not in loss_terms:
                # Older or non-flux checkpoints do not split state supervision by transported pair.
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
                if float(train_cfg.get("flux_loss_weight", 0.0)) > 0.0:
                    loss = loss + (float(train_cfg.get("flux_loss_weight", 0.0)) * flux_consistency)
                if float(train_cfg.get("residual_conservative_weight", 0.0)) > 0.0:
                    loss = loss + (
                        float(train_cfg.get("residual_conservative_weight", 0.0))
                        * residual_transport
                    )
            if "source_conservative_physical" in aux_outputs:
                source_conservative = residual_conservative_loss(
                    residual_conservative_physical=aux_outputs["source_conservative_physical"],
                    stats_torch=stats_torch,
                )
                if bool(model_cfg.redistribution_transport_head) and "transport_delta_physical" in aux_outputs:
                    support_band_source_suppression = support_band_source_suppression_loss(
                        source_conservative_physical=aux_outputs["source_conservative_physical"],
                        transport_delta_physical=aux_outputs["transport_delta_physical"],
                        stats_torch=stats_torch,
                    )
                if float(train_cfg.get("source_conservative_weight", 0.0)) > 0.0:
                    # Replaying the same source regularizer keeps operator-split checkpoints
                    # comparable to the objective they were actually trained against.
                    loss = loss + (
                        float(train_cfg.get("source_conservative_weight", 0.0))
                        * source_conservative
                    )
                if bool(model_cfg.redistribution_transport_head) and float(train_cfg.get("support_band_source_suppression_weight", 0.0)) > 0.0:
                    loss = loss + (
                        float(train_cfg.get("support_band_source_suppression_weight", 0.0))
                        * support_band_source_suppression
                    )
            if bool(model_cfg.redistribution_transport_head) and "support_outflow_fraction_per_node" in aux_outputs:
                support_outflow_saturation = support_outflow_saturation_loss(
                    support_outflow_fraction_per_node=aux_outputs["support_outflow_fraction_per_node"].squeeze(-1),
                    stats_torch=stats_torch,
                    max_support_outflow_fraction=float(model_cfg.max_support_outflow_fraction),
                )
                if float(train_cfg.get("support_outflow_saturation_weight", 0.0)) > 0.0:
                    loss = loss + (
                        float(train_cfg.get("support_outflow_saturation_weight", 0.0))
                        * support_outflow_saturation
                    )
            focused_boundary_loss = boundary_source_loss(
                pred_dynamic=pred_dynamic,
                target_dynamic=target_dynamic,
                stats_torch=stats_torch,
            )
            if float(train_cfg.get("boundary_source_loss_weight", 0.0)) > 0.0:
                # Evaluation replays the same lower-boundary emphasis so checkpoints are scored
                # under the same localized source objective they were trained with.
                loss = loss + (
                    float(train_cfg.get("boundary_source_loss_weight", 0.0))
                    * focused_boundary_loss
                )
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
            if float(train_cfg.get("support_band_loss_weight", 0.0)) > 0.0:
                # Evaluation should report the same focused objective that training optimized in
                # the diagnosed support-collapse band.
                loss = loss + (float(train_cfg.get("support_band_loss_weight", 0.0)) * support_focus_loss)
            if float(train_cfg.get("secondary_support_band_loss_weight", 0.0)) > 0.0:
                # The evaluation objective includes the weaker upper-band support term when the
                # checkpoint was trained with that residual-altitude correction enabled.
                loss = loss + (
                    float(train_cfg.get("secondary_support_band_loss_weight", 0.0))
                    * secondary_support_focus_loss
                )
            pred_dynamic_physical = inverse_transform_dynamic_torch(
                dynamic_model=pred_dynamic,
                stats_torch=stats_torch,
            )
            diagnostic_mean_vpar_loss, diagnostic_mean_vpar_valid_fraction = masked_mean_vpar_diagnostic_loss(
                pred_dynamic_physical=pred_dynamic_physical,
                target_dynamic_physical=target_dynamic_physical,
                occupied_shell_threshold=float(train_cfg.get("occupied_shell_threshold", 1.0e-3)),
                diagnostic_mean_vpar_scale=float(checkpoint.get("diagnostic_mean_vpar_scale", 1.0)),
                target_schema=target_schema,
            )
            diagnostic_mean_vperp_loss, diagnostic_mean_vperp_valid_fraction = masked_mean_vperp_diagnostic_loss(
                pred_dynamic_physical=pred_dynamic_physical,
                target_dynamic_physical=target_dynamic_physical,
                occupied_shell_threshold=float(train_cfg.get("occupied_shell_threshold", 1.0e-3)),
                diagnostic_mean_vperp_scale=float(checkpoint.get("diagnostic_mean_vperp_scale", 1.0)),
                target_schema=target_schema,
            )
            diagnostic_std_vpar_loss, diagnostic_std_vpar_valid_fraction = masked_std_vpar_diagnostic_loss(
                pred_dynamic_physical=pred_dynamic_physical,
                target_dynamic_physical=target_dynamic_physical,
                occupied_shell_threshold=float(train_cfg.get("occupied_shell_threshold", 1.0e-3)),
                diagnostic_std_vpar_scale=float(checkpoint.get("diagnostic_std_vpar_scale", 1.0)),
                target_schema=target_schema,
            )
            if effective_mean_vpar_loss_weight > 0.0:
                # The reported loss includes the same auxiliary occupied-shell parallel-drift term
                # used during training so checkpoint comparisons remain apples-to-apples.
                loss = loss + (
                    float(effective_mean_vpar_loss_weight)
                    * diagnostic_mean_vpar_loss
                )
            if effective_mean_vperp_loss_weight > 0.0:
                loss = loss + (
                    float(effective_mean_vperp_loss_weight)
                    * diagnostic_mean_vperp_loss
                )
            if effective_std_vpar_loss_weight > 0.0:
                # The same checkpoint-replayed weighting is used for the occupied-shell spread term
                # so the saved objective matches what was optimized during training.
                loss = loss + (
                    float(effective_std_vpar_loss_weight)
                    * diagnostic_std_vpar_loss
                )
            raw_mse = torch.mean(torch.square(pred_dynamic_physical - target_dynamic_physical))
            pred_batches.append(pred_dynamic_physical.detach().cpu().numpy())
            true_batches.append(target_dynamic_physical.detach().cpu().numpy())

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
        raise ValueError("Evaluation dataset is empty.")

    pred_dynamic_all = np.concatenate(pred_batches, axis=0)
    true_dynamic_all = np.concatenate(true_batches, axis=0)
    diagnostic_channel_mse: dict[str, float] = {}
    valid_diagnostic_channel_mse: dict[str, float | None] = {}
    diagnostic_valid_counts: dict[str, int] = {}
    for feature_name in available_dynamic_diagnostic_feature_names(target_schema):
        # Diagnostics recover physical shell means from density-weighted targets so one-step
        # validation still reports interpretable mean-velocity errors.
        true_channel = extract_dynamic_diagnostic_channel(
            dynamic_physical=true_dynamic_all,
            feature_name=feature_name,
            target_schema=target_schema,
        )
        pred_channel = extract_dynamic_diagnostic_channel(
            dynamic_physical=pred_dynamic_all,
            feature_name=feature_name,
            target_schema=target_schema,
        )
        diagnostic_channel_mse[feature_name] = float(np.mean(np.square(pred_channel - true_channel)))
        valid_mask = dynamic_diagnostic_valid_mask(
            dynamic_physical=true_dynamic_all,
            feature_name=feature_name,
            target_schema=target_schema,
            population_epsilon=float(train_cfg.get("occupied_shell_threshold", 1.0e-3)),
        )
        diagnostic_valid_counts[feature_name] = int(np.sum(valid_mask))
        if np.any(valid_mask):
            valid_diagnostic_channel_mse[feature_name] = float(
                np.mean(np.square(pred_channel[valid_mask] - true_channel[valid_mask]))
            )
        else:
            valid_diagnostic_channel_mse[feature_name] = None

    metrics = {
        "checkpoint": os.path.abspath(args.checkpoint),
        "split": args.split,
        "num_samples": total_examples,
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
        "effective_diagnostic_mean_vpar_loss_weight": float(effective_mean_vpar_loss_weight),
        "effective_diagnostic_mean_vperp_loss_weight": float(effective_mean_vperp_loss_weight),
        "effective_diagnostic_std_vpar_loss_weight": float(effective_std_vpar_loss_weight),
        "window_length": int(data_cfg["window_length"]),
        "num_altitude_shells": int(data_cfg["num_altitude_shells"]),
        "num_particles": None if data_cfg["num_particles"] is None else int(data_cfg["num_particles"]),
        "species_ids": list(species_ids),
        "node_input_dim": int(model_cfg.node_input_dim),
        "node_output_dim": int(model_cfg.node_output_dim),
        "diagnostic_channel_mse": diagnostic_channel_mse,
        "diagnostic_channel_mse_valid": valid_diagnostic_channel_mse,
        "diagnostic_valid_counts": diagnostic_valid_counts,
    }

    print("[eval] evaluation metrics", flush=True)
    print(metrics, flush=True)

    if args.output_json:
        output_path = os.path.abspath(args.output_json)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2)
        print(f"[eval] saved {output_path}", flush=True)


if __name__ == "__main__":
    main()
