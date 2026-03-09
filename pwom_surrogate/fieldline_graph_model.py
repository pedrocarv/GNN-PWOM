"""Graph model for the field-line PWOM surrogate.

Physical interpretation:
- Message passing moves information between neighboring altitude shells and across species.
- The network predicts shell-state corrections from graph-structured shell history.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig
from .fieldline_graph_data import (
    BOUNDARY_SOURCE_CHANNEL_INDICES,
    DYNAMIC_CLOSURE_CHANNEL_INDICES,
    DYNAMIC_DENSITY_CHANNEL_INDICES,
    DYNAMIC_FORCING_CHANNEL_INDICES,
    DYNAMIC_GEOMETRY_CHANNEL_INDICES,
    DYNAMIC_MOMENTUM_CHANNEL_INDICES,
    DYNAMIC_NODE_FEATURE_NAMES,
    DYNAMIC_OCCUPANCY_SHAPE_CHANNEL_INDICES,
    DYNAMIC_TOPOLOGY_CHANNEL_INDICES,
    POPULATION_FRACTION_CHANNEL_INDEX,
    NONNEGATIVE_VELOCITY_ASINH_CHANNEL_INDICES,
    SIGNED_VELOCITY_ASINH_CHANNEL_INDICES,
    TARGET_SCHEMA_POPULATION_WEIGHTED,
    TRANSPORT_CONSERVATIVE_CHANNEL_INDICES,
    TRANSPORT_INTERFACE_FEATURE_NAMES,
    # Conservative updates are clamped in asinh space so recovered shell velocities stay finite.
    VELOCITY_ASINH_ABS_MAX,
    validate_target_schema,
)


class MLP(nn.Module):
    """Small MLP used by the graph encoder, edge update, node update, and decoder."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, layers: int) -> None:
        super().__init__()
        blocks: list[nn.Module] = []

        # Layer stacking is kept explicit so hidden widths remain easy to audit and change.
        for layer_index in range(layers):
            in_dim = input_dim if layer_index == 0 else hidden_dim
            out_dim = output_dim if layer_index == layers - 1 else hidden_dim
            blocks.append(nn.Linear(in_dim, out_dim))
            if layer_index != layers - 1:
                # GELU is smooth enough for regression while remaining simple and stable.
                blocks.append(nn.GELU())

        self.net = nn.Sequential(*blocks)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize linears with a small variance so early rollout behavior is well conditioned."""
        for module in self.net:
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(0.0, 1.0 / math.sqrt(module.in_features))
                module.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the MLP block to one tensor batch."""
        return self.net(x)


class GraphResidualBlock(nn.Module):
    """One residual message-passing block for the field-line graph.

    Physical interpretation:
    - Edge messages approximate local transport/coupling between neighboring shells.
    - Residual node updates preserve stable autoregressive state propagation.
    """

    def __init__(self, hidden_dim: int, mlp_layers: int) -> None:
        super().__init__()

        # Edge messages consume sender and receiver latent states.
        self.edge_mlp = MLP(input_dim=hidden_dim * 2, hidden_dim=hidden_dim, output_dim=hidden_dim, layers=mlp_layers)
        # Node updates combine the previous node latent with the sum of incoming messages.
        self.node_mlp = MLP(input_dim=hidden_dim * 2, hidden_dim=hidden_dim, output_dim=hidden_dim, layers=mlp_layers)

    def forward(self, node_latent: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Apply one message-passing step with sum aggregation."""
        src_index = edge_index[0]
        dst_index = edge_index[1]

        # Sender-receiver concatenation gives the block access to directional shell coupling.
        edge_input = torch.cat((node_latent.index_select(0, src_index), node_latent.index_select(0, dst_index)), dim=-1)
        message = self.edge_mlp(edge_input)

        # Sum aggregation follows the superposition view of flux and source contributions.
        aggregated = torch.zeros_like(node_latent)
        if dst_index.numel() > 0:
            aggregated.index_add_(0, dst_index, message)

        # Residual node updates make long-rollout training less brittle.
        node_update = self.node_mlp(torch.cat((node_latent, aggregated), dim=-1))
        return node_latent + node_update


class FieldlineGraphForecaster(nn.Module):
    """Encode-process-decode graph forecaster for shell-state evolution."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        # Explicit target-schema validation keeps closure-aware training from silently mixing
        # incompatible output parameterizations with the same graph topology.
        validate_target_schema(cfg.target_schema)
        if cfg.flux_transport_head and cfg.target_schema != TARGET_SCHEMA_POPULATION_WEIGHTED:
            raise ValueError("flux_transport_head currently supports only the population_weighted target schema.")
        if cfg.flux_transport_head and not cfg.predict_delta_state:
            raise ValueError("flux_transport_head requires predict_delta_state=True.")
        if cfg.operator_split_head and not cfg.flux_transport_head:
            raise ValueError("operator_split_head requires flux_transport_head=True.")
        if cfg.redistribution_transport_head and not cfg.flux_transport_head:
            raise ValueError("redistribution_transport_head requires flux_transport_head=True.")
        if cfg.bounded_outflow_fraction_transport and not cfg.flux_transport_head:
            raise ValueError("bounded_outflow_fraction_transport requires flux_transport_head=True.")
        if cfg.simplex_support_transport and not cfg.flux_transport_head:
            raise ValueError("simplex_support_transport requires flux_transport_head=True.")
        if cfg.redistribution_transport_head and cfg.bounded_outflow_fraction_transport:
            raise ValueError("redistribution_transport_head and bounded_outflow_fraction_transport are mutually exclusive.")
        if cfg.redistribution_transport_head and cfg.simplex_support_transport:
            raise ValueError("redistribution_transport_head and simplex_support_transport are mutually exclusive.")
        if cfg.simplex_support_transport and cfg.bounded_outflow_fraction_transport:
            raise ValueError("simplex_support_transport and bounded_outflow_fraction_transport are mutually exclusive.")
        if cfg.operator_split_head and cfg.target_schema != TARGET_SCHEMA_POPULATION_WEIGHTED:
            raise ValueError("operator_split_head currently supports only the population_weighted target schema.")
        if cfg.operator_split_head and not cfg.predict_delta_state:
            raise ValueError("operator_split_head requires predict_delta_state=True.")
        if cfg.conditional_momentum_head and cfg.factorized_decoder_heads:
            raise ValueError("conditional_momentum_head and factorized_decoder_heads are mutually exclusive.")
        if (cfg.split_decoder_heads or cfg.factorized_decoder_heads) and cfg.node_output_dim != len(DYNAMIC_NODE_FEATURE_NAMES):
            raise ValueError(
                "Split/factorized decoder heads require node_output_dim to match the canonical dynamic feature width. "
                f"Got node_output_dim={cfg.node_output_dim} expected={len(DYNAMIC_NODE_FEATURE_NAMES)}."
            )
        self.register_buffer(
            "closure_channel_index",
            torch.as_tensor(DYNAMIC_CLOSURE_CHANNEL_INDICES, dtype=torch.long),
            persistent=False,
        )

        # The encoder lifts raw shell descriptors into a latent transport state.
        self.node_encoder = MLP(
            input_dim=cfg.node_input_dim,
            hidden_dim=cfg.hidden_dim,
            output_dim=cfg.hidden_dim,
            layers=cfg.mlp_layers,
        )
        if cfg.global_context_dim > 0:
            if cfg.global_context_film:
                # FiLM conditioning treats line-global controls as regime modifiers. This is more
                # faithful for `SmLat` and cusp/aurora state than injecting them as extra shell
                # state, because they should change how transport behaves everywhere at once.
                self.global_context_film_encoder = MLP(
                    input_dim=cfg.global_context_dim,
                    hidden_dim=cfg.global_context_hidden_dim,
                    output_dim=cfg.hidden_dim * 2,
                    layers=cfg.mlp_layers,
                )
            else:
                # The additive context path remains available for ablations and compatibility.
                self.global_context_encoder = MLP(
                    input_dim=cfg.global_context_dim,
                    hidden_dim=cfg.global_context_hidden_dim,
                    output_dim=cfg.hidden_dim,
                    layers=cfg.mlp_layers,
                )
        if cfg.boundary_injection_head:
            self.register_buffer(
                "boundary_source_channel_index",
                torch.as_tensor(BOUNDARY_SOURCE_CHANNEL_INDICES, dtype=torch.long),
                persistent=False,
            )
            # The boundary branch only sees lower-boundary controls plus the local latent state,
            # because its job is to model source injection at the base of the field line rather
            # than generic interior shell dynamics.
            self.boundary_injection_decoder = MLP(
                input_dim=cfg.hidden_dim + int(self.boundary_source_channel_index.numel()),
                hidden_dim=cfg.boundary_hidden_dim,
                output_dim=cfg.node_output_dim,
                layers=cfg.mlp_layers,
            )
        # Residual graph blocks propagate transport information through the field-line topology.
        self.processor = nn.ModuleList(
            [GraphResidualBlock(hidden_dim=cfg.hidden_dim, mlp_layers=cfg.mlp_layers) for _ in range(cfg.message_passing_layers)]
        )
        if cfg.flux_transport_head:
            self.register_buffer(
                "transport_conservative_channel_index",
                torch.as_tensor(TRANSPORT_CONSERVATIVE_CHANNEL_INDICES, dtype=torch.long),
                persistent=False,
            )
            if cfg.redistribution_transport_head or cfg.simplex_support_transport:
                # The redistribution transport decoder predicts an `up/down` split, a bounded
                # total outflow fraction, and carried parallel velocities for outgoing support.
                # This keeps shell support nonnegative and prevents one-step full-shell drainout.
                self.transport_flux_decoder = MLP(
                    input_dim=cfg.hidden_dim,
                    hidden_dim=cfg.flux_hidden_dim,
                    output_dim=5,
                    layers=cfg.mlp_layers,
                )
            else:
                # The interface decoder predicts net per-step exchange across same-species shell
                # boundaries. The divergence of these exchanges gives a conservative node update.
                self.transport_flux_decoder = MLP(
                    input_dim=(cfg.hidden_dim * 2) + len(TRANSPORT_INTERFACE_FEATURE_NAMES),
                    hidden_dim=cfg.flux_hidden_dim,
                    output_dim=int(self.transport_conservative_channel_index.numel()),
                    layers=cfg.mlp_layers,
                )
            # The residual conservative head captures local source terms not explained by pure
            # along-field transport, such as heating or closure error in the coarse shell state.
            self.transport_residual_decoder = MLP(
                input_dim=cfg.hidden_dim,
                hidden_dim=cfg.hidden_dim,
                output_dim=int(self.transport_conservative_channel_index.numel()),
                layers=cfg.mlp_layers,
            )
        if cfg.conditional_momentum_head:
            self.register_buffer(
                "occupancy_shape_channel_index",
                torch.as_tensor(DYNAMIC_OCCUPANCY_SHAPE_CHANNEL_INDICES, dtype=torch.long),
                persistent=False,
            )
            self.register_buffer(
                "momentum_channel_index",
                torch.as_tensor(DYNAMIC_MOMENTUM_CHANNEL_INDICES, dtype=torch.long),
                persistent=False,
            )
            self.register_buffer(
                "forcing_channel_index",
                torch.as_tensor(DYNAMIC_FORCING_CHANNEL_INDICES, dtype=torch.long),
                persistent=False,
            )
            # Population fraction is the gating variable because momentum is only meaningful when a
            # shell actually contains enough particle weight to define a distribution.
            self.register_buffer(
                "population_channel_index",
                torch.as_tensor([POPULATION_FRACTION_CHANNEL_INDEX], dtype=torch.long),
                persistent=False,
            )
            self.occupancy_shape_decoder = MLP(
                input_dim=cfg.hidden_dim,
                hidden_dim=cfg.hidden_dim,
                output_dim=int(self.occupancy_shape_channel_index.numel()),
                layers=cfg.mlp_layers,
            )
            self.forcing_decoder = MLP(
                input_dim=cfg.hidden_dim,
                hidden_dim=cfg.hidden_dim,
                output_dim=int(self.forcing_channel_index.numel()),
                layers=cfg.mlp_layers,
            )
            self.momentum_decoder = MLP(
                input_dim=cfg.hidden_dim + int(self.occupancy_shape_channel_index.numel()) + int(self.forcing_channel_index.numel()) + 1,
                hidden_dim=cfg.hidden_dim,
                output_dim=int(self.momentum_channel_index.numel()),
                layers=cfg.mlp_layers,
            )
            if cfg.uncertainty_aware_closure_head:
                # The closure log-variance head shares the same conditioning path as the momentum
                # decoder because closure uncertainty should depend on predicted occupancy and the
                # local forcing context that also shape the closure mean.
                self.closure_logvar_decoder = MLP(
                    input_dim=cfg.hidden_dim + int(self.occupancy_shape_channel_index.numel()) + int(self.forcing_channel_index.numel()) + 1,
                    hidden_dim=cfg.hidden_dim,
                    output_dim=int(self.closure_channel_index.numel()),
                    layers=cfg.mlp_layers,
                )
        elif cfg.factorized_decoder_heads:
            # These channel groups follow the coarse-shell physics decomposition:
            # density/occupancy, shell geometry, topology, momentum-like moments, and forcing.
            self.register_buffer(
                "density_channel_index",
                torch.as_tensor(DYNAMIC_DENSITY_CHANNEL_INDICES, dtype=torch.long),
                persistent=False,
            )
            self.register_buffer(
                "topology_channel_index",
                torch.as_tensor(DYNAMIC_TOPOLOGY_CHANNEL_INDICES, dtype=torch.long),
                persistent=False,
            )
            self.register_buffer(
                "geometry_channel_index",
                torch.as_tensor(DYNAMIC_GEOMETRY_CHANNEL_INDICES, dtype=torch.long),
                persistent=False,
            )
            self.register_buffer(
                "momentum_channel_index",
                torch.as_tensor(DYNAMIC_MOMENTUM_CHANNEL_INDICES, dtype=torch.long),
                persistent=False,
            )
            self.register_buffer(
                "forcing_channel_index",
                torch.as_tensor(DYNAMIC_FORCING_CHANNEL_INDICES, dtype=torch.long),
                persistent=False,
            )

            # Each group gets its own decoder directly from the latent graph state so density and
            # momentum errors do not have to share one decoder trunk.
            self.density_decoder = MLP(
                input_dim=cfg.hidden_dim,
                hidden_dim=cfg.hidden_dim,
                output_dim=int(self.density_channel_index.numel()),
                layers=cfg.mlp_layers,
            )
            self.topology_decoder = MLP(
                input_dim=cfg.hidden_dim,
                hidden_dim=cfg.hidden_dim,
                output_dim=int(self.topology_channel_index.numel()),
                layers=cfg.mlp_layers,
            )
            self.geometry_decoder = MLP(
                input_dim=cfg.hidden_dim,
                hidden_dim=cfg.hidden_dim,
                output_dim=int(self.geometry_channel_index.numel()),
                layers=cfg.mlp_layers,
            )
            self.momentum_decoder = MLP(
                input_dim=cfg.hidden_dim,
                hidden_dim=cfg.hidden_dim,
                output_dim=int(self.momentum_channel_index.numel()),
                layers=cfg.mlp_layers,
            )
            self.forcing_decoder = MLP(
                input_dim=cfg.hidden_dim,
                hidden_dim=cfg.hidden_dim,
                output_dim=int(self.forcing_channel_index.numel()),
                layers=cfg.mlp_layers,
            )
            if cfg.uncertainty_aware_closure_head:
                # In factorized mode the closure uncertainty should be predicted directly from the
                # latent graph state, independently of density or forcing decoder errors.
                self.closure_logvar_decoder = MLP(
                    input_dim=cfg.hidden_dim,
                    hidden_dim=cfg.hidden_dim,
                    output_dim=int(self.closure_channel_index.numel()),
                    layers=cfg.mlp_layers,
                )
        elif cfg.split_decoder_heads:
            # Velocity moments are the hardest channels in rollout, so they get their own decoder
            # path instead of sharing the exact same output head as occupancy and geometry terms.
            non_velocity_channel_indices = tuple(
                channel_index
                for channel_index in range(len(DYNAMIC_NODE_FEATURE_NAMES))
                if channel_index not in SIGNED_VELOCITY_ASINH_CHANNEL_INDICES
                and channel_index not in NONNEGATIVE_VELOCITY_ASINH_CHANNEL_INDICES
            )
            self.register_buffer(
                "non_velocity_channel_index",
                torch.as_tensor(non_velocity_channel_indices, dtype=torch.long),
                persistent=False,
            )
            self.register_buffer(
                "signed_velocity_channel_index",
                torch.as_tensor(SIGNED_VELOCITY_ASINH_CHANNEL_INDICES, dtype=torch.long),
                persistent=False,
            )
            self.register_buffer(
                "nonnegative_velocity_channel_index",
                torch.as_tensor(NONNEGATIVE_VELOCITY_ASINH_CHANNEL_INDICES, dtype=torch.long),
                persistent=False,
            )

            # One shared decoder trunk still maps latent transport state into an output-ready
            # representation before the channel groups branch apart.
            self.decoder_trunk = MLP(
                input_dim=cfg.hidden_dim,
                hidden_dim=cfg.hidden_dim,
                output_dim=cfg.hidden_dim,
                layers=cfg.mlp_layers,
            )
            self.non_velocity_decoder = MLP(
                input_dim=cfg.hidden_dim,
                hidden_dim=cfg.hidden_dim,
                output_dim=int(self.non_velocity_channel_index.numel()),
                layers=cfg.mlp_layers,
            )
            self.signed_velocity_decoder = MLP(
                input_dim=cfg.hidden_dim,
                hidden_dim=cfg.hidden_dim,
                output_dim=int(self.signed_velocity_channel_index.numel()),
                layers=cfg.mlp_layers,
            )
            self.nonnegative_velocity_decoder = MLP(
                input_dim=cfg.hidden_dim,
                hidden_dim=cfg.hidden_dim,
                output_dim=int(self.nonnegative_velocity_channel_index.numel()),
                layers=cfg.mlp_layers,
            )
            if cfg.uncertainty_aware_closure_head:
                # The split-head path already isolates the hardest velocity-like channels, so
                # the uncertainty head reads the same decoder trunk before assigning per-channel
                # confidence to the closure outputs.
                self.closure_logvar_decoder = MLP(
                    input_dim=cfg.hidden_dim,
                    hidden_dim=cfg.hidden_dim,
                    output_dim=int(self.closure_channel_index.numel()),
                    layers=cfg.mlp_layers,
                )
        else:
            # The shared decoder maps latent shell state back to next-step state increments or
            # absolute values with one shared head for all channels.
            self.decoder = MLP(
                input_dim=cfg.hidden_dim,
                hidden_dim=cfg.hidden_dim,
                output_dim=cfg.node_output_dim,
                layers=cfg.mlp_layers,
            )
            if cfg.uncertainty_aware_closure_head:
                # The single-head path can still expose closure uncertainty from the shared latent
                # graph state without changing default decoder behavior.
                self.closure_logvar_decoder = MLP(
                    input_dim=cfg.hidden_dim,
                    hidden_dim=cfg.hidden_dim,
                    output_dim=int(self.closure_channel_index.numel()),
                    layers=cfg.mlp_layers,
                )

    def _predict_population_fraction_raw(
        self,
        occupancy_shape_pred_norm: torch.Tensor,
        last_dynamic: torch.Tensor,
        stats_torch: dict[str, torch.Tensor],
        predict_delta_state: bool,
    ) -> torch.Tensor:
        """Recover predicted population fraction in raw shell-state units for momentum gating.

        Physical interpretation:
        - The momentum branch should only activate when the predicted shell occupancy is large
          enough to support a meaningful velocity distribution.
        - Population fraction is not transformed with `asinh`, so its raw value is the same
          physically interpretable occupancy proxy used throughout this pipeline.
        """
        population_local_index = 0
        population_pred_norm = occupancy_shape_pred_norm[..., population_local_index : population_local_index + 1]
        population_channel_index = self.population_channel_index
        if predict_delta_state:
            population_delta_std = stats_torch["delta_std"].index_select(0, population_channel_index)
            population_delta_mean = stats_torch["delta_mean"].index_select(0, population_channel_index)
            population_delta_raw = (population_pred_norm * population_delta_std) + population_delta_mean
            population_prev = last_dynamic.index_select(dim=-1, index=population_channel_index)
            population_raw = population_prev + population_delta_raw
        else:
            population_dynamic_std = stats_torch["dynamic_std"].index_select(0, population_channel_index)
            population_dynamic_mean = stats_torch["dynamic_mean"].index_select(0, population_channel_index)
            population_raw = (population_pred_norm * population_dynamic_std) + population_dynamic_mean
        return torch.clamp(population_raw, min=0.0, max=1.0)

    def _gate_momentum_prediction_norm(
        self,
        momentum_pred_norm: torch.Tensor,
        occupancy_gate: torch.Tensor,
        stats_torch: dict[str, torch.Tensor],
        predict_delta_state: bool,
    ) -> torch.Tensor:
        """Apply the occupancy gate in raw output space and map back to normalized coordinates.

        Physical interpretation:
        - In delta-state mode, the gate should suppress the physical momentum increment itself, not
          merely the normalized decoder value, because zero normalized output is not generally zero
          physical increment.
        """
        momentum_channel_index = self.momentum_channel_index
        gate_expanded = occupancy_gate.unsqueeze(-1)
        if predict_delta_state:
            momentum_delta_std = stats_torch["delta_std"].index_select(0, momentum_channel_index)
            momentum_delta_mean = stats_torch["delta_mean"].index_select(0, momentum_channel_index)
            momentum_delta_raw = (momentum_pred_norm * momentum_delta_std) + momentum_delta_mean
            gated_momentum_delta_raw = gate_expanded * momentum_delta_raw
            return (gated_momentum_delta_raw - momentum_delta_mean) / momentum_delta_std

        momentum_dynamic_std = stats_torch["dynamic_std"].index_select(0, momentum_channel_index)
        momentum_dynamic_mean = stats_torch["dynamic_mean"].index_select(0, momentum_channel_index)
        momentum_raw = (momentum_pred_norm * momentum_dynamic_std) + momentum_dynamic_mean
        gated_momentum_raw = gate_expanded * momentum_raw
        return (gated_momentum_raw - momentum_dynamic_mean) / momentum_dynamic_std

    def _clamp_closure_logvar(self, closure_logvar: torch.Tensor) -> torch.Tensor:
        """Clamp predicted closure log-variance to a numerically safe range.

        Physical interpretation:
        - Sparse-shell closure targets should be allowed to carry higher uncertainty than dense
          shells.
        - The clamp prevents the uncertainty head from making the optimizer numerically unstable
          before the full closure-aware trainer is introduced.
        """
        return torch.clamp(
            closure_logvar,
            min=float(self.cfg.closure_logvar_min),
            max=float(self.cfg.closure_logvar_max),
        )

    def _apply_global_context_conditioning(
        self,
        node_latent: torch.Tensor,
        global_context: torch.Tensor | None,
    ) -> torch.Tensor:
        """Inject line-global controls into the node latent state.

        Physical interpretation:
        - `SmLat` and cusp/aurora regime do not describe one shell; they set the line-wide regime.
        - FiLM conditioning lets those controls modulate transport/source behavior without
          pretending they are shell-resolved plasma variables.
        """
        if global_context is None or self.cfg.global_context_dim <= 0:
            return node_latent
        if self.cfg.global_context_film:
            film_params = self.global_context_film_encoder(global_context)
            film_scale, film_shift = torch.chunk(film_params, chunks=2, dim=-1)
            bounded_scale = torch.tanh(film_scale)
            return (node_latent * (1.0 + bounded_scale)) + film_shift
        return node_latent + self.global_context_encoder(global_context)

    def _compute_boundary_injection_update(
        self,
        node_latent: torch.Tensor,
        boundary_context: torch.Tensor | None,
        stats_torch: dict[str, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Predict a dedicated lower-boundary source correction on the lowest shell nodes only.

        Physical interpretation:
        - PWOM boundary moments replenish or bias particle populations at the lower boundary.
        - The boundary branch returns only the additive update so operator-split mode can combine
          it explicitly with transport and source terms instead of hiding it inside one decoder.
        """
        aux: dict[str, torch.Tensor] = {}
        if not self.cfg.boundary_injection_head:
            empty_update = torch.zeros(
                node_latent.shape[0],
                self.cfg.node_output_dim,
                dtype=node_latent.dtype,
                device=node_latent.device,
            )
            aux["boundary_injection_update"] = empty_update
            return empty_update, aux
        if boundary_context is None or stats_torch is None:
            raise ValueError("boundary_injection_head requires boundary_context and stats_torch.")
        boundary_source_node_mask = stats_torch.get("boundary_source_node_mask")
        if boundary_source_node_mask is None:
            raise ValueError("boundary_injection_head requires boundary_source_node_mask in stats_torch.")
        boundary_features = boundary_context.index_select(dim=-1, index=self.boundary_source_channel_index)
        boundary_input = torch.cat((node_latent, boundary_features), dim=-1)
        boundary_update = self.boundary_injection_decoder(boundary_input)
        boundary_mask = boundary_source_node_mask.to(dtype=boundary_update.dtype)
        if boundary_update.dim() == 2:
            # Training/evaluation flatten `[batch, nodes, features]` into `[batch*nodes, features]`
            # before decoding. The physical boundary mask is defined per node, so it must be
            # repeated once per graph in the batch to keep the injection restricted to the lowest
            # shell of each species on every field line sample.
            if boundary_mask.dim() != 1:
                boundary_mask = boundary_mask.reshape(-1)
            if boundary_mask.numel() == 0:
                raise ValueError("boundary_source_node_mask cannot be empty.")
            if boundary_update.shape[0] % boundary_mask.numel() != 0:
                raise ValueError(
                    "boundary_source_node_mask length must divide the flattened decoder batch."
                )
            batch_repeat = boundary_update.shape[0] // boundary_mask.numel()
            boundary_mask = boundary_mask.repeat(batch_repeat).unsqueeze(-1)
        else:
            # Unflattened paths keep the node axis explicit, so the physical mask can be broadcast
            # across batch and feature dimensions directly.
            if boundary_mask.dim() == 1:
                boundary_mask = boundary_mask.unsqueeze(0)
            boundary_mask = boundary_mask.unsqueeze(-1)
        # `NaN * 0` is still `NaN`, so the boundary mask must be applied with an explicit where
        # instead of simple multiplication. Physically, non-boundary shells should receive exactly
        # zero boundary injection regardless of what the raw decoder emits there.
        masked_boundary_update = torch.where(
            boundary_mask > 0.0,
            boundary_update,
            torch.zeros_like(boundary_update),
        )
        aux["boundary_injection_update"] = masked_boundary_update
        return masked_boundary_update, aux

    def _apply_boundary_injection(
        self,
        base_output: torch.Tensor,
        node_latent: torch.Tensor,
        boundary_context: torch.Tensor | None,
        stats_torch: dict[str, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Add the dedicated lower-boundary update onto an existing node-state prediction.

        Physical interpretation:
        - Non-operator branches still use the simpler additive path where the boundary source is
          folded into the main decoder output before the trainer sees the final state update.
        """
        boundary_update, aux = self._compute_boundary_injection_update(
            node_latent=node_latent,
            boundary_context=boundary_context,
            stats_torch=stats_torch,
        )
        return base_output + boundary_update, aux

    def _physical_conservative_to_model_space(
        self,
        conservative_physical: torch.Tensor,
        conservative_channel_index: torch.Tensor,
        stats_torch: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Map conservative channels from physical units back into learned model-space channels.

        Physical interpretation:
        - Population support is already stored in physical units.
        - Population-weighted parallel drift is learned in compressed `asinh` coordinates, so the
          conservative transport update must be mapped back through the same compression before it
          can replace the decoder output.
        """
        asinh_channel_index = stats_torch["asinh_channel_index"]
        asinh_channel_scale = stats_torch["asinh_channel_scale"]
        conservative_columns: list[torch.Tensor] = []
        for local_offset in range(int(conservative_channel_index.numel())):
            channel_index = int(conservative_channel_index[local_offset].item())
            matching_index = torch.nonzero(asinh_channel_index == channel_index, as_tuple=False)
            if matching_index.numel() == 0:
                conservative_columns.append(conservative_physical[..., local_offset])
                continue
            scale = asinh_channel_scale[int(matching_index[0, 0].item())]
            # Conservative shell channels are rebuilt out-of-place so the operator-split backward
            # pass never sees a view overwritten while it is still part of the computation graph.
            conservative_columns.append(torch.asinh(conservative_physical[..., local_offset] / scale))

        return torch.stack(conservative_columns, dim=-1)

    def _model_delta_to_conservative_physical(
        self,
        conservative_pred_norm: torch.Tensor,
        last_dynamic: torch.Tensor,
        last_dynamic_physical: torch.Tensor,
        stats_torch: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Convert a conservative-channel model delta into a physical conservative delta.

        Physical interpretation:
        - Source and boundary branches predict updates in learned delta coordinates.
        - For diagnostics and regularization we need their implied physical support and momentum
          increments, not just the normalized decoder values.
        """
        conservative_channel_index = self.transport_conservative_channel_index
        delta_mean = stats_torch["delta_mean"].index_select(0, conservative_channel_index)
        delta_std = stats_torch["delta_std"].index_select(0, conservative_channel_index)
        conservative_delta_model = (conservative_pred_norm * delta_std) + delta_mean
        last_conservative_model = last_dynamic.index_select(dim=-1, index=conservative_channel_index)
        next_conservative_model = last_conservative_model + conservative_delta_model
        next_conservative_model = self._constrain_conservative_model_state(
            conservative_model=next_conservative_model,
            conservative_channel_index=conservative_channel_index,
            stats_torch=stats_torch,
        )
        next_conservative_physical = self._conservative_model_to_physical(
            conservative_model=next_conservative_model,
            conservative_channel_index=conservative_channel_index,
            stats_torch=stats_torch,
        )
        last_conservative_physical = last_dynamic_physical.index_select(dim=-1, index=conservative_channel_index)
        return next_conservative_physical - last_conservative_physical

    def _constrain_conservative_model_state(
        self,
        conservative_model: torch.Tensor,
        conservative_channel_index: torch.Tensor,
        stats_torch: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Apply physical bounds to the conservative pair before inverse transforms.

        Physical interpretation:
        - Population fraction must stay inside `[0, 1]` after transport, boundary injection, and
          source updates are combined.
        - The population-weighted parallel-drift channel is stored in bounded `asinh` space, so it
          should be clamped before any `sinh()` call to avoid artificial numerical overflow.
        """
        constrained_columns: list[torch.Tensor] = []
        for local_offset in range(int(conservative_channel_index.numel())):
            channel_index = int(conservative_channel_index[local_offset].item())
            if channel_index == int(POPULATION_FRACTION_CHANNEL_INDEX):
                constrained_columns.append(torch.clamp(
                    conservative_model[..., local_offset],
                    min=0.0,
                    max=1.0,
                ))
                continue
            matching_index = torch.nonzero(
                stats_torch["signed_velocity_asinh_channel_index"] == channel_index,
                as_tuple=False,
            )
            if matching_index.numel() > 0:
                constrained_columns.append(torch.clamp(
                    conservative_model[..., local_offset],
                    min=-VELOCITY_ASINH_ABS_MAX,
                    max=VELOCITY_ASINH_ABS_MAX,
                ))
                continue
            constrained_columns.append(conservative_model[..., local_offset])
        return torch.stack(constrained_columns, dim=-1)

    def _conservative_model_to_physical(
        self,
        conservative_model: torch.Tensor,
        conservative_channel_index: torch.Tensor,
        stats_torch: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Map conservative model-space channels back into physical units.

        Physical interpretation:
        - Population fraction is already a physical shell-support quantity.
        - Population-weighted parallel drift uses the same `asinh` compression as the shell-state
          targets, so it must be decompressed before a physical transport/source delta is formed.
        """
        asinh_channel_index = stats_torch["asinh_channel_index"]
        asinh_channel_scale = stats_torch["asinh_channel_scale"]
        conservative_columns: list[torch.Tensor] = []
        for local_offset in range(int(conservative_channel_index.numel())):
            channel_index = int(conservative_channel_index[local_offset].item())
            matching_index = torch.nonzero(asinh_channel_index == channel_index, as_tuple=False)
            if matching_index.numel() == 0:
                conservative_columns.append(conservative_model[..., local_offset])
                continue
            scale = asinh_channel_scale[int(matching_index[0, 0].item())]
            # Converting back to physical conservative variables also stays out-of-place for the
            # same autograd reason as the model-space reconstruction above.
            conservative_columns.append(torch.sinh(conservative_model[..., local_offset]) * scale)
        return torch.stack(conservative_columns, dim=-1)

    def _apply_conservative_channel_update(
        self,
        base_output: torch.Tensor,
        conservative_channel_index: torch.Tensor,
        conservative_update: torch.Tensor,
        additive: bool,
    ) -> torch.Tensor:
        """Update the conservative pair without in-place slice assignment.

        Physical interpretation:
        - The operator-split model only modifies the conservative shell pair through transport,
          while all other shell channels remain the decoder output.
        - Reconstructing the full tensor out-of-place avoids autograd aliasing on the flattened
          shell dimension used during one-step training.
        """
        expanded_index = conservative_channel_index
        while expanded_index.dim() < conservative_update.dim():
            expanded_index = expanded_index.unsqueeze(0)
        expanded_index = expanded_index.expand_as(conservative_update)
        conservative_update_full = torch.zeros_like(base_output).scatter(
            dim=-1,
            index=expanded_index,
            src=conservative_update,
        )
        if additive:
            return base_output + conservative_update_full
        conservative_mask = F.one_hot(
            conservative_channel_index,
            num_classes=base_output.shape[-1],
        ).sum(dim=0).to(dtype=base_output.dtype, device=base_output.device)
        while conservative_mask.dim() < base_output.dim():
            conservative_mask = conservative_mask.unsqueeze(0)
        return (base_output * (1.0 - conservative_mask)) + conservative_update_full

    def _bound_support_interface_flux(
        self,
        support_transport_score: torch.Tensor,
        last_support_physical: torch.Tensor,
        transport_left_index: torch.Tensor,
        transport_right_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert signed interface transport scores into bounded shell-support transfer amounts.

        Physical interpretation:
        - The support channel is a shell-averaged density-like quantity, so the transport branch
          should move a bounded fraction of the shell support each step, not an arbitrary raw flux.
        - A positive score means left-to-right transport, while a negative score means right-to-left
          transport across the interface.
        - Per-shell outgoing requests are normalized before transport is applied, so a shell cannot
          export more than `max_support_outflow_fraction` of the support it currently contains.
        """
        if support_transport_score.numel() == 0:
            # The empty-interface case appears in degenerate graphs with no same-species shell
            # neighbors. Returning zeros in the score tensor's dtype/device keeps the transport
            # branch algebra well-defined without introducing a fake flux scale.
            zero_fraction = torch.zeros(
                (),
                dtype=support_transport_score.dtype,
                device=support_transport_score.device,
            )
            zero_like = torch.zeros_like(support_transport_score)
            return zero_like, zero_like, zero_like, zero_fraction

        requested_fraction = torch.tanh(support_transport_score)
        left_support = torch.clamp(last_support_physical.index_select(0, transport_left_index), min=0.0)
        right_support = torch.clamp(last_support_physical.index_select(0, transport_right_index), min=0.0)
        requested_left_to_right = torch.clamp(requested_fraction, min=0.0) * left_support
        requested_right_to_left = torch.clamp(-requested_fraction, min=0.0) * right_support

        outgoing_demand = torch.zeros_like(last_support_physical)
        outgoing_demand.index_add_(0, transport_left_index, requested_left_to_right)
        outgoing_demand.index_add_(0, transport_right_index, requested_right_to_left)

        available_support = torch.clamp(last_support_physical, min=0.0)
        available_outflow = available_support * float(self.cfg.max_support_outflow_fraction)
        outgoing_scale = torch.ones_like(outgoing_demand)
        active_outflow_mask = outgoing_demand > 0.0
        outgoing_scale = torch.where(
            active_outflow_mask,
            torch.clamp(available_outflow / torch.clamp(outgoing_demand, min=1.0e-12), max=1.0),
            outgoing_scale,
        )

        left_sender_scale = outgoing_scale.index_select(0, transport_left_index)
        right_sender_scale = outgoing_scale.index_select(0, transport_right_index)
        bounded_left_to_right = requested_left_to_right * left_sender_scale
        bounded_right_to_left = requested_right_to_left * right_sender_scale
        bounded_support_transfer = bounded_left_to_right - bounded_right_to_left

        total_requested_outflow = torch.sum(requested_left_to_right + requested_right_to_left)
        total_capped_outflow = torch.sum(
            (requested_left_to_right - bounded_left_to_right) +
            (requested_right_to_left - bounded_right_to_left)
        )
        capped_fraction = torch.where(
            total_requested_outflow > 0.0,
            total_capped_outflow / torch.clamp(total_requested_outflow, min=1.0e-12),
            torch.zeros((), dtype=support_transport_score.dtype, device=support_transport_score.device),
        )
        return bounded_support_transfer, left_sender_scale, right_sender_scale, capped_fraction

    def _build_flux_transport_update(
        self,
        node_latent: torch.Tensor,
        last_dynamic: torch.Tensor,
        last_dynamic_physical: torch.Tensor,
        stats_torch: dict[str, torch.Tensor],
        transport_left_index: torch.Tensor,
        transport_right_index: torch.Tensor,
        transport_interface_features: torch.Tensor,
        transport_shell_node_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Predict conservative shell updates from interface fluxes plus a local residual source.

        Physical interpretation:
        - Interface fluxes move population support and parallel momentum between neighboring shells.
        - The residual conservative head captures local source/sink effects that cannot be written
          as pure 1-D exchange between adjacent shells.
        """
        conservative_channel_index = self.transport_conservative_channel_index
        interface_flux_physical = torch.zeros(
            transport_left_index.shape[0],
            int(conservative_channel_index.numel()),
            dtype=node_latent.dtype,
            device=node_latent.device,
        )
        transport_delta_physical = torch.zeros(
            node_latent.shape[0],
            int(conservative_channel_index.numel()),
            dtype=node_latent.dtype,
            device=node_latent.device,
        )
        support_flux_capped_fraction = torch.zeros((), dtype=node_latent.dtype, device=node_latent.device)
        support_stay_fraction_mean = torch.zeros((), dtype=node_latent.dtype, device=node_latent.device)
        support_outflow_fraction_mean = torch.zeros((), dtype=node_latent.dtype, device=node_latent.device)
        support_outflow_cap_saturation_mean = torch.zeros((), dtype=node_latent.dtype, device=node_latent.device)
        support_outflow_fraction_per_node = torch.zeros(node_latent.shape[0], dtype=node_latent.dtype, device=node_latent.device)
        if self.cfg.redistribution_transport_head or self.cfg.simplex_support_transport:
            support_local_index = torch.nonzero(
                conservative_channel_index == int(POPULATION_FRACTION_CHANNEL_INDEX),
                as_tuple=False,
            )
            if support_local_index.numel() == 0:
                raise ValueError("redistribution transport conservative pair must include population_fraction.")
            support_local_index_int = int(support_local_index[0, 0].item())
            momentum_local_index = 1 - support_local_index_int
            shell_node_index = torch.nonzero(transport_shell_node_mask > 0.0, as_tuple=False).squeeze(-1)
            if shell_node_index.numel() > 0:
                redistribution_raw = self.transport_flux_decoder(node_latent.index_select(0, shell_node_index))
                direction_split = torch.softmax(redistribution_raw[:, :2], dim=-1)
                bounded_total_outflow = float(self.cfg.max_support_outflow_fraction) * torch.sigmoid(redistribution_raw[:, 2])
                upward_fraction = bounded_total_outflow * direction_split[:, 0]
                downward_fraction = bounded_total_outflow * direction_split[:, 1]
                stay_fraction = 1.0 - bounded_total_outflow
                support_stay_fraction_mean = torch.mean(stay_fraction)
                support_outflow_fraction_mean = torch.mean(bounded_total_outflow)
                support_outflow_cap_saturation_mean = torch.mean(
                    bounded_total_outflow / max(float(self.cfg.max_support_outflow_fraction), 1.0e-6)
                )
                support_outflow_fraction_per_node.index_add_(0, shell_node_index, bounded_total_outflow)
                last_support_physical = last_dynamic_physical.index_select(
                    dim=-1,
                    index=conservative_channel_index[support_local_index_int:support_local_index_int + 1],
                ).squeeze(-1)
                shell_support = torch.clamp(last_support_physical.index_select(0, shell_node_index), min=0.0)
                transport_delta_scale = stats_torch.get("transport_delta_scale")
                if transport_delta_scale is None:
                    raise ValueError("redistribution transport requires transport_delta_scale in stats_torch.")
                support_scale = float(transport_delta_scale[support_local_index_int].item())
                momentum_scale = float(transport_delta_scale[momentum_local_index].item())
                carry_velocity_scale = momentum_scale / max(support_scale, 1.0e-6)
                carry_velocity = torch.tanh(redistribution_raw[:, 3:5]) * carry_velocity_scale
                upward_support = shell_support * upward_fraction
                downward_support = shell_support * downward_fraction

                upward_support_full = torch.zeros(node_latent.shape[0], dtype=node_latent.dtype, device=node_latent.device)
                downward_support_full = torch.zeros_like(upward_support_full)
                upward_carry_full = torch.zeros_like(upward_support_full)
                downward_carry_full = torch.zeros_like(upward_support_full)
                # Shell-node indices are unique, so index_add_ is equivalent to index_copy_ here and
                # keeps the redistribution transport path compatible with the MPS backend.
                upward_support_full.index_add_(0, shell_node_index, upward_support)
                downward_support_full.index_add_(0, shell_node_index, downward_support)
                upward_carry_full.index_add_(0, shell_node_index, carry_velocity[:, 0])
                downward_carry_full.index_add_(0, shell_node_index, carry_velocity[:, 1])

                left_to_right_support = upward_support_full.index_select(0, transport_left_index)
                right_to_left_support = downward_support_full.index_select(0, transport_right_index)
                left_to_right_momentum = left_to_right_support * upward_carry_full.index_select(0, transport_left_index)
                right_to_left_momentum = right_to_left_support * downward_carry_full.index_select(0, transport_right_index)

                interface_support_transfer = left_to_right_support - right_to_left_support
                interface_momentum_transfer = left_to_right_momentum - right_to_left_momentum
                interface_flux_physical = torch.stack(
                    (interface_support_transfer, interface_momentum_transfer),
                    dim=-1,
                )

                transport_delta_physical.index_add_(
                    0,
                    transport_left_index,
                    torch.stack(
                        (
                            -left_to_right_support + right_to_left_support,
                            -left_to_right_momentum + right_to_left_momentum,
                        ),
                        dim=-1,
                    ),
                )
                transport_delta_physical.index_add_(
                    0,
                    transport_right_index,
                    torch.stack(
                        (
                            left_to_right_support - right_to_left_support,
                            left_to_right_momentum - right_to_left_momentum,
                        ),
                        dim=-1,
                    ),
                )
        elif transport_left_index.numel() > 0:
            left_latent = node_latent.index_select(0, transport_left_index)
            right_latent = node_latent.index_select(0, transport_right_index)
            interface_input = torch.cat((left_latent, right_latent, transport_interface_features), dim=-1)
            interface_flux_norm = self.transport_flux_decoder(interface_input)
            transport_delta_scale = stats_torch.get("transport_delta_scale")
            if transport_delta_scale is None:
                raise ValueError("flux_transport_head requires transport_delta_scale in stats_torch.")
            if self.cfg.bounded_outflow_fraction_transport:
                # In the bounded outflow-fraction variant, the interface head predicts a signed
                # fraction of local shell support to move across each face, not a raw physical
                # support flux. The momentum channel is interpreted as a carried parallel-velocity
                # factor that will be applied to the bounded moved support below.
                interface_flux_physical = interface_flux_norm
            else:
                transport_interface_width_km = stats_torch.get("transport_interface_width_km")
                if transport_interface_width_km is None:
                    raise ValueError("flux_transport_head requires transport_interface_width_km in stats_torch.")
                if transport_interface_width_km.shape[0] != transport_left_index.shape[0]:
                    if transport_left_index.shape[0] % max(int(transport_interface_width_km.shape[0]), 1) != 0:
                        raise ValueError(
                            "transport_interface_width_km does not match the flattened interface count."
                        )
                    batch_repeats = transport_left_index.shape[0] // max(int(transport_interface_width_km.shape[0]), 1)
                    # Batched one-step training flattens identical-topology graphs. Interface widths are
                    # graph metadata, so they must be repeated once per batch element before converting
                    # normalized interface predictions back into physical per-face fluxes.
                    transport_interface_width_km = transport_interface_width_km.repeat(batch_repeats)
                safe_interface_width = torch.clamp(transport_interface_width_km.unsqueeze(-1), min=1.0e-6)
                # The flux-form branch predicts a dimensionless transported amount per face.
                # Multiplying by the conservative shell-delta scale and interface width produces a
                # physical per-step face flux before the divergence divides by the shell width.
                interface_flux_scale = transport_delta_scale.unsqueeze(0) * safe_interface_width
                interface_flux_physical = interface_flux_norm * interface_flux_scale
            support_local_index = torch.nonzero(
                conservative_channel_index == int(POPULATION_FRACTION_CHANNEL_INDEX),
                as_tuple=False,
            )
            if support_local_index.numel() == 0:
                raise ValueError("flux_transport_head conservative pair must include population_fraction.")
            support_local_index_int = int(support_local_index[0, 0].item())
            last_support_physical = last_dynamic_physical.index_select(
                dim=-1,
                index=conservative_channel_index[support_local_index_int:support_local_index_int + 1],
            ).squeeze(-1)
            if self.cfg.bounded_outflow_fraction_transport:
                bounded_support_transfer, left_sender_scale, right_sender_scale, support_flux_capped_fraction = self._bound_support_interface_flux(
                    support_transport_score=interface_flux_physical[:, support_local_index_int],
                    last_support_physical=last_support_physical,
                    transport_left_index=transport_left_index,
                    transport_right_index=transport_right_index,
                )
                momentum_local_index = 1 - support_local_index_int
                support_scale = float(transport_delta_scale[support_local_index_int].item())
                momentum_scale = float(transport_delta_scale[momentum_local_index].item())
                carry_velocity_scale = momentum_scale / max(support_scale, 1.0e-6)
                requested_fraction = torch.tanh(interface_flux_physical[:, support_local_index_int])
                carry_velocity = torch.tanh(interface_flux_physical[:, momentum_local_index]) * carry_velocity_scale
                bounded_sender_scale = torch.where(
                    requested_fraction >= 0.0,
                    left_sender_scale,
                    right_sender_scale,
                )
                bounded_momentum_transfer = bounded_support_transfer * carry_velocity * bounded_sender_scale
                bounded_flux_columns = []
                for local_offset in range(int(conservative_channel_index.numel())):
                    if local_offset == support_local_index_int:
                        # The support channel is moved as a bounded transferred amount, not an
                        # unconstrained flux, so each shell exports at most a fixed fraction of the
                        # support it currently contains.
                        bounded_flux_columns.append(bounded_support_transfer)
                    else:
                        # Parallel momentum is carried by the bounded transported support through a
                        # learned interface velocity factor. This ties momentum transport directly to
                        # how much support actually moves between neighboring shells.
                        bounded_flux_columns.append(bounded_momentum_transfer)
                interface_flux_physical = torch.stack(bounded_flux_columns, dim=-1)
                transport_delta_physical.index_add_(0, transport_left_index, -interface_flux_physical)
                transport_delta_physical.index_add_(0, transport_right_index, interface_flux_physical)
            else:
                transport_cell_width_km = stats_torch.get("transport_cell_width_km")
                if transport_cell_width_km is None:
                    raise ValueError("flux_transport_head requires transport_cell_width_km in stats_torch.")
                if transport_cell_width_km.shape[0] != node_latent.shape[0]:
                    if node_latent.shape[0] % max(int(transport_cell_width_km.shape[0]), 1) != 0:
                        raise ValueError("transport_cell_width_km does not match the flattened node count.")
                    batch_repeats = node_latent.shape[0] // max(int(transport_cell_width_km.shape[0]), 1)
                    # The same shell geometry applies to every graph in the minibatch, so repeat the
                    # per-node cell widths before applying the finite-volume divergence.
                    transport_cell_width_km = transport_cell_width_km.repeat(batch_repeats)
                bounded_support_flux, _, _, support_flux_capped_fraction = self._bound_support_interface_flux(
                    support_transport_score=interface_flux_physical[:, support_local_index_int],
                    last_support_physical=last_support_physical,
                    transport_left_index=transport_left_index,
                    transport_right_index=transport_right_index,
                )
                bounded_flux_columns = []
                for local_offset in range(int(conservative_channel_index.numel())):
                    if local_offset == support_local_index_int:
                        bounded_flux_columns.append(bounded_support_flux)
                    else:
                        bounded_flux_columns.append(interface_flux_physical[:, local_offset])
                interface_flux_physical = torch.stack(bounded_flux_columns, dim=-1)
                left_cell_width = transport_cell_width_km.index_select(0, transport_left_index).unsqueeze(-1)
                right_cell_width = transport_cell_width_km.index_select(0, transport_right_index).unsqueeze(-1)
                safe_left_cell_width = torch.clamp(left_cell_width, min=1.0e-6)
                safe_right_cell_width = torch.clamp(right_cell_width, min=1.0e-6)
                left_flux_divergence = -interface_flux_physical / safe_left_cell_width
                right_flux_divergence = interface_flux_physical / safe_right_cell_width
                transport_delta_physical.index_add_(0, transport_left_index, left_flux_divergence)
                transport_delta_physical.index_add_(0, transport_right_index, right_flux_divergence)

        shell_mask = transport_shell_node_mask.to(dtype=node_latent.dtype).unsqueeze(-1)
        if self.cfg.operator_split_head:
            # In operator-split mode the source branch is responsible for local non-transport
            # effects, so the transport branch should remain a pure conservative flux operator.
            residual_conservative_physical = torch.zeros_like(transport_delta_physical)
        else:
            residual_conservative_physical = self.transport_residual_decoder(node_latent) * shell_mask

        last_conservative_physical = last_dynamic_physical.index_select(dim=-1, index=conservative_channel_index)
        next_conservative_physical = last_conservative_physical + transport_delta_physical + residual_conservative_physical
        next_conservative_model = self._physical_conservative_to_model_space(
            conservative_physical=next_conservative_physical,
            conservative_channel_index=conservative_channel_index,
            stats_torch=stats_torch,
        )
        last_conservative_model = last_dynamic.index_select(dim=-1, index=conservative_channel_index)
        conservative_delta_model = next_conservative_model - last_conservative_model
        delta_mean = stats_torch["delta_mean"].index_select(0, conservative_channel_index)
        delta_std = stats_torch["delta_std"].index_select(0, conservative_channel_index)
        conservative_pred_norm = (conservative_delta_model - delta_mean) / delta_std
        return conservative_pred_norm, {
            "transport_interface_flux_physical": interface_flux_physical,
            "transport_delta_physical": transport_delta_physical,
            "residual_conservative_physical": residual_conservative_physical,
            "next_conservative_physical": next_conservative_physical,
            "support_flux_capped_fraction": support_flux_capped_fraction,
            "support_stay_fraction_mean": support_stay_fraction_mean,
            "support_outflow_fraction_mean": support_outflow_fraction_mean,
            "support_outflow_cap_saturation_mean": support_outflow_cap_saturation_mean,
            "support_outflow_fraction_per_node": support_outflow_fraction_per_node,
        }

    def _decode_base_with_aux(
        self,
        node_latent: torch.Tensor,
        last_dynamic: torch.Tensor | None = None,
        boundary_context: torch.Tensor | None = None,
        stats_torch: dict[str, torch.Tensor] | None = None,
        predict_delta_state: bool | None = None,
        source_global_context: torch.Tensor | None = None,
        apply_boundary_injection: bool = True,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Decode latent shell states and optional closure-uncertainty auxiliaries.

        Physical interpretation:
        - The main decoder still predicts next-step shell-state channels.
        - The optional auxiliary output carries closure-channel uncertainty so future trainers can
          score sparse-shell transport targets with a heteroscedastic loss instead of a fixed MSE.
        """
        aux: dict[str, torch.Tensor] = {}
        source_node_latent = node_latent
        if self.cfg.operator_split_head:
            # Operator-split mode reserves line-global controls for the source branch only, which
            # matches the interpretation that `SmLat` and cusp/aurora state modulate source/heating
            # behavior rather than directly acting as shell-interface transport variables.
            source_node_latent = self._apply_global_context_conditioning(
                node_latent=node_latent,
                global_context=source_global_context,
            )
        if self.cfg.conditional_momentum_head:
            if last_dynamic is None or stats_torch is None or predict_delta_state is None:
                raise ValueError(
                    "conditional_momentum_head requires last_dynamic, stats_torch, and predict_delta_state."
                )
            output = torch.zeros(
                node_latent.shape[0],
                self.cfg.node_output_dim,
                dtype=node_latent.dtype,
                device=node_latent.device,
            )
            occupancy_shape_pred_norm = self.occupancy_shape_decoder(source_node_latent)
            forcing_pred_norm = self.forcing_decoder(source_node_latent)
            population_raw = self._predict_population_fraction_raw(
                occupancy_shape_pred_norm=occupancy_shape_pred_norm,
                last_dynamic=last_dynamic,
                stats_torch=stats_torch,
                predict_delta_state=predict_delta_state,
            ).squeeze(-1)
            # The smooth gate avoids discontinuous on/off behavior while still suppressing
            # momentum-like outputs in effectively empty shells.
            occupancy_gate = torch.sigmoid(
                (population_raw - float(self.cfg.occupancy_gate_center))
                / max(float(self.cfg.occupancy_gate_width), 1.0e-6)
            )
            momentum_input = torch.cat(
                (
                    source_node_latent,
                    occupancy_shape_pred_norm,
                    forcing_pred_norm,
                    occupancy_gate.unsqueeze(-1),
                ),
                dim=-1,
            )
            momentum_pred_norm = self.momentum_decoder(momentum_input)
            momentum_pred_norm = self._gate_momentum_prediction_norm(
                momentum_pred_norm=momentum_pred_norm,
                occupancy_gate=occupancy_gate,
                stats_torch=stats_torch,
                predict_delta_state=predict_delta_state,
            )
            output[..., self.occupancy_shape_channel_index] = occupancy_shape_pred_norm
            output[..., self.forcing_channel_index] = forcing_pred_norm
            output[..., self.momentum_channel_index] = momentum_pred_norm
            if self.cfg.uncertainty_aware_closure_head:
                aux["closure_logvar"] = self._clamp_closure_logvar(self.closure_logvar_decoder(momentum_input))
            if apply_boundary_injection:
                output, boundary_aux = self._apply_boundary_injection(
                    base_output=output,
                    node_latent=node_latent,
                    boundary_context=boundary_context,
                    stats_torch=stats_torch,
                )
                aux.update(boundary_aux)
            return output, aux

        if self.cfg.factorized_decoder_heads:
            output = torch.zeros(
                node_latent.shape[0],
                self.cfg.node_output_dim,
                dtype=node_latent.dtype,
                device=node_latent.device,
            )
            output[..., self.density_channel_index] = self.density_decoder(source_node_latent)
            output[..., self.topology_channel_index] = self.topology_decoder(source_node_latent)
            output[..., self.geometry_channel_index] = self.geometry_decoder(source_node_latent)
            output[..., self.momentum_channel_index] = self.momentum_decoder(source_node_latent)
            output[..., self.forcing_channel_index] = self.forcing_decoder(source_node_latent)
            if self.cfg.uncertainty_aware_closure_head:
                aux["closure_logvar"] = self._clamp_closure_logvar(self.closure_logvar_decoder(source_node_latent))
            if apply_boundary_injection:
                output, boundary_aux = self._apply_boundary_injection(
                    base_output=output,
                    node_latent=node_latent,
                    boundary_context=boundary_context,
                    stats_torch=stats_torch,
                )
                aux.update(boundary_aux)
            return output, aux

        if not self.cfg.split_decoder_heads:
            output = self.decoder(source_node_latent)
            if self.cfg.uncertainty_aware_closure_head:
                aux["closure_logvar"] = self._clamp_closure_logvar(self.closure_logvar_decoder(source_node_latent))
            if apply_boundary_injection:
                output, boundary_aux = self._apply_boundary_injection(
                    base_output=output,
                    node_latent=node_latent,
                    boundary_context=boundary_context,
                    stats_torch=stats_torch,
                )
                aux.update(boundary_aux)
            return output, aux

        decoder_latent = self.decoder_trunk(source_node_latent)
        output = torch.zeros(
            node_latent.shape[0],
            self.cfg.node_output_dim,
            dtype=node_latent.dtype,
            device=node_latent.device,
        )
        output[..., self.non_velocity_channel_index] = self.non_velocity_decoder(decoder_latent)
        output[..., self.signed_velocity_channel_index] = self.signed_velocity_decoder(decoder_latent)
        output[..., self.nonnegative_velocity_channel_index] = self.nonnegative_velocity_decoder(decoder_latent)
        if self.cfg.uncertainty_aware_closure_head:
            aux["closure_logvar"] = self._clamp_closure_logvar(self.closure_logvar_decoder(decoder_latent))
        if apply_boundary_injection:
            output, boundary_aux = self._apply_boundary_injection(
                base_output=output,
                node_latent=node_latent,
                boundary_context=boundary_context,
                stats_torch=stats_torch,
            )
            aux.update(boundary_aux)
        return output, aux

    def _decode_with_aux(
        self,
        node_latent: torch.Tensor,
        last_dynamic: torch.Tensor | None = None,
        boundary_context: torch.Tensor | None = None,
        source_global_context: torch.Tensor | None = None,
        last_dynamic_physical: torch.Tensor | None = None,
        stats_torch: dict[str, torch.Tensor] | None = None,
        predict_delta_state: bool | None = None,
        transport_left_index: torch.Tensor | None = None,
        transport_right_index: torch.Tensor | None = None,
        transport_interface_features: torch.Tensor | None = None,
        transport_shell_node_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Decode latent shell states and optionally overwrite the conservative pair via transport.

        Physical interpretation:
        - The base decoder still predicts the full shell-state update.
        - When enabled, the flux head replaces the conservative pair with a transport-plus-source
          update derived from shell-interface exchanges, which is closer to the 1-D field-line
          transport picture than direct node-wise regression.
        """
        if self.cfg.operator_split_head:
            if last_dynamic is None or last_dynamic_physical is None or stats_torch is None or predict_delta_state is None:
                raise ValueError("operator_split_head requires last_dynamic, last_dynamic_physical, stats_torch, and predict_delta_state.")
            source_output, aux = self._decode_base_with_aux(
                node_latent=node_latent,
                last_dynamic=last_dynamic,
                boundary_context=None,
                stats_torch=stats_torch,
                predict_delta_state=predict_delta_state,
                source_global_context=source_global_context,
                apply_boundary_injection=False,
            )
            conservative_channel_index = self.transport_conservative_channel_index
            if self.cfg.boundary_injection_head:
                boundary_update, boundary_aux = self._compute_boundary_injection_update(
                    node_latent=node_latent,
                    boundary_context=boundary_context,
                    stats_torch=stats_torch,
                )
                aux.update(boundary_aux)
                output = source_output + boundary_update
                aux["boundary_conservative_physical"] = self._model_delta_to_conservative_physical(
                    conservative_pred_norm=boundary_update.index_select(dim=-1, index=conservative_channel_index),
                    last_dynamic=last_dynamic,
                    last_dynamic_physical=last_dynamic_physical,
                    stats_torch=stats_torch,
                )
            else:
                boundary_update = torch.zeros_like(source_output)
                # When the boundary branch is disabled, the operator split reduces to transport
                # plus source only, so the boundary contribution must be exactly zero in both
                # model space and physical conservative diagnostics.
                output = source_output
                aux["boundary_injection_update"] = boundary_update
                aux["boundary_conservative_physical"] = torch.zeros_like(
                    last_dynamic_physical.index_select(dim=-1, index=conservative_channel_index)
                )
            aux["source_update_model"] = source_output
            aux["source_conservative_physical"] = self._model_delta_to_conservative_physical(
                conservative_pred_norm=source_output.index_select(dim=-1, index=conservative_channel_index),
                last_dynamic=last_dynamic,
                last_dynamic_physical=last_dynamic_physical,
                stats_torch=stats_torch,
            )
        else:
            output, aux = self._decode_base_with_aux(
                node_latent=node_latent,
                last_dynamic=last_dynamic,
                boundary_context=boundary_context,
                stats_torch=stats_torch,
                predict_delta_state=predict_delta_state,
                source_global_context=None,
                apply_boundary_injection=True,
            )
        if not self.cfg.flux_transport_head:
            return output, aux

        if (
            last_dynamic is None
            or last_dynamic_physical is None
            or stats_torch is None
            or predict_delta_state is None
            or transport_left_index is None
            or transport_right_index is None
            or transport_interface_features is None
            or transport_shell_node_mask is None
        ):
            raise ValueError("flux_transport_head requires last_dynamic, last_dynamic_physical, stats_torch, and transport metadata.")

        conservative_pred_norm, transport_aux = self._build_flux_transport_update(
            node_latent=node_latent,
            last_dynamic=last_dynamic,
            last_dynamic_physical=last_dynamic_physical,
            stats_torch=stats_torch,
            transport_left_index=transport_left_index,
            transport_right_index=transport_right_index,
            transport_interface_features=transport_interface_features,
            transport_shell_node_mask=transport_shell_node_mask,
        )
        conservative_channel_index = self.transport_conservative_channel_index
        if self.cfg.operator_split_head:
            # The explicit operator split adds the transport contribution onto the conservative
            # channels already produced by the source and boundary branches.
            output = self._apply_conservative_channel_update(
                base_output=output,
                conservative_channel_index=conservative_channel_index,
                conservative_update=conservative_pred_norm,
                additive=True,
            )
        else:
            # Legacy flux mode overwrites the conservative pair entirely with the transport branch.
            output = self._apply_conservative_channel_update(
                base_output=output,
                conservative_channel_index=conservative_channel_index,
                conservative_update=conservative_pred_norm,
                additive=False,
            )
        aux.update(transport_aux)
        return output, aux

    def _decode(
        self,
        node_latent: torch.Tensor,
        last_dynamic: torch.Tensor | None = None,
        boundary_context: torch.Tensor | None = None,
        source_global_context: torch.Tensor | None = None,
        last_dynamic_physical: torch.Tensor | None = None,
        stats_torch: dict[str, torch.Tensor] | None = None,
        predict_delta_state: bool | None = None,
        transport_left_index: torch.Tensor | None = None,
        transport_right_index: torch.Tensor | None = None,
        transport_interface_features: torch.Tensor | None = None,
        transport_shell_node_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Decode latent shell states into the ordered output channel vector."""
        output, _ = self._decode_with_aux(
            node_latent=node_latent,
            last_dynamic=last_dynamic,
            boundary_context=boundary_context,
            source_global_context=source_global_context,
            last_dynamic_physical=last_dynamic_physical,
            stats_torch=stats_torch,
            predict_delta_state=predict_delta_state,
            transport_left_index=transport_left_index,
            transport_right_index=transport_right_index,
            transport_interface_features=transport_interface_features,
            transport_shell_node_mask=transport_shell_node_mask,
        )
        return output

    def _forward_flat(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        global_context: torch.Tensor | None = None,
        last_dynamic: torch.Tensor | None = None,
        boundary_context: torch.Tensor | None = None,
        last_dynamic_physical: torch.Tensor | None = None,
        stats_torch: dict[str, torch.Tensor] | None = None,
        predict_delta_state: bool | None = None,
        transport_left_index: torch.Tensor | None = None,
        transport_right_index: torch.Tensor | None = None,
        transport_interface_features: torch.Tensor | None = None,
        transport_shell_node_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forecast shell-state outputs for one flattened graph batch."""
        node_latent = self.node_encoder(node_features)
        if not self.cfg.operator_split_head:
            node_latent = self._apply_global_context_conditioning(
                node_latent=node_latent,
                global_context=global_context,
            )
        for layer in self.processor:
            node_latent = layer(node_latent=node_latent, edge_index=edge_index)
        return self._decode(
            node_latent=node_latent,
            last_dynamic=last_dynamic,
            boundary_context=boundary_context,
            source_global_context=global_context,
            last_dynamic_physical=last_dynamic_physical,
            stats_torch=stats_torch,
            predict_delta_state=predict_delta_state,
            transport_left_index=transport_left_index,
            transport_right_index=transport_right_index,
            transport_interface_features=transport_interface_features,
            transport_shell_node_mask=transport_shell_node_mask,
        )

    def _forward_flat_with_aux(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        global_context: torch.Tensor | None = None,
        last_dynamic: torch.Tensor | None = None,
        boundary_context: torch.Tensor | None = None,
        last_dynamic_physical: torch.Tensor | None = None,
        stats_torch: dict[str, torch.Tensor] | None = None,
        predict_delta_state: bool | None = None,
        transport_left_index: torch.Tensor | None = None,
        transport_right_index: torch.Tensor | None = None,
        transport_interface_features: torch.Tensor | None = None,
        transport_shell_node_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Forecast shell-state outputs together with optional closure auxiliaries."""
        node_latent = self.node_encoder(node_features)
        if not self.cfg.operator_split_head:
            node_latent = self._apply_global_context_conditioning(
                node_latent=node_latent,
                global_context=global_context,
            )
        for layer in self.processor:
            node_latent = layer(node_latent=node_latent, edge_index=edge_index)
        return self._decode_with_aux(
            node_latent=node_latent,
            last_dynamic=last_dynamic,
            boundary_context=boundary_context,
            source_global_context=global_context,
            last_dynamic_physical=last_dynamic_physical,
            stats_torch=stats_torch,
            predict_delta_state=predict_delta_state,
            transport_left_index=transport_left_index,
            transport_right_index=transport_right_index,
            transport_interface_features=transport_interface_features,
            transport_shell_node_mask=transport_shell_node_mask,
        )

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        global_context: torch.Tensor | None = None,
        last_dynamic: torch.Tensor | None = None,
        boundary_context: torch.Tensor | None = None,
        last_dynamic_physical: torch.Tensor | None = None,
        stats_torch: dict[str, torch.Tensor] | None = None,
        predict_delta_state: bool | None = None,
    ) -> torch.Tensor:
        """Forecast shell-state outputs for one graph or a batch of identical-topology graphs."""
        if node_features.ndim == 2:
            # One graph snapshot uses the base edge index directly.
            single_global_context = global_context
            if single_global_context is not None and single_global_context.ndim == 1:
                single_global_context = single_global_context.unsqueeze(0).expand(node_features.shape[0], -1)
            return self._forward_flat(
                node_features=node_features,
                edge_index=edge_index,
                global_context=single_global_context,
                last_dynamic=last_dynamic,
                boundary_context=boundary_context,
                last_dynamic_physical=last_dynamic_physical,
                stats_torch=stats_torch,
                predict_delta_state=predict_delta_state,
                transport_left_index=None if stats_torch is None else stats_torch.get("transport_left_index"),
                transport_right_index=None if stats_torch is None else stats_torch.get("transport_right_index"),
                transport_interface_features=None if stats_torch is None else stats_torch.get("transport_interface_features"),
                transport_shell_node_mask=None if stats_torch is None else stats_torch.get("transport_shell_node_mask"),
            )

        if node_features.ndim != 3:
            raise ValueError(
                f"Expected node_features with 2 or 3 dims, got shape={tuple(node_features.shape)}."
            )

        batch_size, num_nodes, feature_dim = node_features.shape
        del feature_dim

        # Batched graphs share one topology, so the edge list is repeated with a node-offset per
        # batch element instead of depending on an external graph library.
        flat_features = node_features.reshape(batch_size * num_nodes, -1)
        flat_global_context = None
        if global_context is not None:
            if global_context.ndim == 1:
                global_context = global_context.unsqueeze(0)
            flat_global_context = (
                global_context.unsqueeze(1)
                .expand(batch_size, num_nodes, -1)
                .reshape(batch_size * num_nodes, -1)
            )
        flat_last_dynamic = None
        if last_dynamic is not None:
            flat_last_dynamic = last_dynamic.reshape(batch_size * num_nodes, -1)
        flat_boundary_context = None
        if boundary_context is not None:
            flat_boundary_context = boundary_context.reshape(batch_size * num_nodes, -1)
        flat_last_dynamic_physical = None
        if last_dynamic_physical is not None:
            flat_last_dynamic_physical = last_dynamic_physical.reshape(batch_size * num_nodes, -1)
        if edge_index.numel() == 0:
            batched_edge_index = edge_index
        else:
            offsets = (
                torch.arange(batch_size, device=edge_index.device, dtype=edge_index.dtype)
                .view(batch_size, 1, 1)
                * num_nodes
            )
            batched_edge_index = (edge_index.unsqueeze(0) + offsets).permute(1, 0, 2).reshape(2, -1)

        batched_transport_left_index = None
        batched_transport_right_index = None
        batched_transport_interface_features = None
        batched_transport_shell_node_mask = None
        if self.cfg.flux_transport_head and stats_torch is not None:
            single_transport_left_index = stats_torch.get("transport_left_index")
            single_transport_right_index = stats_torch.get("transport_right_index")
            single_transport_interface_features = stats_torch.get("transport_interface_features")
            single_transport_shell_node_mask = stats_torch.get("transport_shell_node_mask")
            if (
                single_transport_left_index is not None
                and single_transport_right_index is not None
                and single_transport_interface_features is not None
                and single_transport_shell_node_mask is not None
            ):
                transport_offsets = (
                    torch.arange(batch_size, device=edge_index.device, dtype=edge_index.dtype)
                    .view(batch_size, 1)
                    * num_nodes
                )
                if single_transport_left_index.numel() > 0:
                    batched_transport_left_index = (
                        single_transport_left_index.unsqueeze(0) + transport_offsets
                    ).reshape(-1)
                    batched_transport_right_index = (
                        single_transport_right_index.unsqueeze(0) + transport_offsets
                    ).reshape(-1)
                    batched_transport_interface_features = (
                        single_transport_interface_features.unsqueeze(0)
                        .expand(batch_size, -1, -1)
                        .reshape(batch_size * single_transport_interface_features.shape[0], -1)
                    )
                else:
                    batched_transport_left_index = single_transport_left_index
                    batched_transport_right_index = single_transport_right_index
                    batched_transport_interface_features = single_transport_interface_features
                batched_transport_shell_node_mask = (
                    single_transport_shell_node_mask.unsqueeze(0)
                    .expand(batch_size, -1)
                    .reshape(batch_size * num_nodes)
                )

        flat_output = self._forward_flat(
            node_features=flat_features,
            edge_index=batched_edge_index,
            global_context=flat_global_context,
            last_dynamic=flat_last_dynamic,
            boundary_context=flat_boundary_context,
            last_dynamic_physical=flat_last_dynamic_physical,
            stats_torch=stats_torch,
            predict_delta_state=predict_delta_state,
            transport_left_index=batched_transport_left_index,
            transport_right_index=batched_transport_right_index,
            transport_interface_features=batched_transport_interface_features,
            transport_shell_node_mask=batched_transport_shell_node_mask,
        )
        return flat_output.reshape(batch_size, num_nodes, -1)

    def forward_with_aux(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        global_context: torch.Tensor | None = None,
        last_dynamic: torch.Tensor | None = None,
        boundary_context: torch.Tensor | None = None,
        last_dynamic_physical: torch.Tensor | None = None,
        stats_torch: dict[str, torch.Tensor] | None = None,
        predict_delta_state: bool | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Forecast shell-state outputs and optional closure uncertainty tensors.

        Physical interpretation:
        - This method keeps the stable `forward()` signature untouched for current trainers.
        - Future closure-aware trainers can call it to retrieve per-node closure log-variance
          predictions alongside the standard shell-state output tensor.
        """
        if node_features.ndim == 2:
            single_global_context = global_context
            if single_global_context is not None and single_global_context.ndim == 1:
                single_global_context = single_global_context.unsqueeze(0).expand(node_features.shape[0], -1)
            return self._forward_flat_with_aux(
                node_features=node_features,
                edge_index=edge_index,
                global_context=single_global_context,
                last_dynamic=last_dynamic,
                boundary_context=boundary_context,
                last_dynamic_physical=last_dynamic_physical,
                stats_torch=stats_torch,
                predict_delta_state=predict_delta_state,
                transport_left_index=None if stats_torch is None else stats_torch.get("transport_left_index"),
                transport_right_index=None if stats_torch is None else stats_torch.get("transport_right_index"),
                transport_interface_features=None if stats_torch is None else stats_torch.get("transport_interface_features"),
                transport_shell_node_mask=None if stats_torch is None else stats_torch.get("transport_shell_node_mask"),
            )

        if node_features.ndim != 3:
            raise ValueError(
                f"Expected node_features with 2 or 3 dims, got shape={tuple(node_features.shape)}."
            )

        batch_size, num_nodes, feature_dim = node_features.shape
        del feature_dim
        flat_features = node_features.reshape(batch_size * num_nodes, -1)
        flat_global_context = None
        if global_context is not None:
            if global_context.ndim == 1:
                global_context = global_context.unsqueeze(0)
            flat_global_context = (
                global_context.unsqueeze(1)
                .expand(batch_size, num_nodes, -1)
                .reshape(batch_size * num_nodes, -1)
            )
        flat_last_dynamic = None
        if last_dynamic is not None:
            flat_last_dynamic = last_dynamic.reshape(batch_size * num_nodes, -1)
        flat_boundary_context = None
        if boundary_context is not None:
            flat_boundary_context = boundary_context.reshape(batch_size * num_nodes, -1)
        flat_last_dynamic_physical = None
        if last_dynamic_physical is not None:
            flat_last_dynamic_physical = last_dynamic_physical.reshape(batch_size * num_nodes, -1)
        if edge_index.numel() == 0:
            batched_edge_index = edge_index
        else:
            offsets = (
                torch.arange(batch_size, device=edge_index.device, dtype=edge_index.dtype)
                .view(batch_size, 1, 1)
                * num_nodes
            )
            batched_edge_index = (edge_index.unsqueeze(0) + offsets).permute(1, 0, 2).reshape(2, -1)

        batched_transport_left_index = None
        batched_transport_right_index = None
        batched_transport_interface_features = None
        batched_transport_shell_node_mask = None
        if self.cfg.flux_transport_head and stats_torch is not None:
            single_transport_left_index = stats_torch.get("transport_left_index")
            single_transport_right_index = stats_torch.get("transport_right_index")
            single_transport_interface_features = stats_torch.get("transport_interface_features")
            single_transport_shell_node_mask = stats_torch.get("transport_shell_node_mask")
            if (
                single_transport_left_index is not None
                and single_transport_right_index is not None
                and single_transport_interface_features is not None
                and single_transport_shell_node_mask is not None
            ):
                transport_offsets = (
                    torch.arange(batch_size, device=edge_index.device, dtype=edge_index.dtype)
                    .view(batch_size, 1)
                    * num_nodes
                )
                if single_transport_left_index.numel() > 0:
                    batched_transport_left_index = (
                        single_transport_left_index.unsqueeze(0) + transport_offsets
                    ).reshape(-1)
                    batched_transport_right_index = (
                        single_transport_right_index.unsqueeze(0) + transport_offsets
                    ).reshape(-1)
                    batched_transport_interface_features = (
                        single_transport_interface_features.unsqueeze(0)
                        .expand(batch_size, -1, -1)
                        .reshape(batch_size * single_transport_interface_features.shape[0], -1)
                    )
                else:
                    batched_transport_left_index = single_transport_left_index
                    batched_transport_right_index = single_transport_right_index
                    batched_transport_interface_features = single_transport_interface_features
                batched_transport_shell_node_mask = (
                    single_transport_shell_node_mask.unsqueeze(0)
                    .expand(batch_size, -1)
                    .reshape(batch_size * num_nodes)
                )

        flat_output, flat_aux = self._forward_flat_with_aux(
            node_features=flat_features,
            edge_index=batched_edge_index,
            global_context=flat_global_context,
            last_dynamic=flat_last_dynamic,
            boundary_context=flat_boundary_context,
            last_dynamic_physical=flat_last_dynamic_physical,
            stats_torch=stats_torch,
            predict_delta_state=predict_delta_state,
            transport_left_index=batched_transport_left_index,
            transport_right_index=batched_transport_right_index,
            transport_interface_features=batched_transport_interface_features,
            transport_shell_node_mask=batched_transport_shell_node_mask,
        )
        batched_output = flat_output.reshape(batch_size, num_nodes, -1)
        batched_aux: dict[str, torch.Tensor] = {}
        single_interface_count = 0
        if stats_torch is not None and stats_torch.get("transport_left_index") is not None:
            single_interface_count = int(stats_torch["transport_left_index"].shape[0])
        for aux_name, aux_value in flat_aux.items():
            if aux_value.ndim == 0:
                # Scalar auxiliaries summarize the whole batched graph (for example the fraction
                # of support-interface fluxes that had to be capped), so they should stay scalar
                # instead of being forced into a node/interface-shaped tensor.
                batched_aux[aux_name] = aux_value
                continue
            if aux_value.shape[0] == batch_size * num_nodes:
                batched_aux[aux_name] = aux_value.reshape(batch_size, num_nodes, -1)
            elif single_interface_count > 0 and aux_value.shape[0] == batch_size * single_interface_count:
                # Interface flux auxiliaries live on shell interfaces, not nodes.
                batched_aux[aux_name] = aux_value.reshape(batch_size, single_interface_count, -1)
            else:
                raise ValueError(
                    f"Cannot reshape auxiliary '{aux_name}' with leading dimension {aux_value.shape[0]} "
                    f"for batch_size={batch_size}, num_nodes={num_nodes}, interfaces={single_interface_count}."
                )
        return batched_output, batched_aux
