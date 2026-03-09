"""Configuration surface for the PWOM surrogate workspace.

Physical interpretation:
- `DataConfig` describes how a PWOM field line is coarse-grained into graph nodes.
- `ModelConfig` describes the graph forecaster that propagates coarse shell dynamics.
- `TrainConfig` describes rollout depth and optimization behavior for surrogate training.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DataConfig:
    """Dataset and coarse-graining parameters for field-line graphs."""

    # The data directory points at the shared PWOM binary history used by the rebuild workspace.
    data_dir: str = "../pwom-surrogate-rebuild/dataset"
    # A particle subset keeps preprocessing bounded while still representing the distribution shape.
    num_particles: int | None = 512
    # Altitude shells define the graph nodes along the field line.
    num_altitude_shells: int = 32
    # Temporal history length sets the Markov context available to the surrogate.
    window_length: int = 8
    # Rank-based alignment remains a sampling policy only; supervision is distribution-level.
    temporal_alignment: str = "species_altitude_rank"
    # A global node lets exogenous forcing act on every shell in one message-passing step.
    include_global_node: bool = True
    # Cross-species shell coupling is needed because chemistry and collisions are species dependent.
    include_species_coupling_edges: bool = True
    # `None` means species stacks are inferred from the sampled split, which stays robust to
    # datasets that do not contain all species in every run.
    species_ids: tuple[int, ...] | None = None
    # The target schema controls how shell dynamics are parameterized for the learning objective.
    # The default keeps the current population-weighted moment targets, while future branches can
    # switch to closure/transport-style targets without changing the coarse field-line graph.
    target_schema: str = "population_weighted"


@dataclass
class ModelConfig:
    """Model hyperparameters for the field-line graph forecaster."""

    # Node inputs summarize shell-level moments and forcing-conditioned state descriptors.
    node_input_dim: int = 16
    # Line-global PWOM controls use a separate conditioning path instead of being mixed into node
    # state, because latitude/timestep/regime inputs are not shell-resolved plasma variables.
    global_context_dim: int = 0
    # Node outputs predict the next-step change of the same shell descriptors.
    node_output_dim: int = 16
    # Hidden width controls message capacity across neighboring shells and species.
    hidden_dim: int = 128
    # Multiple message-passing layers allow transport information to move along the field line.
    message_passing_layers: int = 4
    # A small MLP depth keeps the model interpretable.
    mlp_layers: int = 2
    # Residual dynamics are the default because shell states evolve incrementally between snapshots.
    predict_delta_state: bool = True
    # Split decoder heads let velocity moments use a dedicated output branch instead of competing
    # directly with occupancy and geometry channels in one shared linear head.
    split_decoder_heads: bool = False
    # Factorized decoder heads go one step further and separate density, geometry, momentum, and
    # forcing groups so errors in one physical family do not have to share the same decoder path.
    factorized_decoder_heads: bool = False
    # The conditional momentum head predicts occupancy/shape first and then predicts momentum-like
    # channels conditioned on that intermediate shell state.
    conditional_momentum_head: bool = False
    # The target schema tells the decoder which physical parameterization the output channels use.
    # This keeps closure-aware branches explicit instead of overloading one set of channel names.
    target_schema: str = "population_weighted"
    # An uncertainty-aware closure head predicts per-channel log-variance for closure/transport
    # channels so sparse shells can express lower confidence instead of forcing one fixed MSE scale.
    uncertainty_aware_closure_head: bool = False
    # The log-variance clamp prevents the uncertainty head from collapsing into numerically extreme
    # values before the closure-aware trainer is wired into the full optimization stack.
    closure_logvar_min: float = -8.0
    # The upper clamp keeps predicted uncertainty finite while still allowing noisy shells to be
    # downweighted relative to dense, well-resolved shells.
    closure_logvar_max: float = 6.0
    # The gate center marks the population-fraction scale where shell momentum predictions should
    # activate smoothly instead of staying near zero in effectively empty shells.
    occupancy_gate_center: float = 1.0e-3
    # The gate width controls how sharply the conditional momentum head turns on with occupancy.
    occupancy_gate_width: float = 5.0e-4
    # The flux transport head predicts same-species shell-interface exchanges whose discrete
    # divergence updates the conservative shell pair before the residual decoder adds local sources.
    flux_transport_head: bool = False
    # A dedicated hidden width lets the interface flux branch stay lightweight even when the main
    # node latent is wider.
    flux_hidden_dim: int = 128
    # A small context encoder maps line-global PWOM controls into the same latent space as the
    # node encoder so exogenous line drivers can bias every shell through one broadcast addition.
    global_context_hidden_dim: int = 128
    # FiLM-style global conditioning treats line-global PWOM controls as regime modulators rather
    # than as pseudo-state features. This matches `SmLat` and cusp/aurora flags better than a
    # plain additive context embedding.
    global_context_film: bool = False
    # The boundary injection head models lower-boundary replenishment separately from interior
    # shell evolution, which is closer to how PWOM imposes boundary moments.
    boundary_injection_head: bool = False
    # A dedicated boundary hidden width lets the lower-boundary source branch stay lightweight and
    # interpretable even when the main graph latent is wider.
    boundary_hidden_dim: int = 128
    # The operator-split head makes the one-step update explicit as transport plus boundary plus
    # source terms, instead of letting one decoder implicitly absorb all three mechanisms.
    operator_split_head: bool = False
    # The redistribution transport head predicts shellwise up/down/stay fractions plus carried
    # parallel velocities, so shell support is redistributed conservatively by construction.
    redistribution_transport_head: bool = False
    # The bounded outflow-fraction transport variant predicts a signed fraction of local shell
    # support to move across each interface, instead of an unconstrained per-face support flux.
    bounded_outflow_fraction_transport: bool = False
    # The simplex transport variant predicts shellwise `up/down/stay` fractions that sum to one.
    # Physically, this makes support preservation native to the parameterization instead of being
    # enforced by post-hoc clipping of an unconstrained transport score. It is kept as an alias
    # for the redistribution-transport branch.
    simplex_support_transport: bool = False
    # This caps the total shell support that can be exported in one explicit transport step.
    # Physically, it prevents a shell from draining nearly all represented particles in one update.
    max_support_outflow_fraction: float = 0.5


@dataclass
class TrainConfig:
    """Training controls for rollout-aware field-line graph learning."""

    # Epoch count controls how long optimization runs.
    epochs: int = 100
    # Batch size is expressed in graph windows, not individual particles.
    batch_size: int = 8
    # Learning rate controls optimizer step size.
    lr: float = 1e-4
    # Weight decay is small because rollout stability is more important than aggressive regularization.
    weight_decay: float = 1e-6
    # Rollout horizon should be increased by curriculum rather than all at once.
    horizon: int = 3
    # Rollout loss keeps the graph model honest under autoregressive use.
    rollout_weight: float = 0.2
    # Teacher forcing stabilizes early autoregressive training by mixing true and predicted shell
    # states when updating the rollout history.
    teacher_forcing_prob: float = 0.0
    # Gradient clipping protects long-rollout training from numerical bursts.
    grad_clip_norm: float = 1.0
    # Worker count controls host-side batching only; training can run with zero workers.
    num_workers: int = 0
    # Early stopping prevents wasting time once validation loss stops improving.
    early_stopping_patience: int = 20
    # A small minimum delta avoids saving checkpoints for purely numerical jitter.
    early_stopping_min_delta: float = 1e-4
    # Species-aware velocity balancing optionally upweights underrepresented ion populations in the
    # velocity-moment loss so the surrogate does not overfit the dominant species only.
    velocity_species_balance_power: float = 0.0
    # A cap keeps extremely rare species from dominating the regression objective through inverse
    # population weighting alone.
    velocity_species_balance_max_weight: float = 4.0
    # Momentum losses are only physically meaningful where shell occupancy is large enough to
    # define a species-conditioned velocity distribution.
    occupied_shell_threshold: float = 1.0e-3
    # The momentum branch can be weighted separately from occupancy/shape and forcing losses.
    momentum_loss_weight: float = 1.0
    # Closure/transport targets can be weighted separately when the target schema switches away
    # from direct population-weighted moments to uncertainty-aware closure regression.
    closure_loss_weight: float = 1.0
    # Flux-consistency supervision compares the conservative transport update inferred from
    # interface exchanges against the true one-step conservative shell increment.
    flux_loss_weight: float = 0.0
    # The direct conservative-pair loss is kept smaller than the non-conservative state loss when
    # the flux head is active, because support and parallel-momentum updates should be explained
    # primarily by interface transport rather than by a generic per-node regression path.
    conservative_state_loss_weight: float = 0.25
    # Non-conservative channels such as geometry, topology, and perpendicular moments still need
    # strong node-wise supervision because the flux branch does not model their shell-to-shell
    # transport explicitly.
    non_conservative_state_loss_weight: float = 1.0
    # The conservative residual penalty keeps the flux branch responsible for as much of the
    # support/parallel-drift update as possible while still allowing local source terms.
    residual_conservative_weight: float = 0.0
    # The source conservative penalty keeps the generic source branch from absorbing shell-to-shell
    # transport that should instead be explained by the explicit flux operator.
    source_conservative_weight: float = 0.0
    # The redistribution branch needs an additional support-band penalty that keeps the operator-
    # split source term smaller than the transported conservative update in the diagnosed collapse
    # band, otherwise the transport decoder can converge to a trivial all-stay solution.
    support_band_source_suppression_weight: float = 1.0
    # A mild saturation penalty discourages the redistribution branch from parking at the maximum
    # allowed one-step outflow fraction on every shell once the hard cap is introduced.
    support_outflow_saturation_weight: float = 0.05
    # Lower-boundary supervision keeps the boundary injection branch focused on the shells where
    # PWOM boundary moments actually enter the particle system.
    boundary_source_loss_weight: float = 0.0
    # The support-band loss focuses extra supervision on the altitude region where rollout
    # diagnostics showed shell population support collapsing and dragging velocity estimates down.
    support_band_loss_weight: float = 0.0
    # The lower edge of the problematic altitude band in kilometers.
    support_band_lower_km: float = 2900.0
    # The upper edge of the primary problematic altitude band in kilometers.
    support_band_upper_km: float = 4700.0
    # A second weaker support-band loss can target the residual upper-altitude support collapse
    # without broadening the primary band that already works well in the mid-altitude regime.
    secondary_support_band_loss_weight: float = 0.0
    # Lower edge of the residual upper-altitude failure band in kilometers.
    secondary_support_band_lower_km: float = 6900.0
    # Upper edge of the residual upper-altitude failure band in kilometers.
    secondary_support_band_upper_km: float = 7800.0
    # A localized occupied-shell mean_vpar corrective term targets the persistent transport error
    # inside the primary support-collapse band without perturbing the rest of the field line.
    localized_support_band_mean_vpar_loss_weight: float = 0.0
    # A small auxiliary loss on recovered mean parallel velocity can improve velocity diagnostics
    # without replacing the main shell-state objective.
    diagnostic_mean_vpar_loss_weight: float = 0.0
    # A matching occupied-shell mean_vperp loss helps the redistribution branch keep outward
    # support transport from drifting into the wrong perpendicular-velocity regime.
    diagnostic_mean_vperp_loss_weight: float = 0.0
    # A matching auxiliary loss on recovered parallel spread can improve velocity-width fidelity
    # in occupied shells without forcing undefined empty-shell diagnostics.
    diagnostic_std_vpar_loss_weight: float = 0.0
    # Primitive diagnostic losses can optionally be restricted to the primary support-collapse band
    # so the velocity correction acts only in the altitude regime where rollout drift is known to
    # matter physically instead of perturbing the whole field line.
    diagnostic_band_mask_mode: str = "global"
    # Primitive diagnostic losses can be warmed in gradually so the model first locks onto the
    # stable shell-state objective before a small occupied-shell velocity correction is applied.
    diagnostic_loss_warmup_epochs: int = 0
    # The warmup starts at this fraction of the requested diagnostic weight and ramps linearly to
    # the full value over `diagnostic_loss_warmup_epochs`.
    diagnostic_loss_warmup_start_factor: float = 0.0
    # Rollout checkpoint selection can follow the optimized objective, a normalized composite score,
    # or raw physical diagnostic MSE recovered globally or inside the
    # primary support-collapse band.
    selection_metric: str = "valid_total_loss"
    # Some rollout failures are species-specific, so selection can optionally focus on one species
    # inside the support-collapse band instead of averaging across all species.
    selection_focus_species_id: int | None = None
    # A constrained species-focus artifact should only be saved when rollout loss remains inside a
    # user-defined physically acceptable range, so local velocity gains cannot come from a broken
    # global rollout.
    selection_focus_rollout_loss_cap: float | None = None
    # The rollout component stays dominant in the composite score because temporal stability is the
    # core surrogate requirement once one-step behavior is already acceptable.
    selection_rollout_loss_weight: float = 1.0
    # A small occupied-shell mean_vpar term biases checkpoint choice toward better parallel-drift
    # recovery without fully sacrificing the stable rollout solution.
    selection_mean_vpar_weight: float = 0.02
    # A weaker occupied-shell std_vpar term helps separate good drift checkpoints from those that
    # only look good because shell support collapsed.
    selection_std_vpar_weight: float = 0.01
    # Optional per-epoch checkpoints make it possible to inspect or recover epochs that are best
    # under one diagnostic but not under the main selection score.
    save_epoch_checkpoints: bool = False
    # Seed keeps sampling and initialization reproducible across comparisons.
    seed: int = 42
