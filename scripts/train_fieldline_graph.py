#!/usr/bin/env python3
"""Train the one-step field-line surrogate with a concise top-level orchestration flow.

This script intentionally keeps `main()` close to the top-level control flow.
Detailed training primitives and loss implementations live in the core training module.
"""

from __future__ import annotations

import os
import sys

# Allow running this script directly from the repository root.
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Re-export core helpers used by scripts that import from `train_fieldline_graph`.
from pwom_surrogate.train_fieldline_core import *  # noqa: F401,F403
from pwom_surrogate.train_fieldline_core import (
    _build_run_configs,
    _initialize_runtime,
    _prepare_training_data,
    _run_training_loop,
    _stats_to_torch,
    _validate_training_args,
    _write_training_config,
)


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
