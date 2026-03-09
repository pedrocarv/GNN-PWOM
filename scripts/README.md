# Scripts Overview

This folder contains the runnable training, evaluation, and plotting implementations.

## Core Training and Evaluation

- `train_fieldline_graph.py`: one-step trainer.
- `evaluate_fieldline_graph.py`: one-step evaluator.
- `train_fieldline_graph_rollout.py`: rollout fine-tuning.
- `evaluate_fieldline_graph_rollout.py`: rollout evaluator.

## Diagnostics and Plots

- `plot_fieldline_shell_profiles.py`: shell-profile plotting.
- `plot_species_vpar_focus.py`: species-wise `v_parallel` plotting.

## Typical Execution Order

1. Train one-step model:
   `python scripts/train_fieldline_graph.py ...`
2. Evaluate one-step model:
   `python scripts/evaluate_fieldline_graph.py --checkpoint runs/<run>/best.pt`
3. (Optional) Fine-tune for rollout:
   `python scripts/train_fieldline_graph_rollout.py --checkpoint runs/<run>/best.pt ...`
4. Evaluate rollout model:
   `python scripts/evaluate_fieldline_graph_rollout.py --checkpoint runs/<rollout_run>/best.pt`

Shared data/model code lives in `pwom_surrogate/`; scripts are organized by workflow stage.
The one-step trainer is split across:
- `scripts/train_fieldline_graph.py`: top-level orchestration and entrypoint
- `pwom_surrogate/train_fieldline_core.py`: reusable training primitives, objectives, and epoch loop
