# PWOM Surrogate Model

## What Is PWOM?

The **Polar Wind Outflow Model (PWOM)** is a first-principles model of ion outflow from Earth’s polar ionosphere into the magnetosphere along magnetic field lines. It captures multi-species plasma behavior, altitude-dependent transport, and externally driven forcing. The model has a hybrid-DSMC approach, where in higher altitudes it switches from a fluid tretament to a kinetic treatment, which has proven to be quite expensive computationally. 

## Project Purpose

This repository implements a machine-learning surrogate that approximates PWOM field-line dynamics in the altitudes where particle simulation is needed. The goal is to preserve key transport and diagnostic behavior while reducing runtime for model development workflows and downstream integration.

This repository is still evolving, so defaults and experiment settings may change as new ideas are tested.

## Modeling Approach

The surrogate is formulated as a graph-learning problem, where each graph represents one PWOM field line at one time.

- **Shell**: one altitude bin along the field line (for example, 2000-2500 km).  
  For each ion species, the field line is split into multiple shells.  
  A node therefore represents one `(species, altitude_shell)` cell.
- **Topology**: the connectivity pattern between nodes.  
  In this project, topology mainly means:
  - links between neighboring altitude shells of the same species (transport along the field line),
  - optional links between different species at similar altitude (cross-species coupling),
  - optional links to one global context node (shared drivers/forcing).

The model reads a short history of shell states and predicts the next shell state. It is trained first in one-step mode, with optional rollout fine-tuning for autoregressive stability.

## Repository Layout

- Core package directory (`pwom_surrogate_*`): data transforms, normalization, model, configuration dataclasses
- `scripts/`: training, evaluation, rollout, and plotting programs
- `tests/`: topology and contract-focused tests
- `docs/`: concise workflow and architecture notes
- `shell_layouts/`: shell edge presets for non-uniform altitude grids
- `dataset/`: local data placeholder and notes (not tracked)
- `runs/`: checkpoints, histories, and evaluation outputs (not tracked)

## Environment Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install pytest
```

Recommended Python version: `3.10+`.  
Device selection is CLI-controlled (`--device auto|cpu|mps|cuda`).

## Training and Evaluation Workflow

### 1) Train one-step model

```bash
python scripts/train_fieldline_graph.py \
  --data-dir <path-to-dataset> \
  --output-dir runs/fieldline_onestep_baseline \
  --epochs 80 \
  --batch-size 8 \
  --target-schema population_weighted \
  --device auto
```

Useful knobs:
- architecture: `--hidden-dim`, `--message-passing-layers`, `--mlp-layers`
- transport variants: `--operator-split-head`, `--redistribution-transport-head`, `--bounded-outflow-fraction-transport`
- loss shaping: `--flux-loss-weight`, `--conservative-state-loss-weight`, `--support-band-loss-weight`

### 2) Evaluate checkpoint

```bash
python scripts/evaluate_fieldline_graph.py \
  --checkpoint runs/fieldline_onestep_baseline/best.pt \
  --split valid \
  --output-json runs/fieldline_onestep_baseline/eval_best.json
```

### 3) Optional rollout fine-tuning

```bash
python scripts/train_fieldline_graph_rollout.py \
  --checkpoint runs/fieldline_onestep_baseline/best.pt \
  --output-dir runs/fieldline_rollout_h3 \
  --horizon 3 \
  --rollout-weight 0.2 \
  --teacher-forcing-prob 0.5
```

## Run Artifacts

A typical run directory contains:
- `best.pt`: best model according to selection metric
- `last.pt`: final epoch checkpoint
- `config.json`: full training/evaluation configuration
- `normalization.json`: feature scaling metadata
- `history.json`: epoch-wise losses/metrics
- `eval_best.json`: post-training evaluation report

These artifacts provide enough information to reproduce model selection and compare experiment behavior across runs.

## Development and Quality

- Keep local data and run artifacts out of Git history.
- Validate code before sharing:

```bash
python -m compileall .
python -m pytest tests/test_topology.py
```
