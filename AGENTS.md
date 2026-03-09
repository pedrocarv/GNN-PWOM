# Repository Guidelines

## Project Structure & Module Organization
Core library code lives in `pwom_surrogate/`. Use `config.py` for dataclass-based experiment settings, `fieldline_graph_data.py` for graph/data transforms, and `fieldline_graph_model.py` for PyTorch model code. CLI entry points live in `scripts/`, mainly `train_fieldline_graph.py` and `evaluate_fieldline_graph.py`. Tests live in `tests/` and currently focus on topology, schema, and tensor-shape contracts. Keep research plans and design notes in the top-level `*.md` files, shell definitions in `shell_layouts/`, sample data in `dataset/`, and local checkpoints under `runs/`.

## Build, Test, and Development Commands
Create an environment and install runtime dependencies with `python -m venv .venv && source .venv/bin/activate` and `pip install -r requirements.txt`. Install `pytest` separately if it is not already available in your environment. Use `python scripts/train_fieldline_graph.py --help` to inspect training options, then run training from the repo root so local imports resolve correctly. Example: `python scripts/train_fieldline_graph.py --output-dir runs/active`. Evaluate a checkpoint with `python scripts/evaluate_fieldline_graph.py --checkpoint runs/active/best.pt --split valid`. Run tests with `python -m pytest tests/test_topology.py` or `python -m pytest`.

## Coding Style & Naming Conventions
Follow existing Python style: 4-space indentation, type hints on public functions, dataclasses for configuration, and short module or function docstrings when physical intent is not obvious. Use `snake_case` for functions, variables, and CLI flags, `PascalCase` for classes, and keep tensor/channel names explicit rather than abbreviated. Match the current pattern of small, composable helpers over large monolithic routines.

## Testing Guidelines
Add `pytest` tests in `tests/test_*.py`. Prefer fast synthetic fixtures built from small NumPy or Torch arrays rather than large binary inputs. When changing graph layout, feature schemas, or decoder heads, add assertions for output shape, mask behavior, and round-trip transforms. If a script interface changes, at minimum verify `--help` and one representative code path locally.

## Commit & Pull Request Guidelines
Use short imperative commit subjects such as `Add closure transport mask test`. Keep commits scoped to one change. If you open pull requests later, include intent and validation commands; otherwise, keep that context in commit history or project notes.

## Data & Artifact Hygiene
Do not commit large datasets, cached files under `dataset/.pwom_graph_cache/`, or transient outputs from `runs/`. Treat checkpoints, plots, and experiment JSON as local artifacts unless they are intentionally needed for reproducibility.
