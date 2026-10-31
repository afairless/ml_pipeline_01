# Machine-Learning Pipeline Template

A reusable template for batch machine-learning pipelines. The pipeline moves
data through clearly separated stages: ingestion, outcome and feature
derivation, date-based partitioning, training, prediction, and evaluation.

This repository is under construction. Stage modules and documentation will be
added incrementally.

## Development

The project uses `pixi` for environment management.

```bash
pixi install
pixi run pytest -q
```

Replace the demo producers in `src/project/adapters.py` with your own data
transformations; the remaining stage plumbing stays unchanged.
