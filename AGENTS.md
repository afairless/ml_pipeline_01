# Agent Instructions

This file tells coding agents how to work in this repository. Follow it
together with `README.md` and `docs/ARCHITECTURE.md`.

## Project summary

This is a reusable batch machine-learning pipeline template. It derives
outcomes and features from external source tables, divides cases into
date-based partitions, trains model strategies, predicts, and evaluates. The
demo is deterministic and runs without external data.

The extension point is `src/project/adapters.py`: `build_outcome_table` and
`build_feature_table` are where a new project supplies its own calculations.

## Toolchain

- Use `pixi run` for every Python command. Do not assume a bare `python` exists.
- Python is pinned to 3.12. Do not change the version without a reviewed
  lockfile update.
- Runtime, test, lint, and type-check dependencies are declared in
  `pixi.toml` and pinned in `pixi.lock`.

## Commands

```bash
pixi install                          # create or refresh the environment
pixi run pytest -q                    # full test suite
pixi run ruff check .                 # lint
pixi run ruff format --check .        # format check
pixi run pyright                      # type check
OMP_NUM_THREADS=1 pixi run python -m src.main   # full demo run
pixi run python scripts/run_stage.py --list     # stage inventory
pixi run python scripts/make_demo_data.py       # regenerate demo source table
```

Before every commit, also run the vocabulary gate:

```bash
pixi run python scripts/scrub_repo.py --worktree
pixi run python scripts/scrub_repo.py --staged
```

After changing history or when verifying a fresh clone, run
`pixi run python scripts/scrub_repo.py --history`.

## Architecture boundaries

- `src/core/` is project-agnostic. It must never import `src/project/` or a
  stage module. `tests/test_module_imports.py` enforces this.
- `src/project/` owns configuration, registries, boundary schemas, external
  input discovery, the adapters, artifact I/O, the training driver, and the
  stage table.
- `src/stages/` contains thin entry points: resolve paths through the layout,
  call `context.ensure`, validate at boundaries, delegate, and write declared
  artifacts.
- `src/main.py` calls the three explicit stages and delegates the rest to the
  registry. Keep stage entry points lazy: the stage table stores dotted module
  paths and imports a stage only when it executes.
- Do not add module-level path constants or `Path.cwd()` defaults. A stage
  receives a frozen `PipelineContext` and an explicit output root.

## Data and artifact rules

- Use the canonical vocabulary from `src/project/columns.py` and
  `src/project/outcomes.py`. `KEY_COLUMNS = (case_id, case_date)` is the
  identity and join order; never reverse it.
- Validate every frame crossing a stage boundary against the relevant schema
  in `src/project/schemas.py`.
- Resolve external inputs through `src/project/inputs.py`. Do not discover
  files by guessing names outside that registry.
- Resolve stage directories and artifacts through the declarative stage table
  in `src/project/pipeline.py`. Do not infer a run's identity from a filename
  or directory depth.
- Read and write per-run artifacts through manifests. An unknown manifest
  version or a missing declared artifact must fail, not fall back.
- A stage replaces only its own declared artifacts, using a temporary sibling
  and an atomic replace. Never delete another stage's output.

## Determinism rules

- The demo source table is committed and regenerated only from
  `CONFIG.demo_seed` and `CONFIG.demo_case_count`.
- Demo adapters create their own `numpy.random.default_rng(demo_seed)` and
  must not read or mutate the global `numpy` random state.
- The estimator is the only consumer of the global RNG. Do not seed it inside
  a stage or adapter; the test configuration seeds it once.
- Keep `OMP_NUM_THREADS=1` and the estimator's `nthread=1` behavior intact.
- Preserve draw order in feature policies. Reordering a draw changes every
  later seeded value and the golden baseline.
- Do not normalize or reformat golden baseline artifacts. If a baseline diff
  is intentional, review it and recapture explicitly; never recapture to hide
  a regression.

## Testing requirements

- Run the full suite, not only the tests for the file you changed.
- Add unit tests for new public functions and boundary failures. Add
  property-based tests for round-trip, idempotency, and ordering invariants.
- Add an integration or smoke test when a change crosses stages.
- Every schema constraint needs a failure test that asserts the boundary
  label and the failing rule.
- Binary labels follow the strict rule `value > threshold`; a value equal to
  the threshold is negative. Pin that boundary in tests.
- The vocabulary gate test is part of the default suite. Do not exclude it.

## Commit and documentation conventions

- Use conventional commits and keep one logical unit per commit.
- Run the quality gates and both scrub modes before committing.
- Do not commit `.scrub/` or `.port-staging/` content. They are local work
  product and are gitignored.
- Do not commit generated output, caches, or environment directories.
- Update `README.md`, `docs/ARCHITECTURE.md`, or `AGENTS.md` in the same commit
  that changes behavior, commands, paths, or contracts.
- Keep the README about current behavior. Do not add development history,
  commit counts, machine-specific paths, or transient status.
- Keep documentation in the repository's generic vocabulary. Do not describe
  a particular domain's data.

## Safety

- Never commit secrets, credentials, local absolute paths, or machine-specific
  configuration.
- Do not force-push or rewrite shared history.
- Do not weaken or bypass the vocabulary gate, the import-direction test, the
  boundary schemas, or the golden diff to make a change pass.
- If a dependency is added or upgraded, record a dependency audit and update
  `pixi.lock` in a reviewed change.
- If a required decision is missing, stop and ask rather than inventing a
  contract that other modules will depend on.

## Documentation map

- `README.md` — user-facing purpose, quick start, output tree, configuration,
  customization, and stage commands.
- `docs/ARCHITECTURE.md` — layering, module map, data flow, contracts, design
  decisions, testing, quality gates, and future work.
- `docs/research/` — design notes and plans. These are historical records, not
  the current user guide.
