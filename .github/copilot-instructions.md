# GitHub Copilot / AI Agent Instructions

Purpose: Give an AI coding agent the minimal, concrete context needed to be productive in this repository.

Project big-picture
- Data-driven MILP scheduler: the repo centers on building schedules for corn planting/harvest using a MILP solver. The key solver code is in `src/optimization/milp_scheduler.py`.
- Data flow: raw CSVs live in `data/raw/`, data cleaning happens in `data_cleaning.ipynb` and processed CSVs are written to `data/processed/` (examples: `master_weekly_table.csv`, `illinois_corn_fields_clean.csv`).

Critical files to inspect first
- `src/optimization/milp_scheduler.py`: primary entrypoint for scheduling. It expects `data/processed/master_weekly_table.csv` and `data/processed/illinois_corn_fields_clean.csv` by default and uses `gurobipy` + `pandas`.
- `data_cleaning.ipynb`: the canonical notebook used to build `data/processed` inputs.
- `data/processed/`: contains cleaned CSVs that are the inputs to optimization.

Runtime & developer workflows
- Virtual environment: a project `.venv/` exists. Activate before running Python code:

  ``source .venv/bin/activate``

- Dependencies: at minimum the code requires `pandas` and `gurobipy`. Gurobi requires a local installation and valid license—this is mandatory to run `milp_scheduler.py`.
- Quick run example (one-liner):

  ``source .venv/bin/activate && python -c "from src.optimization.milp_scheduler import build_and_solve_schedule; df=build_and_solve_schedule(target_year=2015, time_limit=10); print(df.head())"``

Key conventions and patterns (project-specific)
- File paths are repo-root-relative and often passed as default str args to functions (see `build_and_solve_schedule` defaults).
- `master_weekly_table.csv` is authoritative for scheduling weeks. The MILP expects these columns (case-sensitive):
  - `year`, `week`, `capacity_factor`, `labor_hours`, `is_plant_window`, `is_harvest_window`
- Window flags are boolean-like (converted with `.astype(bool)` in code). Weeks are normalized to `int` and sorted.
- Missing `labor_hours` are treated conservatively (filled with a high value) in `milp_scheduler.py` to avoid artificial restrictions—be cautious when changing this logic.

Model / solver notes (important for agents)
- The model uses Gurobi (`gurobipy`). Agents must not attempt to run the solver unless Gurobi is available and licensed. For offline testing, mock calls or small synthetic inputs can be used.
- `build_and_solve_schedule` sets `OutputFlag=1` and accepts `time_limit` argument (seconds). To reduce chatter set `m.setParam(GRB.Param.OutputFlag, 0)`.
- The objective minimizes makespan; capacities are computed as `base_capacity * capacity_factor` per week.

Editing guidance for agents
- Small, focused changes only. Prefer adding unit-test-like scripts or small runner scripts rather than changing core algorithm logic without tests.
- If altering data assumptions, update the relevant checks near the top of `build_and_solve_schedule` (existence checks, required columns, dtype coercion). Use defensive errors as currently implemented.
- When adding I/O or CLI, keep default file paths aligned with `data/processed/` and preserve existing function signatures where possible.

Examples of useful, concrete tasks an agent can do now
- Add a small CLI wrapper `scripts/run_scheduler.py` that parses args and calls `build_and_solve_schedule` (helpful for CI/manual runs).
- Create a `README` snippet documenting how to acquire/configure a Gurobi license for local runs.
- Add a small synthetic-data test that constructs minimal `master_weekly_table` + `illinois_corn_fields_clean` to run the model quickly in CI with a mocked solver or with a short `time_limit`.

What the agent should ask the user before risky changes
- Do you want me to modify default data-cleaning steps (notebook) or the solver defaults (e.g., labor fill strategy)?
- Is it acceptable to add a lightweight test that runs a tiny instance of the MILP in CI (requires either a Gurobi docker image/license or a mocked path)?

Where to look for more context
- `data_cleaning.ipynb`: shows how processed CSVs are produced and columns expected by the scheduler.
- `data/processed/master_weekly_table.csv`: real example rows to confirm expected column values and ranges.

If anything here is unclear or you want additional examples (CLI, tests, or CI snippets), tell me which sections to expand.
