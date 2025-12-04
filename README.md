# Corn Planting & Harvest Scheduler

Mixed-integer optimization workflow to plan corn planting and harvest across Illinois fields. The current solver (`src/optimization/milp_schedulerv7.py`) uses Gurobi to balance machinery capacity, labor, agronomic lags, NASS windows, and weather-derived slowdowns, producing a week-by-week schedule.

## Repository layout
- `src/optimization/milp_schedulerv7.py` – main MILP scheduler (v7 weather and capacity tuning).
- `src/scripts/NOAA_Daily_dataGetter` – pulls NOAA CDO daily weather for Illinois and aggregates to weekly.
- `data/processed/` – cleaned inputs (fields, labor, NASS windows, NOAA aggregates) and sample optimizer outputs.
- `data/forecasting_opt_data/` – features prepared for forecasting experiments.
- `install_gurobi.sh` – convenience installer for Gurobi 10.0.3 (downloads, installs to `/opt`, updates shell paths).

## Setup
1) Python 3.10+ and a Gurobi license are required. Run `bash install_gurobi.sh` if you want the scripted install (needs sudo and downloads from `packages.gurobi.com`).
2) Create a virtual environment and install the small dependency set:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install gurobipy pandas requests
   ```
3) If you need fresh NOAA data, open `src/scripts/NOAA_Daily_dataGetter` and replace `NOAA_TOKEN` with your API token before running the script.

## Running the MILP scheduler
The solver expects two CSV inputs:
- Fields file (e.g., `data/processed/illinois_corn_fields_clean.csv`) with `field_id` and `acres`.
- Weekly master table (e.g., `data/processed/master_weekly_table_2017_2024.csv`) with weather (`prcp_week_in`, `TAVG`, `AWND`, etc.), capacity factors, labor hours, and planting/harvest window flags.

Example run from the repo root:
```bash
python3 - <<'PY'
from src.optimization.milp_schedulerv7 import build_and_solve_schedule_v7
df = build_and_solve_schedule_v7(
    fields_path="data/processed/illinois_corn_fields_clean.csv",
    weekly_master_path="data/processed/master_weekly_table_2017_2024.csv",
    target_year=2017,
    time_limit=180  # seconds
)
print(df.head())
df.to_csv("data/processed/schedule_output_latest.csv", index=False)
PY
```

The returned DataFrame (and saved CSV) includes `field_id`, `plant_week`, `harvest_week`, penalty components, and `status` flags (early, late, frost risk). If the model is infeasible, an IIS is written to `infeasiblev7.ilp` for debugging.

## Weather data ingestion
To refresh NOAA weather locally:
```bash
python3 src/scripts/NOAA_Daily_dataGetter
```
Outputs land in `data/raw/noaa_il_daily_raw.csv` and `data/processed/noaa_il_weekly_agg.csv`. The weekly aggregates feed the master weekly table used by the optimizer.

## Notes
- Gurobi needs a valid license (`grbgetkey ...`) and access to the solver binaries on your PATH/LD_LIBRARY_PATH.
- The solver parameters (capacity, labor per acre, penalties, frost deadline, and windows) are configurable via function arguments in `milp_schedulerv7.py`.
- Sample outputs from earlier runs are stored in `data/processed/schedule_output*.csv` for reference.
