# src/optimization/milp_scheduler.py

from pathlib import Path
from typing import Optional

import gurobipy as gp
from gurobipy import GRB
import pandas as pd


def build_and_solve_schedule(
    fields_path: str = "data/processed/illinois_corn_fields_clean.csv",
    weekly_master_path: str = "data/processed/master_weekly_table.csv",
    target_year: int = 2015,
    # base equipment capacities at capacity_factor = 1.0 (realistic values)
    base_planter_capacity: float = 1400.0,   # acres/week (e.g. 24-row planter)
    base_harvester_capacity: float = 950.0,  # acres/week (e.g. two 12-row combines)
    # labor per acre assumptions (you can tweak these later)
    labor_plant_per_acre: float = 0.15,      # hours / acre to plant
    labor_harvest_per_acre: float = 0.20,    # hours / acre to harvest
    min_harvest_lag_weeks: int = 6,          # minimum weeks between plant & harvest
    time_limit: Optional[int] = 60           # Gurobi time limit in seconds
) -> pd.DataFrame:
    """
    Build and solve a corn planting/harvest schedule MILP for a single year.

    Inputs:
        fields_path: path to cleaned fields CSV (field_id, acres, region, ...)
        weekly_master_path: path to master weekly table CSV
        target_year: which year from master table to schedule
        base_planter_capacity: max acres/week in perfect conditions
        base_harvester_capacity: same for harvest
        labor_plant_per_acre: hours of labor per planted acre
        labor_harvest_per_acre: hours of labor per harvested acre
        min_harvest_lag_weeks: min weeks between planting and harvest
        time_limit: optional solver time limit in seconds

    Returns:
        schedule_df: DataFrame with columns:
            field_id, plant_week, harvest_week,
            plant_week_continuous, harvest_week_continuous,
            status, objective_makespan
    """

    # -----------------------
    # 1. Load data
    # -----------------------
    fields_path = Path(fields_path)
    weekly_master_path = Path(weekly_master_path)

    if not fields_path.exists():
        raise FileNotFoundError(f"Fields file not found: {fields_path}")
    if not weekly_master_path.exists():
        raise FileNotFoundError(f"Weekly master file not found: {weekly_master_path}")

    fields_df = pd.read_csv(fields_path)
    weekly_master = pd.read_csv(weekly_master_path)

    # Filter to target year
    wm_year = weekly_master[weekly_master["year"] == target_year].copy()
    if wm_year.empty:
        raise ValueError(f"No rows in master_weekly_table for year={target_year}")

    # Ensure types and sort
    wm_year["week"] = wm_year["week"].astype(int)
    wm_year = wm_year.sort_values("week").reset_index(drop=True)

    # Make sure required columns exist
    required_cols = [
        "week", "capacity_factor", "labor_hours",
        "is_plant_window", "is_harvest_window"
    ]
    missing = [c for c in required_cols if c not in wm_year.columns]
    if missing:
        raise ValueError(f"Missing required columns in weekly master: {missing}")

    # Coerce window flags to bool
    wm_year["is_plant_window"] = wm_year["is_plant_window"].astype(bool)
    wm_year["is_harvest_window"] = wm_year["is_harvest_window"].astype(bool)

    # -----------------------
    # 1a. Restrict to relevant weeks
    # -----------------------
    plant_weeks_all = wm_year.loc[wm_year["is_plant_window"], "week"].unique()
    harvest_weeks_all = wm_year.loc[wm_year["is_harvest_window"], "week"].unique()

    if len(plant_weeks_all) == 0:
        raise ValueError("No planting weeks (is_plant_window == True) for this year.")
    if len(harvest_weeks_all) == 0:
        raise ValueError("No harvest weeks (is_harvest_window == True) for this year.")

    w_min = min(plant_weeks_all.min(), harvest_weeks_all.min())
    w_max = max(plant_weeks_all.max(), harvest_weeks_all.max())

    # Optional 1-week buffer on each side
    w_min_buffer = max(1, w_min - 1)
    w_max_buffer = min(52, w_max + 1)

    wm_year = wm_year[
        (wm_year["week"] >= w_min_buffer) &
        (wm_year["week"] <= w_max_buffer)
    ].copy()
    wm_year = wm_year.sort_values("week").reset_index(drop=True)

    # After trimming, recompute weeks and window flags
    weeks = wm_year["week"].unique().tolist()

    is_plant_window = {
        int(row["week"]): bool(row["is_plant_window"])
        for _, row in wm_year.iterrows()
    }
    is_harvest_window = {
        int(row["week"]): bool(row["is_harvest_window"])
        for _, row in wm_year.iterrows()
    }

    plant_weeks = [w for w in weeks if is_plant_window[w]]
    harvest_weeks = [w for w in weeks if is_harvest_window[w]]

    if not plant_weeks:
        raise ValueError("No planting weeks after trimming.")
    if not harvest_weeks:
        raise ValueError("No harvest weeks after trimming.")

    # -----------------------
    # 1b. Build sets & parameters
    # -----------------------
    fields = fields_df["field_id"].tolist()
    area = dict(zip(fields_df["field_id"], fields_df["acres"]))

    # Ensure numeric, clean NaNs
    wm_year["capacity_factor"] = pd.to_numeric(wm_year["capacity_factor"], errors="coerce")
    wm_year["labor_hours"] = pd.to_numeric(wm_year["labor_hours"], errors="coerce")

    if wm_year["labor_hours"].isna().any():
        # Fill NaNs conservatively with a very large labor capacity
        # so they don't restrict the model artificially
        wm_year["labor_hours"] = wm_year["labor_hours"].fillna(wm_year["labor_hours"].max())

    cap_factor = {
        int(row["week"]): float(row["capacity_factor"])
        for _, row in wm_year.iterrows()
    }
    labor_hours = {
        int(row["week"]): float(row["labor_hours"])
        for _, row in wm_year.iterrows()
    }

    # -----------------------
    # 2. Compute weekly capacities from equipment + weather
    # -----------------------
    plant_capacity = {
        w: base_planter_capacity * cap_factor[w] for w in weeks
    }
    harvest_capacity = {
        w: base_harvester_capacity * cap_factor[w] for w in weeks
    }

    # -----------------------
    # 3. Build Gurobi model
    # -----------------------
    m = gp.Model("corn_plant_harvest_schedule")

    # Decision variables: only on valid weeks
    Plant = m.addVars(fields, plant_weeks, vtype=GRB.BINARY, name="Plant")
    Harvest = m.addVars(fields, harvest_weeks, vtype=GRB.BINARY, name="Harvest")

    # Helper continuous vars: plant_week_f, harvest_week_f
    plant_week_var = m.addVars(fields, vtype=GRB.CONTINUOUS, name="PlantWeek")
    harvest_week_var = m.addVars(fields, vtype=GRB.CONTINUOUS, name="HarvestWeek")

    # Makespan: latest harvest week across all fields
    makespan = m.addVar(vtype=GRB.CONTINUOUS, name="Makespan")

    # -----------------------
    # 4. Constraints
    # -----------------------

    # 4.1 Each field planted exactly once; harvested exactly once
    for f in fields:
        m.addConstr(
            gp.quicksum(Plant[f, w] for w in plant_weeks) == 1,
            name=f"Plant_once_{f}"
        )
        m.addConstr(
            gp.quicksum(Harvest[f, w] for w in harvest_weeks) == 1,
            name=f"Harvest_once_{f}"
        )

    # 4.2 Define plant_week_f and harvest_week_f as the chosen week (weighted sum)
    for f in fields:
        m.addConstr(
            plant_week_var[f] == gp.quicksum(w * Plant[f, w] for w in plant_weeks),
            name=f"DefPlantWeek_{f}"
        )
        m.addConstr(
            harvest_week_var[f] == gp.quicksum(w * Harvest[f, w] for w in harvest_weeks),
            name=f"DefHarvestWeek_{f}"
        )

    # 4.3 Harvest must be at least min_harvest_lag_weeks after planting
    for f in fields:
        m.addConstr(
            harvest_week_var[f] >= plant_week_var[f] + min_harvest_lag_weeks,
            name=f"HarvestAfterPlant_{f}"
        )

    # 4.4 Weekly equipment & labor capacity constraints
    for w in weeks:
        # Planting expression (no vars if w not in plant_weeks)
        plant_expr = gp.quicksum(
            area[f] * Plant[f, w] for f in fields if w in plant_weeks
        )
        harvest_expr = gp.quicksum(
            area[f] * Harvest[f, w] for f in fields if w in harvest_weeks
        )

        # Planting capacity (acres)
        m.addConstr(
            plant_expr <= plant_capacity[w],
            name=f"PlantCap_week{w}"
        )

        # Harvest capacity (acres)
        m.addConstr(
            harvest_expr <= harvest_capacity[w],
            name=f"HarvestCap_week{w}"
        )

        # Labor needed this week
        plant_labor = gp.quicksum(
            area[f] * labor_plant_per_acre * Plant[f, w]
            for f in fields if w in plant_weeks
        )
        harvest_labor = gp.quicksum(
            area[f] * labor_harvest_per_acre * Harvest[f, w]
            for f in fields if w in harvest_weeks
        )
        labor_needed = plant_labor + harvest_labor

        m.addConstr(
            labor_needed <= labor_hours[w],
            name=f"LaborCap_week{w}"
        )

    # 4.5 Makespan: maximum of harvest_week_var[f]
    for f in fields:
        m.addConstr(
            harvest_week_var[f] <= makespan,
            name=f"Makespan_ge_harvest_{f}"
        )

    # -----------------------
    # 5. Objective
    # -----------------------
    # Minimize makespan (finish all harvests as early as possible)
    m.setObjective(makespan, GRB.MINIMIZE)

    # Solver params
    if time_limit is not None:
        m.setParam(GRB.Param.TimeLimit, time_limit)

    # Verbosity (set to 0 to mute)
    m.setParam(GRB.Param.OutputFlag, 1)

    # -----------------------
    # 6. Optimize
    # -----------------------
    m.optimize()

    status = m.status
    obj_val = None
    if status in (GRB.OPTIMAL, GRB.TIME_LIMIT):
        try:
            obj_val = makespan.X
        except AttributeError:
            obj_val = None
        print(f"Model status: {status}, makespan = {obj_val}")
    else:
        print(f"Model ended with status {status}")

    # -----------------------
    # 7. Extract solution
    # -----------------------
    sol_rows = []
    for f in fields:
        p_week = None
        h_week = None

        for w in plant_weeks:
            if Plant[f, w].X > 0.5:
                p_week = w
                break

        for w in harvest_weeks:
            if Harvest[f, w].X > 0.5:
                h_week = w
                break

        sol_rows.append({
            "field_id": f,
            "plant_week": p_week,
            "harvest_week": h_week,
            "plant_week_continuous": float(plant_week_var[f].X) if plant_week_var[f].X is not None else None,
            "harvest_week_continuous": float(harvest_week_var[f].X) if harvest_week_var[f].X is not None else None,
            "status": status,
            "objective_makespan": obj_val,
        })

    schedule_df = pd.DataFrame(sol_rows)
    return schedule_df
