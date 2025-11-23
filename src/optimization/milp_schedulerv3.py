# src/optimization/milp_schedulerv3.py

from pathlib import Path
from typing import Optional

import gurobipy as gp
from gurobipy import GRB
import pandas as pd


# -------------------------------------------------------------------
# Helper: weather-based harvest slowdown factor
# -------------------------------------------------------------------
def _compute_harvest_weather_factor(row: pd.Series) -> float:
    """
    Heuristic factor (0.3–1.0) that scales harvest capacity based on weather.

    Uses:
      - weekly rainfall (prcp_week_in)
      - TAVG, TMAX, TMIN
      - AWND (avg wind)

    Heavy rain / cool temps / low drying / low wind => slower harvest.
    """
    factor = 1.0

    prcp = row.get("prcp_week_in", 0.0)
    tavg = row.get("TAVG", None)
    tmax = row.get("TMAX", None)
    tmin = row.get("TMIN", None)
    awnd = row.get("AWND", None)

    # Rain impact (inches / week)
    if pd.notna(prcp):
        if prcp >= 1.5:
            factor -= 0.4
        elif prcp >= 0.75:
            factor -= 0.2

    # Cool average temperature (poor drying)
    if pd.notna(tavg) and tavg < 45.0:
        factor -= 0.1

    # Low diurnal temp range -> low drying degree
    if pd.notna(tmax) and pd.notna(tmin):
        ddi = tmax - tmin
        if ddi < 10.0:
            factor -= 0.1

    # Very low wind -> less air drying
    if pd.notna(awnd) and awnd < 5.0:
        factor -= 0.05

    # Clamp between 0.3 and 1.0 so we never completely shut down
    factor = max(0.3, min(1.0, factor))
    return float(factor)


# -------------------------------------------------------------------
# Main MILP: V3
# -------------------------------------------------------------------
def build_and_solve_schedule_v3(
    fields_path: str = "data/processed/illinois_corn_fields_clean.csv",
    weekly_master_path: str = "data/processed/master_weekly_table.csv",
    target_year: int = 2017,
    # base equipment capacities at capacity_factor = 1.0 (acres/week)
    base_planter_capacity: float = 1400.0,
    base_harvester_capacity: float = 950.0,
    # labor per acre assumptions
    labor_plant_per_acre: float = 0.15,
    labor_harvest_per_acre: float = 0.20,
    # growth / moisture modelling
    min_harvest_lag_weeks: int = 6,
    phys_maturity_lag_weeks: int = 16,
    late_buffer_weeks: int = 3,
    early_penalty_weight: float = 10.0,
    late_penalty_weight: float = 5.0,
    time_limit: Optional[int] = 60,
) -> pd.DataFrame:
    """
    MILP for corn planting / harvest scheduling with:

      1. Weather-dependent harvest capacity (NOAA-based slowdown).
      2. Grain moisture penalties (harvesting too early or too late).
      3. Labor capacity constraints.

    Objective:
        minimize makespan
        + early_penalty_weight * sum(early_penalty_f)
        + late_penalty_weight  * sum(late_penalty_f)

    Returns:
        schedule_df with columns:
        field_id, plant_week, harvest_week,
        plant_week_continuous, harvest_week_continuous,
        early_penalty, late_penalty, status, objective_makespan
    """

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    fields_path = Path(fields_path)
    weekly_master_path = Path(weekly_master_path)

    if not fields_path.exists():
        raise FileNotFoundError(f"Fields file not found: {fields_path}")
    if not weekly_master_path.exists():
        raise FileNotFoundError(f"Weekly master file not found: {weekly_master_path}")

    fields_df = pd.read_csv(fields_path)
    weekly_master = pd.read_csv(weekly_master_path)

    wm_year = weekly_master[weekly_master["year"] == target_year].copy()
    if wm_year.empty:
        raise ValueError(f"No rows in master_weekly_table for year={target_year}")

    wm_year["week"] = wm_year["week"].astype(int)
    wm_year = wm_year.sort_values("week").reset_index(drop=True)

    required_cols = [
        "week",
        "capacity_factor",
        "labor_hours",
        "is_plant_window",
        "is_harvest_window",
    ]
    missing = [c for c in required_cols if c not in wm_year.columns]
    if missing:
        raise ValueError(f"Missing required columns in weekly master: {missing}")

    wm_year["is_plant_window"] = wm_year["is_plant_window"].astype(bool)
    wm_year["is_harvest_window"] = wm_year["is_harvest_window"].astype(bool)

    # ------------------------------------------------------------------
    # 1a. Trim horizon to relevant weeks
    # ------------------------------------------------------------------
    plant_weeks_all = wm_year.loc[wm_year["is_plant_window"], "week"].unique()
    harvest_weeks_all = wm_year.loc[wm_year["is_harvest_window"], "week"].unique()

    if len(plant_weeks_all) == 0:
        raise ValueError("No planting weeks (is_plant_window == True) for this year.")
    if len(harvest_weeks_all) == 0:
        raise ValueError("No harvest weeks (is_harvest_window == True) for this year.")

    w_min = min(plant_weeks_all.min(), harvest_weeks_all.min())
    w_max = max(plant_weeks_all.max(), harvest_weeks_all.max())

    # small buffer on each side
    w_min_buffer = max(1, w_min - 1)
    w_max_buffer = min(52, w_max + 1)

    wm_year = wm_year[
        (wm_year["week"] >= w_min_buffer) & (wm_year["week"] <= w_max_buffer)
    ].copy()
    wm_year = wm_year.sort_values("week").reset_index(drop=True)

    weeks = wm_year["week"].unique().tolist()

    is_plant_window = {
        int(row["week"]): bool(row["is_plant_window"]) for _, row in wm_year.iterrows()
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

    # ------------------------------------------------------------------
    # 1b. Sets & parameters
    # ------------------------------------------------------------------
    fields = fields_df["field_id"].tolist()
    area = dict(zip(fields_df["field_id"], fields_df["acres"]))

    wm_year["capacity_factor"] = pd.to_numeric(
        wm_year["capacity_factor"], errors="coerce"
    )
    wm_year["labor_hours"] = pd.to_numeric(wm_year["labor_hours"], errors="coerce")

    if wm_year["labor_hours"].isna().any():
        # conservatively use max labor hours so NaNs are never binding
        wm_year["labor_hours"] = wm_year["labor_hours"].fillna(
            wm_year["labor_hours"].max()
        )

    cap_factor = {
        int(row["week"]): float(row["capacity_factor"]) for _, row in wm_year.iterrows()
    }
    labor_hours = {
        int(row["week"]): float(row["labor_hours"]) for _, row in wm_year.iterrows()
    }

    # weather-dependent harvest slowdown factor
    wm_year["harvest_weather_factor"] = wm_year.apply(
        _compute_harvest_weather_factor, axis=1
    )
    harvest_weather_factor = {
        int(row["week"]): float(row["harvest_weather_factor"])
        for _, row in wm_year.iterrows()
    }

    # ------------------------------------------------------------------
    # 2. Weekly capacities (equipment × weather)
    # ------------------------------------------------------------------
    plant_capacity = {w: base_planter_capacity * cap_factor[w] for w in weeks}
    harvest_capacity = {
        w: base_harvester_capacity * cap_factor[w] * harvest_weather_factor[w]
        for w in weeks
    }

    # ------------------------------------------------------------------
    # 3. Build Gurobi model
    # ------------------------------------------------------------------
    m = gp.Model("corn_plant_harvest_schedule_v3")

    # Binary decisions only on valid weeks
    Plant = m.addVars(fields, plant_weeks, vtype=GRB.BINARY, name="Plant")
    Harvest = m.addVars(fields, harvest_weeks, vtype=GRB.BINARY, name="Harvest")

    # Helper continuous vars
    plant_week_var = m.addVars(fields, vtype=GRB.CONTINUOUS, name="PlantWeek")
    harvest_week_var = m.addVars(fields, vtype=GRB.CONTINUOUS, name="HarvestWeek")

    # Makespan
    makespan = m.addVar(vtype=GRB.CONTINUOUS, name="Makespan")

    # Grain moisture penalties (one per field)
    early_penalty = m.addVars(fields, vtype=GRB.CONTINUOUS, name="EarlyPenalty")
    late_penalty = m.addVars(fields, vtype=GRB.CONTINUOUS, name="LatePenalty")

    # ------------------------------------------------------------------
    # 4. Constraints
    # ------------------------------------------------------------------

    # 4.1 Each field planted & harvested exactly once
    for f in fields:
        m.addConstr(
            gp.quicksum(Plant[f, w] for w in plant_weeks) == 1,
            name=f"Plant_once_{f}",
        )
        m.addConstr(
            gp.quicksum(Harvest[f, w] for w in harvest_weeks) == 1,
            name=f"Harvest_once_{f}",
        )

    # 4.2 Define plant_week_var and harvest_week_var (weighted sums)
    for f in fields:
        m.addConstr(
            plant_week_var[f] == gp.quicksum(w * Plant[f, w] for w in plant_weeks),
            name=f"DefPlantWeek_{f}",
        )
        m.addConstr(
            harvest_week_var[f] == gp.quicksum(w * Harvest[f, w] for w in harvest_weeks),
            name=f"DefHarvestWeek_{f}",
        )

    # 4.3 Growth lag: harvest at least min_harvest_lag_weeks after planting
    for f in fields:
        m.addConstr(
            harvest_week_var[f] >= plant_week_var[f] + min_harvest_lag_weeks,
            name=f"HarvestAfterPlant_{f}",
        )

    # 4.4 Weekly equipment & labor capacity
    for w in weeks:
        plant_expr = gp.quicksum(
            area[f] * Plant[f, w] for f in fields if w in plant_weeks
        )
        harvest_expr = gp.quicksum(
            area[f] * Harvest[f, w] for f in fields if w in harvest_weeks
        )

        # machinery capacity
        m.addConstr(
            plant_expr <= plant_capacity[w],
            name=f"PlantCap_week{w}",
        )
        m.addConstr(
            harvest_expr <= harvest_capacity[w],
            name=f"HarvestCap_week{w}",
        )

        # labor capacity
        plant_labor = gp.quicksum(
            area[f] * labor_plant_per_acre * Plant[f, w]
            for f in fields
            if w in plant_weeks
        )
        harvest_labor = gp.quicksum(
            area[f] * labor_harvest_per_acre * Harvest[f, w]
            for f in fields
            if w in harvest_weeks
        )
        labor_needed = plant_labor + harvest_labor

        m.addConstr(
            labor_needed <= labor_hours[w],
            name=f"LaborCap_week{w}",
        )

    # 4.5 Makespan ≥ harvest week for each field
    for f in fields:
        m.addConstr(
            harvest_week_var[f] <= makespan,
            name=f"Makespan_ge_harvest_{f}",
        )

    # 4.6 Grain moisture penalties (early & late)
    # maturity_week ≈ plant_week + phys_maturity_lag_weeks
    for f in fields:
        maturity_expr = plant_week_var[f] + phys_maturity_lag_weeks

        # Early penalty: harvest before maturity
        # early_penalty[f] >= maturity_week - harvest_week
        m.addConstr(
            early_penalty[f] >= maturity_expr - harvest_week_var[f],
            name=f"EarlyPenaltyDef_{f}",
        )
        m.addConstr(
            early_penalty[f] >= 0.0,
            name=f"EarlyPenaltyNonNeg_{f}",
        )

        # Late penalty: harvest after maturity + buffer
        # late_penalty[f] >= harvest_week - (maturity_week + late_buffer)
        m.addConstr(
            late_penalty[f]
            >= harvest_week_var[f] - (maturity_expr + late_buffer_weeks),
            name=f"LatePenaltyDef_{f}",
        )
        m.addConstr(
            late_penalty[f] >= 0.0,
            name=f"LatePenaltyNonNeg_{f}",
        )

    # ------------------------------------------------------------------
    # 5. Objective
    # ------------------------------------------------------------------
    m.setObjective(
        makespan
        + early_penalty_weight * gp.quicksum(early_penalty[f] for f in fields)
        + late_penalty_weight * gp.quicksum(late_penalty[f] for f in fields),
        GRB.MINIMIZE,
    )

    # Solver params
    if time_limit is not None:
        m.setParam(GRB.Param.TimeLimit, time_limit)
    m.setParam(GRB.Param.OutputFlag, 1)

    # ------------------------------------------------------------------
    # 6. Optimize
    # ------------------------------------------------------------------
    m.optimize()

    # ------------------------------------------------------------------
    # 6a. DEBUG: Capacity checks on the solution
    # ------------------------------------------------------------------
    print("\n--- Capacity Check (solution) ---")
    for w in weeks:
        # Plant capacity check
        plant_load = sum(
            area[f] * Plant[f, w].X for f in fields if w in plant_weeks
        )
        if plant_load > plant_capacity[w] + 1e-5:
            print(
                f"[PLANT] OVER CAPACITY w{w}: "
                f"load={plant_load:.2f}, cap={plant_capacity[w]:.2f}"
            )

        # Harvest capacity check
        harvest_load = sum(
            area[f] * Harvest[f, w].X for f in fields if w in harvest_weeks
        )
        if harvest_load > harvest_capacity[w] + 1e-5:
            print(
                f"[HARVEST] OVER CAPACITY w{w}: "
                f"load={harvest_load:.2f}, cap={harvest_capacity[w]:.2f}"
            )

        # Labor capacity check
        labor_need = (
            sum(
                area[f] * labor_plant_per_acre * Plant[f, w].X
                for f in fields
                if w in plant_weeks
            )
            + sum(
                area[f] * labor_harvest_per_acre * Harvest[f, w].X
                for f in fields
                if w in harvest_weeks
            )
        )
        if labor_need > labor_hours[w] + 1e-5:
            print(
                f"[LABOR] OVER CAPACITY w{w}: "
                f"load={labor_need:.2f}, cap={labor_hours[w]:.2f}"
            )

    # ------------------------------------------------------------------
    # 7. Extract solution
    # ------------------------------------------------------------------
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

        sol_rows.append(
            {
                "field_id": f,
                "plant_week": p_week,
                "harvest_week": h_week,
                "plant_week_continuous": float(plant_week_var[f].X),
                "harvest_week_continuous": float(harvest_week_var[f].X),
                "early_penalty": float(early_penalty[f].X),
                "late_penalty": float(late_penalty[f].X),
                "status": status,
                "objective_makespan": obj_val,
            }
        )

    schedule_df = pd.DataFrame(sol_rows)
    return schedule_df
