
from pathlib import Path
from typing import Optional

import gurobipy as gp
from gurobipy import GRB
import pandas as pd


def _compute_harvest_weather_factor(row: pd.Series) -> float:
    factor = 1.0
    prcp = row.get("prcp_week_in", 0.0)
    tavg = row.get("TAVG", None)
    tmax = row.get("TMAX", None)
    tmin = row.get("TMIN", None)
    awnd = row.get("AWND", None)
    if pd.notna(prcp):
        if prcp >= 1.5:
            factor -= 0.4
        elif prcp >= 0.75:
            factor -= 0.2
    if pd.notna(tavg) and tavg < 45.0:
        factor -= 0.1
    if pd.notna(tmax) and pd.notna(tmin):
        ddi = tmax - tmin
        if ddi < 10.0:
            factor -= 0.1
    if pd.notna(awnd) and awnd < 5.0:
        factor -= 0.05
    return max(0.3, min(1.0, factor))


def build_and_solve_schedule_v4(
    fields_path: str,
    weekly_master_path: str,
    target_year: int = 2017,
    base_planter_capacity: float = 1400.0,
    base_harvester_capacity: float = 950.0,
    labor_plant_per_acre: float = 0.35,
    labor_harvest_per_acre: float = 0.30,
    labor_adjustment: float = 0.25,    # NEW
    min_harvest_lag_weeks: int = 6,
    phys_maturity_lag_weeks: int = 16,
    early_penalty_weight: float = 10.0,
    late_penalty_weight: float = 5.0,
    statewide_corn_acres: float = 12_000_000.0,
    num_planters: int = 1,             # NEW
    num_harvesters: int = 1,           # NEW
    time_limit: Optional[int] = 60,
):
    fields_df = pd.read_csv(fields_path)
    weekly_master = pd.read_csv(weekly_master_path)

    wm_year = weekly_master[weekly_master["year"] == target_year].copy()
    wm_year["week"] = wm_year["week"].astype(int)

    # --- Adjust labor scaling properly ---
    synthetic_acres = fields_df["acres"].sum()
    scale = (synthetic_acres / statewide_corn_acres) * labor_adjustment

    wm_year["labor_hours"] = wm_year["labor_hours"] * scale

    # --- Compute improved harvest weather factor ---
    def compute_weather(row):
        factor = 1.0
        prcp = row.get("prcp_week_in", 0)
        tavg = row.get("TAVG", None)
        tmax = row.get("TMAX", None)
        tmin = row.get("TMIN", None)

        if prcp >= 1.5: factor -= 0.25
        elif prcp >= 0.75: factor -= 0.15

        if pd.notna(tavg) and tavg < 45: factor -= 0.05
        if pd.notna(tmax) and pd.notna(tmin) and (tmax - tmin) < 10: factor -= 0.05

        return max(0.5, min(1.0, factor))

    wm_year["harvest_weather_factor"] = wm_year.apply(compute_weather, axis=1)

    weeks = wm_year["week"].tolist()
    fields = fields_df["field_id"].tolist()
    area = dict(zip(fields_df["field_id"], fields_df["acres"]))

    is_plant_window = dict(zip(wm_year["week"], wm_year["is_plant_window"]))
    is_harvest_window = dict(zip(wm_year["week"], wm_year["is_harvest_window"]))

    plant_weeks = [w for w in weeks if is_plant_window[w] == 1]
    harvest_weeks = [w for w in weeks if is_harvest_window[w] == 1]

    cap_factor = dict(zip(wm_year["week"], wm_year["capacity_factor"]))
    hweather = dict(zip(wm_year["week"], wm_year["harvest_weather_factor"]))
    labor_hours = dict(zip(wm_year["week"], wm_year["labor_hours"]))

    plant_capacity = {w: num_planters * base_planter_capacity * cap_factor[w] for w in weeks}
    harvest_capacity = {w: num_harvesters * base_harvester_capacity * cap_factor[w] * hweather[w] for w in weeks}

    # ------------------------ MODEL ------------------------
    m = gp.Model("corn_sched_v5")

    Plant = m.addVars(fields, plant_weeks, vtype=GRB.BINARY)
    Harvest = m.addVars(fields, harvest_weeks, vtype=GRB.BINARY)

    plant_week_var = m.addVars(fields, lb=0)
    harvest_week_var = m.addVars(fields, lb=0)
    makespan = m.addVar(lb=0)
    penalty = m.addVars(fields, lb=0)

    # --- Each field exactly once ---
    for f in fields:
        m.addConstr(sum(Plant[f, w] for w in plant_weeks) == 1)
        m.addConstr(sum(Harvest[f, w] for w in harvest_weeks) == 1)

        m.addConstr(plant_week_var[f] == sum(w * Plant[f, w] for w in plant_weeks))
        m.addConstr(harvest_week_var[f] == sum(w * Harvest[f, w] for w in harvest_weeks))

        m.addConstr(harvest_week_var[f] >= plant_week_var[f] + min_harvest_lag_weeks)

        # enforce makespan
        m.addConstr(makespan >= harvest_week_var[f])

        # Simple penalty: PWL on delta
        delta = m.addVar(lb=0)
        m.addConstr(delta == harvest_week_var[f] - plant_week_var[f])
        m.addGenConstrPWL(delta, penalty[f], [14, 16, 19, 21], [2, 0, 0, 1.5])

    # --- Weekly capacity + labor constraints ---
    for w in weeks:
        m.addConstr(sum(area[f] * Plant[f, w] for f in fields if w in plant_weeks) <= plant_capacity[w])
        m.addConstr(sum(area[f] * Harvest[f, w] for f in fields if w in harvest_weeks) <= harvest_capacity[w])

        plant_labor = sum(area[f] * labor_plant_per_acre * Plant[f, w] for f in fields if w in plant_weeks)
        harvest_labor = sum(area[f] * labor_harvest_per_acre * hweather[w] * Harvest[f, w] for f in fields if w in harvest_weeks)
        m.addConstr(plant_labor + harvest_labor <= labor_hours[w])

    # ------------------------ OBJECTIVE ------------------------
    m.setObjective(makespan + early_penalty_weight * sum(penalty[f] for f in fields), GRB.MINIMIZE)

    if time_limit:
        m.setParam("TimeLimit", time_limit)
    m.setParam("OutputFlag", 1)

    m.optimize()
    print("====== MILP DEBUG ======")
    print("Status!!!!!!!!!:", m.status)
    print("SolCount!!!!!!!!:", m.SolCount)

    if m.status == GRB.INFEASIBLE:
        print("Model infeasible. Computing IIS...")
        m.computeIIS()
        m.write("infeasible.ilp")
        print("IIS written to infeasible.ilp")


    # ------------------------ SOLUTION ------------------------
    sol = []
    for f in fields:
        p_week = next((w for w in plant_weeks if Plant[f, w].X > 0.5), None)
        h_week = next((w for w in harvest_weeks if Harvest[f, w].X > 0.5), None)
        sol.append({
            "field_id": f,
            "plant_week": p_week,
            "harvest_week": h_week,
            "penalty": penalty[f].X,
            "makespan": makespan.X,
            "status": m.status
        })

    return pd.DataFrame(sol)
