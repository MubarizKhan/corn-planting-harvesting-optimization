
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
    labor_plant_per_acre: float = 0.15,
    labor_harvest_per_acre: float = 0.20,
    min_harvest_lag_weeks: int = 6,
    phys_maturity_lag_weeks: int = 16,
    late_buffer_weeks: int = 3,
    early_penalty_weight: float = 10.0,
    late_penalty_weight: float = 5.0,
    statewide_corn_acres: float = 12_000_000.0,
    time_limit: Optional[int] = 60,
) -> pd.DataFrame:
    fields_path = Path(fields_path)
    weekly_master_path = Path(weekly_master_path)

    fields_df = pd.read_csv(fields_path)
    weekly_master = pd.read_csv(weekly_master_path)
    wm_year = weekly_master[weekly_master["year"] == target_year].copy()
    wm_year["week"] = wm_year["week"].astype(int)

    agg_spec = {
        "capacity_factor": "mean",
        "labor_hours": "sum",
        "is_plant_window": "max",
        "is_harvest_window": "max",
    }
    for col in ["prcp_week_in", "TAVG", "TMAX", "TMIN", "AWND"]:
        if col in wm_year.columns:
            agg_spec[col] = "mean"

    wm_year = wm_year.groupby("week", as_index=False).agg(agg_spec).sort_values("week")

    weeks = wm_year["week"].unique().tolist()
    fields = fields_df["field_id"].tolist()
    area = dict(zip(fields_df["field_id"], fields_df["acres"]))
    synthetic_acres = float(sum(area.values()))
    labor_scale = synthetic_acres / statewide_corn_acres

    cap_factor = dict(zip(wm_year["week"], wm_year["capacity_factor"]))
    wm_year["harvest_weather_factor"] = wm_year.apply(
        _compute_harvest_weather_factor, axis=1
    )
    harvest_weather_factor = dict(zip(wm_year["week"], wm_year["harvest_weather_factor"]))
    labor_hours = dict(zip(wm_year["week"], wm_year["labor_hours"] * labor_scale))
    is_plant_window = dict(zip(wm_year["week"], wm_year["is_plant_window"]))
    is_harvest_window = dict(zip(wm_year["week"], wm_year["is_harvest_window"]))

    plant_weeks = [w for w in weeks if is_plant_window[w]]
    harvest_weeks = [w for w in weeks if is_harvest_window[w]]
    plant_capacity = {w: base_planter_capacity * cap_factor[w] for w in weeks}
    harvest_capacity = {
        w: base_harvester_capacity * cap_factor[w] * harvest_weather_factor[w] for w in weeks
    }

    m = gp.Model("corn_schedule_v4")
    Plant = m.addVars(fields, plant_weeks, vtype=GRB.BINARY, name="Plant")
    Harvest = m.addVars(fields, harvest_weeks, vtype=GRB.BINARY, name="Harvest")
    plant_week_var = m.addVars(fields, vtype=GRB.CONTINUOUS, name="PlantWeek")
    harvest_week_var = m.addVars(fields, vtype=GRB.CONTINUOUS, name="HarvestWeek")
    makespan = m.addVar(vtype=GRB.CONTINUOUS, name="Makespan")
    penalty = m.addVars(fields, vtype=GRB.CONTINUOUS, name="MoisturePenalty")

    for f in fields:
        m.addConstr(gp.quicksum(Plant[f, w] for w in plant_weeks) == 1)
        m.addConstr(gp.quicksum(Harvest[f, w] for w in harvest_weeks) == 1)
        m.addConstr(plant_week_var[f] == gp.quicksum(w * Plant[f, w] for w in plant_weeks))
        m.addConstr(harvest_week_var[f] == gp.quicksum(w * Harvest[f, w] for w in harvest_weeks))
        m.addConstr(harvest_week_var[f] >= plant_week_var[f] + min_harvest_lag_weeks)

    for w in weeks:
        plant_expr = gp.quicksum(area[f] * Plant[f, w] for f in fields if w in plant_weeks)
        harvest_expr = gp.quicksum(area[f] * Harvest[f, w] for f in fields if w in harvest_weeks)
        plant_labor = gp.quicksum(area[f] * labor_plant_per_acre * Plant[f, w] for f in fields if w in plant_weeks)
        harvest_labor = gp.quicksum(area[f] * labor_harvest_per_acre * harvest_weather_factor[w] * Harvest[f, w] for f in fields if w in harvest_weeks)
        m.addConstr(plant_expr <= plant_capacity[w])
        m.addConstr(harvest_expr <= harvest_capacity[w])
        m.addConstr(plant_labor + harvest_labor <= labor_hours[w])

    for f in fields:
        m.addConstr(harvest_week_var[f] <= makespan)
        delta = m.addVar(lb=0.0, name=f"DeltaHarvest_{f}")
        m.addConstr(delta == harvest_week_var[f] - plant_week_var[f])
        m.addGenConstrPWL(
            delta,
            penalty[f],
            [14, 16, 19, 21],  # relative to planting
            [2.0, 0.0, 0.0, 1.5],
            name=f"PenaltyCurve_{f}"
        )

    m.setObjective(
        makespan + gp.quicksum(penalty[f] * early_penalty_weight for f in fields),
        GRB.MINIMIZE
    )

    if time_limit is not None:
        m.setParam(GRB.Param.TimeLimit, time_limit)
    m.setParam(GRB.Param.OutputFlag, 1)
    m.optimize()

    sol_rows = []
    for f in fields:
        p_week = next((w for w in plant_weeks if Plant[f, w].X > 0.5), None)
        h_week = next((w for w in harvest_weeks if Harvest[f, w].X > 0.5), None)
        sol_rows.append({
            "field_id": f,
            "plant_week": p_week,
            "harvest_week": h_week,
            "plant_week_continuous": float(plant_week_var[f].X),
            "harvest_week_continuous": float(harvest_week_var[f].X),
            "penalty": float(penalty[f].X),
            "status": m.status,
            "objective_makespan": makespan.X if m.status in (GRB.OPTIMAL, GRB.TIME_LIMIT) else None,
        })
    return pd.DataFrame(sol_rows)
