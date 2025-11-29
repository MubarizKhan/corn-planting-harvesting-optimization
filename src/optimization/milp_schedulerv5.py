from pathlib import Path
from typing import Optional
import gurobipy as gp
from gurobipy import GRB
import os
import pandas as pd


# ==============================================================
# 1. Harvest Weather Factor
# ==============================================================
def compute_harvest_weather_factor(row: pd.Series) -> float:
    factor = 1.0

    prcp = row.get("prcp_week_in", 0.0)
    tavg = row.get("TAVG", None)
    tmax = row.get("TMAX", None)
    tmin = row.get("TMIN", None)
    awnd = row.get("AWND", None)

    if pd.notna(prcp):
        if prcp >= 1.5:
            factor -= 0.35
        elif prcp >= 0.75:
            factor -= 0.20

    if pd.notna(tavg) and tavg < 45:
        factor -= 0.1

    if pd.notna(tmax) and pd.notna(tmin):
        if (tmax - tmin) < 10:
            factor -= 0.05

    if pd.notna(awnd) and awnd < 5:
        factor -= 0.05

    return max(0.35, min(1.0, factor))


# ==============================================================
# 2. FIXED MILP MODEL (v5)
# ==============================================================
def build_and_solve_schedule_v5(
    fields_path: str,
    weekly_master_path: str,
    target_year: int = 2017,
    base_planter_capacity: float = 700.0,
    base_harvester_capacity: float = 450.0,
    labor_plant_per_acre: float = 0.2,
    labor_harvest_per_acre: float = 0.3,
    min_harvest_lag_weeks: int = 6,
    phys_maturity_lag_weeks: int = 12,
    late_buffer_weeks: int = 3,
    early_penalty_weight: float = 10.0,
    late_penalty_weight: float = 5.0,
    statewide_corn_acres: float = 12_000_000.0,
    time_limit: Optional[int] = 120,
):
    fields_df = pd.read_csv(fields_path)
    wm = pd.read_csv(weekly_master_path)

    wm_year = wm[wm["year"] == target_year].copy()
    wm_year["week"] = wm_year["week"].astype(int)
    wm_year = wm_year.sort_values("week")

    agg_spec = {
        "capacity_factor": "mean",
        "labor_hours": "mean",
        "is_plant_window": "max",
        "is_harvest_window": "max",
        "prcp_week_in": "mean",
        "TAVG": "mean",
        "TMAX": "mean",
        "TMIN": "mean",
        "AWND": "mean",
    }

    wm_year = wm_year.groupby("week", as_index=False).agg(agg_spec)
    wm_year["harvest_weather_factor"] = wm_year.apply(
        compute_harvest_weather_factor, axis=1
    )

    weeks = wm_year["week"].tolist()
    fields = fields_df["field_id"].tolist()
    area = dict(zip(fields_df["field_id"], fields_df["acres"]))

    # synthetic_acres = sum(area.values())
    # labor_scale = synthetic_acres / statewide_corn_acres

    cap_factor = dict(zip(wm_year["week"], wm_year["capacity_factor"]))
    harvest_weather_factor = dict(zip(wm_year["week"], wm_year["harvest_weather_factor"]))

    # ⬇️ Use labor_hours exactly as stored in weekly_master
    labor_hours = dict(zip(wm_year["week"], wm_year["labor_hours"]))

    is_plant_window = dict(zip(wm_year["week"], wm_year["is_plant_window"]))
    is_harvest_window = dict(zip(wm_year["week"], wm_year["is_harvest_window"]))

    # --------------------------------------------
    # ORIGINAL NASS WINDOW (for penalties)
    # --------------------------------------------
    harvest_weeks_nass = [w for w in weeks if is_harvest_window[w]]
    nass_start = min(harvest_weeks_nass)
    nass_end = max(harvest_weeks_nass)

    # --------------------------------------------
    # EXPANDED OPERATING WINDOW (for MILP feasibility)
    # --------------------------------------------
    # expanded_range = range(nass_start - 2, nass_end + 2)
    # harvest_weeks = [w for w in expanded_range if w in weeks]
    harvest_weeks = weeks.copy()


    plant_weeks = [w for w in weeks if is_plant_window[w]]

    plant_capacity = {w: base_planter_capacity * cap_factor[w] for w in weeks}
    harvest_capacity = {
        w: base_harvester_capacity * cap_factor[w] * harvest_weather_factor[w]
        for w in weeks
    }

    # --------------------------------------------
    # BUILD MODEL
    # --------------------------------------------
    m = gp.Model("CornScheduler_v5")

    Plant = m.addVars(fields, plant_weeks, vtype=GRB.BINARY)
    Harvest = m.addVars(fields, harvest_weeks, vtype=GRB.BINARY)

    plant_week_var = m.addVars(fields, vtype=GRB.CONTINUOUS)
    harvest_week_var = m.addVars(fields, vtype=GRB.CONTINUOUS)

    makespan = m.addVar(vtype=GRB.CONTINUOUS)
    early_pen = m.addVars(fields, vtype=GRB.CONTINUOUS)
    late_pen = m.addVars(fields, vtype=GRB.CONTINUOUS)

    # --------------------------------------------
    # Plant & Harvest Constraints
    # --------------------------------------------
    for f in fields:
        m.addConstr(gp.quicksum(Plant[f, w] for w in plant_weeks) == 1)
        m.addConstr(gp.quicksum(Harvest[f, w] for w in harvest_weeks) == 1)

        m.addConstr(plant_week_var[f] == gp.quicksum(w * Plant[f, w] for w in plant_weeks))
        m.addConstr(harvest_week_var[f] == gp.quicksum(w * Harvest[f, w] for w in harvest_weeks))

        m.addConstr(harvest_week_var[f] >= plant_week_var[f] + phys_maturity_lag_weeks)
        m.addConstr(harvest_week_var[f] >= plant_week_var[f] + min_harvest_lag_weeks)

    # --------------------------------------------
    # Weekly Resource Constraints
    # --------------------------------------------
    for w in weeks:
        if w in plant_weeks:
            m.addConstr(gp.quicksum(area[f] * Plant[f, w] for f in fields) <= plant_capacity[w])
            m.addConstr(gp.quicksum(area[f] * labor_plant_per_acre * Plant[f, w] for f in fields) <= labor_hours[w])

        if w in harvest_weeks:
            m.addConstr(gp.quicksum(area[f] * Harvest[f, w] for f in fields) <= harvest_capacity[w])
            m.addConstr(gp.quicksum(area[f] * labor_harvest_per_acre * Harvest[f, w] for f in fields) <= labor_hours[w])

    # --------------------------------------------
    # Penalties based on ORIGINAL NASS window
    # --------------------------------------------
    for f in fields:
        early_dev = m.addVar(lb=0)
        late_dev = m.addVar(lb=0)

        m.addConstr(early_dev >= nass_start - harvest_week_var[f])
        m.addConstr(late_dev >= harvest_week_var[f] - (nass_end + late_buffer_weeks))

        m.addConstr(early_pen[f] == early_dev)
        m.addConstr(late_pen[f] == late_dev)

        m.addConstr(harvest_week_var[f] <= makespan)

    # --------------------------------------------
    # Objective
    # --------------------------------------------
    m.setObjective(
        makespan +
        early_penalty_weight * gp.quicksum(early_pen[f] for f in fields) +
        late_penalty_weight * gp.quicksum(late_pen[f] for f in fields),
        GRB.MINIMIZE
    )

    if time_limit:
        m.setParam(GRB.Param.TimeLimit, time_limit)
    # m.setParam("DualReductions", 0)


    if time_limit:
        m.setParam(GRB.Param.TimeLimit, time_limit)

    m.setParam("Presolve", 0)
    m.setParam("DualReductions", 0)

    m.optimize()


    # print("STATUS CODE!!!!:", m.status)
    # print("❌ Model is infeasible. Generating IIS…")
    # m.computeIIS()
    # m.write("infeasible.ilp")
    # print("📄 IIS written to infeasible.ilp")
    # m.write("infeasible.ilp")
    # print("STATUS NAME:", gp.GRB.Status[m.status])
    # print("STATUS NAME:", gp.GRB.Status(m.status))

    # ----- Case 1: INF_OR_UNBD -----
    if m.status == GRB.INF_OR_UNBD:
        print("🔄 Re-solving without presolve reductions…")
        m.setParam("DualReductions", 0)
        m.setParam("Presolve", 0)
        m.optimize()
        print("STATUS AFTER SECOND RUN:", m.status)

    # ----- Case 2: INFEASIBLE -----
    if m.status == GRB.INFEASIBLE:
        print("❌ Model is infeasible. Generating IIS…")
        m.computeIIS()
        m.write("infeasible.ilp")
        print("📄 !!!!!!!!!!IIS written to infeasible.ilp")
        return pd.DataFrame({"error": ["infeasible"], "status": [m.status]})




    # --------------------------------------------
    # Extract solution
    # --------------------------------------------
    rows = []
    for f in fields:
        p_week = next((w for w in plant_weeks if Plant[f, w].X > 0.5), None)
        h_week = next((w for w in harvest_weeks if Harvest[f, w].X > 0.5), None)

        if h_week < nass_start:
            status = "EARLY"
        elif h_week > nass_end + late_buffer_weeks:
            status = "LATE"
        else:
            status = "OK"

        rows.append({
            "field_id": f,
            "plant_week": p_week,
            "harvest_week": h_week,
            "plant_week_cont": plant_week_var[f].X,
            "harvest_week_cont": harvest_week_var[f].X,
            "early_penalty": early_pen[f].X,
            "late_penalty": late_pen[f].X,
            "status": status,
            "makespan": makespan.X,
        })

    return pd.DataFrame(rows)
