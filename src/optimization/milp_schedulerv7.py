from pathlib import Path
from typing import Optional
import gurobipy as gp
from gurobipy import GRB
import os
import pandas as pd


def compute_harvest_weather_factor_v7(row: pd.Series) -> float:
    """V8: Stronger weather shocks, more realistic harvest slowdowns."""
    prcp = row.get("prcp_week_in", 0.0)
    awnd = row.get("AWND", 5.0)
    tavg = row.get("TAVG", 50.0)

    factor = 1.0

    # 🌧 Rain — now MUCH stronger
    if prcp > 2.0:
        factor -= 0.70
    elif prcp > 1.0:
        factor -= 0.45
    elif prcp > 0.5:
        factor -= 0.20

    # 🌬️ Wind
    if awnd < 4:
        factor -= 0.10
    elif awnd > 12:
        factor -= 0.10

    # 🥶 Cold
    if tavg < 45:
        factor -= 0.20
    if tavg < 40:
        factor -= 0.30

    # Never below 10%
    return max(0.10, min(1.0, factor))

def build_and_solve_schedule_v7(
    fields_path: str,
    weekly_master_path: str,
    target_year: int = 2017,

    # 🧩 v8 machinery speeds (realistic)
    base_planter_capacity: float = 750.0,
    base_harvester_capacity: float = 400.0,

    labor_plant_per_acre: float = 0.25,
    labor_harvest_per_acre: float = 0.35,

    phys_maturity_lag_weeks: int = 13,
    min_harvest_lag_weeks: int = 6,

    # 🧩 v8 tighter NASS harvest window (±1 week)
    nass_window_extension: int = 3,

    # Penalty weights
    early_penalty_weight: float = 1.0,
    late_penalty_weight: float = 4.0,
    frost_penalty_weight: float = 12.0,

    frost_deadline: int = 44,   # v8 realistic frost date

    time_limit: Optional[int] = 150
):
    import gurobipy as gp
    from gurobipy import GRB

    # Load data
    fields_df = pd.read_csv(fields_path)
    wm = pd.read_csv(weekly_master_path)

    wm_year = wm[wm["year"] == target_year].copy().sort_values("week")
    wm_year["week"] = wm_year["week"].astype(int)

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

    # 🚀 v8 stronger weather factor
    wm_year["harvest_weather_factor"] = wm_year.apply(
        compute_harvest_weather_factor_v7, axis=1
    )

    weeks = wm_year["week"].tolist()
    fields = fields_df["field_id"].tolist()
    area = dict(zip(fields_df["field_id"], fields_df["acres"]))

    cap_factor = dict(zip(wm_year["week"], wm_year["capacity_factor"]))
    weather_factor = dict(zip(wm_year["week"], wm_year["harvest_weather_factor"]))
    labor_hours = dict(zip(wm_year["week"], wm_year["labor_hours"]))
    is_plant_window = dict(zip(wm_year["week"], wm_year["is_plant_window"]))
    is_harvest_window = dict(zip(wm_year["week"], wm_year["is_harvest_window"]))

    nass_weeks = [w for w in weeks if is_harvest_window[w] == 1]
    nass_start = min(nass_weeks)
    nass_end = max(nass_weeks)

    # 🔒 v8 restricted harvest window
    harvest_weeks = [
        w for w in weeks
        if (nass_start - nass_window_extension <= w <= nass_end + nass_window_extension)
    ]
        # v8 strict realistic harvest window
    # HARVEST_START = 38     # Early October
    # HARVEST_END   = 44     # Early November

    # harvest_weeks = [w for w in weeks if HARVEST_START <= w <= HARVEST_END]


    plant_weeks = [w for w in weeks if is_plant_window[w] == 1]

    harvest_capacity = {
        w: base_harvester_capacity * cap_factor[w] * weather_factor[w]
        for w in harvest_weeks
    }
    plant_capacity = {
        w: base_planter_capacity * cap_factor[w]
        for w in plant_weeks
    }

    # Build model
    m = gp.Model("CornScheduler_v8")

    Plant = m.addVars(fields, plant_weeks, vtype=GRB.BINARY)
    Harvest = m.addVars(fields, harvest_weeks, vtype=GRB.BINARY)

    plant_week_var = m.addVars(fields, vtype=GRB.CONTINUOUS)
    harvest_week_var = m.addVars(fields, vtype=GRB.CONTINUOUS)

    makespan = m.addVar(vtype=GRB.CONTINUOUS)

    early_pen = m.addVars(fields, vtype=GRB.CONTINUOUS)
    late_pen = m.addVars(fields, vtype=GRB.CONTINUOUS)
    frost_pen = m.addVars(fields, vtype=GRB.CONTINUOUS)

    # Constraints
    for f in fields:
        m.addConstr(gp.quicksum(Plant[f, w] for w in plant_weeks) == 1)
        m.addConstr(gp.quicksum(Harvest[f, w] for w in harvest_weeks) == 1)

        m.addConstr(plant_week_var[f] ==
                    gp.quicksum(w * Plant[f, w] for w in plant_weeks))
        m.addConstr(harvest_week_var[f] ==
                    gp.quicksum(w * Harvest[f, w] for w in harvest_weeks))

        m.addConstr(harvest_week_var[f] >= plant_week_var[f] + phys_maturity_lag_weeks)
        m.addConstr(harvest_week_var[f] >= plant_week_var[f] + min_harvest_lag_weeks)

        # m.addConstr(makespan >= harvest_week_var[f])
        m.addConstr(harvest_week_var[f] <= makespan )

        # penalties
        early_dev = m.addVar(lb=0)
        late_dev = m.addVar(lb=0)
        frost_dev = m.addVar(lb=0)

        m.addConstr(early_dev >= nass_start - harvest_week_var[f])
        m.addConstr(late_dev >= harvest_week_var[f] - nass_end)
        m.addConstr(frost_dev >= harvest_week_var[f] - frost_deadline)

        m.addConstr(early_pen[f] == early_dev)
        m.addConstr(late_pen[f] == late_dev)
        m.addConstr(frost_pen[f] == frost_dev)

    # Weekly caps
    for w in plant_weeks:
        m.addConstr(sum(area[f] * Plant[f, w] for f in fields) <= plant_capacity[w])
        m.addConstr(sum(area[f] * labor_plant_per_acre * Plant[f, w]
                    for f in fields) <= labor_hours[w])

    for w in harvest_weeks:
        m.addConstr(sum(area[f] * Harvest[f, w] for f in fields) <= harvest_capacity[w])
        m.addConstr(sum(area[f] * labor_harvest_per_acre * Harvest[f, w]
                    for f in fields) <= labor_hours[w] * weather_factor[w])

    # Objective
    m.setObjective(
        makespan
        + 1.0 * gp.quicksum(early_pen[f] for f in fields)
        + 4.0 * gp.quicksum(late_pen[f] for f in fields)
        + 12.0 * gp.quicksum(frost_pen[f] for f in fields),
        GRB.MINIMIZE
    )

    m.setParam("TimeLimit", time_limit)
    m.setParam("Presolve", 1)
    m.setParam("Threads", 4)
    m.setParam("Heuristics", 0.1)
    m.setParam("MIPFocus", 1)

    m.optimize()
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
        m.write("infeasiblev7.ilp")
        print("📄 !!!!!!!!!!IIS written to infeasible.ilp")
        return pd.DataFrame({"error": ["infeasible"], "status": [m.status]})


    if m.status == GRB.INFEASIBLE:
        return pd.DataFrame({"error": ["infeasible"], "status": [m.status]})

    # Output
    rows = []
    for f in fields:
        p_week = next((w for w in plant_weeks if Plant[f, w].X > 0.5), None)
        h_week = next((w for w in harvest_weeks if Harvest[f, w].X > 0.5), None)

        status = "OK"
        if h_week < nass_start:
            status = "EARLY"
        elif h_week > nass_end:
            status = "LATE"
        if h_week > frost_deadline:
            status = "FROST_RISK"

        rows.append({
            "field_id": f,
            "plant_week": p_week,
            "harvest_week": h_week,
            "early_penalty": early_pen[f].X,
            "late_penalty": late_pen[f].X,
            "frost_penalty": frost_pen[f].X,
            "total_penalty": early_pen[f].X + late_pen[f].X + frost_pen[f].X,
            "status": status,
            "makespan": makespan.X,
        })

    return pd.DataFrame(rows)
