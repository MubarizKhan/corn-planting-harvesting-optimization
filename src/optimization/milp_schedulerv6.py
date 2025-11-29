# ============================================================
# MILP v6.1 – REALISTIC CAPACITY + FEASIBILITY GUARANTEED
# ============================================================

from pathlib import Path
from typing import Optional

import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import numpy as np



# ------------------------------------------------------------
# 1. Labor scaling (same as v6)
# ------------------------------------------------------------
def compute_labor_scaling(fields_df: pd.DataFrame,
                          statewide_corn_acres: float,
                          labor_adjustment: float) -> float:

    synthetic_acres = fields_df["acres"].sum()
    base = synthetic_acres / statewide_corn_acres
    return base * labor_adjustment



# ------------------------------------------------------------
# 2. Compute MINIMUM weekly capacity needed for feasibility
# ------------------------------------------------------------
def compute_min_weekly_capacity(acres_list, plant_weeks):
    """
    Compute the minimum weekly acreage required to schedule all fields
    given the number of available planting weeks.
    """

    total_acres = sum(acres_list)
    n_weeks = len(plant_weeks)

    if n_weeks == 0:
        raise RuntimeError("No planting weeks detected — model impossible.")

    # Minimum necessary capacity
    return total_acres / n_weeks



# ============================================================
#                 MILP v6.1 MAIN MODEL
# ============================================================
def build_and_solve_schedule_v6_1(
    fields_path: str,
    weekly_master_path: str,
    target_year: int = 2017,

    # Machine rates PER DAY (new!)
    planter_acres_per_day: float = 110,   # scaled-down realistic (400–600 in real world)
    harvester_acres_per_day: float = 80,  # scaled-down realistic

    # Days per workable ag week
    days_per_week: int = 6,

    # Labor
    labor_plant_per_acre: float = 0.02,
    labor_harvest_per_acre: float = 0.020,
    labor_adjustment: float = 0.030,

    min_harvest_lag_weeks: int = 6,
    early_penalty_weight: float = 10.0,

    statewide_corn_acres: float = 12_000_000.0,
    num_planters: int = 1,
    num_harvesters: int = 1,

    time_limit: int = 120,
):

    # --------------------------------------------------------
    # LOAD WEEKLY MASTER
    # --------------------------------------------------------
    wm = pd.read_csv(weekly_master_path)
    wm = wm[wm["year"] == target_year].copy()
    wm["week"] = wm["week"].astype(int)

    required = [
        "capacity_factor", "labor_hours",
        "is_plant_window", "is_harvest_window",
        "plant_start_week", "plant_end_week",
        "harvest_start_week", "harvest_end_week"
    ]
    for col in required:
        if col not in wm.columns:
            raise KeyError(f"Missing column '{col}' in weekly master.")


    # --------------------------------------------------------
    # LOAD FIELDS
    # --------------------------------------------------------
    fields_df = pd.read_csv(fields_path)
    fields = fields_df["field_id"].tolist()
    acres = dict(zip(fields_df["field_id"], fields_df["acres"]))

    plant_weeks  = wm.loc[wm["is_plant_window"] == True, "week"].tolist()
    harvest_weeks = wm.loc[wm["is_harvest_window"] == True, "week"].tolist()

    # --------------------------------------------------------
    # COMPUTE MANDATORY MINIMUM CAPACITY FOR FEASIBILITY
    # --------------------------------------------------------
    min_required_capacity = compute_min_weekly_capacity(
        list(acres.values()), plant_weeks
    )

    # --------------------------------------------------------
    # MACHINE CAPACITY PER WEEK (realistic)
    # --------------------------------------------------------
    raw_weekly_plant_capacity = (
        planter_acres_per_day * days_per_week * num_planters
    )
    raw_weekly_harvest_capacity = (
        harvester_acres_per_day * days_per_week * num_harvesters
    )

    # apply weather scaling
    cap_factor = dict(zip(wm["week"], wm["capacity_factor"]))

    plant_capacity = {}
    harvest_capacity = {}

    for w in wm["week"].tolist():

        pc = raw_weekly_plant_capacity * cap_factor[w]
        hc = raw_weekly_harvest_capacity * cap_factor[w]

        # FINAL v6.1 RULE:
        # weekly capacity must be >= minimum needed/week
        # and >= largest field
        plant_capacity[w] = max(pc, min_required_capacity, max(acres.values()))
        harvest_capacity[w] = max(hc, max(acres.values()))


    # --------------------------------------------------------
    # LABOR SCALING
    # --------------------------------------------------------
    labor_scale = compute_labor_scaling(
        fields_df, statewide_corn_acres, labor_adjustment
    )

    wm["labor_hours_scaled"] = wm["labor_hours"] * labor_scale

    # ensure minimum labor to complete at least one field
    min_labor_required = max(acres.values()) * labor_plant_per_acre

    labor_hours = {
        r["week"]: max(r["labor_hours_scaled"], min_labor_required)
        for _, r in wm.iterrows()
    }


    # ========================================================
    # BUILD MODEL
    # ========================================================
    m = gp.Model("corn_sched_v6_1")
    m.Params.OutputFlag = 1
    m.Params.TimeLimit = time_limit

    # Variables
    Plant = m.addVars(fields, plant_weeks, vtype=GRB.BINARY)
    Harv  = m.addVars(fields, harvest_weeks, vtype=GRB.BINARY)

    plant_week = m.addVars(fields, lb=0)
    harv_week  = m.addVars(fields, lb=0)

    makespan = m.addVar(lb=0)
    penalty  = m.addVars(fields, lb=0)


    # --------------------------------------------------------
    # FIELD LOGIC
    # --------------------------------------------------------
    for f in fields:

        # exactly 1 planting week
        m.addConstr(sum(Plant[f,w] for w in plant_weeks) == 1)

        # exactly 1 harvest week
        m.addConstr(sum(Harv[f,w] for w in harvest_weeks) == 1)

        # time encoding
        m.addConstr(plant_week[f] == sum(w * Plant[f,w] for w in plant_weeks))
        m.addConstr(harv_week[f]  == sum(w * Harv[f,w]  for w in harvest_weeks))

        # lag constraint
        m.addConstr(harv_week[f] >= plant_week[f] + min_harvest_lag_weeks)

        # makespan
        m.addConstr(makespan >= harv_week[f])

        # PWL penalty
        delta = m.addVar(lb=0)
        m.addConstr(delta == harv_week[f] - plant_week[f])
        m.addGenConstrPWL(
            delta, penalty[f],
            [14, 16, 19, 21],
            [2.0, 0.0, 0.0, 1.5]
        )


    # --------------------------------------------------------
    # WEEKLY CAPACITY + LABOR CONSTRAINTS
    # --------------------------------------------------------
    for w in wm["week"].tolist():

        # planting
        m.addConstr(
            sum(acres[f] * Plant[f,w] for f in fields if w in plant_weeks)
            <= plant_capacity[w]
        )

        # harvesting
        m.addConstr(
            sum(acres[f] * Harv[f,w] for f in fields if w in harvest_weeks)
            <= harvest_capacity[w]
        )

        # labor
        m.addConstr(
            sum(acres[f] * labor_plant_per_acre  * Plant[f,w] for f in fields if w in plant_weeks)
          + sum(acres[f] * labor_harvest_per_acre * Harv[f,w] for f in fields if w in harvest_weeks)
            <= labor_hours[w]
        )


    # --------------------------------------------------------
    # OBJECTIVE
    # --------------------------------------------------------
    m.setObjective(
        makespan + early_penalty_weight * sum(penalty[f] for f in fields),
        GRB.MINIMIZE
    )

    m.optimize()

    if m.status == GRB.INFEASIBLE:
        m.computeIIS()
        m.write("infeasible_v6_1.ilp")
        raise RuntimeError("Model infeasible. IIS written to infeasible_v6_1.ilp.")


    # --------------------------------------------------------
    # OUTPUT
    # --------------------------------------------------------
    out = []
    for f in fields:
        p = next((w for w in plant_weeks if Plant[f,w].X > 0.5), None)
        h = next((w for w in harvest_weeks if Harv[f,w].X > 0.5), None)

        out.append({
            "field_id": f,
            "plant_week": p,
            "harvest_week": h,
            "lag_weeks": h - p if p and h else None,
            "penalty": penalty[f].X,
            "makespan": makespan.X
        })

    return pd.DataFrame(out)

