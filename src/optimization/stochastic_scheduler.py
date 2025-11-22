# src/optimization/stochastic_scheduler.py

# src/optimization/stochastic_scheduler.py

from __future__ import annotations

import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB

from src.optimization.weather_capacity import (
    compute_capacity_factor,
    compute_harvest_weather_factor,
)

# --------------------------------------------------------------------
# 1. Monte Carlo scenario generation (NO solving here)
# --------------------------------------------------------------------


def _sample_weather_year(
    noaa_weekly_all: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Bootstrap-style weather sampling:
    Pick a random historical year and return its weekly weather profile.

    Expects noaa_weekly_all with columns at least:
      ['year', 'week', 'prcp_week_in', 'TAVG', 'TMAX', 'TMIN', 'AWND', ...]
    """
    candidate_years = sorted(noaa_weekly_all["year"].unique())
    sampled_year = int(rng.choice(candidate_years))
    weather_scen = noaa_weekly_all[noaa_weekly_all["year"] == sampled_year].copy()

    return weather_scen[["week", "prcp_week_in", "TAVG", "TMAX", "TMIN", "AWND"]]


def _apply_labor_uncertainty(
    wm_year: pd.DataFrame,
    rng: np.random.Generator,
    sd: float = 0.1,
) -> tuple[pd.DataFrame, float]:
    """
    Apply a single seasonal labor multiplier (e.g. N(1, sd)).
    Clips to [0.7, 1.3] to avoid insane values.
    """
    mult = float(np.clip(rng.normal(loc=1.0, scale=sd), 0.7, 1.3))
    wm = wm_year.copy()
    wm["labor_hours"] = wm["labor_hours"] * mult
    return wm, mult


def generate_scenarios_for_year(
    noaa_weekly_path: str,
    weekly_master_path: str,
    target_year: int,
    n_scenarios: int = 10,
    random_seed: int = 42,
    labor_sd: float = 0.1,
) -> pd.DataFrame:
    """
    Generate a Monte Carlo scenario table for a given year.

    Returns DataFrame with columns:
      ['scenario', 'week', 'capacity_factor',
       'harvest_weather_factor', 'labor_hours']

    This table is used as INPUT to the stochastic Gurobi model
    (build_stochastic_schedule).
    """
    rng = np.random.default_rng(random_seed)

    noaa_weekly_all = pd.read_csv(noaa_weekly_path)
    master = pd.read_csv(weekly_master_path)

    base_year = master[master["year"] == target_year].copy()
    base_year = base_year.sort_values("week")

    if base_year.empty:
        raise ValueError(f"No rows for year {target_year} in master_weekly_table.")

    if "labor_hours" not in base_year.columns:
        raise ValueError("weekly_master must contain 'labor_hours' column.")

    records: list[dict] = []

    for s in range(n_scenarios):
        # 1) Start from baseline yearly slice
        wm_scen = base_year.copy()

        # 2) Sample a random historical weather year
        weather_scen = _sample_weather_year(noaa_weekly_all, rng)

        # 3) Replace weather columns for target year
        wm_scen = wm_scen.drop(
            columns=["prcp_week_in", "TAVG", "TMAX", "TMIN", "AWND"],
            errors="ignore",
        )
        wm_scen = wm_scen.merge(weather_scen, on="week", how="left")

        # 4) Recompute capacity factors
        wm_scen["capacity_factor"] = wm_scen.apply(compute_capacity_factor, axis=1)
        wm_scen["harvest_weather_factor"] = wm_scen.apply(
            compute_harvest_weather_factor, axis=1
        )

        # 5) Apply labor uncertainty
        wm_scen, labor_mult = _apply_labor_uncertainty(wm_scen, rng, sd=labor_sd)

        # 6) Store per-week scenario records
        for _, row in wm_scen.iterrows():
            records.append(
                {
                    "scenario": s,
                    "week": int(row["week"]),
                    "capacity_factor": float(row["capacity_factor"]),
                    "harvest_weather_factor": float(row["harvest_weather_factor"]),
                    "labor_hours": float(row["labor_hours"]),
                    # optional: keep labor multiplier per scenario if you like
                    # "labor_multiplier": labor_mult,
                }
            )

    scen_df = pd.DataFrame(records)
    scen_df = scen_df.sort_values(["scenario", "week"]).reset_index(drop=True)
    return scen_df


# --------------------------------------------------------------------
# 2. Stochastic MILP: robust schedule across many scenarios
# --------------------------------------------------------------------


def build_stochastic_schedule(
    fields_path: str,
    weekly_master_path: str,
    scenario_df: pd.DataFrame,
    target_year: int,
    base_planter_capacity: float = 1400.0,
    base_harvester_capacity: float = 950.0,
    labor_plant_per_acre: float = 0.15,
    labor_harvest_per_acre: float = 0.20,
    min_harvest_lag_weeks: int = 6,
    time_limit: int | None = 120,
    mip_focus: int = 1,
):
    """
    Build a ROBUST stochastic schedule:

    - Plant / Harvest decisions are shared across all scenarios
      (1st-stage decisions).
    - Each scenario has its own weekly machinery & labor capacities.
    - Schedule must satisfy capacity & labor constraints in *every* scenario.
    - Objective: minimize a single robust makespan (latest harvest week).

    Returns:
      schedule_df  (field-level planting/harvest weeks + objective)
      model_status (int)
    """

    # ------------------------------
    # Load data and sets
    # ------------------------------
    fields_df = pd.read_csv(fields_path)
    master = pd.read_csv(weekly_master_path)

    year_slice = master[master["year"] == target_year].copy()
    if year_slice.empty:
        raise ValueError(f"No rows for year {target_year} in master_weekly_table.")

    fields = fields_df["field_id"].tolist()
    area = dict(zip(fields_df["field_id"], fields_df["acres"]))

    all_weeks = sorted(year_slice["week"].unique().tolist())

    # NASS-based windows (from your master table)
    plant_start = int(year_slice["plant_start_week"].iloc[0])
    plant_end = int(year_slice["plant_end_week"].iloc[0])
    harvest_start = int(year_slice["harvest_start_week"].iloc[0])
    harvest_end = int(year_slice["harvest_end_week"].iloc[0])

    plant_weeks = [w for w in all_weeks if plant_start <= w <= plant_end]
    harvest_weeks = [w for w in all_weeks if harvest_start <= w <= harvest_end]

    # scenario sets
    scenarios = sorted(scenario_df["scenario"].unique().tolist())
    weeks = sorted(scenario_df["week"].unique().tolist())

    # dictionaries: capacity & labor indexed by (w, s)
    cap_plant: dict[tuple[int, int], float] = {}
    cap_harvest: dict[tuple[int, int], float] = {}
    labor_cap: dict[tuple[int, int], float] = {}

    for _, row in scenario_df.iterrows():
        w = int(row["week"])
        s = int(row["scenario"])
        cap_plant[w, s] = base_planter_capacity * row["capacity_factor"]
        cap_harvest[w, s] = (
            base_harvester_capacity
            * row["capacity_factor"]
            * row["harvest_weather_factor"]
        )
        labor_cap[w, s] = row["labor_hours"]

    # ------------------------------
    # Build model
    # ------------------------------
    m = gp.Model("stochastic_corn_schedule")

    if time_limit is not None:
        m.setParam(GRB.Param.TimeLimit, time_limit)
    m.setParam(GRB.Param.MIPFocus, mip_focus)

    # 1st-stage decisions (shared across scenarios)
    Plant = m.addVars(fields, plant_weeks, vtype=GRB.BINARY, name="Plant")
    Harvest = m.addVars(fields, harvest_weeks, vtype=GRB.BINARY, name="Harvest")

    # Field-level harvest week and global makespan
    harvest_week_var = m.addVars(
        fields,
        lb=harvest_start,
        ub=harvest_end,
        vtype=GRB.CONTINUOUS,
        name="HarvestWeek",
    )
    makespan = m.addVar(
        lb=harvest_start,
        ub=harvest_end,
        vtype=GRB.CONTINUOUS,
        name="Makespan",
    )

    # ------------------------------
    # Constraints
    # ------------------------------

    # 1) Each field planted exactly once within planting window
    for f in fields:
        m.addConstr(
            gp.quicksum(Plant[f, w] for w in plant_weeks) == 1,
            name=f"PlantOnce_{f}",
        )

    # 2) Each field harvested exactly once within harvest window
    for f in fields:
        m.addConstr(
            gp.quicksum(Harvest[f, w] for w in harvest_weeks) == 1,
            name=f"HarvestOnce_{f}",
        )

    # 3) Harvest after planting + minimum lag
    for f in fields:
        for wp in plant_weeks:
            for wh in harvest_weeks:
                if wh < wp + min_harvest_lag_weeks:
                    m.addConstr(
                        Plant[f, wp] + Harvest[f, wh] <= 1,
                        name=f"NoEarlyHarvest_{f}_{wp}_{wh}",
                    )

    # 4) Define continuous harvest_week_var[f] as weighted sum of harvest weeks
    for f in fields:
        m.addConstr(
            harvest_week_var[f]
            == gp.quicksum(wh * Harvest[f, wh] for wh in harvest_weeks),
            name=f"HarvestWeekDef_{f}",
        )

    # 5) Makespan is max over fields
    for f in fields:
        m.addConstr(
            makespan >= harvest_week_var[f],
            name=f"Makespan_ge_field_{f}",
        )

    # 6) Scenario-specific capacities & labor
    for s in scenarios:
        for w in weeks:

            # planter capacity (only if w is a planting week)
            if w in plant_weeks:
                m.addConstr(
                    gp.quicksum(area[f] * Plant[f, w] for f in fields)
                    <= cap_plant[w, s],
                    name=f"PlantCap_w{w}_s{s}",
                )

            # harvester capacity (only if w is a harvest week)
            if w in harvest_weeks:
                m.addConstr(
                    gp.quicksum(area[f] * Harvest[f, w] for f in fields)
                    <= cap_harvest[w, s],
                    name=f"HarvCap_w{w}_s{s}",
                )

            # Labor term – always supply a valid LinExpr (not a raw float!)
            plant_term = (
                gp.quicksum(
                    area[f] * labor_plant_per_acre * Plant[f, w] for f in fields
                )
                if w in plant_weeks
                else gp.LinExpr(0)
            )
            harvest_term = (
                gp.quicksum(
                    area[f] * labor_harvest_per_acre * Harvest[f, w] for f in fields
                )
                if w in harvest_weeks
                else gp.LinExpr(0)
            )

            if (w, s) in labor_cap:
                m.addConstr(
                    plant_term + harvest_term <= labor_cap[w, s],
                    name=f"LaborCap_w{w}_s{s}",
                )

    # ------------------------------
    # Objective: minimize robust makespan
    # ------------------------------
    m.setObjective(makespan, GRB.MINIMIZE)

    m.optimize()

    status = m.Status
    obj_val = (
        makespan.X
        if status in (GRB.OPTIMAL, GRB.TIME_LIMIT)
        else None
    )

    # ------------------------------
    # Build schedule DataFrame (field-level)
    # ------------------------------
    rows: list[dict] = []
    for f in fields:
        plant_w = [w for w in plant_weeks if Plant[f, w].X > 0.5]
        harvest_w = [w for w in harvest_weeks if Harvest[f, w].X > 0.5]

        rows.append(
            {
                "field_id": f,
                "plant_week": plant_w[0] if plant_w else None,
                "harvest_week": harvest_w[0] if harvest_w else None,
                "plant_week_continuous": float(plant_w[0]) if plant_w else np.nan,
                "harvest_week_continuous": harvest_week_var[f].X,
                "status": status,
                "objective_makespan": obj_val,
            }
        )

    schedule_df = pd.DataFrame(rows)
    return schedule_df, status
