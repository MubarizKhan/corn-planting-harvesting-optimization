# src/visualization/viz_v3.py

from pathlib import Path
from typing import Optional

import pandas as pd
import matplotlib.pyplot as plt

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    _HAS_PLOTLY = True
except ImportError:
    _HAS_PLOTLY = False


# ---------------------------------------------------------------------
# 1. Build weekly_view that matches MILP V3 capacity + labor logic
# ---------------------------------------------------------------------
def build_weekly_view_v3(
    schedule_path: str,
    fields_path: str,
    weekly_master_path: str,
    target_year: int,
    base_planter_capacity: float,
    base_harvester_capacity: float,
    labor_plant_per_acre: float,
    labor_harvest_per_acre: float,
    statewide_corn_acres: float,
    planter_machines: int = 1,
    harvester_machines: int = 1,
) -> pd.DataFrame:
    """
    Build a weekly summary view that is consistent with MILP Scheduler V3.

    - Aggregates master_weekly to one state-level row per week
    - Scales statewide labor down to the synthetic farm using acreage ratio
    - Reconstructs planter / harvester capacities using the same logic
      as the MILP (capacity_factor, weather/travel factors, machines)

    Returns one row per week with columns:
      week,
      capacity_factor,
      labor_hours_statewide,   # original statewide labor
      labor_hours_scaled,      # scaled labor for synthetic farm
      plant_acres,
      harvest_acres,
      plant_capacity_model,
      harvest_capacity_model,
      plant_utilization,
      harvest_utilization,
      labor_demand,
      labor_utilization
    """

    schedule_path = Path(schedule_path)
    fields_path = Path(fields_path)
    weekly_master_path = Path(weekly_master_path)

    if not schedule_path.exists():
        raise FileNotFoundError(f"Schedule file not found: {schedule_path}")
    if not fields_path.exists():
        raise FileNotFoundError(f"Fields file not found: {fields_path}")
    if not weekly_master_path.exists():
        raise FileNotFoundError(f"Weekly master file not found: {weekly_master_path}")

    sched = pd.read_csv(schedule_path)
    fields = pd.read_csv(fields_path)
    wm = pd.read_csv(weekly_master_path)

    # ------------------------------------------------------------------
    # Merge acres onto schedule
    # ------------------------------------------------------------------
    if "acres" not in fields.columns:
        raise ValueError("fields_path must contain an 'acres' column.")

    sched = sched.merge(fields[["field_id", "acres"]], on="field_id", how="left")

    # ------------------------------------------------------------------
    # Filter master to target year & basic checks
    # ------------------------------------------------------------------
    wm_year = wm[wm["year"] == target_year].copy()
    if wm_year.empty:
        raise ValueError(f"No rows in master_weekly_table for year={target_year}")

    wm_year["week"] = wm_year["week"].astype(int)

    required_cols = ["week", "capacity_factor", "labor_hours"]
    missing = [c for c in required_cols if c not in wm_year.columns]
    if missing:
        raise ValueError(f"Missing required columns in weekly master: {missing}")

    # Optional columns used by V3; default to 1.0 if missing
    if "planter_travel_factor" not in wm_year.columns:
        wm_year["planter_travel_factor"] = 1.0
    if "harvest_weather_factor" not in wm_year.columns:
        wm_year["harvest_weather_factor"] = 1.0

    # ------------------------------------------------------------------
    # 1a. Collapse any region-level rows to a single row per week
    # ------------------------------------------------------------------
    wm_weekly = (
        wm_year.groupby("week", as_index=False)
        .agg(
            capacity_factor=("capacity_factor", "mean"),
            labor_hours=("labor_hours", "sum"),  # statewide total
            planter_travel_factor=("planter_travel_factor", "mean"),
            harvest_weather_factor=("harvest_weather_factor", "mean"),
        )
        .sort_values("week")
        .reset_index(drop=True)
    )

    # ------------------------------------------------------------------
    # 1b. Compute acreage-based labor scaling (same idea as MILP)
    # ------------------------------------------------------------------
    synthetic_acres = float(fields["acres"].sum())
    if statewide_corn_acres <= 0:
        raise ValueError("statewide_corn_acres must be positive.")

    labor_scale = synthetic_acres / float(statewide_corn_acres)

    print(
        f"[VIZ V3] Synthetic acres = {synthetic_acres:.1f}, "
        f"statewide reference = {statewide_corn_acres:.1f}, "
        f"labor scale = {labor_scale:.6f}"
    )

    wm_weekly["labor_hours_statewide"] = pd.to_numeric(
        wm_weekly["labor_hours"], errors="coerce"
    )
    if wm_weekly["labor_hours_statewide"].isna().any():
        wm_weekly["labor_hours_statewide"] = wm_weekly["labor_hours_statewide"].fillna(
            wm_weekly["labor_hours_statewide"].max()
        )

    # scaled labor capacity for the synthetic farm
    wm_weekly["labor_hours_scaled"] = wm_weekly["labor_hours_statewide"] * labor_scale

    # ------------------------------------------------------------------
    # 2. Plant & harvest acres per week (from MILP schedule)
    # ------------------------------------------------------------------
    plant_by_week = (
        sched.groupby("plant_week")["acres"]
        .sum()
        .rename("plant_acres")
        .reset_index()
        .rename(columns={"plant_week": "week"})
    )

    harvest_by_week = (
        sched.groupby("harvest_week")["acres"]
        .sum()
        .rename("harvest_acres")
        .reset_index()
        .rename(columns={"harvest_week": "week"})
    )

    # ------------------------------------------------------------------
    # 3. Build weekly_view
    # ------------------------------------------------------------------
    weekly_view = (
        wm_weekly[
            [
                "week",
                "capacity_factor",
                "labor_hours_statewide",
                "labor_hours_scaled",
                "planter_travel_factor",
                "harvest_weather_factor",
            ]
        ]
        .merge(plant_by_week, on="week", how="left")
        .merge(harvest_by_week, on="week", how="left")
        .sort_values("week")
        .reset_index(drop=True)
    )

    weekly_view["plant_acres"] = weekly_view["plant_acres"].fillna(0.0)
    weekly_view["harvest_acres"] = weekly_view["harvest_acres"].fillna(0.0)

    # ------------------------------------------------------------------
    # 4. Capacity model: match MILP V3 logic
    # ------------------------------------------------------------------
    weekly_view["plant_capacity_model"] = (
        base_planter_capacity
        * weekly_view["capacity_factor"]
        * weekly_view["planter_travel_factor"]
        * float(planter_machines)
    )

    weekly_view["harvest_capacity_model"] = (
        base_harvester_capacity
        * weekly_view["capacity_factor"]
        * weekly_view["harvest_weather_factor"]
        * float(harvester_machines)
    )

    # ------------------------------------------------------------------
    # 5. Utilization (0–1)
    # ------------------------------------------------------------------
    weekly_view["plant_utilization"] = (
        weekly_view["plant_acres"]
        / weekly_view["plant_capacity_model"].replace(0, pd.NA)
    ).fillna(0.0)

    weekly_view["harvest_utilization"] = (
        weekly_view["harvest_acres"]
        / weekly_view["harvest_capacity_model"].replace(0, pd.NA)
    ).fillna(0.0)

    # Labor demand & utilization vs *scaled* labor hours
    weekly_view["labor_demand"] = (
        weekly_view["plant_acres"] * labor_plant_per_acre
        + weekly_view["harvest_acres"] * labor_harvest_per_acre
    )

    weekly_view["labor_utilization"] = (
        weekly_view["labor_demand"]
        / weekly_view["labor_hours_scaled"].replace(0, pd.NA)
    ).fillna(0.0)

    return weekly_view


# ---------------------------------------------------------------------
# 2. KPI printer
# ---------------------------------------------------------------------
def print_utilization_kpis_v3(weekly_view: pd.DataFrame, target_year: int) -> None:
    def kpi(col_name: str, label: str) -> None:
        max_u = weekly_view[col_name].max()
        max_week = int(weekly_view.loc[weekly_view[col_name].idxmax(), "week"])
        weeks_high = int((weekly_view[col_name] >= 0.8).sum())
        print(f"{label}:")
        print(f"  • Peak utilization = {max_u * 100:.1f}% (week {max_week})")
        print(f"  • Weeks ≥ 80% utilization = {weeks_high}")
        print()

    print(f"=== Utilization KPIs – {target_year} ===")
    kpi("plant_utilization", "Planter")
    kpi("harvest_utilization", "Harvester")
    kpi("labor_utilization", "Labor")


# ---------------------------------------------------------------------
# 3. Matplotlib weekly utilization dashboard
# ---------------------------------------------------------------------
def plot_weekly_utilization_dashboard_v3(
    weekly_view: pd.DataFrame,
    target_year: int,
    figsize=(10, 8),
) -> None:
    weeks = weekly_view["week"].values

    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)

    # ----------------- Planter -----------------
    ax = axes[0]
    ax.plot(
        weeks,
        weekly_view["plant_utilization"] * 100,
        marker="o",
        label="Planter utilization (%)",
    )
    ax.axhline(100, linestyle="--", alpha=0.4)
    ax.set_ylabel("Planter utilization (%)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")

    ax2 = ax.twinx()
    ax2.plot(
        weeks,
        weekly_view["plant_acres"],
        marker="x",
        alpha=0.6,
        label="Plant acres",
    )
    ax2.plot(
        weeks,
        weekly_view["plant_capacity_model"],
        linestyle="--",
        alpha=0.6,
        label="Plant capacity (acres)",
    )
    ax2.set_ylabel("Plant acres / capacity")
    ax2.legend(loc="upper right")

    # ----------------- Harvester -----------------
    ax = axes[1]
    ax.plot(
        weeks,
        weekly_view["harvest_utilization"] * 100,
        marker="o",
        label="Harvester utilization (%)",
    )
    ax.axhline(100, linestyle="--", alpha=0.4)
    ax.set_ylabel("Harvester utilization (%)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")

    ax2 = ax.twinx()
    ax2.plot(
        weeks,
        weekly_view["harvest_acres"],
        marker="x",
        alpha=0.6,
        label="Harvest acres",
    )
    ax2.plot(
        weeks,
        weekly_view["harvest_capacity_model"],
        linestyle="--",
        alpha=0.6,
        label="Harvest capacity (acres)",
    )
    ax2.set_ylabel("Harvest acres / capacity")
    ax2.legend(loc="upper right")

    # ----------------- Labor -----------------
    ax = axes[2]
    ax.plot(
        weeks,
        weekly_view["labor_utilization"] * 100,
        marker="o",
        label="Labor utilization (%)",
    )
    ax.axhline(100, linestyle="--", alpha=0.4)
    ax.set_ylabel("Labor utilization (%)")
    ax.set_xlabel("Week of Year")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")

    ax2 = ax.twinx()
    ax2.plot(
        weeks,
        weekly_view["labor_demand"],
        marker="x",
        alpha=0.6,
        label="Labor demand (hours)",
    )
    ax2.plot(
        weeks,
        weekly_view["labor_hours_scaled"],
        linestyle="--",
        alpha=0.6,
        label="Labor available (scaled, hours)",
    )
    ax2.set_ylabel("Labor hours")
    ax2.legend(loc="upper right")

    fig.suptitle(f"Weekly Utilization Dashboard – {target_year}")
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------
# 4. Plotly interactive version (optional)
# ---------------------------------------------------------------------
def plot_weekly_utilization_plotly_v3(
    weekly_view: pd.DataFrame,
    target_year: int,
    height: int = 900,
) -> Optional["go.Figure"]:
    if not _HAS_PLOTLY:
        print("Plotly is not installed; cannot build interactive dashboard.")
        return None

    weeks = weekly_view["week"].values

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.07,
        specs=[
            [{"secondary_y": True}],
            [{"secondary_y": True}],
            [{"secondary_y": True}],
        ],
        subplot_titles=[
            "Planter Utilization & Capacity",
            "Harvester Utilization & Capacity",
            "Labor Utilization & Capacity",
        ],
    )

    # Planter
    fig.add_trace(
        go.Scatter(
            x=weeks,
            y=weekly_view["plant_utilization"] * 100,
            mode="lines+markers",
            name="Planter utilization (%)",
        ),
        row=1,
        col=1,
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=weeks,
            y=weekly_view["plant_acres"],
            mode="lines+markers",
            name="Plant acres",
        ),
        row=1,
        col=1,
        secondary_y=True,
    )
    fig.add_trace(
        go.Scatter(
            x=weeks,
            y=weekly_view["plant_capacity_model"],
            mode="lines",
            name="Plant capacity (acres)",
        ),
        row=1,
        col=1,
        secondary_y=True,
    )

    # Harvester
    fig.add_trace(
        go.Scatter(
            x=weeks,
            y=weekly_view["harvest_utilization"] * 100,
            mode="lines+markers",
            name="Harvester utilization (%)",
        ),
        row=2,
        col=1,
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=weeks,
            y=weekly_view["harvest_acres"],
            mode="lines+markers",
            name="Harvest acres",
        ),
        row=2,
        col=1,
        secondary_y=True,
    )
    fig.add_trace(
        go.Scatter(
            x=weeks,
            y=weekly_view["harvest_capacity_model"],
            mode="lines",
            name="Harvest capacity (acres)",
        ),
        row=2,
        col=1,
        secondary_y=True,
    )

    # Labor
    fig.add_trace(
        go.Scatter(
            x=weeks,
            y=weekly_view["labor_utilization"] * 100,
            mode="lines+markers",
            name="Labor utilization (%)",
        ),
        row=3,
        col=1,
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=weeks,
            y=weekly_view["labor_demand"],
            mode="lines+markers",
            name="Labor demand (hours)",
        ),
        row=3,
        col=1,
        secondary_y=True,
    )
    fig.add_trace(
        go.Scatter(
            x=weeks,
            y=weekly_view["labor_hours_scaled"],
            mode="lines",
            name="Labor available (scaled, hours)",
        ),
        row=3,
        col=1,
        secondary_y=True,
    )

    fig.update_xaxes(title_text="Week of Year", row=3, col=1)
    fig.update_yaxes(
        title_text="Planter utilization (%)", row=1, col=1, secondary_y=False
    )
    fig.update_yaxes(title_text="Acres", row=1, col=1, secondary_y=True)
    fig.update_yaxes(
        title_text="Harvester utilization (%)", row=2, col=1, secondary_y=False
    )
    fig.update_yaxes(title_text="Acres", row=2, col=1, secondary_y=True)
    fig.update_yaxes(
        title_text="Labor utilization (%)", row=3, col=1, secondary_y=False
    )
    fig.update_yaxes(title_text="Labor hours", row=3, col=1, secondary_y=True)

    fig.update_layout(
        title_text=f"Weekly Utilization Dashboard (Interactive) – {target_year}",
        height=height,
        showlegend=True,
    )

    fig.show()
    return fig
