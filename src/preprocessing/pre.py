import pandas as pd
import numpy as np
from pathlib import Path

# ===================================================================
#                         1. SYNTHETIC FIELDS
# ===================================================================

def build_synthetic_fields(n_fields=25):
    MIN_LAT, MAX_LAT = 39.0, 41.0
    MIN_LON, MAX_LON = -90.5, -87.5

    rng = np.random.default_rng(42)
    raw_acres = rng.lognormal(mean=4.0, sigma=0.35, size=n_fields)
    scaled_acres = np.interp(raw_acres,
                             (raw_acres.min(), raw_acres.max()),
                             (40, 120))

    lats = rng.uniform(MIN_LAT, MAX_LAT, size=n_fields)
    lons = rng.uniform(MIN_LON, MAX_LON, size=n_fields)

    fields = pd.DataFrame({
        "field_id": [f"F{i+1:03d}" for i in range(n_fields)],
        "crop_name": "CORN",
        "acres": scaled_acres,
        "centroid_lat": lats,
        "centroid_lon": lons
    })

    # Simple regional assignment
    def region(lat):
        if lat < 39.7: return "South"
        if lat < 40.3: return "Central"
        return "North"

    fields["region"] = fields["centroid_lat"].apply(region)

    return fields.sort_values("field_id").reset_index(drop=True)


# ===================================================================
#                2. NASS PLANTING – CLEANING & WINDOWS
# ===================================================================

def build_planting(df_plant):
    df = df_plant[
        (df_plant["State"] == "ILLINOIS") &
        (df_plant["Data Item"].str.contains("PCT PLANTED"))
    ].copy()

    df["week"] = df["Period"].str.extract(r"(\d+)").astype(int)
    df["week_ending"] = pd.to_datetime(df["Week Ending"])
    df["pct_planted"] = pd.to_numeric(df["Value"], errors="coerce")

    df_clean = df[["Year", "week", "week_ending", "pct_planted"]] \
        .dropna().sort_values(["Year", "week"])

    # --- Percentile-based window ---
    windows = []
    for year, g in df_clean.groupby("Year"):
        g = g.sort_values("week").copy()
        g["pct"] = g["pct_planted"].clip(0, 100)

        start = g[g["pct"] >= 5]["week"].min()
        end   = g[g["pct"] >= 85]["week"].min()

        if pd.isna(start):
            nz = g[g["pct"] > 0]["week"].min()
            start = nz if pd.notna(nz) else g["week"].min()

        if pd.isna(end):
            end = g["week"].max()

        windows.append({
            "Year": year,
            "plant_start_week": int(start),
            "plant_end_week": int(end)
        })

    windows_df = pd.DataFrame(windows).sort_values("Year")

    return df_clean, windows_df


# ===================================================================
#                3. NASS HARVEST – CLEANING & WINDOWS
# ===================================================================

def build_harvest(df_harv):
    df = df_harv[
        (df_harv["State"] == "ILLINOIS") &
        (df_harv["Data Item"].str.contains("PCT HARVESTED"))
    ].copy()

    df["week"] = df["Period"].str.extract(r"(\d+)").astype(int)
    df["week_ending"] = pd.to_datetime(df["Week Ending"])
    df["pct_harvested"] = pd.to_numeric(df["Value"], errors="coerce")

    df_clean = df[["Year", "week", "week_ending", "pct_harvested"]] \
        .dropna().sort_values(["Year", "week"])

    # --- Percentile-based window ---
    windows = []
    for year, g in df_clean.groupby("Year"):
        g = g.sort_values("week").copy()
        g["pct"] = g["pct_harvested"].clip(0, 100)

        start = g[g["pct"] >= 5]["week"].min()
        end   = g[g["pct"] >= 85]["week"].min()

        if pd.isna(start):
            nz = g[g["pct"] > 0]["week"].min()
            start = nz if pd.notna(nz) else g["week"].min()

        if pd.isna(end):
            end = g["week"].max()

        windows.append({
            "Year": year,
            "harvest_start_week": int(start),
            "harvest_end_week": int(end)
        })

    windows_df = pd.DataFrame(windows).sort_values("Year")

    return df_clean, windows_df


# ===================================================================
#                   4. DAILY NOAA WEATHER → WEEKLY
# ===================================================================

def build_weekly_weather(daily):
    daily = daily.copy()
    daily["date"] = pd.to_datetime(daily["date"])

    for col in ["PRCP", "TMAX", "TMIN", "TAVG", "AWND"]:
        daily[col] = pd.to_numeric(daily[col], errors="coerce")

    # Impute missing TAVG
    bad = daily["TAVG"].isna()
    daily.loc[bad, "TAVG"] = daily.loc[bad, ["TMAX", "TMIN"]].mean(axis=1)

    daily["year"] = daily["date"].dt.year
    daily["week"] = daily["date"].dt.isocalendar().week.astype(int)

    weekly = (
        daily.groupby(["year", "week"], as_index=False)
             .agg(
                 prcp_week_in=("PRCP", "sum"),
                 TMAX=("TMAX", "mean"),
                 TMIN=("TMIN", "mean"),
                 TAVG=("TAVG", "mean"),
                 AWND=("AWND", "mean"),
             )
    )

    # Capacity factor from rain
    def cap_from_rain(r):
        if pd.isna(r): return 0.8
        if r == 0: return 1.0
        if r < 0.5: return 0.9
        if r < 1.5: return 0.7
        if r < 3.0: return 0.4
        return 0.2

    weekly["capacity_factor"] = weekly["prcp_week_in"].apply(cap_from_rain)

    # bucket
    def bucket(r):
        if pd.isna(r): return "missing"
        if r == 0: return "dry"
        if r < 0.5: return "light"
        if r < 1.5: return "moderate"
        if r < 3.0: return "heavy"
        return "very_heavy"

    weekly["rain_category"] = weekly["prcp_week_in"].apply(bucket)

    return weekly


# ===================================================================
#                     5. USDA LABOR → WEEKLY LABOR
# ===================================================================

def build_weekly_labor(df_labor):
    df = df_labor[df_labor["Data Item"].str.contains("LABOR", case=False)].copy()
    df["Value"] = df["Value"].str.replace(",", "")
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")

    yearly = df.groupby("Year")["Value"].sum().reset_index()
    yearly = yearly.rename(columns={"Value": "total_workers"})

    def season_curve(year, workers):
        base = workers * 40
        out = []
        for wk in range(1, 53):
            if 16 <= wk <= 22:
                mul = 1.2
            elif 35 <= wk <= 45:
                mul = 1.5
            else:
                mul = 0.75
            out.append({"year": year, "week": wk, "labor_hours": base * mul})
        return pd.DataFrame(out)

    weekly = pd.concat([season_curve(r["Year"], r["total_workers"]) for _, r in yearly.iterrows()])
    return yearly, weekly


# ===================================================================
#               6. MASTER WEEKLY TABLE (MERGE EVERYTHING)
# ===================================================================

def build_master_table(weekly_weather, weekly_labor, plant_wins, harv_wins):
    wm = weekly_weather.copy()

    wm = wm.merge(weekly_labor, on=["year", "week"], how="left")
    wm = wm.merge(plant_wins.rename(columns={"Year": "year"}), on="year", how="left")
    wm = wm.merge(harv_wins.rename(columns={"Year": "year"}), on="year", how="left")

    wm["is_plant_window"] = (wm["week"] >= wm["plant_start_week"]) & \
                            (wm["week"] <= wm["plant_end_week"])
    wm["is_harvest_window"] = (wm["week"] >= wm["harvest_start_week"]) & \
                              (wm["week"] <= wm["harvest_end_week"])

    wm = wm.sort_values(["year", "week"]).reset_index(drop=True)
    return wm


# ===================================================================
#                     7. RUN PIPELINE + SAVE OUTPUTS
# ===================================================================

def run_pipeline_v6():
    # ------- Load raw data -------
    df_plant = pd.read_csv("../../data/raw/CORNPROGRESSMEASURED IN PCT PLANTED.csv")
    df_harv = pd.read_csv("../../data/raw/CORN, GRAIN – PROGRESS, MEASURED IN PCT HARVESTED.csv")
    df_daily = pd.read_csv("../../data/raw/noaa_il_daily_raw.csv")
    df_labor = pd.read_csv("../../data/raw/no_of_worker2.csv")

    # ------- Step-by-step -------
    fields = build_synthetic_fields()
    plant_clean, plant_wins = build_planting(df_plant)
    harv_clean, harv_wins = build_harvest(df_harv)
    weekly_weather = build_weekly_weather(df_daily)

    labor_yearly, labor_weekly = build_weekly_labor(df_labor)

    weekly_master = build_master_table(
        weekly_weather, labor_weekly, plant_wins, harv_wins
    )

    # ------- Save -------
    Path("../../data/processed").mkdir(exist_ok=True)

    fields.to_csv("../../data/processed/illinois_corn_fields_clean.csv", index=False)
    plant_clean.to_csv("../../data/processed/nass_corn_planting_weekly_clean.csv", index=False)
    plant_wins.to_csv("../../data/processed/nass_corn_planting_windows.csv", index=False)
    harv_clean.to_csv("../../data/processed/nass_corn_harvest_weekly_clean.csv", index=False)
    harv_wins.to_csv("../../data/processed/nass_corn_harvest_windows.csv", index=False)
    weekly_weather.to_csv("../../data/processed/noaa_il_weekly_clean.csv", index=False)
    labor_yearly.to_csv("../../data/processed/labor_illinois_yearly_clean.csv", index=False)
    labor_weekly.to_csv("../../data/processed/labor_weekly_capacity_clean.csv", index=False)
    weekly_master.to_csv("../../data/processed/master_weekly_table.csv", index=False)

    print("\n🎉 Pipeline v6 completed. Files saved under data/processed/")

