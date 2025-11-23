import numpy as np

def compute_capacity_factor(row):
    """
    Compute machinery capacity factor based on weekly precipitation.
    Matches logic used when building weekly_master.
    
    rain_category:
        light     → 1.0
        moderate  → 0.7
        heavy     → 0.4
    """
    pr = row.get("prcp_week_in", 0)

    if pr < 0.25:
        return 1.0      # light
    elif pr < 0.75:
        return 0.7      # moderate
    else:
        return 0.4      # heavy


# def compute_harvest_weather_factor(row):
#     """
#     Compute harvest slowdown factor using:
#     - precipitation (strong effect)
#     - low drying-degree days
#     - high winds

#     Returns a multiplier ∈ [0.3, 1.0].
#     """

#     pr = row.get("prcp_week_in", 0)
#     tavg = row.get("TAVG", 60)
#     wind = row.get("AWND", 8)

#     # Base from rain
#     if pr < 0.25:
#         f = 1.0
#     elif pr < 0.75:
#         f = 0.75
#     else:
#         f = 0.45

#     # Drying-degree penalty (cool weeks slow dry-down)
#     if tavg < 50:
#         f *= 0.85

#     # High winds slightly reduce harvest speed
#     if wind > 15:
#         f *= 0.9

#     return max(min(f, 1.0), 0.3)    # clamp to [0.3,1.0]
def compute_harvest_weather_factor(row) -> float:
    """
    Extra harvest penalty for bad drying conditions.

    Uses:
      - prcp_week_in : weekly rainfall (inches)
      - TAVG         : avg temp (F)
    """
    prcp = row.get("prcp_week_in", 0.0) or 0.0
    tavg = row.get("TAVG", 60.0) or 60.0

    # Start from 1.0 (no penalty)
    factor = 1.0

    # Heavy rain → strong penalty
    if prcp > 1.5:
        factor *= 0.4
    elif prcp > 0.8:
        factor *= 0.7
    else:
        factor *= 1.0

    # Cool temps → slower drying
    if tavg < 40:
        factor *= 0.6
    elif tavg < 50:
        factor *= 0.8

    # Clamp to [0.2, 1.0]
    return float(np.clip(factor, 0.2, 1.0))