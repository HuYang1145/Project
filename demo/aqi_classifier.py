# aqi_classifier.py
# AQI long-range forecast classification system - regression-to-classification post-processing

import numpy as np
import pandas as pd

# AQI six-level classification standard based on PM2.5
AQI_LEVELS = {
    0: {"name": "Excellent", "name_cn": "Excellent", "color": "#00E400", "range": (0, 35)},
    1: {"name": "Good", "name_cn": "Good", "color": "#FFFF00", "range": (35, 75)},
    2: {"name": "Lightly Polluted", "name_cn": "Lightly Polluted", "color": "#FF7E00", "range": (75, 115)},
    3: {"name": "Moderately Polluted", "name_cn": "Moderately Polluted", "color": "#FF0000", "range": (115, 150)},
    4: {"name": "Heavily Polluted", "name_cn": "Heavily Polluted", "color": "#99004C", "range": (150, 250)},
    5: {"name": "Severely Polluted", "name_cn": "Severely Polluted", "color": "#7E0023", "range": (250, 500)},
}


def pm25_to_aqi_level(pm25_value):
    """Map a PM2.5 value to an AQI level."""
    if pm25_value <= 35:
        return 0
    if pm25_value <= 75:
        return 1
    if pm25_value <= 115:
        return 2
    if pm25_value <= 150:
        return 3
    if pm25_value <= 250:
        return 4
    return 5


def classify_predictions(pred_df, horizon_days=2):
    """
    Convert numerical predictions to AQI level classification.

    Args:
        pred_df: Prediction DataFrame (must contain the ``yhat`` column)
        horizon_days: Forecast days (default: 2 days / 48 hours)

    Returns:
        dict: Level distribution, dominant level, confidence intervals, etc.
    """
    if pred_df is None or pred_df.empty:
        return None

    # Extract the forecast for the next N days and average by day.
    hours_per_day = 24
    total_hours = min(len(pred_df), horizon_days * hours_per_day)

    daily_levels = []
    daily_pm25 = []

    for day in range(horizon_days):
        start_idx = day * hours_per_day
        end_idx = min(start_idx + hours_per_day, total_hours)

        if start_idx >= total_hours:
            break

        day_avg = pred_df["yhat"].iloc[start_idx:end_idx].mean()
        daily_pm25.append(day_avg)
        daily_levels.append(pm25_to_aqi_level(day_avg))

    # Count the level distribution.
    level_counts = {i: daily_levels.count(i) for i in range(6)}
    dominant_level = max(level_counts, key=level_counts.get)

    # Estimate classification uncertainty if interval forecasts are available.
    uncertainty = None
    if "y_lower" in pred_df.columns and "y_upper" in pred_df.columns:
        lower_levels = [
            pm25_to_aqi_level(pred_df["y_lower"].iloc[i * 24 : (i + 1) * 24].mean())
            for i in range(min(horizon_days, len(pred_df) // 24))
        ]
        upper_levels = [
            pm25_to_aqi_level(pred_df["y_upper"].iloc[i * 24 : (i + 1) * 24].mean())
            for i in range(min(horizon_days, len(pred_df) // 24))
        ]
        uncertainty = {
            "lower_bound": min(lower_levels) if lower_levels else dominant_level,
            "upper_bound": max(upper_levels) if upper_levels else dominant_level,
        }

    return {
        "daily_levels": daily_levels,
        "daily_pm25": daily_pm25,
        "level_distribution": level_counts,
        "dominant_level": dominant_level,
        "uncertainty": uncertainty,
        "horizon_days": len(daily_levels),
    }


def get_health_advice(level):
    """Return health advice for a given AQI level."""
    advice_map = {
        0: "Air quality is excellent and suitable for outdoor activity.",
        1: "Air quality is good and normal outdoor activity is fine.",
        2: "Sensitive groups should reduce outdoor activity.",
        3: "The general public should reduce outdoor activity and keep windows closed.",
        4: "Avoid outdoor activity and wear an N95 mask when outside.",
        5: "Stay indoors, use an air purifier, and treat this as a health emergency.",
    }
    return advice_map.get(level, "")
