# aqi_classifier.py
# AQI 长程预测分类系统 - Regression-to-Classification Post-processing

import numpy as np
import pandas as pd

# AQI 6级分类标准 (基于 PM2.5)
AQI_LEVELS = {
    0: {'name': 'Excellent', 'name_cn': '优', 'color': '#00E400', 'range': (0, 35)},
    1: {'name': 'Good', 'name_cn': '良', 'color': '#FFFF00', 'range': (35, 75)},
    2: {'name': 'Lightly Polluted', 'name_cn': '轻度污染', 'color': '#FF7E00', 'range': (75, 115)},
    3: {'name': 'Moderately Polluted', 'name_cn': '中度污染', 'color': '#FF0000', 'range': (115, 150)},
    4: {'name': 'Heavily Polluted', 'name_cn': '重度污染', 'color': '#99004C', 'range': (150, 250)},
    5: {'name': 'Severely Polluted', 'name_cn': '严重污染', 'color': '#7E0023', 'range': (250, 500)}
}

def pm25_to_aqi_level(pm25_value):
    """PM2.5 数值 → AQI 等级映射"""
    if pm25_value <= 35:
        return 0
    elif pm25_value <= 75:
        return 1
    elif pm25_value <= 115:
        return 2
    elif pm25_value <= 150:
        return 3
    elif pm25_value <= 250:
        return 4
    else:
        return 5

def classify_predictions(pred_df, horizon_days=2):
    """
    Convert numerical predictions to AQI level classification

    Args:
        pred_df: Prediction DataFrame (must contain 'yhat' column)
        horizon_days: Forecast days (default 2 days / 48 hours)

    Returns:
        dict: Level distribution, dominant level, confidence intervals, etc.
    """
    if pred_df is None or pred_df.empty:
        return None

    # 提取未来 N 天的预测值 (每天取平均)
    hours_per_day = 24
    total_hours = min(len(pred_df), horizon_days * hours_per_day)

    daily_levels = []
    daily_pm25 = []

    for day in range(horizon_days):
        start_idx = day * hours_per_day
        end_idx = min(start_idx + hours_per_day, total_hours)

        if start_idx >= total_hours:
            break

        # 计算当天平均 PM2.5
        day_avg = pred_df['yhat'].iloc[start_idx:end_idx].mean()
        daily_pm25.append(day_avg)
        daily_levels.append(pm25_to_aqi_level(day_avg))

    # 统计等级分布
    level_counts = {i: daily_levels.count(i) for i in range(6)}
    dominant_level = max(level_counts, key=level_counts.get)

    # 计算置信区间 (如果有)
    uncertainty = None
    if 'y_lower' in pred_df.columns and 'y_upper' in pred_df.columns:
        lower_levels = [pm25_to_aqi_level(pred_df['y_lower'].iloc[i*24:(i+1)*24].mean())
                       for i in range(min(horizon_days, len(pred_df)//24))]
        upper_levels = [pm25_to_aqi_level(pred_df['y_upper'].iloc[i*24:(i+1)*24].mean())
                       for i in range(min(horizon_days, len(pred_df)//24))]
        uncertainty = {
            'lower_bound': min(lower_levels) if lower_levels else dominant_level,
            'upper_bound': max(upper_levels) if upper_levels else dominant_level
        }

    return {
        'daily_levels': daily_levels,
        'daily_pm25': daily_pm25,
        'level_distribution': level_counts,
        'dominant_level': dominant_level,
        'uncertainty': uncertainty,
        'horizon_days': len(daily_levels)
    }

def get_health_advice(level):
    """根据 AQI 等级返回健康建议"""
    advice_map = {
        0: "空气质量优秀，适合户外活动。",
        1: "空气质量良好，可正常户外活动。",
        2: "敏感人群应减少户外活动。",
        3: "一般人群应减少户外活动，关闭门窗。",
        4: "避免户外活动，外出佩戴 N95 口罩。",
        5: "留在室内，使用空气净化器，健康紧急状态。"
    }
    return advice_map.get(level, "")
