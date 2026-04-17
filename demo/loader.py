from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
import streamlit as st

import db_utils

# ==========================================
# 1. API configuration
# ==========================================
API_CONFIG = {
    "OWM": {
        "key": "46368cf05a7d13f95bd414fac0674b9c",
        "lat": 39.98,
        "lon": 116.39,
        "enabled": True,
    },
    "QWEATHER": {
        "key": "c7d726eaa5bd4a8b8006dc47a9f8c237",
        "location_id": "101010100",  # Beijing
        "host": "https://jh3p4233y7.re.qweatherapi.com",
        "enabled": True,
    },
}


# ==========================================
# 2. Utility functions
# ==========================================
def calculate_dew_point(temp_c, humidity_percent):
    """Calculate the dew point using a standard physical formula."""
    if humidity_percent == 0:
        return temp_c
    a = 17.27
    b = 237.7
    humidity_percent = max(humidity_percent, 0.1)
    alpha = ((a * temp_c) / (b + temp_c)) + np.log(humidity_percent / 100.0)
    dew_point = (b * alpha) / (a - alpha)
    return dew_point


def get_aqi_info(pm25):
    """Compute AQI labels using the Chinese PM2.5 standard."""
    if pm25 is None:
        return "--", "off"
    if pm25 <= 35:
        return "Excellent", "success"
    if pm25 <= 75:
        return "Good", "success"
    if pm25 <= 115:
        return "Lightly Polluted", "warning"
    if pm25 <= 150:
        return "Moderately Polluted", "warning"
    if pm25 <= 250:
        return "Heavily Polluted", "error"
    return "Severely Polluted", "error"


# ==========================================
# 3. Data fetching and local synchronization
# ==========================================
db_utils.init_db()


@st.cache_data(ttl=1800)
def _sync_owm_to_db():
    """Pull recent data from the API and append it to the local SQLite cache."""
    cfg = API_CONFIG["OWM"]
    if not cfg["enabled"]:
        return False

    end = int(datetime.now().timestamp())
    start = int((datetime.now() - timedelta(days=45)).timestamp())
    url = (
        "http://api.openweathermap.org/data/2.5/air_pollution/history"
        f"?lat={cfg['lat']}&lon={cfg['lon']}&start={start}&end={end}&appid={cfg['key']}"
    )

    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            for item in data.get("list", []):
                dt = datetime.fromtimestamp(item["dt"])
                pm25 = item["components"]["pm2_5"]
                db_utils.save_realtime_data(station="Dongsi", timestamp=dt, pm25=pm25)
            return True
    except Exception as e:
        print(f"API Sync Error: {e}")
    return False


def fetch_owm_history():
    """Read a long historical series from the local database and feed it to the model."""
    # Trigger a background sync. Cache TTL ensures the real API call only happens every 30 minutes.
    _sync_owm_to_db()

    # Pull up to 1000 recent points from SQLite.
    df = db_utils.get_recent_data(station="Dongsi", limit=1000)

    if df.empty:
        st.error("The local database is empty and the API sync failed. Check the network or the OWM key.")
        return None

    # If the cache is still short, mirror the sequence to stabilize signal decomposition.
    if len(df) < 168:
        st.toast(
            f"Database still warming up ({len(df)} records). Applied mirrored extension to stabilize the sequence end.",
            icon=None,
        )
        mirrored_df = df.copy().iloc[::-1]

        time_diffs = df["ds"].diff().mean()
        if pd.isna(time_diffs):
            time_diffs = timedelta(hours=1)

        first_time = df["ds"].iloc[0]
        mirrored_df["ds"] = [first_time - (i + 1) * time_diffs for i in range(len(mirrored_df))]
        df = pd.concat([mirrored_df, df]).sort_values("ds").reset_index(drop=True)

    return df[["ds", "y"]]


@st.cache_data(ttl=1800)
def fetch_qweather_forecast():
    """Fetch future weather features from QWeather."""
    cfg = API_CONFIG["QWEATHER"]
    if not cfg["enabled"]:
        return None

    url = f"{cfg['host']}/v7/weather/24h?location={cfg['location_id']}&key={cfg['key']}"

    try:
        response = requests.get(url, timeout=5)
        if response.status_code != 200:
            return None

        data = response.json()
        if data["code"] != "200":
            return None

        records = []
        for item in data["hourly"]:
            temp = float(item["temp"])
            humi = float(item["humidity"])

            # The API returns km/h; the model expects m/s.
            wspm_kmh = float(item["windSpeed"])
            wspm_ms = wspm_kmh / 3.6

            records.append(
                {
                    "ds": pd.to_datetime(item["fxTime"]).replace(tzinfo=None),
                    "TEMP": temp,
                    "PRES": float(item["pressure"]),
                    "DEWP": calculate_dew_point(temp, humi),
                    "WSPM": wspm_ms,
                    "RAIN": float(item["precip"]),
                }
            )

        weather_df = pd.DataFrame(records)

        expected_cols = ["TEMP", "PRES", "DEWP", "RAIN", "WSPM"]
        for col in expected_cols:
            if col not in weather_df.columns:
                st.error(f"API weather data is missing the required column: {col}")
                return None

        return weather_df

    except Exception as e:
        st.error(f"Weather API Error: {e}")
        return None
