# fetch_weather_data.py

import requests
import pandas as pd
from datetime import datetime

def get_daily_weather(start_date: datetime, end_date: datetime) -> pd.DataFrame:
    latitude = 40.1106
    longitude = -88.2073

    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")

    url = (
        "https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={latitude}&longitude={longitude}"
        f"&start_date={start_str}&end_date={end_str}"
        "&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,windspeed_10m_max"
        "&timezone=America/Chicago"
    )

    try:
        print("Requesting URL:", url)
        response = requests.get(url, timeout=15)
        print("Received response")
        response.raise_for_status()
        data = response.json()
        daily = data["daily"]

        rows = []
        for i in range(len(daily["time"])):
            date = daily["time"][i]
            max_c = daily["temperature_2m_max"][i]
            min_c = daily["temperature_2m_min"][i]
            precip_mm = daily["precipitation_sum"][i]
            wind_kmh = daily["windspeed_10m_max"][i]

            # Skip rows with missing data
            if None in (max_c, min_c, precip_mm, wind_kmh):
                continue

            max_f = round(max_c * 9 / 5 + 32, 2)
            min_f = round(min_c * 9 / 5 + 32, 2)
            wind_mph = round(wind_kmh * 0.621371, 2)
            precip_in = round(precip_mm / 25.4, 6)

            rows.append({
                "date": date,
                "max_temp_celsius": max_c,
                "min_temp_celsius": min_c,
                "precipitation_mm": precip_mm,
                "max_wind_speed_kmh": wind_kmh,
                "max_temp_fahrenheit": max_f,
                "min_temp_fahrenheit": min_f,
                "max_wind_speed_mph": wind_mph,
                "precipitation_inches": precip_in
            })

        return pd.DataFrame(rows)

    except Exception as e:
        print(f"❌ Error fetching weather data: {e}")
        return pd.DataFrame()

if __name__ == "__main__":
    import sys
    print("sys.argv:", sys.argv)  # Debugging line
    from datetime import datetime

# Parse command line arguments for date range (YYYY-MM-DD)
    if len(sys.argv) >= 3:
        start = datetime.strptime(sys.argv[1], "%Y-%m-%d")
        end = datetime.strptime(sys.argv[2], "%Y-%m-%d")
        print(f"Parsed start: {start}, end: {end}")  # Debugging line
    else:
        start = datetime(2024, 8, 16)
        end = datetime(2024, 8, 25)

    print(f"Fetching weather data from {start.date()} to {end.date()}")
    weather_df = get_daily_weather(start, end)

