import requests
import pandas as pd
import matplotlib.pyplot as plt

latitude = 22.5726
longitude = 88.3639

url = "https://archive-api.open-meteo.com/v1/archive"

params = {
    "latitude": latitude,
    "longitude": longitude,
    "start_date": "2020-05-15",
    "end_date": "2020-06-17",
    "hourly": "shortwave_radiation,temperature_2m,cloud_cover",
    "timezone": "Asia/Kolkata"
}

response = requests.get(url, params=params)

response.raise_for_status()

# Convert response to JSON
weather_data = response.json()

print("\nOpen-Meteo returned location:")
print("Latitude:", weather_data["latitude"])
print("Longitude:", weather_data["longitude"])
print("Timezone:", weather_data["timezone"])

weather_df = pd.DataFrame({
    "datetime": weather_data["hourly"]["time"],
    "sw radiation": weather_data["hourly"]["shortwave_radiation"],
    "temp 2m": weather_data["hourly"]["temperature_2m"],
    "cloud cover": weather_data["hourly"]["cloud_cover"]
})


# Convert datetime
weather_df["datetime"] = pd.to_datetime(
    weather_df["datetime"]
)


# -----------------------------------
# Display Open-Meteo data
# -----------------------------------

print("\nOpen-Meteo Weather Data:")
print(weather_df.head())

print("\nOpen-Meteo data shape:")
print(weather_df.shape)

print("\nOpen-Meteo data columns:")
print(weather_df.columns)


# -----------------------------------
# Save Open-Meteo data
# -----------------------------------

weather_df.to_csv(
    "data/plant1_openmeteo.csv",
    index=False
)

print("\nData saved to:")
print("data/plant1_openmeteo.csv")


# ============================================================
# 3.3 Merge Open-Meteo Data with Plant 1 Hourly Data
# ============================================================

# Load our previously created hourly Plant 1 dataset
plant_hourly = pd.read_csv(
    "data/plant1_hourly.csv"
)


# Convert datetime columns
plant_hourly["datetime"] = pd.to_datetime(
    plant_hourly["datetime"]
)

weather_df["datetime"] = pd.to_datetime(
    weather_df["datetime"]
)


# -----------------------------------
# Convert sensor irradiation
# from kW/m2 to W/m2
# -----------------------------------

# Sensor irradiation is in kW/m2.
# Open-Meteo shortwave radiation is in W/m2.
#
# 1 kW/m2 = 1000 W/m2

plant_hourly["irradiation_wm2"] = (
    plant_hourly["irradiation"] * 1000
)


# -----------------------------------
# Merge both datasets
# -----------------------------------

merged_hourly = pd.merge(
    plant_hourly,
    weather_df,
    on="datetime",
    how="inner"
)


# -----------------------------------
# Display merged data
# -----------------------------------

print("\nMerged Plant + Open-Meteo Data:")
print(merged_hourly.head())

print("\nMerged data shape:")
print(merged_hourly.shape)


# -----------------------------------
# Save merged dataset
# -----------------------------------

merged_hourly.to_csv(
    "data/plant1_hourly_openmeteo.csv",
    index=False
)

print(
    "\nMerged data saved to:"
    " data/plant1_hourly_openmeteo.csv"
)


# ============================================================
# 3.4 Verify Location
# Compare Sensor Irradiation with Open-Meteo Radiation
# for three days
# ============================================================


# -----------------------------------
# Select three days
# -----------------------------------

three_days = merged_hourly[
    merged_hourly["datetime"].dt.date.isin([
        pd.Timestamp("2020-05-20").date(),
        pd.Timestamp("2020-05-21").date(),
        pd.Timestamp("2020-05-22").date()
    ])
].copy()


# -----------------------------------
# Plot both radiation curves
# -----------------------------------

plt.figure(figsize=(12, 6))

plt.plot(
    three_days["datetime"],
    three_days["irradiation_wm2"],
    label="Sensor Irradiation (W/m²)"
)

plt.plot(
    three_days["datetime"],
    three_days["sw radiation"],
    label="Open-Meteo Shortwave Radiation (W/m²)"
)

plt.xlabel("Datetime")
plt.ylabel("Radiation (W/m²)")
plt.title(
    "Sensor Irradiation vs Open-Meteo Shortwave Radiation"
)
plt.legend()
plt.grid(True)

plt.xticks(rotation=45)

plt.tight_layout()

plt.show()


# ============================================================
# Report Peak Hour for Each Day
# ============================================================

print("\n========================================")
print("3.4 PEAK HOUR COMPARISON")
print("========================================")


for day in [
    pd.Timestamp("2020-05-20").date(),
    pd.Timestamp("2020-05-21").date(),
    pd.Timestamp("2020-05-22").date()
]:

    daily_data = merged_hourly[
        merged_hourly["datetime"].dt.date == day
    ].copy()


    # Sensor peak
    sensor_peak_index = daily_data[
        "irradiation_wm2"
    ].idxmax()

    sensor_peak_time = daily_data.loc[
        sensor_peak_index,
        "datetime"
    ]

    sensor_peak_value = daily_data.loc[
        sensor_peak_index,
        "irradiation_wm2"
    ]


    # Open-Meteo peak
    openmeteo_peak_index = daily_data[
        "sw radiation"
    ].idxmax()

    openmeteo_peak_time = daily_data.loc[
        openmeteo_peak_index,
        "datetime"
    ]

    openmeteo_peak_value = daily_data.loc[
        openmeteo_peak_index,
        "sw radiation"
    ]


    # Calculate difference in peak hours
    peak_difference = abs(
        (
            sensor_peak_time - openmeteo_peak_time
        ).total_seconds()
    ) / 3600


    print("\nDate:", day)

    print(
        "Sensor peak:",
        sensor_peak_time.strftime("%H:%M"),
        "| Value:",
        round(sensor_peak_value, 2),
        "W/m²"
    )

    print(
        "Open-Meteo peak:",
        openmeteo_peak_time.strftime("%H:%M"),
        "| Value:",
        round(openmeteo_peak_value, 2),
        "W/m²"
    )

    print(
        "Peak time difference:",
        peak_difference,
        "hour(s)"
    )


    # -----------------------------------
    # Location/timezone check
    # -----------------------------------

    if peak_difference >= 1:
        print(
            "WARNING: Peak times differ by "
            "1 hour or more."
        )
        print(
            "Check coordinates and timezone."
        )
    else:
        print(
            "OK: Peak times are aligned "
            "within 1 hour."
        )

