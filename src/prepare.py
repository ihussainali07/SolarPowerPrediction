import pandas as pd

# -----------------------------------
# Load Plant 1 generation data
# -----------------------------------

gen1 = pd.read_csv("data/Plant_1_Generation_Data.csv")

sensor1 = pd.read_csv("data/Plant_1_Weather_Sensor_Data.csv")


# -----------------------------------
# Inspect original DATE_TIME
# -----------------------------------

print("Generation DATE_TIME:")
print(gen1["DATE_TIME"].head())

print("\nSensor DATE_TIME:")
print(sensor1["DATE_TIME"].head())


# -----------------------------------
# 1.2 Convert DATE_TIME to datetime
# -----------------------------------

gen1["DATE_TIME"] = pd.to_datetime(
    gen1["DATE_TIME"],
    format="%d-%m-%Y %H:%M"
)

sensor1["DATE_TIME"] = pd.to_datetime(
    sensor1["DATE_TIME"]
)


# -----------------------------------
# Print first and last timestamps
# -----------------------------------

print("\nGeneration first timestamp:")
print(gen1["DATE_TIME"].min())

print("Generation last timestamp:")
print(gen1["DATE_TIME"].max())

print("\nSensor first timestamp:")
print(sensor1["DATE_TIME"].min())

print("Sensor last timestamp:")
print(sensor1["DATE_TIME"].max())


# -----------------------------------
# 1.3 Aggregate inverter power
#     to plant-level power
# -----------------------------------

# Define Day/Night period
gen1["PERIOD"] = gen1["DATE_TIME"].dt.hour.apply(
    lambda x: "Night" if x < 6 or x >= 18 else "Daytime"
)


# -----------------------------------
# Separate Nighttime and Daytime
# -----------------------------------

# Nighttime: 18:00 – 05:59
night_power = gen1[
    (gen1["DATE_TIME"].dt.hour < 6) |
    (gen1["DATE_TIME"].dt.hour >= 18)
].groupby("DATE_TIME")[["AC_POWER", "DC_POWER"]].sum()


# Daytime: 06:00 – 17:59
day_power = gen1[
    (gen1["DATE_TIME"].dt.hour >= 6) &
    (gen1["DATE_TIME"].dt.hour < 18)
].groupby("DATE_TIME")[["AC_POWER", "DC_POWER"]].sum()


print("\nNight-time plant power:")
print(night_power.head())

print("\nDaytime plant power:")
print(day_power.head())


# -----------------------------------
# 1.4 Merge nighttime power with
#     sensor data
# -----------------------------------

merged_data = pd.merge(
    night_power,
    sensor1,
    on="DATE_TIME",
    how="outer",
    indicator=True
)


print("\nMerged nighttime data:")
print(merged_data.head())


# -----------------------------------
# Count timestamps existing in
# only one of the two files
# -----------------------------------

only_night = (
    merged_data["_merge"] == "left_only"
).sum()

only_sensor = (
    merged_data["_merge"] == "right_only"
).sum()


print(
    "\nTimestamps only in nighttime data:",
    only_night
)

print(
    "Timestamps only in sensor data:",
    only_sensor
)

print(
    "\nTotal timestamps existing in only one file:",
    only_night + only_sensor
)


# =================================================
# 1.5 Resample to hourly means
#     INCLUDING BOTH DAYTIME AND NIGHTTIME
# =================================================

# -----------------------------------
# Combine Daytime + Nighttime power
# -----------------------------------

full_power = pd.concat([
    night_power,
    day_power
]).sort_index()


print("\nFull plant-level power:")
print(full_power.head(10))


# -----------------------------------
# Merge full power with sensor data
# -----------------------------------

hourly_data = pd.merge(
    full_power,
    sensor1,
    on="DATE_TIME",
    how="inner"
)


# -----------------------------------
# Keep only required columns
# -----------------------------------

hourly_data = hourly_data[
    [
        "DATE_TIME",
        "AC_POWER",
        "DC_POWER",
        "AMBIENT_TEMPERATURE",
        "MODULE_TEMPERATURE",
        "IRRADIATION"
    ]
]


# -----------------------------------
# Set DATE_TIME as index
# -----------------------------------

hourly_data = hourly_data.set_index("DATE_TIME")


# -----------------------------------
# Resample to hourly means
# -----------------------------------

hourly_data = hourly_data.resample("1h").mean()


# -----------------------------------
# Convert index back to column
# -----------------------------------

hourly_data = hourly_data.reset_index()


# -----------------------------------
# Rename columns
# -----------------------------------

hourly_data = hourly_data.rename(columns={
    "DATE_TIME": "datetime",
    "AC_POWER": "ac_power",
    "DC_POWER": "dc_power",
    "AMBIENT_TEMPERATURE": "ambient_temp",
    "MODULE_TEMPERATURE": "module_temp",
    "IRRADIATION": "irradiation"
})


# -----------------------------------
# Save complete hourly dataset
# -----------------------------------

hourly_data.to_csv(
    "data/plant1_hourly.csv",
    index=False
)


# -----------------------------------
# Display hourly dataset
# -----------------------------------

print("\nHourly data:")
print(hourly_data.head(10))

print("\nHourly data shape:")
print(hourly_data.shape)


# -----------------------------------
# Check daytime and nighttime rows
# -----------------------------------

print("\nDaytime rows:")
print(
    hourly_data[
        (pd.to_datetime(hourly_data["datetime"]).dt.hour >= 6) &
        (pd.to_datetime(hourly_data["datetime"]).dt.hour < 18)
    ].head()
)


print("\nNighttime rows:")
print(
    hourly_data[
        (pd.to_datetime(hourly_data["datetime"]).dt.hour < 6) |
        (pd.to_datetime(hourly_data["datetime"]).dt.hour >= 18)
    ].head()
)


# =================================================
# 1.6 Report missing values
# =================================================

# Number of hourly rows
number_of_rows = len(hourly_data)


# Count rows containing at least one missing value
missing_rows = hourly_data.isna().any(axis=1).sum()


# Count total missing values
total_missing = hourly_data.isna().sum().sum()


print("\n--- Step 1.6 Report ---")

print(
    "Number of hourly rows:",
    number_of_rows
)

print(
    "Number of rows with missing values:",
    missing_rows
)

print(
    "Total missing values:",
    total_missing
)


# -----------------------------------
# Handle missing values
# -----------------------------------

hourly_data = hourly_data.interpolate()


# Check again
print(
    "Missing values after handling:",
    hourly_data.isna().sum().sum()
)


# -----------------------------------
# Save final cleaned dataset again
# -----------------------------------

hourly_data.to_csv(
    "data/plant1_hourly.csv",
    index=False
)

print("\nFinal hourly dataset saved successfully.")