import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------------
# Task 2.1 - AC Power vs Irradiation
# -----------------------------------

# Load hourly data
hourly_data = pd.read_csv("data/plant1_hourly.csv")

# Create scatter plot
plt.figure(figsize=(8, 6))

plt.scatter(
    hourly_data["irradiation"],
    hourly_data["ac_power"],
    alpha=0.6
)

# Labels and title
plt.xlabel("Irradiation")
plt.ylabel("AC Power (W)")
plt.title("AC Power vs Irradiation")

# Grid
plt.grid(True)

# Display plot
plt.show()
# -----------------------------------
# Task 2.2 - Module Temp vs Ambient Temp
# -----------------------------------

plt.figure(figsize=(8, 6))

scatter = plt.scatter(
    hourly_data["ambient_temp"],
    hourly_data["module_temp"],
    c=hourly_data["irradiation"],
    alpha=0.7
)

plt.xlabel("Ambient Temperature (°C)")
plt.ylabel("Module Temperature (°C)")
plt.title("Module Temperature vs Ambient Temperature")

plt.colorbar(scatter, label="Irradiation")

plt.grid(True)

plt.show()
# -----------------------------------
# Task 2.3 - AC Power vs DC Power
# -----------------------------------

plt.figure(figsize=(8, 6))

plt.scatter(
    hourly_data["dc_power"],
    hourly_data["ac_power"],
    alpha=0.6
)

plt.xlabel("DC Power (W)")
plt.ylabel("AC Power (W)")
plt.title("AC Power vs DC Power")

plt.grid(True)

plt.show()
# -----------------------------------
# Task 2.4 - Average AC Power
#           for each hour of the day
# -----------------------------------

import pandas as pd
import matplotlib.pyplot as plt

hourly_data = pd.read_csv("data/plant1_hourly.csv")

hourly_data["datetime"] = pd.to_datetime(
    hourly_data["datetime"]
)

hourly_data["hour"] = hourly_data["datetime"].dt.hour

average_ac_power = (
    hourly_data
    .groupby("hour")["ac_power"]
    .mean()
)

plt.figure(figsize=(9, 6))

plt.plot(
    average_ac_power.index,
    average_ac_power.values,
    marker="o"
)

plt.xlabel("Hour of the Day")
plt.ylabel("Average AC Power (W)")
plt.title("Average AC Power for Each Hour of the Day")
plt.xticks(range(24))

plt.grid(True)

plt.show()

print("\nAverage AC Power for each hour:")

for hour, power in average_ac_power.items():
    print(f"{hour:02d}:00 -> {power:.2f} W")