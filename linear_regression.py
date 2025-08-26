import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

#Reading CSV:
df = pd.read_csv("pak_inflation.csv")

#convert YEAR object into integer:
df["Year"] = pd.to_datetime(df["Year"]).dt.year.astype(int)

x = df[["Year"]]
y = df["Inflation"]

#Creating Figure and axis:
fig, axis = plt.subplots(figsize=(10,5))

#Train Model:
model = LinearRegression()
model.fit(x, y)

#Predict Future:
future_years = np.array(range(2025, 2030+1)).reshape(-1, 1)
predictions = model.predict(future_years)

for year, predic in zip(range(2025, 2030+1), predictions):
    print(f"Year: {year}\nInflation: {predic:.2f}")

#Peak Inflation over the Years:
peak_inflation_year = df.loc[df["Inflation"].idxmax(), "Year"]
peak_inflation_value = df["Inflation"].max()

#Highlight on Chart:
axis.scatter(peak_inflation_year, peak_inflation_value, s=100, zorder=5, color="red", label="Peak Inflation")

#Plotting
plt.title("Inflation Predictions Using Linear Regression")
plt.xlabel("Year", fontsize=14)
plt.ylabel("Inflation", fontsize=14)
# plt.scatter(x, y, color='red')
# plt.plot(future_years, predictions, color='blue')
plt.plot(color='red')
plt.plot(future_years, predictions, x, y, color='black', marker='D', linestyle='-')
axis.plot(future_years, predictions, color='blue', linestyle='--', marker='o', label="Predicted Inflation (2025–2030)")
plt.grid(True, linestyle='--', color='gray', alpha=0.5)
plt.xticks(df["Year"][::5], rotation=45)
plt.tight_layout()
axis.legend(loc='upper left')
plt.show()




