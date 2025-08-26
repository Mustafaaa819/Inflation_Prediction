import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import mplcursors


#Reading from the CSV:
df = pd.read_csv("pak_inflation.csv")
df["Year"] = pd.to_datetime(df["Year"]).dt.year.astype(int)

x = df[["Year"]]
y = df["Inflation"]

#Train Random Forest Model:
forest = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=None, min_samples_leaf=1, min_samples_split=2)
forest.fit(x, y)

#prediction of Future Years:
future_years = np.array(range(2025, 2031)).reshape(-1, 1)
future_prediction = forest.predict(future_years)

#Printing the Predictions:
for year, predic in zip(range(2025, 2031), future_prediction):
    print(f"{year}: {predic:.2f}%")

# Plotting on Graph:
plt.figure(figsize=(12, 8))

#Actual Data of Inflation:
scatter = plt.scatter(df["Year"], df["Inflation"], color="red", marker='o', label="Actual Inflation over the Years")

 # Forest Predicts Inflation of all the Years:
all_years = np.array(range(df["Year"].min(), 2030+1)).reshape(-1, 1)
all_predictions = forest.predict(all_years)
plt.plot(all_years, all_predictions, color='black', linestyle='--',marker='s', linewidth=2, label="Random Forest Prediction for All Years")

#Forest's Prediction for future Years:
line, = plt.plot(future_years, future_prediction, color="red", linestyle='-', linewidth=2, marker='o', label="Future Prediction of Inflation")

plt.axvline(x=2025, color="red", linestyle='--', linewidth=1.5)

plt.xlabel("Year", fontsize=12, fontweight='bold')
plt.ylabel("Inflation (%)", fontsize=12, fontweight='bold')
plt.title("Random Forest Regression on Inflation Data")
plt.legend()
plt.tight_layout()
plt.grid(True, linestyle='--', alpha=0.6)

#Hovering Function:
cursors = mplcursors.cursor([line, scatter], hover=True)
@cursors.connect("add")

def get_hover(sel):
    x, y = sel.target
    sel.annotation.set(text=f"Year: {int(x)}, Inflation: {y:.2f}%", fontsize=12, backgroundcolor="white")

plt.show()




















