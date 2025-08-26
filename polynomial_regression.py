import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

#Read from the CSV:
df = pd.read_csv("pak_inflation.csv")
df["Year"] = pd.to_datetime(df["Year"]).dt.year.astype(int)

x = df[["Year"]]
y = df["Inflation"]

#Polynomial Features:
degree = 3
poly = PolynomialFeatures(degree=degree)
x_poly = poly.fit_transform(x)

#Training Model:
model = LinearRegression()
model.fit(x_poly, y)

#Prediction:
x_range = pd.DataFrame({"Year": range(df["Year"].min(), 2031)})
x_range_poly = poly.transform(x_range)
y_pred = model.predict(x_range_poly)

#split history and future:
history_mask = x_range["Year"] <= 2024
future_mask = x_range["Year"] >= 2025

#Plotting on the Graph:
plt.figure(figsize=(12, 8))
plt.scatter(df["Year"], df["Inflation"], color="red", label="Actual Inflation")

#History on Graph:
plt.plot(x_range["Year"][history_mask], y_pred[history_mask], color="black", label=f"Past Inflation")

#Future Predictions on graph:
plt.plot(x_range["Year"][future_mask], y_pred[future_mask], color="magenta", linestyle="-.", linewidth=2, label="Future Predictions(2025-2030) ")


plt.xlabel("Year", fontsize=14)
plt.ylabel("Inflation", fontsize=14)
plt.title("Polynomial Regression on Inflation Data")
plt.legend()
plt.show()

#Print
future_years = list(range(2025, 2031))
future_poly = poly.transform(pd.DataFrame({"Year": future_years}))
future_pred = model.predict(future_poly)

for year, predic in zip(future_years, future_pred):
    print(f"{year}: {predic:.2f}%")











