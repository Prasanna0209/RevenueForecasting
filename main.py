
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# 1. Generate Synthetic Dataset

np.random.seed(42)

months = pd.date_range(start="2018-01-01", periods=60, freq="M")
revenue = 20000 + np.arange(60) * 200 + np.random.randint(-3000, 3000, size=60)
promo_spend = np.random.randint(2000, 8000, size=60)
holiday_flag = np.random.choice([0, 1], size=60, p=[0.7, 0.3])

data = pd.DataFrame({
    "Month": months,
    "Revenue": revenue,
    "Promo_Spend": promo_spend,
    "Holiday_Flag": holiday_flag
})

print("Sample dataset:\n", data.head())


# 2. Feature Engineering

data["Month_Num"] = data["Month"].dt.month
data["Year"] = data["Month"].dt.year

X = data[["Month_Num", "Year", "Promo_Spend", "Holiday_Flag"]]
y = data["Revenue"]


# 3. Train-Test Split & Model Training

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)


# 4. Evaluate Model

print("\nModel Coefficients:")
for feature, coef in zip(X.columns, model.coef_):
    print(f"{feature}: {coef:.2f}")

print("\nIntercept:", model.intercept_)


# 5. Scenario Simulation Functions

def simulate_promotion_increase(percentage):
    """Simulate revenue if promo spend increases by X%"""
    X_future = X_test.copy()
    X_future["Promo_Spend"] = X_future["Promo_Spend"] * (1 + percentage / 100)
    return model.predict(X_future)

def simulate_traffic_drop(percentage):
    """Simulate revenue if overall customer traffic drops by X%"""
    y_future = model.predict(X_test)
    return y_future * (1 - percentage / 100)


# 6. Run Simulations

base_forecast = model.predict(X_test)
promo_forecast = simulate_promotion_increase(10)   # +10% promotion
traffic_forecast = simulate_traffic_drop(5)        # -5% traffic


# 7. Plot Results

plt.figure(figsize=(12,6))
plt.plot(data["Month"], data["Revenue"], label="Historical Revenue", color="blue")
plt.plot(data["Month"].iloc[-len(y_test):], base_forecast, label="Base Forecast", color="green", linestyle="--")
plt.plot(data["Month"].iloc[-len(y_test):], promo_forecast, label="Forecast with +10% Promo", color="orange", linestyle="--")
plt.plot(data["Month"].iloc[-len(y_test):], traffic_forecast, label="Forecast with -5% Traffic", color="red", linestyle="--")
plt.xlabel("Month")
plt.ylabel("Revenue")
plt.title("Revenue Forecasting with Scenario Simulation")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
