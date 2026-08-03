import pandas as pd
import numpy as np
from catboost import CatBoostRegressor

# Load trained model
model = CatBoostRegressor()
model.load_model("catboost_model.cbm")

# Load validation data
validation = pd.read_csv("data/validation.csv")

# ---------- Apply the SAME preprocessing ----------

validation.loc[validation["weight"] <= 0, "weight"] = np.nan

validation["weight"] = validation["weight"].fillna(validation["weight"].median())
validation["market_index"] = validation["market_index"].fillna(
    validation["market_index"].median()
)

validation["route"] = (
    validation["pickup"] + "_" + validation["delivery"]
)

validation["date"] = pd.to_datetime(validation["date"])

validation["month"] = validation["date"].dt.month
validation["day"] = validation["date"].dt.day
validation["dayofweek"] = validation["date"].dt.dayofweek

validation["month_sin"] = np.sin(
    2 * np.pi * validation["month"] / 12
)
validation["month_cos"] = np.cos(
    2 * np.pi * validation["month"] / 12
)

validation["day_sin"] = np.sin(
    2 * np.pi * validation["day"] / 31
)
validation["day_cos"] = np.cos(
    2 * np.pi * validation["day"] / 31
)

validation["lat_diff"] = abs(
    validation["pickup_lat"] - validation["delivery_lat"]
)
validation["lon_diff"] = abs(
    validation["pickup_lon"] - validation["delivery_lon"]
)

# Save IDs
load_ids = validation["load_id"]

# Remove columns not used for prediction
validation = validation.drop(columns=["load_id", "date"])

# Generate predictions
predictions = model.predict(validation)

# Create submission
submission = pd.DataFrame({
    "load_id": load_ids,
    "predicted_rate": predictions
})

submission.to_csv(
    "validation_predictions.csv",
    index=False
)

print("Saved validation_predictions.csv")