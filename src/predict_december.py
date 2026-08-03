import pandas as pd
import numpy as np
from catboost import CatBoostRegressor

# Load model
model = CatBoostRegressor()
model.load_model("catboost_model.cbm")

# Load December file
december = pd.read_csv("data/december-chart-inputs.csv")

# ------------------------------------------------------------------
# Fill in the missing features expected by the model
# (computed from the Lexington -> Fort Wayne training records)
# ------------------------------------------------------------------

december["pickup_lat"] = 36.991520
december["pickup_lon"] = -84.998760
december["delivery_lat"] = 41.315610
december["delivery_lon"] = -85.362060

december["market_index"] = 1.041472
december["quote_signal"] = 2.010673

# Route feature
december["route"] = (
    december["pickup"] + "_" + december["delivery"]
)

# Date features
december["date"] = pd.to_datetime(december["date"])

december["month"] = december["date"].dt.month
december["day"] = december["date"].dt.day
december["dayofweek"] = december["date"].dt.dayofweek

# Cyclical features
december["month_sin"] = np.sin(2 * np.pi * december["month"] / 12)
december["month_cos"] = np.cos(2 * np.pi * december["month"] / 12)

december["day_sin"] = np.sin(2 * np.pi * december["day"] / 31)
december["day_cos"] = np.cos(2 * np.pi * december["day"] / 31)

# Geographic difference features
december["lat_diff"] = abs(
    december["pickup_lat"] - december["delivery_lat"]
)

december["lon_diff"] = abs(
    december["pickup_lon"] - december["delivery_lon"]
)

# ------------------------------------------------------------------
# Build features in the SAME order as training
# ------------------------------------------------------------------

features = december[
    [
        "pickup",
        "delivery",
        "pickup_lat",
        "pickup_lon",
        "delivery_lat",
        "delivery_lon",
        "distance",
        "equipment",
        "weight",
        "market_index",
        "quote_signal",
        "route",
        "month",
        "day",
        "dayofweek",
        "month_sin",
        "month_cos",
        "day_sin",
        "day_cos",
        "lat_diff",
        "lon_diff",
    ]
]

# Predict
december["predicted_rate"] = model.predict(features)

# Save ONLY the required columns
output = december[
    [
        "pickup",
        "delivery",
        "distance",
        "equipment",
        "weight",
        "date",
        "predicted_rate",
    ]
]

output.to_csv(
    "data/december-chart-inputs.csv",
    index=False,
)

print("Saved December predictions.")