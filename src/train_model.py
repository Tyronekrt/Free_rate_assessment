import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Load data
train = pd.read_csv("data/train-test.csv")

# Clean weight
train.loc[train["weight"] <= 0, "weight"] = np.nan

# Fill missing values
train["weight"] = train["weight"].fillna(train["weight"].median())
train["market_index"] = train["market_index"].fillna(train["market_index"].median())

# Create route feature
train["route"] = train["pickup"] + "_" + train["delivery"]

# Convert date
train["date"] = pd.to_datetime(train["date"])

# Date features
train["month"] = train["date"].dt.month
train["day"] = train["date"].dt.day
train["dayofweek"] = train["date"].dt.dayofweek

# Cyclical features
train["month_sin"] = np.sin(2 * np.pi * train["month"] / 12)
train["month_cos"] = np.cos(2 * np.pi * train["month"] / 12)

train["day_sin"] = np.sin(2 * np.pi * train["day"] / 31)
train["day_cos"] = np.cos(2 * np.pi * train["day"] / 31)

# Geographic differences
train["lat_diff"] = abs(train["pickup_lat"] - train["delivery_lat"])
train["lon_diff"] = abs(train["pickup_lon"] - train["delivery_lon"])

# Drop unused columns
train = train.drop(columns=["load_id", "date"])

# Split features/target
X = train.drop(columns=["posted_rate"])
y = train["posted_rate"]

# Categorical columns
categorical_features = [
    "pickup",
    "delivery",
    "equipment",
    "route"
]
# Train/validation split
X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = CatBoostRegressor(
    iterations=2000,
    learning_rate=0.03,
    depth=8,
    loss_function="RMSE",
    eval_metric="RMSE",
    random_seed=42,
    early_stopping_rounds=100,
    verbose=100
)

model.fit(
    X_train,
    y_train,
    cat_features=categorical_features,
    eval_set=(X_valid, y_valid),
    use_best_model=True
)

predictions = model.predict(X_valid)

rmse = np.sqrt(mean_squared_error(y_valid, predictions))
mae = mean_absolute_error(y_valid, predictions)

print(f"\nRMSE: {rmse:.2f}")
print(f"MAE : {mae:.2f}")

model.save_model("catboost_model.cbm")
print("Model saved.")

importance = model.get_feature_importance()

importance_df = (
    pd.DataFrame({
        "Feature": X.columns,
        "Importance": importance
    })
    .sort_values(by="Importance", ascending=False)
)

print("\nFeature Importance")
print(importance_df)

importance_df.to_csv("feature_importance.csv", index=False)
print("\nFeature importance saved to feature_importance.csv")