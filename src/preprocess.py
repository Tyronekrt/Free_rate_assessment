import pandas as pd

train = pd.read_csv("data/train-test.csv")

# Treat invalid weights as missing
train.loc[train["weight"] <= 0, "weight"] = pd.NA

# Fill missing values with the median
train["weight"] = train["weight"].fillna(train["weight"].median())
train["market_index"] = train["market_index"].fillna(train["market_index"].median())

# Convert date
train["date"] = pd.to_datetime(train["date"])

# Extract date features
train["month"] = train["date"].dt.month
train["day"] = train["date"].dt.day
train["dayofweek"] = train["date"].dt.dayofweek


# Create a route feature
train["route"] = train["pickup"] + "_" + train["delivery"]

# Rate per mile (only for EDA, not as a feature if it uses the target)
train["rate_per_mile"] = train["posted_rate"] / train["distance"]

print(train[["route", "distance", "posted_rate", "rate_per_mile"]].head())

print(train.head())
