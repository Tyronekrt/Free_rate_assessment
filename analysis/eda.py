import pandas as pd

# Load datasets
train = pd.read_csv("data/train-test.csv")
validation = pd.read_csv("data/validation.csv")

# Convert date
train["date"] = pd.to_datetime(train["date"])

# Create date features
train["month"] = train["date"].dt.month
train["day"] = train["date"].dt.day
train["dayofweek"] = train["date"].dt.dayofweek

numeric = train.select_dtypes(include=["number"])

print("\nCorrelation with posted_rate")
print(
    numeric.corr()["posted_rate"]
    .sort_values(ascending=False)
)