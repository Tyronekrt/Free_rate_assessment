import pandas as pd

train = pd.read_csv("data/train-test.csv")

route = train[
    (train["pickup"] == "Lexington") &
    (train["delivery"] == "Fort Wayne")
]

print("Rows:", len(route))
print(route.head())

print("\nNumeric averages:")
print(route[[
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
    "market_index",
    "quote_signal"
]].mean())