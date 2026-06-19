# Target: predict direction over a 5-day horizon instead of 1 day.
# A 5-day move is usually a bit easier to predict (short-term noise averages out).
# Note: the "always up" baseline will also be higher on 5 days — compare against it.
HORIZON = 5
df["target"] = (df["Close"].shift(-HORIZON) > df["Close"]).astype(int)

print(df["target"].value_counts())
