# Easy win: make the moving-average features stationary.
# AAPL went from ~$40 to ~$250, so raw ma_10 / ma_50 have totally different ranges
# in training vs test, which confuses tree models. Using the price RELATIVE to its
# moving average (a ratio) keeps the feature meaningful across all price levels.

# --- 1) Replace the feature-engineering cell with this ---

# Feature engineering

# Past returns
df["return_1"] = df["Close"].pct_change(1)
df["return_5"] = df["Close"].pct_change(5)
df["return_10"] = df["Close"].pct_change(10)

# Moving averages
df["ma_10"] = df["Close"].rolling(10).mean()
df["ma_50"] = df["Close"].rolling(50).mean()

# Price position relative to its moving averages (stationary across price levels)
# Positive = price above the average (uptrend), negative = below
df["price_vs_ma10"] = df["Close"] / df["ma_10"] - 1
df["price_vs_ma50"] = df["Close"] / df["ma_50"] - 1

# Volatility: standard deviation of returns over the last 10 days
df["volatility_10"] = df["return_1"].rolling(10).std()


# --- 2) Replace the features list with this ---

# Use the relative MA features instead of the raw price levels
features = ["return_1", "return_5", "return_10", "price_vs_ma10", "price_vs_ma50", "volatility_10"]

X = df[features]
y = df["target"]

# Then re-run the Random Forest, XGBoost and baseline cells and compare the new
# mean accuracies against the 53.5% baseline.
