import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from datetime import datetime
import matplotlib.pyplot as plt

# -------------------------
# 1. Load Stock Data
# -------------------------
ticker = "AAPL"
start_date = "2018-01-01"
end_date = datetime.today().strftime("%Y-%m-%d")

df = yf.download(ticker, start=start_date, end=end_date)
df.dropna(inplace=True)

# -------------------------
# 2. Feature Engineering
# -------------------------
df["Return"] = df["Close"].pct_change()
df["SMA_5"] = df["Close"].rolling(5).mean()
df["SMA_20"] = df["Close"].rolling(20).mean()
df["Volatility"] = df["Return"].rolling(20).std()

# RSI function
def compute_rsi(series, window=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

df["RSI"] = compute_rsi(df["Close"])

# Drop NaN rows from indicators
df.dropna(inplace=True)

# -------------------------
# 3. Target Variable
# -------------------------
df["Target"] = np.where(df["Close"].shift(-1) > df["Close"], 1, 0)

features = ["SMA_5", "SMA_20", "Volatility", "RSI"]
X = df[features]
y = df["Target"]

# -------------------------
# 4. Train/Test Split
# -------------------------
split = int(len(df) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# -------------------------
# 5. Train Model
# -------------------------
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# -------------------------
# 6. Backtest Strategy
# -------------------------
df_test = df.iloc[split:].copy()
df_test["Pred"] = y_pred
df_test["Strategy_Return"] = df_test["Pred"].shift(1) * df_test["Return"]

cum_strategy = (1 + df_test["Strategy_Return"]).cumprod()
cum_market = (1 + df_test["Return"]).cumprod()

plt.figure(figsize=(10,6))
plt.plot(cum_market, label="Market (Buy & Hold)")
plt.plot(cum_strategy, label="Model Strategy")
plt.legend()
plt.title(f"Backtest: {ticker}")
plt.show()

# -------------------------
# 7. Export to Excel
# -------------------------
output_file = "stock_predictions.xlsx"
df_test.to_excel(output_file, index=True)
print(f"Results exported to {output_file}")
