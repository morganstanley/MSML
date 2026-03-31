"""Generate a synthetic multi-asset OHLCV dataset with features for testing."""

import numpy as np
import pandas as pd

np.random.seed(42)

# 3 years of daily data for 5 tickers
dates = pd.bdate_range("2021-01-04", "2023-12-29")
tickers = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]

rows = []
for ticker in tickers:
    # Random walk price
    base_price = {"AAPL": 130, "GOOGL": 1800, "MSFT": 220, "AMZN": 3200, "TSLA": 700}[ticker]
    volatility = {"AAPL": 0.02, "GOOGL": 0.018, "MSFT": 0.017, "AMZN": 0.022, "TSLA": 0.035}[ticker]

    returns = np.random.normal(0.0003, volatility, len(dates))
    # Add some autocorrelation and mean-reversion
    for i in range(1, len(returns)):
        returns[i] += -0.05 * returns[i - 1]  # slight mean reversion

    close = base_price * np.cumprod(1 + returns)
    high = close * (1 + np.abs(np.random.normal(0, 0.008, len(dates))))
    low = close * (1 - np.abs(np.random.normal(0, 0.008, len(dates))))
    open_ = low + (high - low) * np.random.uniform(0.3, 0.7, len(dates))
    volume = np.random.lognormal(mean=17, sigma=0.5, size=len(dates)).astype(int)

    # Some features
    sma_20 = pd.Series(close).rolling(20).mean().values
    sma_50 = pd.Series(close).rolling(50).mean().values
    rsi_14 = 50 + np.random.normal(0, 15, len(dates))  # fake RSI-like
    rsi_14 = np.clip(rsi_14, 5, 95)

    # Forward return (target) - 5-day forward return
    fwd_return_5d = pd.Series(close).pct_change(5).shift(-5).values

    for i, date in enumerate(dates):
        row = {
            "date": date,
            "ticker": ticker,
            "open": round(open_[i], 2),
            "high": round(high[i], 2),
            "low": round(low[i], 2),
            "close": round(close[i], 2),
            "volume": int(volume[i]),
            "sma_20": round(sma_20[i], 2) if not np.isnan(sma_20[i]) else None,
            "sma_50": round(sma_50[i], 2) if not np.isnan(sma_50[i]) else None,
            "rsi_14": round(rsi_14[i], 2),
            "fwd_return_5d": round(fwd_return_5d[i], 6) if not np.isnan(fwd_return_5d[i]) else None,
        }
        # Sprinkle in some data quality issues
        if np.random.random() < 0.005:
            row["volume"] = None  # missing volume
        if np.random.random() < 0.003:
            row["close"] = -abs(row["close"])  # impossible negative price
        rows.append(row)

df = pd.DataFrame(rows)

# Add a duplicate row (~0.1% of data)
dup_indices = np.random.choice(len(df), size=5, replace=False)
dups = df.iloc[dup_indices].copy()
df = pd.concat([df, dups], ignore_index=True)
df = df.sort_values(["date", "ticker"]).reset_index(drop=True)

df.to_csv("sample_equity_data.csv", index=False)
df.to_parquet("sample_equity_data.parquet", index=False)
print(f"Generated {len(df)} rows, {len(df.columns)} columns")
print(f"Date range: {df['date'].min()} to {df['date'].max()}")
print(f"Tickers: {df['ticker'].unique().tolist()}")
print(f"Saved to sample_equity_data.csv and .parquet")
