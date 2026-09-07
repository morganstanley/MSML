"""Generate synthetic exchange rate data for testing.

Creates realistic-looking daily exchange rates for 8 countries vs USD,
spanning 20 years (~58k rows). Uses random walks with mean reversion,
volatility clustering, and seasonal patterns to mimic real FX behavior.

Usage:
    generate-synthetic-data --out-dir DIR [--start-date YYYY-MM-DD] ...
    # or, without install:
    python -m alpha_lab.generate_synthetic_data [...]
"""

import math
import os
import random
from datetime import datetime, timedelta

import click

# Country configs: (name, base_rate, volatility, mean_reversion_speed)
COUNTRIES = [
    ("Australia", 1.35, 0.008, 0.02),
    ("United_Kingdom", 0.65, 0.007, 0.015),
    ("Canada", 1.25, 0.005, 0.02),
    ("Switzerland", 0.95, 0.006, 0.025),
    ("China", 7.0, 0.003, 0.01),
    ("Japan", 110.0, 0.006, 0.015),
    ("New_Zealand", 1.45, 0.009, 0.02),
    ("Singapore", 1.35, 0.004, 0.02),
]



def generate_exchange_rate(n_days, base_rate, daily_vol, mean_reversion, seed):
    """Generate a realistic exchange rate series using Ornstein-Uhlenbeck process."""
    random.seed(seed)

    rates = [0.0] * n_days
    rates[0] = base_rate

    vol_state = daily_vol

    for t in range(1, n_days):
        # Mean reversion toward base rate
        drift = mean_reversion * (base_rate - rates[t-1])

        # Volatility clustering
        vol_state = 0.9 * vol_state + 0.1 * daily_vol * (1 + random.expovariate(2.0))

        # Random shock
        shock = random.gauss(0, vol_state * rates[t-1])

        # Small seasonal component
        day_of_year = t % 365
        seasonal = 0.002 * base_rate * math.sin(2 * math.pi * day_of_year / 365)

        rates[t] = rates[t-1] + drift + shock + seasonal

        # Ensure positive
        rates[t] = max(rates[t], base_rate * 0.5)

    return rates


def generate_features(n_days, seed):
    """Generate exogenous features."""
    random.seed(seed)

    # Generate raw features
    interest_rate_diff = [0.0] * n_days
    inflation_diff = [0.0] * n_days
    trade_balance = [0.0] * n_days
    volatility_index = [0.0] * n_days

    cumsum_ir = 0.0
    cumsum_inf = 0.0

    for i in range(n_days):
        cumsum_ir += random.gauss(0, 0.5) * 0.01
        cumsum_inf += random.gauss(0, 0.3) * 0.005
        interest_rate_diff[i] = cumsum_ir
        inflation_diff[i] = cumsum_inf
        trade_balance[i] = random.gauss(0, 1)
        volatility_index[i] = abs(random.gauss(15, 5))

    # Simple moving average smoothing
    def smooth(arr, window=20):
        result = []
        for i in range(len(arr)):
            start = max(0, i - window + 1)
            result.append(sum(arr[start:i+1]) / (i - start + 1))
        return result

    return {
        "interest_rate_diff": smooth(interest_rate_diff),
        "inflation_diff": smooth(inflation_diff),
        "trade_balance": smooth(trade_balance),
        "volatility_index": smooth(volatility_index),
    }


@click.command(help="Generate synthetic exchange rate data for testing.")
@click.option(
    "--start-date",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    default="1990-01-01",
    show_default=True,
    help="First date (inclusive), YYYY-MM-DD.",
)
@click.option(
    "--end-date",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    default="2010-12-31",
    show_default=True,
    help="Last date (inclusive), YYYY-MM-DD.",
)
@click.option(
    "--out-dir",
    type=click.Path(file_okay=False, writable=True),
    required=True,
    help="Output directory (CSV is written as exchange_rates.csv inside it).",
)
@click.option(
    "--seed",
    type=int,
    default=42,
    show_default=True,
    help="Base random seed (each country uses seed+i).",
)
def main(start_date: datetime, end_date: datetime, out_dir: str, seed: int) -> None:
    print("Generating synthetic exchange rate data...")

    dates = []
    current = start_date
    while current <= end_date:
        dates.append(current)
        current += timedelta(days=1)

    n_days = len(dates)

    print("Generating {} days x {} countries = {} rows".format(
        n_days, len(COUNTRIES), n_days * len(COUNTRIES)
    ))

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "exchange_rates.csv")

    with open(out_path, "w") as f:
        f.write("date,country,exchange_rate,interest_rate_diff,inflation_diff,trade_balance,volatility_index\n")

        all_data = []

        for i, (country, base_rate, vol, mr) in enumerate(COUNTRIES):
            print("  {}: base={}, vol={}".format(country, base_rate, vol))

            rates = generate_exchange_rate(
                n_days=n_days,
                base_rate=base_rate,
                daily_vol=vol,
                mean_reversion=mr,
                seed=seed + i,
            )

            features = generate_features(n_days, seed=seed + 100 + i)

            for j in range(n_days):
                all_data.append((
                    dates[j],
                    country,
                    rates[j],
                    features["interest_rate_diff"][j],
                    features["inflation_diff"][j],
                    features["trade_balance"][j],
                    features["volatility_index"][j],
                ))

        # Sort by date, then country
        all_data.sort(key=lambda x: (x[0], x[1]))

        # Write rows
        for row in all_data:
            f.write("{},{},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f}\n".format(
                row[0].strftime("%Y-%m-%d"),
                row[1],
                row[2],
                row[3],
                row[4],
                row[5],
                row[6],
            ))

    print("\nSaved {:,} rows to {}".format(len(all_data), out_path))
    print("Date range: {} to {}".format(
        dates[0].strftime("%Y-%m-%d"),
        dates[-1].strftime("%Y-%m-%d")
    ))
    print("Countries: {}".format(sorted(set(c for _, c, *_ in all_data))))

    # Compute stats per country
    print("\nSample statistics (mean exchange rate per country):")
    by_country = {}
    for row in all_data:
        c = row[1]
        if c not in by_country:
            by_country[c] = []
        by_country[c].append(row[2])

    for c in sorted(by_country.keys()):
        rates = by_country[c]
        mean_rate = sum(rates) / len(rates)
        print("  {}: mean={:.3f}, min={:.3f}, max={:.3f}".format(
            c, mean_rate, min(rates), max(rates)
        ))


if __name__ == "__main__":
    main()
