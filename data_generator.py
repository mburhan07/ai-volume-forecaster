# data_generator.py
# Quantitative AI: Generate synthetic numerical data with mathematical patterns

import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt


def generate_volume_data():
    """
    Generate 3 years of numerical volume data using mathematical functions.
    Includes trend, seasonality, weekly pattern, monthly spikes, and noise.
    """

    # Create date range
    start_date = datetime(2021, 1, 1)
    end_date = datetime(2023, 12, 31)
    dates = pd.date_range(start=start_date, end=end_date, freq="D")

    n_days = len(dates)

    # 1. Linear trend (growing over time)
    trend = np.linspace(1000, 1500, n_days)

    # 2. Yearly seasonal pattern (up and down during the year)
    seasonal_year = 200 * np.sin(2 * np.pi * np.arange(n_days) / 365)

    # 3. Weekly pattern (weekdays higher than weekends)
    weekly = np.array([150 if d.weekday() < 5 else 50 for d in dates])

    # 4. Monthly spike (end of the month bump)
    monthly = np.array([100 if d.day > 25 else 0 for d in dates])

    # 5. Random noise
    np.random.seed(42)
    noise = np.random.normal(0, 50, n_days)

    # Add everything together
    volume = trend + seasonal_year + weekly + monthly + noise

    # Make sure values are not negative
    volume = np.maximum(volume, 0)

    # Create DataFrame
    df = pd.DataFrame(
        {
            "date": dates,
            "volume": volume.round().astype(int),
            "day_of_week": [d.day_name() for d in dates],
            "month": [d.month for d in dates],
            "year": [d.year for d in dates],
        }
    )

    # Nice banner
    print("\033[91m" + "=" * 50)  # Red
    print("\033[97m" + " 🇺🇸 QUANTITATIVE AI DATA GENERATOR 🇺🇸 ")  # White
    print("\033[94m" + "=" * 50 + "\033[0m")  # Blue

    # Show basic stats
    print(f"Generated {len(df)} numerical data points")
    print(f"Date range: {start_date.date()} to {end_date.date()}")
    print(f"Mean (μ): {df['volume'].mean():.2f}")
    print(f"Std Dev (σ): {df['volume'].std():.2f}")
    print(
        f"Coefficient of Variation: {(df['volume'].std() / df['volume'].mean()):.3f}"
    )

    return df


if __name__ == "__main__":
    data = generate_volume_data()
    data.to_csv("data/historical_volumes.csv", index=False)
    print("\n\033[92m✓ Data saved to data/historical_volumes.csv\033[0m")
