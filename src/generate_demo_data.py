"""Create fictional survey responses. No real people or client data."""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def generate_data(path=ROOT / "examples" / "synthetic_survey.csv", rows=1000, seed=42):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "age": rng.integers(18, 76, rows),
        "monthly_visits": rng.poisson(3, rows),
        "satisfaction": rng.integers(1, 6, rows),
        "value_rating": rng.integers(1, 6, rows),
        "channel": rng.choice(["online", "store", "both"], rows),
    })
    score = (0.9 * (df["satisfaction"] - 3)
             + 0.6 * (df["value_rating"] - 3)
             + 0.2 * (df["monthly_visits"] - 3)
             + 0.3 * (df["channel"] == "both") - 0.5)
    probability = 1 / (1 + np.exp(-score))
    df["would_recommend"] = rng.binomial(1, probability)
    for column in ["satisfaction", "value_rating"]:
        df.loc[rng.choice(rows, size=rows // 25, replace=False), column] = np.nan
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


if __name__ == "__main__":
    data = generate_data()
    print(f"Created {len(data)} synthetic responses in examples/synthetic_survey.csv")
