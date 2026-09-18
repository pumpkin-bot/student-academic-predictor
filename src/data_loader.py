import os
import numpy as np
import pandas as pd

def get_or_create_data(file_path: str = "data/student_data.csv", n_samples: int = 600) -> pd.DataFrame:
    if os.path.exists(file_path):
        return pd.read_csv(file_path)

    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    np.random.seed(42)

    study_hours = np.random.uniform(1.0, 10.0, n_samples)
    attendance = np.random.uniform(50.0, 100.0, n_samples)
    past_failures = np.random.choice([0, 1, 2, 3], size=n_samples, p=[0.7, 0.18, 0.08, 0.04])
    internal_score = np.random.uniform(30.0, 95.0, n_samples)
    internet_access = np.random.choice(["yes", "no"], size=n_samples, p=[0.85, 0.15])

    score_metric = (
        0.35 * (attendance / 100.0) +
        0.40 * (internal_score / 100.0) +
        0.25 * (study_hours / 10.0) -
        0.15 * past_failures
    )
    passed = (score_metric > 0.48).astype(int)

    df = pd.DataFrame({
        "study_hours": np.round(study_hours, 2),
        "attendance": np.round(attendance, 2),
        "past_failures": past_failures,
        "internal_score": np.round(internal_score, 2),
        "internet_access": internet_access,
        "target": passed
    })

    df.to_csv(file_path, index=False)
    return df
