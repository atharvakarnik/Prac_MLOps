import joblib
import pandas as pd
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor

FEATURES = ["CHAS", "RM", "TAX", "PTRATIO", "B", "LSTAT"]
TARGET = "MEDV"

def load_boston_frame():
    """
    Try OpenML first (works with modern sklearn and fetches a DataFrame).
    If that fails (no internet), fall back to a local CSV at webapp/boston.csv
    containing the FEATURES plus MEDV.
    """
    try:
        from sklearn.datasets import fetch_openml
        boston = fetch_openml(name="boston", version=1, as_frame=True)
        df = boston.frame
        # OpenML version has lowercase names; normalize to expected uppercase
        df.columns = [c.upper() for c in df.columns]
        return df
    except Exception:
        csv_path = Path(__file__).with_name("boston.csv")
        if not csv_path.exists():
            raise RuntimeError(
                "Could not fetch Boston dataset from OpenML and no local CSV found.\n"
                "Provide webapp/boston.csv with columns: "
                f"{FEATURES + [TARGET]} (MEDV as target)."
            )
        return pd.read_csv(csv_path)

def main():
    df = load_boston_frame()
    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in dataset: {missing}")

    X = df[FEATURES]
    y = df[TARGET]

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", GradientBoostingRegressor(random_state=0)),
    ])
    pipe.fit(X, y)

    model_path = Path(__file__).with_name("boston_housing_prediction.joblib")
    joblib.dump(pipe, model_path)
    print(f"Saved model to {model_path.resolve()}")

if __name__ == "__main__":
    main()