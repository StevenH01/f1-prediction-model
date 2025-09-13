from __future__ import annotations
import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import log_loss, brier_score_loss, accuracy_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier
import matplotlib.pyplot as plt

from .config import DATA_INTERIM, MODELS_DIR, REPORTS_DIR, RANDOM_SEED

METRIC_FIGSIZE = (6,4)

def _load_features(mode: str) -> pd.DataFrame:
    path = DATA_INTERIM / f"features_{mode}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Features not found: {path}. Run build_features first.")
    return pd.read_parquet(path)

def _select_xy(df: pd.DataFrame):
    drop_cols = {"is_win","driverRef","code","forename","surname","constructor"}
    X = df.drop(columns=[c for c in drop_cols if c in df.columns]).copy()
    y = df["is_win"].astype(int).values
    groups = df["year"].values  # year-based grouping / chronological split helper
    # Keep IDs for later reference
    meta_cols = [c for c in ["driverId","constructorId","raceId","year","round"] if c in df.columns]
    meta = df[meta_cols].copy()
    return X, y, groups, meta

def _time_aware_folds(years: np.ndarray, n_splits: int = 5):
    """Yield train/test boolean masks for incremental yearly splits."""
    unique_years = sorted(np.unique(years))
    if len(unique_years) < n_splits + 1:
        n_splits = max(1, len(unique_years) - 1)
    for i in range(n_splits):
        # train up to Y_i (inclusive), test on Y_{i+1}
        train_years = set(unique_years[: i+1 ])
        test_year = unique_years[i+1] if i+1 < len(unique_years) else None
        if test_year is None: 
            break
        train_mask = np.isin(years, list(train_years))
        test_mask = (years == test_year)
        yield train_mask, test_mask, test_year

def train_and_evaluate(mode: str = "prequal"):
    df = _load_features(mode)
    X, y, years, meta = _select_xy(df)

    # Separate features from IDs/targets
    id_cols = {"driverId","constructorId","raceId","year","round"}
    feature_cols = [c for c in X.columns if c not in id_cols]

    # Basic scaler for robustness (LightGBM is robust but scaling aids calibration)
    scaler = StandardScaler(with_mean=False)  # sparse-safe
    X_feat = scaler.fit_transform(X[feature_cols])

    best_model = None
    all_metrics = []
    rng = np.random.RandomState(RANDOM_SEED)

    for train_mask, test_mask, test_year in _time_aware_folds(years, n_splits=5):
        X_tr = X_feat[train_mask]
        y_tr = y[train_mask]
        X_te = X_feat[test_mask]
        y_te = y[test_mask]

        # Base classifier
        clf = LGBMClassifier(
            objective="binary",
            n_estimators=600,
            learning_rate=0.03,
            num_leaves=63,
            subsample=0.8,
            colsample_bytree=0.9,
            random_state=RANDOM_SEED,
            class_weight="balanced",
        )

        # Calibrate on a slice of training set for stability
        calibrator = CalibratedClassifierCV(estimator=clf, method="isotonic", cv=3)
        calibrator.fit(X_tr, y_tr)

        proba = calibrator.predict_proba(X_te)[:,1]
        preds = (proba == proba) & (proba >= 0.5)

        # Metrics
        ll = log_loss(y_te, proba, labels=[0,1])
        bs = brier_score_loss(y_te, proba)
        # Evaluate "picked the winner correctly per race"
        fold_meta = meta[test_mask].copy()
        fold_meta["proba"] = proba
        fold_meta["y"] = y_te
        # top-1 accuracy: for each race, did the highest proba row have y=1?
        top1_hits, race_count = 0, 0
        for rid, grp in fold_meta.groupby("raceId"):
            if len(grp) == 0: 
                continue
            race_count += 1
            best_idx = grp["proba"].idxmax()
            top1_hits += int(grp.loc[best_idx, "y"] == 1)
        top1 = top1_hits / race_count if race_count else np.nan

        all_metrics.append({"year": int(test_year), "log_loss": ll, "brier": bs, "top1": top1})
        print(f"[{mode}] Test {test_year}: logloss={ll:.4f}  brier={bs:.4f}  top1={top1:.3f}")

        best_model = calibrator  # last fold's calibrated model as representative

    # Save model & scaler
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"scaler": scaler, "model": best_model, "feature_cols": feature_cols}, MODELS_DIR / f"model_{mode}.pkl")

    # Save metrics CSV and a simple plot
    metrics_df = pd.DataFrame(all_metrics)
    metrics_path = MODELS_DIR / f"metrics_{mode}.csv"
    metrics_df.to_csv(metrics_path, index=False)

    # Plot log loss over years
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6,4))
    plt.plot(metrics_df["year"], metrics_df["log_loss"], marker="o")
    plt.title(f"{mode} - Log Loss by Test Year")
    plt.xlabel("Year"); plt.ylabel("Log Loss"); plt.grid(True, alpha=0.3)
    fig_path = REPORTS_DIR / f"logloss_{mode}.png"
    plt.tight_layout(); plt.savefig(fig_path, dpi=160)
    plt.close()

    return metrics_path, fig_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["prequal","postqual"], default="prequal")
    args = parser.parse_args()
    train_and_evaluate(mode=args.mode)
