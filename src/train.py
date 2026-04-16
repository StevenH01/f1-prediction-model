from __future__ import annotations
import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import DATA_INTERIM, MODELS_DIR, REPORTS_DIR, RANDOM_SEED

# ---------------------------------------------------------------------------
# Minimum year for walk-forward validation (ignore ancient pre-turbo data
# when evaluating model quality; still trains on everything ≥ year_min).
# ---------------------------------------------------------------------------
WALKFORWARD_START = 2005
N_FOLDS = 20  # maximum test years; capped by available data


def _load_features(mode: str) -> pd.DataFrame:
    path = DATA_INTERIM / f"features_{mode}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Features not found: {path}. Run build_features first."
        )
    return pd.read_parquet(path)


def _select_xy(df: pd.DataFrame):
    drop_cols = {"is_win", "driverRef", "code", "forename", "surname", "constructor"}
    X = df.drop(columns=[c for c in drop_cols if c in df.columns]).copy()
    y = df["is_win"].astype(int).values
    id_cols = {"driverId", "constructorId", "raceId", "year", "round"}
    feature_cols = [c for c in X.columns if c not in id_cols]
    meta_cols = [c for c in ["driverId", "constructorId", "raceId", "year", "round"] if c in df.columns]
    meta = df[meta_cols].copy()
    return X, y, feature_cols, meta


def _time_aware_folds(years_arr: np.ndarray, start_year: int = WALKFORWARD_START, n_splits: int = N_FOLDS):
    """Walk-forward yearly folds.

    Trains on all data up to year Y, tests on year Y+1.
    Only yields folds where the test year >= start_year, so early pre-turbo
    eras don't dominate the validation signal.
    """
    unique_years = sorted(np.unique(years_arr))
    folds_yielded = 0
    for i in range(len(unique_years) - 1):
        test_year = unique_years[i + 1]
        if test_year < start_year:
            continue
        train_years = set(unique_years[: i + 1])
        train_mask = np.isin(years_arr, list(train_years))
        test_mask = years_arr == test_year
        yield train_mask, test_mask, test_year
        folds_yielded += 1
        if folds_yielded >= n_splits:
            break


def _default_lgbm_params() -> dict:
    return dict(
        objective="binary",
        n_estimators=600,
        learning_rate=0.03,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.9,
        min_child_samples=20,
        reg_alpha=0.0,
        reg_lambda=0.0,
        random_state=RANDOM_SEED,
        class_weight="balanced",
        verbose=-1,
    )


def _tune_hyperparams(X_tune: np.ndarray, y_tune: np.ndarray, n_trials: int = 40) -> dict:
    """Use Optuna to search LightGBM hyperparameters on a held-out validation slice."""
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        print("[train] optuna not installed — skipping tuning, using defaults.")
        return _default_lgbm_params()

    split = int(len(X_tune) * 0.8)
    X_tr, X_val = X_tune[:split], X_tune[split:]
    y_tr, y_val = y_tune[:split], y_tune[split:]

    def objective(trial: optuna.Trial) -> float:
        params = dict(
            objective="binary",
            n_estimators=trial.suggest_int("n_estimators", 200, 1000),
            learning_rate=trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
            num_leaves=trial.suggest_int("num_leaves", 15, 127),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            min_child_samples=trial.suggest_int("min_child_samples", 5, 60),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            random_state=RANDOM_SEED,
            class_weight="balanced",
            verbose=-1,
        )
        clf = LGBMClassifier(**params)
        clf.fit(X_tr, y_tr)
        proba = clf.predict_proba(X_val)[:, 1]
        return log_loss(y_val, proba, labels=[0, 1])

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = study.best_params
    best.update(objective="binary", random_state=RANDOM_SEED, class_weight="balanced", verbose=-1)
    print(f"[train] Optuna best log-loss={study.best_value:.4f}  params={best}")
    return best


def _plot_feature_importance(model: CalibratedClassifierCV, feature_cols: list[str], mode: str):
    """Extract and plot LightGBM feature importances (mean across calibration folds)."""
    importances = []
    for est in model.calibrated_classifiers_:
        base = est.estimator
        if hasattr(base, "feature_importances_"):
            importances.append(base.feature_importances_)
    if not importances:
        return
    imp = np.mean(importances, axis=0)
    order = np.argsort(imp)[::-1][:20]  # top 20

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(
        [feature_cols[i] for i in reversed(order)],
        [imp[i] for i in reversed(order)],
        color="#e10600",
    )
    ax.set_xlabel("Importance (mean gain)")
    ax.set_title(f"Feature Importance — {mode}")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    path = REPORTS_DIR / f"feature_importance_{mode}.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"[train] Saved feature importance → {path}")


def _plot_calibration(all_y: list, all_proba: list, mode: str):
    """Reliability diagram across all test folds."""
    y_cat = np.concatenate(all_y)
    p_cat = np.concatenate(all_proba)
    prob_true, prob_pred = calibration_curve(y_cat, p_cat, n_bins=10, strategy="uniform")

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(prob_pred, prob_true, marker="o", label="Model", color="#e10600")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title(f"Calibration Curve — {mode}")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = REPORTS_DIR / f"calibration_{mode}.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"[train] Saved calibration curve → {path}")


def _plot_metrics(metrics_df: pd.DataFrame, mode: str):
    """3-panel plot: log-loss, Brier score, and top-1 accuracy by test year."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].plot(metrics_df["year"], metrics_df["log_loss"], marker="o", color="#e10600")
    axes[0].set_title("Log Loss by Test Year")
    axes[0].set_xlabel("Year"); axes[0].set_ylabel("Log Loss")
    axes[0].grid(alpha=0.3)

    axes[1].plot(metrics_df["year"], metrics_df["brier"], marker="o", color="#1f77b4")
    axes[1].set_title("Brier Score by Test Year")
    axes[1].set_xlabel("Year"); axes[1].set_ylabel("Brier Score")
    axes[1].grid(alpha=0.3)

    axes[2].plot(metrics_df["year"], metrics_df["top1"], marker="o", color="#2ca02c")
    axes[2].axhline(metrics_df["top1"].mean(), linestyle="--", color="gray", alpha=0.7,
                    label=f"Mean={metrics_df['top1'].mean():.3f}")
    axes[2].set_title("Top-1 Accuracy by Test Year")
    axes[2].set_xlabel("Year"); axes[2].set_ylabel("Top-1 Acc")
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.3)

    fig.suptitle(f"Walk-forward CV metrics — {mode}", fontsize=13)
    fig.tight_layout()
    path = REPORTS_DIR / f"metrics_{mode}.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"[train] Saved metrics plot → {path}")


def train_and_evaluate(
    mode: str = "prequal",
    year_min: int | None = None,
    year_max: int | None = None,
    years: str | None = None,
    tune: bool = False,
    n_tune_trials: int = 40,
):
    df = _load_features(mode)

    if years:
        year_list = [int(y) for y in str(years).replace(" ", "").split(",") if y]
        df = df[df["year"].isin(year_list)].copy()
    else:
        lo = int(year_min) if year_min is not None else int(df["year"].min())
        hi = int(year_max) if year_max is not None else int(df["year"].max())
        df = df[(df["year"] >= lo) & (df["year"] <= hi)].copy()

    X, y, feature_cols, meta = _select_xy(df)
    years_arr = df["year"].values

    # StandardScaler (sparse-safe; aids isotonic calibration)
    scaler = StandardScaler(with_mean=False)
    X_feat = scaler.fit_transform(X[feature_cols])

    # ------------------------------------------------------------------
    # Optional Optuna tuning — use first 80 % of chronological data
    # ------------------------------------------------------------------
    if tune:
        print(f"[train] Running Optuna hyperparameter search ({n_tune_trials} trials)…")
        lgbm_params = _tune_hyperparams(X_feat, y, n_trials=n_tune_trials)
    else:
        lgbm_params = _default_lgbm_params()

    # ------------------------------------------------------------------
    # Walk-forward cross-validation
    # ------------------------------------------------------------------
    best_model = None
    all_metrics: list[dict] = []
    all_y_test: list[np.ndarray] = []
    all_proba_test: list[np.ndarray] = []

    fold_gen = list(_time_aware_folds(years_arr, start_year=WALKFORWARD_START, n_splits=N_FOLDS))
    print(f"[train] {len(fold_gen)} walk-forward folds (test years ≥ {WALKFORWARD_START})")

    for train_mask, test_mask, test_year in fold_gen:
        X_tr, y_tr = X_feat[train_mask], y[train_mask]
        X_te, y_te = X_feat[test_mask], y[test_mask]

        clf = LGBMClassifier(**lgbm_params)
        calibrator = CalibratedClassifierCV(estimator=clf, method="isotonic", cv=3)
        calibrator.fit(X_tr, y_tr)

        proba = calibrator.predict_proba(X_te)[:, 1]
        ll = log_loss(y_te, proba, labels=[0, 1])
        bs = brier_score_loss(y_te, proba)

        # Top-1 accuracy: per race, did the highest-probability driver actually win?
        fold_meta = meta[test_mask].copy()
        fold_meta["proba"] = proba
        fold_meta["y"] = y_te
        top1_hits, race_count = 0, 0
        for _, grp in fold_meta.groupby("raceId"):
            if grp.empty:
                continue
            race_count += 1
            best_idx = grp["proba"].idxmax()
            top1_hits += int(grp.loc[best_idx, "y"] == 1)
        top1 = top1_hits / race_count if race_count else np.nan

        all_metrics.append({"year": int(test_year), "log_loss": ll, "brier": bs, "top1": top1})
        all_y_test.append(y_te)
        all_proba_test.append(proba)

        print(
            f"[{mode}] Test {test_year}: logloss={ll:.4f}  brier={bs:.4f}  top1={top1:.3f}  "
            f"(train n={train_mask.sum()}, test n={test_mask.sum()})"
        )
        best_model = calibrator  # retain the last fold's model

    if best_model is None:
        raise RuntimeError("No folds were generated. Check WALKFORWARD_START vs data years.")

    # ------------------------------------------------------------------
    # Persist model bundle
    # ------------------------------------------------------------------
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    bundle_path = MODELS_DIR / f"model_{mode}.pkl"
    joblib.dump(
        {"scaler": scaler, "model": best_model, "feature_cols": feature_cols},
        bundle_path,
    )
    print(f"[train] Saved model → {bundle_path}")

    # ------------------------------------------------------------------
    # Save metrics CSV
    # ------------------------------------------------------------------
    metrics_df = pd.DataFrame(all_metrics)
    metrics_path = MODELS_DIR / f"metrics_{mode}.csv"
    metrics_df.to_csv(metrics_path, index=False)

    # ------------------------------------------------------------------
    # Generate all plots
    # ------------------------------------------------------------------
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    _plot_metrics(metrics_df, mode)
    _plot_calibration(all_y_test, all_proba_test, mode)
    _plot_feature_importance(best_model, feature_cols, mode)

    return metrics_path, REPORTS_DIR


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["prequal", "postqual"], default="prequal")
    parser.add_argument("--year-min", type=int, default=None,
                        help="Exclude years before this from training data")
    parser.add_argument("--year-max", type=int, default=None)
    parser.add_argument("--tune", action="store_true",
                        help="Run Optuna hyperparameter search before training")
    parser.add_argument("--n-trials", type=int, default=40,
                        help="Number of Optuna trials (default 40)")
    args = parser.parse_args()
    train_and_evaluate(
        mode=args.mode,
        year_min=args.year_min,
        year_max=args.year_max,
        tune=args.tune,
        n_tune_trials=args.n_trials,
    )
