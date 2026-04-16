"""visualize.py — plotting utilities for F1 win-probability predictions.

All functions accept a save_path kwarg. When provided the figure is saved
to that path; when None the figure is returned for the caller to display
(e.g. in Streamlit via st.pyplot).
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Team colour palette (approximate 2025 livery colours)
# ---------------------------------------------------------------------------
TEAM_COLOURS: dict[str, str] = {
    "Red Bull":         "#3671C6",
    "Red Bull Racing":  "#3671C6",
    "Ferrari":          "#E8002D",
    "Mercedes":         "#27F4D2",
    "McLaren":          "#FF8000",
    "Aston Martin":     "#229971",
    "Alpine":           "#FF87BC",
    "Williams":         "#64C4FF",
    "RB":               "#6692FF",
    "Visa Cash App RB": "#6692FF",
    "Haas":             "#B6BABD",
    "Haas F1 Team":     "#B6BABD",
    "Sauber":           "#52E252",
    "Kick Sauber":      "#52E252",
}
DEFAULT_COLOUR = "#888888"


def _team_colour(team: str) -> str:
    for key, colour in TEAM_COLOURS.items():
        if key.lower() in str(team).lower():
            return colour
    return DEFAULT_COLOUR


def _save_or_return(fig: plt.Figure, save_path: Optional[Path | str]):
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        return None
    return fig


# ---------------------------------------------------------------------------
# 1. Race Card — horizontal bar chart of win probabilities
# ---------------------------------------------------------------------------
def plot_race_card(
    df: pd.DataFrame,
    title: str = "Win Probability",
    top_n: int = 20,
    save_path: Optional[Path | str] = None,
) -> Optional[plt.Figure]:
    """Horizontal bar chart showing each driver's win probability.

    Parameters
    ----------
    df : DataFrame with columns  win_pct (0-100), code, forename, surname,
         constructor (team name), and optionally grid_pos.
    title : Chart title.
    top_n : Maximum number of drivers to show.
    save_path : File path to save to, or None to return the Figure.
    """
    df = df.sort_values("win_pct", ascending=False).head(top_n).reset_index(drop=True)

    # Build display labels
    def _driver_label(row) -> str:
        code = str(row.get("code", "")).strip()
        fname = str(row.get("forename", "")).strip()
        lname = str(row.get("surname", "")).strip()
        if code:
            return code
        return f"{fname[0]}. {lname}" if fname else lname

    labels = df.apply(_driver_label, axis=1).tolist()
    values = df["win_pct"].tolist()
    teams = df.get("constructor", pd.Series([""] * len(df))).fillna("").tolist()
    colours = [_team_colour(t) for t in teams]

    fig, ax = plt.subplots(figsize=(9, max(4, len(labels) * 0.42)))
    bars = ax.barh(range(len(labels)), values, color=colours, edgecolor="white", linewidth=0.4)

    # Value labels on bars
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}%", va="center", ha="left", fontsize=8.5,
        )

    # Grid position badge (optional)
    if "grid_pos" in df.columns:
        for i, (_, row) in enumerate(df.iterrows()):
            gp = row.get("grid_pos")
            if pd.notna(gp):
                ax.text(
                    -0.5, i, f"P{int(gp)}", va="center", ha="right",
                    fontsize=7.5, color="#555555",
                )

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Win probability (%)")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=100))
    ax.set_xlim(0, max(values) * 1.25)
    ax.grid(axis="x", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return _save_or_return(fig, save_path)


# ---------------------------------------------------------------------------
# 2. Elo History — line chart of top-N drivers over time
# ---------------------------------------------------------------------------
def plot_elo_history(
    features_df: pd.DataFrame,
    top_n: int = 10,
    year_min: int = 2010,
    save_path: Optional[Path | str] = None,
) -> Optional[plt.Figure]:
    """Line chart of driver Elo ratings over time.

    Parameters
    ----------
    features_df : The features parquet (postqual or prequal).
    top_n : Number of highest-rated drivers to highlight.
    year_min : Restrict display to this year and later.
    save_path : File path or None.
    """
    df = features_df.copy()
    df = df[df["year"] >= year_min]

    # One row per (year, round, driver) — take first occurrence
    df = df.sort_values(["year", "round"]).drop_duplicates(["raceId", "driverId"])

    if "driverId_elo_pre" not in df.columns or df.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No Elo data available", ha="center", va="center")
        return _save_or_return(fig, save_path)

    # Identify top-N drivers by peak Elo
    peak = df.groupby("driverId")["driverId_elo_pre"].max()
    top_ids = peak.nlargest(top_n).index.tolist()

    # Build a display name map
    name_map: dict[int, str] = {}
    for did in top_ids:
        rows = df[df["driverId"] == did]
        code = rows["code"].dropna().iloc[0] if "code" in rows.columns and not rows["code"].dropna().empty else ""
        surname = rows["surname"].dropna().iloc[0] if "surname" in rows.columns and not rows["surname"].dropna().empty else str(did)
        name_map[did] = code if code else str(surname)

    fig, ax = plt.subplots(figsize=(12, 6))
    for did in top_ids:
        sub = df[df["driverId"] == did].sort_values(["year", "round"])
        # Create a continuous x-axis: year + round/25
        x = sub["year"] + sub["round"] / 25.0
        y = sub["driverId_elo_pre"]
        ax.plot(x, y, linewidth=1.4, label=name_map.get(did, str(did)), alpha=0.85)

    ax.set_xlabel("Season")
    ax.set_ylabel("Elo Rating")
    ax.set_title(f"Driver Elo History (top {top_n}, {year_min}–present)")
    ax.legend(fontsize=8, ncol=2, loc="upper left")
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return _save_or_return(fig, save_path)


# ---------------------------------------------------------------------------
# 3. Calibration Curve
# ---------------------------------------------------------------------------
def plot_calibration(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mode: str = "",
    n_bins: int = 10,
    save_path: Optional[Path | str] = None,
) -> Optional[plt.Figure]:
    """Reliability diagram: predicted probability vs. fraction of true wins."""
    from sklearn.calibration import calibration_curve

    prob_true, prob_pred = calibration_curve(y_true, y_pred, n_bins=n_bins, strategy="uniform")

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(prob_pred, prob_true, marker="o", label="Model", color="#e10600", linewidth=2)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    ax.fill_between(prob_pred, prob_true, prob_pred,
                    where=prob_true > prob_pred, alpha=0.08, color="green", label="Under-confident")
    ax.fill_between(prob_pred, prob_true, prob_pred,
                    where=prob_true < prob_pred, alpha=0.08, color="red", label="Over-confident")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives (actual win rate)")
    ax.set_title(f"Calibration Curve{' — ' + mode if mode else ''}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return _save_or_return(fig, save_path)


# ---------------------------------------------------------------------------
# 4. Feature Importance
# ---------------------------------------------------------------------------
def plot_feature_importance(
    model,
    feature_cols: list[str],
    top_n: int = 20,
    mode: str = "",
    save_path: Optional[Path | str] = None,
) -> Optional[plt.Figure]:
    """Bar chart of mean LightGBM feature importances across calibration folds."""
    importances = []
    for est in model.calibrated_classifiers_:
        base = est.estimator
        if hasattr(base, "feature_importances_"):
            importances.append(base.feature_importances_)
    if not importances:
        return None

    imp = np.mean(importances, axis=0)
    std = np.std(importances, axis=0)
    order = np.argsort(imp)[::-1][:top_n]

    names = [feature_cols[i] for i in reversed(order)]
    vals = [imp[i] for i in reversed(order)]
    errs = [std[i] for i in reversed(order)]

    fig, ax = plt.subplots(figsize=(9, max(5, top_n * 0.38)))
    ax.barh(names, vals, xerr=errs, color="#e10600", ecolor="#888888",
            capsize=3, edgecolor="white", linewidth=0.4)
    ax.set_xlabel("Importance (mean gain ± std)")
    ax.set_title(f"Feature Importance{' — ' + mode if mode else ''}")
    ax.grid(axis="x", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return _save_or_return(fig, save_path)


# ---------------------------------------------------------------------------
# 5. Metrics over time (log loss, Brier, top-1)
# ---------------------------------------------------------------------------
def plot_metrics(
    metrics_df: pd.DataFrame,
    mode: str = "",
    save_path: Optional[Path | str] = None,
) -> Optional[plt.Figure]:
    """3-panel metrics plot: log loss, Brier score, top-1 accuracy."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].plot(metrics_df["year"], metrics_df["log_loss"], marker="o", color="#e10600")
    axes[0].set_title("Log Loss"); axes[0].set_xlabel("Year"); axes[0].grid(alpha=0.25)

    axes[1].plot(metrics_df["year"], metrics_df["brier"], marker="o", color="#1f77b4")
    axes[1].set_title("Brier Score"); axes[1].set_xlabel("Year"); axes[1].grid(alpha=0.25)

    axes[2].plot(metrics_df["year"], metrics_df["top1"], marker="o", color="#2ca02c")
    axes[2].axhline(
        metrics_df["top1"].mean(), linestyle="--", color="gray", alpha=0.7,
        label=f"Mean = {metrics_df['top1'].mean():.3f}",
    )
    axes[2].set_title("Top-1 Accuracy"); axes[2].set_xlabel("Year"); axes[2].grid(alpha=0.25)
    axes[2].legend(fontsize=8)

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"Walk-forward CV{' — ' + mode if mode else ''}", fontsize=12)
    fig.tight_layout()
    return _save_or_return(fig, save_path)


# ---------------------------------------------------------------------------
# 6. Constructor Elo History
# ---------------------------------------------------------------------------
def plot_constructor_elo(
    features_df: pd.DataFrame,
    top_n: int = 10,
    year_min: int = 2010,
    save_path: Optional[Path | str] = None,
) -> Optional[plt.Figure]:
    """Line chart of constructor Elo ratings over time."""
    df = features_df.copy()
    df = df[df["year"] >= year_min]
    df = df.sort_values(["year", "round"]).drop_duplicates(["raceId", "constructorId"])

    if "constructorId_elo_pre" not in df.columns or df.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No constructor Elo data", ha="center", va="center")
        return _save_or_return(fig, save_path)

    peak = df.groupby("constructorId")["constructorId_elo_pre"].max()
    top_ids = peak.nlargest(top_n).index.tolist()

    name_map: dict[int, str] = {}
    for cid in top_ids:
        rows = df[df["constructorId"] == cid]
        cname = rows["constructor"].dropna().iloc[0] if "constructor" in rows.columns and not rows["constructor"].dropna().empty else str(cid)
        name_map[cid] = str(cname)

    fig, ax = plt.subplots(figsize=(12, 6))
    for cid in top_ids:
        sub = df[df["constructorId"] == cid].sort_values(["year", "round"])
        x = sub["year"] + sub["round"] / 25.0
        y = sub["constructorId_elo_pre"]
        colour = _team_colour(name_map.get(cid, ""))
        ax.plot(x, y, linewidth=1.6, label=name_map.get(cid, str(cid)),
                color=colour, alpha=0.85)

    ax.set_xlabel("Season")
    ax.set_ylabel("Elo Rating")
    ax.set_title(f"Constructor Elo History (top {top_n}, {year_min}–present)")
    ax.legend(fontsize=8, ncol=2, loc="upper left")
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return _save_or_return(fig, save_path)
