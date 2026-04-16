"""app.py — Streamlit dashboard for the F1 win-probability model.

Run with:
    streamlit run app.py
"""
from __future__ import annotations
import sys
from pathlib import Path

# Ensure src/ is importable when running from the project root
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import streamlit as st
import joblib

from src.config import DATA_INTERIM, MODELS_DIR, REPORTS_DIR, KAGGLE_DATA_MAX_YEAR, REG_ERA_NAMES
from src.live_season import save_live_supplement, get_live_supplement_status

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="F1 Win Probability",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Helpers
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
    "Sauber":           "#52E252",
    "Kick Sauber":      "#52E252",
}


def _team_colour(team: str) -> str:
    for key, colour in TEAM_COLOURS.items():
        if key.lower() in str(team).lower():
            return colour
    return "#888888"


@st.cache_data(show_spinner=False)
def load_features(mode: str = "postqual") -> pd.DataFrame | None:
    path = DATA_INTERIM / f"features_{mode}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return None


@st.cache_resource(show_spinner=False)
def load_model(mode: str = "postqual"):
    path = MODELS_DIR / f"model_{mode}.pkl"
    if path.exists():
        return joblib.load(path)
    return None


@st.cache_data(show_spinner=False)
def load_metrics(mode: str = "postqual") -> pd.DataFrame | None:
    path = MODELS_DIR / f"metrics_{mode}.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


def _driver_label(row: pd.Series) -> str:
    code = str(row.get("code", "")).strip()
    fname = str(row.get("forename", "")).strip()
    lname = str(row.get("surname", "")).strip()
    if code and code != "nan":
        return code
    return f"{fname[0]}. {lname}" if fname and fname != "nan" else lname


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("🏎️ F1 Win Probability")
    mode = st.selectbox("Prediction mode", ["postqual", "prequal"],
                        help="postqual uses qualifying positions; prequal uses only rolling stats")

    st.divider()

    # ------------------------------------------------------------------
    # Live season data panel
    # ------------------------------------------------------------------
    import datetime
    current_year = datetime.date.today().year
    st.subheader("Live Season Data")

    supplement_status = get_live_supplement_status()
    fetched_years = {s["year"]: s["races"] for s in supplement_status}

    if fetched_years:
        for yr, rc in sorted(fetched_years.items()):
            st.caption(f"✅  {yr}: {rc} races cached")
    else:
        st.caption("No live supplements cached yet.")

    refresh_year = st.number_input(
        "Season to refresh", min_value=KAGGLE_DATA_MAX_YEAR + 1,
        max_value=current_year + 1, value=current_year, step=1,
    )

    if st.button("🔄 Fetch / Refresh via FastF1", use_container_width=True):
        with st.spinner(f"Fetching {refresh_year} results from FastF1…"):
            try:
                path = save_live_supplement(int(refresh_year))
                st.success(f"Saved: {path.name}")
                st.cache_data.clear()   # force feature/metrics reload
                st.cache_resource.clear()
            except Exception as exc:
                st.error(f"Fetch failed: {exc}")

    st.caption("After refreshing, re-run build_features + train to update the model.")

    st.divider()

    # 2026 regulation-change notice
    if current_year >= 2026:
        st.info(
            "**2026 regulations are active.**  \n"
            "New power unit architecture + active aerodynamics (era 7).  \n"
            "The model treats 2026 as an unseen regulation era — predictions "
            "rely more heavily on Elo and rolling stats than era-specific patterns.",
            icon="⚠️",
        )

    st.divider()
    st.caption("Powered by LightGBM + isotonic calibration")
    st.caption(f"Kaggle data: 1950–{KAGGLE_DATA_MAX_YEAR} · Live: FastF1")

# ---------------------------------------------------------------------------
# Load state
# ---------------------------------------------------------------------------
features_df = load_features(mode)
bundle = load_model(mode)
metrics_df = load_metrics(mode)

model_ready = bundle is not None
features_ready = features_df is not None

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_race, tab_history, tab_elo, tab_metrics = st.tabs([
    "🏁 Race Prediction",
    "📈 Historical Predictions",
    "⚡ Elo Ratings",
    "📊 Model Performance",
])

# ============================================================
# TAB 1 — Next race / upcoming prediction
# ============================================================
with tab_race:
    st.header("Upcoming Race Win Probabilities")

    # Regulation-change warning when the current season is beyond model training data
    import datetime as _dt
    _cy = _dt.date.today().year
    if _cy > KAGGLE_DATA_MAX_YEAR:
        from src.config import year_to_reg_era, REG_ERA_NAMES
        current_era = year_to_reg_era(_cy)
        era_name = REG_ERA_NAMES.get(current_era, "unknown")
        st.warning(
            f"**{_cy} is beyond the Kaggle training dataset ({KAGGLE_DATA_MAX_YEAR}).**  \n"
            f"Regulation era: **{era_name}** (era {current_era}) — this value is outside the "
            f"model's training distribution.  \n"
            f"Predictions rely on Elo ratings and rolling stats carried forward from "
            f"{KAGGLE_DATA_MAX_YEAR}. Refresh live data in the sidebar and retrain for "
            f"best accuracy.",
        )

    if not model_ready:
        st.warning(
            "No trained model found. Run the pipeline first:\n"
            "```bash\n"
            "python -m src.build_features --mode postqual\n"
            "python -m src.train --mode postqual\n"
            "```"
        )
    else:
        col_left, col_right = st.columns([1, 2])

        with col_left:
            use_upcoming = st.toggle("Predict NEXT race (live FastF1)", value=True)
            if not use_upcoming and features_ready:
                all_races = (
                    features_df[["year", "round", "raceId"]]
                    .drop_duplicates()
                    .sort_values(["year", "round"], ascending=False)
                )
                all_races["label"] = (
                    all_races["year"].astype(str) + " — Round "
                    + all_races["round"].astype(str)
                    + "  (raceId " + all_races["raceId"].astype(str) + ")"
                )
                selected = st.selectbox("Select past race", all_races["label"].tolist())
                race_id = int(all_races.loc[all_races["label"] == selected, "raceId"].iloc[0])
            st.divider()

        if use_upcoming:
            with st.spinner("Fetching next event from FastF1…"):
                try:
                    from src.upcoming import predict_next
                    result_df = predict_next(mode=mode)
                    race_label = f"{result_df['year'].iloc[0]} Round {result_df['round'].iloc[0]}"
                except Exception as exc:
                    st.error(f"FastF1 fetch failed: {exc}")
                    result_df = None
                    race_label = "Unknown"
        else:
            with st.spinner("Scoring past race…"):
                try:
                    scaler = bundle["scaler"]
                    model = bundle["model"]
                    feat_cols = bundle["feature_cols"]
                    df_race = features_df[features_df["raceId"] == race_id].copy()
                    missing = [c for c in feat_cols if c not in df_race.columns]
                    for c in missing:
                        df_race[c] = np.nan
                    Xs = scaler.transform(df_race[feat_cols])
                    proba = model.predict_proba(Xs)[:, 1]
                    result_df = df_race[[
                        c for c in ["raceId", "year", "round", "driverId", "constructorId",
                                    "code", "forename", "surname", "constructor", "grid_pos"]
                        if c in df_race.columns
                    ]].copy()
                    result_df["win_proba"] = proba
                    result_df["win_pct"] = (proba * 100).round(1)
                    result_df = result_df.sort_values("win_pct", ascending=False).reset_index(drop=True)
                    race_label = f"{result_df['year'].iloc[0]} Round {result_df['round'].iloc[0]}"
                except Exception as exc:
                    st.error(f"Prediction failed: {exc}")
                    result_df = None
                    race_label = "Unknown"

        if result_df is not None and not result_df.empty:
            st.subheader(f"Race: {race_label}")

            # Win probability bar chart
            from src.visualize import plot_race_card
            if "win_pct" not in result_df.columns and "win_proba" in result_df.columns:
                result_df["win_pct"] = (result_df["win_proba"] * 100).round(1)

            fig = plot_race_card(result_df, title=f"{race_label} — Win Probability")
            if fig is not None:
                st.pyplot(fig)

            # Data table
            with st.expander("Full data table"):
                disp_cols = [c for c in [
                    "code", "forename", "surname", "constructor",
                    "grid_pos", "win_pct",
                ] if c in result_df.columns]
                st.dataframe(
                    result_df[disp_cols].rename(columns={"win_pct": "Win %"}),
                    use_container_width=True,
                )

# ============================================================
# TAB 2 — Historical prediction browser
# ============================================================
with tab_history:
    st.header("Historical Win Probability Browser")

    if not features_ready or not model_ready:
        st.info("Build features and train the model to use this tab.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            year_sel = st.selectbox(
                "Season",
                sorted(features_df["year"].unique(), reverse=True),
            )
        with col2:
            rounds_in_year = sorted(
                features_df.loc[features_df["year"] == year_sel, "round"].unique()
            )
            round_sel = st.selectbox("Round", rounds_in_year)

        rid_mask = (features_df["year"] == year_sel) & (features_df["round"] == round_sel)
        df_race = features_df[rid_mask].copy()

        if df_race.empty:
            st.warning("No data for this race.")
        else:
            scaler = bundle["scaler"]
            model = bundle["model"]
            feat_cols = bundle["feature_cols"]
            missing = [c for c in feat_cols if c not in df_race.columns]
            for c in missing:
                df_race[c] = np.nan
            Xs = scaler.transform(df_race[feat_cols])
            proba = model.predict_proba(Xs)[:, 1]
            df_race["win_pct"] = (proba * 100).round(1)
            df_race = df_race.sort_values("win_pct", ascending=False).reset_index(drop=True)

            from src.visualize import plot_race_card
            fig = plot_race_card(df_race, title=f"{year_sel} Round {round_sel} — Win Probability")
            if fig is not None:
                st.pyplot(fig)

            if "is_win" in df_race.columns:
                winner_row = df_race[df_race["is_win"] == 1]
                if not winner_row.empty:
                    winner = _driver_label(winner_row.iloc[0])
                    winner_rank = int(df_race[df_race["is_win"] == 1].index[0]) + 1
                    st.info(
                        f"Actual winner: **{winner}**  (model ranked them #{winner_rank})"
                    )

# ============================================================
# TAB 3 — Elo ratings
# ============================================================
with tab_elo:
    st.header("Elo Rating History")

    if not features_ready:
        st.info("Build features to view Elo history.")
    else:
        col1, col2, col3 = st.columns(3)
        with col1:
            elo_type = st.radio("Rating type", ["Driver", "Constructor"])
        with col2:
            top_n_elo = st.slider("Top N", min_value=3, max_value=20, value=10)
        with col3:
            year_from = st.slider(
                "From year", min_value=int(features_df["year"].min()),
                max_value=int(features_df["year"].max()) - 1,
                value=max(2010, int(features_df["year"].min())),
            )

        from src.visualize import plot_elo_history, plot_constructor_elo
        if elo_type == "Driver":
            fig = plot_elo_history(features_df, top_n=top_n_elo, year_min=year_from)
        else:
            fig = plot_constructor_elo(features_df, top_n=top_n_elo, year_min=year_from)
        if fig is not None:
            st.pyplot(fig)

        # Current standings table
        st.subheader(f"Current {'Driver' if elo_type == 'Driver' else 'Constructor'} Elo Standings")
        latest_year = int(features_df["year"].max())
        elo_col = "driverId_elo_pre" if elo_type == "Driver" else "constructorId_elo_pre"
        id_col = "driverId" if elo_type == "Driver" else "constructorId"

        latest = (
            features_df[features_df["year"] == latest_year]
            .sort_values(["year", "round"])
            .drop_duplicates([id_col], keep="last")
            .sort_values(elo_col, ascending=False)
        )

        if elo_type == "Driver":
            disp = latest[
                [c for c in ["code", "forename", "surname", "constructor", elo_col] if c in latest.columns]
            ].head(20).reset_index(drop=True)
            disp.index += 1
            disp = disp.rename(columns={elo_col: "Elo"})
        else:
            disp = latest[
                [c for c in ["constructor", elo_col] if c in latest.columns]
            ].drop_duplicates("constructor").head(15).reset_index(drop=True)
            disp.index += 1
            disp = disp.rename(columns={elo_col: "Elo"})

        disp["Elo"] = disp["Elo"].round(1)
        st.dataframe(disp, use_container_width=True)

# ============================================================
# TAB 4 — Model performance
# ============================================================
with tab_metrics:
    st.header("Model Performance — Walk-forward CV")

    if metrics_df is None:
        st.warning(
            "No metrics found. Train the model to generate metrics:\n"
            "```bash\npython -m src.train --mode postqual\n```"
        )
    else:
        mean_top1 = metrics_df["top1"].mean()
        mean_ll = metrics_df["log_loss"].mean()
        mean_brier = metrics_df["brier"].mean()
        n_folds = len(metrics_df)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Avg Top-1 Accuracy", f"{mean_top1:.1%}")
        m2.metric("Avg Log Loss", f"{mean_ll:.4f}")
        m3.metric("Avg Brier Score", f"{mean_brier:.4f}")
        m4.metric("CV Folds", str(n_folds))

        from src.visualize import plot_metrics
        fig = plot_metrics(metrics_df, mode=mode)
        if fig is not None:
            st.pyplot(fig)

        with st.expander("Raw metrics table"):
            st.dataframe(metrics_df.round(4), use_container_width=True)

    # Feature importance (from saved PNG or live)
    st.divider()
    st.subheader("Feature Importance")
    fi_png = REPORTS_DIR / f"feature_importance_{mode}.png"
    if fi_png.exists():
        st.image(str(fi_png))
    elif bundle is not None:
        from src.visualize import plot_feature_importance
        feat_cols = bundle["feature_cols"]
        fig = plot_feature_importance(bundle["model"], feat_cols, mode=mode)
        if fig is not None:
            st.pyplot(fig)
    else:
        st.info("Train the model to see feature importance.")

    # Calibration curve
    st.divider()
    st.subheader("Calibration Curve")
    cal_png = REPORTS_DIR / f"calibration_{mode}.png"
    if cal_png.exists():
        st.image(str(cal_png))
    else:
        st.info("Calibration plot is generated automatically during training.")
