from __future__ import annotations
import pandas as pd
import numpy as np

def compute_multiplayer_elo(results_merged: pd.DataFrame,
                            init_rating: float = 1500.0,
                            k: float = 16.0,
                            by: str = "driverId") -> pd.DataFrame:
    """Approximate multi-competitor Elo from race finish orders.

    Parameters
    ----------
    results_merged : DataFrame
        Must contain columns: ['raceId','year','round', by, 'positionOrder'].
        One row per driver in a race.
    by : str
        Either 'driverId' or 'constructorId' for which to compute Elo.

    Returns
    -------
    DataFrame with columns: ['raceId', by, f'{by}_elo_pre'] providing the
    pre-race Elo prior to the race update, computed sequentially by (year, round).
    """
    required = {"raceId","year","round",by,"positionOrder"}
    missing = required - set(results_merged.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = results_merged.copy()
    df = df.sort_values(["year","round","raceId","positionOrder"], kind="mergesort")

    # Initial ratings
    entities = pd.unique(df[by])
    rating = {int(ent): init_rating for ent in entities if pd.notna(ent)}

    rows = []
    # Iterate race by race
    for (year, rnd, raceId), grp in df.groupby(["year","round","raceId"], sort=False):
        # Pre-race snapshot
        pre = {int(ent): rating.get(int(ent), init_rating) for ent in grp[by].astype("Int64").dropna().astype(int)}
        # Save pre-race ratings
        for ent in grp[by].dropna().astype(int).unique():
            rows.append({"raceId": int(raceId), by: int(ent), f"{by}_elo_pre": pre[int(ent)]})

        # Pairwise comparisons (winner = finished ahead)
        participants = grp[[by,"positionOrder"]].dropna().astype({"positionOrder": int, by: int})
        ids = participants[by].tolist()
        # For stability, compute expected score per entity vs field
        for i, ent_i in enumerate(ids):
            ri = rating.get(ent_i, init_rating)
            actual_wins = 0.0
            expected_wins = 0.0
            for j, ent_j in enumerate(ids):
                if ent_i == ent_j:
                    continue
                rj = rating.get(ent_j, init_rating)
                # Expected win prob vs opponent
                pij = 1.0 / (1.0 + 10 ** ((rj - ri)/400.0))
                expected_wins += pij
                # Actual: did i finish ahead of j?
                pos_i = int(participants.loc[participants[by]==ent_i, "positionOrder"].iloc[0])
                pos_j = int(participants.loc[participants[by]==ent_j, "positionOrder"].iloc[0])
                actual_wins += 1.0 if pos_i < pos_j else 0.0
            # Normalize by number of opponents
            if len(ids) > 1:
                s = actual_wins / (len(ids)-1)
                e = expected_wins / (len(ids)-1)
                rating[ent_i] = ri + k * (s - e)

    out = pd.DataFrame(rows).drop_duplicates(subset=["raceId", by], keep="last")
    return out
