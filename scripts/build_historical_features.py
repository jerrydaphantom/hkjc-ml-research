from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
INTERIM_DIR = DATA_DIR / "interim"

SOURCE_DB_PATH = DATA_DIR / "hkjc_races_v2.db"
OUTPUT_DB_PATH = DATA_DIR / "hkjc_features_v2.db"

SOURCE_MODELING_TABLE = "modeling_table_v2"

FEATURE_TABLE_MARKET_FREE = "model_features_v2_market_free"
FEATURE_TABLE_MARKET_AWARE = "model_features_v2_market_aware"
FEATURE_SUMMARY_TABLE = "model_features_v2_summary"

FEATURE_CSV_MARKET_FREE = INTERIM_DIR / "model_features_v2_market_free.csv"
FEATURE_CSV_MARKET_AWARE = INTERIM_DIR / "model_features_v2_market_aware.csv"
FEATURE_SUMMARY_CSV = INTERIM_DIR / "model_features_v2_summary.csv"

WRITE_SQLITE_TABLES = True
WRITE_CSV_FILES = True

# IMPORTANT:
# This first version uses chronological row order:
# race_date, racecourse, race_no.
# That means for jockey/trainer features, earlier races on the same day
# can count toward later races on that same day.
# This is acceptable for a first version, but if you later want a
# strict “pre-card only” feature set, we can amend it.


def to_safe_string(series: pd.Series, fill_value: str) -> pd.Series:
    s = series.astype("string")
    s = s.fillna(fill_value)
    s = s.str.strip()
    s = s.replace("", fill_value)
    return s


def previous_count(df: pd.DataFrame, group_keys: list[str]) -> pd.Series:
    return df.groupby(group_keys, sort=False).cumcount().astype("Int64")


def previous_cumsum(df: pd.DataFrame, value_col: str, group_keys: list[str]) -> pd.Series:
    out = df.groupby(group_keys, sort=False)[value_col].cumsum() - df[value_col]
    return out.astype("Int64")


def previous_shift(df: pd.DataFrame, value_col: str, group_keys: list[str]) -> pd.Series:
    return df.groupby(group_keys, sort=False)[value_col].shift(1)


def rolling_prior_mean(
    df: pd.DataFrame,
    value_col: str,
    group_keys: list[str],
    window: int,
) -> pd.Series:
    return (
        df.groupby(group_keys, sort=False)[value_col]
        .transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
    )


def safe_rate(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    out = numerator / denominator.replace(0, np.nan)
    return out.fillna(0.0).astype(float)


def build_features(base: pd.DataFrame) -> pd.DataFrame:
    df = base.copy()

    # -----------------------------
    # Basic cleanup / keys
    # -----------------------------
    df["race_date_dt"] = pd.to_datetime(df["race_date"], errors="coerce")

    df["racecourse"] = to_safe_string(df["racecourse"], "UNKNOWN")
    df["jockey"] = to_safe_string(df["jockey"], "UNKNOWN")
    df["trainer"] = to_safe_string(df["trainer"], "UNKNOWN")
    df["horse"] = to_safe_string(df["horse"], "UNKNOWN_HORSE")

    if "horse_code" in df.columns:
        df["horse_key"] = to_safe_string(df["horse_code"], "NO_CODE")
        no_code_mask = df["horse_key"].eq("NO_CODE")
        df.loc[no_code_mask, "horse_key"] = df.loc[no_code_mask, "horse"]
    else:
        df["horse_key"] = df["horse"]

    if "race_distance_m" in df.columns:
        df["race_distance_key"] = df["race_distance_m"].astype("Int64").astype("string").fillna("UNKNOWN_DISTANCE")
    else:
        df["race_distance_key"] = "UNKNOWN_DISTANCE"

    if "race_surface" in df.columns:
        df["race_surface_key"] = to_safe_string(df["race_surface"], "UNKNOWN_SURFACE")
    else:
        df["race_surface_key"] = "UNKNOWN_SURFACE"

    # Ensure numeric types
    numeric_cols = [
        "race_no",
        "act_wt",
        "declar_horse_wt",
        "dr",
        "win_odds",
        "target_win",
        "target_place_top3",
        "outcome_place_num",
        "outcome_dead_heat",
        "outcome_finish_time_seconds",
        "race_distance_m",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Sort in chronological race order
    df = df.sort_values(
        ["race_date_dt", "racecourse", "race_no", "horse"],
        kind="mergesort",
    ).reset_index(drop=True)

    # -----------------------------
    # Current-race context features
    # -----------------------------
    df["field_size"] = (
        df.groupby(["race_date", "racecourse", "race_no"], sort=False)["horse"]
        .transform("count")
        .astype("Int64")
    )

    df["draw_norm_by_field"] = df["dr"] / df["field_size"]
    df["weight_diff_from_declared"] = df["act_wt"] - df["declar_horse_wt"]

    df["log_win_odds"] = np.where(
        df["win_odds"].notna() & (df["win_odds"] > 0),
        np.log(df["win_odds"]),
        np.nan,
    )

    # -----------------------------
    # Horse history features
    # -----------------------------
    horse_keys = ["horse_key"]

    df["horse_prev_starts"] = previous_count(df, horse_keys)
    df["horse_prev_wins"] = previous_cumsum(df, "target_win", horse_keys)
    df["horse_prev_top3"] = previous_cumsum(df, "target_place_top3", horse_keys)

    df["horse_prev_win_rate"] = safe_rate(df["horse_prev_wins"], df["horse_prev_starts"])
    df["horse_prev_top3_rate"] = safe_rate(df["horse_prev_top3"], df["horse_prev_starts"])

    df["horse_last_finish_pos"] = previous_shift(df, "outcome_place_num", horse_keys).astype("Float64")
    df["horse_last_finish_time_seconds"] = previous_shift(df, "outcome_finish_time_seconds", horse_keys).astype("Float64")

    df["horse_avg_finish_pos_last3"] = rolling_prior_mean(df, "outcome_place_num", horse_keys, 3)
    df["horse_avg_finish_pos_last5"] = rolling_prior_mean(df, "outcome_place_num", horse_keys, 5)

    df["horse_days_since_last_run"] = (
        df.groupby(horse_keys, sort=False)["race_date_dt"].diff().dt.days.astype("Float64")
    )

    # Horse same-course
    horse_course_keys = ["horse_key", "racecourse"]
    df["horse_prev_starts_same_course"] = previous_count(df, horse_course_keys)
    df["horse_prev_top3_same_course"] = previous_cumsum(df, "target_place_top3", horse_course_keys)
    df["horse_prev_top3_rate_same_course"] = safe_rate(
        df["horse_prev_top3_same_course"],
        df["horse_prev_starts_same_course"],
    )

    # Horse same-distance
    horse_distance_keys = ["horse_key", "race_distance_key"]
    df["horse_prev_starts_same_distance"] = previous_count(df, horse_distance_keys)
    df["horse_prev_top3_same_distance"] = previous_cumsum(df, "target_place_top3", horse_distance_keys)
    df["horse_prev_top3_rate_same_distance"] = safe_rate(
        df["horse_prev_top3_same_distance"],
        df["horse_prev_starts_same_distance"],
    )

    # Horse same-surface
    horse_surface_keys = ["horse_key", "race_surface_key"]
    df["horse_prev_starts_same_surface"] = previous_count(df, horse_surface_keys)
    df["horse_prev_top3_same_surface"] = previous_cumsum(df, "target_place_top3", horse_surface_keys)
    df["horse_prev_top3_rate_same_surface"] = safe_rate(
        df["horse_prev_top3_same_surface"],
        df["horse_prev_starts_same_surface"],
    )

    # -----------------------------
    # Jockey history features
    # -----------------------------
    jockey_keys = ["jockey"]

    df["jockey_prev_rides"] = previous_count(df, jockey_keys)
    df["jockey_prev_wins"] = previous_cumsum(df, "target_win", jockey_keys)
    df["jockey_prev_top3"] = previous_cumsum(df, "target_place_top3", jockey_keys)

    df["jockey_prev_win_rate"] = safe_rate(df["jockey_prev_wins"], df["jockey_prev_rides"])
    df["jockey_prev_top3_rate"] = safe_rate(df["jockey_prev_top3"], df["jockey_prev_rides"])

    df["jockey_win_rate_last30"] = rolling_prior_mean(df, "target_win", jockey_keys, 30).fillna(0.0)
    df["jockey_top3_rate_last30"] = rolling_prior_mean(df, "target_place_top3", jockey_keys, 30).fillna(0.0)

    # -----------------------------
    # Trainer history features
    # -----------------------------
    trainer_keys = ["trainer"]

    df["trainer_prev_runners"] = previous_count(df, trainer_keys)
    df["trainer_prev_wins"] = previous_cumsum(df, "target_win", trainer_keys)
    df["trainer_prev_top3"] = previous_cumsum(df, "target_place_top3", trainer_keys)

    df["trainer_prev_win_rate"] = safe_rate(df["trainer_prev_wins"], df["trainer_prev_runners"])
    df["trainer_prev_top3_rate"] = safe_rate(df["trainer_prev_top3"], df["trainer_prev_runners"])

    df["trainer_win_rate_last30"] = rolling_prior_mean(df, "target_win", trainer_keys, 30).fillna(0.0)
    df["trainer_top3_rate_last30"] = rolling_prior_mean(df, "target_place_top3", trainer_keys, 30).fillna(0.0)

    # -----------------------------
    # Horse-jockey combination
    # -----------------------------
    horse_jockey_keys = ["horse_key", "jockey"]

    df["horse_jockey_prev_starts"] = previous_count(df, horse_jockey_keys)
    df["horse_jockey_prev_wins"] = previous_cumsum(df, "target_win", horse_jockey_keys)
    df["horse_jockey_prev_top3"] = previous_cumsum(df, "target_place_top3", horse_jockey_keys)

    df["horse_jockey_prev_win_rate"] = safe_rate(
        df["horse_jockey_prev_wins"],
        df["horse_jockey_prev_starts"],
    )
    df["horse_jockey_prev_top3_rate"] = safe_rate(
        df["horse_jockey_prev_top3"],
        df["horse_jockey_prev_starts"],
    )

    # -----------------------------
    # Horse-trainer combination
    # -----------------------------
    horse_trainer_keys = ["horse_key", "trainer"]

    df["horse_trainer_prev_starts"] = previous_count(df, horse_trainer_keys)
    df["horse_trainer_prev_wins"] = previous_cumsum(df, "target_win", horse_trainer_keys)
    df["horse_trainer_prev_top3"] = previous_cumsum(df, "target_place_top3", horse_trainer_keys)

    df["horse_trainer_prev_win_rate"] = safe_rate(
        df["horse_trainer_prev_wins"],
        df["horse_trainer_prev_starts"],
    )
    df["horse_trainer_prev_top3_rate"] = safe_rate(
        df["horse_trainer_prev_top3"],
        df["horse_trainer_prev_starts"],
    )

    # Drop purely helper keys not needed downstream
    df = df.drop(columns=["race_distance_key", "race_surface_key"])

    return df


def build_summary(df_market_free: pd.DataFrame, df_market_aware: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {"metric": "market_free_rows", "value": len(df_market_free)},
        {"metric": "market_aware_rows", "value": len(df_market_aware)},
        {"metric": "market_free_columns", "value": df_market_free.shape[1]},
        {"metric": "market_aware_columns", "value": df_market_aware.shape[1]},
        {
            "metric": "distinct_races_market_free",
            "value": df_market_free[["race_date", "racecourse", "race_no"]].drop_duplicates().shape[0],
        },
        {"metric": "distinct_horses_market_free", "value": df_market_free["horse"].nunique()},
        {"metric": "target_win_sum_market_free", "value": int(df_market_free["target_win"].sum())},
        {"metric": "target_top3_sum_market_free", "value": int(df_market_free["target_place_top3"].sum())},
        {"metric": "dead_heat_rows_market_free", "value": int(df_market_free["outcome_dead_heat"].sum())},
    ]
    return pd.DataFrame(rows)


def main() -> None:
    if not SOURCE_DB_PATH.exists():
        raise FileNotFoundError(f"Source database not found: {SOURCE_DB_PATH}")

    print(f"Reading source database: {SOURCE_DB_PATH}")

    with sqlite3.connect(SOURCE_DB_PATH) as conn:
        base = pd.read_sql_query(f"SELECT * FROM {SOURCE_MODELING_TABLE}", conn)

    print(f"Loaded {len(base):,} modeling rows from {SOURCE_MODELING_TABLE}")

    features_all = build_features(base)

    # Market-aware keeps current-race odds features
    df_market_aware = features_all.copy()

    # Market-free removes current-race market columns
    market_free_drop_cols = [
        "win_odds",
        "log_win_odds",
    ]
    market_free_drop_cols = [col for col in market_free_drop_cols if col in features_all.columns]
    df_market_free = features_all.drop(columns=market_free_drop_cols).copy()

    summary_df = build_summary(df_market_free, df_market_aware)

    if OUTPUT_DB_PATH.exists():
        OUTPUT_DB_PATH.unlink()

    if WRITE_SQLITE_TABLES:
        with sqlite3.connect(OUTPUT_DB_PATH) as conn_out:
            df_market_free.to_sql(FEATURE_TABLE_MARKET_FREE, conn_out, index=False, if_exists="replace")
            df_market_aware.to_sql(FEATURE_TABLE_MARKET_AWARE, conn_out, index=False, if_exists="replace")
            summary_df.to_sql(FEATURE_SUMMARY_TABLE, conn_out, index=False, if_exists="replace")

    if WRITE_CSV_FILES:
        INTERIM_DIR.mkdir(parents=True, exist_ok=True)
        df_market_free.to_csv(FEATURE_CSV_MARKET_FREE, index=False)
        df_market_aware.to_csv(FEATURE_CSV_MARKET_AWARE, index=False)
        summary_df.to_csv(FEATURE_SUMMARY_CSV, index=False)

    print("\nCreated separate feature database:")
    print(OUTPUT_DB_PATH)

    print("\nSaved SQLite tables:")
    print(f"- {FEATURE_TABLE_MARKET_FREE}")
    print(f"- {FEATURE_TABLE_MARKET_AWARE}")
    print(f"- {FEATURE_SUMMARY_TABLE}")

    if WRITE_CSV_FILES:
        print("\nSaved CSV files:")
        print(f"- {FEATURE_CSV_MARKET_FREE}")
        print(f"- {FEATURE_CSV_MARKET_AWARE}")
        print(f"- {FEATURE_SUMMARY_CSV}")

    print("\nFeature summary:")
    print(summary_df.to_string(index=False))

    print("\nDone.")


if __name__ == "__main__":
    main()