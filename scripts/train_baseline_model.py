from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sqlite3

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
INTERIM_DIR = DATA_DIR / "interim"
MODELS_DIR = DATA_DIR / "models"

FEATURE_DB_PATH = DATA_DIR / "hkjc_features_v2.db"

MARKET_FREE_TABLE = "model_features_v2_market_free"
MARKET_AWARE_TABLE = "model_features_v2_market_aware"

# ---------------------------------
# TRAIN / VALID / TEST SPLIT
# ---------------------------------
TRAIN_END_DATE = "2023-12-31"
VALID_END_DATE = "2024-12-31"
TEST_END_DATE = "2026-03-29"

TARGET_COL = "target_win"

# ---------------------------------
# OUTPUT FILES
# ---------------------------------
METRICS_CSV = MODELS_DIR / "baseline_metrics_v2.csv"
PREDICTIONS_VALID_CSV = MODELS_DIR / "baseline_validation_predictions_v2.csv"
PREDICTIONS_TEST_CSV = MODELS_DIR / "baseline_test_predictions_v2.csv"
SPLIT_SUMMARY_CSV = MODELS_DIR / "baseline_split_summary_v2.csv"

MARKET_FREE_MODEL_PATH = MODELS_DIR / "baseline_logreg_market_free_v2.joblib"
MARKET_AWARE_MODEL_PATH = MODELS_DIR / "baseline_logreg_market_aware_v2.joblib"

MARKET_FREE_FEATURES_JSON = MODELS_DIR / "baseline_logreg_market_free_features_v2.json"
MARKET_AWARE_FEATURES_JSON = MODELS_DIR / "baseline_logreg_market_aware_features_v2.json"

# ---------------------------------
# FEATURE LISTS
# ---------------------------------
CATEGORICAL_FEATURES = [
    "racecourse",
    "race_surface",
    "race_going",
    "race_class_label",
    "race_rail_code",
]

NUMERIC_FEATURES_COMMON = [
    # Current-race context
    "race_distance_m",
    "dr",
    "act_wt",
    "declar_horse_wt",
    "field_size",
    "draw_norm_by_field",
    "weight_diff_from_declared",

    # Horse history
    "horse_prev_starts",
    "horse_prev_wins",
    "horse_prev_top3",
    "horse_prev_win_rate",
    "horse_prev_top3_rate",
    "horse_last_finish_pos",
    "horse_last_finish_time_seconds",
    "horse_avg_finish_pos_last3",
    "horse_avg_finish_pos_last5",
    "horse_days_since_last_run",
    "horse_prev_starts_same_course",
    "horse_prev_top3_rate_same_course",
    "horse_prev_starts_same_distance",
    "horse_prev_top3_rate_same_distance",
    "horse_prev_starts_same_surface",
    "horse_prev_top3_rate_same_surface",

    # Jockey history
    "jockey_prev_rides",
    "jockey_prev_wins",
    "jockey_prev_top3",
    "jockey_prev_win_rate",
    "jockey_prev_top3_rate",
    "jockey_win_rate_last30",
    "jockey_top3_rate_last30",

    # Trainer history
    "trainer_prev_runners",
    "trainer_prev_wins",
    "trainer_prev_top3",
    "trainer_prev_win_rate",
    "trainer_prev_top3_rate",
    "trainer_win_rate_last30",
    "trainer_top3_rate_last30",

    # Horse-jockey combo
    "horse_jockey_prev_starts",
    "horse_jockey_prev_wins",
    "horse_jockey_prev_top3",
    "horse_jockey_prev_win_rate",
    "horse_jockey_prev_top3_rate",

    # Horse-trainer combo
    "horse_trainer_prev_starts",
    "horse_trainer_prev_wins",
    "horse_trainer_prev_top3",
    "horse_trainer_prev_win_rate",
    "horse_trainer_prev_top3_rate",
]

NUMERIC_FEATURES_MARKET_AWARE_EXTRA = [
    "win_odds",
    "log_win_odds",
]

# Columns that should never be used as model inputs
NEVER_USE_AS_FEATURES = {
    "target_finished",
    "target_win",
    "target_place_top2",
    "target_place_top3",
    "target_place_top4",
    "outcome_place_raw",
    "outcome_place_num",
    "outcome_dead_heat",
    "outcome_result_status",
    "outcome_finish_time",
    "outcome_finish_time_seconds",
    "outcome_lbw",
    "outcome_running_position",
}


def read_table(db_path: Path, table_name: str) -> pd.DataFrame:
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query(f"SELECT * FROM {table_name}", conn)


def ensure_dirs() -> None:
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)


def assert_columns_exist(df: pd.DataFrame, required_cols: list[str], table_name: str) -> None:
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns in {table_name}: {missing}"
        )


def build_split_column(df: pd.DataFrame) -> pd.Series:
    race_date = pd.to_datetime(df["race_date"], errors="coerce")

    split = pd.Series(index=df.index, dtype="string")

    split.loc[race_date <= pd.Timestamp(TRAIN_END_DATE)] = "train"
    split.loc[(race_date > pd.Timestamp(TRAIN_END_DATE)) & (race_date <= pd.Timestamp(VALID_END_DATE))] = "valid"
    split.loc[(race_date > pd.Timestamp(VALID_END_DATE)) & (race_date <= pd.Timestamp(TEST_END_DATE))] = "test"

    return split


def make_preprocessor(categorical_cols: list[str], numeric_cols: list[str]) -> ColumnTransformer:
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="MISSING")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, categorical_cols),
            ("num", numeric_transformer, numeric_cols),
        ]
    )

    return preprocessor


def make_model_pipeline(categorical_cols: list[str], numeric_cols: list[str]) -> Pipeline:
    preprocessor = make_preprocessor(categorical_cols, numeric_cols)

    model = LogisticRegression(
        max_iter=2000,
        solver="lbfgs",
        n_jobs=None,
        random_state=42,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )
    return pipeline


def build_feature_list(df: pd.DataFrame, market_aware: bool) -> tuple[list[str], list[str]]:
    categorical = [c for c in CATEGORICAL_FEATURES if c in df.columns]
    numeric = [c for c in NUMERIC_FEATURES_COMMON if c in df.columns]

    if market_aware:
        numeric += [c for c in NUMERIC_FEATURES_MARKET_AWARE_EXTRA if c in df.columns]

    # Final safety filter
    categorical = [c for c in categorical if c not in NEVER_USE_AS_FEATURES]
    numeric = [c for c in numeric if c not in NEVER_USE_AS_FEATURES]

    return categorical, numeric


def race_level_metrics(pred_df: pd.DataFrame) -> dict[str, float]:
    """
    Race-level metrics based on predicted probabilities.

    top_pick_win_rate:
        For each race, choose the highest-probability horse.
        Measure how often it actually won.

    winner_in_top3_rate:
        For each race, take the top 3 horses by model probability.
        Check whether any true winner is in that top 3.
    """
    race_keys = ["race_date", "racecourse", "race_no"]

    ranked = pred_df.sort_values(
        race_keys + ["pred_win_prob", "horse"],
        ascending=[True, True, True, False, True],
        kind="mergesort",
    ).copy()

    ranked["rank_in_race"] = ranked.groupby(race_keys, sort=False).cumcount() + 1

    top_pick = ranked[ranked["rank_in_race"] == 1].copy()
    top3 = ranked[ranked["rank_in_race"] <= 3].copy()

    top_pick_win_rate = float(top_pick[TARGET_COL].mean()) if len(top_pick) else float("nan")

    top3_by_race = (
        top3.groupby(race_keys, sort=False)[TARGET_COL]
        .max()
        .reset_index(name="winner_in_top3")
    )
    winner_in_top3_rate = (
        float(top3_by_race["winner_in_top3"].mean()) if len(top3_by_race) else float("nan")
    )

    return {
        "top_pick_win_rate": top_pick_win_rate,
        "winner_in_top3_rate": winner_in_top3_rate,
    }


def evaluate_split(pred_df: pd.DataFrame, model_name: str, split_name: str) -> dict[str, Any]:
    y_true = pred_df[TARGET_COL].astype(int).to_numpy()
    y_prob = pred_df["pred_win_prob"].astype(float).to_numpy()

    metrics = {
        "model_name": model_name,
        "split": split_name,
        "row_count": len(pred_df),
        "race_count": pred_df[["race_date", "racecourse", "race_no"]].drop_duplicates().shape[0],
        "positive_count": int(pred_df[TARGET_COL].sum()),
        "positive_rate": float(pred_df[TARGET_COL].mean()),
        "log_loss": float(log_loss(y_true, y_prob, labels=[0, 1])),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
    }
    metrics.update(race_level_metrics(pred_df))
    return metrics


def fit_and_score(
    df: pd.DataFrame,
    model_name: str,
    market_aware: bool,
    model_path: Path,
    features_json_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    df = df.copy()

    categorical_cols, numeric_cols = build_feature_list(df, market_aware=market_aware)
    feature_cols = categorical_cols + numeric_cols

    assert_columns_exist(df, ["race_date", "racecourse", "race_no", "horse", TARGET_COL], model_name)
    assert_columns_exist(df, feature_cols, model_name)

    df["split"] = build_split_column(df)
    df = df[df["split"].isin(["train", "valid", "test"])].copy()

    train_df = df[df["split"] == "train"].copy()
    valid_df = df[df["split"] == "valid"].copy()
    test_df = df[df["split"] == "test"].copy()

    if train_df.empty or valid_df.empty or test_df.empty:
        raise ValueError(
            f"{model_name}: one or more splits are empty. "
            f"train={len(train_df)}, valid={len(valid_df)}, test={len(test_df)}"
        )

    X_train = train_df[feature_cols]
    y_train = train_df[TARGET_COL].astype(int)

    pipeline = make_model_pipeline(categorical_cols, numeric_cols)
    pipeline.fit(X_train, y_train)

    joblib.dump(pipeline, model_path)

    feature_metadata = {
        "model_name": model_name,
        "categorical_features": categorical_cols,
        "numeric_features": numeric_cols,
        "all_feature_columns": feature_cols,
        "target_column": TARGET_COL,
        "train_end_date": TRAIN_END_DATE,
        "valid_end_date": VALID_END_DATE,
        "test_end_date": TEST_END_DATE,
    }
    with features_json_path.open("w", encoding="utf-8") as f:
        json.dump(feature_metadata, f, indent=2)

    split_metrics: list[dict[str, Any]] = []
    prediction_frames: dict[str, pd.DataFrame] = {}

    for split_name, split_df in [("valid", valid_df), ("test", test_df)]:
        X_split = split_df[feature_cols]
        pred_prob = pipeline.predict_proba(X_split)[:, 1]

        pred_out = split_df[
            [
                "race_date",
                "racecourse",
                "race_no",
                "horse",
                "jockey",
                "trainer",
                TARGET_COL,
            ]
        ].copy()

        if "win_odds" in split_df.columns:
            pred_out["win_odds"] = split_df["win_odds"].values

        pred_out["model_name"] = model_name
        pred_out["split"] = split_name
        pred_out["pred_win_prob"] = pred_prob

        pred_out = pred_out.sort_values(
            ["race_date", "racecourse", "race_no", "pred_win_prob", "horse"],
            ascending=[True, True, True, False, True],
            kind="mergesort",
        ).reset_index(drop=True)

        pred_out["rank_in_race"] = (
            pred_out.groupby(["race_date", "racecourse", "race_no"], sort=False)
            .cumcount() + 1
        )

        prediction_frames[split_name] = pred_out
        split_metrics.append(evaluate_split(pred_out, model_name=model_name, split_name=split_name))

    return prediction_frames["valid"], prediction_frames["test"], split_metrics


def build_split_summary(
    df_market_free: pd.DataFrame,
    df_market_aware: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for model_name, df in [
        ("market_free", df_market_free.copy()),
        ("market_aware", df_market_aware.copy()),
    ]:
        df["split"] = build_split_column(df)
        df = df[df["split"].isin(["train", "valid", "test"])].copy()

        for split_name in ["train", "valid", "test"]:
            sub = df[df["split"] == split_name].copy()
            rows.append(
                {
                    "model_name": model_name,
                    "split": split_name,
                    "row_count": len(sub),
                    "race_count": sub[["race_date", "racecourse", "race_no"]].drop_duplicates().shape[0],
                    "positive_count": int(sub[TARGET_COL].sum()) if len(sub) else 0,
                    "positive_rate": float(sub[TARGET_COL].mean()) if len(sub) else np.nan,
                }
            )

    return pd.DataFrame(rows)


def main() -> None:
    ensure_dirs()

    if not FEATURE_DB_PATH.exists():
        raise FileNotFoundError(f"Feature database not found: {FEATURE_DB_PATH}")

    print(f"Reading feature database: {FEATURE_DB_PATH}")

    df_market_free = read_table(FEATURE_DB_PATH, MARKET_FREE_TABLE)
    df_market_aware = read_table(FEATURE_DB_PATH, MARKET_AWARE_TABLE)

    split_summary = build_split_summary(df_market_free, df_market_aware)
    split_summary.to_csv(SPLIT_SUMMARY_CSV, index=False)

    print("\nSplit summary:")
    print(split_summary.to_string(index=False))

    print("\nTraining market-free baseline...")
    valid_free, test_free, metrics_free = fit_and_score(
        df=df_market_free,
        model_name="market_free",
        market_aware=False,
        model_path=MARKET_FREE_MODEL_PATH,
        features_json_path=MARKET_FREE_FEATURES_JSON,
    )

    print("Training market-aware baseline...")
    valid_aware, test_aware, metrics_aware = fit_and_score(
        df=df_market_aware,
        model_name="market_aware",
        market_aware=True,
        model_path=MARKET_AWARE_MODEL_PATH,
        features_json_path=MARKET_AWARE_FEATURES_JSON,
    )

    metrics_df = pd.DataFrame(metrics_free + metrics_aware)
    metrics_df.to_csv(METRICS_CSV, index=False)

    valid_preds = pd.concat([valid_free, valid_aware], ignore_index=True, sort=False)
    test_preds = pd.concat([test_free, test_aware], ignore_index=True, sort=False)

    valid_preds.to_csv(PREDICTIONS_VALID_CSV, index=False)
    test_preds.to_csv(PREDICTIONS_TEST_CSV, index=False)

    print("\nMetrics:")
    print(metrics_df.to_string(index=False))

    print("\nSaved files:")
    print(f"- {MARKET_FREE_MODEL_PATH}")
    print(f"- {MARKET_AWARE_MODEL_PATH}")
    print(f"- {MARKET_FREE_FEATURES_JSON}")
    print(f"- {MARKET_AWARE_FEATURES_JSON}")
    print(f"- {METRICS_CSV}")
    print(f"- {PREDICTIONS_VALID_CSV}")
    print(f"- {PREDICTIONS_TEST_CSV}")
    print(f"- {SPLIT_SUMMARY_CSV}")

    print("\nDone.")


if __name__ == "__main__":
    main()