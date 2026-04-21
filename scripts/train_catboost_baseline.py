from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

try:
    from catboost import CatBoostClassifier
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "catboost is required for this script. Install it in your venv with: python -m pip install catboost"
    ) from exc


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = DATA_DIR / "models"

FEATURE_DB_PATH = DATA_DIR / "hkjc_features_v2.db"

MARKET_FREE_TABLE = "model_features_v2_market_free"
MARKET_AWARE_TABLE = "model_features_v2_market_aware"

TRAIN_END_DATE = "2023-12-31"
VALID_END_DATE = "2024-12-31"
TEST_END_DATE = "2026-03-29"

TARGET_COL = "target_win"

METRICS_CSV = MODELS_DIR / "baseline_metrics_catboost_v2.csv"
PREDICTIONS_VALID_CSV = MODELS_DIR / "baseline_validation_predictions_catboost_v2.csv"
PREDICTIONS_TEST_CSV = MODELS_DIR / "baseline_test_predictions_catboost_v2.csv"
SPLIT_SUMMARY_CSV = MODELS_DIR / "baseline_split_summary_catboost_v2.csv"
TRAINING_DIAGNOSTICS_JSON = MODELS_DIR / "baseline_training_diagnostics_catboost_v2.json"

MARKET_FREE_MODEL_PATH = MODELS_DIR / "baseline_catboost_market_free_v2.joblib"
MARKET_AWARE_MODEL_PATH = MODELS_DIR / "baseline_catboost_market_aware_v2.joblib"

MARKET_FREE_FEATURES_JSON = MODELS_DIR / "baseline_catboost_market_free_features_v2.json"
MARKET_AWARE_FEATURES_JSON = MODELS_DIR / "baseline_catboost_market_aware_features_v2.json"

MARKET_FREE_IMPORTANCE_CSV = MODELS_DIR / "baseline_catboost_market_free_feature_importance_v2.csv"
MARKET_AWARE_IMPORTANCE_CSV = MODELS_DIR / "baseline_catboost_market_aware_feature_importance_v2.csv"

CATEGORICAL_FEATURES = [
    "racecourse",
    "race_surface",
    "race_going",
    "race_class_label",
    "race_rail_code",
]

NUMERIC_FEATURES_COMMON = [
    "race_distance_m",
    "dr",
    "act_wt",
    "declar_horse_wt",
    "field_size",
    "draw_norm_by_field",
    "weight_diff_from_declared",
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
    "jockey_prev_rides",
    "jockey_prev_wins",
    "jockey_prev_top3",
    "jockey_prev_win_rate",
    "jockey_prev_top3_rate",
    "jockey_win_rate_last30",
    "jockey_top3_rate_last30",
    "trainer_prev_runners",
    "trainer_prev_wins",
    "trainer_prev_top3",
    "trainer_prev_win_rate",
    "trainer_prev_top3_rate",
    "trainer_win_rate_last30",
    "trainer_top3_rate_last30",
    "horse_jockey_prev_starts",
    "horse_jockey_prev_wins",
    "horse_jockey_prev_top3",
    "horse_jockey_prev_win_rate",
    "horse_jockey_prev_top3_rate",
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

MIN_CATEGORY_COUNT = 20
CLIP_LOWER_Q = 0.001
CLIP_UPPER_Q = 0.999
EARLY_STOPPING_ROUNDS = 100

CATBOOST_PARAMS: dict[str, Any] = {
    "loss_function": "Logloss",
    "eval_metric": "Logloss",
    "iterations": 2000,
    "learning_rate": 0.03,
    "depth": 6,
    "l2_leaf_reg": 5.0,
    "min_data_in_leaf": 50,
    "subsample": 0.8,
    "rsm": 0.8,
    "random_seed": 42,
    "verbose": False,
    "allow_writing_files": False,
}


def read_table(db_path: Path, table_name: str) -> pd.DataFrame:
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query(f"SELECT * FROM {table_name}", conn)


def ensure_dirs() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)


def assert_columns_exist(df: pd.DataFrame, required_cols: list[str], table_name: str) -> None:
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {table_name}: {missing}")


def build_split_column(df: pd.DataFrame) -> pd.Series:
    race_date = pd.to_datetime(df["race_date"], errors="coerce")
    split = pd.Series(index=df.index, dtype="string")
    split.loc[race_date <= pd.Timestamp(TRAIN_END_DATE)] = "train"
    split.loc[(race_date > pd.Timestamp(TRAIN_END_DATE)) & (race_date <= pd.Timestamp(VALID_END_DATE))] = "valid"
    split.loc[(race_date > pd.Timestamp(VALID_END_DATE)) & (race_date <= pd.Timestamp(TEST_END_DATE))] = "test"
    return split


def build_feature_list(df: pd.DataFrame, market_aware: bool) -> tuple[list[str], list[str]]:
    categorical = [c for c in CATEGORICAL_FEATURES if c in df.columns and c not in NEVER_USE_AS_FEATURES]
    numeric = [c for c in NUMERIC_FEATURES_COMMON if c in df.columns and c not in NEVER_USE_AS_FEATURES]
    if market_aware:
        numeric += [c for c in NUMERIC_FEATURES_MARKET_AWARE_EXTRA if c in df.columns and c not in NEVER_USE_AS_FEATURES]
    return categorical, numeric


def prepare_categorical_train(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    categorical_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    diagnostics: dict[str, Any] = {"rare_category_levels": {}, "kept_category_levels": {}}

    for col in categorical_cols:
        for frame in [train_df, valid_df, test_df]:
            s = frame[col].astype("string").fillna("MISSING").str.strip()
            s = s.replace("", "MISSING")
            frame[col] = s

        value_counts = train_df[col].value_counts(dropna=False)
        keep_levels = set(value_counts[value_counts >= MIN_CATEGORY_COUNT].index.astype(str).tolist())
        keep_levels.add("MISSING")

        diagnostics["rare_category_levels"][col] = int((value_counts < MIN_CATEGORY_COUNT).sum())
        diagnostics["kept_category_levels"][col] = sorted(keep_levels)

        for frame in [train_df, valid_df, test_df]:
            frame[col] = frame[col].where(frame[col].isin(keep_levels), "RARE")

    return train_df, valid_df, test_df, diagnostics


def prepare_numeric_train(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    numeric_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str], dict[str, Any]]:
    diagnostics: dict[str, Any] = {
        "numeric_clip_bounds": {},
        "dropped_constant_numeric": [],
    }
    keep_numeric: list[str] = []

    for col in numeric_cols:
        for frame in [train_df, valid_df, test_df]:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
            frame[col] = frame[col].replace([np.inf, -np.inf], np.nan)

        nunique = train_df[col].dropna().nunique()
        if nunique <= 1:
            diagnostics["dropped_constant_numeric"].append(col)
            continue

        lower = float(train_df[col].quantile(CLIP_LOWER_Q)) if train_df[col].notna().any() else np.nan
        upper = float(train_df[col].quantile(CLIP_UPPER_Q)) if train_df[col].notna().any() else np.nan
        diagnostics["numeric_clip_bounds"][col] = {"lower": lower, "upper": upper}

        if np.isfinite(lower) and np.isfinite(upper) and lower < upper:
            for frame in [train_df, valid_df, test_df]:
                frame[col] = frame[col].clip(lower=lower, upper=upper)

        keep_numeric.append(col)

    return train_df, valid_df, test_df, keep_numeric, diagnostics


def median_impute_using_train(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    numeric_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, float]]:
    medians: dict[str, float] = {}
    for col in numeric_cols:
        median = float(train_df[col].median()) if train_df[col].notna().any() else 0.0
        medians[col] = median
        for frame in [train_df, valid_df, test_df]:
            frame[col] = frame[col].fillna(median)
    return train_df, valid_df, test_df, medians


def race_level_metrics(pred_df: pd.DataFrame) -> dict[str, float]:
    from sklearn.metrics import brier_score_loss, log_loss  # noqa: F401

    race_keys = ["race_date", "racecourse", "race_no"]
    ranked = pred_df.sort_values(
        race_keys + ["pred_win_prob", "horse"],
        ascending=[True, True, True, False, True],
        kind="mergesort",
    ).copy()
    ranked["rank_in_race"] = ranked.groupby(race_keys, sort=False, observed=False).cumcount() + 1

    top_pick = ranked[ranked["rank_in_race"] == 1].copy()
    top3 = ranked[ranked["rank_in_race"] <= 3].copy()

    top_pick_win_rate = float(top_pick[TARGET_COL].mean()) if len(top_pick) else float("nan")
    top3_by_race = top3.groupby(race_keys, sort=False, observed=False)[TARGET_COL].max().reset_index(name="winner_in_top3")
    winner_in_top3_rate = float(top3_by_race["winner_in_top3"].mean()) if len(top3_by_race) else float("nan")
    return {
        "top_pick_win_rate": top_pick_win_rate,
        "winner_in_top3_rate": winner_in_top3_rate,
    }


def evaluate_split(pred_df: pd.DataFrame, model_name: str, split_name: str) -> dict[str, Any]:
    from sklearn.metrics import brier_score_loss, log_loss

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
        "mean_pred_win_prob": float(np.mean(y_prob)),
        "std_pred_win_prob": float(np.std(y_prob)),
    }
    metrics.update(race_level_metrics(pred_df))
    return metrics


def build_prediction_frame(split_df: pd.DataFrame, pred_prob: np.ndarray, model_name: str, split_name: str) -> pd.DataFrame:
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
    pred_out["rank_in_race"] = pred_out.groupby(["race_date", "racecourse", "race_no"], sort=False, observed=False).cumcount() + 1
    return pred_out


def build_feature_importance_df(model: CatBoostClassifier, feature_cols: list[str]) -> pd.DataFrame:
    importance = model.get_feature_importance(type="FeatureImportance")
    df = pd.DataFrame({
        "feature_name": feature_cols,
        "importance": importance,
    })
    df = df.sort_values(["importance", "feature_name"], ascending=[False, True]).reset_index(drop=True)
    return df


def fit_and_score(
    df: pd.DataFrame,
    model_name: str,
    market_aware: bool,
    model_path: Path,
    features_json_path: Path,
    importance_csv_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]], dict[str, Any]]:
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
        raise ValueError(f"{model_name}: one or more splits are empty. train={len(train_df)}, valid={len(valid_df)}, test={len(test_df)}")

    train_df, valid_df, test_df, cat_diag = prepare_categorical_train(train_df, valid_df, test_df, categorical_cols)
    train_df, valid_df, test_df, numeric_cols_kept, num_diag = prepare_numeric_train(train_df, valid_df, test_df, numeric_cols)
    train_df, valid_df, test_df, median_map = median_impute_using_train(train_df, valid_df, test_df, numeric_cols_kept)

    feature_cols_final = categorical_cols + numeric_cols_kept
    X_train = train_df[feature_cols_final].copy()
    y_train = train_df[TARGET_COL].astype(int)
    X_valid = valid_df[feature_cols_final].copy()
    y_valid = valid_df[TARGET_COL].astype(int)
    X_test = test_df[feature_cols_final].copy()

    for col in categorical_cols:
        X_train[col] = X_train[col].astype(str)
        X_valid[col] = X_valid[col].astype(str)
        X_test[col] = X_test[col].astype(str)

    model = CatBoostClassifier(**CATBOOST_PARAMS)
    model.fit(
        X_train,
        y_train,
        cat_features=categorical_cols,
        eval_set=(X_valid, y_valid),
        use_best_model=True,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        verbose=False,
    )

    joblib.dump(model, model_path)

    feature_metadata = {
        "model_name": model_name,
        "categorical_features": categorical_cols,
        "numeric_features_requested": numeric_cols,
        "numeric_features_used": numeric_cols_kept,
        "all_feature_columns": feature_cols_final,
        "target_column": TARGET_COL,
        "train_end_date": TRAIN_END_DATE,
        "valid_end_date": VALID_END_DATE,
        "test_end_date": TEST_END_DATE,
        "catboost_params": CATBOOST_PARAMS,
        "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
        "min_category_count": MIN_CATEGORY_COUNT,
        "clip_lower_q": CLIP_LOWER_Q,
        "clip_upper_q": CLIP_UPPER_Q,
    }
    with features_json_path.open("w", encoding="utf-8") as f:
        json.dump(feature_metadata, f, indent=2)

    importance_df = build_feature_importance_df(model, feature_cols_final)
    importance_df.to_csv(importance_csv_path, index=False)

    valid_pred = model.predict_proba(X_valid)[:, 1]
    test_pred = model.predict_proba(X_test)[:, 1]

    valid_out = build_prediction_frame(valid_df, valid_pred, model_name=model_name, split_name="valid")
    test_out = build_prediction_frame(test_df, test_pred, model_name=model_name, split_name="test")

    metrics = [
        evaluate_split(valid_out, model_name=model_name, split_name="valid"),
        evaluate_split(test_out, model_name=model_name, split_name="test"),
    ]

    diagnostics = {
        "model_name": model_name,
        "market_aware": market_aware,
        "train_rows": len(train_df),
        "valid_rows": len(valid_df),
        "test_rows": len(test_df),
        "train_positive_count": int(y_train.sum()),
        "train_positive_rate": float(y_train.mean()),
        "categorical_feature_count": len(categorical_cols),
        "numeric_feature_count_requested": len(numeric_cols),
        "numeric_feature_count_used": len(numeric_cols_kept),
        "feature_count_total_used": len(feature_cols_final),
        "best_iteration": int(model.get_best_iteration()) if model.get_best_iteration() is not None else None,
        "best_score": model.get_best_score(),
        "catboost_params": CATBOOST_PARAMS,
        "categorical_diagnostics": cat_diag,
        "numeric_diagnostics": num_diag,
        "numeric_medians": median_map,
    }
    return valid_out, test_out, metrics, diagnostics


def build_split_summary(df_market_free: pd.DataFrame, df_market_aware: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model_name, df in [("market_free", df_market_free.copy()), ("market_aware", df_market_aware.copy())]:
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

    print("\nTraining CatBoost market-free baseline...")
    valid_free, test_free, metrics_free, diag_free = fit_and_score(
        df=df_market_free,
        model_name="market_free",
        market_aware=False,
        model_path=MARKET_FREE_MODEL_PATH,
        features_json_path=MARKET_FREE_FEATURES_JSON,
        importance_csv_path=MARKET_FREE_IMPORTANCE_CSV,
    )

    print("Training CatBoost market-aware baseline...")
    valid_aware, test_aware, metrics_aware, diag_aware = fit_and_score(
        df=df_market_aware,
        model_name="market_aware",
        market_aware=True,
        model_path=MARKET_AWARE_MODEL_PATH,
        features_json_path=MARKET_AWARE_FEATURES_JSON,
        importance_csv_path=MARKET_AWARE_IMPORTANCE_CSV,
    )

    metrics_df = pd.DataFrame(metrics_free + metrics_aware)
    metrics_df.to_csv(METRICS_CSV, index=False)

    valid_preds = pd.concat([valid_free, valid_aware], ignore_index=True, sort=False)
    test_preds = pd.concat([test_free, test_aware], ignore_index=True, sort=False)
    valid_preds.to_csv(PREDICTIONS_VALID_CSV, index=False)
    test_preds.to_csv(PREDICTIONS_TEST_CSV, index=False)

    diagnostics = {"market_free": diag_free, "market_aware": diag_aware}
    with TRAINING_DIAGNOSTICS_JSON.open("w", encoding="utf-8") as f:
        json.dump(diagnostics, f, indent=2)

    print("\nMetrics:")
    print(metrics_df.to_string(index=False))

    print("\nSaved files:")
    print(f"- {MARKET_FREE_MODEL_PATH}")
    print(f"- {MARKET_AWARE_MODEL_PATH}")
    print(f"- {MARKET_FREE_FEATURES_JSON}")
    print(f"- {MARKET_AWARE_FEATURES_JSON}")
    print(f"- {MARKET_FREE_IMPORTANCE_CSV}")
    print(f"- {MARKET_AWARE_IMPORTANCE_CSV}")
    print(f"- {METRICS_CSV}")
    print(f"- {PREDICTIONS_VALID_CSV}")
    print(f"- {PREDICTIONS_TEST_CSV}")
    print(f"- {SPLIT_SUMMARY_CSV}")
    print(f"- {TRAINING_DIAGNOSTICS_JSON}")

    print("\nDone.")


if __name__ == "__main__":
    main()
