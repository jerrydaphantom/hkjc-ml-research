from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = DATA_DIR / "models"

VALID_GLOB = "baseline_validation_predictions*_v2.csv"
EXTRA_VALID_FILES = [
    "baseline_validation_predictions_v3.csv",
]

TARGET_COL = "target_win"
RACE_KEYS = ["race_date", "racecourse", "race_no"]

PREDICTIONS_VALID_CSV = MODELS_DIR / "calibrated_validation_predictions_v2.csv"
PREDICTIONS_TEST_CSV = MODELS_DIR / "calibrated_test_predictions_v2.csv"
METRICS_CSV = MODELS_DIR / "calibrated_probability_metrics_v2.csv"
TEST_RANKING_CSV = MODELS_DIR / "calibrated_probability_test_ranking_v2.csv"
BIN_STATS_CSV = MODELS_DIR / "calibrated_probability_bin_stats_v2.csv"
MODEL_REGISTRY_CSV = MODELS_DIR / "calibrated_probability_model_registry_v2.csv"
SUMMARY_JSON = MODELS_DIR / "calibrated_probability_summary_v2.json"
CALIBRATORS_JOBLIB = MODELS_DIR / "calibrated_probability_models_v2.joblib"

EPS = 1e-6
RELIABILITY_BINS = 10


class PlattCalibrator:
    """Platt-style scaling on the logit of an input probability."""

    def __init__(self) -> None:
        self.model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000, random_state=42)

    @staticmethod
    def _to_feature(p: np.ndarray) -> np.ndarray:
        p = np.clip(np.asarray(p, dtype=float), EPS, 1.0 - EPS)
        x = np.log(p / (1.0 - p))
        return x.reshape(-1, 1)

    def fit(self, p: np.ndarray, y: np.ndarray) -> "PlattCalibrator":
        x = self._to_feature(p)
        self.model.fit(x, np.asarray(y, dtype=int))
        return self

    def predict(self, p: np.ndarray) -> np.ndarray:
        x = self._to_feature(p)
        out = self.model.predict_proba(x)[:, 1]
        return np.clip(out, EPS, 1.0 - EPS)


class IdentityCalibrator:
    def fit(self, p: np.ndarray, y: np.ndarray) -> "IdentityCalibrator":
        _ = y
        return self

    def predict(self, p: np.ndarray) -> np.ndarray:
        out = np.asarray(p, dtype=float)
        return np.clip(out, EPS, 1.0 - EPS)


def ensure_dirs() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)


def infer_family_from_filename(path: Path) -> str:
    stem = path.stem
    if stem == "baseline_validation_predictions_v2":
        return "logreg_lbfgs"
    if stem == "baseline_validation_predictions_v3":
        return "logreg_stable"
    if stem == "baseline_validation_predictions_alt_solver_v2":
        return "logreg_alt_solver"
    if stem == "baseline_validation_predictions_lgbm_v2":
        return "lightgbm"
    if stem == "baseline_validation_predictions_catboost_v2":
        return "catboost"
    if stem == "baseline_validation_predictions_rf_v2":
        return "random_forest"

    stem = re.sub(r"^baseline_validation_predictions_", "", stem)
    stem = re.sub(r"_v\d+$", "", stem)
    return stem


def discover_prediction_file_pairs() -> list[tuple[Path, Path, str]]:
    valid_files = sorted(MODELS_DIR.glob(VALID_GLOB))
    for extra_name in EXTRA_VALID_FILES:
        extra_path = MODELS_DIR / extra_name
        if extra_path.exists() and extra_path not in valid_files:
            valid_files.append(extra_path)

    pairs: list[tuple[Path, Path, str]] = []
    for valid_path in sorted(valid_files):
        test_name = valid_path.name.replace("baseline_validation_predictions", "baseline_test_predictions")
        test_path = MODELS_DIR / test_name
        if not test_path.exists():
            continue
        family = infer_family_from_filename(valid_path)
        pairs.append((valid_path, test_path, family))
    return pairs


def variant_from_row(row: pd.Series, group_cols: list[str]) -> str:
    parts: list[str] = []
    for col in group_cols:
        value = row[col]
        if pd.isna(value):
            continue
        if col == "model_name":
            parts.append(str(value))
        else:
            parts.append(f"{value}")
    return "__".join(parts)


def make_model_key(family: str, variant: str) -> str:
    return f"{family}::{variant}"


def race_level_metrics(pred_df: pd.DataFrame, prob_col: str) -> dict[str, float]:
    ranked = pred_df.sort_values(
        RACE_KEYS + [prob_col, "horse"],
        ascending=[True, True, True, False, True],
        kind="mergesort",
    ).copy()

    ranked["rank_in_race"] = ranked.groupby(RACE_KEYS, sort=False, observed=False).cumcount() + 1
    top_pick = ranked[ranked["rank_in_race"] == 1].copy()
    top3 = ranked[ranked["rank_in_race"] <= 3].copy()

    top_pick_win_rate = float(top_pick[TARGET_COL].mean()) if len(top_pick) else float("nan")
    top3_by_race = (
        top3.groupby(RACE_KEYS, sort=False, observed=False)[TARGET_COL]
        .max()
        .reset_index(name="winner_in_top3")
    )
    winner_in_top3_rate = float(top3_by_race["winner_in_top3"].mean()) if len(top3_by_race) else float("nan")

    return {
        "top_pick_win_rate": top_pick_win_rate,
        "winner_in_top3_rate": winner_in_top3_rate,
    }


def race_sum_stats(pred_df: pd.DataFrame, prob_col: str) -> dict[str, float]:
    race_sum = pred_df.groupby(RACE_KEYS, sort=False, observed=False)[prob_col].sum().reset_index(name="race_prob_sum")
    err = race_sum["race_prob_sum"] - 1.0
    return {
        "mean_race_prob_sum": float(race_sum["race_prob_sum"].mean()),
        "median_race_prob_sum": float(race_sum["race_prob_sum"].median()),
        "mae_race_prob_sum_vs_1": float(err.abs().mean()),
    }


def evaluate_predictions(pred_df: pd.DataFrame, prob_col: str) -> dict[str, Any]:
    y_true = pred_df[TARGET_COL].astype(int).to_numpy()
    y_prob = np.clip(pred_df[prob_col].astype(float).to_numpy(), EPS, 1.0 - EPS)

    metrics = {
        "row_count": len(pred_df),
        "race_count": pred_df[RACE_KEYS].drop_duplicates().shape[0],
        "positive_count": int(pred_df[TARGET_COL].sum()),
        "positive_rate": float(pred_df[TARGET_COL].mean()),
        "log_loss": float(log_loss(y_true, y_prob, labels=[0, 1])),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "mean_pred_win_prob": float(np.mean(y_prob)),
        "std_pred_win_prob": float(np.std(y_prob)),
    }
    metrics.update(race_level_metrics(pred_df, prob_col=prob_col))
    metrics.update(race_sum_stats(pred_df, prob_col=prob_col))
    return metrics


def build_bin_stats(pred_df: pd.DataFrame, prob_col: str, n_bins: int = RELIABILITY_BINS) -> pd.DataFrame:
    df = pred_df[[TARGET_COL, prob_col]].copy()
    df[prob_col] = pd.to_numeric(df[prob_col], errors="coerce").clip(EPS, 1.0 - EPS)
    df = df.dropna(subset=[prob_col]).copy()

    if df.empty:
        return pd.DataFrame()

    try:
        df["bin"] = pd.qcut(df[prob_col], q=min(n_bins, df[prob_col].nunique()), duplicates="drop")
    except ValueError:
        return pd.DataFrame()

    out = (
        df.groupby("bin", observed=False)
        .agg(
            row_count=(TARGET_COL, "size"),
            mean_pred_prob=(prob_col, "mean"),
            observed_rate=(TARGET_COL, "mean"),
            min_pred_prob=(prob_col, "min"),
            max_pred_prob=(prob_col, "max"),
        )
        .reset_index()
    )
    out["calibration_gap"] = out["observed_rate"] - out["mean_pred_prob"]
    out["bin"] = out["bin"].astype(str)
    return out


def normalize_within_race(pred_df: pd.DataFrame, prob_col: str, out_col: str) -> pd.DataFrame:
    out = pred_df.copy()
    race_sum = out.groupby(RACE_KEYS, sort=False, observed=False)[prob_col].transform("sum")
    field_size = out.groupby(RACE_KEYS, sort=False, observed=False)[prob_col].transform("size")

    base_prob = np.where(race_sum > 0, out[prob_col] / race_sum, 1.0 / field_size)
    out[out_col] = np.clip(base_prob.astype(float), EPS, 1.0 - EPS)
    return out


def fit_calibrators(valid_df: pd.DataFrame) -> dict[str, Any]:
    p_valid = valid_df["pred_win_prob"].astype(float).to_numpy()
    y_valid = valid_df[TARGET_COL].astype(int).to_numpy()

    identity = IdentityCalibrator().fit(p_valid, y_valid)
    sigmoid = PlattCalibrator().fit(p_valid, y_valid)
    isotonic = IsotonicRegression(out_of_bounds="clip", y_min=EPS, y_max=1.0 - EPS)
    isotonic.fit(np.clip(p_valid, EPS, 1.0 - EPS), y_valid)

    return {
        "raw": identity,
        "sigmoid": sigmoid,
        "isotonic": isotonic,
    }


def apply_calibrators(frame: pd.DataFrame, calibrators: dict[str, Any]) -> pd.DataFrame:
    out = frame.copy()
    p = out["pred_win_prob"].astype(float).to_numpy()

    out["prob_raw"] = np.clip(p, EPS, 1.0 - EPS)
    out["prob_sigmoid"] = calibrators["sigmoid"].predict(p)
    out["prob_isotonic"] = np.clip(calibrators["isotonic"].predict(np.clip(p, EPS, 1.0 - EPS)), EPS, 1.0 - EPS)

    for src_col, dst_col in [
        ("prob_raw", "prob_raw_race_norm"),
        ("prob_sigmoid", "prob_sigmoid_race_norm"),
        ("prob_isotonic", "prob_isotonic_race_norm"),
    ]:
        out = normalize_within_race(out, prob_col=src_col, out_col=dst_col)

    return out


def main() -> None:
    ensure_dirs()

    file_pairs = discover_prediction_file_pairs()
    if not file_pairs:
        raise FileNotFoundError(
            f"No validation/test prediction file pairs found in {MODELS_DIR}. "
            "Run the baseline training scripts first."
        )

    print("Discovered prediction file pairs:")
    for valid_path, test_path, family in file_pairs:
        print(f"- {family}: {valid_path.name} | {test_path.name}")

    registry_rows: list[dict[str, Any]] = []
    metrics_rows: list[dict[str, Any]] = []
    bin_rows: list[pd.DataFrame] = []
    valid_outputs: list[pd.DataFrame] = []
    test_outputs: list[pd.DataFrame] = []
    calibrator_bundle: dict[str, Any] = {}

    for valid_path, test_path, family in file_pairs:
        valid_all = pd.read_csv(valid_path)
        test_all = pd.read_csv(test_path)

        group_cols = [c for c in ["model_name", "solver_name"] if c in valid_all.columns]
        if not group_cols:
            raise ValueError(f"Could not infer model grouping columns from {valid_path.name}")

        if set(group_cols) != set([c for c in group_cols if c in test_all.columns]):
            raise ValueError(f"Grouping columns do not match between {valid_path.name} and {test_path.name}")

        valid_models = valid_all[group_cols].drop_duplicates().reset_index(drop=True)

        for _, model_row in valid_models.iterrows():
            mask_valid = np.ones(len(valid_all), dtype=bool)
            mask_test = np.ones(len(test_all), dtype=bool)
            for col in group_cols:
                mask_valid &= valid_all[col].astype(str).eq(str(model_row[col])).to_numpy()
                mask_test &= test_all[col].astype(str).eq(str(model_row[col])).to_numpy()

            valid_df = valid_all.loc[mask_valid].copy()
            test_df = test_all.loc[mask_test].copy()
            if valid_df.empty or test_df.empty:
                continue

            variant = variant_from_row(model_row, group_cols=group_cols)
            model_key = make_model_key(family, variant)
            print(f"\nCalibrating {model_key}...")

            calibrators = fit_calibrators(valid_df)
            calibrator_bundle[model_key] = calibrators

            valid_scored = apply_calibrators(valid_df, calibrators=calibrators)
            test_scored = apply_calibrators(test_df, calibrators=calibrators)

            for split_name, scored_df, bucket in [
                ("valid", valid_scored, valid_outputs),
                ("test", test_scored, test_outputs),
            ]:
                methods = [
                    ("raw", "prob_raw"),
                    ("raw_race_norm", "prob_raw_race_norm"),
                    ("sigmoid", "prob_sigmoid"),
                    ("sigmoid_race_norm", "prob_sigmoid_race_norm"),
                    ("isotonic", "prob_isotonic"),
                    ("isotonic_race_norm", "prob_isotonic_race_norm"),
                ]

                for method_name, prob_col in methods:
                    metrics = evaluate_predictions(scored_df, prob_col=prob_col)
                    metrics_rows.append(
                        {
                            "model_family": family,
                            "model_variant": variant,
                            "model_key": model_key,
                            "split": split_name,
                            "method": method_name,
                            **metrics,
                        }
                    )

                    bins = build_bin_stats(scored_df, prob_col=prob_col)
                    if not bins.empty:
                        bins["model_family"] = family
                        bins["model_variant"] = variant
                        bins["model_key"] = model_key
                        bins["split"] = split_name
                        bins["method"] = method_name
                        bin_rows.append(bins)

                out_subset = scored_df.copy()
                out_subset["model_family"] = family
                out_subset["model_variant"] = variant
                out_subset["model_key"] = model_key
                bucket.append(out_subset)

            registry_rows.append(
                {
                    "model_family": family,
                    "model_variant": variant,
                    "model_key": model_key,
                    "validation_source_file": valid_path.name,
                    "test_source_file": test_path.name,
                    "row_count_valid": len(valid_df),
                    "row_count_test": len(test_df),
                }
            )

    metrics_df = pd.DataFrame(metrics_rows)
    if metrics_df.empty:
        raise ValueError("No calibration metrics were created.")

    metrics_df = metrics_df.sort_values(["split", "log_loss", "brier_score", "top_pick_win_rate"], ascending=[True, True, True, False]).reset_index(drop=True)

    test_rank = metrics_df[metrics_df["split"] == "test"].copy()
    test_rank["rank_log_loss"] = test_rank["log_loss"].rank(method="dense", ascending=True)
    test_rank["rank_brier_score"] = test_rank["brier_score"].rank(method="dense", ascending=True)
    test_rank["rank_top_pick_win_rate"] = test_rank["top_pick_win_rate"].rank(method="dense", ascending=False)
    test_rank["rank_winner_in_top3_rate"] = test_rank["winner_in_top3_rate"].rank(method="dense", ascending=False)
    test_rank["rank_mae_race_prob_sum_vs_1"] = test_rank["mae_race_prob_sum_vs_1"].rank(method="dense", ascending=True)
    test_rank["rank_mean"] = test_rank[
        [
            "rank_log_loss",
            "rank_brier_score",
            "rank_top_pick_win_rate",
            "rank_winner_in_top3_rate",
            "rank_mae_race_prob_sum_vs_1",
        ]
    ].mean(axis=1)
    test_rank = test_rank.sort_values(["rank_mean", "log_loss", "brier_score"], ascending=[True, True, True]).reset_index(drop=True)

    valid_out_df = pd.concat(valid_outputs, ignore_index=True, sort=False)
    test_out_df = pd.concat(test_outputs, ignore_index=True, sort=False)
    registry_df = pd.DataFrame(registry_rows).sort_values(["model_family", "model_variant"]).reset_index(drop=True)
    bin_df = pd.concat(bin_rows, ignore_index=True, sort=False) if bin_rows else pd.DataFrame()

    metrics_df.to_csv(METRICS_CSV, index=False)
    test_rank.to_csv(TEST_RANKING_CSV, index=False)
    valid_out_df.to_csv(PREDICTIONS_VALID_CSV, index=False)
    test_out_df.to_csv(PREDICTIONS_TEST_CSV, index=False)
    registry_df.to_csv(MODEL_REGISTRY_CSV, index=False)
    if not bin_df.empty:
        bin_df.to_csv(BIN_STATS_CSV, index=False)
    joblib.dump(calibrator_bundle, CALIBRATORS_JOBLIB)

    best_rows = test_rank.head(10).to_dict(orient="records")
    summary = {
        "prediction_file_pairs": [
            {
                "family": family,
                "validation_file": valid_path.name,
                "test_file": test_path.name,
            }
            for valid_path, test_path, family in file_pairs
        ],
        "model_count": int(registry_df["model_key"].nunique()),
        "metric_row_count": len(metrics_df),
        "top_test_rows": best_rows,
    }
    with SUMMARY_JSON.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nTest ranking preview:")
    preview_cols = [
        "model_key",
        "method",
        "log_loss",
        "brier_score",
        "top_pick_win_rate",
        "winner_in_top3_rate",
        "mean_race_prob_sum",
        "mae_race_prob_sum_vs_1",
        "rank_mean",
    ]
    print(test_rank[preview_cols].head(20).to_string(index=False))

    print("\nSaved files:")
    print(f"- {METRICS_CSV}")
    print(f"- {TEST_RANKING_CSV}")
    print(f"- {PREDICTIONS_VALID_CSV}")
    print(f"- {PREDICTIONS_TEST_CSV}")
    print(f"- {MODEL_REGISTRY_CSV}")
    if not bin_df.empty:
        print(f"- {BIN_STATS_CSV}")
    print(f"- {CALIBRATORS_JOBLIB}")
    print(f"- {SUMMARY_JSON}")
    print("\nDone.")


if __name__ == "__main__":
    main()
