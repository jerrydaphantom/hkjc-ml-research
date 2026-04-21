from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = DATA_DIR / "models"

OUTPUT_COMPARISON_CSV = MODELS_DIR / "baseline_model_comparison_v2.csv"
OUTPUT_TEST_RANK_CSV = MODELS_DIR / "baseline_model_test_ranking_v2.csv"
OUTPUT_SUMMARY_JSON = MODELS_DIR / "baseline_model_comparison_summary_v2.json"

SOURCE_FILES = [
    {
        "path": MODELS_DIR / "baseline_metrics_v2.csv",
        "family": "logreg_lbfgs",
        "variant_builder": lambda row: row["model_name"],
    },
    {
        "path": MODELS_DIR / "baseline_metrics_v3.csv",
        "family": "logreg_stable",
        "variant_builder": lambda row: row["model_name"],
    },
    {
        "path": MODELS_DIR / "baseline_metrics_alt_solver_v2.csv",
        "family": "logreg_alt_solver",
        "variant_builder": lambda row: f"{row['model_name']}__{row['solver_name']}",
    },
    {
        "path": MODELS_DIR / "baseline_metrics_lgbm_v2.csv",
        "family": "lightgbm",
        "variant_builder": lambda row: row["model_name"],
    },
    {
        "path": MODELS_DIR / "baseline_metrics_catboost_v2.csv",
        "family": "catboost",
        "variant_builder": lambda row: row["model_name"],
    },
    {
        "path": MODELS_DIR / "baseline_metrics_rf_v2.csv",
        "family": "random_forest",
        "variant_builder": lambda row: row["model_name"],
    },
]


METRIC_COLS = [
    "log_loss",
    "brier_score",
    "top_pick_win_rate",
    "winner_in_top3_rate",
]


def read_metrics() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    for spec in SOURCE_FILES:
        path = spec["path"]
        if not path.exists():
            continue

        df = pd.read_csv(path)
        if df.empty:
            continue

        df["source_file"] = path.name
        df["model_family"] = spec["family"]
        df["model_variant"] = df.apply(spec["variant_builder"], axis=1)
        df["model_key"] = df["model_family"] + "::" + df["model_variant"]
        frames.append(df)

    if not frames:
        raise FileNotFoundError("No baseline metrics files found under data/models/.")

    combined = pd.concat(frames, ignore_index=True, sort=False)
    return combined


def add_rank_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["rank_log_loss"] = out.groupby("split")["log_loss"].rank(method="dense", ascending=True)
    out["rank_brier_score"] = out.groupby("split")["brier_score"].rank(method="dense", ascending=True)
    out["rank_top_pick_win_rate"] = out.groupby("split")["top_pick_win_rate"].rank(method="dense", ascending=False)
    out["rank_winner_in_top3_rate"] = out.groupby("split")["winner_in_top3_rate"].rank(method="dense", ascending=False)

    out["rank_mean"] = out[
        [
            "rank_log_loss",
            "rank_brier_score",
            "rank_top_pick_win_rate",
            "rank_winner_in_top3_rate",
        ]
    ].mean(axis=1)
    return out


def build_test_ranking(df: pd.DataFrame) -> pd.DataFrame:
    test_df = df[df["split"] == "test"].copy()
    if test_df.empty:
        return pd.DataFrame()

    keep_cols = [
        "model_family",
        "model_variant",
        "model_key",
        "source_file",
        "log_loss",
        "brier_score",
        "top_pick_win_rate",
        "winner_in_top3_rate",
        "rank_log_loss",
        "rank_brier_score",
        "rank_top_pick_win_rate",
        "rank_winner_in_top3_rate",
        "rank_mean",
    ]
    test_df = test_df[keep_cols].sort_values(
        ["rank_mean", "rank_log_loss", "rank_brier_score", "model_key"],
        ascending=[True, True, True, True],
    )
    return test_df.reset_index(drop=True)


def build_summary(df: pd.DataFrame, test_ranking: pd.DataFrame) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "files_used": sorted(df["source_file"].dropna().unique().tolist()),
        "splits_present": sorted(df["split"].dropna().unique().tolist()),
        "model_keys": sorted(df["model_key"].dropna().unique().tolist()),
    }

    best_by_metric: dict[str, Any] = {}
    test_df = df[df["split"] == "test"].copy()
    if not test_df.empty:
        best_by_metric["best_log_loss"] = test_df.sort_values(["log_loss", "model_key"]).iloc[0][["model_key", "log_loss"]].to_dict()
        best_by_metric["best_brier_score"] = test_df.sort_values(["brier_score", "model_key"]).iloc[0][["model_key", "brier_score"]].to_dict()
        best_by_metric["best_top_pick_win_rate"] = test_df.sort_values(["top_pick_win_rate", "model_key"], ascending=[False, True]).iloc[0][["model_key", "top_pick_win_rate"]].to_dict()
        best_by_metric["best_winner_in_top3_rate"] = test_df.sort_values(["winner_in_top3_rate", "model_key"], ascending=[False, True]).iloc[0][["model_key", "winner_in_top3_rate"]].to_dict()
    summary["best_test_by_metric"] = best_by_metric

    if not test_ranking.empty:
        summary["top_3_test_models_by_rank_mean"] = test_ranking.head(3)[
            ["model_key", "rank_mean", "log_loss", "brier_score", "top_pick_win_rate", "winner_in_top3_rate"]
        ].to_dict(orient="records")

    return summary


def main() -> None:
    if not MODELS_DIR.exists():
        raise FileNotFoundError(f"Models directory not found: {MODELS_DIR}")

    combined = read_metrics()
    combined = add_rank_columns(combined)
    combined.to_csv(OUTPUT_COMPARISON_CSV, index=False)

    test_ranking = build_test_ranking(combined)
    test_ranking.to_csv(OUTPUT_TEST_RANK_CSV, index=False)

    summary = build_summary(combined, test_ranking)
    with OUTPUT_SUMMARY_JSON.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Loaded metric files:")
    for name in summary["files_used"]:
        print(f"- {name}")

    print("\nCombined comparison preview:")
    print(combined.head(20).to_string(index=False))

    print("\nTest ranking:")
    if test_ranking.empty:
        print("No test rows found.")
    else:
        print(test_ranking.to_string(index=False))

    print("\nSaved files:")
    print(f"- {OUTPUT_COMPARISON_CSV}")
    print(f"- {OUTPUT_TEST_RANK_CSV}")
    print(f"- {OUTPUT_SUMMARY_JSON}")

    print("\nDone.")


if __name__ == "__main__":
    main()
