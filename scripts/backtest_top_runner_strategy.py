from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = DATA_DIR / "models"

CALIBRATED_TEST_CSV = MODELS_DIR / "calibrated_test_predictions_v2.csv"
CALIBRATED_RANKING_CSV = MODELS_DIR / "calibrated_probability_test_ranking_v2.csv"

SELECTED_MODELS_OUT = MODELS_DIR / "top_runner_selected_models_v2.csv"
TOP_RUNNER_HORSE_LEVEL_OUT = MODELS_DIR / "top_runner_horse_level_v2.csv"
TOP_RUNNER_STRATEGY_SUMMARY_OUT = MODELS_DIR / "top_runner_strategy_summary_v2.csv"
TOP_RUNNER_TEST_RANKING_OUT = MODELS_DIR / "top_runner_test_ranking_v2.csv"
TOP_RUNNER_SUMMARY_JSON_OUT = MODELS_DIR / "top_runner_summary_v2.json"

SHORTLIST = [
    ("catboost::market_aware", "sigmoid_race_norm"),
    ("lightgbm::market_aware", "raw_race_norm"),
    ("logreg_lbfgs::market_aware", "raw_race_norm"),
]

RACE_KEYS = ["race_date", "racecourse", "race_no"]
EPS = 1e-12
STAKE_PER_BET = 1.0


def read_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    test_df = pd.read_csv(CALIBRATED_TEST_CSV, low_memory=False)
    ranking_df = pd.read_csv(CALIBRATED_RANKING_CSV)
    return test_df, ranking_df


def select_models(ranking_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    rank_lookup = ranking_df.set_index(["model_key", "method"]).to_dict("index")

    for model_key, method in SHORTLIST:
        meta = rank_lookup.get((model_key, method), {})
        rows.append(
            {
                "model_key": model_key,
                "method": method,
                "selection_reason": "preferred",
                "rank_mean": meta.get("rank_mean"),
                "log_loss": meta.get("log_loss"),
                "brier_score": meta.get("brier_score"),
                "top_pick_win_rate": meta.get("top_pick_win_rate"),
                "winner_in_top3_rate": meta.get("winner_in_top3_rate"),
            }
        )

    return pd.DataFrame(rows)


def choose_probability_column(method: str) -> str:
    mapping = {
        "raw": "prob_raw",
        "raw_race_norm": "prob_raw_race_norm",
        "sigmoid": "prob_sigmoid",
        "sigmoid_race_norm": "prob_sigmoid_race_norm",
        "isotonic": "prob_isotonic",
        "isotonic_race_norm": "prob_isotonic_race_norm",
    }
    if method not in mapping:
        raise KeyError(f"Unknown method: {method}")
    return mapping[method]


def build_selected_predictions(test_df: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    out_parts: list[pd.DataFrame] = []

    for row in selected.itertuples(index=False):
        model_key = row.model_key
        method = row.method
        prob_col = choose_probability_column(method)

        sub = test_df[test_df["model_key"] == model_key].copy()
        if sub.empty:
            continue
        if prob_col not in sub.columns:
            raise KeyError(f"Probability column not found for {model_key} / {method}: {prob_col}")

        sub["selected_prob"] = pd.to_numeric(sub[prob_col], errors="coerce")
        sub["method"] = method
        sub["model_key"] = model_key
        sub["win_odds"] = pd.to_numeric(sub["win_odds"], errors="coerce")
        sub["target_win"] = pd.to_numeric(sub["target_win"], errors="coerce").fillna(0).astype(int)

        # top pick within each race using the chosen probability column
        sub = sub.sort_values(
            RACE_KEYS + ["selected_prob", "horse"],
            ascending=[True, True, True, False, True],
            kind="mergesort",
        ).reset_index(drop=True)
        sub["model_rank_in_race"] = sub.groupby(RACE_KEYS, sort=False).cumcount() + 1
        top = sub[sub["model_rank_in_race"] == 1].copy()

        # direct EV-style quantities using posted odds
        top["expected_gross_return_per_unit"] = top["selected_prob"] * top["win_odds"]
        top["expected_net_profit_per_unit"] = top["expected_gross_return_per_unit"] - 1.0
        top["realized_net_profit_per_unit"] = np.where(
            top["target_win"].eq(1),
            top["win_odds"] - 1.0,
            -1.0,
        )
        top["stake_per_bet"] = STAKE_PER_BET
        out_parts.append(top)

    if not out_parts:
        return pd.DataFrame()

    return pd.concat(out_parts, ignore_index=True, sort=False)


def summarize_strategy(top_df: pd.DataFrame) -> pd.DataFrame:
    strategies: list[tuple[str, float | None, float | None]] = [
        ("top_pick_all", None, None),
        ("top_pick_ev_ge_0", 0.00, None),
        ("top_pick_ev_ge_0_05", 0.05, None),
        ("top_pick_ev_ge_0_10", 0.10, None),
        ("top_pick_ev_ge_0_odds_le_20", 0.00, 20.0),
        ("top_pick_ev_ge_0_odds_le_12", 0.00, 12.0),
        ("top_pick_all_odds_le_20", None, 20.0),
        ("top_pick_all_odds_le_12", None, 12.0),
    ]

    rows: list[dict[str, Any]] = []

    for (model_key, method), group in top_df.groupby(["model_key", "method"], sort=False):
        for strategy_name, ev_threshold, max_odds in strategies:
            sel = group.copy()
            if ev_threshold is not None:
                sel = sel[sel["expected_net_profit_per_unit"] >= ev_threshold].copy()
            if max_odds is not None:
                sel = sel[sel["win_odds"] <= max_odds].copy()

            if sel.empty:
                rows.append(
                    {
                        "model_key": model_key,
                        "method": method,
                        "strategy_name": strategy_name,
                        "selection_count": 0,
                        "race_count": 0,
                        "hit_rate": np.nan,
                        "avg_win_odds": np.nan,
                        "median_win_odds": np.nan,
                        "avg_selected_prob": np.nan,
                        "avg_expected_net_profit_per_unit": np.nan,
                        "avg_realized_net_profit_per_unit": np.nan,
                        "sum_realized_net_profit_per_unit": 0.0,
                        "roi_pct": np.nan,
                    }
                )
                continue

            avg_realized = float(sel["realized_net_profit_per_unit"].mean())
            rows.append(
                {
                    "model_key": model_key,
                    "method": method,
                    "strategy_name": strategy_name,
                    "selection_count": int(len(sel)),
                    "race_count": int(sel[RACE_KEYS].drop_duplicates().shape[0]),
                    "hit_rate": float(sel["target_win"].mean()),
                    "avg_win_odds": float(sel["win_odds"].mean()),
                    "median_win_odds": float(sel["win_odds"].median()),
                    "avg_selected_prob": float(sel["selected_prob"].mean()),
                    "avg_expected_net_profit_per_unit": float(sel["expected_net_profit_per_unit"].mean()),
                    "avg_realized_net_profit_per_unit": avg_realized,
                    "sum_realized_net_profit_per_unit": float(sel["realized_net_profit_per_unit"].sum()),
                    "roi_pct": avg_realized * 100.0,
                }
            )

    out = pd.DataFrame(rows)
    out["rank_roi"] = out["avg_realized_net_profit_per_unit"].rank(method="dense", ascending=False)
    out["rank_hit_rate"] = out["hit_rate"].rank(method="dense", ascending=False)
    out["rank_sum_profit"] = out["sum_realized_net_profit_per_unit"].rank(method="dense", ascending=False)
    out["rank_mean"] = out[["rank_roi", "rank_hit_rate", "rank_sum_profit"]].mean(axis=1)
    out = out.sort_values(["rank_mean", "model_key", "strategy_name"], kind="mergesort").reset_index(drop=True)
    return out


def build_test_ranking(summary_df: pd.DataFrame) -> pd.DataFrame:
    keep = summary_df.copy()
    keep = keep.sort_values(["rank_mean", "rank_roi", "rank_hit_rate"], kind="mergesort").reset_index(drop=True)
    return keep


def main() -> None:
    test_df, ranking_df = read_inputs()
    selected = select_models(ranking_df)

    print("Selected model/method combinations:")
    for row in selected.itertuples(index=False):
        print(f"- {row.model_key} | {row.method} ({row.selection_reason})")

    top_df = build_selected_predictions(test_df, selected)
    if top_df.empty:
        raise ValueError("No top-runner rows were created.")

    print("\nTop-runner horse-level preview:")
    preview_cols = [
        "model_key",
        "method",
        "race_date",
        "racecourse",
        "race_no",
        "horse",
        "selected_prob",
        "win_odds",
        "expected_net_profit_per_unit",
        "target_win",
        "realized_net_profit_per_unit",
    ]
    print(top_df[preview_cols].head(20).to_string(index=False))

    summary_df = summarize_strategy(top_df)

    print("\nTop-runner strategy summary preview:")
    print(summary_df.head(20).to_string(index=False))

    ranking_out = build_test_ranking(summary_df)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    selected.to_csv(SELECTED_MODELS_OUT, index=False)
    top_df.to_csv(TOP_RUNNER_HORSE_LEVEL_OUT, index=False)
    summary_df.to_csv(TOP_RUNNER_STRATEGY_SUMMARY_OUT, index=False)
    ranking_out.to_csv(TOP_RUNNER_TEST_RANKING_OUT, index=False)

    summary_payload = {
        "selected_models": selected.to_dict(orient="records"),
        "best_strategy": ranking_out.head(1).to_dict(orient="records"),
        "file_outputs": [
            str(SELECTED_MODELS_OUT),
            str(TOP_RUNNER_HORSE_LEVEL_OUT),
            str(TOP_RUNNER_STRATEGY_SUMMARY_OUT),
            str(TOP_RUNNER_TEST_RANKING_OUT),
        ],
    }
    with TOP_RUNNER_SUMMARY_JSON_OUT.open("w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)

    print("\nSaved files:")
    print(f"- {SELECTED_MODELS_OUT}")
    print(f"- {TOP_RUNNER_HORSE_LEVEL_OUT}")
    print(f"- {TOP_RUNNER_STRATEGY_SUMMARY_OUT}")
    print(f"- {TOP_RUNNER_TEST_RANKING_OUT}")
    print(f"- {TOP_RUNNER_SUMMARY_JSON_OUT}")

    print("\nDone.")


if __name__ == "__main__":
    main()
