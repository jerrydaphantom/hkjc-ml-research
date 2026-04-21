from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = DATA_DIR / "models"
FEATURE_DB_PATH = DATA_DIR / "hkjc_features_v2.db"
FEATURE_TABLE_MARKET_AWARE = "model_features_v2_market_aware"

CALIBRATED_TEST_CSV = MODELS_DIR / "calibrated_test_predictions_v2.csv"
CALIBRATED_RANKING_CSV = MODELS_DIR / "calibrated_probability_test_ranking_v2.csv"

HORSE_LEVEL_CSV = MODELS_DIR / "model_market_comparison_horse_level_v2b.csv"
CONTEXT_SUMMARY_CSV = MODELS_DIR / "model_market_context_summary_v2b.csv"
RACE_SUMMARY_CSV = MODELS_DIR / "model_market_race_summary_v2b.csv"
SELECTED_MODELS_CSV = MODELS_DIR / "model_market_selected_models_v2b.csv"
OVERLAY_CANDIDATES_CSV = MODELS_DIR / "model_market_overlay_candidates_v2b.csv"
CONSENSUS_CANDIDATES_CSV = MODELS_DIR / "model_market_consensus_candidates_v2b.csv"
THRESHOLD_SUMMARY_CSV = MODELS_DIR / "model_market_threshold_summary_v2b.csv"
SUMMARY_JSON = MODELS_DIR / "model_market_summary_v2b.json"

RACE_KEYS = ["race_date", "racecourse", "race_no"]
HORSE_KEYS = RACE_KEYS + ["horse"]
EPS = 1e-9

# Prefer the current champion + closest challenger + transparent benchmark.
PREFERRED_SELECTIONS: list[tuple[str, str]] = [
    ("catboost::market_aware", "sigmoid_race_norm"),
    ("lightgbm::market_aware", "raw_race_norm"),
    ("logreg_lbfgs::market_aware", "raw_race_norm"),
]

# These are research thresholds, not a final strategy.
EV_THRESHOLDS = [0.00, 0.02, 0.05, 0.10]
PROB_DIFF_THRESHOLDS = [0.00, 0.01, 0.02, 0.03]
TOP_N_PER_MODEL = 500


def ensure_dirs() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)


def odds_to_market_strength(odds: pd.Series) -> pd.Series:
    out = pd.to_numeric(odds, errors="coerce")
    out = pd.Series(np.where(out > 0, 1.0 / out, np.nan), index=odds.index, dtype=float)
    return out


def normalize_within_race(df: pd.DataFrame, value_col: str, out_col: str) -> pd.DataFrame:
    out = df.copy()
    race_sum = out.groupby(RACE_KEYS, sort=False, observed=False)[value_col].transform("sum")
    race_n = out.groupby(RACE_KEYS, sort=False, observed=False)[value_col].transform("size")

    normalized = np.where(race_sum > 0, out[value_col] / race_sum, 1.0 / race_n)
    out[out_col] = normalized.astype(float)
    return out


def assign_odds_band(odds: pd.Series) -> pd.Series:
    x = pd.to_numeric(odds, errors="coerce")
    bins = [-np.inf, 3.0, 5.0, 10.0, 20.0, np.inf]
    labels = ["<=3", "3-5", "5-10", "10-20", ">20"]
    return pd.cut(x, bins=bins, labels=labels, include_lowest=True)


def assign_field_size_band(field_size: pd.Series) -> pd.Series:
    x = pd.to_numeric(field_size, errors="coerce")
    bins = [-np.inf, 8, 10, 12, 14, np.inf]
    labels = ["<=8", "9-10", "11-12", "13-14", ">=15"]
    return pd.cut(x, bins=bins, labels=labels, include_lowest=True)


def choose_models(ranking_df: pd.DataFrame) -> pd.DataFrame:
    available = ranking_df[["model_key", "method"]].drop_duplicates().copy()

    preferred_rows = []
    for model_key, method in PREFERRED_SELECTIONS:
        mask = available["model_key"].eq(model_key) & available["method"].eq(method)
        if mask.any():
            preferred_rows.append({"model_key": model_key, "method": method, "selection_reason": "preferred"})

    if preferred_rows:
        return pd.DataFrame(preferred_rows)

    fallback = ranking_df.copy()
    fallback = fallback[fallback["model_key"].astype(str).str.contains("market_aware", na=False)]
    fallback = fallback[fallback["method"].astype(str).str.contains("race_norm", na=False)]
    fallback = fallback.sort_values(["rank_mean", "log_loss", "brier_score"], ascending=[True, True, True])
    fallback = fallback[["model_key", "method"]].drop_duplicates().head(3).copy()
    fallback["selection_reason"] = "ranking_fallback"
    return fallback.reset_index(drop=True)


def enrich_with_feature_context(df: pd.DataFrame) -> pd.DataFrame:
    if not FEATURE_DB_PATH.exists():
        return df

    keep_cols = [
        "race_date",
        "racecourse",
        "race_no",
        "horse",
        "race_distance_m",
        "race_surface",
        "race_going",
        "field_size",
    ]

    with sqlite3.connect(FEATURE_DB_PATH) as conn:
        feat = pd.read_sql_query(
            f"SELECT {', '.join(keep_cols)} FROM {FEATURE_TABLE_MARKET_AWARE}",
            conn,
        )

    feat = feat.drop_duplicates(subset=HORSE_KEYS)
    merged = df.merge(feat, on=HORSE_KEYS, how="left", suffixes=("", "_feat"))

    if "field_size" not in merged.columns or merged["field_size"].isna().all():
        merged["field_size"] = merged.groupby(RACE_KEYS, sort=False, observed=False)["horse"].transform("size")

    return merged


def build_horse_level(selected_df: pd.DataFrame) -> pd.DataFrame:
    out = selected_df.copy()
    out = enrich_with_feature_context(out)

    out["selected_prob"] = pd.to_numeric(out["selected_prob"], errors="coerce").clip(EPS, 1.0 - EPS)
    out["win_odds"] = pd.to_numeric(out["win_odds"], errors="coerce")
    out = out[out["win_odds"].notna() & (out["win_odds"] > 0)].copy()

    out["market_strength_raw"] = odds_to_market_strength(out["win_odds"])
    out = normalize_within_race(out, value_col="market_strength_raw", out_col="market_prob_norm")

    overround = out.groupby(RACE_KEYS, sort=False, observed=False)["market_strength_raw"].transform("sum")
    out["market_overround"] = overround
    out["market_payout_fraction_implied"] = np.where(overround > 0, 1.0 / overround, np.nan)

    out["prob_minus_market"] = out["selected_prob"] - out["market_prob_norm"]
    out["prob_over_market_ratio"] = np.where(out["market_prob_norm"] > 0, out["selected_prob"] / out["market_prob_norm"], np.nan)

    # Direct EV-style research columns per 1 unit stake.
    out["expected_gross_return_multiple"] = out["selected_prob"] * out["win_odds"]
    out["expected_net_profit_per_unit"] = out["expected_gross_return_multiple"] - 1.0
    out["realized_net_profit_per_unit"] = np.where(out["target_win"].astype(int) == 1, out["win_odds"] - 1.0, -1.0)

    # Rank comparisons.
    out = out.sort_values(RACE_KEYS + ["market_prob_norm", "horse"], ascending=[True, True, True, False, True], kind="mergesort")
    out["market_rank_in_race"] = out.groupby(RACE_KEYS, sort=False, observed=False).cumcount() + 1

    out = out.sort_values(RACE_KEYS + ["selected_prob", "horse"], ascending=[True, True, True, False, True], kind="mergesort")
    out["model_rank_in_race"] = out.groupby(["model_key", "method"] + RACE_KEYS, sort=False, observed=False).cumcount() + 1
    out["rank_advantage_vs_market"] = out["market_rank_in_race"] - out["model_rank_in_race"]

    out["odds_band"] = assign_odds_band(out["win_odds"]).astype("string")
    out["field_size_band"] = assign_field_size_band(out["field_size"]).astype("string") if "field_size" in out.columns else pd.Series(dtype="string")

    return out.sort_values(["model_key", "method"] + RACE_KEYS + ["model_rank_in_race", "horse"]).reset_index(drop=True)


def summarize_by_context(horse_df: pd.DataFrame) -> pd.DataFrame:
    group_sets = [
        ["model_key", "method", "odds_band"],
        ["model_key", "method", "field_size_band"],
        ["model_key", "method", "racecourse"],
    ]

    if "race_surface" in horse_df.columns:
        group_sets.append(["model_key", "method", "race_surface"])
    if "race_going" in horse_df.columns:
        group_sets.append(["model_key", "method", "race_going"])
    if "race_distance_m" in horse_df.columns:
        distance_df = horse_df.copy()
        distance_df["distance_band"] = pd.cut(
            pd.to_numeric(distance_df["race_distance_m"], errors="coerce"),
            bins=[-np.inf, 1200, 1400, 1650, 2000, np.inf],
            labels=["<=1200", "1201-1400", "1401-1650", "1651-2000", ">2000"],
            include_lowest=True,
        ).astype("string")
        horse_df = distance_df
        group_sets.append(["model_key", "method", "distance_band"])

    pieces: list[pd.DataFrame] = []
    for group_cols in group_sets:
        context_col = group_cols[-1]
        sub = horse_df.dropna(subset=[context_col]).copy()
        if sub.empty:
            continue

        agg = (
            sub.groupby(group_cols, observed=False)
            .agg(
                row_count=("horse", "size"),
                race_count=("race_no", "nunique"),
                hit_rate=("target_win", "mean"),
                avg_model_prob=("selected_prob", "mean"),
                avg_market_prob=("market_prob_norm", "mean"),
                avg_prob_minus_market=("prob_minus_market", "mean"),
                avg_prob_over_market_ratio=("prob_over_market_ratio", "mean"),
                avg_expected_net_profit_per_unit=("expected_net_profit_per_unit", "mean"),
                avg_realized_net_profit_per_unit=("realized_net_profit_per_unit", "mean"),
            )
            .reset_index()
        )
        agg["context_type"] = context_col
        agg["context_value"] = agg[context_col].astype(str)
        pieces.append(agg)

    if not pieces:
        return pd.DataFrame()

    out = pd.concat(pieces, ignore_index=True, sort=False)
    front_cols = [
        "model_key",
        "method",
        "context_type",
        "context_value",
        "row_count",
        "race_count",
        "hit_rate",
        "avg_model_prob",
        "avg_market_prob",
        "avg_prob_minus_market",
        "avg_prob_over_market_ratio",
        "avg_expected_net_profit_per_unit",
        "avg_realized_net_profit_per_unit",
    ]
    keep_cols = front_cols + [c for c in out.columns if c not in front_cols]
    return out[keep_cols].sort_values(["model_key", "method", "context_type", "context_value"]).reset_index(drop=True)


def summarize_by_race(horse_df: pd.DataFrame) -> pd.DataFrame:
    agg = (
        horse_df.groupby(["model_key", "method"] + RACE_KEYS, observed=False)
        .agg(
            race_market_overround=("market_overround", "first"),
            race_payout_fraction_implied=("market_payout_fraction_implied", "first"),
            model_top_pick=("horse", "first"),
            model_top_pick_prob=("selected_prob", "first"),
            model_top_pick_odds=("win_odds", "first"),
            model_top_pick_ev_per_unit=("expected_net_profit_per_unit", "first"),
            model_top_pick_hit=("target_win", "first"),
            max_prob_minus_market=("prob_minus_market", "max"),
            mean_prob_minus_market=("prob_minus_market", "mean"),
            max_rank_advantage_vs_market=("rank_advantage_vs_market", "max"),
        )
        .reset_index()
    )
    return agg.sort_values(["model_key", "method"] + RACE_KEYS).reset_index(drop=True)


def build_overlay_candidates(horse_df: pd.DataFrame) -> pd.DataFrame:
    out = horse_df.copy()
    out["overlay_flag_basic"] = (out["expected_net_profit_per_unit"] > 0) & (out["prob_minus_market"] > 0)
    out["overlay_flag_strict"] = (
        (out["expected_net_profit_per_unit"] >= 0.05)
        & (out["prob_minus_market"] >= 0.02)
        & (out["model_rank_in_race"] <= 3)
    )

    overlays = out[out["overlay_flag_basic"]].copy()
    overlays = overlays.sort_values(
        ["model_key", "method", "expected_net_profit_per_unit", "prob_minus_market", "selected_prob"],
        ascending=[True, True, False, False, False],
        kind="mergesort",
    )

    overlays["overlay_rank_within_model"] = overlays.groupby(["model_key", "method"], sort=False, observed=False).cumcount() + 1
    return overlays.head(TOP_N_PER_MODEL * max(1, overlays[["model_key", "method"]].drop_duplicates().shape[0])).reset_index(drop=True)


def build_consensus_candidates(horse_df: pd.DataFrame) -> pd.DataFrame:
    focus = horse_df[horse_df["model_key"].isin(["catboost::market_aware", "lightgbm::market_aware"])].copy()
    if focus.empty:
        return pd.DataFrame()

    keep_cols = [
        "selected_prob",
        "prob_minus_market",
        "expected_net_profit_per_unit",
        "model_rank_in_race",
        "target_win",
        "win_odds",
        "market_prob_norm",
        "realized_net_profit_per_unit",
    ]
    wide = focus[HORSE_KEYS + ["model_key"] + keep_cols].copy()
    wide = wide.pivot_table(index=HORSE_KEYS, columns="model_key", values=keep_cols, aggfunc="first")
    if wide.empty:
        return pd.DataFrame()

    wide.columns = [f"{metric}__{model_key}" for metric, model_key in wide.columns]
    wide = wide.reset_index()

    required = [
        "selected_prob__catboost::market_aware",
        "selected_prob__lightgbm::market_aware",
        "prob_minus_market__catboost::market_aware",
        "prob_minus_market__lightgbm::market_aware",
    ]
    if any(col not in wide.columns for col in required):
        return pd.DataFrame()

    wide["consensus_avg_prob"] = wide[[
        "selected_prob__catboost::market_aware",
        "selected_prob__lightgbm::market_aware",
    ]].mean(axis=1)
    wide["consensus_avg_prob_minus_market"] = wide[[
        "prob_minus_market__catboost::market_aware",
        "prob_minus_market__lightgbm::market_aware",
    ]].mean(axis=1)
    wide["consensus_avg_ev_per_unit"] = wide[[
        "expected_net_profit_per_unit__catboost::market_aware",
        "expected_net_profit_per_unit__lightgbm::market_aware",
    ]].mean(axis=1)
    wide["consensus_agree_positive"] = (
        (wide["prob_minus_market__catboost::market_aware"] > 0)
        & (wide["prob_minus_market__lightgbm::market_aware"] > 0)
        & (wide["expected_net_profit_per_unit__catboost::market_aware"] > 0)
        & (wide["expected_net_profit_per_unit__lightgbm::market_aware"] > 0)
    )

    out = wide[wide["consensus_agree_positive"]].copy()
    if out.empty:
        return out

    out = out.sort_values(
        ["consensus_avg_ev_per_unit", "consensus_avg_prob_minus_market", "consensus_avg_prob"],
        ascending=[False, False, False],
        kind="mergesort",
    ).reset_index(drop=True)
    out["consensus_rank"] = np.arange(1, len(out) + 1)
    return out.head(TOP_N_PER_MODEL)


def build_threshold_summary(horse_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (model_key, method), group in horse_df.groupby(["model_key", "method"], observed=False):
        for ev_thr in EV_THRESHOLDS:
            for diff_thr in PROB_DIFF_THRESHOLDS:
                sel = group[
                    (group["expected_net_profit_per_unit"] >= ev_thr)
                    & (group["prob_minus_market"] >= diff_thr)
                ].copy()
                rows.append(
                    {
                        "model_key": model_key,
                        "method": method,
                        "ev_threshold": ev_thr,
                        "prob_diff_threshold": diff_thr,
                        "selection_count": len(sel),
                        "race_count": sel[RACE_KEYS].drop_duplicates().shape[0],
                        "hit_rate": float(sel["target_win"].mean()) if len(sel) else np.nan,
                        "avg_expected_net_profit_per_unit": float(sel["expected_net_profit_per_unit"].mean()) if len(sel) else np.nan,
                        "avg_realized_net_profit_per_unit": float(sel["realized_net_profit_per_unit"].mean()) if len(sel) else np.nan,
                        "sum_realized_net_profit_per_unit": float(sel["realized_net_profit_per_unit"].sum()) if len(sel) else np.nan,
                    }
                )
    return pd.DataFrame(rows).sort_values(["model_key", "method", "ev_threshold", "prob_diff_threshold"]).reset_index(drop=True)


def main() -> None:
    ensure_dirs()

    if not CALIBRATED_TEST_CSV.exists():
        raise FileNotFoundError(f"Missing file: {CALIBRATED_TEST_CSV}. Run 18_calibrate_probabilities_v2.py first.")
    if not CALIBRATED_RANKING_CSV.exists():
        raise FileNotFoundError(f"Missing file: {CALIBRATED_RANKING_CSV}. Run 18_calibrate_probabilities_v2.py first.")

    ranking_df = pd.read_csv(CALIBRATED_RANKING_CSV)
    selected_models = choose_models(ranking_df)
    if selected_models.empty:
        raise ValueError("Could not find any selected model/method combinations in calibration ranking output.")

    test_df = pd.read_csv(CALIBRATED_TEST_CSV, low_memory=False)

    print("Selected model/method combinations:")
    for _, row in selected_models.iterrows():
        print(f"- {row['model_key']} | {row['method']} ({row['selection_reason']})")

    # The calibrated test predictions are stored in wide format: one row per horse/model,
    # with multiple probability columns (prob_raw, prob_sigmoid, ...), not one row per method.
    # So we select one preferred method per model_key here and then map that method to the
    # corresponding probability column.
    selected_models = selected_models.rename(columns={"method": "selected_method"}).copy()
    merged = test_df.merge(selected_models, on=["model_key"], how="inner")
    if merged.empty:
        raise ValueError("No rows matched the selected models in calibrated test predictions.")

    method_to_prob_col = {
        "raw": "prob_raw",
        "raw_race_norm": "prob_raw_race_norm",
        "sigmoid": "prob_sigmoid",
        "sigmoid_race_norm": "prob_sigmoid_race_norm",
        "isotonic": "prob_isotonic",
        "isotonic_race_norm": "prob_isotonic_race_norm",
    }

    merged["selected_prob"] = np.nan
    for method_name, prob_col in method_to_prob_col.items():
        mask = merged["selected_method"].eq(method_name)
        if mask.any() and prob_col in merged.columns:
            merged.loc[mask, "selected_prob"] = pd.to_numeric(merged.loc[mask, prob_col], errors="coerce")

    merged["method"] = merged["selected_method"]
    merged = merged[merged["selected_prob"].notna()].copy()
    if merged.empty:
        raise ValueError("Selected probabilities were all missing after method lookup.")

    horse_df = build_horse_level(merged)
    context_df = summarize_by_context(horse_df)
    race_df = summarize_by_race(horse_df)
    overlay_df = build_overlay_candidates(horse_df)
    consensus_df = build_consensus_candidates(horse_df)
    threshold_df = build_threshold_summary(horse_df)

    selected_models.to_csv(SELECTED_MODELS_CSV, index=False)
    horse_df.to_csv(HORSE_LEVEL_CSV, index=False)
    if not context_df.empty:
        context_df.to_csv(CONTEXT_SUMMARY_CSV, index=False)
    race_df.to_csv(RACE_SUMMARY_CSV, index=False)
    if not overlay_df.empty:
        overlay_df.to_csv(OVERLAY_CANDIDATES_CSV, index=False)
    if not consensus_df.empty:
        consensus_df.to_csv(CONSENSUS_CANDIDATES_CSV, index=False)
    threshold_df.to_csv(THRESHOLD_SUMMARY_CSV, index=False)

    summary = {
        "selected_models": selected_models.to_dict(orient="records"),
        "horse_level_rows": int(len(horse_df)),
        "selected_model_count": int(selected_models.shape[0]),
        "overlay_candidate_rows": int(len(overlay_df)),
        "consensus_candidate_rows": int(len(consensus_df)),
        "threshold_rows": int(len(threshold_df)),
        "top_overlay_preview": overlay_df[
            [
                "model_key",
                "method",
                "race_date",
                "racecourse",
                "race_no",
                "horse",
                "selected_prob",
                "market_prob_norm",
                "prob_minus_market",
                "expected_net_profit_per_unit",
                "win_odds",
            ]
        ].head(20).to_dict(orient="records") if not overlay_df.empty else [],
        "top_consensus_preview": consensus_df.head(20).to_dict(orient="records") if not consensus_df.empty else [],
    }
    with SUMMARY_JSON.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nHorse-level comparison preview:")
    preview_cols = [
        "model_key",
        "method",
        "race_date",
        "racecourse",
        "race_no",
        "horse",
        "selected_prob",
        "market_prob_norm",
        "prob_minus_market",
        "expected_net_profit_per_unit",
        "win_odds",
        "model_rank_in_race",
        "market_rank_in_race",
        "rank_advantage_vs_market",
    ]
    print(horse_df[preview_cols].head(20).to_string(index=False))

    print("\nThreshold summary preview:")
    th_preview = threshold_df.sort_values(
        ["avg_realized_net_profit_per_unit", "selection_count"],
        ascending=[False, False],
        kind="mergesort",
    ).head(20)
    print(th_preview.to_string(index=False))

    if not consensus_df.empty:
        print("\nConsensus candidate preview:")
        print(consensus_df.head(20).to_string(index=False))

    print("\nSaved files:")
    print(f"- {SELECTED_MODELS_CSV}")
    print(f"- {HORSE_LEVEL_CSV}")
    if not context_df.empty:
        print(f"- {CONTEXT_SUMMARY_CSV}")
    print(f"- {RACE_SUMMARY_CSV}")
    if not overlay_df.empty:
        print(f"- {OVERLAY_CANDIDATES_CSV}")
    if not consensus_df.empty:
        print(f"- {CONSENSUS_CANDIDATES_CSV}")
    print(f"- {THRESHOLD_SUMMARY_CSV}")
    print(f"- {SUMMARY_JSON}")
    print("\nDone.")


if __name__ == "__main__":
    main()
