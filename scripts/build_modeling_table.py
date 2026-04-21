from __future__ import annotations

import re
import sqlite3
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
INTERIM_DIR = DATA_DIR / "interim"
DB_PATH = DATA_DIR / "hkjc_races_v2.db"

# -----------------------------
# OUTPUT CONFIG
# -----------------------------
WRITE_SQLITE_TABLES = True
WRITE_CSV_FILES = True

MODELING_TABLE_NAME = "modeling_table_v2"
MODELING_EXCLUDED_TABLE_NAME = "modeling_table_v2_excluded"
RACE_INFO_WIDE_TABLE_NAME = "race_info_wide_v2"
MODELING_SUMMARY_TABLE_NAME = "modeling_table_v2_summary"

MODELING_CSV_PATH = INTERIM_DIR / "modeling_table_v2.csv"
MODELING_EXCLUDED_CSV_PATH = INTERIM_DIR / "modeling_table_v2_excluded.csv"
RACE_INFO_WIDE_CSV_PATH = INTERIM_DIR / "race_info_wide_v2.csv"
MODELING_SUMMARY_CSV_PATH = INTERIM_DIR / "modeling_table_v2_summary.csv"


def table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    sql = """
    SELECT 1
    FROM sqlite_master
    WHERE type = 'table'
      AND name = ?
    LIMIT 1;
    """
    row = conn.execute(sql, (table_name,)).fetchone()
    return row is not None


def read_table(conn: sqlite3.Connection, table_name: str) -> pd.DataFrame:
    return pd.read_sql_query(f"SELECT * FROM {table_name}", conn)


def to_numeric_nullable(series: pd.Series, as_int: bool = False) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    if as_int:
        return s.astype("Int64")
    return s


def parse_finish_time_to_seconds(value: object) -> float | None:
    """
    Convert finish time strings like:
    - 1:48.95
    - 58.34
    into seconds.
    """
    if pd.isna(value):
        return None

    text = str(value).strip()
    if not text:
        return None

    try:
        if ":" in text:
            mins, secs = text.split(":", 1)
            return int(mins) * 60 + float(secs)
        return float(text)
    except Exception:
        return None


def parse_money_to_number(value: object) -> int | None:
    """
    Convert strings like:
    - HK$ 1,170,000
    into integer 1170000
    """
    if pd.isna(value):
        return None

    text = str(value).strip()
    if not text:
        return None

    text = text.replace("HK$", "").replace(",", "").strip()
    if not text:
        return None

    try:
        return int(float(text))
    except Exception:
        return None


def extract_horse_name(value: object) -> str | None:
    """
    Extract horse name without the code in brackets.

    Example:
    SPICY GOLD (H440) -> SPICY GOLD
    """
    if pd.isna(value):
        return None

    text = str(value).strip()
    if not text:
        return None

    match = re.match(r"^(.*?)\s*\([A-Z0-9]+\)\s*$", text)
    if match:
        return match.group(1).strip()

    return text


def extract_horse_code(value: object) -> str | None:
    """
    Extract horse code in brackets.

    Example:
    SPICY GOLD (H440) -> H440
    """
    if pd.isna(value):
        return None

    text = str(value).strip()
    if not text:
        return None

    match = re.search(r"\(([A-Z0-9]+)\)\s*$", text)
    if match:
        return match.group(1).strip()

    return None


def build_race_info_wide(race_info: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot race_info key-value rows into one row per race.
    """
    keep_cols = ["race_date", "racecourse", "race_no", "field", "value"]
    df = race_info[keep_cols].copy()

    df["field"] = df["field"].astype("string").str.strip()
    df["value"] = df["value"].astype("string").str.strip()

    wide = (
        df.pivot_table(
            index=["race_date", "racecourse", "race_no"],
            columns="field",
            values="value",
            aggfunc="first",
        )
        .reset_index()
    )

    # Flatten pivoted column names and prefix race-info fields
    renamed_cols: list[str] = []
    for col in wide.columns:
        if col in {"race_date", "racecourse", "race_no"}:
            renamed_cols.append(col)
        else:
            renamed_cols.append(f"race_{col}")
    wide.columns = renamed_cols

    # -----------------------------
    # Derived race-level fields
    # -----------------------------
    if "race_class_distance_band" in wide.columns:
        wide["race_distance_m"] = (
            wide["race_class_distance_band"]
            .astype("string")
            .str.extract(r"(\d+)\s*M", expand=False)
        )
        wide["race_distance_m"] = to_numeric_nullable(wide["race_distance_m"], as_int=True)

        wide["race_rating_band"] = (
            wide["race_class_distance_band"]
            .astype("string")
            .str.extract(r"\(([^()]*)\)\s*$", expand=False)
        )

        wide["race_class_label"] = (
            wide["race_class_distance_band"]
            .astype("string")
            .str.extract(r"^(.*?)\s*-\s*\d+\s*M", expand=False)
        )

        # fallback if regex did not match
        missing_class_mask = wide["race_class_label"].isna()
        wide.loc[missing_class_mask, "race_class_label"] = wide.loc[
            missing_class_mask, "race_class_distance_band"
        ]

    else:
        wide["race_distance_m"] = pd.Series(dtype="Int64")
        wide["race_rating_band"] = pd.Series(dtype="string")
        wide["race_class_label"] = pd.Series(dtype="string")

    if "race_course" in wide.columns:
        course_text = wide["race_course"].astype("string")

        wide["race_surface"] = course_text.str.extract(
            r"^(TURF|ALL WEATHER TRACK|AWT)",
            expand=False,
        )

        wide["race_rail_code"] = course_text.str.extract(
            r'"([^"]+)"',
            expand=False,
        )

    else:
        wide["race_surface"] = pd.Series(dtype="string")
        wide["race_rail_code"] = pd.Series(dtype="string")

    if "race_prize_money" in wide.columns:
        wide["race_prize_money_num"] = wide["race_prize_money"].apply(parse_money_to_number)
        wide["race_prize_money_num"] = to_numeric_nullable(wide["race_prize_money_num"], as_int=True)
    else:
        wide["race_prize_money_num"] = pd.Series(dtype="Int64")

    wide["race_date_dt"] = pd.to_datetime(wide["race_date"], errors="coerce")
    wide["race_year"] = wide["race_date_dt"].dt.year.astype("Int64")
    wide["race_month"] = wide["race_date_dt"].dt.month.astype("Int64")
    wide["race_day"] = wide["race_date_dt"].dt.day.astype("Int64")

    return wide


def build_base_runner_results(runner_results: pd.DataFrame) -> pd.DataFrame:
    """
    Clean runner_results columns for modeling-table construction.
    """
    df = runner_results.copy()

    # Numeric cleanup
    int_like_cols = ["horse_no", "act_wt", "declar_horse_wt", "dr", "pla_num", "dead_heat"]
    for col in int_like_cols:
        if col in df.columns:
            df[col] = to_numeric_nullable(df[col], as_int=True)

    if "win_odds" in df.columns:
        df["win_odds"] = to_numeric_nullable(df["win_odds"], as_int=False)

    # Horse-name derived fields
    df["horse_name"] = df["horse"].apply(extract_horse_name)
    df["horse_code"] = df["horse"].apply(extract_horse_code)

    # Finish-time numeric version
    df["finish_time_seconds"] = df["finish_time"].apply(parse_finish_time_to_seconds)

    # Outcome / target columns
    df["target_finished"] = (df["result_status"] == "FINISHED").astype(int)
    df["target_win"] = ((df["result_status"] == "FINISHED") & (df["pla_num"] == 1)).astype(int)
    df["target_place_top2"] = ((df["result_status"] == "FINISHED") & (df["pla_num"] <= 2)).astype(int)
    df["target_place_top3"] = ((df["result_status"] == "FINISHED") & (df["pla_num"] <= 3)).astype(int)
    df["target_place_top4"] = ((df["result_status"] == "FINISHED") & (df["pla_num"] <= 4)).astype(int)

    # Clearer outcome column naming
    df["outcome_place_raw"] = df["pla_raw"]
    df["outcome_place_num"] = df["pla_num"]
    df["outcome_dead_heat"] = df["dead_heat"]
    df["outcome_result_status"] = df["result_status"]
    df["outcome_finish_time"] = df["finish_time"]
    df["outcome_finish_time_seconds"] = df["finish_time_seconds"]
    df["outcome_lbw"] = df["lbw"]
    df["outcome_running_position"] = df["running_position"]

    return df


def join_race_info(
    runners: pd.DataFrame,
    race_info_wide: pd.DataFrame,
) -> pd.DataFrame:
    """
    Join one-row-per-race race info onto one-row-per-horse runner rows.
    """
    merged = runners.merge(
        race_info_wide,
        on=["race_date", "racecourse", "race_no"],
        how="left",
        validate="many_to_one",
    )
    return merged


def build_modeling_outputs(
    runner_results: pd.DataFrame,
    excluded_runner_results: pd.DataFrame,
    race_info_wide: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build:
    1. main modeling table with FINISHED rows only
    2. excluded modeling table with non-FINISHED rows and separately excluded rows
    """
    rr = build_base_runner_results(runner_results)

    rr_joined = join_race_info(rr, race_info_wide)

    # Main modeling table: FINISHED rows only
    modeling_table = rr_joined[rr_joined["result_status"] == "FINISHED"].copy()
    modeling_table["modeling_row_source"] = "runner_results_finished"
    modeling_table["modeling_exclusion_reason"] = pd.NA

    # Non-finished rows from main runner_results table
    excluded_nonfinished = rr_joined[rr_joined["result_status"] != "FINISHED"].copy()
    excluded_nonfinished["modeling_row_source"] = "runner_results_nonfinished"
    excluded_nonfinished["modeling_exclusion_reason"] = (
        "Excluded from main modeling table because result_status != FINISHED."
    )

    excluded_parts = [excluded_nonfinished]

    # Excluded rows from v2 rebuild (mostly old MISSING rows)
    if not excluded_runner_results.empty:
        ex = build_base_runner_results(excluded_runner_results.copy())
        ex_joined = join_race_info(ex, race_info_wide)
        ex_joined["modeling_row_source"] = "excluded_runner_results"
        if "exclusion_reason" in ex_joined.columns:
            ex_joined["modeling_exclusion_reason"] = ex_joined["exclusion_reason"]
        else:
            ex_joined["modeling_exclusion_reason"] = (
                "Excluded earlier in v2 rebuild."
            )
        excluded_parts.append(ex_joined)

    modeling_excluded = pd.concat(excluded_parts, ignore_index=True, sort=False)

    # Sort outputs
    sort_cols = ["race_date", "racecourse", "race_no", "horse"]
    modeling_table = modeling_table.sort_values(sort_cols).reset_index(drop=True)
    modeling_excluded = modeling_excluded.sort_values(sort_cols).reset_index(drop=True)

    return modeling_table, modeling_excluded


def build_summary(
    modeling_table: pd.DataFrame,
    modeling_excluded: pd.DataFrame,
    race_info_wide: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a compact summary table for the modeling build.
    """
    summary_rows = [
        {"metric": "modeling_rows", "value": len(modeling_table)},
        {"metric": "excluded_rows", "value": len(modeling_excluded)},
        {
            "metric": "distinct_races_in_modeling_table",
            "value": modeling_table[["race_date", "racecourse", "race_no"]].drop_duplicates().shape[0],
        },
        {"metric": "distinct_horses_in_modeling_table", "value": modeling_table["horse"].nunique()},
        {"metric": "distinct_jockeys_in_modeling_table", "value": modeling_table["jockey"].nunique()},
        {"metric": "distinct_trainers_in_modeling_table", "value": modeling_table["trainer"].nunique()},
        {"metric": "race_info_wide_rows", "value": len(race_info_wide)},
        {"metric": "modeling_columns", "value": modeling_table.shape[1]},
        {"metric": "excluded_columns", "value": modeling_excluded.shape[1]},
        {"metric": "target_win_sum", "value": int(modeling_table["target_win"].sum())},
        {"metric": "target_place_top3_sum", "value": int(modeling_table["target_place_top3"].sum())},
        {"metric": "dead_heat_rows_in_modeling_table", "value": int(modeling_table["outcome_dead_heat"].sum())},
    ]
    return pd.DataFrame(summary_rows)


def write_outputs_to_sqlite(
    conn: sqlite3.Connection,
    race_info_wide: pd.DataFrame,
    modeling_table: pd.DataFrame,
    modeling_excluded: pd.DataFrame,
    summary_df: pd.DataFrame,
) -> None:
    race_info_wide.to_sql(RACE_INFO_WIDE_TABLE_NAME, conn, index=False, if_exists="replace")
    modeling_table.to_sql(MODELING_TABLE_NAME, conn, index=False, if_exists="replace")
    modeling_excluded.to_sql(MODELING_EXCLUDED_TABLE_NAME, conn, index=False, if_exists="replace")
    summary_df.to_sql(MODELING_SUMMARY_TABLE_NAME, conn, index=False, if_exists="replace")


def write_outputs_to_csv(
    race_info_wide: pd.DataFrame,
    modeling_table: pd.DataFrame,
    modeling_excluded: pd.DataFrame,
    summary_df: pd.DataFrame,
) -> None:
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)

    race_info_wide.to_csv(RACE_INFO_WIDE_CSV_PATH, index=False)
    modeling_table.to_csv(MODELING_CSV_PATH, index=False)
    modeling_excluded.to_csv(MODELING_EXCLUDED_CSV_PATH, index=False)
    summary_df.to_csv(MODELING_SUMMARY_CSV_PATH, index=False)


def main() -> None:
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Database not found: {DB_PATH}")

    with sqlite3.connect(DB_PATH) as conn:
        runner_results = read_table(conn, "runner_results")
        race_info = read_table(conn, "race_info")

        if table_exists(conn, "excluded_runner_results"):
            excluded_runner_results = read_table(conn, "excluded_runner_results")
        else:
            excluded_runner_results = pd.DataFrame()

        race_info_wide = build_race_info_wide(race_info)
        modeling_table, modeling_excluded = build_modeling_outputs(
            runner_results=runner_results,
            excluded_runner_results=excluded_runner_results,
            race_info_wide=race_info_wide,
        )
        summary_df = build_summary(
            modeling_table=modeling_table,
            modeling_excluded=modeling_excluded,
            race_info_wide=race_info_wide,
        )

        if WRITE_SQLITE_TABLES:
            write_outputs_to_sqlite(
                conn=conn,
                race_info_wide=race_info_wide,
                modeling_table=modeling_table,
                modeling_excluded=modeling_excluded,
                summary_df=summary_df,
            )

    if WRITE_CSV_FILES:
        write_outputs_to_csv(
            race_info_wide=race_info_wide,
            modeling_table=modeling_table,
            modeling_excluded=modeling_excluded,
            summary_df=summary_df,
        )

    print("Built modeling outputs from:")
    print(DB_PATH)
    print("\nSaved SQLite tables:")
    print(f"- {RACE_INFO_WIDE_TABLE_NAME}")
    print(f"- {MODELING_TABLE_NAME}")
    print(f"- {MODELING_EXCLUDED_TABLE_NAME}")
    print(f"- {MODELING_SUMMARY_TABLE_NAME}")

    if WRITE_CSV_FILES:
        print("\nSaved CSV files:")
        print(f"- {RACE_INFO_WIDE_CSV_PATH}")
        print(f"- {MODELING_CSV_PATH}")
        print(f"- {MODELING_EXCLUDED_CSV_PATH}")
        print(f"- {MODELING_SUMMARY_CSV_PATH}")

    print("\nModeling summary:")
    print(summary_df.to_string(index=False))

    print("\nDone.")


if __name__ == "__main__":
    main()