from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
DB_PATH = BASE_DIR / "data" / "hkjc_features_v2.db"

MARKET_FREE_TABLE = "model_features_v2_market_free"
MARKET_AWARE_TABLE = "model_features_v2_market_aware"
SUMMARY_TABLE = "model_features_v2_summary"

MAX_ROWS = 15


def print_section(title: str) -> None:
    print(f"\n{title}")
    print("=" * len(title))


def show_df(df: pd.DataFrame, max_rows: int = MAX_ROWS) -> None:
    if df.empty:
        print("No rows returned.")
        return
    if len(df) <= max_rows:
        print(df.to_string(index=False))
    else:
        print(df.head(max_rows).to_string(index=False))
        print(f"\n... showing first {max_rows} of {len(df)} rows ...")


def table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        """
        SELECT 1
        FROM sqlite_master
        WHERE type = 'table' AND name = ?
        LIMIT 1
        """,
        (table_name,),
    ).fetchone()
    return row is not None


def get_columns(conn: sqlite3.Connection, table_name: str) -> list[str]:
    pragma = pd.read_sql_query(f"PRAGMA table_info({table_name})", conn)
    return pragma["name"].tolist()


def main() -> None:
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Database not found: {DB_PATH}")

    with sqlite3.connect(DB_PATH) as conn:
        print_section("1. Tables")
        tables = pd.read_sql_query(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
            ORDER BY name
            """,
            conn,
        )
        show_df(tables, max_rows=20)

        print_section("2. Summary table")
        if table_exists(conn, SUMMARY_TABLE):
            summary = pd.read_sql_query(f"SELECT * FROM {SUMMARY_TABLE}", conn)
            show_df(summary, max_rows=50)
        else:
            print("Summary table not found.")

        print_section("3. Row counts")
        counts = []
        for table_name in [MARKET_FREE_TABLE, MARKET_AWARE_TABLE]:
            if table_exists(conn, table_name):
                row_count = pd.read_sql_query(f"SELECT COUNT(*) AS n FROM {table_name}", conn)["n"].iloc[0]
                counts.append({"table_name": table_name, "row_count": int(row_count)})
        show_df(pd.DataFrame(counts), max_rows=10)

        print_section("4. Column counts")
        col_rows = []
        for table_name in [MARKET_FREE_TABLE, MARKET_AWARE_TABLE]:
            if table_exists(conn, table_name):
                cols = get_columns(conn, table_name)
                col_rows.append({"table_name": table_name, "column_count": len(cols)})
        show_df(pd.DataFrame(col_rows), max_rows=10)

        print_section("5. A few important columns that exist")
        important = [
            "race_date",
            "racecourse",
            "race_no",
            "horse",
            "jockey",
            "trainer",
            "race_distance_m",
            "race_going",
            "dr",
            "act_wt",
            "field_size",
            "horse_prev_starts",
            "horse_prev_win_rate",
            "horse_last_finish_pos",
            "horse_avg_finish_pos_last3",
            "horse_days_since_last_run",
            "jockey_prev_win_rate",
            "trainer_prev_win_rate",
            "horse_jockey_prev_starts",
            "horse_trainer_prev_starts",
            "win_odds",
            "log_win_odds",
            "target_win",
            "target_place_top3",
            "outcome_place_num",
            "outcome_dead_heat",
        ]
        for table_name in [MARKET_FREE_TABLE, MARKET_AWARE_TABLE]:
            if not table_exists(conn, table_name):
                continue
            cols = set(get_columns(conn, table_name))
            present = [c for c in important if c in cols]
            print(f"\n{table_name}:")
            print(", ".join(present))

        print_section("6. Sample rows from market-free table")
        sample_sql = f"""
        SELECT
            race_date,
            racecourse,
            race_no,
            horse,
            jockey,
            trainer,
            race_distance_m,
            race_going,
            dr,
            act_wt,
            field_size,
            horse_prev_starts,
            horse_prev_win_rate,
            horse_last_finish_pos,
            horse_avg_finish_pos_last3,
            horse_days_since_last_run,
            jockey_prev_win_rate,
            trainer_prev_win_rate,
            horse_jockey_prev_starts,
            horse_trainer_prev_starts,
            target_win,
            target_place_top3,
            outcome_place_num,
            outcome_dead_heat
        FROM {MARKET_FREE_TABLE}
        ORDER BY race_date, racecourse, race_no, horse
        LIMIT 15
        """
        show_df(pd.read_sql_query(sample_sql, conn), max_rows=15)

        print_section("7. Date coverage")
        date_sql = f"""
        SELECT
            MIN(race_date) AS min_race_date,
            MAX(race_date) AS max_race_date,
            COUNT(DISTINCT race_date || '|' || racecourse || '|' || race_no) AS race_count
        FROM {MARKET_FREE_TABLE}
        """
        show_df(pd.read_sql_query(date_sql, conn), max_rows=5)

        print_section("8. Quick missingness check for key features")
        missing_sql = f"""
        SELECT
            SUM(CASE WHEN horse_prev_starts IS NULL THEN 1 ELSE 0 END) AS miss_horse_prev_starts,
            SUM(CASE WHEN horse_prev_win_rate IS NULL THEN 1 ELSE 0 END) AS miss_horse_prev_win_rate,
            SUM(CASE WHEN horse_last_finish_pos IS NULL THEN 1 ELSE 0 END) AS miss_horse_last_finish_pos,
            SUM(CASE WHEN horse_days_since_last_run IS NULL THEN 1 ELSE 0 END) AS miss_horse_days_since_last_run,
            SUM(CASE WHEN jockey_prev_win_rate IS NULL THEN 1 ELSE 0 END) AS miss_jockey_prev_win_rate,
            SUM(CASE WHEN trainer_prev_win_rate IS NULL THEN 1 ELSE 0 END) AS miss_trainer_prev_win_rate,
            SUM(CASE WHEN field_size IS NULL THEN 1 ELSE 0 END) AS miss_field_size
        FROM {MARKET_FREE_TABLE}
        """
        show_df(pd.read_sql_query(missing_sql, conn), max_rows=5)

        print_section("9. Sanity check: first career starts")
        first_start_sql = f"""
        SELECT
            SUM(CASE WHEN horse_prev_starts = 0 THEN 1 ELSE 0 END) AS rows_with_zero_prev_starts,
            SUM(CASE WHEN horse_prev_starts = 0 AND horse_last_finish_pos IS NULL THEN 1 ELSE 0 END) AS zero_prev_and_no_last_finish,
            SUM(CASE WHEN horse_prev_starts = 0 AND horse_days_since_last_run IS NULL THEN 1 ELSE 0 END) AS zero_prev_and_no_days_since_last_run
        FROM {MARKET_FREE_TABLE}
        """
        show_df(pd.read_sql_query(first_start_sql, conn), max_rows=5)

        print_section("10. Sanity check: impossible values")
        bad_sql = f"""
        SELECT
            SUM(CASE WHEN horse_prev_starts < 0 THEN 1 ELSE 0 END) AS neg_horse_prev_starts,
            SUM(CASE WHEN horse_prev_win_rate < 0 OR horse_prev_win_rate > 1 THEN 1 ELSE 0 END) AS bad_horse_prev_win_rate,
            SUM(CASE WHEN jockey_prev_win_rate < 0 OR jockey_prev_win_rate > 1 THEN 1 ELSE 0 END) AS bad_jockey_prev_win_rate,
            SUM(CASE WHEN trainer_prev_win_rate < 0 OR trainer_prev_win_rate > 1 THEN 1 ELSE 0 END) AS bad_trainer_prev_win_rate,
            SUM(CASE WHEN horse_days_since_last_run < 0 THEN 1 ELSE 0 END) AS neg_days_since_last_run,
            SUM(CASE WHEN field_size < 2 THEN 1 ELSE 0 END) AS bad_field_size
        FROM {MARKET_FREE_TABLE}
        """
        show_df(pd.read_sql_query(bad_sql, conn), max_rows=5)

        print_section("11. Dead-heat winner count")
        dh_sql = f"""
        SELECT
            SUM(CASE WHEN outcome_dead_heat = 1 THEN 1 ELSE 0 END) AS dead_heat_rows,
            SUM(CASE WHEN target_win = 1 THEN 1 ELSE 0 END) AS target_win_sum,
            SUM(CASE WHEN target_place_top3 = 1 THEN 1 ELSE 0 END) AS target_top3_sum
        FROM {MARKET_FREE_TABLE}
        """
        show_df(pd.read_sql_query(dh_sql, conn), max_rows=5)

        print_section("12. Market-aware-only columns check")
        free_cols = set(get_columns(conn, MARKET_FREE_TABLE))
        aware_cols = set(get_columns(conn, MARKET_AWARE_TABLE))
        only_in_aware = sorted(list(aware_cols - free_cols))
        print("Only in market-aware:")
        print(", ".join(only_in_aware) if only_in_aware else "(none)")

        print_section("13. Columns you should NOT feed into the model")
        likely_leakage = [
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
        ]
        print(", ".join([c for c in likely_leakage if c in free_cols]))


if __name__ == "__main__":
    main()