"""
sync_hris_attendance.py
--------------------------------------------------------------------------------
Standalone bridge script to read verified attendance punches from the local
SQLite database, aggregate the FIRST and LAST attendance punch for each
employee per day, and sync them to the Oracle HRIS database (HRISOLD.LV_TTEMPATT).

Features:
- Separate codebase; zero changes to existing pipeline or detection logic.
- Graceful dry-run mode when Oracle credentials are empty: prints formatted
  data, types, and summary table to terminal for verification.
- Optional date filter (--date YYYY-MM-DD, --today, or all dates).
- Pre-configured batching and error rollback for Oracle execution.
--------------------------------------------------------------------------------
"""

import sys
import argparse
import sqlite3
from pathlib import Path
from datetime import datetime, date
from typing import List, Tuple, Dict, Any, Optional

# Optional import of oracledb so the script runs and prints data even before oracledb is installed
try:
    import oracledb
except ImportError:
    oracledb = None


# ==============================================================================
# CONFIGURATION
# ==============================================================================

# 1. Local SQLite Database Path
# Auto-detects whether script is run from project root or inside "Attendence System"
SCRIPT_DIR = Path(__file__).resolve().parent
DB_CANDIDATES = [
    SCRIPT_DIR / "data" / "attendance.db",
    SCRIPT_DIR.parent / "Attendence System" / "data" / "attendance.db",
    Path("data/attendance.db"),
    Path("Attendence System/data/attendance.db")
]

SQLITE_DB = None
for candidate in DB_CANDIDATES:
    if candidate.exists():
        SQLITE_DB = candidate.resolve()
        break

if SQLITE_DB is None:
    # Default fallback
    SQLITE_DB = SCRIPT_DIR / "data" / "attendance.db"

# 2. Oracle HRIS Database Configuration
# Fill in your Oracle host, port, service_name, user, and password when ready.
ORACLE_CONFIG = {
    "user": "",
    "password": "",
    "host": "",
    "port": 1521,
    "service_name": ""
}

# 3. Default Terminal & Location IDs for HRIS
DEFAULT_TERMINAL_ID = "111111"
DEFAULT_LOCATION_ID = 1

# 4. Insertion Logic Options
# If True: When a staff only has 1 punch on a given day (first_punch == last_punch),
#          it outputs two identical records (last_punch, first_punch) matching the legacy C# bridge.
# If False: It only outputs one record if first_punch == last_punch.
SEND_DUPLICATE_FOR_SINGLE_PUNCH = True


# ==============================================================================
# SQLITE DATA FETCHING
# ==============================================================================

def get_daily_punches(
    sqlite_db_path: Path,
    target_date: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Reads verified attendance from SQLite and calculates first & last punch per employee per day.
    
    :param sqlite_db_path: Path to attendance.db
    :param target_date: Optional filter in 'YYYY-MM-DD' format. If None, retrieves all days.
    :return: List of dict records containing staff_id, staff_name, work_date, first_punch, last_punch, punch_count
    """
    if not sqlite_db_path.exists():
        raise FileNotFoundError(f"SQLite database not found at: {sqlite_db_path}")

    conn = sqlite3.connect(str(sqlite_db_path))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    query = """
        SELECT
            employee_id AS staff_id,
            employee_name AS staff_name,
            DATE(timestamp) AS work_date,
            MIN(TIME(timestamp)) AS first_punch,
            MAX(TIME(timestamp)) AS last_punch,
            COUNT(*) AS punch_count
        FROM attendance_events
        WHERE status = 'CONFIRMED'
    """
    params = []

    if target_date:
        query += " AND DATE(timestamp) = ?"
        params.append(target_date)

    query += """
        GROUP BY employee_id, DATE(timestamp)
        ORDER BY work_date ASC, employee_id ASC
    """

    cursor.execute(query, params)
    raw_rows = cursor.fetchall()
    conn.close()

    results = []
    for r in raw_rows:
        results.append({
            "staff_id": str(r["staff_id"]).strip(),
            "staff_name": str(r["staff_name"]).strip(),
            "work_date": str(r["work_date"]).strip(),
            "first_punch": str(r["first_punch"]).strip() if r["first_punch"] else "",
            "last_punch": str(r["last_punch"]).strip() if r["last_punch"] else "",
            "punch_count": int(r["punch_count"])
        })

    return results


# ==============================================================================
# HRIS RECORD PREPARATION
# ==============================================================================

def prepare_oracle_rows(
    daily_punches: List[Dict[str, Any]],
    terminal_id: str = DEFAULT_TERMINAL_ID,
    location_id: int = DEFAULT_LOCATION_ID,
    duplicate_for_single_punch: bool = SEND_DUPLICATE_FOR_SINGLE_PUNCH
) -> List[Tuple[str, str, str, str, int]]:
    """
    Converts SQLite daily records into Oracle HRIS tuples matching HRISOLD.LV_TTEMPATT schema:
    (STAFFID, CDATE, CTIME, TARMINALID, LOCATIONID)

    Matches legacy C# insertion sequence:
    For each employee/day, creates records for (last_time, first_time).
    """
    oracle_rows: List[Tuple[str, str, str, str, int]] = []

    for item in daily_punches:
        staff_id = item["staff_id"]
        work_date_str = item["work_date"]
        first_time = item["first_punch"]
        last_time = item["last_punch"]

        # CDATE format: 'YYYY-MM-DD' -> 'YYYYMMDD'
        dt = datetime.strptime(work_date_str, "%Y-%m-%d")
        cdate = dt.strftime("%Y%m%d")

        # Punches to include
        if first_time == last_time and not duplicate_for_single_punch:
            punches = (first_time,)
        else:
            # Matches original C# bridge logic: (last_time, first_time)
            punches = (last_time, first_time)

        for punch_time in punches:
            if not punch_time:
                continue
            # CTIME format: 'HH:MM:SS' -> 'HHMMSS'
            ctime = punch_time.replace(":", "")
            oracle_rows.append((
                str(staff_id),
                str(cdate),
                str(ctime),
                str(terminal_id),
                int(location_id)
            ))

    return oracle_rows


# ==============================================================================
# ORACLE SYNC EXECUTION
# ==============================================================================

def is_oracle_configured() -> bool:
    """Checks if Oracle connection parameters are filled."""
    user = ORACLE_CONFIG.get("user", "").strip()
    host = ORACLE_CONFIG.get("host", "").strip()
    service_name = ORACLE_CONFIG.get("service_name", "").strip()
    return bool(user and (host or service_name))


def sync_to_oracle(oracle_rows: List[Tuple[str, str, str, str, int]]) -> int:
    """
    Inserts prepared rows into Oracle table HRISOLD.LV_TTEMPATT in batches of 2000.
    """
    if not oracle_rows:
        print("[INFO] No records to sync.")
        return 0

    if not is_oracle_configured():
        print("\n[NOTICE] Oracle configuration is empty.")
        print("[NOTICE] Skipping database insert. (Values are printed below for inspection)")
        return 0

    if oracledb is None:
        raise ImportError(
            "The 'oracledb' library is required to sync with Oracle.\n"
            "Install it via: pip install oracledb"
        )

    dsn = oracledb.makedsn(
        ORACLE_CONFIG["host"],
        ORACLE_CONFIG.get("port", 1521),
        service_name=ORACLE_CONFIG["service_name"]
    )

    connection = oracledb.connect(
        user=ORACLE_CONFIG["user"],
        password=ORACLE_CONFIG["password"],
        dsn=dsn
    )

    inserted_count = 0
    sql = """
        INSERT INTO HRISOLD.LV_TTEMPATT
        (STAFFID, CDATE, CTIME, TARMINALID, LOCATIONID)
        VALUES (:1, :2, :3, :4, :5)
    """

    try:
        with connection.cursor() as cursor:
            batch_size = 2000
            for start in range(0, len(oracle_rows), batch_size):
                batch = oracle_rows[start:start + batch_size]
                cursor.executemany(sql, batch)
                inserted_count += len(batch)
                print(f"[PROGRESS] Synced {inserted_count}/{len(oracle_rows)} records...")

        connection.commit()
        print(f"\n[SUCCESS] Successfully committed {inserted_count} records to Oracle HRIS.")
        return inserted_count

    except Exception as e:
        connection.rollback()
        print(f"\n[ERROR] Oracle transaction rolled back due to error: {e}", file=sys.stderr)
        raise
    finally:
        connection.close()


# ==============================================================================
# MAIN & DISPLAY LOGIC
# ==============================================================================

def display_summary_and_rows(
    daily_punches: List[Dict[str, Any]],
    oracle_rows: List[Tuple[str, str, str, str, int]]
):
    """Prints human-readable daily attendance summaries and the exact Oracle payloads."""
    print("\n" + "=" * 92)
    print("           HRIS ATTENDANCE SYNC - DAILY FIRST & LAST PUNCH SUMMARY")
    print("=" * 92)
    print(f"Source SQLite DB : {SQLITE_DB}")
    print(f"Oracle Config    : {'CONFIGURED' if is_oracle_configured() else 'EMPTY (Dry-Run / Preview Mode)'}")
    print(f"Terminal ID      : {DEFAULT_TERMINAL_ID} | Location ID: {DEFAULT_LOCATION_ID}")
    print("-" * 92)

    if not daily_punches:
        print("  [!] No confirmed attendance records found in SQLite.")
        print("=" * 92 + "\n")
        return

    # 1. Print Summary Table
    print(f"{'Staff ID':<10} | {'Staff Name':<16} | {'Date':<10} | {'First Punch':<12} | {'Last Punch':<12} | {'Total'}")
    print("-" * 92)
    for p in daily_punches:
        print(
            f"{p['staff_id']:<10} | "
            f"{p['staff_name']:<16} | "
            f"{p['work_date']:<10} | "
            f"{p['first_punch']:<12} | "
            f"{p['last_punch']:<12} | "
            f"{p['punch_count']} punch(es)"
        )

    print("=" * 92)
    print(f" Total Employee-Day Groups : {len(daily_punches)}")
    print(f" Total Oracle Rows Prepared : {len(oracle_rows)}")
    print("=" * 92)

    # 2. Print exact Oracle rows payload for format validation
    print("\n" + "-" * 92)
    print(" [PAYLOAD PREVIEW] Rows ready for HRISOLD.LV_TTEMPATT:")
    print(" Schema: (STAFFID, CDATE, CTIME, TARMINALID, LOCATIONID)")
    print("-" * 92)
    print(f"{'#':<4} | {'STAFFID':<10} | {'CDATE':<10} | {'CTIME':<8} | {'TARMINALID':<12} | {'LOCATIONID':<10}")
    print("-" * 92)
    for idx, row in enumerate(oracle_rows, start=1):
        staff_id, cdate, ctime, term_id, loc_id = row
        print(f"#{idx:<3} | {staff_id:<10} | {cdate:<10} | {ctime:<8} | {term_id:<12} | {loc_id:<10}")

    print("-" * 92)
    if not is_oracle_configured():
        print("\n [INFO] Oracle config is currently empty.")
        print("        To sync live to Oracle HRIS:")
        print("        1. Install oracledb: pip install oracledb")
        print("        2. Fill ORACLE_CONFIG in sync_hris_attendance.py")
        print("        3. Re-run: python sync_hris_attendance.py\n")


def main():
    parser = argparse.ArgumentParser(
        description="Sync daily first/last attendance punches from SQLite to Oracle HRIS."
    )
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Target date in 'YYYY-MM-DD' format (e.g. --date 2026-10-11). Defaults to all days."
    )
    parser.add_argument(
        "--today",
        action="store_true",
        help="Sync only today's attendance."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview and print data without inserting into Oracle."
    )
    args = parser.parse_args()

    # Determine date filter
    target_date = None
    if args.today:
        target_date = date.today().isoformat()
    elif args.date:
        target_date = args.date

    # 1. Fetch from SQLite
    try:
        daily_punches = get_daily_punches(SQLITE_DB, target_date=target_date)
    except Exception as e:
        print(f"[ERROR] Failed to query SQLite database: {e}", file=sys.stderr)
        sys.exit(1)

    # 2. Prepare Oracle rows
    oracle_rows = prepare_oracle_rows(daily_punches)

    # 3. Print terminal preview & format check
    display_summary_and_rows(daily_punches, oracle_rows)

    # 4. Sync to Oracle if configured and not dry-run
    if is_oracle_configured() and not args.dry_run:
        try:
            sync_to_oracle(oracle_rows)
        except Exception as e:
            print(f"[FATAL] Oracle sync failed: {e}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
