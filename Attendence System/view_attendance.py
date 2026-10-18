import sqlite3
from pathlib import Path
from datetime import datetime

def format_time(ts_str):
    try:
        dt = datetime.fromisoformat(ts_str)
        return dt.strftime("%Y-%m-%d %I:%M:%S %p")
    except Exception:
        return ts_str[:19].replace("T", " ")

def main():
    # Resolve path relative to this script so it works from any working directory
    db_path = Path(__file__).resolve().parent / "data" / "attendance.db"
    if not db_path.exists():
        db_path = Path("data/attendance.db")
    if not db_path.exists():
        print(f"[!] Database not found at {db_path}")
        return

    conn = sqlite3.connect(str(db_path))
    c = conn.cursor()

    print("\n" + "=" * 88)
    print("                     OFFICIAL VERIFIED ATTENDANCE LOG")
    print("=" * 88)

    c.execute("""
        SELECT
            employee_id,
            DATE(timestamp) AS emp_date,
            MAX(TIME(timestamp)) AS max_time,
            MIN(TIME(timestamp)) AS min_time
        FROM attendance_events
        WHERE status = 'CONFIRMED'
        GROUP BY
            employee_id,
            DATE(timestamp)
        ORDER BY emp_date
    """)
    events = c.fetchall()
    if not events:
        print("  No verified attendance events recorded yet.\n")
        return

    print(f"{'Employee ID':<14} | {'Date':<12} | {'First Punch (In)':<18} | {'Last Punch (Out)':<18}")
    print("-" * 70)

    myString = []
    for ev in events:
        staffId = str(ev[0])
        attnDateRaw = str(ev[1])
        maxTimeRaw = str(ev[2])
        minTimeRaw = str(ev[3])

        print(f"{staffId:<14} | {attnDateRaw:<12} | {minTimeRaw:<18} | {maxTimeRaw:<18}")

        # Format date as YYYYMMDD and time as HHMMSS
        attnDate = attnDateRaw.replace("-", "")
        maxTime = maxTimeRaw.replace(":", "")
        minTime = minTimeRaw.replace(":", "")

        myString.append(
            f"INSERT INTO HRISOLD.LV_TTEMPATT "
            f"(STAFFID, CDATE, CTIME, TARMINALID, LOCATIONID) "
            f"VALUES ('{staffId}', '{attnDate}', '{maxTime}', '111111', 5)"
        )

        myString.append(
            f"INSERT INTO HRISOLD.LV_TTEMPATT "
            f"(STAFFID, CDATE, CTIME, TARMINALID, LOCATIONID) "
            f"VALUES ('{staffId}', '{attnDate}', '{minTime}', '111111', 5)"
        )

    print("=" * 70)
    print(f"Total Records: {len(events)} (Prepared {len(myString)} Oracle SQL statements)\n")

  
  

if __name__ == '__main__':
    main()
