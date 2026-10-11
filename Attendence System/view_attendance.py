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
        SELECT id, employee_id, employee_name, timestamp, camera_id, confidence, status, face_image_path, embedding_path
        FROM attendance_events
        ORDER BY id ASC
    """)
    events = c.fetchall()

    if not events:
        print("  No verified attendance events recorded yet.")
        print("  (A face must be recognized with >= 38% confidence across 3 consecutive frames to confirm)")
    else:
        print(f"{'Event #':<8} | {'Employee Name':<16} | {'ID':<10} | {'Time':<24} | {'Match %':<8} | {'Status'}")
        print("-" * 88)
        for ev in events:
            eid, emp_id, name, ts, cam, conf, status, img_p, emb_p = ev
            formatted_t = format_time(ts)
            pct = f"{conf * 100:.1f}%"
            print(f"#{eid:<7} | {name:<16} | {emp_id:<10} | {formatted_t:<24} | {pct:<8} | {status}")
            if img_p and Path(img_p).exists():
                print(f"         +-- Snapshot: {Path(img_p).name}")

    print("=" * 88)

    # Summary Stats
    c.execute("SELECT COUNT(*), COUNT(DISTINCT employee_id) FROM attendance_events")
    total_events, distinct_emps = c.fetchone()

    c.execute("SELECT employee_name, COUNT(*) FROM attendance_events GROUP BY employee_id")
    breakdown = c.fetchall()

    print(f"\n Summary: Total Confirmed Check-ins: {total_events} | Unique Staff: {distinct_emps}")
    if breakdown:
        print(" Check-ins per Staff:")
        for name, count in breakdown:
            print(f"   * {name}: {count} event(s)")

    print("\n Where to view:")
    print(" 1. Web Dashboard : Open http://127.0.0.1:8080 and click 'Verified Attendance Events' tab")
    print(" 2. Terminal      : Run 'python view_attendance.py'")
    print(" 3. SQLite DB     : Query table 'attendance_events' in data/attendance.db")
    print("=" * 88 + "\n")

if __name__ == '__main__':
    main()
