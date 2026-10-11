import sqlite3

def check():
    conn = sqlite3.connect('data/attendance.db')
    c = conn.cursor()

    print("\n" + "=" * 70)
    print("           VERIFIED ATTENDANCE RECORDS (attendance_events)")
    print("=" * 70)
    c.execute("SELECT id, employee_id, employee_name, timestamp, camera_id, confidence, status, face_image_path FROM attendance_events")
    events = c.fetchall()
    if not events:
        print("No verified attendance events recorded yet.")
    else:
        for e in events:
            print(f" Event #{e[0]}: Employee: {e[2]} (ID: {e[1]}) | Conf: {e[5]:.2f} | Status: {e[6]} | Time: {e[3]}")

    print("\n" + "=" * 70)
    print("           IDENTIFIED MATCHES IN LIVE CAPTURES (face_captures)")
    print("=" * 70)
    c.execute("SELECT id, timestamp, match_name, match_score, passed_quality, face_image_path FROM face_captures WHERE match_name != 'UNKNOWN' ORDER BY id DESC LIMIT 20")
    matches = c.fetchall()
    if not matches:
        print("No matches != UNKNOWN found.")
    else:
        for m in matches:
            print(f" Capture #{m[0]}: Matched: {m[2]} | Score: {m[3]:.3f} | Passed Quality: {bool(m[4])} | Time: {m[1]}")

    print("\n" + "=" * 70)
    print("           ENROLLED EMPLOYEES & STAFF")
    print("=" * 70)
    c.execute("SELECT id, name, department, enrolled_at FROM employees")
    for emp in c.fetchall():
        print(f" ID: {emp[0]:<10} | Name: {emp[1]:<16} | Dept: {emp[2]:<10} | Enrolled: {emp[3]}")
    print("=" * 70 + "\n")

if __name__ == '__main__':
    check()
