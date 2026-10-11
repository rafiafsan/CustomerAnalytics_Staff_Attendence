import sqlite3
from storage import StorageManager
from recognizer import SFaceRecognizer

def main():
    storage = StorageManager()
    rec = SFaceRecognizer()

    conn = sqlite3.connect('data/attendance.db')
    c = conn.cursor()
    c.execute("SELECT id, name, embedding_path FROM employees WHERE id LIKE 'STAFF%'")
    staffs = [(row[0], row[1], storage.load_embedding(row[2])) for row in c.fetchall()]

    c.execute("SELECT id, match_name, embedding_path FROM face_captures WHERE passed_quality = 1 AND camera_id = 'CAM_01_MAIN' ORDER BY id DESC LIMIT 5")
    live_caps = c.fetchall()

    print("\nComparing Latest Live RTSP Stream Captures against Enrolled Staffs:")
    print("-" * 75)
    for cap_id, mname, ep in live_caps:
        if not ep:
            continue
        try:
            emb = storage.load_embedding(ep)
        except Exception:
            continue
        scores = [(s_name, rec.compute_similarity(emb, s_emb)) for s_id, s_name, s_emb in staffs]
        scores.sort(key=lambda x: x[1], reverse=True)
        best_name, best_score = scores[0]
        verdict = "[MATCH!]" if best_score >= 0.38 else "[Below Threshold]"
        print(f"Capture #{cap_id} -> Best: {best_name:<8} ({best_score:.3f}) | {verdict}")
        all_scores = ", ".join([f"{name}: {sc:.2f}" for name, sc in scores])
        print(f"   Breakdown: {all_scores}")
    print("-" * 75 + "\n")

if __name__ == '__main__':
    main()
