import sys
import argparse
import json
from pathlib import Path
import numpy as np

from config import (
    COSINE_SIMILARITY_THRESHOLD,
    L2_DISTANCE_THRESHOLD,
    EMBEDDINGS_DIR
)

from storage import StorageManager
from recognizer import SFaceRecognizer

def format_verdict(sim: float, threshold: float = COSINE_SIMILARITY_THRESHOLD) -> str:
    if sim >= threshold:
        pct = min(100.0, (sim / 1.0) * 100.0)
        return f"\033[92m[MATCH / SAME IDENTITY]\033[0m (Similarity: {sim:.4f} >= {threshold:.4f}, Confidence: {pct:.1f}%)"
    else:
        return f"\033[91m[NO MATCH / DIFFERENT PERSON]\033[0m (Similarity: {sim:.4f} < {threshold:.4f})"

def verify_two_embeddings(file1: str, file2: str):
    """Compare two .npy or .json embedding files."""
    storage = StorageManager()
    recognizer = SFaceRecognizer()

    print("\n" + "=" * 65)
    print("        SFace 128-d Face Embedding Verification")
    print("=" * 65)
    print(f"File 1: {file1}")
    print(f"File 2: {file2}")

    emb1 = storage.load_embedding(file1)
    emb2 = storage.load_embedding(file2)

    sim = recognizer.compute_similarity(emb1, emb2)
    dist = recognizer.compute_l2_distance(emb1, emb2)

    print("-" * 65)
    print(f"Cosine Similarity : {sim:+.4f}  (Threshold: >={COSINE_SIMILARITY_THRESHOLD})")
    print(f"Euclidean (L2) Dist: {dist:.4f}   (Threshold: <={L2_DISTANCE_THRESHOLD})")
    print(f"Verdict           : {format_verdict(sim)}")
    print("=" * 65 + "\n")
    return sim, dist

def list_embeddings():
    """List all saved face embeddings in the data/embeddings directory."""
    storage = StorageManager()
    saved = storage.list_saved_embeddings()
    print("\n" + "=" * 78)
    print(f"  Saved Face Embeddings in '{EMBEDDINGS_DIR}' (Total: {len(saved)})")
    print("=" * 78)
    if not saved:
        print("  No face embeddings found yet. Run the demo to capture faces from RTSP stream.")
        print("=" * 78 + "\n")
        return

    for idx, item in enumerate(saved, 1):
        uid = item.get("id")
        created = item.get("created_at")
        cam = item.get("camera_id")
        meta = item.get("quality_metrics", {})
        blur = meta.get("blur_score", 0.0)
        matched = meta.get("matched_name", "UNKNOWN")
        sim = meta.get("similarity_score", 0.0)

        print(f"[{idx:02d}] ID: {uid} | Time: {created} | Cam: {cam}")
        print(f"     Match: {matched} ({sim:.3f}) | Blur Score: {blur:.1f}")
        print(f"     NPY : {Path(item.get('embedding_npy', '')).name}")
        print(f"     Face: {Path(item.get('aligned_image', '')).name}")
        print("-" * 78)
    print()

def main():
    parser = argparse.ArgumentParser(description="Verify and compare saved face embeddings")
    parser.add_argument("--file1", type=str, help="Path to first embedding (.npy or .json)")
    parser.add_argument("--file2", type=str, help="Path to second embedding (.npy or .json)")
    parser.add_argument("--list", action="store_true", help="List all saved embeddings on drive")

    args = parser.parse_args()

    if args.list:
        list_embeddings()
    elif args.file1 and args.file2:
        verify_two_embeddings(args.file1, args.file2)
    else:
        # Default behavior: list embeddings or compare the latest two if available
        storage = StorageManager()
        items = storage.list_saved_embeddings()
        if len(items) >= 2:
            print("Auto-comparing the two most recently saved embeddings:")
            verify_two_embeddings(items[0]["embedding_npy"], items[1]["embedding_npy"])
        else:
            list_embeddings()
            print("Usage: python verify_embeddings.py --file1 <path1.npy> --file2 <path2.npy>")
            print("       python verify_embeddings.py --list")

if __name__ == "__main__":
    main()
