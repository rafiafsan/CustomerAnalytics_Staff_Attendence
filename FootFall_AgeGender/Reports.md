# Retail AI Edge System — Full Audit, Optimization & Deployment Report

**Project:** `Low_Resource_ai_system` (Retail AI Analytics Engine — footfall, bag-exit, demographics)
**Report date:** 2026-10-07
**Analyst role:** Computer Vision / ML / Edge Deployment review
**Target:** Low-configuration PC (Intel 4-core CPU, 4–8 GB RAM, no NVIDIA GPU, Windows/Linux, 24/7 operation)

---

## 0. How to read this report (evidence rules)

Everything below is labelled with one of three confidence levels:

| Label | Meaning |
|---|---|
| ✅ **CONFIRMED** | Verified by reading the code *and* (where possible) by running it in this environment. |
| 🟡 **LIKELY** | Read from code with high confidence, but not reproduced (missing model/video assets or missing library). |
| 💡 **POSSIBLE** | Improvement / risk that depends on your hardware, camera or data — must be measured. |

**What I actually ran in your repo (Python 3.13, CPU-only, no torch/ultralytics/supervision installed):**

| Check | Result |
|---|---|
| `python tests/test_edge_changes.py` | ✅ PASSED (letterbox math, `mask_url`, `BoundedIdSet` are correct) |
| Synthetic test: one empty-detection frame in `TrackManager.track()` | ✅ BUG REPRODUCED — all tracks lost in 1 frame |
| Synthetic test: consequence of that bug on `EventManager` | ✅ BUG REPRODUCED — gate crossing silently missed |
| Scheduler effective FPS with 200 ms work @ 4 FPS target | ✅ BUG REPRODUCED — 3 passes/s instead of 4 |
| Micro-benchmark of non-model stages (preprocess / NMS / HUD / copy) | ✅ MEASURED (numbers below) |
| `python app.py --no-display --no-demographics` (fresh clone) | ✅ CRASHES — traceback captured |
| Model inference latency | ❌ **NOT MEASURED** — `models/` is not in the repo. Command to measure it is in §9. |

> Anything about inference time, RAM of the full app, mAP, or low-end FPS in this report is an
> **estimate clearly marked as such**. I do not invent measurements.

---

# A. System Diagnosis — What is this system, and what is wrong right now?

## A.1 What the system does

A single-camera **retail store analytics edge agent**. From one video source (RTSP CCTV / USB cam / video file) it:

1. Detects **persons and bags** (YOLOv8 ONNX on CPU).
2. Tracks them with **ByteTrack** (via `supervision`).
3. Counts **entries / exits** through a user-drawn virtual tripwire ("footfall gate"), with dwell validation (entry only counts after 60 s inside, exit after 10 s outside).
4. Detects **person-leaving-with-a-bag** through a second gate (association + crossing + movement heuristic).
5. Optionally runs **MiVOLO age/gender** classification on an async background thread.
6. Sends all validated events **asynchronously over HTTP POST** to a central FastAPI gateway (with a local SQLite spill buffer for network outages).
7. Shows a live OpenCV preview window with HUD (optional, can be headless).

## A.2 The real architecture (data flow as implemented)

```
                                 ┌────────────────────────────────────────────────┐
                                 │            RTSP / Video / Webcam              │
                                 └──────────────────────┬─────────────────────────┘
                                                        │ ffmpeg H.264 (TCP transport forced)
                                          ┌─────────────▼──────────────┐
   THREAD 1 (daemon)                     │  RTSPReader._capture_loop  │  grab() every frame,
   "RTSPReader-Thread"                   │  single-slot latest-frame  │  retrieve() only on request
                                          │  buffer (_latest_frame)    │  auto-reconnect every 2 s
                                          └──────┬──────────────┬──────┘
                                    fresh frame  │              │  latest frame (copy)
                                                 │              │
   THREAD 2 (daemon)                    ┌────────▼───────┐ ┌────▼──────────────┐
   "AI-InferenceThread"                 │  ONNX letterbox│ │ DISPLAY LOOP      │  (main thread)
   (gated to inference.target_fps=4)    │  720p → 480²   │ │ gates + boxes +   │
                                        │  onnxruntime   │ │ HUD + imshow @15  │
                                        │  (2 threads)   │ │ cv2.waitKey       │
                                        └────────┬───────┘ └────▲──────────────┘
                                                 │              │ thread-safe snapshot
                              person/bag dicts   │              │ (self.cached_persons …)
                                                 ▼              │
                                        ┌────────────────┐      │
                                        │ ByteTrack      │      │
                                        │ (supervision)  │      │
                                        └───────┬────────┘      │
                                                │               │
        ┌───────────────────────────────────────┼───────────────┤
        │  lost-track eviction fan-out          ▼               │
        │  (event_manager / associator /        │               │
        │   exit_validator / MiVOLO worker)     │               │
        │                                       ▼               │
        │                     ┌──────────────────────────────┐  │
        │                     │ EventManager: gate crossing, │  │
        │                     │ pending entries/exits,       │  │
        │                     │ 60 s / 10 s dwell validation │  │
        │                     └──────────┬───────────────────┘  │
        │                                │ validated ENTRY/EXIT │
        │                                ▼                      │
        │                     ┌──────────────────────────────┐  │
        │                     │ business_logic hooks →       │  │
        │                     │ DatabaseManager (ApiManager) │  │
        │                     │  job_queue (10 000)          │  │
        │                     └───────┬──────────────┬───────┘  │
        │                             │              │          │
        │              THREAD 3       │              │  THREAD 4│
        │   "API-HTTP-WriterThread"   │              │ "MiVOLO-WorkerThread"
        │   POST/PATCH → gateway      │              │ crop → age/gender (lazy torch load)
        │   fail → SQLite edge_events │              │ bounded queue (4)
        │   replay every 10 s         │              │
        │                             ▼              ▼          │
        │                    FastAPI gateway → PostgreSQL       │
        │                                                       │
        └── PerformanceMonitor (1 Hz RAM/CPU), LightweightMonitor (30 s health log)
```

**This architecture is fundamentally sound.** The single-slot frame buffer, the capture/inference/display decoupling, the bounded queues, the SQLite spill, the RTSP TCP transport and auto-reconnect, the adaptive inference scheduler, the bounded ID sets, and the supervisor scripts are all *correct, deliberate low-resource design choices*. **I am explicitly telling you these are implemented correctly — do not change them.**

## A.3 What is currently wrong (summary)

1. **A fresh clone of this repo cannot start at all** (confirmed crash — see C-1). Assets are gitignored (`models/`, `test_videos/`, MiVOLO checkpoint) and the ONNX-missing fallback path imports `torch`, which is a *dev-only* dependency.
2. **One blank detection frame destroys every active track** (confirmed) — this silently deletes gate-crossing state and produces **missed entries/exits**. This is the single biggest accuracy bug.
3. **All counters reset to 0 on every restart/crash**, and are then pushed to the central server as daily metrics (confirmed by code path) — a crash corrupts the central dashboard.
4. **The event queue blocks the AI thread** when the network is down for many hours (confirmed by code reading) — the pipeline can freeze.
5. **Gate coordinates are resolution-dependent but resolution is never validated** — a 1080p RTSP stream instead of 720p silently breaks all counting.
6. Gate crossing uses the **top of the bounding box (head)** rather than the feet, which is where the calibrated floor line actually is.
7. Assorted: dead config keys, ignored config values, unpinned dependencies, `mivolo_system/requirements.txt` pulling **CUDA torch**, log-rotation races, GUI-mode watchdog that silently kills analytics, no fail-fast on a missing MiVOLO checkpoint.

None of these require a rewrite. They are surgical fixes (§C, §16).

---

# B. Architecture Diagrams

## B.1 Current (as-is)

```
Camera ─► Capture thread ─► Latest-frame slot ─┬─► Display thread (15 FPS, main)
                                               └─► AI thread (4 FPS):
                                                     letterbox → YOLO ONNX → NMS
                                                       → ByteTrack → gate/dwell logic
                                                         → bag association → demographics
                                                           → HTTP queue → gateway
```

Strengths: decoupled, bounded memory, non-blocking display, auto-reconnect, spill buffer.
Weaknesses: all AI stages run in **one thread** (so one slow stage delays all), eviction fan-out on a single blank frame, in-memory-only counters, no health heartbeat.

## B.2 Recommended (production, still simple enough for a 4-core box)

```
                         ┌──────────────────────────────────────────────────────┐
                         │                 Supervisor (bat loop / systemd)      │
                         │   crash → 5 s backoff → restart, RAM ceiling, logs    │
                         └──────────────────────────┬───────────────────────────┘
                                                    ▼
 ┌──────────────┐   single-slot   ┌───────────────────────────────────────────────┐
 │ Capture      │   latest frame  │            UnifiedRetailPipeline             │
 │ thread       ├────────────────►│  Stage 1  preprocess (letterbox, 1×resize)   │
 │ (RTSP TCP,   │                 │  Stage 2  YOLO ONNX (INT8 candidate)         │
 │  reconnect,  │                 │  Stage 3  NMS (vectorised, per-class conf)   │
 │  frame drop) │                 │  Stage 4  ByteTrack (hysteresis ALWAYS on)   │
 └──────────────┘                 │  Stage 5  gate/dwell + bag logic  (<1 ms)    │
                                  │  → publishes snapshot under lock            │
                                  └───────┬──────────────────────┬───────────────┘
                                          │ validated events     │ crops (bounded q=4)
                        ┌─────────────────▼───────┐   ┌──────────▼──────────────┐
                        │ Event/Counter Store     │   │ MiVOLO worker (opt-in)  │
                        │ SQLite (counters+pend.) │   │ 1 torch thread, lazy    │
                        │ survives restart        │   │ fail-fast if no ckpt    │
                        └───────────┬─────────────┘   └─────────────────────────┘
                                    │ non-blocking put_nowait
                        ┌───────────▼─────────────┐
                        │ Sender thread            │  HTTP → gateway
                        │ queue(10k) → fail→spill  │  heartbeat every 60 s
                        │ replay every 10 s        │  offline: SQLite spill
                        └─────────────────────────┘
                                  ▲
                        ┌─────────┴─────────┐
                        │ status.json (30 s)│ ← FPS, RAM, CPU, camera, errors
                        └───────────────────┘  (tiny, for external watchdog/UI)
```

Key differences from current: **counters persisted locally**, **non-blocking event enqueue**, **stage-level timing**, optional **status.json heartbeat**, hysteresis guaranteed, no functional restructure.

---

# C. Critical Issues (prioritized)

Severity: 🔴 Critical (crash / wrong results / unusable) · 🟠 High · 🟡 Medium · 🟢 Low

## C-1 🔴 CONFIRMED — Fresh clone crashes at startup; no actionable error message

**What is wrong.** `models/`, `test_videos/` and the MiVOLO checkpoint are gitignored, so on a new PC `inference.detector.create_detector()` finds no `.onnx`, falls through to `TorchDetector`, and dies:

```
File "inference\detector.py", line 55, in create_detector
    from .torch_detector import TorchDetector
File "inference\torch_detector.py", line 12, in <module>
    import torch
ModuleNotFoundError: No module named 'torch'
```

Also observed before the crash: `[RTSPReader] Could not open video source: test_videos/Outlet_Sample_2_Banasree.avi` — the capture thread is already spinning on a non-existent source.

**Why.** The runtime `requirements.txt` intentionally excludes torch (correct for the edge box), but the ONNX→torch fallback silently depends on it.

**Impact.** 🔴 Any deployment/redeploy fails; the operator gets a Python traceback instead of "run `export_and_quantize.py`".

**Fix.**

```python
# inference/detector.py  (replace the tail of create_detector)
    # PyTorch fallback only if torch is actually installed AND weights exist
    target_pt = model_path if model_path.endswith(".pt") and os.path.exists(model_path) else fallback_model
    if not os.path.exists(target_pt):
        for candidate in ("models/yolov8n.pt", "models/yolov8m.pt"):
            if os.path.exists(candidate):
                target_pt = candidate
                break

    if not os.path.exists(target_pt):
        raise FileNotFoundError(
            "No usable model found.\n"
            "  Expected: models/yolov8n_480.onnx (or another *.onnx)\n"
            "  Create it with:  python export_and_quantize.py --weights yolov8n.pt --imgsz 480\n"
            "  (run that on a build machine with: pip install -r requirements-dev.txt)"
        )

    try:
        import torch  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "ONNX model missing and torch is not installed. "
            "Install the ONNX model (preferred) or requirements-dev.txt (heavy)."
        ) from exc
```

Also add a **fail-fast preflight in `UnifiedRetailPipeline.__init__`** for the video source (see C-5) so a missing file does not start a reconnect loop forever.

## C-2 🔴 CONFIRMED — One frame with zero detections wipes ALL tracks (hysteresis bypassed)

**What is wrong.** `tracking/track_manager.py:77-81`:

```python
if not all_dets or self.tracker is None:
    newly_lost = self.active_ids - current_active_ids   # current_active_ids is ALWAYS empty here
    self.lost_ids.update(newly_lost)
    self.active_ids = current_active_ids                # clears every active id
    return tracked_persons, tracked_bags, newly_lost
```

The non-empty path carefully implements a `max_missing_frames = max(10, frame_rate*5) = 20` frame hysteresis (lines 159-167). The empty-detections path **bypasses it entirely**.

**Reproduced:** with 3 healthy active tracks and a single `track([], [])` call →
`newly_lost = [10, 11, 12]`, `active_ids = []`.

**Why it is wrong.** Zero-detection frames are *normal* at 4 AI-FPS: motion blur, backlight, a crowd occluding everyone, a single decode glitch, a dark frame at night.

**How it affects the system (reproduced end-to-end).** The pipeline then runs the eviction fan-out (`services/unified_pipeline.py:374-381`):

* `event_manager.evict_lost_tracks()` deletes `previous_side[track]` → the next frame's side becomes the *baseline*, so an **OUTSIDE→INSIDE transition in progress is silently lost → no entry counted**. Reproduced: `crossing event returned: None`.
* bag-exit `pending_associations` deleted → bag-exit events lost.
* MiVOLO evidence evicted → demographics restart sampling (wasted CPU).
* associator candidates/`used_*` cleared → duplicate `person_with_bag` counts possible.

**Severity:** 🔴 — this alone can cost you a large percentage of daily footfall.

**Fix (minimal, preserves behaviour otherwise):**

```python
# tracking/track_manager.py  -> inside track(), replace the early-return block
        if self.tracker is None:
            # No tracker installed: return empty without corrupting state,
            # and make the failure loud instead of silent.
            if not getattr(self, "_warned_no_tracker", False):
                print("[TrackManager] ERROR: supervision not installed — tracking disabled!")
                self._warned_no_tracker = True
            return [], [], set()

        if not all_dets:
            # ZERO detections this frame -> must use the SAME hysteresis as the
            # normal path, never evict everything at once.
            pass                       # fall through to the hysteresis block below
```

and make the tracking block tolerate empty `tracked` output (it already does), i.e. let execution reach:

```python
        newly_lost = set()
        for tid in list(self.active_ids):
            if tid not in current_active_ids:
                self.missing_counts[tid] = self.missing_counts.get(tid, 0) + 1
                if self.missing_counts[tid] >= self.max_missing_frames:
                    newly_lost.add(tid)
                    self.active_ids.discard(tid)
                    self.missing_counts.pop(tid, None)
                    self.history.pop(tid, None)
```

(In the empty-detection case `current_active_ids` is empty, so every track simply accrues +1 missing count — exactly the intended behaviour.)

**Add a regression test:**

```python
def test_blank_frame_does_not_evict_tracks():
    tm = TrackManager(trail_length=5, frame_rate=4)
    tm.tracker = object()                 # pretend ByteTrack is available
    tm.active_ids = {10, 11}; tm.missing_counts = {10: 0, 11: 0}
    _, _, lost = tm.track([], [])
    assert lost == set()                  # 1 blank frame must NOT kill tracks
    for _ in range(tm.max_missing_frames):
        _, _, lost = tm.track([], [])
    assert lost == {10, 11}               # ...but 20 consecutive blanks must
```

## C-3 🔴 CONFIRMED (by code path) — Restart resets all counters, then reports them as daily metrics

**What is wrong.** `validated_entry_count`, `validated_exit_count`, `person_with_bag_exit_count`, `demographics_count` live only in RAM. Every 5 s the pipeline POSTs them:

```python
self.db_manager.upsert_daily_metrics(in_count=..., out_count=..., ...)
```

After any crash/restart (RAM watchdog exit code 3, power cut, app update), counts restart at 0 and the next POST tells the server **`in_count=0` for today**. If the gateway upserts (the name says "upsert"), today's numbers are destroyed. The same applies to `insert_occupancy()`.

**Impact.** 🔴 Silent data corruption in the central database; invisible until someone compares totals.

**Fix (recommended, simple):** persist the counters in the existing SQLite spill DB and restore at startup; **or** (if you control the gateway) switch to *deltas* instead of absolute values.

```python
# database/db_manager.py — add
def load_counters(self) -> dict:
    try:
        row = self._spill_conn.execute(
            "SELECT in_count, out_count, bag_exits, demo_count, day FROM counters WHERE day = ?",
            (date.today().strftime("%Y-%m-%d"),),
        ).fetchone()
        if row:
            return {"in": row[0], "out": row[1], "bag": row[2], "demo": row[3]}
    except Exception:
        pass
    return {"in": 0, "out": 0, "bag": 0, "demo": 0}

def save_counters(self, in_count, out_count, bag_exits, demo_count) -> None:
    # called at most every 5 s and at shutdown; REPLACE INTO counters ...
```

```python
# services/unified_pipeline.py __init__ (after db_manager creation)
        saved = self.db_manager.load_counters()
        self.event_manager.validated_entry_count = saved["in"]
        self.event_manager.validated_exit_count  = saved["out"]
        # restore bag/demographic totals the same way, then re-seed
        # event_manager.entry_count/exit_count so occupancy maths stays consistent
```

Also: `recorded_entry_db_ids` / `completed_entries` are *not* restored, so after a restart a person still inside can be counted again — acceptable (rare), but note it in the docs.

## C-4 🔴 CONFIRMED (by code path) — Blocking `job_queue.put()` can freeze the AI thread

**What is wrong.** `DatabaseManager.insert_entry/exit/demographic/upsert_daily_metrics/...` all call `self.job_queue.put(...)` — **blocking** put on a `Queue(maxsize=10000)`.

**Why it matters.** With the network down, the writer thread fails `_send()` after ~2×5 s timeouts, then spills *that one item* to SQLite — but items already in the queue are only drained at ~1 per 10 s when offline (retries dominate). Daily metrics POSTs alone add 17 280 items/day. In a prolonged outage the queue reaches 10 000 and **the next `put()` blocks the AI inference thread → the whole pipeline stops detecting.**

**Impact.** 🔴 "System froze but process is alive" — the worst kind of production failure.

**Fix (one-liner per call site — add a safe enqueue to ApiManager):**

```python
# database/db_manager.py
    def _enqueue(self, method: str, endpoint: str, payload: Dict[str, Any]) -> None:
        """Never block the caller: spill straight to disk if the queue is saturated."""
        try:
            self.job_queue.put_nowait((method, endpoint, payload))
        except queue.Full:
            self._spill(method, endpoint, payload)   # already exists & is bounded
```

Then replace every `self.job_queue.put((...))` with `self._enqueue(...)`.

**Also:** on `close()`, drain the remaining queue into the spill instead of dropping it:

```python
    def close(self, timeout: float = 5.0, **kwargs):
        self._stop_event.set()
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=timeout)
        # persist anything still in the queue so an offline shutdown loses nothing
        while True:
            try:
                self._spill(*self.job_queue.get_nowait())
            except queue.Empty:
                break
```

## C-5 🟠 CONFIRMED (code) — Gate coordinates are resolution-dependent, resolution never validated

**What is wrong.** `config/gates.json` holds absolute pixel coordinates (e.g. `[679,474]–[1240,386]`) calibrated on a 1280×720 view. `RTSPReader` only forces width/height for **USB camera indices** (`isinstance(self.source, int)`); for RTSP the stream's native resolution is used untouched. If the NVR gives 1920×1080 (or 704×576), every gate line is in the wrong place — **counting is silently wrong**, no error is logged.

**Fix.**

```python
# services/unified_pipeline.py  __init__, right after self.reader = RTSPReader(...)
        # wait briefly for the first frame so we know the true resolution
        for _ in range(50):
            if self.reader.frame_shape:
                break
            time.sleep(0.1)
        shape = self.reader.frame_shape
        if shape:
            self._warn_if_gate_resolution_mismatch(shape)
```

```python
    def _warn_if_gate_resolution_mismatch(self, shape) -> None:
        """Gate coords are absolute pixels: refuse to silently mis-count."""
        if getattr(self, "_gates_calibrated_for", None) is None:
            return                                   # store calibration res. in gates.json going forward
        ch, cw = shape
        gw, gh = self._gates_calibrated_for
        if (ch, cw) != (gh, gw):
            print(f"[UnifiedPipeline] FATAL: stream is {cw}x{ch} but gates were "
                  f"calibrated for {gw}x{gh}. Re-run: python calibrate_gates.py --source ...")
            self.gate_manager.gates = {k: {"gate_points": None, "inside_reference": None}
                                       for k in self.gate_manager.gates}   # disable gates loudly
```

Minimum viable version (no schema change): **scale** the gate points when the resolution differs, and print a warning. Going forward, save `{"resolution": [1280,720], ...}` inside `gates.json` in `UnifiedGateManager.save_config()`.

## C-6 🟠 LIKELY (accuracy) — Gate crossing uses the **head** (top of bbox), not the feet

**What is wrong.** `person["center"] = (cx, y1)` (bbox top). Both `EventManager.process()` and `PersonWithBagExitValidator.update()` use `person["center"]`.

Your gate lines are drawn on the *floor* (the calibrator clicks on the ground). The bbox top is a head-height point, typically **100–200 px above the feet at 720p** — so the crossing is detected when the *head* reaches the line, i.e. roughly one body-length early/late, and a person standing next to the line can flicker between sides (the deadband is only 15 px).

The correct point already exists in the detection dict: `person["person_lower_body_midpoint"] = (cx, y2)`.

**Fix (2 lines).**

```python
# services/unified_pipeline.py  line ~389
ev = self.event_manager.process(t_id, person["person_lower_body_midpoint"],
                                footfall_pts, footfall_ref)
# detectors/person_bag_exit_validator.py  line ~91
midpoint = person["person_lower_body_midpoint"]
```

⚠️ **This changes counting behaviour → re-run `calibrate_gates.py` and re-validate against a labelled clip before deploying.** Expected impact: fewer false triggers for people walking parallel to the gate, more deterministic crossing timestamps. Mark as 🟠 HIGH accuracy impact.

## C-7 🟠 CONFIRMED (code) — `upsert_daily_metrics` every 5 s; two config keys ignored

```python
# unified_pipeline.py:506
if (now_t - self.last_metrics_sync) >= 5.0:      # <-- hardcoded, ignores
    ...                                           # api.metrics_sync_interval_seconds
```

and `insert_occupancy(...)` is called without `interval_minutes=`, so the payload always claims 20.0 even if you change `production.yaml`.

**Why it matters:** 17 280 POSTs/day per outlet for data that changes once a minute; on a slow uplink it competes with event traffic; on the gateway it's 17 280 unnecessary DB writes.

**Fix:**

```python
self.metrics_interval = float(api_cfg.get("metrics_sync_interval_seconds", 60.0))  # __init__
...
if (now_t - self.last_metrics_sync) >= self.metrics_interval:
    ...
    self.db_manager.insert_occupancy(..., interval_minutes=self.occupancy_interval / 60.0)
```

Recommended value on a low-end box: **60 s** (not 5 s).

## C-8 🟠 CONFIRMED (code) — MiVOLO checkpoint missing ⇒ infinite retry + log spam, no fail-fast

`AsyncDemographicWorker._worker_loop()` calls `_ensure_model_loaded()` per item; `Meta.load_from_ckpt()` → `torch.load(ckpt)` raises `FileNotFoundError`; the `except Exception` in `_worker_loop` logs a warning and continues — and the next queued crop repeats the whole load attempt (every 0.25 s while the queue has items). Demographics then never work, `retail_ai.log` grows, CPU burns.

**Fix:**

```python
# mivolo_system/async_worker.py
    def _ensure_model_loaded(self) -> None:
        if self.model is None and not self._load_failed:
            if not os.path.exists(self.checkpoint):
                self._load_failed = True            # set in __init__ = False
                print(f"[AsyncDemographicWorker] FATAL: checkpoint not found: {self.checkpoint}. "
                      f"Demographics DISABLED for this session. "
                      f"(place model_imdb_cross_person_4.22_99.46.pth.tar in mivolo_system/model/)")
                self.stopped = True
                return
            ... existing load code ...

    def submit_crop(self, track_id, crop):
        if self.stopped or self._load_failed:
            return False
```

(Or simply check `os.path.exists(self.checkpoint)` once in `UnifiedRetailPipeline.__init__` and set `enable_demographics=False` with a clear message.)

## C-9 🟠 LIKELY (robustness) — GUI mode: the RAM watchdog only kills its own thread

`UnifiedRetailPipeline.run()` in headless mode calls `sys.exit(3)` to trigger a supervisor restart. But `gui/retail_gui.py._run_pipeline_worker()` runs `pipeline.run()` **inside a thread**, where `sys.exit(3)` raises `SystemExit` — which is *not* caught by the surrounding `except Exception`. The thread dies, `finally` prints "terminated cleanly", and **the Tkinter GUI keeps showing frozen last-known KPIs** with no restart.

**Fix:** in GUI mode, don't `sys.exit`. Instead set `self.running = False`, raise a custom `PipelineRestartRequested`, and have the GUI restart the pipeline (or show a red "restarting" banner). Minimum viable:

```python
# unified_pipeline.py run(), headless RAM branch
if self.perf_monitor.ram_mb > self.ram_restart_mb:
    if getattr(self, "in_gui_thread", False):
        self.running = False
        print("[UnifiedPipeline] RAM limit hit: pipeline stopped (GUI must restart it).")
        return
    sys.exit(3)
```

## C-10 🟡 CONFIRMED (code) — Pending entry/exit never cancelled when a track is lost

`EventManager.evict_lost_tracks()` only pops `previous_side`. `pending_entries` / `pending_exits` survive.

Consequence: a person crosses in, then is lost at 45 s (occlusion) — at 60 s the entry is **validated and POSTed even though the person may have left at 46 s**, with no matching exit → occupancy drifts upward over a day (occupancy = entries − exits is displayed and reported).

**Fix:** decide the policy explicitly. Recommended:

```python
def evict_lost_tracks(self, lost_ids):
    for tid in lost_ids:
        self.previous_side.pop(tid, None)
        # do NOT keep validating events for tracks that no longer exist
        self.pending_entries.pop(tid, None)
        self.pending_exits.pop(tid, None)
```

Trade-off: you may drop a real entry whose track broke (with C-2 fixed, tracks break far less often). Document whichever policy you choose.

## C-11 🟡 CONFIRMED (measurement) — `cv2` uses all cores by default → oversubscription with onnxruntime

`cv2.getNumThreads()` returned **8** on this machine. During every preprocess/display frame OpenCV spawns threads that fight with onnxruntime's `intra_op_num_threads=2` and the capture thread. On a 4-core CPU this causes context-switch latency spikes (higher P95 inference time).

**Fix (add near the top of `app.py`, after imports):**

```python
import cv2
cv2.setNumThreads(1)     # 1 for 4-core boxes; 2 if you see preprocess > 5 ms
```

## C-12 🟡 CONFIRMED (measurement) — Scheduler interval starts *after* work completes

`record_inference_completed()` sets `last_inference_time = now` *after* the work, so the achieved rate = `1/(interval + work)`. Measured: with 200 ms work and a 4 FPS target → **3 passes/s** instead of 4 (25 % throughput loss, and it degrades further on a slow CPU).

**Fix:**

```python
# inference/inference_scheduler.py
    def should_process_next_frame(self) -> bool:
        now = time.time()
        interval = 1.0 / max(1.0, self.current_target_fps)
        if (now - self.last_inference_time) < interval:
            return False
        self.last_inference_time = now      # reserve the slot at START
        return True

    def record_inference_completed(self, latency_ms: float) -> None:
        # (do not touch last_inference_time here — only stats + adaptive backoff)
        ...
```

## C-13 🟡 CONFIRMED (code) — Dead / ignored configuration keys

| Key | Status |
|---|---|
| `camera.target_fps: 15` | **ignored** — local-file playback is hardcoded to `0.033 s` (`_target_interval` computed from `CAP_PROP_FPS` then thrown away: `v_fps` assigned, never used) |
| `camera.buffer_size: 1` | **ignored** (hardcoded `CAP_PROP_BUFFERSIZE=1`, which is fine) |
| `tracking.tracker: "bytetrack"` | **ignored** (hardcoded) |
| `bag_detection.enabled` / `event_driven` | **ignored** |
| `api.metrics_sync_interval_seconds` | **ignored** (see C-7) |
| `inference.width/height` | only used by the torch path |
| `config/settings.py` `MODEL_PATH="models/yolov8l.pt"` | legacy — a 100 MB model constant that misleads readers |

**Fix:** either wire them up or delete them. Dead config is worse than no config: it makes operators believe they tuned something.

Concretely for playback pacing:

```python
# camera/rtsp_reader.py _open_stream()
v_fps = self.cap.get(cv2.CAP_PROP_FPS) or 15.0
if v_fps <= 0 or v_fps > 60:
    v_fps = 15.0
self._target_interval = 1.0 / v_fps
```

This matters for **evaluation**: at a hardcoded 30 FPS a 15 FPS recording plays 2× fast, so a 60 s dwell is satisfied after 30 s of source video → your offline accuracy tests are not measuring what you think.

## C-14 🟡 CONFIRMED (code) — Logging hazards

1. **`_RotatingFile` has no lock.** `print()` (via `_TeeWriter`) and `logging.StreamHandler(log_fh)` both write to the same file object from ≥3 threads. During `_rotate()` another thread can write to a closed handle (exception swallowed by the tee → lost lines) or race `os.replace` on Windows (`PermissionError` → lost log lines).
2. `retail_ai.log` rotation is bounded (5 MB × 3) ✅ but **`restarts.log` grows forever** (append-only from `run_service.bat`).
3. `_TeeWriter.__getattr__` delegates to `self._primary`, which may be `None` → `sys.stdout.encoding` raises `AttributeError` for some libraries.

**Fix (small):** add `threading.Lock` around `write`/`_rotate` in `_RotatingFile`; cap `restarts.log` in the bat (`if exist restarts.log if %~z restarts.log GTR 1048576 del restarts.log`); make `__getattr__` return `None`-safe values.

## C-15 🟡 CONFIRMED (code) — Shutdown drops queued events when offline

`close(timeout=2.0)` gives the writer thread 2 s; with the network down one `_send` attempt can take up to `2×(5 s timeout) + 1 s sleep` ⇒ most queued items are dropped instead of spilled. Fix given in C-4 (`close()` drain → `_spill`).

## C-16 🟡 CONFIRMED (code) — Dependency management risks

1. `requirements.txt` has **no version pins** → a fresh install next month may pull a `supervision` release with a changed `ByteTrack(...)` API and break tracking silently.
2. `mivolo_system/requirements.txt` pins **`torch==2.15.0.dev…+cu130`** (a *CUDA nightly*) and `opencv-python==5.0.0.93` (non-headless) — installing it on the edge PC pulls several GB of GPU libraries you don't need, conflicts with `opencv-python-headless`, and contradicts the "no GPU, ~2 GB RAM" goal.
3. README says Python **3.12** is required; this machine has only **3.13** and the code runs fine on it → the instruction is overly strict/confusing (keep 3.12 if your deployment images use it, but say "3.10–3.13 supported").

**Fix:**

```
# requirements.txt (pin what you actually tested)
numpy>=1.26,<3
opencv-python-headless>=4.9,<4.12
onnxruntime>=1.17,<2
pyyaml>=6.0
requests>=2.31
psutil>=5.9
supervision>=0.21,<0.24        # verify ByteTrack API before bumping
```

```
# mivolo_system/requirements.txt — replace torch line with CPU wheels:
torch==2.4.1+cpu  torchvision==0.19.1+cpu
  --extra-index-url https://download.pytorch.org/whl/cpu
```

## C-17 🟡 CONFIRMED (code) — Secrets hygiene

**Good (keep):** `API_KEY` comes only from the environment; `mask_url()` hides RTSP credentials in every log line; there is a startup warning for cleartext HTTP; no hardcoded password in any `.py` file. ✅

**Problems:**

1. `README.md:130` publishes what looks like a real gateway token: `API_KEY=secret_outlet_token_banasree`. **Rotate it** and replace with `API_KEY=<redacted>`.
2. **`.gitignore` does not cover `edge.env.bat`** (the file `run_service.bat` tells you to create *with the API key in it*). If this repo is ever `git init`-ed, the token gets committed. Add `edge.env.bat` / `*.env.bat`.
3. Gateway is plain `http://` (code warns, good). For production put TLS in front or at minimum restrict the outlet→server path with a firewall/VPN.
4. The camera RTSP URL passed on the command line is visible in the process list (`wmic process`/Task Manager) — prefer `config/camera.yaml` (already supported) over `--source rtsp://admin:pw@...`.
5. No inbound ports are opened by this app (outbound-only) ✅ — keep it that way.

## C-18 🟡 CONFIRMED (code) — `run_service.bat` can silently run the wrong Python

```bat
call venv\Scripts\activate.bat
python app.py ...
```

If `venv/` is missing (it is gitignored), `python` resolves to whatever is on PATH (your README itself warns about Python 3.14) → crash loop every 5 s, forever, with a growing `restarts.log`.

**Fix:**

```bat
@echo off
cd /d "%~dp0"
if exist "%~dp0edge.env.bat" call "%~dp0edge.env.bat"
if not exist "venv\Scripts\python.exe" (
  echo [RetailAI] FATAL: venv not found. Run: python -m venv venv ^&^& pip install -r requirements.txt
  pause & exit /b 1
)
:loop
"venv\Scripts\python.exe" app.py --no-display --no-demographics --config config/production.yaml %*
echo [%date% %time%] exited with %ERRORLEVEL%, restarting in 5s >> restarts.log
if exist restarts.log if %~zrestarts.log GTR 1048576 del restarts.log
timeout /t 5 /nobreak >nul
goto loop
```

(Use the venv python directly — no `activate` needed, and it cannot silently fall back.)

## C-19 🟢 CONFIRMED — Dead code / duplicated code (cleanup, no functional need)

* `BagPersonAassociator.associate()` ≡ `associate_person_bag_exit()` (200 lines duplicated); `draw_confirmed_associations()` ≡ `draw_confirmed_associations_bag_exit()`.
* `detectors/person_detector.py`, `detectors/bag_detector.py`, `services/customer_counter.py`, `services/customer_bag_counter.py` — legacy paths that import **ultralytics+torch** (unavailable on the edge box) and are only reachable via `--legacy-*`.
* `video/threaded_stream.py` — superseded by `camera/rtsp_reader.py`, unused.
* `TrackManager.history`/`get_trail()` and `tracking.trail_length` are maintained but `draw_tracks()` never draws trails → pure per-frame overhead.
* `TrackManager.lost_ids` is written but never read; the `>1000 → clear()` branch is unreachable logic.
* `EventManager.track_aliases`/`get_canonical_id()` — written nowhere.
* `config/settings.py` model constants (yolov8l/yolov8m) unused by the ONNX path.
* `services/unified_pipeline.py` imports `CONFIDENCE, GATE_BAND_PIXELS` but never uses them.

**Recommendation:** delete in a dedicated cleanup pass (Priority 4), not before the P1/P2 fixes.

## C-20 🟢 CONFIRMED — Small logic notes (not bugs, but be aware)

* **Duplicate drawing:** in display mode each person's box is drawn up to 3× (draw_tracks + main loop + associator label). Cost measured at <1 ms, but it's noise.
* **`completed_exits` freezes a track forever** — correct, because `supervision` ByteTrack IDs are monotonically increasing (never reused) — **as long as you keep that tracker**. If you ever switch trackers to one that reuses IDs, re-entry breaks.
* **`print()` is used for operational messages**, `logging` only for health/errors — mixing both means the tee writes some lines twice (once via print→tee, once via logging handler). Cosmetic.
* **`test_scenarios.py` cannot run in a fresh clone** (imports torch-dependent MiVOLO, needs `test_videos/`). Only `test_edge_changes.py` runs. Your "ALL 10 SCENARIO TESTS PASSED" banner is therefore not reproducible today.
* **Occupancy = `entries − exits`** in the HUD: correct only if both counters are complete (see C-3/C-10).

---

# D. Performance Analysis (find the REAL bottleneck)

## D.1 Measured on this machine (Python 3.13, CPU-only, 8 logical cores, cv2 threads = 8)

| Component | Cost | Bottleneck? | Notes |
|---|---|---|---|
| Letterbox preprocess 720p→480² float tensor | **1.0 ms** | No | includes resize+pad+cvtColor+transpose+normalize |
| Post-process (8400 raw boxes → conf filter → Python box build → `cv2.dnn.NMSBoxes`), realistic ~120 candidates | **7.4 ms** | 🟡 becoming one | ~6 ms of it is the Python list comprehension |
| Post-process, pathological (5000 above-threshold boxes) | **56 ms** | Yes (worst case) | happens if conf threshold is lowered a lot |
| Single-slot display frame copy 1280×720 | **0.1 ms** | No | ✅ design is good |
| HUD overlay (frame copy + 20 `putText`) | **0.3 ms** | No | ✅ |
| Resize 720p→640×360 (torch path) | **0.1 ms** | No | |
| **Model inference (yolov8n@480 ONNX)** | **NOT MEASURED** | Almost certainly **YES** | no `models/` in repo → run §9.1 |

## D.2 Estimates for the *target* low-end PC (LABELLED AS ESTIMATES)

Assume Intel N5105/J4125-class (4C, ~2.0–2.9 GHz boost, no AVX-512) — roughly **2.5–4× slower per core** than this machine:

| Stage | Thread | Est. cost @720p | Bottleneck? | Optimization |
|---|---|---|---|---|
| H.264 decode (grab, 15 fps) | capture | 5–15 ms/frame, continuous | 🟡 background load | use the camera's **sub-stream (D1/720p)**, keep TCP transport (already) |
| YUV→BGR retrieve (only when requested) | capture | 2–5 ms | No | already skipped when no consumer needs it ✅ |
| Letterbox + tensor build | AI | **3–8 ms** | No | `cv2.setNumThreads(1)`; already single-resize ✅ |
| **YOLOv8n@480 FP32 inference** | AI | **150–450 ms** | 🔴 **YES** | §G: INT8 / 416 / 320 / OpenVINO EP |
| NMS + Python box loop | AI | 15–40 ms | 🟡 second | vectorize box build (§17.1) |
| ByteTrack | AI | 1–4 ms | No | ✅ |
| Gate/dwell/bag logic | AI | <1 ms | No | ✅ |
| MiVOLO (per crop, if enabled) | demo thread | 100–350 ms | 🟠 eats a whole core | keep off on 2–4 GB boxes (as `run_service.bat` already does) |
| HTTP send + SQLite spill | sender thread | async (50–500 ms net) | No | ✅ non-blocking by design — *except* the `put()` in C-4 |
| Display copy+draw+imshow (GUI mode) | main | 8–20 ms @15 fps | 🟡 GUI mode only | headless in production ✅ |

**Conclusion (estimate):** with `yolov8n@480` FP32 on a low-end CPU you will spend **70–90 % of the AI thread time inside onnxruntime**. Everything else in the AI thread is ≤10 %. The realistic target is **3–5 AI FPS at 50–80 % of one core**, i.e. the configured `target_fps: 4` is appropriate — do **not** raise it before doing §G.

**Biggest non-model CPU consumers on the box:** (1) H.264 decode at camera rate, (2) OpenCV thread pool, (3) MiVOLO if enabled, (4) GUI `imshow`.

## D.3 How to actually measure each stage (do this on the target PC)

`PerformanceMonitor` already collects `inference_ms`, `tracking_ms`, `gate_ms`, `bag_ms`, `demographic_ms`, `capture_fps`, `ai_fps`, `ram_mb`, `cpu_percent` — but **`inference_ms` is measured around `detector.detect()` only** and nothing is logged over time. Add a 10-second stage report inside `_ai_worker_loop_inner` (after `record_inference_completed`):

```python
if now_t - getattr(self, "_stage_log_t", 0) >= 10.0:
    self._stage_log_t = now_t
    m = self.perf_monitor
    print(f"[Stages] inf={m.inference_ms:.0f} track={m.tracking_ms:.1f} "
          f"gate={m.gate_ms:.1f} bag={m.bag_ms:.1f} demo={m.demographic_ms:.1f} "
          f"ai_fps={m.ai_fps:.1f} cam_fps={m.capture_fps:.1f} "
          f"ram={m.ram_mb:.0f}MB cpu={m.cpu_percent:.0f}%")
```

External measurement:

* **Windows:** `typestat -n "\Process(app|python)\%" 1 5` or Task Manager → Performance → Resource Monitor; `powershell "(Get-Process -Id $pid).CPU"` deltas.
* **Linux:** `pidstat -p $(pgrep -f app.py) 1 10`, `uptime`, `free -m`.
* **Per-stage micro:** `python benchmark.py models/yolov8n_480.onnx --threads 2` (already in the repo ✅).
* **End-to-end:** `python tests/long_run_stability_test.py --seconds 600` (already in the repo ✅ — prints RAM drift, AI FPS, latency, and asserts no leak).

---

# E. Computer Vision Analysis — where accuracy is lost

## E.1 What is already correct ✅

* Letterbox preprocessing math verified by `tests/test_edge_changes.py` (ratio/padding round-trip exact) — **don't touch it**.
* Output parsing for YOLOv8 `[1,84,N]`, class-column mapping (`4+cid` for 0/24/26/28) is correct.
* Detections are scaled back into original-frame coordinates correctly (both ONNX and torch paths).
* NMS via `cv2.dnn.NMSBoxes` (C++, fast). Global (cross-class) NMS is acceptable here: a bag inside a person box has IoU ≈ bag/person ≈ 0.05 ≪ 0.45, so it is **not** suppressed.
* Dwell validation (60 s entry / 10 s exit) + reversal cancellation is a sound anti-bounce design, unit-tested (`test_scenarios.py` scenarios 1–4).
* ByteTrack `lost_track_buffer` scaling with `frame_rate=4` gives ~5 s of coasting — reasonable.

## E.2 Where accuracy is being lost (ranked)

| # | Issue | Effect | Severity | Fix |
|---|---|---|---|---|
| 1 | Blank-frame track eviction (C-2) | **missed entries/exits, restarted demographics, lost bag associations** | 🔴 | §C-2 |
| 2 | Head (bbox top) used for gate crossing (C-6) | crossing detected ~1 body-length early/late; flicker for people walking parallel to the line | 🟠 | §C-6 + re-calibrate |
| 3 | Gate coords vs stream resolution (C-5) | total mis-counting if stream isn't 720p | 🟠 | §C-5 |
| 4 | Single confidence threshold 0.40 for **persons and bags** | bags are small & thin ⇒ strong **false negatives** for bag-exit; persons at distance also drop out | 🟠 | per-class thresholds (below) |
| 5 | `yolov8n` @480 for small/far objects | a handbag at 720p ≈ 15 px after letterbox ⇒ below what a nano model resolves; distant persons (<30 px tall) missed | 🟡 | see §G — do **not** jump to a bigger model; measure first |
| 6 | No per-track temporal smoothing of gate side (15 px deadband only) | jitter near the line | 🟡 | require N consecutive frames on the new side (cheap, below) |
| 7 | Pending events survive track loss (C-10) | phantom entries → occupancy drift | 🟡 | §C-10 |
| 8 | 4 AI-FPS | fast crossers (running) can traverse the deadband between two samples → missed crossing | 🟡 | accept, or add motion interpolation; 4 FPS ≈ 250 ms is adequate for walking |
| 9 | Camera-dependent: backlight, motion blur, night | standard low-light degradation | 💡 | camera-side: WDR, IR, shutter 1/100–1/250; app-side: nothing more expensive than nano model will fix |

### Per-class confidence thresholds (recommended, small change)

```python
# inference/onnx_detector.py __init__
        self.conf_by_class = {0: self.conf_threshold,          # person 0.40
                              24: self.conf_threshold * 0.7,    # bag    0.28
                              26: self.conf_threshold * 0.7,
                              28: self.conf_threshold * 0.7}
# detect(): replace the scalar mask
        thr = np.array([self.conf_by_class[int(c)] for c in self._target_cids], dtype=np.float32)
        valid_mask = best_scores >= thr
```

(You may keep `cv2.dnn.NMSBoxes`' score threshold at the *lowest* of those.)

### Side-stability (anti-flicker) for the tripwire

```python
# gate/event_manager.py
MIN_SIDE_CONFIRM_FRAMES = 2

# in process(): store (side, consecutive_count); only commit a transition after
# MIN_SIDE_CONFIRM_FRAMES consecutive observations on the new side.
```

At 4 FPS that adds 250 ms of latency to a crossing — irrelevant given the 60 s/10 s dwell, and it removes most jitter-induced false triggers.

## E.3 Tracking quality (ID switches, occlusion)

* ByteTrack with `track_activation_threshold=0.35` (detections already ≥0.40 ⇒ effectively always activated), `minimum_matching_threshold=0.80`, ~5 s coasting: **appropriate**.
* The track↔detection re-association IoU loop in `TrackManager.track()` is O(tracks×dets) and redundant — `sv.ByteTrack.update_with_detections()` already returns detections with `tracker_id` attached; you could map by index. At <50 objects this costs <1 ms: **leave it** (Low priority).
* ID switches after long occlusions will re-open a *new* track: `previous_side` is rebuilt from the first observation, which is the correct conservative behaviour (no phantom crossing).
* No re-identification model — correct decision for this hardware; the 60 s dwell absorbs most re-ID failures.

---

# F. Low-End Optimization Plan (the target PC)

## F.1 Resource budget (headless, demographics OFF — the configuration `run_service.bat` uses)

| Resource | Expected | Guard |
|---|---|---|
| RAM (Python + numpy + cv2 + onnxruntime + yolov8n) | **~250–450 MB** | `safeguards.ram_limit_mb: 1500`, `ram_restart_mb: 1200` (currently 3500/2400 — too high for a 4 GB box, see below) |
| RAM with MiVOLO enabled | **+400–800 MB** (torch) | keep demographics off unless you have ≥8 GB |
| CPU, headless, 4 FPS, model only | **30–70 % of one core** (estimate) | `inference.cpu_threads: 2`, `cpu_limit_percent: 70` |
| CPU with GUI preview | +10–25 % | headless in production |
| Disk | ~600 MB venv+model; `retail_ai.log` ≤20 MB (rotated ✅); `edge_events.db` capped at 200 k rows ✅ | monitor DB size weekly |

⚠️ **Config tuning you should change for a 4 GB machine:**

```yaml
safeguards:
  ram_limit_mb: 1500        # was 3500 → demographics were never throttled on a 4 GB box
  ram_restart_mb: 1200      # was 2400 → the OS starts swapping before the watchdog fires
  cpu_limit_percent: 70     # was 90 → 90% sustained = thermal throttling on fanless mini PCs
inference:
  cpu_threads: 2            # correct for 4 cores (capture + sender + GUI get the rest)
api:
  metrics_sync_interval_seconds: 60   # was effectively 5
```

For **8 GB** machines with demographics on, `ram_limit_mb: 2500 / ram_restart_mb: 2000` is reasonable.

## F.2 CPU optimization checklist

1. ✅ already: ONNX Runtime with `intra_op_num_threads=2`, `inter_op=1`, `ORT_SEQUENTIAL`, `ORT_ENABLE_ALL`.
2. Add: `cv2.setNumThreads(1)` (C-11).
3. Add: keep OpenCV/ffmpeg decode the only multi-threaded part; don't raise `cpu_threads` above physical cores.
4. ✅ already: inference gating at 4 FPS, adaptive backoff under CPU pressure.
5. Change: scheduler interval-from-start (C-12) so 4 FPS really means 4 FPS.
6. Keep: headless mode (no `imshow`).
7. Optional: set `torch.set_num_threads(cpu_threads)` once at startup if demographics are on (already done inside the worker ✅).

## F.3 Memory optimization checklist

1. ✅ already: single-slot latest frame (no queue growth), bounded ID sets, bounded MiVOLO queue (4), spill row cap, log rotation, `evict_lost_tracks` fan-out.
2. Fix: unbounded `used_bag_ids`/`used_person_ids` in the associator (only person IDs are discarded today) and unbounded `confirmed_exit_ids` in the exit validator — both are int sets (tiny) but should be bounded for 24/7 hygiene:

```python
from gate.event_manager import BoundedIdSet
self.used_bag_ids = BoundedIdSet(max_size=5000)      # supports discard/__contains__/add
self.confirmed_exit_ids = BoundedIdSet(max_size=5000)
```

3. Fix: `recorded_entry_db_ids` / `recorded_db_demographics` (unbounded int sets) → `BoundedIdSet(20000)`.
4. ✅ `finalized_history` bounded at 1000; `person_bag_candidates` cleaned on eviction.

## F.4 Thread/process plan (keep it simple — this is a 4-core box)

| Component | Model | Count |
|---|---|---|
| Camera capture | daemon thread | 1 |
| AI pipeline (preprocess→infer→track→logic) | thread | 1 |
| HTTP sender + SQLite spill | thread | 1 |
| MiVOLO (optional) | thread | 1 (+1 torch intra-op) |
| Main/GUI | main thread | 1 |
| onnxruntime intra-op | 2 | |
| OpenCV intra-op | 1 | |
| **Total processes** | **1** (no multiprocessing — correct for this workload; multiprocessing would duplicate the model memory) | |

**Do not** move inference to a separate process: you'd pay ~100–200 MB for a second interpreter plus IPC copies of 2.7 MB frames for no gain on 4 cores.

---

# G. Model Optimization Plan (best strategy for a CPU-only low-end PC)

## G.1 The ladder (in the order you should try it)

```
yolov8m.pt (640)  ──(too heavy: 20M params, ~80 MB FP32, 0.5–2 s/frame on old CPU)── ✗ never on edge
        │  you already left this behind
        ▼
yolov8n @480 FP32 ONNX  ── current default ── 3.2M params ≈ 12.8 MB weights
        │
        ├─(A) try @416 / @384 / @320     → 1.5–2.5× faster, small recall loss on far/small objects
        │                                   (measure; 480 is a good default for 720p input)
        ├─(B) INT8 dynamic quantization  → model 12.8 MB → ~3.4 MB, RAM ↓, often 1.2–1.8× faster
        │                                   existing tool: export_and_quantize.py --quantize
        │                                   ⚠ accuracy −0.5…−2 mAP typical → verify
        ├─(C) ONNX Runtime OpenVINO EP   → on Intel iGPU/CPU, often +30…80% vs plain CPU EP
        │                                   good if the box has Intel HD/UHD graphics (they all do)
        ├─(D) static QDQ INT8 (calibrated) → more accurate than dynamic INT8, more work
        └─(E) pruning / distillation     → ✗ not worth it: you are already at the smallest
                                            practical YOLO; effort goes into (A)-(C)
```

**Not applicable:** TensorRT (needs NVIDIA GPU), CUDA EP (no GPU — and `CUDA_VISIBLE_DEVICES=-1` is already set ✅).

## G.2 Which one is best? Decision rule

1. Run `python benchmark.py models/yolov8n_480.onnx --threads 2` → record mean/P95/RAM.
2. Run `python export_and_quantize.py --weights yolov8n.pt --imgsz 480 --quantize`, benchmark the `_int8` file the same way.
3. Run the same two for `--imgsz 416`.
4. Pick the config that gives **≥4 FPS at ≤70 % CPU with <2 points of recall loss** on 20 minutes of *your* footage (use `tests/long_run_stability_test.py` + visual check).

**My recommendation for your stated hardware (state this as a recommendation, not a measurement):**

| Priority | Choice | Why |
|---|---|---|
| Default | `yolov8n_480.onnx` FP32, 2 threads, 4 FPS | best accuracy/effort ratio; you already ship it |
| First experiment | `yolov8n_480_int8.onnx` (dynamic INT8) | 4× smaller file (important for limited storage), less RAM, usually faster on CPU — **but verify accuracy on bags**, dynamic quantization hurts thin/small objects more than large ones |
| Second experiment | `yolov8n_416.onnx` | if the CPU can't hold 4 FPS at 480 |
| Advanced | OpenVINO EP | if you have Intel integrated graphics and want free throughput |
| Never | `yolov8m`/`yolov8l` on the edge box | `config/settings.py` still mentions yolov8l — ignore it |

## G.3 Accuracy vs speed trade-off table (typical, must be verified on your data)

| Model/config | Size | RAM (session) | CPU/frame (old CPU, est.) | FPS (est.) | Accuracy |
|---|---|---|---|---|---|
| yolov8**m**@640 FP32 | 52 MB | ~350 MB | 600–1500 ms | <2 | best |
| yolov8**n**@480 FP32 *(current)* | ~13 MB | ~100–150 MB | 150–400 ms | 3–5 | good for persons, weak for small bags |
| yolov8**n**@480 INT8 | ~3.5 MB | ~60–100 MB | 100–280 ms | 4–7 | −0.5…−2 mAP (est.) |
| yolov8**n**@416 FP32 | ~13 MB | ~100 MB | 110–300 ms | 4–8 | slightly lower recall on far persons |
| yolov8**n**@320 FP32 | ~13 MB | ~90 MB | 70–180 ms | 6–12 | noticeably lower small-object recall |

*(all FPS/RAM columns are ESTIMATES until you run §9.)*

## G.4 Non-model inference wins

* **Batch size must stay 1** ✅ (already).
* **Frame skip 4 FPS** ✅ (already) — the single biggest CPU lever, already applied.
* **ROI inference:** only worth it if a large part of the frame can never contain a person (e.g. ceiling/signage). Your gates are mid-frame, so ROI would clip people → **not recommended** unless the camera is very wide.
* **Vectorize the post-processing Python loop** (§17.1) to shave ~5–25 ms/frame.

---

# H. PC Deployment Guide (step-by-step)

> Assumptions (stated, not guessed): target = Windows 10/11 64-bit mini PC, 4 cores, 8 GB RAM (4 GB minimum, headless, demographics off), one RTSP camera, outbound-only network to your gateway. Linux variant at H.8.

## Step 1 — Prepare the PC

1. **OS:** Windows 10/11 64-bit (LTSC/Pro preferred, disable sleep/hibernation: `powercfg /change standby-timeout-ac 0`, `hibernate off`). For Linux: Ubuntu 22.04 LTS server.
2. **GPU drivers:** none needed (CPU-only). Do *not* install CUDA.
3. **Python:** install **Python 3.12 (64-bit)** from python.org, tick *“Add python.exe to PATH”*.
   Verify: `py -3.12 --version` (your README's `py -3.12` convention is good — keep it).
4. **Page file:** keep automatic (Windows manages it); do not disable swap — the RAM watchdog relies on the process actually ballooning.
5. **Power plan:** High Performance (mini PCs throttle otherwise).
6. **Camera:** confirm you can open the stream first:
   `ffplay rtsp://USER:PASS@CAM_IP:554/...` (or VLC). Prefer an **ONVIF sub-stream at 1280×720 @ 15 fps, H.264 (Baseline/Main), CBR 1–2 Mbps, GOP 15–30, no camera-side analytics**.

## Step 2 — Install dependencies

```powershell
cd C:\path\to\Low_Resource_ai_system

py -3.12 -m venv venv
.\venv\Scripts\python.exe -m pip install --upgrade pip

# Runtime only (edge box) — this is the one you want:
.\venv\Scripts\python.exe -m pip install -r requirements.txt

# ONLY on a build machine (model export / torch fallback / MiVOLO):
# .\venv\Scripts\python.exe -m pip install -r requirements-dev.txt
```

Notes:
* `requirements.txt` is currently unpinned — after your first successful install, freeze it: `.\venv\Scripts\python.exe -m pip freeze > requirements.lock.txt` and deploy *that* on the other PCs.
* If you enable demographics on Linux, install **CPU-only torch** (never `mivolo_system/requirements.txt` as-is — see C-16).

## Step 3 — Prepare the model

The model file lives in **`models/`** (gitignored → must be produced or copied).

**Option A — build machine (has internet):**
```powershell
.\venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\venv\Scripts\python.exe export_and_quantize.py --weights yolov8n.pt --imgsz 480
# optional INT8:
.\venv\Scripts\python.exe export_and_quantize.py --weights yolov8n.pt --imgsz 480 --quantize
```
→ produces `models/yolov8n_480.onnx` (+ `models/yolov8n_480_int8.onnx`).

**Option B — air-gapped edge PC:** copy the `models/` folder (and `mivolo_system/model/*.pth.tar` if demographics are on) from the build machine. ~15 MB.

**Load-time best practice (already implemented ✅):** the session is created once in `ONNXDetector.__init__` with `ORT_ENABLE_ALL` and pinned thread counts; nothing reloads per frame. Keep it that way.

## Step 4 — Configure

1. `config/production.yaml` — set values from §F.1 (`ram_limit_mb`, `cpu_limit_percent`, `metrics_sync_interval_seconds`), `camera.rtsp_url` (or leave it empty and use `config/camera.yaml`).
2. **Secrets:** create `edge.env.bat` (from `deploy\edge.env.example.bat`):
   ```bat
   set API_GATEWAY_URL=https://your-gateway.example.com
   set API_KEY=real_token_here
   set SHOP_ID=2
   ```
   **Add `edge.env.bat` to `.gitignore` before you ever `git init`.**
3. **Calibrate gates at the production resolution:**
   ```powershell
   .\venv\Scripts\python.exe calibrate_gates.py --source "rtsp://..."
   ```
   Click the two line points + an inside reference, press `s` → `config/gates.json`.
   ⚠️ Re-calibrate after any change of camera, resolution, or crop (C-5, C-6).

## Step 5 — Run the application

```powershell
# production (headless, no demographics) — the recommended edge mode:
.\venv\Scripts\python.exe app.py --no-display --no-demographics --config config/production.yaml

# with a specific camera:
.\venv\Scripts\python.exe app.py --source "rtsp://admin:***@192.168.1.100:554/..." --no-display --no-demographics

# development with preview:
.\venv\Scripts\python.exe app.py

# demographics on (needs ≥8 GB RAM):
.\venv\Scripts\python.exe app.py --no-display
```

Expected startup log sequence:
```
[ONNXDetector] Loading model models/yolov8n_480.onnx with 2 CPU threads...
[ONNXDetector] Model loaded in 0.5s! Input size: 480x480
[ApiManager] Gateway: http://... | Outlet ID: 2 | Status: ONLINE
[UnifiedPipeline] Running in Headless Mode ...
[Health] uptime=30s ai_fps=4.0 inf_ms=180 ram=310MB cpu=42% cam=ON reconnects=0 ...
```

## Step 6 — Start automatically at boot

**Windows (recommended): the supplied Task Scheduler script** — it already sets `RestartCount 999`, `StartWhenAvailable`, `ExecutionTimeLimit=Zero`, runs as SYSTEM:

```powershell
# elevated PowerShell:
powershell -ExecutionPolicy Bypass -File deploy\install_windows_task.ps1
Start-ScheduledTask -TaskName RetailAI_Service
```

Why Task Scheduler over the Startup folder: it survives user logoff, restarts on failure, and runs elevated. Why not a Windows *service*: you'd need `nssm`/`pywin32` — not worth it for a single exe/script.

**Linux:**

```bash
sudo cp deploy/retailai.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now retailai
journalctl -u retailai -f
```
The unit already has `Restart=always`, `RestartSec=5`, `MemoryMax=1500M` ✅ (adjust `MemoryMax` to ~60 % of RAM).

## Step 7 — Automatic recovery (what exists vs what to add)

| Failure | Existing mechanism | Status | Add |
|---|---|---|---|
| App crashes / exits | `run_service.bat` loop (5 s) + Task Scheduler `RestartCount 999` / systemd `Restart=always` | ✅ | startup guard in C-18 (venv check), backoff after N rapid crashes |
| RAM grows | `ram_restart_mb` → `sys.exit(3)` → supervisor restarts (headless only) | ✅ 🟡 | GUI mode fix (C-9); lower thresholds (§F.1) |
| Camera/RTSP drops | `RTSPReader` reconnect every 2 s, TCP transport, no crash | ✅ | log reconnect rate (already in `[Health]`) |
| Network drops | in-memory queue → SQLite spill (200 k row cap) → replay every 10 s | ✅ 🟡 | non-blocking `put` (C-4), drain-on-shutdown (C-15) |
| Power loss | boot trigger + `StartWhenAvailable` + SQLite WAL | ✅ | persist counters (C-3) |
| Wrong Python / missing venv | — | ❌ | C-18 |
| Missing model/video assets | — | ❌ | C-1 |
| Central server needs to know the edge died | — | ❌ | 60 s heartbeat POST + `status.json` (§12) |
| Disk full (`edge_events.db`) | row cap 200 000 ✅ / logs rotated ✅ | ✅ | weekly size check in health log |

---

# I. Benchmarking Guide

## I.1 Model-only (run before & after any model change)

```powershell
.\venv\Scripts\python.exe benchmark.py models/yolov8n_480.onnx --threads 2 --iters 300
.\venv\Scripts\python.exe benchmark.py models/yolov8n_480_int8.onnx --threads 2 --iters 300
.\venv\Scripts\python.exe benchmark.py models/yolov8n_416.onnx --threads 2 --iters 300
```
Output to record: `load time`, `RAM added`, `latency mean`, `p95`, `max throughput`, `process CPU`.

## I.2 End-to-end pipeline (run before & after any code change)

```powershell
# 10 minutes, prints 5-second samples + leak assertions:
.\venv\Scripts\python.exe tests\long_run_stability_test.py --seconds 600
```
Record: startup RAM, peak RAM, RAM drift (net), average CPU, average AI FPS, entries/exits recorded, reconnects.

## I.3 Accuracy benchmark (you must build this — I will not invent numbers)

1. Take **30–60 minutes of real footage** covering day/night, crowd/no-crowd.
2. Hand-label: every crossing with timestamp + direction (a CSV is enough).
3. Run the pipeline on the file (`--source "clip.avi"`), dump `validated ENTRY/EXIT` lines from `retail_ai.log`.
4. Compute **precision / recall / F1** for entries and exits:

```python
# accuracy_benchmark.py (sketch — add to repo as Priority 3)
import re, csv, sys
events = []
for line in open("retail_ai.log", encoding="utf-8", errors="ignore"):
    m = re.search(r"VALIDATED (ENTRY|EXIT) RECORDED: Track (\d+)", line)
    if m: events.append((m.group(1), m.group(2)))
truth = list(csv.DictReader(open("ground_truth.csv")))   # type,time
# match predicted to truth within ±2 s tolerance, direction must agree
tp = fp = fn = 0
...
print(f"entry precision={tp/(tp+fp):.3f} recall={tp/(tp+fn):.3f}")
```

5. Run it for every candidate model/threshold (§G) and keep the best F1 that still hits your FPS budget.

## I.4 Metrics to track for the life of the system

| Metric | Source (already exists?) | Frequency |
|---|---|---|
| AI FPS / capture FPS | `PerformanceMonitor` ✅ | log every 10 s (add §D.3) |
| Inference latency mean + P95 | `scheduler.last_latency_ms` ✅ (mean only — keep a 100-sample deque for P95) | 10 s |
| RAM / CPU % | `PerformanceMonitor` ✅ | 1 s internal, 30 s log |
| Camera status + reconnect count | `reader.is_connected/reconnect_count` ✅ | 30 s |
| Detection count / errors | `LightweightMonitor` ✅ | 30 s |
| Uptime | `LightweightMonitor.start_time` ✅ | 30 s |
| Queue depths (DB/demo) | `PerformanceMonitor` ✅ | 10 s |
| GPU | n/a (no GPU) | — |

**Startup / model-load time:** already printed (`[ONNXDetector] Model loaded in X s`) ✅.

---

# J. Hardware Recommendation

*(Stated as a recommendation; not derived from your repo — your repo only says "Intel Celeron/Pentium/i3/i5, 4 cores, ~2 GB usable".)*

| Tier | CPU | RAM | Storage | Notes |
|---|---|---|---|---|
| **Minimum** (headless, demographics OFF, 1 camera) | Intel Celeron **N4020/N5095** class or Core i3-6100/7100 (4 threads) | **4 GB** (2 GB free after OS) | 64 GB SSD (eMMC acceptable) | expect 2–4 AI FPS, CPU 60–90 % |
| **Recommended** (headless, demographics ON, 1 camera, 24/7) | Intel **N100/N95** or Core i3-10100 / Ryzen 3 4300U class | **8 GB** | 128–256 GB SSD | comfortable 4+ FPS with headroom for MiVOLO |
| **Comfortable** (2 cameras or heavy demographics) | i5-10400 / N305 class | 16 GB | 256 GB SSD | |

Additional requirements:
* **Wired Ethernet** (never Wi-Fi for RTSP).
* UPS or at least a BIOS setting to power-on after AC loss (`AC Power Recovery → On`).
* Ventilation: fanless mini PCs throttle under sustained 70 %+ CPU — set `cpu_limit_percent: 70`.
* **No NVIDIA GPU is required and none should be bought for this system.** Intel integrated graphics is only useful if you adopt the OpenVINO EP (§G).

---

# K. Final Production Architecture

```
┌──────────────── EDGE PC (per outlet) ─────────────────┐
│ Supervisor: Task Scheduler / systemd (Restart=always) │
│   └─ run_service.bat (venv guard, 5 s backoff, log cap)│
│        └─ app.py --no-display --no-demographics        │
│             ├ RTSPReader (TCP, reconnect, latest-frame)│
│             ├ AI thread: letterbox→ONNX(INT8?)→NMS     │
│             │            →ByteTrack→gate/dwell→bags    │
│             ├ Counter store: SQLite (restores on boot) │
│             ├ Sender thread: queue→HTTP→spill→replay   │
│             ├ MiVOLO worker (optional, ≥8 GB only)     │
│             ├ status.json + heartbeat (60 s)           │
│             └ logs: retail_ai.log (rotated, locked)    │
│ Assets: models/yolov8n_480[_int8].onnx, config/, gates │
└────────────────────────┬───────────────────────────────┘
                         │ outbound HTTPS only
                         ▼
┌──────────────── CENTRAL SERVER ────────────────────────┐
│ FastAPI gateway (auth via X-API-Key, rate-limit)       │
│   entries / exits / demographics / occupancy / metrics │
│   + heartbeat endpoint → "outlet offline" alerting     │
│ PostgreSQL (indexes on shop_id+timestamp, daily rollup)│
└────────────────────────────────────────────────────────┘
```

Design rules for this final form:
1. **One process, five threads, zero child processes.**
2. Everything bounded: frame slot = 1, demo queue = 4, event queue = 10 000 (+ disk spill), ID sets bounded, logs rotated.
3. **Nothing in RAM is the source of truth** — counters live in SQLite, events live in the gateway.
4. Every failure mode has a recovery path that has been *tested* (§H.7 table).
5. Monitoring is ≤1 % CPU (one psutil sample per second + one log line per 30 s).

---

# 16. Prioritized Improvement Plan

## 🔴 Priority 1 — Must Fix (crashes / wrong results / freezes)

| # | Item | Difficulty | Perf impact | Accuracy impact | RAM/CPU impact | Effort |
|---|---|---|---|---|---|---|
| P1-1 | Blank-frame track eviction (C-2) + regression test | Easy | none | 🔺 **large** (recovers missed entries/exits) | none | 30 min |
| P1-2 | Missing model/assets → clear fail-fast (C-1) + preflight source check | Easy | none | none (but system actually starts) | none | 1 h |
| P1-3 | Persist & restore counters (C-3) | Medium | none | 🔺 prevents corrupted central data | +<1 MB | 2–3 h |
| P1-4 | Non-blocking queue enqueue + drain-on-shutdown (C-4, C-15) | Easy | prevents freeze | none | none | 1 h |
| P1-5 | Gate resolution validation (C-5) | Easy | none | 🔺 prevents silent total mis-count | none | 1–2 h |

## 🟠 Priority 2 — High Impact (FPS / latency / accuracy / reliability)

| # | Item | Difficulty | Perf impact | Accuracy impact | RAM/CPU | Effort |
|---|---|---|---|---|---|---|
| P2-1 | Gate crossing on feet, not head (C-6) + **re-calibrate + re-validate** | Easy | none | 🔺 positive (needs re-tune) | none | 1 h + validation |
| P2-2 | Per-class confidence thresholds (person 0.40 / bag 0.28) (E.2) | Easy | +1–3 ms | 🔺 bag recall ↑, some FP ↑ | none | 1 h |
| P2-3 | `cv2.setNumThreads(1)` (C-11) | Trivial | 🔺 lower P95 latency | none | CPU steadier | 5 min |
| P2-4 | Scheduler interval-from-start (C-12) | Easy | 🔺 +20–25 % effective FPS | none | none | 15 min |
| P2-5 | Config: RAM/CPU limits, metrics interval (§F.1, C-7) | Easy | 🔺 fewer throttles/restarts | none | CPU ↓ | 15 min |
| P2-6 | MiVOLO fail-fast on missing checkpoint (C-8) | Easy | log/CPU ↓ | none | none | 30 min |
| P2-7 | Pin dependencies; CPU-only torch note (C-16) | Easy | stability | none | avoids 2 GB CUDA install | 30 min |
| P2-8 | `run_service.bat` venv guard + `restarts.log` cap (C-18) | Easy | reliability | none | none | 30 min |
| P2-9 | Cancel pending events on track loss (C-10) | Easy | none | 🔺 occupancy accuracy | none | 30 min |
| P2-10 | GUI watchdog fix (C-9) | Easy | reliability | none | none | 30 min |
| P2-11 | Model experiments: INT8 / 416 (§G) | Medium | 🔺🔺 1.2–2× FPS | ⚠ verify (−0.5…−2 mAP est.) | RAM ↓ | 2–4 h incl. benchmark |

## 🟡 Priority 3 — Recommended

| # | Item | Difficulty | Benefit |
|---|---|---|---|
| P3-1 | Side-stability confirm frames for tripwire (E.2) | Medium | kills jitter false-triggers |
| P3-2 | Stage-timing log every 10 s (§D.3) + P95 latency deque | Easy | makes every future optimization measurable |
| P3-3 | `status.json` + 60 s heartbeat to gateway | Easy | central "outlet offline" alerting |
| P3-4 | Thread-safe log rotation lock (C-14) | Easy | no lost log lines |
| P3-5 | Bound `used_bag_ids` / `confirmed_exit_ids` / `recorded_*` sets (§F.3) | Easy | 24/7 hygiene |
| P3-6 | Accuracy benchmark script vs hand-labeled clip (§I.3) | Medium | objective accuracy before/after any change |
| P3-7 | Vectorized post-processing box build (§17.1) | Medium | −5…−25 ms/frame |
| P3-8 | Wire up or delete dead config keys (C-13) | Easy | operator trust |
| P3-9 | Rotate/remove README example API token (C-17) | Trivial | security |
| P3-10 | Make `test_scenarios.py` runnable without torch (mock MiVOLO import) | Easy | CI-able tests |

## 🟢 Priority 4 — Optional

| # | Item | Benefit |
|---|---|---|
| P4-1 | Delete legacy/duplicated code (C-19): `associate()`, legacy detectors, `threaded_stream.py`, unused trails | −300 KB, clarity |
| P4-2 | OpenVINO EP experiment (§G.2) | +30–80 % on Intel iGPU |
| P4-3 | Tkinter GUI: restart pipeline on watchdog exit; show real "restarting" state | UX |
| P4-4 | Map ByteTrack→detection by index instead of IoU search | micro |
| P4-5 | Per-class NMS | micro |
| P4-6 | ROI inference (only if scene allows) | 10–30 % |

**Do NOT do:** pruning/distillation, multiprocessing, TensorRT, PostgreSQL/MySQL on the edge box, a message broker, Docker (on Windows mini PCs), or raising `target_fps` above 4 before §G is done.

---

# 17. Actual code changes (before → problem → after)

### 17.1 Post-processing vectorization (P3-7)

```python
# BEFORE (Python loop over every above-threshold box)  ~6 ms of the 7.4 ms total
for box, score, cid in zip(valid_boxes, valid_scores, valid_classes):
    cx, cy, bw, bh = box
    x1 = ((cx - bw / 2.0) - pad_x) / ratio * sx
    ...
    boxes_list.append([int(x1), int(y1), int(w_box), int(h_box)])

# AFTER (vectorised; only the final NMS list stays in Python)
cx, cy, bw, bh = valid_boxes[:, 0], valid_boxes[:, 1], valid_boxes[:, 2], valid_boxes[:, 3]
x1 = ((cx - bw / 2.0) - pad_x) / ratio * sx
y1 = ((cy - bh / 2.0) - pad_y) / ratio * sy
w_box = (bw / ratio) * sx
h_box = (bh / ratio) * sy
boxes_list = np.stack([x1, y1, w_box, h_box], axis=1).astype(np.int32).tolist()
# (keep the existing clamping inside the final reconstruction loop)
```
Expected saving: 3–15 ms/frame on a low-end CPU (measured 6 ms of pure Python on this machine).

### 17.2 Blank-frame hysteresis (P1-1) — full corrected function

```python
def track(self, person_detections, bag_detections=None):
    all_dets, is_person_flags = [], []
    for p in person_detections:
        all_dets.append((p["bbox"], p.get("confidence", 0.9), 0, p)); is_person_flags.append(True)
    if bag_detections:
        for b in bag_detections:
            all_dets.append((b["bbox"], b.get("confidence", 0.8), 26, b)); is_person_flags.append(False)

    tracked_persons, tracked_bags = [], []
    current_active_ids: Set[int] = set()

    if self.tracker is None:
        if not getattr(self, "_warned_no_tracker", False):
            print("[TrackManager] ERROR: supervision missing -> tracking disabled")
            self._warned_no_tracker = True
        return tracked_persons, tracked_bags, set()

    if all_dets:                       # <-- only run ByteTrack when there is something to track
        xyxy = np.array([d[0] for d in all_dets], dtype=np.float32)
        conf = np.array([d[1] for d in all_dets], dtype=np.float32)
        cls = np.array([d[2] for d in all_dets], dtype=np.int32)
        tracked = self.tracker.update_with_detections(sv.Detections(xyxy=xyxy, confidence=conf, class_id=cls))
        ... (unchanged IoU re-association code) ...

    self.update(tracked_persons)
    for tid in current_active_ids:
        self.missing_counts[tid] = 0
        self.active_ids.add(tid)

    newly_lost = set()                 # <-- hysteresis ALWAYS runs, even with 0 detections
    for tid in list(self.active_ids):
        if tid not in current_active_ids:
            self.missing_counts[tid] = self.missing_counts.get(tid, 0) + 1
            if self.missing_counts[tid] >= self.max_missing_frames:
                newly_lost.add(tid); self.active_ids.discard(tid)
                self.missing_counts.pop(tid, None); self.history.pop(tid, None)
    self.lost_ids.update(newly_lost)
    return tracked_persons, tracked_bags, newly_lost
```

### 17.3 Non-blocking enqueue (P1-4)

```python
# BEFORE
self.job_queue.put(("POST", "/api/v1/entries", payload))          # can block the AI thread forever

# AFTER
self._enqueue("POST", "/api/v1/entries", payload)                 # put_nowait, spill on Full
```

### 17.4 FPS-correct scheduler (P2-4)

```python
# BEFORE: last_inference_time = now   # set AFTER work  -> 1/(interval+work)
# AFTER : reserve the slot in should_process_next_frame() BEFORE work runs
```

### 17.5 Feet-based crossing (P2-1)

```python
# BEFORE: ev = self.event_manager.process(t_id, person["center"], ...)
# AFTER : ev = self.event_manager.process(t_id, person["person_lower_body_midpoint"], ...)
```
→ then re-run `calibrate_gates.py` and the accuracy benchmark.

*(Applying only these five changes addresses Priority 1 completely and most of Priority 2. Nothing else in the pipeline needs restructuring.)*

---

# 18. Final Deliverables Checklist

| Request | Where |
|---|---|
| A. System Diagnosis | §A |
| B. Architecture diagrams (current + recommended) | §B |
| C. Critical issues (prioritized, with fixes) | §C (20 findings, 5 🔴) |
| D. Performance analysis + bottleneck table | §D (measured vs estimate clearly separated) |
| E. Computer vision analysis | §E |
| F. Low-end optimization plan | §F |
| G. Model optimization plan | §G |
| H. Step-by-step PC deployment guide | §H (Windows primary, Linux alternate) |
| I. Benchmarking guide + scripts | §I (uses your existing `benchmark.py` + `long_run_stability_test.py`) |
| J. Hardware recommendation | §J |
| K. Final production architecture | §K |
| Prioritized improvement plan (P1–P4 with effort/impact) | §16 |
| Actual code changes | §17 + each §C finding |
| Implementation roadmap | below |

## Implementation Roadmap

**Week 1 — Stop the bleeding (P1)**
1. P1-1 blank-frame hysteresis fix + regression test (30 min → largest accuracy win).
2. P1-2 fail-fast for missing model/source (1 h → system boots).
3. P1-4 non-blocking enqueue + shutdown drain (1 h → no freezes).
4. P1-5 gate resolution guard (1–2 h → no silent mis-counting).
5. P1-3 counter persistence (2–3 h → central data stays correct).
*Verify:* run `tests/test_edge_changes.py` + your scenario tests + a 30 min video clip; compare entry/exit counts before/after.

**Week 2 — Make it fast and honest (P2)**
6. P2-3/P2-4/P2-5/P2-8 (all < 30 min each).
7. P2-1 feet crossing + **re-calibrate gates** + validate on labeled clip.
8. P2-2 per-class thresholds + validate bag-exit recall.
9. P2-11: benchmark FP32 vs INT8 vs 416 with `benchmark.py`; deploy the winner.
10. P2-6/P2-7/P2-9/P2-10.

**Week 3 — Production hardening (P3)**
11. P3-1 side stability, P3-2 stage logging, P3-3 heartbeat, P3-4 log locking, P3-6 accuracy benchmark, P3-8/9/10.

**Week 4 — Deploy**
12. Build model on the build PC → copy `models/`; install venv on target PC per §H; calibrate gates at production resolution; install Task Scheduler task; run 24 h soak with `long_run_stability_test.py --seconds 86400` (or watch `[Health]` lines); confirm RAM drift < 50 MB and AI FPS ≥ 4; only then enable demographics if RAM allows.

**Definition of done:** boots unattended after power loss, ≥4 AI FPS at ≤70 % CPU, RAM < 1.2 GB (no demographics) with zero drift over 7 days, camera/network/power outages recover automatically, entry/exit F1 ≥ your business threshold on the labeled clip, and central metrics never go backwards.
