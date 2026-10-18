# System Architecture & Workflow Diagrams
=========================================

This document contains visual architectural blueprints and operational workflow diagrams for the **OmniRetail AI Suite**.

---

## 1. High-Level System Architecture

```mermaid
graph TB
    subgraph EdgeDevice["🏢 Retail Outlet (Edge Mini PC — Windows 10/11)"]
        subgraph Cameras["Edge CCTV Cameras"]
            CAM1["📹 Store Entrance / Gate Camera<br/>(RTSP / TCP)"]
            CAM2["📹 Staff Attendance Terminal Camera<br/>(RTSP / USB)"]
        end

        subgraph GUILayer["OmniRetail Desktop GUI (Tkinter/TTK)"]
            APP["Master App (unified_gui.py)"]
            TAB1["Tab 1: Executive Dashboard<br/>(Live Store KPIs & Dual Feeds)"]
            TAB2["Tab 2: Customer Analytics<br/>(Virtual Gates & Crossings)"]
            TAB3["Tab 3: VisionGuard Attendance<br/>(Staff Roster & Registration)"]
            TAB4["Tab 4: Settings & Diagnostics<br/>(Config & 1-Click Self-Test)"]
        end

        subgraph AnalyticsEngine["👥 Customer Analytics Engine (FootFall_AgeGender)"]
            RTSP1["Decoupled RTSP Reader"]
            YOLO["YOLOv8 ONNX (320x320)<br/>Person & Bag Detector"]
            TRACK["ByteTrack Multi-Object Tracker"]
            GATE["Virtual Gate Manager<br/>(Entry / Exit / Dwell / Bag)"]
            MIVOLO["MiVOLO Profiler<br/>(Age & Gender Estimation)"]
        end

        subgraph AttendanceEngine["👤 VisionGuard Attendance Engine (Attendence System)"]
            RTSP2["Sampled RTSP Reader (2-5 FPS)"]
            YUNET["YuNet ONNX Face & 5-Landmark Detector"]
            QUAL["Quality Assessor (Sharpness, Yaw, Roll)"]
            SFACE["SFace 128-d Embedding Extractor"]
            TEMP["Temporal Multi-Frame Confirmation"]
        end

        subgraph LocalEdgeStore["💾 Local Edge SQLite Persistence"]
            DB1[("edge_events.db<br/>Footfall & Occupancy")]
            DB2[("attendance.db<br/>Staff Roster & Punch Audit")]
            DISK["Local Disk Storage<br/>(Staffs/ & Aligned Face Chips)"]
        end

        subgraph APIClient["📡 Async HTTP Event Dispatchers"]
            CLIENT1["Footfall HTTP Client<br/>(Background Queue)"]
            CLIENT2["AttendanceApiClient<br/>(Non-blocking Daemon Thread)"]
        end
    end

    subgraph CloudCloud["☁️ Enterprise Central Cloud Gateway"]
        GATEWAY["API Gateway (FastAPI / Nginx)<br/>https://ai.prangroup.com"]
        FALLBACK["Direct Fallback Endpoint<br/>http://ai.prangroup.com:8021"]
        MAINDB[("Central Enterprise Database<br/>(PostgreSQL / Oracle HRIS)")]
    end

    CAM1 --> RTSP1
    CAM2 --> RTSP2

    RTSP1 --> YOLO --> TRACK --> GATE
    GATE --> MIVOLO
    GATE --> DB1
    GATE --> CLIENT1

    RTSP2 --> YUNET --> QUAL --> SFACE --> TEMP
    TEMP --> DB2
    TEMP --> DISK
    TEMP --> CLIENT2

    APP --- TAB1
    APP --- TAB2
    APP --- TAB3
    APP --- TAB4

    AnalyticsEngine -.->|Real-time Frames| TAB1
    AttendanceEngine -.->|Annotated Video| TAB1

    CLIENT1 ==>|POST /api/v1/events/footfall| GATEWAY
    CLIENT2 ==>|POST /api/v1/attendance| GATEWAY
    GATEWAY -.->|Retry Fallback| FALLBACK
    GATEWAY --> MAINDB
    FALLBACK --> MAINDB
```

---

## 2. End-to-End Operational Workflow

```mermaid
sequenceDiagram
    autonumber
    actor Staff as Store Staff / Customer
    participant Cam as CCTV Camera (RTSP)
    participant Reader as RTSP Reader (TCP)
    participant AI as Edge Vision AI Pipeline
    participant GUI as Desktop GUI Display
    participant LocalDB as Local SQLite Database
    participant API as Async HTTP Client
    participant Cloud as Central Retail AI Cloud

    Note over Cam,Cloud: 1. Continuous Video Ingestion
    Cam->>Reader: Continuous H.264 / RTSP video stream
    Reader->>AI: Fetches sampled frame (2-5 FPS for Attendance, 10-15 FPS for Footfall)

    Note over AI: 2. Edge AI Inference & Logic
    alt Staff Attendance Pipeline
        AI->>AI: YuNet detects face & 5 landmarks
        AI->>AI: Quality assessor checks blur, roll, yaw
        AI->>AI: SFace aligns 112x112 chip and extracts 128-d embedding
        AI->>AI: Matches against cached in-memory staff gallery
        AI->>AI: Temporal tracker confirms identity across consecutive frames
    else Customer Footfall Pipeline
        AI->>AI: YOLOv8 detects persons and carried bags
        AI->>AI: ByteTrack maintains trajectory and identity ID
        AI->>AI: Evaluates line crossing & dwell anti-bounce rules
        AI->>AI: MiVOLO estimates age and gender
    end

    Note over AI,GUI: 3. Visual Overlay & UI Feedback
    AI->>GUI: Renders color-coded bounding boxes and badges on canvas
    GUI-->>Staff: Visual real-time verification feedback

    Note over AI,Cloud: 4. Local Persistence & Async Cloud Dispatch
    AI->>LocalDB: Write event record (instant edge audit safety)
    AI->>API: Queue event payload (asynchronous, non-blocking)
    API->>Cloud: POST payload with X-API-Key header
    alt Cloud returns 201 Created
        Cloud-->>API: 201 Created (Success JSON)
        API->>LocalDB: Mark event as synced
    else Primary URL fails / times out
        API->>Cloud: Retries against fallback endpoint (Port 8021)
    end
```

---

## 3. VisionGuard Staff Attendance 8-Stage Pipeline

```mermaid
flowchart TD
    S1["Stage 1: RTSP Input<br/>TCP stream from Terminal Camera"] --> S2["Stage 2: Frame Sampling<br/>2–5 FPS throttle to preserve CPU"]
    S2 --> S3["Stage 3: YuNet Face & Landmark Detection<br/>Bounding box [x,y,w,h] + 5 facial points"]
    
    S3 --> S4{"Stage 4: Quality Assessment<br/>Blur > threshold?<br/>Roll & Yaw within bounds?<br/>Face size > 40x40?"}
    
    S4 -- "❌ Failed Quality" --> REJ["Log Quality Failure to SQLite<br/>Draw RED Rejection Box on UI<br/>(Skip Recognition)"]
    
    S4 -- "✅ Passed Quality" --> S5["Stage 5: Alignment & SFace Embedding<br/>Affine transform to 112x112 chip<br/>Extract 128-d L2 normalized vector"]
    
    S5 --> S6["Stage 6: In-Memory Gallery Matching<br/>Fast cosine similarity comparison<br/>Threshold: Cosine >= 0.380"]
    
    S6 --> S7{"Stage 7: Temporal Confirmation<br/>Same person recognized in<br/>N consecutive frames?"}
    
    S7 -- "⏳ Streak Pending" --> DRAW_PENDING["Draw YELLOW/ORANGE Box<br/>Status: PENDING"]
    
    S7 -- "🎉 Confirmed Streak" --> S8["Stage 8: Persistence & Cloud Dispatch"]
    
    subgraph PersistenceStage["Stage 8 Actions"]
        S8A["Save aligned 112x112 chip & .npy to disk"]
        S8B["Insert punch record to SQLite attendance.db"]
        S8C["Spawn background thread in AttendanceApiClient"]
        S8D["POST JSON punch to https://ai.prangroup.com/api/v1/attendance"]
        S8E["Draw GREEN Verified Box on Desktop UI"]
    end
    
    S8 --> S8A
    S8 --> S8B
    S8 --> S8C --> S8D
    S8 --> S8E
```

---

## 4. Customer Footfall & Virtual Gate Tracking Workflow

```mermaid
flowchart LR
    subgraph DetectionTracking["Detection & Tracking"]
        FRAME["Camera Frame"] --> YOLO["YOLOv8 ONNX<br/>(320x320)"]
        YOLO --> BBOX["Detections<br/>(Person & Bags)"]
        BBOX --> BYTE["ByteTrack Tracker<br/>(Kalman Filter + Hungarian Matching)"]
        BYTE --> TRAJ["Tracked Person Trajectory<br/>(History of centroids)"]
    end

    subgraph GateLogic["Virtual Gate Geometry"]
        TRAJ --> GATE_CHECK{"Crossed Gate Boundary?"}
        GATE_CHECK -- "Entry Line" --> DWELL_IN{"Dwell Validation<br/>Inside > 60s?"}
        GATE_CHECK -- "Exit Line" --> DWELL_OUT{"Dwell Validation<br/>Inside > 10s?"}
        GATE_CHECK -- "Bag Gate" --> BAG_VAL{"Person carrying bag?"}
        
        DWELL_IN -- "Yes" --> EVT_IN["Validated Customer IN<br/>Increment Occupancy"]
        DWELL_OUT -- "Yes" --> EVT_OUT["Validated Customer OUT<br/>Decrement Occupancy"]
        BAG_VAL -- "Yes" --> EVT_BAG["Validated Bag Exit Event"]
    end

    subgraph Demographics["Demographic Profiling"]
        EVT_IN --> MIVOLO["MiVOLO Model<br/>(Face / Upper body crop)"]
        MIVOLO --> DEMO_RES["Age Group & Gender<br/>(e.g., Male, 25-35)"]
    end

    subgraph CloudSync["Cloud Sync"]
        EVT_IN --> SYNC["Zero-Drop Event Queue"]
        EVT_OUT --> SYNC
        EVT_BAG --> SYNC
        DEMO_RES --> SYNC
        SYNC --> HTTP["POST to Cloud Gateway"]
    end
```

---

## 5. Staff 3-Photo Multi-Angle Registration Workflow

```mermaid
flowchart TD
    START["User Opens 3-Photo Enrollment Modal"] --> SELECT["Selects 3 Photos (Frontal, Left Angle, Right Angle)"]
    SELECT --> SUBMIT["Click 'Register Staff'"]
    
    subgraph EnrollmentPipeline["Enrollment Pipeline (enroll_staff_from_photos)"]
        SUBMIT --> EXIF["Auto-orient via EXIF transposition (fixes smartphone photos)"]
        EXIF --> RESIZE["Downscale large photos to 1280 max dimension<br/>(Speeds up YuNet 10x-20x)"]
        RESIZE --> DIR["Create storage directory: Attendence System/Staffs/{ID}_{Name}"]
        DIR --> LOOP["Iterate over each photo:"]
        
        LOOP --> DET["YuNet Face & Landmark Detection (Adaptive threshold 0.45 -> 0.30)"]
        DET --> ALIGN["Align face to 112x112 chip"]
        ALIGN --> EMB["Extract SFace 128-d Embedding vector"]
        EMB --> EVAL["Calculate Frontal Score:<br/>Score = (1 - |yaw - 1| * 0.5) * log(1 + blur)"]
        
        EVAL --> BEST{"Best Frontal Photo?"}
        BEST -- "Yes" --> SET_PRI["Designate as Primary Avatar"]
        BEST -- "No" --> SET_SEC["Save as Secondary Multi-Angle Template"]
        
        SET_PRI --> CENTROID["Compute Normalized Centroid Mean Embedding:<br/>Mean Vector = sum(Embeddings) / N<br/>Centroid = Mean Vector / ||Mean Vector||"]
        SET_SEC --> CENTROID
        
        CENTROID --> SQL["Insert Employee & Embedding rows into SQLite"]
        SQL --> RELOAD["Hot-reload in-memory recognition gallery<br/>(Zero app restart required)"]
    end
    
    RELOAD --> DONE["Live recognition instantly active for new staff! 🎉"]
```
