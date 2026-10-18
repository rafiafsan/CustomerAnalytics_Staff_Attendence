# Pipeline & REST API Specifications
====================================

This document defines the REST API integration contracts, network fallback architecture, and local edge SQLite database schemas for the **OmniRetail AI Suite**.

---

## 1. Cloud REST API Integration

Edge devices communicate with the central enterprise backend exclusively via standard HTTP POST calls. Direct edge database access (e.g. to Oracle HRIS or PostgreSQL) is eliminated, keeping the corporate network secure and simplifying packaging.

### Base URLs & Fallback Hierarchy
* **Primary Endpoint:** `https://ai.prangroup.com/api/v1/attendance` (Reverse-proxied HTTPS, port 443)
* **Direct Fallback Endpoint:** `http://ai.prangroup.com:8021/api/v1/attendance` (Direct gateway port 8021)

> [!NOTE]
> `AttendanceApiClient` implements an automatic failover list. If the primary HTTPS URL times out or returns HTTP 404, it immediately attempts delivery to the direct port 8021 endpoint before reporting an error.

---

### Endpoint A: Staff Attendance Punch Dispatch

* **Method:** `POST`
* **Path:** `/api/v1/attendance`
* **Authentication:** Header `X-API-Key: <OUTLET_SECRET_TOKEN>`
* **Content-Type:** `application/json`

#### Request Headers
| Header | Value | Description |
| :--- | :--- | :--- |
| `Content-Type` | `application/json` | Standard JSON payload |
| `X-API-Key` | `secret_outlet_token_banasree` | Edge outlet authorization token |

#### Request Payload Schema
```json
{
  "staff_id": "100234",
  "employee_name": "Rahim Ahmed",
  "cdate": "20261018",
  "ctime": "091530",
  "camera_id": "CAM_01_MAIN",
  "terminal_id": "111111",
  "location_id": 3,
  "confidence": 0.8845,
  "status": "CONFIRMED",
  "punch_type": "AUTO"
}
```

#### Field Specifications
| Field | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `staff_id` | `string` | **Yes** | Employee company identification code (e.g., `"100234"`). |
| `employee_name` | `string` | No | Full name of the staff member (e.g., `"Rahim Ahmed"`). |
| `cdate` | `string` | **Yes** | 8-character numeric date formatted as `YYYYMMDD` (e.g., `"20261018"`). |
| `ctime` | `string` | **Yes** | 6-character numeric time formatted as `HHMMSS` (e.g., `"091530"`). |
| `camera_id` | `string` | **Yes** | Edge camera identifier (e.g., `"CAM_01_MAIN"`). |
| `terminal_id` | `string` | **Yes** | Physical attendance terminal device identifier (e.g., `"111111"`). |
| `location_id` | `integer` | **Yes** | Outlet/Shop branch ID in the central enterprise system (e.g., `3`). |
| `confidence` | `float` | No | Cosine similarity match score rounded to 4 decimals (e.g., `0.8845`). |
| `status` | `string` | **Yes** | Verification status: `"CONFIRMED"` or `"MANUAL"`. |
| `punch_type` | `string` | **Yes** | Capture mode: `"AUTO"` (live face match) or `"CARD"`. |

#### Response Codes
* **`201 Created`**: Punch successfully stored and forwarded to enterprise payroll/HRIS.
* **`400 Bad Request`**: Malformed payload or invalid date/time format.
* **`401 Unauthorized`**: Missing or invalid `X-API-Key`.
* **`409 Conflict`**: Duplicate punch detected within the lockout window (handled gracefully by edge).

---

### Endpoint B: Customer Footfall & Gate Crossing Dispatch

* **Method:** `POST`
* **Path:** `/api/v1/events/footfall`
* **Authentication:** Header `X-API-Key: <OUTLET_SECRET_TOKEN>`

#### Request Payload Schema
```json
{
  "shop_id": 3,
  "camera_id": "CAM_RETAIL_ENTRANCE",
  "event_type": "IN",
  "timestamp": "2026-10-18T09:15:30.123456",
  "dwell_time_seconds": 65.4,
  "bag_detected": false,
  "demographics": {
    "age_group": "25-34",
    "gender": "Male",
    "confidence": 0.82
  }
}
```

---

## 2. Local Edge SQLite Database Schemas

The edge PC maintains local SQLite databases to guarantee offline resilience and comprehensive auditability.

### Database 1: `attendance.db` (`Attendence System/data/attendance.db`)

#### Table: `employees`
Stores enrolled staff identity profiles and master photo references.
```sql
CREATE TABLE IF NOT EXISTS employees (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    department TEXT DEFAULT 'General',
    photo_path TEXT,
    embedding_path TEXT,
    email TEXT,
    enrolled_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

#### Table: `employee_embeddings`
Stores multi-angle template embeddings (frontal, angled, and centroid representations) for each staff member.
```sql
CREATE TABLE IF NOT EXISTS employee_embeddings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    emp_id TEXT NOT NULL,
    embedding_path TEXT NOT NULL,
    photo_path TEXT,
    is_primary INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (emp_id) REFERENCES employees(id) ON DELETE CASCADE
);
```

#### Table: `attendance_logs`
Historical archive of all verified attendance events generated at the edge.
```sql
CREATE TABLE IF NOT EXISTS attendance_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    emp_id TEXT NOT NULL,
    name TEXT NOT NULL,
    camera_id TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    confidence REAL,
    status TEXT DEFAULT 'CONFIRMED',
    face_image_path TEXT,
    embedding_path TEXT,
    synced_to_cloud INTEGER DEFAULT 0
);
```

#### Table: `face_captures`
Diagnostic audit trail recording every detected face (both passed and rejected quality checks).
```sql
CREATE TABLE IF NOT EXISTS face_captures (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    camera_id TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    bbox_w INTEGER,
    bbox_h INTEGER,
    blur_score REAL,
    roll_angle REAL,
    yaw_ratio REAL,
    passed_quality BOOLEAN,
    rejection_reason TEXT,
    match_name TEXT,
    match_score REAL,
    face_image_path TEXT,
    embedding_path TEXT
);
```

---

### Database 2: `edge_events.db` (`FootFall_AgeGender/edge_events.db`)

#### Table: `events`
Stores all validated footfall crossings, bag exits, and demographic profiling records.
```sql
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    track_id INTEGER NOT NULL,
    event_type TEXT NOT NULL,          -- 'IN', 'OUT', 'BAG_EXIT'
    timestamp TEXT NOT NULL,
    dwell_time REAL,
    age_group TEXT,
    gender TEXT,
    synced INTEGER DEFAULT 0
);
```

#### Table: `occupancy_snapshots`
Periodic snapshots of current outlet occupancy.
```sql
CREATE TABLE IF NOT EXISTS occupancy_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    current_inside INTEGER NOT NULL,
    total_in INTEGER NOT NULL,
    total_out INTEGER NOT NULL,
    synced INTEGER DEFAULT 0
);
```
