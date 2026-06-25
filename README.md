# VisionTrack AI — Smart Object Detection & Tracking System

> Real-time multi-object detection and tracking powered by **YOLOv8** + **DeepSORT**, wrapped in a cyberpunk glassmorphism web dashboard built with **Flask**.

---

## Features

| Category | Details |
|---|---|
| **Detection** | YOLOv8n (80 COCO classes), confidence filtering, neon bounding boxes |
| **Tracking** | DeepSORT — unique IDs, motion trails, multi-object simultaneous |
| **Analytics** | FPS counter, total detections, label frequency, avg confidence |
| **Heatmap** | Gaussian accumulation overlay of crowd density |
| **Zone Monitoring** | Polygon-based restricted zones with live alerts |
| **Counting Line** | Inbound/Outbound object counter via virtual line |
| **Snapshots** | Save annotated frames to `static/snapshots/` |
| **Reports** | Export detection logs as CSV or JSON |
| **Input Sources** | Webcam, video file upload, drag-and-drop |
| **UI** | Glassmorphism dark dashboard, particle background, animated cards |

---

## Project Structure

```
VisionTrack_AI/
├── app.py              ← Flask app & all API routes
├── detector.py         ← YOLOv8 detection module
├── tracker.py          ← DeepSORT tracking module
├── analytics.py        ← Stats, heatmap, zones, reports
├── requirements.txt
├── README.md
│
├── models/             ← YOLOv8 weights (auto-downloaded)
│
├── static/
│   ├── css/style.css   ← Cyberpunk glassmorphism styles
│   ├── js/script.js    ← Frontend controller
│   ├── uploads/        ← Uploaded video files
│   ├── reports/        ← CSV & JSON exports
│   └── snapshots/      ← Captured detection frames
│
├── templates/
│   └── index.html      ← Single-page dashboard
│
└── logs/
    └── visiontrack.log
```

---

## Installation

### Prerequisites

- Python 3.9–3.11
- pip
- A webcam (optional) or video files
- 2 GB free disk space (for model weights)

### Steps

```bash
# 1. Clone or extract the project
cd VisionTrack_AI

# 2. Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Linux/macOS
# venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the application
python app.py
```

Open your browser at: **http://localhost:5000**

> The first run will automatically download `yolov8n.pt` (~6 MB) from Ultralytics if it is not present.

---

## Usage

### Start a Stream

**Webcam** — Click "Start Webcam" on the hero page or sidebar dashboard.

**Video File** — Click "Upload Video", drag & drop a video onto the stream panel,
or use the upload button in the control bar. Supported: `.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`.

### Controls

| Control | Description |
|---|---|
| Trails toggle | Show/hide motion trail paths per track |
| Heatmap toggle | Blend Gaussian heatmap onto stream |
| Confidence slider | Adjust detection threshold (10–90%) |
| Reset | Clear analytics counters (keeps stream running) |
| Stop | Release camera/video and stop processing |

### Counting Line

In the **Zones** section or by clicking the `⟶` button under the stream,
set a Y-pixel position to draw a horizontal counting line.
Objects crossing it are counted as **IN** or **OUT**.

### Restricted Zones

In the **Zones** section, enter a zone name and a JSON polygon:
```json
[[100,100],[500,100],[500,400],[100,400]]
```
An alert fires (and a banner appears) whenever a tracked object enters the zone.

### Snapshots

Click **📷** to capture and save the current annotated frame.
All captures appear in the **Snapshots** gallery tab.

### Reports

In the **Reports** tab:
- **Export CSV** — full per-detection log with timestamps and bounding boxes
- **Export JSON** — session summary with label frequency and accuracy stats

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/video_feed` | MJPEG stream |
| `POST` | `/api/start_webcam` | Open webcam |
| `POST` | `/api/upload_video` | Upload & play video |
| `POST` | `/api/stop` | Stop stream |
| `GET` | `/api/stats` | Live stats JSON |
| `POST` | `/api/set_confidence` | `{confidence: 0.0–1.0}` |
| `POST` | `/api/toggle_trails` | Toggle motion trails |
| `POST` | `/api/toggle_heatmap` | Toggle heatmap overlay |
| `POST` | `/api/set_counting_line` | `{pt1:[x,y], pt2:[x,y]}` |
| `POST` | `/api/add_zone` | `{name, polygon:[[x,y],...]}` |
| `POST` | `/api/clear_zones` | Remove all zones |
| `POST` | `/api/snapshot` | Capture frame |
| `GET`  | `/api/list_snapshots` | List saved snapshots |
| `POST` | `/api/export_csv` | Export CSV report |
| `POST` | `/api/export_json` | Export JSON report |
| `POST` | `/api/reset` | Reset analytics |

---

## Configuration

Edit defaults directly in `app.py`:

```python
detector = ObjectDetector(confidence=0.45)          # detection threshold
tracker  = ObjectTracker(max_age=30, n_init=3,      # DeepSORT params
                         trail_length=50)
```

For higher-accuracy detection at the cost of speed, change `yolov8n.pt` to
`yolov8s.pt`, `yolov8m.pt`, or `yolov8l.pt`.

---

## Troubleshooting

**Webcam not found** — Check that no other app is using the camera.
On Linux, verify the device is at `/dev/video0`.

**Low FPS** — Reduce video resolution in `app.py` (`CAP_PROP_FRAME_WIDTH` / `HEIGHT`),
or switch to a smaller YOLO model (`yolov8n`).

**DeepSORT import error** — Run `pip install deep-sort-realtime`. If GPU is unavailable,
the tracker falls back to a sequential ID assignment.

**CORS errors** — `flask-cors` is included. If deploying behind a reverse proxy,
ensure the proxy forwards `Origin` headers.

---

## Tech Stack

- **Python 3.10**
- **Flask 3.0** — web server & REST API
- **Ultralytics YOLOv8** — object detection
- **deep-sort-realtime** — multi-object tracking
- **OpenCV** — video capture & frame rendering
- **NumPy** — heatmap math
- **Vanilla JS** — no frontend framework needed

---

## License

MIT — free to use, modify, and distribute.

---

*Built by Boopathi · VisionTrack AI · 2025*
