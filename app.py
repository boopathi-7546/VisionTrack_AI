"""
VisionTrack AI - Flask Application Entry Point
Routes: live stream, video upload, analytics API, reports, snapshots.
"""

import os
import cv2
import time
import threading
import logging
import base64
import numpy as np
from datetime import datetime
from flask import (Flask, render_template, Response, request,
                   jsonify, send_from_directory)
from flask_cors import CORS
from werkzeug.utils import secure_filename

from detector import ObjectDetector
from tracker import ObjectTracker
from analytics import Analytics

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/visiontrack.log"),
    ]
)
logger = logging.getLogger(__name__)

# ── Flask App ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024   # 500 MB upload limit
app.config["UPLOAD_FOLDER"] = "static/uploads"
ALLOWED_EXTENSIONS = {"mp4", "avi", "mov", "mkv", "webm", "m4v"}

os.makedirs("logs", exist_ok=True)
os.makedirs("static/uploads", exist_ok=True)

# ── AI Components (initialized once) ─────────────────────────────────────────
detector = ObjectDetector(confidence=0.45)
tracker = ObjectTracker(max_age=30, n_init=3, trail_length=50)
analytics = Analytics()

# ── Shared stream state ───────────────────────────────────────────────────────
_lock = threading.Lock()
_state = {
    "cap": None,            # cv2.VideoCapture
    "source": "none",       # "webcam" | "video" | "none"
    "running": False,
    "show_trails": True,
    "show_heatmap": False,
    "show_zones": True,
    "latest_stats": {},
    "latest_tracks": [],
    "last_snapshot": None,
}


def _allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def _generate_frames():
    """
    Generator that reads frames from the active VideoCapture,
    runs detection + tracking + analytics, and yields MJPEG bytes.
    """
    frame_skip = 0

    while True:
        with _lock:
            cap = _state.get("cap")
            running = _state.get("running")

        if not running or cap is None:
            # Yield a blank placeholder frame
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(blank, "No Active Stream", (160, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100, 100, 100), 2)
            _, buf = cv2.imencode(".jpg", blank)
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" +
                   buf.tobytes() + b"\r\n")
            time.sleep(0.1)
            continue

        ret, frame = cap.read()
        if not ret:
            # Video ended — loop or stop
            with _lock:
                if _state["source"] == "video":
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                else:
                    _state["running"] = False
            continue

        # ── Detection & Tracking ──────────────────────────────────────────
        try:
            detections = detector.detect(frame)
            tracks = tracker.update(detections, frame)
            stats = analytics.update(detections, tracks, frame)

            with _lock:
                _state["latest_stats"] = stats
                _state["latest_tracks"] = tracks

        except Exception as exc:
            logger.error(f"Pipeline error: {exc}")
            detections, tracks, stats = [], [], {}

        # ── Overlay rendering ─────────────────────────────────────────────
        vis = frame.copy()

        if _state.get("show_heatmap"):
            vis = analytics.apply_heatmap(vis)

        if _state.get("show_zones"):
            vis = analytics.draw_zones(vis)

        if _state.get("show_trails"):
            vis = tracker.draw_trails(vis)

        vis = tracker.draw_counting_line(vis)

        # Draw bounding boxes with tracking IDs
        id_map = {i: t["track_id"] for i, t in enumerate(tracks)}
        vis = detector.draw_detections(vis, detections, id_map)

        # ── FPS overlay ───────────────────────────────────────────────────
        fps = stats.get("fps", 0.0)
        cv2.putText(vis, f"FPS {fps:.1f}", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 200), 2, cv2.LINE_AA)
        cv2.putText(vis, f"Objects: {len(detections)}", (10, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2, cv2.LINE_AA)

        # ── MJPEG encode ─────────────────────────────────────────────────
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, 85]
        _, buf = cv2.imencode(".jpg", vis, encode_params)
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" +
               buf.tobytes() + b"\r\n")


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    """MJPEG stream endpoint consumed by <img> tag in frontend."""
    return Response(
        _generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/api/start_webcam", methods=["POST"])
def start_webcam():
    """Open the default webcam (device 0)."""
    with _lock:
        if _state["cap"] is not None:
            _state["cap"].release()
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return jsonify({"success": False, "error": "Webcam not found."}), 400
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        _state["cap"] = cap
        _state["source"] = "webcam"
        _state["running"] = True

    tracker.reset()
    analytics.reset()
    logger.info("Webcam stream started.")
    return jsonify({"success": True, "source": "webcam"})


@app.route("/api/upload_video", methods=["POST"])
def upload_video():
    """Accept a video file upload and start detection on it."""
    if "video" not in request.files:
        return jsonify({"success": False, "error": "No file part."}), 400
    file = request.files["video"]
    if file.filename == "":
        return jsonify({"success": False, "error": "No file selected."}), 400
    if not _allowed_file(file.filename):
        return jsonify({"success": False,
                        "error": f"Unsupported format. Allowed: {ALLOWED_EXTENSIONS}"}), 400

    filename = secure_filename(file.filename)
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(save_path)
    logger.info(f"Video uploaded: {save_path}")

    with _lock:
        if _state["cap"] is not None:
            _state["cap"].release()
        cap = cv2.VideoCapture(save_path)
        if not cap.isOpened():
            return jsonify({"success": False, "error": "Could not open video file."}), 400
        _state["cap"] = cap
        _state["source"] = "video"
        _state["running"] = True

    tracker.reset()
    analytics.reset()
    return jsonify({"success": True, "source": "video", "filename": filename})


@app.route("/api/stop", methods=["POST"])
def stop_stream():
    """Stop the active stream and release resources."""
    with _lock:
        _state["running"] = False
        if _state["cap"] is not None:
            _state["cap"].release()
            _state["cap"] = None
        _state["source"] = "none"
    logger.info("Stream stopped.")
    return jsonify({"success": True})


@app.route("/api/stats")
def get_stats():
    """Live statistics JSON polled by the frontend every second."""
    with _lock:
        stats = dict(_state.get("latest_stats", {}))
        tracks = list(_state.get("latest_tracks", []))
        source = _state.get("source", "none")
        running = _state.get("running", False)

    stats["source"] = source
    stats["running"] = running

    # Build a summary track table (limit 20 rows for UI)
    stats["tracks_table"] = [
        {
            "id": t["track_id"],
            "label": t["label"],
            "confidence": f"{t['confidence']:.0%}",
        }
        for t in tracks[:20]
    ]
    return jsonify(stats)


@app.route("/api/set_confidence", methods=["POST"])
def set_confidence():
    data = request.get_json(force=True)
    conf = float(data.get("confidence", 0.45))
    detector.set_confidence(conf)
    return jsonify({"success": True, "confidence": conf})


@app.route("/api/toggle_trails", methods=["POST"])
def toggle_trails():
    with _lock:
        _state["show_trails"] = not _state["show_trails"]
        val = _state["show_trails"]
    return jsonify({"success": True, "show_trails": val})


@app.route("/api/toggle_heatmap", methods=["POST"])
def toggle_heatmap():
    with _lock:
        _state["show_heatmap"] = not _state["show_heatmap"]
        val = _state["show_heatmap"]
    return jsonify({"success": True, "show_heatmap": val})


@app.route("/api/set_counting_line", methods=["POST"])
def set_counting_line():
    """Set a virtual counting line. Body: {pt1:[x,y], pt2:[x,y]}"""
    data = request.get_json(force=True)
    pt1 = tuple(data.get("pt1", [0, 360]))
    pt2 = tuple(data.get("pt2", [1280, 360]))
    tracker.set_counting_line(pt1, pt2)
    return jsonify({"success": True, "pt1": list(pt1), "pt2": list(pt2)})


@app.route("/api/add_zone", methods=["POST"])
def add_zone():
    """Add a restricted zone. Body: {name:str, polygon:[[x,y],...]}"""
    data = request.get_json(force=True)
    name = data.get("name", "Zone")
    polygon = [tuple(p) for p in data.get("polygon", [])]
    if len(polygon) < 3:
        return jsonify({"success": False, "error": "Polygon needs ≥3 points."}), 400
    analytics.add_zone(name, polygon)
    return jsonify({"success": True, "zone": name})


@app.route("/api/clear_zones", methods=["POST"])
def clear_zones():
    analytics.clear_zones()
    return jsonify({"success": True})


@app.route("/api/snapshot", methods=["POST"])
def take_snapshot():
    """Capture the current frame and save as a snapshot."""
    with _lock:
        cap = _state.get("cap")
        running = _state.get("running")
    if not running or cap is None:
        return jsonify({"success": False, "error": "No active stream."}), 400
    ret, frame = cap.read()
    if not ret:
        return jsonify({"success": False, "error": "Could not read frame."}), 400

    detections = detector.detect(frame)
    vis = detector.draw_detections(frame.copy(), detections)
    filename = analytics.save_snapshot(vis)
    if filename:
        return jsonify({"success": True, "filename": filename,
                        "url": f"/static/snapshots/{filename}"})
    return jsonify({"success": False, "error": "Snapshot failed."}), 500


@app.route("/api/export_csv", methods=["POST"])
def export_csv():
    filename = analytics.export_csv()
    if filename:
        return jsonify({"success": True, "filename": filename,
                        "url": f"/static/reports/{filename}"})
    return jsonify({"success": False, "error": "Export failed."}), 500


@app.route("/api/export_json", methods=["POST"])
def export_json():
    filename = analytics.export_json()
    if filename:
        return jsonify({"success": True, "filename": filename,
                        "url": f"/static/reports/{filename}"})
    return jsonify({"success": False, "error": "Export failed."}), 500


@app.route("/api/reset", methods=["POST"])
def reset_session():
    """Reset analytics and tracker (keep stream running)."""
    tracker.reset()
    analytics.reset()
    return jsonify({"success": True})


@app.route("/static/snapshots/<filename>")
def serve_snapshot(filename):
    return send_from_directory("static/snapshots", filename)


@app.route("/static/reports/<filename>")
def serve_report(filename):
    return send_from_directory("static/reports", filename)


@app.route("/api/list_snapshots")
def list_snapshots():
    files = sorted(os.listdir("static/snapshots"), reverse=True)[:20]
    return jsonify({"snapshots": [f"/static/snapshots/{f}" for f in files if f.endswith(".jpg")]})


# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logger.info("VisionTrack AI starting…")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
