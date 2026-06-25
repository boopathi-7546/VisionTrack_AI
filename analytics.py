"""
VisionTrack AI - Analytics Module
Tracks detection statistics, generates heatmaps, and exports reports.
"""

import csv
import json
import os
import time
import numpy as np
import cv2
from collections import Counter, deque
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class Analytics:
    """
    Manages real-time detection statistics, heatmap accumulation,
    zone monitoring, snapshot capture, and report generation.
    """

    def __init__(self, reports_dir: str = "static/reports",
                 snapshots_dir: str = "static/snapshots",
                 heatmap_decay: float = 0.98):
        """
        Args:
            reports_dir: Where CSV and JSON reports are saved.
            snapshots_dir: Where detected frames are saved.
            heatmap_decay: Per-frame decay multiplier for the heatmap (0–1).
        """
        self.reports_dir = reports_dir
        self.snapshots_dir = snapshots_dir
        self.heatmap_decay = heatmap_decay

        os.makedirs(reports_dir, exist_ok=True)
        os.makedirs(snapshots_dir, exist_ok=True)

        # ── Per-session stats ─────────────────────────────────────────────
        self.session_start = time.time()
        self.frame_count = 0
        self.total_detections = 0
        self.label_counts: Counter = Counter()
        self.confidence_sum = 0.0
        self.detection_log: list[dict] = []        # full log for export
        self.fps_history: deque = deque(maxlen=60)  # rolling FPS buffer
        self._last_frame_time = time.time()

        # ── Heatmap ───────────────────────────────────────────────────────
        self.heatmap: np.ndarray | None = None     # initialized on first frame

        # ── Restricted zones ─────────────────────────────────────────────
        # Each zone: {"name": str, "polygon": [(x,y),...], "triggered": bool}
        self.zones: list[dict] = []

    # ── Per-Frame Update ──────────────────────────────────────────────────────

    def update(self, detections: list, tracks: list, frame: np.ndarray) -> dict:
        """
        Process one frame's detections and return live stats dict.
        Args:
            detections: From ObjectDetector.detect().
            tracks: From ObjectTracker.update().
            frame: BGR frame (used for heatmap dimensions).
        Returns:
            stats dict consumed by Flask endpoint and frontend.
        """
        now = time.time()
        elapsed = now - self._last_frame_time
        if elapsed > 0:
            self.fps_history.append(1.0 / elapsed)
        self._last_frame_time = now

        self.frame_count += 1
        self.total_detections += len(detections)

        for det in detections:
            self.label_counts[det["label"]] += 1
            self.confidence_sum += det["confidence"]

        # ── Heatmap accumulation ─────────────────────────────────────────
        if frame is not None:
            h, w = frame.shape[:2]
            if self.heatmap is None:
                self.heatmap = np.zeros((h, w), dtype=np.float32)
            else:
                self.heatmap *= self.heatmap_decay

            for det in detections:
                x1, y1, x2, y2 = det["bbox"]
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                sigma = max(20, (x2 - x1 + y2 - y1) // 6)
                self._add_gaussian(self.heatmap, cy, cx, sigma)

        # ── Zone monitoring ───────────────────────────────────────────────
        zone_alerts = []
        for zone in self.zones:
            pts = np.array(zone["polygon"], dtype=np.int32)
            triggered = False
            for det in detections:
                x1, y1, x2, y2 = det["bbox"]
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                if cv2.pointPolygonTest(pts, (cx, cy), False) >= 0:
                    triggered = True
                    break
            if triggered and not zone.get("triggered"):
                zone_alerts.append(zone["name"])
            zone["triggered"] = triggered

        # ── Detection log ─────────────────────────────────────────────────
        if detections:
            ts = datetime.now().isoformat(timespec="milliseconds")
            for det in detections:
                self.detection_log.append({
                    "timestamp": ts,
                    "frame": self.frame_count,
                    "label": det["label"],
                    "confidence": det["confidence"],
                    "bbox": det["bbox"],
                })

        return self._build_stats(tracks, zone_alerts)

    def _build_stats(self, tracks: list, zone_alerts: list) -> dict:
        """Assemble the stats dict sent to frontend."""
        fps = round(np.mean(self.fps_history), 1) if self.fps_history else 0.0
        avg_conf = (self.confidence_sum / max(self.total_detections, 1))
        most_frequent = self.label_counts.most_common(1)

        return {
            "fps": fps,
            "frame_count": self.frame_count,
            "total_detections": self.total_detections,
            "active_tracks": len(tracks),
            "most_frequent": most_frequent[0][0] if most_frequent else "—",
            "most_frequent_count": most_frequent[0][1] if most_frequent else 0,
            "avg_confidence": round(avg_conf * 100, 1),
            "label_counts": dict(self.label_counts.most_common(10)),
            "zone_alerts": zone_alerts,
            "session_seconds": round(time.time() - self.session_start),
            "track_ids": [t["track_id"] for t in tracks],
            "track_labels": [t["label"] for t in tracks],
        }

    # ── Heatmap Overlay ───────────────────────────────────────────────────────

    def apply_heatmap(self, frame: np.ndarray) -> np.ndarray:
        """Blend the accumulated heatmap onto the frame."""
        if self.heatmap is None:
            return frame
        h, w = frame.shape[:2]
        hm = cv2.resize(self.heatmap, (w, h))
        # Normalize and colorize
        norm = cv2.normalize(hm, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        colored = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
        blended = cv2.addWeighted(frame, 0.6, colored, 0.4, 0)
        return blended

    # ── Zone Monitoring ───────────────────────────────────────────────────────

    def add_zone(self, name: str, polygon: list[tuple]):
        """
        Register a restricted zone.
        Args:
            name: Human-readable zone name.
            polygon: List of (x, y) vertex tuples.
        """
        self.zones.append({"name": name, "polygon": polygon, "triggered": False})

    def clear_zones(self):
        """Remove all registered zones."""
        self.zones.clear()

    def draw_zones(self, frame: np.ndarray) -> np.ndarray:
        """Overlay zone polygons on the frame with alert coloring."""
        for zone in self.zones:
            pts = np.array(zone["polygon"], dtype=np.int32)
            color = (0, 0, 255) if zone.get("triggered") else (0, 255, 150)
            overlay = frame.copy()
            cv2.fillPoly(overlay, [pts], color)
            frame = cv2.addWeighted(overlay, 0.2, frame, 0.8, 0)
            cv2.polylines(frame, [pts], True, color, 2, cv2.LINE_AA)
            cv2.putText(frame, zone["name"],
                        (pts[0][0], pts[0][1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
        return frame

    # ── Snapshot ──────────────────────────────────────────────────────────────

    def save_snapshot(self, frame: np.ndarray, label: str = "detection") -> str | None:
        """Save an annotated frame as a JPEG snapshot."""
        try:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"{label}_{ts}.jpg"
            path = os.path.join(self.snapshots_dir, filename)
            cv2.imwrite(path, frame)
            logger.info(f"Snapshot saved: {path}")
            return filename
        except Exception as exc:
            logger.error(f"Snapshot save failed: {exc}")
            return None

    # ── Reports ───────────────────────────────────────────────────────────────

    def export_csv(self) -> str | None:
        """Export detection log to CSV. Returns filename."""
        try:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"detection_report_{ts}.csv"
            path = os.path.join(self.reports_dir, filename)
            with open(path, "w", newline="") as f:
                writer = csv.DictWriter(
                    f, fieldnames=["timestamp", "frame", "label", "confidence", "bbox"])
                writer.writeheader()
                writer.writerows(self.detection_log)
            logger.info(f"CSV report saved: {path}")
            return filename
        except Exception as exc:
            logger.error(f"CSV export failed: {exc}")
            return None

    def export_json(self) -> str | None:
        """Export full session stats as JSON. Returns filename."""
        try:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"session_report_{ts}.json"
            path = os.path.join(self.reports_dir, filename)
            report = {
                "session_start": datetime.fromtimestamp(self.session_start).isoformat(),
                "total_frames": self.frame_count,
                "total_detections": self.total_detections,
                "label_counts": dict(self.label_counts),
                "avg_confidence_pct": round(
                    self.confidence_sum / max(self.total_detections, 1) * 100, 2),
                "detection_log": self.detection_log,
            }
            with open(path, "w") as f:
                json.dump(report, f, indent=2)
            logger.info(f"JSON report saved: {path}")
            return filename
        except Exception as exc:
            logger.error(f"JSON export failed: {exc}")
            return None

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _add_gaussian(heatmap: np.ndarray, cy: int, cx: int, sigma: int):
        """Add a 2D Gaussian blob centered at (cx, cy) to the heatmap."""
        h, w = heatmap.shape
        y_min = max(0, cy - 3 * sigma)
        y_max = min(h, cy + 3 * sigma)
        x_min = max(0, cx - 3 * sigma)
        x_max = min(w, cx + 3 * sigma)
        if y_min >= y_max or x_min >= x_max:
            return
        yy, xx = np.mgrid[y_min:y_max, x_min:x_max]
        g = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma ** 2))
        heatmap[y_min:y_max, x_min:x_max] += g.astype(np.float32)

    def reset(self):
        """Clear all session data."""
        self.session_start = time.time()
        self.frame_count = 0
        self.total_detections = 0
        self.label_counts.clear()
        self.confidence_sum = 0.0
        self.detection_log.clear()
        self.fps_history.clear()
        self.heatmap = None
        for z in self.zones:
            z["triggered"] = False
