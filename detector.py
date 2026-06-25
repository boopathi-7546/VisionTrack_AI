"""
VisionTrack AI - Object Detector Module
Uses YOLOv8 for real-time object detection with confidence filtering.
"""

import cv2
import numpy as np
from ultralytics import YOLO
import os
import logging

logger = logging.getLogger(__name__)

# ── COCO class labels (80 classes) ──────────────────────────────────────────
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
    "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
    "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table",
    "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock",
    "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

# Neon color palette for bounding boxes (BGR format for OpenCV)
NEON_COLORS = [
    (0, 255, 255),    # cyan
    (255, 0, 255),    # magenta
    (0, 255, 0),      # neon green
    (255, 165, 0),    # neon orange
    (255, 0, 128),    # neon pink
    (128, 0, 255),    # neon purple
    (0, 200, 255),    # neon blue
    (255, 255, 0),    # yellow
    (0, 128, 255),    # neon sky
    (255, 64, 64),    # neon red
]


class ObjectDetector:
    """
    YOLOv8-based object detector with confidence threshold filtering
    and bounding box rendering.
    """

    def __init__(self, model_path: str = "yolov8n.pt", confidence: float = 0.45):
        """
        Initialize the detector.
        Args:
            model_path: Path to YOLOv8 model weights.
            confidence: Minimum detection confidence (0–1).
        """
        self.confidence = confidence
        self.model = None
        self._load_model(model_path)

    def _load_model(self, model_path: str):
        """Load YOLOv8 model; download automatically if not found."""
        try:
            # models/ folder first, then CWD, then let ultralytics download
            search_paths = [
                os.path.join("models", model_path),
                model_path,
            ]
            resolved = next((p for p in search_paths if os.path.exists(p)), model_path)
            logger.info(f"Loading YOLOv8 model from: {resolved}")
            self.model = YOLO(resolved)
            logger.info("YOLOv8 model loaded successfully.")
        except Exception as exc:
            logger.error(f"Failed to load model: {exc}")
            raise

    def detect(self, frame: np.ndarray):
        """
        Run detection on a single BGR frame.
        Returns:
            List of dicts: {bbox, label, confidence, class_id, color}
        """
        if self.model is None:
            return []

        try:
            results = self.model(frame, conf=self.confidence, verbose=False)
            detections = []

            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    label = COCO_CLASSES[cls_id] if cls_id < len(COCO_CLASSES) else "unknown"
                    color = NEON_COLORS[cls_id % len(NEON_COLORS)]

                    detections.append({
                        "bbox": [x1, y1, x2, y2],
                        "label": label,
                        "confidence": round(conf, 3),
                        "class_id": cls_id,
                        "color": color,
                    })

            return detections

        except Exception as exc:
            logger.error(f"Detection error: {exc}")
            return []

    def draw_detections(self, frame: np.ndarray, detections: list,
                        tracked_ids: dict = None) -> np.ndarray:
        """
        Draw animated bounding boxes with labels and confidence scores.
        Args:
            frame: BGR image.
            detections: From self.detect().
            tracked_ids: {track_id: detection_index} for ID overlay.
        Returns:
            Annotated frame.
        """
        overlay = frame.copy()
        h, w = frame.shape[:2]

        for i, det in enumerate(detections):
            x1, y1, x2, y2 = det["bbox"]
            color = det["color"]
            label = det["label"]
            conf = det["confidence"]

            # ── Filled translucent box ───────────────────────────────────
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
            alpha = 0.08
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

            # ── Solid border ─────────────────────────────────────────────
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # ── Corner accent lines (futuristic look) ─────────────────────
            corner_len = max(10, min(20, (x2 - x1) // 5))
            thickness = 3
            for (cx, cy, dx, dy) in [
                (x1, y1, 1, 1), (x2, y1, -1, 1),
                (x1, y2, 1, -1), (x2, y2, -1, -1)
            ]:
                cv2.line(frame, (cx, cy), (cx + dx * corner_len, cy), color, thickness)
                cv2.line(frame, (cx, cy), (cx, cy + dy * corner_len), color, thickness)

            # ── Label background ─────────────────────────────────────────
            track_id = ""
            if tracked_ids and i in tracked_ids:
                track_id = f" #{tracked_ids[i]}"
            text = f"{label}{track_id} {conf:.0%}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            label_y = max(y1 - 4, th + 4)
            cv2.rectangle(frame, (x1, label_y - th - 6), (x1 + tw + 8, label_y + 2), color, -1)
            cv2.putText(frame, text, (x1 + 4, label_y - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (10, 10, 10), 2, cv2.LINE_AA)

        return frame

    def set_confidence(self, conf: float):
        """Update confidence threshold at runtime."""
        self.confidence = max(0.1, min(0.99, conf))
        logger.info(f"Confidence threshold set to {self.confidence}")
