"""
VisionTrack AI - Object Tracker Module
DeepSORT-based multi-object tracker with motion trail support.
"""

import numpy as np
import cv2
from collections import defaultdict, deque
import logging

logger = logging.getLogger(__name__)

try:
    from deep_sort_realtime.deepsort_tracker import DeepSort
    DEEPSORT_AVAILABLE = True
except ImportError:
    DEEPSORT_AVAILABLE = False
    logger.warning("deep-sort-realtime not installed. Tracking IDs will be sequential fallback.")


class ObjectTracker:
    """
    Wraps DeepSORT tracker.
    Maintains motion trails per track ID and provides crossing-line counting.
    """

    def __init__(self, max_age: int = 30, n_init: int = 3, trail_length: int = 40):
        """
        Args:
            max_age: Frames to keep a lost track alive.
            n_init: Detections needed before a track is confirmed.
            trail_length: Number of past centroids stored per track.
        """
        self.max_age = max_age
        self.n_init = n_init
        self.trail_length = trail_length

        # trail_history[track_id] = deque of (cx, cy)
        self.trail_history: dict[int, deque] = defaultdict(lambda: deque(maxlen=trail_length))
        # last known color per track
        self.track_colors: dict[int, tuple] = {}
        # all track IDs ever seen
        self.all_ids: set = set()

        # Virtual line counting
        self.line_pt1 = None
        self.line_pt2 = None
        self.inbound_count = 0
        self.outbound_count = 0
        self._prev_centroids: dict[int, tuple] = {}

        self._init_tracker()

    def _init_tracker(self):
        """Initialize the DeepSORT instance."""
        if DEEPSORT_AVAILABLE:
            try:
                self.tracker = DeepSort(
                    max_age=self.max_age,
                    n_init=self.n_init,
                    nms_max_overlap=1.0,
                    max_cosine_distance=0.3,
                    nn_budget=None,
                    override_track_class=None,
                    embedder="mobilenet",
                    half=True,
                    bgr=True,
                    embedder_gpu=False,
                    embedder_model_name=None,
                    embedder_wts=None,
                    polygon=False,
                    today=None,
                )
                logger.info("DeepSORT tracker initialized.")
            except Exception as exc:
                logger.warning(f"DeepSORT init error ({exc}); falling back to centroid tracker.")
                self.tracker = None
        else:
            self.tracker = None

        # Fallback counter for when DeepSORT is unavailable
        self._fallback_id_counter = 0

    def update(self, detections: list, frame: np.ndarray) -> list:
        """
        Update tracker with new detections.
        Args:
            detections: From ObjectDetector.detect().
            frame: Current BGR frame (needed by DeepSORT embedder).
        Returns:
            List of track dicts:
              {track_id, bbox, label, confidence, color, centroid}
        """
        tracks = []

        if not detections:
            # Still update tracker with empty list to age out old tracks
            if self.tracker:
                try:
                    self.tracker.update_tracks([], frame=frame)
                except Exception:
                    pass
            return tracks

        if self.tracker and DEEPSORT_AVAILABLE:
            # DeepSORT expects list of ([x, y, w, h], conf, class_label)
            ds_inputs = []
            for det in detections:
                x1, y1, x2, y2 = det["bbox"]
                w, h = x2 - x1, y2 - y1
                ds_inputs.append(([x1, y1, w, h], det["confidence"], det["label"]))

            try:
                raw_tracks = self.tracker.update_tracks(ds_inputs, frame=frame)
            except Exception as exc:
                logger.error(f"DeepSORT update error: {exc}")
                raw_tracks = []

            for t in raw_tracks:
                if not t.is_confirmed():
                    continue
                track_id = int(t.track_id)
                ltrb = t.to_ltrb()
                x1, y1, x2, y2 = map(int, ltrb)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                # Retrieve original detection info
                det_info = self._match_detection(detections, x1, y1, x2, y2)
                color = det_info.get("color", (0, 255, 255))
                label = det_info.get("label", t.get_det_class() or "object")
                conf = det_info.get("confidence", 0.0)

                self._update_trail(track_id, cx, cy, color)
                self._check_line_crossing(track_id, cx, cy)
                self.all_ids.add(track_id)

                tracks.append({
                    "track_id": track_id,
                    "bbox": [x1, y1, x2, y2],
                    "label": label,
                    "confidence": conf,
                    "color": color,
                    "centroid": (cx, cy),
                })
        else:
            # Fallback: assign sequential IDs
            for det in detections:
                self._fallback_id_counter += 1
                tid = self._fallback_id_counter
                x1, y1, x2, y2 = det["bbox"]
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                color = det.get("color", (0, 255, 255))
                self._update_trail(tid, cx, cy, color)
                self.all_ids.add(tid)
                tracks.append({
                    "track_id": tid,
                    "bbox": [x1, y1, x2, y2],
                    "label": det["label"],
                    "confidence": det["confidence"],
                    "color": color,
                    "centroid": (cx, cy),
                })

        return tracks

    def _match_detection(self, detections, x1, y1, x2, y2):
        """Match a tracked bbox to the closest detection by IoU."""
        best, best_iou = {}, 0.0
        for det in detections:
            dx1, dy1, dx2, dy2 = det["bbox"]
            iou = self._iou(x1, y1, x2, y2, dx1, dy1, dx2, dy2)
            if iou > best_iou:
                best_iou = iou
                best = det
        return best

    @staticmethod
    def _iou(ax1, ay1, ax2, ay2, bx1, by1, bx2, by2) -> float:
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        if inter == 0:
            return 0.0
        a_area = (ax2 - ax1) * (ay2 - ay1)
        b_area = (bx2 - bx1) * (by2 - by1)
        return inter / (a_area + b_area - inter + 1e-6)

    def _update_trail(self, track_id: int, cx: int, cy: int, color: tuple):
        """Append centroid to the motion trail for this track."""
        self.trail_history[track_id].append((cx, cy))
        self.track_colors[track_id] = color

    def draw_trails(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw motion trails for all active tracks.
        Trails fade from opaque at the tip to transparent at the tail.
        """
        for track_id, trail in self.trail_history.items():
            points = list(trail)
            color = self.track_colors.get(track_id, (0, 255, 255))
            n = len(points)
            for i in range(1, n):
                alpha = i / n
                faded = tuple(int(c * alpha) for c in color)
                thickness = max(1, int(3 * alpha))
                cv2.line(frame, points[i - 1], points[i], faded, thickness, cv2.LINE_AA)
        return frame

    def set_counting_line(self, pt1: tuple, pt2: tuple):
        """Define a virtual counting line. Objects crossing it are counted."""
        self.line_pt1 = pt1
        self.line_pt2 = pt2
        self.inbound_count = 0
        self.outbound_count = 0
        self._prev_centroids.clear()

    def _check_line_crossing(self, track_id: int, cx: int, cy: int):
        """Detect if track centroid has crossed the counting line."""
        if self.line_pt1 is None or self.line_pt2 is None:
            return
        if track_id not in self._prev_centroids:
            self._prev_centroids[track_id] = (cx, cy)
            return

        prev = self._prev_centroids[track_id]
        side_now = self._side_of_line(cx, cy)
        side_prev = self._side_of_line(*prev)

        if side_now != side_prev:
            if side_now > 0:
                self.inbound_count += 1
            else:
                self.outbound_count += 1

        self._prev_centroids[track_id] = (cx, cy)

    def _side_of_line(self, px: int, py: int) -> int:
        """Return +1 or -1 depending on which side of the counting line (px, py) is."""
        x1, y1 = self.line_pt1
        x2, y2 = self.line_pt2
        val = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
        return 1 if val >= 0 else -1

    def draw_counting_line(self, frame: np.ndarray) -> np.ndarray:
        """Overlay the counting line and counters on the frame."""
        if self.line_pt1 is None or self.line_pt2 is None:
            return frame
        cv2.line(frame, self.line_pt1, self.line_pt2, (0, 255, 255), 2, cv2.LINE_AA)
        mid_x = (self.line_pt1[0] + self.line_pt2[0]) // 2
        mid_y = (self.line_pt1[1] + self.line_pt2[1]) // 2
        cv2.putText(frame, f"IN:{self.inbound_count}  OUT:{self.outbound_count}",
                    (mid_x - 60, mid_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
        return frame

    def reset(self):
        """Clear all tracking state."""
        self.trail_history.clear()
        self.track_colors.clear()
        self.all_ids.clear()
        self._prev_centroids.clear()
        self.inbound_count = 0
        self.outbound_count = 0
        self._fallback_id_counter = 0
        self._init_tracker()
