"""Frame processing pipeline: detection + privacy protection."""
import cv2
import numpy as np
import time
from typing import List
from dataclasses import dataclass

from .camera import CameraStats
from .face_detector import FaceDetector
from .privacy_engine import PrivacyEngine, PrivacySettings
from .config import CONFIG


@dataclass
class ProcessResult:
    """Result of frame processing."""
    frame: np.ndarray
    faces_detected: int
    processing_time_ms: float
    fps: float


class FrameProcessor:
    """Orchestrates face detection and privacy protection."""

    def __init__(self):
        self._detector = FaceDetector(
            scale_factor=CONFIG.FACE_DETECTION_SCALE,
            min_neighbors=CONFIG.FACE_DETECTION_MIN_NEIGHBORS,
            min_size=CONFIG.FACE_DETECTION_MIN_SIZE,
        )
        self._privacy_engine = PrivacyEngine()
        self._frame_times: List[float] = []

    @property
    def privacy_settings(self) -> PrivacySettings:
        return self._privacy_engine.settings

    @privacy_settings.setter
    def privacy_settings(self, settings: PrivacySettings) -> None:
        self._privacy_engine.update_settings(settings)

    def process(self, frame: np.ndarray) -> ProcessResult:
        """Process a single frame: detect faces and apply privacy protection."""
        start_time = time.perf_counter()

        faces = self._detector.detect(frame)
        output = self._privacy_engine.apply(frame, faces)

        elapsed = time.perf_counter() - start_time
        self._frame_times.append(elapsed)
        if len(self._frame_times) > 30:
            self._frame_times.pop(0)

        avg_time = sum(self._frame_times) / len(self._frame_times) if self._frame_times else 0.001
        fps = 1.0 / avg_time if avg_time > 0 else 0

        if self._privacy_engine.settings.show_stats:
            self._draw_hud(output, len(faces), elapsed * 1000, fps)

        return ProcessResult(
            frame=output,
            faces_detected=len(faces),
            processing_time_ms=elapsed * 1000,
            fps=fps,
        )

    def _draw_hud(self, frame: np.ndarray, face_count: int, proc_ms: float, fps: float) -> None:
        """Draw performance HUD on frame."""
        h, w = frame.shape[:2]

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 40), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        mode_name = self._privacy_engine.settings.mode.value.upper()
        hud = f"Privacy: {mode_name} | Faces: {face_count} | {proc_ms:.1f}ms | {fps:.1f} FPS"
        cv2.putText(frame, hud, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

        color = (0, 255, 0) if face_count > 0 else (128, 128, 128)
        cv2.circle(frame, (w - 20, 20), 8, color, -1)
