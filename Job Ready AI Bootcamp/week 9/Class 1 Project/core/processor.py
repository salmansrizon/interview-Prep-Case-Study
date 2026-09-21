"""Frame processing pipeline: detection + face effects."""
import cv2
import numpy as np
import time
from typing import List
from dataclasses import dataclass

from .face_detector import FaceDetector, FaceBox
from .config import CONFIG


@dataclass
class ProcessResult:
    """Result of frame processing."""
    frame: np.ndarray
    faces_detected: int
    processing_time_ms: float
    fps: float


class FrameProcessor:
    """Orchestrates face detection and native OpenCV effects."""

    EFFECTS = ("Blur", "Pixelate", "Grayscale", "Negative")

    def __init__(self):
        self._detector = FaceDetector(
            scale_factor=CONFIG.FACE_DETECTION_SCALE,
            min_neighbors=CONFIG.FACE_DETECTION_MIN_NEIGHBORS,
            min_size=CONFIG.FACE_DETECTION_MIN_SIZE,
        )
        self._current_effect = self.EFFECTS[0]
        self._show_debug = False
        self._frame_times: List[float] = []

    @property
    def current_effect(self) -> str:
        return self._current_effect

    @current_effect.setter
    def current_effect(self, effect: str) -> None:
        self._current_effect = effect

    @property
    def show_debug(self) -> bool:
        return self._show_debug

    @show_debug.setter
    def show_debug(self, value: bool) -> None:
        self._show_debug = value

    def _apply_face_effect(self, frame: np.ndarray, face: FaceBox) -> None:
        x1, y1 = max(0, face.x), max(0, face.y)
        x2 = min(frame.shape[1], face.x + face.width)
        y2 = min(frame.shape[0], face.y + face.height)
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return

        if self._current_effect == "Blur":
            kernel = max(3, (min(roi.shape[:2]) // 4) | 1)
            effect = cv2.GaussianBlur(roi, (kernel, kernel), 0)
        elif self._current_effect == "Pixelate":
            small = cv2.resize(
                roi,
                (max(1, roi.shape[1] // 12), max(1, roi.shape[0] // 12)),
                interpolation=cv2.INTER_LINEAR,
            )
            effect = cv2.resize(small, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_NEAREST)
        elif self._current_effect == "Grayscale":
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            effect = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        else:
            effect = cv2.bitwise_not(roi)

        frame[y1:y2, x1:x2] = effect

    def process(self, frame: np.ndarray) -> ProcessResult:
        """Detect faces and apply the selected effect."""
        start_time = time.perf_counter()

        # Detect faces
        faces = self._detector.detect(frame)

        output = frame.copy()
        for face in faces:
            self._apply_face_effect(output, face)

            if self._show_debug:
                # Draw debug box
                cv2.rectangle(
                    output,
                    (face.x, face.y),
                    (face.x + face.width, face.y + face.height),
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    output,
                    f"Face: {face.width}x{face.height}",
                    (face.x, face.y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                )

        # Calculate processing FPS
        elapsed = time.perf_counter() - start_time
        self._frame_times.append(elapsed)
        if len(self._frame_times) > 30:
            self._frame_times.pop(0)

        avg_time = sum(self._frame_times) / len(self._frame_times) if self._frame_times else 0.001
        fps = 1.0 / avg_time if avg_time > 0 else 0

        # Add HUD overlay
        if self._show_debug:
            hud_text = f"Faces: {len(faces)} | Proc: {elapsed*1000:.1f}ms | FPS: {fps:.1f}"
            cv2.putText(
                output,
                hud_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )

        return ProcessResult(
            frame=output,
            faces_detected=len(faces),
            processing_time_ms=elapsed * 1000,
            fps=fps,
        )
