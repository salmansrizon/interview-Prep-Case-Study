"""Frame processing pipeline: detection + emoji overlay."""
import cv2
import numpy as np
import time
from typing import Optional, List
from dataclasses import dataclass

from .camera import CameraStats
from .face_detector import FaceDetector, FaceBox
from .emoji_manager import EmojiManager
from .config import CONFIG


@dataclass
class ProcessResult:
    """Result of frame processing."""
    frame: np.ndarray
    faces_detected: int
    processing_time_ms: float
    fps: float


class FrameProcessor:
    """Orchestrates face detection and emoji overlay."""

    def __init__(self):
        self._detector = FaceDetector(
            scale_factor=CONFIG.FACE_DETECTION_SCALE,
            min_neighbors=CONFIG.FACE_DETECTION_MIN_NEIGHBORS,
            min_size=CONFIG.FACE_DETECTION_MIN_SIZE,
        )
        self._emoji_manager = EmojiManager()
        self._current_emoji = "😀"
        self._show_debug = False
        self._frame_times: List[float] = []

    @property
    def current_emoji(self) -> str:
        return self._current_emoji

    @current_emoji.setter
    def current_emoji(self, emoji: str) -> None:
        self._current_emoji = emoji

    @property
    def show_debug(self) -> bool:
        return self._show_debug

    @show_debug.setter
    def show_debug(self, value: bool) -> None:
        self._show_debug = value

    def get_available_emojis(self) -> List[str]:
        """Return list of available emoji characters."""
        return self._emoji_manager.get_emoji_list()

    def process(self, frame: np.ndarray) -> ProcessResult:
        """Process a single frame: detect faces and overlay emoji."""
        start_time = time.perf_counter()

        # Detect faces
        faces = self._detector.detect(frame)

        # Overlay emoji on each face
        output = frame.copy()
        for face in faces:
            output = self._emoji_manager.overlay_emoji(
                output,
                self._current_emoji,
                face.x,
                face.y,
                face.width,
                face.height,
                scale_factor=CONFIG.EMOJI_SCALE_FACTOR,
                vertical_offset=CONFIG.EMOJI_VERTICAL_OFFSET,
            )

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
