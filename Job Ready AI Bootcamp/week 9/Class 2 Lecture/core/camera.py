"""Threaded camera capture with frame queue management."""
import cv2
import threading
import queue
import time
from typing import Optional
import numpy as np
from dataclasses import dataclass


@dataclass
class CameraStats:
    """Real-time camera performance statistics."""
    fps: float = 0.0
    frame_count: int = 0
    dropped_frames: int = 0


class ThreadedCamera:
    """Thread-safe camera capture with automatic frame dropping."""

    def __init__(
        self,
        camera_index: int = 0,
        width: int = 1280,
        height: int = 720,
        max_queue_size: int = 2,
    ):
        self._camera_index = camera_index
        self._width = width
        self._height = height
        self._max_queue_size = max_queue_size

        self._cap: Optional[cv2.VideoCapture] = None
        self._queue: queue.Queue = queue.Queue(maxsize=max_queue_size)
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        self._stats = CameraStats()
        self._last_frame_time = time.time()
        self._fps_history: list[float] = []

        self._lock = threading.Lock()

    def start(self) -> bool:
        """Initialize and start camera capture thread."""
        self._cap = cv2.VideoCapture(self._camera_index)

        if not self._cap.isOpened():
            return False

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        for _ in range(3):
            self._cap.read()

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

        return True

    def _capture_loop(self) -> None:
        """Background thread: continuously read frames from camera."""
        while not self._stop_event.is_set():
            ret, frame = self._cap.read()

            if not ret:
                continue

            current_time = time.time()
            delta = current_time - self._last_frame_time
            self._last_frame_time = current_time

            if delta > 0:
                instant_fps = 1.0 / delta
                self._fps_history.append(instant_fps)
                if len(self._fps_history) > 30:
                    self._fps_history.pop(0)

            with self._lock:
                self._stats.frame_count += 1
                if len(self._fps_history) > 0:
                    self._stats.fps = sum(self._fps_history) / len(self._fps_history)

            if self._queue.full():
                try:
                    self._queue.get_nowait()
                    with self._lock:
                        self._stats.dropped_frames += 1
                except queue.Empty:
                    pass

            try:
                self._queue.put_nowait(frame)
            except queue.Full:
                with self._lock:
                    self._stats.dropped_frames += 1

    def read(self) -> Optional[np.ndarray]:
        """Get latest frame (non-blocking)."""
        try:
            return self._queue.get_nowait()
        except queue.Empty:
            return None

    def get_stats(self) -> CameraStats:
        """Get current camera statistics."""
        with self._lock:
            return CameraStats(
                fps=self._stats.fps,
                frame_count=self._stats.frame_count,
                dropped_frames=self._stats.dropped_frames,
            )

    def is_running(self) -> bool:
        """Check if camera thread is active."""
        return self._thread is not None and self._thread.is_alive()

    def stop(self) -> None:
        """Gracefully stop camera capture."""
        self._stop_event.set()

        if self._thread:
            self._thread.join(timeout=1.0)

        if self._cap:
            self._cap.release()
            self._cap = None

        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
