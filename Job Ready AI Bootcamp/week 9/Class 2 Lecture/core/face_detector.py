"""Face detection with Haar Cascade classifier."""
import cv2
import numpy as np
from typing import List, Tuple
from dataclasses import dataclass
import os


@dataclass
class FaceBox:
    """Detected face bounding box."""
    x: int
    y: int
    width: int
    height: int
    confidence: float = 0.0

    @property
    def center(self) -> Tuple[int, int]:
        return (self.x + self.width // 2, self.y + self.height // 2)

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def tl(self) -> Tuple[int, int]:
        return (self.x, self.y)

    @property
    def br(self) -> Tuple[int, int]:
        return (self.x + self.width, self.y + self.height)


class FaceDetector:
    """Production face detector with Haar Cascade classifier."""

    def __init__(
        self,
        scale_factor: float = 1.1,
        min_neighbors: int = 5,
        min_size: Tuple[int, int] = (80, 80),
    ):
        self._scale_factor = scale_factor
        self._min_neighbors = min_neighbors
        self._min_size = min_size

        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        if not os.path.exists(cascade_path):
            raise RuntimeError(f"Haar cascade not found: {cascade_path}")

        self._cascade = cv2.CascadeClassifier(cascade_path)

    def detect(self, frame: np.ndarray) -> List[FaceBox]:
        """Detect faces in frame. Returns list of FaceBox objects."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        detections = self._cascade.detectMultiScale(
            gray,
            scaleFactor=self._scale_factor,
            minNeighbors=self._min_neighbors,
            minSize=self._min_size,
            flags=cv2.CASCADE_SCALE_IMAGE,
        )

        faces = []
        for (x, y, w, h) in detections:
            faces.append(FaceBox(
                x=int(x),
                y=int(y),
                width=int(w),
                height=int(h),
                confidence=1.0,
            ))

        return faces
