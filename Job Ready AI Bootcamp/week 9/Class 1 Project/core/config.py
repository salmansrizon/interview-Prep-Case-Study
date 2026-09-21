"""Production configuration for Face Effects Cam."""
import os
from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class AppConfig:
    """Immutable application configuration."""

    # Camera settings
    CAMERA_INDEX: int = int(os.getenv("CAMERA_INDEX", "0"))
    FRAME_WIDTH: int = int(os.getenv("FRAME_WIDTH", "1280"))
    FRAME_HEIGHT: int = int(os.getenv("FRAME_HEIGHT", "720"))
    FPS_TARGET: int = int(os.getenv("FPS_TARGET", "30"))

    # Face detection
    FACE_DETECTION_SCALE: float = 1.1
    FACE_DETECTION_MIN_NEIGHBORS: int = 5
    FACE_DETECTION_MIN_SIZE: Tuple[int, int] = (80, 80)

    # UI
    STREAMLIT_PAGE_TITLE: str = "Face Effects Cam"
    STREAMLIT_LAYOUT: str = "wide"

    # Performance
    MAX_QUEUE_SIZE: int = 2  # Prevent memory buildup in threaded capture


CONFIG = AppConfig()
