"""Privacy protection engine with multiple anonymization modes."""
import cv2
import numpy as np
from typing import List, Optional
from dataclasses import dataclass
from enum import Enum

from .face_detector import FaceBox
from .config import CONFIG


class PrivacyMode(Enum):
    """Available privacy protection modes."""
    BLUR = "blur"
    PIXELATE = "pixelate"
    SOLID_MASK = "solid_mask"
    BLACK_BAR = "black_bar"
    EMOJI_MASK = "emoji_mask"


@dataclass
class PrivacySettings:
    """User-configurable privacy settings."""
    mode: PrivacyMode = PrivacyMode.BLUR
    blur_strength: int = 51
    pixelate_size: int = 15
    mask_color: Tuple[int, int, int] = (0, 0, 0)
    mask_opacity: float = 0.85
    show_detection_boxes: bool = False
    show_stats: bool = True
    protect_eyes_only: bool = False
    feather_edges: bool = True


class PrivacyEngine:
    """Applies privacy protection to detected faces in video frames."""

    def __init__(self, settings: Optional[PrivacySettings] = None):
        self.settings = settings or PrivacySettings()
        self._face_cache: dict = {}

    def apply(self, frame: np.ndarray, faces: List[FaceBox]) -> np.ndarray:
        """Apply selected privacy mode to all detected faces."""
        output = frame.copy()

        for face in faces:
            if self.settings.protect_eyes_only:
                # Only blur upper 40% of face (eye region)
                eye_region = FaceBox(
                    x=face.x,
                    y=face.y,
                    width=face.width,
                    height=int(face.height * 0.45),
                )
                output = self._apply_mode(output, eye_region)
            else:
                output = self._apply_mode(output, face)

            if self.settings.show_detection_boxes:
                output = self._draw_detection_box(output, face)

        return output

    def _apply_mode(self, frame: np.ndarray, face: FaceBox) -> np.ndarray:
        """Apply the selected privacy mode to a single face region."""
        mode = self.settings.mode

        if mode == PrivacyMode.BLUR:
            return self._apply_blur(frame, face)
        elif mode == PrivacyMode.PIXELATE:
            return self._apply_pixelate(frame, face)
        elif mode == PrivacyMode.SOLID_MASK:
            return self._apply_solid_mask(frame, face)
        elif mode == PrivacyMode.BLACK_BAR:
            return self._apply_black_bar(frame, face)
        elif mode == PrivacyMode.EMOJI_MASK:
            return self._apply_emoji_mask(frame, face)

        return frame

    def _apply_blur(self, frame: np.ndarray, face: FaceBox) -> np.ndarray:
        """Apply Gaussian blur to face region with feathered edges."""
        x1, y1 = max(0, face.x), max(0, face.y)
        x2, y2 = min(frame.shape[1], face.x + face.width), min(frame.shape[0], face.y + face.height)

        if x2 <= x1 or y2 <= y1:
            return frame

        roi = frame[y1:y2, x1:x2].copy()

        # Ensure odd kernel size
        k = max(3, self.settings.blur_strength | 1)

        # Apply strong Gaussian blur
        blurred = cv2.GaussianBlur(roi, (k, k), self.settings.blur_strength / 2)

        if self.settings.feather_edges and face.width > 60 and face.height > 60:
            # Create feathered mask for smooth edges
            mask = np.zeros((y2 - y1, x2 - x1), dtype=np.float32)
            feather = min(20, min(face.width, face.height) // 6)

            cv2.rectangle(
                mask,
                (feather, feather),
                (x2 - x1 - feather, y2 - y1 - feather),
                1.0,
                -1,
            )
            mask = cv2.GaussianBlur(mask, (feather * 2 + 1, feather * 2 + 1), feather)

            mask_3ch = np.stack([mask] * 3, axis=-1)
            blended = (blurred * mask_3ch + roi * (1 - mask_3ch)).astype(np.uint8)
            frame[y1:y2, x1:x2] = blended
        else:
            frame[y1:y2, x1:x2] = blurred

        return frame

    def _apply_pixelate(self, frame: np.ndarray, face: FaceBox) -> np.ndarray:
        """Apply pixelation effect to face region."""
        x1, y1 = max(0, face.x), max(0, face.y)
        x2, y2 = min(frame.shape[1], face.x + face.width), min(frame.shape[0], face.y + face.height)

        if x2 <= x1 or y2 <= y1:
            return frame

        roi = frame[y1:y2, x1:x2]
        h, w = roi.shape[:2]

        block_size = max(4, self.settings.pixelate_size)

        # Downsample
        small = cv2.resize(roi, (w // block_size, h // block_size), interpolation=cv2.INTER_LINEAR)
        # Upsample with nearest neighbor for pixelated look
        pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

        frame[y1:y2, x1:x2] = pixelated
        return frame

    def _apply_solid_mask(self, frame: np.ndarray, face: FaceBox) -> np.ndarray:
        """Apply semi-transparent solid color mask."""
        x1, y1 = max(0, face.x), max(0, face.y)
        x2, y2 = min(frame.shape[1], face.x + face.width), min(frame.shape[0], face.y + face.height)

        if x2 <= x1 or y2 <= y1:
            return frame

        overlay = frame.copy()
        color = self.settings.mask_color
        opacity = self.settings.mask_opacity

        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)

        if self.settings.feather_edges:
            # Feather the edges
            mask = np.zeros(frame.shape[:2], dtype=np.float32)
            feather = min(15, min(face.width, face.height) // 8)
            cv2.rectangle(mask, (x1 + feather, y1 + feather), (x2 - feather, y2 - feather), 1.0, -1)
            mask = cv2.GaussianBlur(mask, (feather * 2 + 1, feather * 2 + 1), feather)

            for c in range(3):
                frame[:, :, c] = (
                    overlay[:, :, c] * mask * opacity +
                    frame[:, :, c] * (1 - mask * opacity)
                ).astype(np.uint8)
        else:
            cv2.addWeighted(overlay, opacity, frame, 1 - opacity, 0, frame)

        return frame

    def _apply_black_bar(self, frame: np.ndarray, face: FaceBox) -> np.ndarray:
        """Apply black bar censor (classic TV style)."""
        x1, y1 = max(0, face.x), max(0, face.y)
        x2, y2 = min(frame.shape[1], face.x + face.width), min(frame.shape[0], face.y + face.height)

        if x2 <= x1 or y2 <= y1:
            return frame

        # Draw black bar with thin border
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), -1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 80, 80), 1)

        # Add "CENSORED" text if face is large enough
        if face.width > 100:
            text = "CENSORED"
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = face.width / 400
            thickness = max(1, int(scale * 2))
            (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
            tx = x1 + (face.width - tw) // 2
            ty = y1 + (face.height + th) // 2
            cv2.putText(frame, text, (tx, ty), font, scale, (180, 180, 180), thickness)

        return frame

    def _apply_emoji_mask(self, frame: np.ndarray, face: FaceBox) -> np.ndarray:
        """Apply large emoji over face as mask."""
        x1, y1 = max(0, face.x), max(0, face.y)
        x2, y2 = min(frame.shape[1], face.x + face.width), min(frame.shape[0], face.y + face.height)

        if x2 <= x1 or y2 <= y1:
            return frame

        # Blur underneath first
        roi = frame[y1:y2, x1:x2].copy()
        k = max(21, min(face.width, face.height) // 4 | 1)
        blurred = cv2.GaussianBlur(roi, (k, k), k / 2)
        frame[y1:y2, x1:x2] = blurred

        # Draw emoji text
        emoji = "🛡️"
        cx = x1 + face.width // 2
        cy = y1 + face.height // 2

        # Use PIL for emoji rendering
        from PIL import Image, ImageDraw, ImageFont

        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)

        font_size = int(min(face.width, face.height) * 0.8)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf", font_size)
        except:
            try:
                font = ImageFont.truetype("/System/Library/Fonts/Apple Color Emoji.ttc", font_size)
            except:
                font = ImageFont.load_default()

        bbox = draw.textbbox((0, 0), emoji, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        tx = cx - tw // 2
        ty = cy - th // 2

        draw.text((tx, ty), emoji, font=font, embedded_color=True)
        frame[:] = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        return frame

    def _draw_detection_box(self, frame: np.ndarray, face: FaceBox) -> np.ndarray:
        """Draw debug detection box around face."""
        cv2.rectangle(
            frame,
            face.tl,
            face.br,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            frame,
            f"Face {face.width}x{face.height}",
            (face.x, face.y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )
        return frame

    def update_settings(self, settings: PrivacySettings) -> None:
        """Update privacy settings."""
        self.settings = settings
