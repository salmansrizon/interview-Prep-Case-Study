"""Emoji rendering and management."""
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import Dict, Optional
import io
import base64


class EmojiManager:
    """Handles emoji rendering, caching, and overlay operations."""

    EMOJI_MAP: Dict[str, str] = {
        "😀": "grinning",
        "😂": "joy",
        "😍": "heart_eyes",
        "🤔": "thinking",
        "😎": "sunglasses",
        "🥳": "partying",
        "😴": "sleeping",
        "🤯": "exploding_head",
        "🤡": "clown",
        "👽": "alien",
        "🤖": "robot",
        "👻": "ghost",
        "💩": "poop",
        "🦄": "unicorn",
        "🐶": "dog",
        "🐱": "cat",
        "🦊": "fox",
        "🐼": "panda",
        "🐨": "koala",
        "🦁": "lion",
        "🐯": "tiger",
        "🐷": "pig",
        "🐸": "frog",
        "🐙": "octopus",
        "🦋": "butterfly",
        "🌵": "cactus",
        "🍄": "mushroom",
        "🎃": "jack_o_lantern",
        "🎅": "santa",
        "🧙": "mage",
        "🧛": "vampire",
        "🧟": "zombie",
        "🦸": "superhero",
        "🧚": "fairy",
        "🧜": "mermaid",
        "🧞": "genie",
        "🧝": "elf",
    }

    def __init__(self):
        self._cache: Dict[str, np.ndarray] = {}
        self._font: Optional[ImageFont.FreeTypeFont] = None
        self._init_font()

    def _init_font(self) -> None:
        """Initialize emoji font with fallback strategy."""
        font_paths = [
            "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf",
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
            "/System/Library/Fonts/Apple Color Emoji.ttc",
            "C:\Windows\Fonts\seguiemj.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]

        for path in font_paths:
            if os.path.exists(path):
                try:
                    self._font = ImageFont.truetype(path, 109)
                    return
                except Exception:
                    continue

        # Fallback to default
        self._font = ImageFont.load_default()

    def get_emoji_list(self) -> list:
        """Return list of available emojis."""
        return list(self.EMOJI_MAP.keys())

    def render_emoji(self, emoji_char: str, size: int = 256) -> np.ndarray:
        """Render emoji to RGBA numpy array with caching."""
        cache_key = f"{emoji_char}_{size}"

        if cache_key in self._cache:
            return self._cache[cache_key].copy()

        # Create transparent canvas
        img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        # Calculate centered position
        bbox = draw.textbbox((0, 0), emoji_char, font=self._font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        x = (size - text_width) // 2 - bbox[0]
        y = (size - text_height) // 2 - bbox[1]

        draw.text((x, y), emoji_char, font=self._font, embedded_color=True)

        # Convert to numpy
        arr = np.array(img)
        self._cache[cache_key] = arr.copy()

        return arr

    def overlay_emoji(
        self,
        frame: np.ndarray,
        emoji_char: str,
        face_x: int,
        face_y: int,
        face_w: int,
        face_h: int,
        scale_factor: float = 1.4,
        vertical_offset: float = 0.1,
    ) -> np.ndarray:
        """Overlay emoji onto frame at face position with alpha blending."""

        # Calculate emoji size
        emoji_w = int(face_w * scale_factor)
        emoji_h = int(face_h * scale_factor)

        # Calculate position (centered on face, slightly up)
        cx = face_x + face_w // 2
        cy = face_y + face_h // 2 - int(face_h * vertical_offset)

        x1 = cx - emoji_w // 2
        y1 = cy - emoji_h // 2
        x2 = x1 + emoji_w
        y2 = y1 + emoji_h

        # Render emoji at target size
        emoji_rgba = self.render_emoji(emoji_char, max(emoji_w, emoji_h))
        emoji_resized = cv2.resize(emoji_rgba, (emoji_w, emoji_h), interpolation=cv2.INTER_AREA)

        # Clip to frame bounds
        h, w = frame.shape[:2]

        fx1, fx2 = max(0, x1), min(w, x2)
        fy1, fy2 = max(0, y1), min(h, y2)

        ex1 = max(0, -x1)
        ey1 = max(0, -y1)
        ex2 = ex1 + (fx2 - fx1)
        ey2 = ey1 + (fy2 - fy1)

        if fx1 >= fx2 or fy1 >= fy2:
            return frame

        # Alpha blend
        roi = frame[fy1:fy2, fx1:fx2]
        emoji_crop = emoji_resized[ey1:ey2, ex1:ex2]

        alpha = emoji_crop[:, :, 3:4].astype(np.float32) / 255.0
        emoji_rgb = emoji_crop[:, :, :3].astype(np.float32)
        roi_float = roi.astype(np.float32)

        blended = (emoji_rgb * alpha + roi_float * (1 - alpha)).astype(np.uint8)
        frame[fy1:fy2, fx1:fx2] = blended

        return frame

    def get_emoji_base64(self, emoji_char: str, size: int = 64) -> str:
        """Get emoji as base64 PNG for UI display."""
        img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        bbox = draw.textbbox((0, 0), emoji_char, font=self._font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        x = (size - text_w) // 2 - bbox[0]
        y = (size - text_h) // 2 - bbox[1]

        draw.text((x, y), emoji_char, font=self._font, embedded_color=True)

        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode()
