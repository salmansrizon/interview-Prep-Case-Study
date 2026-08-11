"""
Preprocessing for user-supplied images (drawn on a canvas or uploaded).

Converts an arbitrary RGB/RGBA/grayscale image into the exact 28x28x1,
[0, 1]-normalized tensor format the CNN models were trained on.

HIGHLIGHTS: a model is only as good as the match between its training
distribution and what it sees at inference time. MNIST was scanned at
28x28 grayscale, white digit strokes on a black background, pixels
normalized to [0, 1]. A user's drawing or phone photo starts out as
*none* of those things (arbitrary resolution, RGBA color, often dark
strokes on a light background) — this module's whole job is bridging that
gap so the model isn't asked to generalize to a distribution it has never
seen, which is a very different (and much harder) problem than the
overfitting/regularization story the rest of this project is about.
"""

from __future__ import annotations

from typing import Union

import numpy as np
from PIL import Image

from config import DataConfig, get_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

ImageLike = Union[Image.Image, np.ndarray]


def _to_pil_grayscale(image: ImageLike) -> Image.Image:
    """Coerce a PIL image or numpy array into single-channel PIL grayscale.

    Streamlit's drawable canvas returns RGBA numpy arrays (an alpha/
    transparency channel included), so we drop the alpha channel (keep
    only RGB) before letting PIL's ``.convert("L")`` do the actual
    grayscale luminance conversion — passing RGBA straight into
    ``convert("L")`` would silently ignore alpha anyway, but slicing it
    off explicitly here makes that assumption visible in the code.
    """
    if isinstance(image, np.ndarray):
        arr = image
        if arr.ndim == 3 and arr.shape[-1] == 4:
            arr = arr[..., :3]
        image = Image.fromarray(arr.astype("uint8"))
    return image.convert("L")


def preprocess_canvas_image(image: ImageLike, cfg: DataConfig | None = None) -> np.ndarray:
    """Convert a drawn/uploaded image into a model-ready input tensor.

    Pipeline:
        1. Convert to single-channel grayscale.
        2. Resize to the configured square image size (default 28x28).
        3. Normalize pixel values to [0, 1] float32.
        4. Reshape to a (1, H, W, 1) batch tensor for ``model.predict``.

    Args:
        image: A PIL Image or numpy array (e.g. from a Streamlit drawable
            canvas, which typically returns RGBA arrays with a dark
            background and light strokes, or the reverse for uploads).
        cfg: Optional data configuration. Defaults to the global config.

    Returns:
        A ``(1, image_size, image_size, 1)`` float32 array in [0, 1].
    """
    cfg = cfg or get_config().data
    gray = _to_pil_grayscale(image)
    # LANCZOS is a high-quality resampling filter — it matters here because
    # we're almost always *downscaling* a much larger canvas/photo to a
    # tiny 28x28, and a poor resampling filter (e.g. nearest-neighbor)
    # would produce jagged, aliased strokes that look nothing like the
    # smooth anti-aliased strokes MNIST digits actually have.
    gray = gray.resize((cfg.image_size, cfg.image_size), Image.LANCZOS)

    arr = np.asarray(gray, dtype="float32")

    # MNIST digits are white-on-black (stroke pixels near 255, background
    # near 0). A Streamlit canvas drawn with a white pen on a black
    # background already matches that. But an *uploaded* image — say, a
    # phone photo of pencil-on-paper — is typically the opposite: a dark
    # stroke on a light background. Without correcting for this, the model
    # would see an almost-inverted version of the pixel distribution it
    # was trained on and predict close to garbage. We approximate "which
    # case is this" with a simple heuristic: if the image is mostly bright
    # (mean pixel > 127, i.e. a light background dominates), assume it's
    # dark-on-light and invert it back to white-on-black.
    if arr.mean() > 127.0:
        arr = 255.0 - arr

    arr = arr / 255.0
    tensor = arr.reshape((1, cfg.image_size, cfg.image_size, cfg.channels))
    logger.debug("Preprocessed input tensor shape=%s", tensor.shape)
    return tensor


def preprocess_batch(images: list[ImageLike], cfg: DataConfig | None = None) -> np.ndarray:
    """Preprocess a list of images into a single stacked batch tensor.

    Args:
        images: List of PIL images or numpy arrays.
        cfg: Optional data configuration.

    Returns:
        A ``(N, image_size, image_size, 1)`` float32 array in [0, 1].
    """
    tensors = [preprocess_canvas_image(img, cfg) for img in images]
    return np.concatenate(tensors, axis=0)
