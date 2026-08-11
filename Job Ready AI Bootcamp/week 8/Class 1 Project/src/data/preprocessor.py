"""
Inference-time preprocessing.

Turns an arbitrary user-supplied image (a drawable-canvas RGBA array or an
uploaded photo) into the exact ``(1, 28, 28, 1)`` float32 tensor in
``[0, 1]`` that :func:`src.models.cnn.build_baseline_cnn` expects,
mirroring the normalization applied to the training data in
:mod:`src.data.loader`.

HIGHLIGHTS — why is this its own module instead of inline code in
`app.py`? The model was trained exclusively on MNIST's specific convention: a
*bright/white digit stroke on a dark/black background*, scaled to
`[0, 1]`. Any real-world input (a mouse-drawn canvas, a phone photo of
pen-on-paper) will not automatically match that convention — and if it
doesn't, the model will confidently produce garbage predictions because
its input distribution looks nothing like what it was trained on. This
module is the single place responsible for bridging that gap, so the app
tab code never has to reason about pixel conventions itself.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageOps

from config import Config, get_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DigitPreprocessor:
    """Converts a raw canvas/upload image into a model-ready tensor.

    This class has exactly two public entry points, `from_array` and
    `from_upload`, because the two input sources genuinely differ in what
    "digit = bright pixels" assumption holds true for them (see each
    method's docstring). Routing both through the same private `_finish`
    step guarantees that resize/normalize/reshape logic — the parts that
    *must* match training-time preprocessing exactly — never drifts
    between the two code paths.
    """

    def __init__(self, config: Config | None = None) -> None:
        self.config = config or get_config()

    def from_array(self, image_array: np.ndarray) -> np.ndarray:
        """Preprocess a numpy image array (e.g. from ``st_canvas``).

        Args:
            image_array: An ``(H, W)``, ``(H, W, 3)``, or ``(H, W, 4)``
                array with values in ``[0, 255]``. Canvases typically draw
                white/colored strokes on a transparent or black background
                (digit = bright, background = dark) — the opposite of
                MNIST, which is white digit on black background stored as
                high pixel values. We normalize both conventions below.

        Returns:
            A ``(1, 28, 28, 1)`` float32 array in ``[0, 1]``.

        HIGHLIGHTS: why no inversion logic here (unlike `from_upload`)?
        `app.py` configures `st_canvas` with a *white* stroke on a
        *black* background — deliberately matching MNIST's convention at
        the source, rather than drawing it the "natural" way (black pen on
        white paper) and inverting afterward. So the canvas path is
        already correct by construction; `already_digit_bright=True` below
        tells `_finish` to skip the brightness check entirely.
        """
        if image_array.ndim == 3:
            image = Image.fromarray(image_array.astype("uint8"), mode="RGBA" if image_array.shape[-1] == 4 else "RGB")
            image = image.convert("L")
        elif image_array.ndim == 2:
            image = Image.fromarray(image_array.astype("uint8"), mode="L")
        else:
            raise ValueError(f"Unsupported image array shape: {image_array.shape}")

        return self._finish(image, already_digit_bright=True)

    def from_upload(self, file_obj) -> np.ndarray:
        """Preprocess an uploaded image file (e.g. from ``st.file_uploader``).

        Uploaded photos are usually dark digit on a light/white background
        (like a photo of pen-on-paper), which is the inverse of MNIST's
        white-digit-on-black convention, so we auto-invert.

        Args:
            file_obj: A file-like object opened by PIL (path or buffer).

        Returns:
            A ``(1, 28, 28, 1)`` float32 array in ``[0, 1]``.

        HIGHLIGHTS: why does the upload path need different handling than
        the canvas path? We cannot control how a user photographed or scanned their
        digit the way we control the canvas widget's colors. A photo of
        pencil-on-paper is dark ink on a light page — visually the exact
        opposite of MNIST's white-stroke-on-black-background convention.
        Feeding that straight into the model without correcting for it
        would mean the model sees an almost fully-inverted version of what
        it learned to recognize, and predictions would be close to random.
        `_finish` below inspects the *average brightness* of the image to
        decide whether an invert is needed, rather than assuming every
        upload is dark-on-light — a scanned white-on-black chalkboard
        photo, for instance, should be left alone.
        """
        image = Image.open(file_obj).convert("L")
        return self._finish(image, already_digit_bright=False)

    def _finish(self, image: Image.Image, already_digit_bright: bool) -> np.ndarray:
        """Resize, orient, normalize, and reshape a grayscale PIL image.

        The four steps below happen in this specific order for a reason:
        resize *before* measuring brightness/normalizing, so the mean-
        brightness heuristic and the final pixel values are computed on
        the same 28x28 grid the model will actually see (a large photo's
        average brightness can look different once background noise is
        cropped away by resizing).
        """
        height, width = self.config.data.image_size
        # LANCZOS is a high-quality downsampling filter — important here
        # because canvases/photos are typically much larger than 28x28
        # (e.g. a 280x280 canvas), and a cheap resize filter can produce
        # aliasing artifacts that make thin digit strokes disappear
        # entirely by the time we're down to MNIST's resolution.
        image = image.resize((width, height), Image.LANCZOS)

        pixels = np.asarray(image).astype("float32")

        if not already_digit_bright:
            # Heuristic: if more than half the image is bright (mean > 127),
            # we assume it's a light background with dark ink strokes —
            # i.e. the opposite of MNIST — and flip it. This is a simple
            # global check rather than per-pixel edge detection because it
            # only needs to get the *overall* polarity right; the CNN
            # itself is robust to the exact stroke width/anti-aliasing
            # differences between a real photo and MNIST's rendering.
            if pixels.mean() > 127.0:
                pixels = 255.0 - pixels

        # Same [0, 255] -> [0, 1] rescaling used on the training data in
        # src/data/loader.py. If this ever drifted out of sync with that
        # module, the model would receive inputs on a different numeric
        # scale than it was trained on and predictions would silently
        # degrade — this is the single most important line to keep
        # consistent between training and inference.
        normalized = pixels / self.config.data.pixel_max_value
        tensor = normalized.reshape(1, height, width, self.config.data.num_channels)

        logger.info(
            "Preprocessed image -> shape=%s, min=%.3f, max=%.3f",
            tensor.shape, tensor.min(), tensor.max(),
        )
        return tensor
