import unittest
from unittest.mock import Mock

import numpy as np

from core.face_detector import FaceBox
from core.processor import FrameProcessor


class FaceEffectsTest(unittest.TestCase):
    def test_each_effect_changes_only_the_detected_face(self):
        frame = np.random.default_rng(0).integers(0, 256, (120, 120, 3), dtype=np.uint8)
        face = FaceBox(30, 30, 60, 60)

        for effect in FrameProcessor.EFFECTS:
            with self.subTest(effect=effect):
                processor = FrameProcessor()
                processor._detector.detect = Mock(return_value=[face])
                processor.current_effect = effect

                result = processor.process(frame)

                self.assertGreater(np.count_nonzero(result.frame[30:90, 30:90] != frame[30:90, 30:90]), 100)
                np.testing.assert_array_equal(result.frame[:30], frame[:30])


if __name__ == "__main__":
    unittest.main()
