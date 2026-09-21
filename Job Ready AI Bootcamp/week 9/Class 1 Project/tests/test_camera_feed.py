import ast
import unittest
from pathlib import Path


class CameraFeedTest(unittest.TestCase):
    def test_feed_updates_one_frame_without_looping_or_rerunning_app(self):
        tree = ast.parse((Path(__file__).parents[1] / "app.py").read_text())
        feed = next(
            (node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "render_camera_feed"),
            None,
        )

        self.assertIsNotNone(feed)
        self.assertFalse(any(isinstance(node, (ast.For, ast.While)) for node in ast.walk(feed)))
        self.assertFalse(
            any(
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "rerun"
                for node in ast.walk(feed)
            )
        )


if __name__ == "__main__":
    unittest.main()
