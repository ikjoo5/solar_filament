import importlib.util
from pathlib import Path
import unittest


path = Path(__file__).resolve().parents[1] / "tools" / "checkpoint_utils.py"
spec = importlib.util.spec_from_file_location("checkpoint_utils", path)
loader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(loader)


class CheckpointPolicyTests(unittest.TestCase):
    def test_known_layer_globals_are_allowed(self):
        names = ["ultralytics.nn.modules.head.Segment26", "torch.nn.modules.conv.Conv2d"]
        self.assertEqual(loader.validate_checkpoint_globals(names), sorted(names))

    def test_nonlayer_global_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unreviewed globals"):
            loader.validate_checkpoint_globals(["os.system"])

    def test_allowlist_does_not_expand_from_checkpoint(self):
        with self.assertRaises(ValueError):
            loader.validate_checkpoint_globals(["ultralytics.custom.Payload"])


if __name__ == "__main__":
    unittest.main()
