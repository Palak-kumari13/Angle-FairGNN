import unittest
from pathlib import Path

class StageAvailability(unittest.TestCase):
    def test_extended_stages_exist(self):
        s=(Path(__file__).resolve().parents[1]/'train.py').read_text()
        for name in ('A4','A5','lambda_robust','lambda_mask_stability','lambda_distill'):
            self.assertIn(name,s)

    def test_optional_bridge_is_separate(self):
        root=Path(__file__).resolve().parents[1]
        self.assertTrue((root/'optional_frme/frme/graph_bridge.py').exists())
        self.assertTrue((root/'global_fallback.py').exists())
