import unittest
import torch

class MaskIntegrityTests(unittest.TestCase):
    def test_near_tied_scores_flagged(self):
        m=torch.full((250,),0.50) + torch.linspace(-1e-6,1e-6,250)
        self.assertLess(float(m.std(unbiased=False)),1e-4)
    def test_variable_scores_pass_std_gate(self):
        m=torch.linspace(.1,.9,250)
        self.assertGreater(float(m.std(unbiased=False)),1e-4)
