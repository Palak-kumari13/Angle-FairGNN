import unittest
from global_fallback import choose_checkpoint, GateConfig

class FallbackTest(unittest.TestCase):
    def row(self, name, auc=.78, dp=.05, eo=.05, drift=.02, sd=.02):
        return dict(model=name,val_auc=auc,val_parity=dp,val_equality=eo,
                    val_prediction_drift=drift,val_mask_std=sd)
    def test_picks_fair_stable_candidate(self):
        d=choose_checkpoint([self.row('A0',auc=.75),self.row('A3',auc=.8)])
        self.assertEqual(d['selected_model'],'A3')
        self.assertEqual(d['explanation'],'abstain')
    def test_rejects_unfair_better_accuracy(self):
        d=choose_checkpoint([self.row('A0'),self.row('A3',auc=.95,dp=.5)])
        self.assertEqual(d['selected_model'],'A0')
    def test_no_certified_checkpoint(self):
        d=choose_checkpoint([self.row('A0',dp=.9),self.row('A3',dp=.8)])
        self.assertEqual(d['decision'],'no_certified_candidate')
        self.assertFalse(d['reference_passed'])
    def test_missing_stability_rejects(self):
        d=choose_checkpoint([self.row('A0'),self.row('A2',auc=.9,drift=float('nan'))])
        self.assertEqual(d['selected_model'],'A0')
    def test_no_a0(self):
        self.assertIsNone(choose_checkpoint([self.row('A3')])['selected_model'])
if __name__=='__main__': unittest.main()
