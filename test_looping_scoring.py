import unittest

from looping.scoring import net_improvement, rewards


class ScoringTests(unittest.TestCase):
    def test_identity_improvement_regressions_and_ceiling(self):
        for repair, transfer, retained, expected in [
            (0, 0, 6, 0), (1, 0, 6, 100/11),
            (2, 1, 5, 200/11), (0, 0, 5, 0), (7, 4, 6, 100),
        ]:
            diagnostics = dict(repair=repair/7, transfer=transfer/4, specificity=retained/6)
            with self.subTest(diagnostics=diagnostics):
                self.assertAlmostEqual(net_improvement(diagnostics)['score'], expected)
                self.assertAlmostEqual(rewards(diagnostics)['reward'] * 100, expected)

    def test_floor_does_not_hide_regression_count(self):
        result = net_improvement(dict(repair=0, transfer=0, specificity=5/6))
        self.assertEqual(result['regressions'], 1)
        self.assertLess(result['signed_net_improvement'], 0)

    def test_reject_mismatched_or_missing_results(self):
        with self.assertRaises(ValueError):
            net_improvement(dict(repair=.5, transfer=0, specificity=1))
        with self.assertRaises(KeyError):
            net_improvement({})
