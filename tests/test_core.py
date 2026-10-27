"""框架单元测试：只用合成数据，不联网。

运行：python tests/test_core.py
"""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from anomaly_toolkit import (  # noqa: E402
    event_stats,
    hac_regression,
    kill_check,
    log_returns,
    placebo_shift,
    tstat_1sample,
)


class TestCore(unittest.TestCase):
    def test_log_returns(self):
        px = pd.Series([100.0, 110.0, 121.0])
        r = log_returns(px)
        self.assertEqual(len(r), 2)
        self.assertAlmostEqual(r.iloc[0], np.log(1.1), places=10)
        self.assertAlmostEqual(r.iloc[1], np.log(1.1), places=10)

    def test_hac_regression_recovers_effect(self):
        rng = np.random.default_rng(0)
        n = 2000
        dummy = pd.Series((rng.random(n) < 0.2).astype(float))
        y = pd.Series(1.0 * dummy + rng.normal(0, 1, n))
        res = hac_regression(y, dummy)
        self.assertAlmostEqual(res["coef"], 1.0, delta=0.15)
        self.assertLess(res["p"], 0.01)

    def test_hac_regression_null(self):
        rng = np.random.default_rng(1)
        n = 2000
        dummy = pd.Series((rng.random(n) < 0.2).astype(float))
        y = pd.Series(rng.normal(0, 1, n))
        res = hac_regression(y, dummy)
        self.assertLess(abs(res["t"]), 2.0)

    def test_event_stats(self):
        y = pd.Series([0.02, -0.01, 0.03, 0.01])
        dummy = pd.Series([1.0, 1.0, 0.0, 0.0])
        s = event_stats(y, dummy)
        self.assertAlmostEqual(s["event"]["mean"], 0.005)
        self.assertEqual(s["event"]["n"], 2)
        self.assertAlmostEqual(s["event"]["positive_share"], 0.5)
        self.assertAlmostEqual(s["normal"]["mean"], 0.02)

    def test_placebo_shift(self):
        dummy = pd.Series([1.0, 0.0, 0.0, 1.0])
        shifted = placebo_shift(dummy, 1)
        self.assertListEqual(list(shifted), [0.0, 1.0, 0.0, 0.0])

    def test_tstat_1sample(self):
        x = pd.Series([0.02, 0.02, 0.02, 0.02])
        # std = 0 会导致 inf；用有波动的序列
        x = pd.Series([0.01, 0.02, 0.03, 0.04])
        t = tstat_1sample(x)
        expected = x.mean() / x.std(ddof=1) * np.sqrt(len(x))
        self.assertAlmostEqual(t, expected)

    def test_kill_check(self):
        killed = kill_check(
            [
                {"name": "K1", "killed": False, "detail": "ok"},
                {"name": "K2", "killed": True, "detail": "bad"},
            ]
        )
        self.assertEqual(killed["verdict"], "killed")
        self.assertEqual(killed["triggered"], ["K2"])
        survived = kill_check([{"name": "K1", "killed": False, "detail": "ok"}])
        self.assertEqual(survived["verdict"], "survived")


if __name__ == "__main__":
    unittest.main(verbosity=2)
