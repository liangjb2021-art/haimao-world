"""案例 03：恒生指数有“周一魔咒”吗？（2000—2026）

市场：恒生指数（^HSI）日线（yfinance），对数收益。
检验：日收益对周一哑变量回归，Newey-West（5 阶）标准误；
安慰剂为周一哑变量平移 +1 个交易日；分段 2000-2012 / 2013-2026。
原文结论：周一平均跌 0.036%，但检验不显著——阴性结果。
研报：见专栏 https://haimao.app/users/haimao2016
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from anomaly_toolkit import (
    download_close,
    hac_regression,
    log_returns,
    placebo_shift,
)


def run():
    px = download_close("^HSI", "2000-01-01", "2026-10-25")
    r = log_returns(px)
    d = pd.DataFrame({"r": r})
    d["dow"] = d.index.dayofweek  # Mon=0
    d["mon"] = (d["dow"] == 0).astype(float)
    print(f"样本：{r.index[0].date()} ~ {r.index[-1].date()}，n={len(d)}")
    print(d.groupby("dow")["r"].agg(["mean", "count"]) * 100)

    res = hac_regression(d["r"], d["mon"])
    print(f"周一全样本：系数={res['coef'] * 100:.4f}% t={res['t']:.2f} p={res['p']:.4f}")
    pl = hac_regression(d["r"], placebo_shift(d["mon"], 1))
    print(f"安慰剂（+1 交易日）：系数={pl['coef'] * 100:.4f}% t={pl['t']:.2f}")

    for a, b in [("2000-01-01", "2012-12-31"), ("2013-01-01", "2026-10-25")]:
        sub = d.loc[a:b]
        s = hac_regression(sub["r"], sub["mon"])
        print(f"分段 {a[:4]}-{b[:4]}：n={s['n']} 系数={s['coef'] * 100:.4f}% t={s['t']:.2f} p={s['p']:.3f}")

    bh = (1 + d["r"]).prod() - 1
    mon_only = (1 + d.loc[d["mon"] == 1, "r"]).prod() - 1
    print(f"买入持有累计：{bh * 100:.1f}% | 仅周一持有累计：{mon_only * 100:.1f}%")
    print(f"周一占交易日：{d['mon'].mean() * 100:.1f}% | 周一占累计对数收益：{d.loc[d['mon'] == 1, 'r'].sum() / d['r'].sum() * 100:.1f}%")


if __name__ == "__main__":
    run()
