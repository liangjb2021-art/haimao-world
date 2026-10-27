"""案例 02：月末月初效应在 SPY 上还存在吗？（2000—2026）

市场：SPY 调整后收盘价日线（yfinance）。
定义（Ariel 1987 原定义）：每月最后一个交易日 + 下月前三个交易日为 TOM 日。
检验：日收益对 TOM 哑变量回归，Newey-West（5 阶）标准误。
预注册杀死条件：
  K1：全样本回归系数 HAC t < 1.96 即杀死；
  K2：安慰剂窗口（TOM 平移 +7 个交易日）效应 >= 真实 TOM 效应即杀死。
原文结论：假设被杀死。全样本 TOM 日均 +0.0653% vs 其余 +0.0327%，
但回归系数 0.0326%，HAC t=0.94，p=0.35；2013—2026 年系数 -0.0041%（t=-0.10），效应归零。
安慰剂系数 -0.0123%（t=-0.35），K2 未触发——杀死结论不是检验本身太松。

注意：本案例是根据第二篇研报的方法节重建的，原运行脚本未存档；
数据版本更新可能导致数字小幅出入。
研报：https://haimao.app/posts/27
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from anomaly_toolkit import (
    download_close,
    event_stats,
    hac_regression,
    kill_check,
    log_returns,
    placebo_shift,
)


def tom_dummy(index):
    """Ariel 1987 定义：每月最后一个交易日 + 下月前三个交易日。"""
    dummy = pd.Series(0.0, index=index)
    tmp = pd.DataFrame(index=index)
    tmp["ym"] = list(zip(index.year, index.month))
    for _, grp in tmp.groupby("ym"):
        days = grp.index
        dummy.loc[days[-1]] = 1.0
        dummy.loc[days[:3]] = 1.0
    return dummy


def run():
    spy = download_close("SPY", "2000-01-01", "2026-10-23")
    r = log_returns(spy)
    tom = tom_dummy(r.index)
    print(f"样本：{r.index[0].date()} ~ {r.index[-1].date()}，n={len(r)}")
    print(f"TOM 日占比：{tom.mean() * 100:.1f}%")

    stats = event_stats(r, tom)
    print(f"TOM 日均：{stats['event']['mean'] * 100:.4f}% | 其余日均：{stats['normal']['mean'] * 100:.4f}%")
    res = hac_regression(r, tom)
    print(f"全样本：系数={res['coef'] * 100:.4f}% HAC t={res['t']:.2f} p={res['p']:.4f}")

    pl = hac_regression(r, placebo_shift(tom, 7))
    print(f"安慰剂（+7 交易日）：系数={pl['coef'] * 100:.4f}% t={pl['t']:.2f}")

    for label, sl in [
        ("2000-2012", slice("2000-01-01", "2012-12-31")),
        ("2013-2026", slice("2013-01-01", "2026-10-22")),
    ]:
        s = hac_regression(r.loc[sl], tom.loc[sl])
        print(f"分段 {label}：系数={s['coef'] * 100:.4f}% t={s['t']:.2f} p={s['p']:.4f} n={s['n']}")

    verdict = kill_check(
        [
            {
                "name": "K1 全样本显著性",
                "killed": abs(res["t"]) < 1.96,
                "detail": f"t={res['t']:.2f}",
            },
            {
                "name": "K2 安慰剂对照",
                "killed": pl["coef"] >= res["coef"],
                "detail": f"安慰剂系数 {pl['coef'] * 100:.4f}% vs 真实 {res['coef'] * 100:.4f}%",
            },
        ]
    )
    print(verdict["summary"])


if __name__ == "__main__":
    run()
