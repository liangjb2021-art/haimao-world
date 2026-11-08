"""案例 07：美债 10 年收益率单日飙升 20bp 后，纳指真的跌得比大盘多吗？（1999—2026）

教科书直觉：长期利率上行时，久期更长的成长股（纳指）应该比大盘跌得更多。
本案例检验这个直觉。数据：Yahoo Finance 日度，^TNX（10 年期美债收益率，
以 % 计）、QQQ、SPY，样本 1999-03-11 至 2026-10-26，共 6942 个交易日。
收益率与 ETF 均用调整后收盘价；ETF 收益为对数收益，收益率变化为水平差分。
事前定义：
  飙升日：dTNX = TNX_t - TNX_{t-1} >= +0.20 个百分点（20bp）。
  暴跌日（安慰剂）：dTNX <= -0.20 个百分点。
  检验：当天 spread = r_QQQ - r_SPY 对飙升日哑变量回归，
  Newey-West（5 阶滞后）标准误。
预注册杀死条件（原假设：飙升日当天 spread < 0，即纳指跑输大盘）：
  K1：飙升日当天 spread 均值 >= 0，则原假设被否决；
  K2：暴跌日当天 spread 同样为负，且幅度不小于飙升日的一半，则 spread
      不是收益率冲击的特有效应，原假设的表述被否决。
原文结论（2026-10-29 报告快照，n=6942，飙升日 21 天）：K1 触发，
原假设被否决。飙升日当天 spread 均值 +0.6377%（中位数 +0.4228%，
为负仅 7/21），HAC 系数 +0.6317%，t=2.62，p=0.0089；非飙升日
spread 均值 +0.0060%。安慰剂暴跌日 n=16，spread 均值 +0.2263%，
HAC 系数 +0.2190%，t=0.65，p=0.5151，K2 未触发。分时段符号一致为正
（1999—2007 +0.7761%，2008—2019 +0.3602%，2020—2026 +0.8814%）；
放宽阈值到 +15bp（n=75）+0.4970%、+10bp（n=307）+0.1929%，符号不变。
次日 QQQ：飙升日 -0.6153% vs 其余 +0.0425%，Welch t=-1.60，p=0.126，
不显著，不宜当作信号。
注：之后重跑会延长样本，数字会小幅移动；docstring 中的数字以研报为准。
研报：https://haimao.app/posts/63
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy import stats

from anomaly_toolkit import download_closes, hac_regression, kill_check


def make_frame(tnx, qqq_close, spy_close):
    """由收益率序列与两只 ETF 收盘价构造检验用数据框（纯函数，不联网）。

    tnx 为收益率水平（%），qqq_close / spy_close 为调整后收盘价。
    """
    d = pd.DataFrame(
        {
            "tnx": pd.Series(tnx, dtype=float),
            "qqq": pd.Series(qqq_close, dtype=float),
            "spy": pd.Series(spy_close, dtype=float),
        }
    ).sort_index()
    d["dtnx"] = d["tnx"] - d["tnx"].shift(1)
    d["r_qqq"] = np.log(d["qqq"] / d["qqq"].shift(1))
    d["r_spy"] = np.log(d["spy"] / d["spy"].shift(1))
    d["spread"] = d["r_qqq"] - d["r_spy"]
    d["fwd1_qqq"] = d["r_qqq"].shift(-1)
    d = d.dropna(subset=["dtnx", "spread"])
    d["spike"] = (d["dtnx"] >= 0.20).astype(float)
    d["plunge"] = (d["dtnx"] <= -0.20).astype(float)
    return d


def _report(d, col, label):
    dd = d.dropna(subset=["spread"])
    ev = dd[dd[col] == 1]["spread"]
    nm = dd[dd[col] == 0]["spread"]
    res = hac_regression(dd["spread"], dd[col])
    print(
        f"[{label}] 事件 {len(ev)} 天（{len(ev) / len(dd) * 100:.2f}%）| "
        f"spread 均值 {ev.mean() * 100:.4f}% 中位数 {ev.median() * 100:.4f}% "
        f"为负占比 {(ev < 0).mean() * 100:.1f}%"
    )
    print(
        f"  QQQ 均值 {dd[dd[col] == 1]['r_qqq'].mean() * 100:.4f}% "
        f"SPY 均值 {dd[dd[col] == 1]['r_spy'].mean() * 100:.4f}% | "
        f"其余交易日 spread 均值 {nm.mean() * 100:.4f}% | "
        f"HAC 系数={res['coef'] * 100:.4f}% t={res['t']:.2f} p={res['p']:.4f}"
    )
    return ev, nm, res


def run():
    px = download_closes(["^TNX", "QQQ", "SPY"], "1999-03-11", "2026-10-27")
    d = make_frame(px["^TNX"], px["QQQ"], px["SPY"])
    print(f"样本：{d.index[0].date()} ~ {d.index[-1].date()}，n={len(d)}")

    ev_sp, nm_sp, _ = _report(d, "spike", "收益率飙升日 (+20bp)")
    ev_pl, nm_pl, _ = _report(d, "plunge", "收益率暴跌日 (-20bp，安慰剂)")

    # 次日表现：飙升日次日 QQQ 均值 vs 其余（Welch t）。
    dd = d.dropna(subset=["fwd1_qqq"])
    ev1 = dd[dd["spike"] == 1]["fwd1_qqq"]
    nm1 = dd[dd["spike"] == 0]["fwd1_qqq"]
    t, p = stats.ttest_ind(ev1, nm1, equal_var=False)
    print(
        f"[次日 QQQ] 飙升日次日均值 {ev1.mean() * 100:.4f}% | "
        f"其余 {nm1.mean() * 100:.4f}% | Welch t={t:.2f} p={p:.4f}"
    )

    # 阈值敏感性：+15bp / +10bp 口径下符号是否不变。
    for thr in (0.15, 0.10):
        col = (d["dtnx"] >= thr).astype(float)
        dd = d.assign(dummy=col).dropna(subset=["spread"])
        ev = dd[dd["dummy"] == 1]["spread"]
        res = hac_regression(dd["spread"], dd["dummy"])
        print(
            f"  阈值 >={thr * 100:.0f}bp：n={len(ev)} spread 均值 {ev.mean() * 100:.4f}% "
            f"| HAC 系数={res['coef'] * 100:.4f}% t={res['t']:.2f} p={res['p']:.4f}"
        )

    # 分时段：三段符号是否一致。
    for lbl, sub in [
        ("1999-2007", d.loc[:"2007-12-31"]),
        ("2008-2019", d.loc["2008-01-01":"2019-12-31"]),
        ("2020-2026", d.loc["2020-01-01":]),
    ]:
        ev = sub[sub["spike"] == 1]["spread"]
        print(f"  {lbl}：飙升日 n={len(ev)} spread 均值 {ev.mean() * 100:.4f}%")

    # 最大的几次跳升：成因各异，教科书案例只是少数。
    sp = d[d["spike"] == 1].nlargest(5, "dtnx")[["dtnx", "r_qqq", "r_spy", "spread"]]
    print("最大的 5 个收益率跳升日：")
    for dt, row in sp.iterrows():
        print(
            f"  {dt.date()}：dTNX +{row['dtnx'] * 100:.1f}bp，"
            f"QQQ {row['r_qqq'] * 100:.2f}%，SPY {row['r_spy'] * 100:.2f}%，"
            f"spread {row['spread'] * 100:.2f}%"
        )
    worst = d[d["spike"] == 1].nsmallest(1, "spread")[["dtnx", "r_qqq", "r_spy", "spread"]]
    for dt, row in worst.iterrows():
        print(
            f"  spread 最负的飙升日 {dt.date()}：dTNX +{row['dtnx'] * 100:.1f}bp，"
            f"QQQ {row['r_qqq'] * 100:.2f}%，SPY {row['r_spy'] * 100:.2f}%，"
            f"spread {row['spread'] * 100:.2f}%"
        )

    verdict = kill_check(
        [
            {
                "name": "K1 飙升日 spread 为正",
                "killed": ev_sp.mean() >= 0,
                "detail": f"飙升日 spread 均值 {ev_sp.mean() * 100:.4f}% vs 其余 {nm_sp.mean() * 100:.4f}%",
            },
            {
                "name": "K2 暴跌日对称负 spread",
                "killed": bool(ev_pl.mean() < 0 and abs(ev_pl.mean()) >= 0.5 * abs(ev_sp.mean())),
                "detail": f"暴跌日 spread 均值 {ev_pl.mean() * 100:.4f}% vs 飙升日 {ev_sp.mean() * 100:.4f}%",
            },
        ]
    )
    print(verdict["summary"])


if __name__ == "__main__":
    run()
