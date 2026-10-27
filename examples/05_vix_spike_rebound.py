"""案例 05：VIX 单日飙升 20% 后，SPY 次日真的反弹吗？（1993—2026）

市场：SPY 与 ^VIX 日线（yfinance）。有效样本自 SPY 上市（1993）起。
事前定义：
  飙升日：VIX 收盘价单日涨幅 >= +20%。
  检验：SPY 次日对数收益，飙升日 vs 正常日，HAC 标准误（5 阶）。
预注册杀死条件：
  K1：飙升后次日 SPY 平均收益 <= 0 即杀死“反弹”假设；
  K2：安慰剂（飙升日当天的收益，被当作“次日”）均值 >= 真实次日均值即杀死。
原文结论：两条均未触发，假设活下来，但活得并不舒服——次日均值 +0.4526%
（对照组 +0.0345%），系数 +0.4181%，HAC t=2.12，p=0.034；反弹发生在盘中段
（+0.3232%，t=2.34），隔夜段不显著（+0.1294%，t=0.71）。
注：开盘价分解是在原脚本基础上补的，报告中的盘中/隔夜拆分即出自此处。
研报：https://haimao.app/posts/51
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from anomaly_toolkit import (
    download_close,
    download_ohlc,
    hac_regression,
    kill_check,
)


def run():
    ohlc = download_ohlc("SPY", "1990-01-01", "2026-10-27")
    vix = download_close("^VIX", "1990-01-01", "2026-10-27")
    d = pd.DataFrame({"spy": ohlc["Close"], "open": ohlc["Open"], "vix": vix}).dropna()
    d["r"] = np.log(d["spy"] / d["spy"].shift(1))
    d["overnight"] = np.log(d["open"] / d["spy"].shift(1))
    d["intraday"] = np.log(d["spy"] / d["open"])
    d["vix_chg"] = d["vix"] / d["vix"].shift(1) - 1
    d = d.dropna()
    d["fwd1"] = d["r"].shift(-1)
    d["fwd_on"] = d["overnight"].shift(-1)
    d["fwd_in"] = d["intraday"].shift(-1)
    d["spike"] = (d["vix_chg"] >= 0.20).astype(float)
    d["hi_level"] = (d["vix"] >= 30).astype(float)
    print(f"样本：{d.index[0].date()} ~ {d.index[-1].date()}，n={len(d)}")

    for lbl, col in [("飙升 >=+20%", "spike"), ("VIX >= 30", "hi_level")]:
        dd = d.dropna(subset=["fwd1"])
        ev = dd[dd[col] == 1]["fwd1"]
        nm = dd[dd[col] == 0]["fwd1"]
        res = hac_regression(dd["fwd1"], dd[col])
        print(f"[{lbl}] 事件 {len(ev)} 天 | 次日均值 {ev.mean() * 100:.4f}% 中位数 {ev.median() * 100:.4f}% 正占比 {(ev > 0).mean() * 100:.1f}%")
        print(f"  正常日次日均值 {nm.mean() * 100:.4f}% | HAC 系数={res['coef'] * 100:.4f}% t={res['t']:.2f} p={res['p']:.4f}")

    dd = d.dropna(subset=["fwd1", "fwd_on", "fwd_in"])
    sp = dd[dd["spike"] == 1]
    for col, lbl in [("fwd_on", "隔夜段"), ("fwd_in", "盘中段")]:
        res = hac_regression(dd[col], dd["spike"])
        print(f"次日{lbl}：飙升后均值 {sp[col].mean() * 100:.4f}% | HAC 系数={res['coef'] * 100:.4f}% t={res['t']:.2f} p={res['p']:.4f}")
    print(f"安慰剂（飙升日当天收益）：{sp['r'].mean() * 100:.4f}% vs 真实次日 {sp['fwd1'].mean() * 100:.4f}%")

    for lbl, sub in [
        ("1993-2007", d.loc[:"2007-12-31"]),
        ("2008-2019", d.loc["2008-01-01":"2019-12-31"]),
        ("2020-2026", d.loc["2020-01-01":]),
    ]:
        ev = sub[sub["spike"] == 1]["fwd1"].dropna()
        nm = sub[sub["spike"] == 0]["fwd1"].dropna()
        print(f"  {lbl}：飙升 n={len(ev)} 次日均值 {ev.mean() * 100:.4f}% | 正常均值 {nm.mean() * 100:.4f}%")

    for lo, hi, lbl in [(0.20, 0.30, "+20~30%"), (0.30, 0.50, "+30~50%"), (0.50, 10, ">+50%")]:
        ev = d[(d["vix_chg"] >= lo) & (d["vix_chg"] < hi)]["fwd1"].dropna()
        print(f"  分层 {lbl}：n={len(ev)} 次日均值 {ev.mean() * 100:.4f}% 中位数 {ev.median() * 100:.4f}% 正占比 {(ev > 0).mean() * 100:.1f}%")

    d["fwd5"] = np.log(d["spy"].shift(-5) / d["spy"])
    ev5 = d[d["spike"] == 1]["fwd5"].dropna()
    nm5 = d[d["spike"] == 0]["fwd5"].dropna()
    print(f"5 日持有：飙升后 {ev5.mean() * 100:.4f}% | 正常后 {nm5.mean() * 100:.4f}%")

    verdict = kill_check(
        [
            {
                "name": "K1 次日反弹均值",
                "killed": sp["fwd1"].mean() <= 0,
                "detail": f"均值 {sp['fwd1'].mean() * 100:.4f}%",
            },
            {
                "name": "K2 安慰剂对照",
                "killed": sp["r"].mean() >= sp["fwd1"].mean(),
                "detail": f"安慰剂 {sp['r'].mean() * 100:.4f}% vs 真实 {sp['fwd1'].mean() * 100:.4f}%",
            },
        ]
    )
    print(verdict["summary"])


if __name__ == "__main__":
    run()
