"""案例 06：沪深 300ETF 大涨大跌之后，下周会反转吗？（2012—2026）

市场：沪深 300ETF（510300.SS）调整后收盘价日线（yfinance），ETF 上市
（2012-05-28）起。用 ETF 而不用指数本身，是因为同一连接下 Yahoo 的
000300.SS 指数序列只返回最近一行数据，无法构成完整样本；ETF 跟踪沪深 300
指数，此处明确披露为可交易的指数代理，结论针对 ETF 价格序列本身。
事前定义：
  极端日：单日对数收益 >= +2.5% 为大涨日，<= -2.5% 为大跌日。
  检验：未来 5 个交易日对数收益（收盘对收盘）对极端日哑变量回归，
  Newey-West（5 阶滞后）标准误。
预注册杀死条件（反转要求“涨后跌、跌后涨”，方向反了即杀死）：
  K1：大涨日后 5 日平均收益 >= 其余交易日后 5 日平均收益，即杀死“大涨后反转”；
  K2：大跌日后 5 日平均收益 <= 其余交易日后 5 日平均收益，即杀死“大跌后反转”。
原文结论（2026-10-27 报告快照，有效样本 2012-05-29 至 2026-10-19，
n=3492，大涨日 129、大跌日 102）：两条同时触发，反转假设被杀死。
大涨日后 5 日平均 +0.3993%（其余 +0.0867%，HAC 系数 +0.3126%，t=0.80，
p=0.423）；大跌日后 5 日平均 -0.2231%（其余 +0.1080%，HAC 系数 -0.3311%，
t=-0.47，p=0.639）。次日、换阈值（2%/3%）、分时段均无对称反转形态。
注：之后重跑会延长样本，数字会小幅移动；docstring 中的数字以研报为准。
研报：https://haimao.app/posts/57
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from anomaly_toolkit import download_close, hac_regression, kill_check


def make_frame(px):
    """由收盘价序列构造检验用数据框（纯函数，不联网，可用合成数据检验）。"""
    d = pd.DataFrame({"close": pd.Series(px, dtype=float)}).sort_index()
    d["r"] = np.log(d["close"] / d["close"].shift(1))
    d["fwd1"] = d["r"].shift(-1)
    d["fwd5"] = np.log(d["close"].shift(-5) / d["close"])
    d = d.dropna(subset=["r", "fwd5"])
    d["big_up"] = (d["r"] >= 0.025).astype(float)
    d["big_down"] = (d["r"] <= -0.025).astype(float)
    return d


def _report(d, col, label):
    dd = d.dropna(subset=["fwd5"])
    ev = dd[dd[col] == 1]["fwd5"]
    nm = dd[dd[col] == 0]["fwd5"]
    res = hac_regression(dd["fwd5"], dd[col])
    print(
        f"[{label}] 事件 {len(ev)} 天（{len(ev) / len(dd) * 100:.2f}%）| "
        f"后 5 日均值 {ev.mean() * 100:.4f}% 中位数 {ev.median() * 100:.4f}% "
        f"上涨占比 {(ev > 0).mean() * 100:.1f}%"
    )
    print(
        f"  其余交易日后 5 日均值 {nm.mean() * 100:.4f}% 中位数 {nm.median() * 100:.4f}% "
        f"上涨占比 {(nm > 0).mean() * 100:.1f}% | "
        f"HAC 系数={res['coef'] * 100:.4f}% t={res['t']:.2f} p={res['p']:.4f}"
    )
    return ev, nm, res


def run():
    px = download_close("510300.SS", "2012-05-28", "2026-10-28")
    d = make_frame(px)
    print(f"样本：{d.index[0].date()} ~ {d.index[-1].date()}，n={len(d)}")
    up = d[d["big_up"] == 1]
    dn = d[d["big_down"] == 1]
    print(
        f"大涨日 {len(up)} 天，当天平均 {up['r'].mean() * 100:.4f}% | "
        f"大跌日 {len(dn)} 天，当天平均 {dn['r'].mean() * 100:.4f}%"
    )

    ev_up, nm_up, _ = _report(d, "big_up", "大涨日")
    ev_dn, nm_dn, _ = _report(d, "big_down", "大跌日")

    # 次日窗口：反转在 1 天尺度上也不存在（原文：t=1.04 / t=0.20）。
    for col, label in [("big_up", "大涨日次日"), ("big_down", "大跌日次日")]:
        dd = d.dropna(subset=["fwd1"])
        res = hac_regression(dd["fwd1"], dd[col])
        print(
            f"[{label}] 事件均值 {dd[dd[col] == 1]['fwd1'].mean() * 100:.4f}% | "
            f"其余均值 {dd[dd[col] == 0]['fwd1'].mean() * 100:.4f}% | "
            f"HAC 系数={res['coef'] * 100:.4f}% t={res['t']:.2f} p={res['p']:.4f}"
        )

    # 阈值敏感性：2% / 3% 口径下也没有对称反转形态。
    for thr in (0.02, 0.03):
        for col, label in [
            ((d["r"] >= thr).astype(float), f">=+{thr * 100:.0f}%"),
            ((d["r"] <= -thr).astype(float), f"<=-{thr * 100:.0f}%"),
        ]:
            dd = d.assign(dummy=col).dropna(subset=["fwd5"])
            ev = dd[dd["dummy"] == 1]["fwd5"]
            print(f"  阈值 {label}：n={len(ev)} 后 5 日均值 {ev.mean() * 100:.4f}%")

    # 分时段：大跌日后 5 日均值前后两段符号相反，效应不稳定。
    for lbl, sub in [
        ("2012-2019", d.loc[:"2019-12-31"]),
        ("2020-2026", d.loc["2020-01-01":]),
    ]:
        for col, name in [("big_up", "大涨"), ("big_down", "大跌")]:
            ev = sub[sub[col] == 1]["fwd5"]
            nm = sub[sub[col] == 0]["fwd5"]
            print(
                f"  {lbl} {name}：n={len(ev)} 后 5 日均值 {ev.mean() * 100:.4f}% "
                f"| 其余 {nm.mean() * 100:.4f}%"
            )

    # 左尾：大跌日后 5 日收益低于 -5% 的占比远高于普通交易日，均值被少数股灾日拖累。
    dn5 = d[d["big_down"] == 1]["fwd5"]
    nm5 = d[d["big_down"] == 0]["fwd5"]
    print(
        f"大跌日后 5 日 < -5% 占比 {(dn5 < -0.05).mean() * 100:.1f}%（{(dn5 < -0.05).sum()}/{len(dn5)}）"
        f" | 普通交易日 {(nm5 < -0.05).mean() * 100:.1f}%"
    )
    worst = d[d["big_down"] == 1].nsmallest(5, "fwd5")[["r", "fwd5"]]
    print("最差 5 个大跌日后 5 日收益：")
    for dt, row in worst.iterrows():
        print(f"  {dt.date()}：当天 {row['r'] * 100:.2f}%，后 5 日 {row['fwd5'] * 100:.2f}%")

    verdict = kill_check(
        [
            {
                "name": "K1 大涨后反转",
                "killed": ev_up.mean() >= nm_up.mean(),
                "detail": f"大涨后 {ev_up.mean() * 100:.4f}% vs 其余 {nm_up.mean() * 100:.4f}%",
            },
            {
                "name": "K2 大跌后反转",
                "killed": ev_dn.mean() <= nm_dn.mean(),
                "detail": f"大跌后 {ev_dn.mean() * 100:.4f}% vs 其余 {nm_dn.mean() * 100:.4f}%",
            },
        ]
    )
    print(verdict["summary"])


if __name__ == "__main__":
    run()
