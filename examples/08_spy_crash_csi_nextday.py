"""案例 08：美股单日暴跌超 2% 后，次日 A 股真的跟跌吗？（2012—2026）

教科书直觉：“美股跌了，明天 A 股低开”。本案例检验这个直觉。
数据：Yahoo Finance 日度调整后收盘价，SPY（标普 500 ETF）、
510300.SS（华泰柏瑞沪深 300ETF，用作沪深 300 的可交易代理——与案例 06
相同，指数 000300.SS 在 Yahoo 上数据不全，故用 ETF 代理并在此明示）。
样本为两者都有数据的交易日对齐后共 3641 个美股交易日，
2012 年 5 月—2026 年 10 月。收益均为对数收益。
时序定义（检验前固定，关键细节）：A 股收盘在前、美股收盘在后，因此某美股
交易日 t 收盘时的信息，A 股最早只能在下一个 A 股交易日反应。本文的
“次日”严格定义为：日历日期严格晚于 t 的第一个 510300 交易日。例如美股
周五暴跌，对应的是下周一的 A 股；美股 2025 年 4 月 3 日、4 日（关税冲击）
暴跌，对应的都是 4 月 7 日的 A 股。
事前定义：
  暴跌日：SPY 对数收益 <= -2%（约等于简单收益 -1.98%）；
  大涨日（安慰剂对照）：SPY 对数收益 >= +2%。
  检验：次日 510300 收益对暴跌日哑变量回归，Newey-West（5 阶滞后）标准误。
预注册杀死条件（原假设：暴跌日后次日 A 股为负，且低于正常日子）：
  K1：暴跌日后次日 A 股均值 >= 0，则原假设被否决；
  K2（安慰剂）：大涨日后次日 A 股同样为负，且幅度不小于暴跌日后跌幅的
      一半，则“次日跟跌”不是暴跌特有的效应，原假设的表述被否决。
原文结论（2026-10-30 报告快照，n=3641，暴跌日 108 天）：K1、K2 均未触发，
原假设活下来，但只是预测关联、不是因果证明。暴跌日后次日 A 股均值
-0.8395%（中位数 -0.5752%，下跌 63.9%），正常日子 +0.0225%；HAC 系数
-0.8620%，t=-3.26，p≈0.001；Welch t=-3.93，p≈0.0001。安慰剂大涨日 n=77，
次日 A 股均值 +0.5768%（中位数 +0.2775%，下跌 37.7%），HAC 系数 +0.5924%，
t=2.60，p≈0.009，符号与暴跌日相反。跌得越狠跟跌越多：-2% 至 -3%（n=69）
-0.4853%，-3% 至 -5%（n=32）-1.3265%，<=-5%（n=7）-2.1050%。三个时段
符号一致：2012—2016 年（n=20）-1.4444%，2017—2021 年（n=50）-0.7060%，
2022—2026 年（n=38）-0.6968%。5 日累计：暴跌日后 -0.8998%，正常日子
+0.1121%，一周内没有快速反转。去重后（108 天映射到 106 个不同次日，
只计第一次）均值 -0.7819%，下跌占比 63.2%，结论不变。同一交易日 sanity
check：暴跌日当天的 A 股（时序上不可能反应当晚美股收盘）均值已为
-0.6408%，正常日子 +0.0342%——说明共同冲击无法排除，统计量是预测关联，
不是因果识别。
注：之后重跑会延长样本，数字会小幅移动；docstring 中的数字以研报为准。
研报：https://haimao.app/posts/68
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy import stats

from anomaly_toolkit import download_close, hac_regression, kill_check, log_returns


def map_next_day(spy_index, csi_index):
    """对每个 SPY 交易日，返回日历日期严格晚于它的第一个 510300 交易日位置。

    纯函数，不联网。返回与 spy_index 等长的整数数组，-1 表示其后没有
    510300 交易日（样本末尾）。
    """
    spy_vals = np.asarray(spy_index.values)
    csi_vals = np.asarray(csi_index.values)
    pos = np.searchsorted(csi_vals, spy_vals, side="right")
    pos = np.where(pos < len(csi_vals), pos, -1)
    return pos


def make_frame(spy_close, csi_close):
    """由两只 ETF 的收盘价构造检验用数据框（纯函数，不联网）。

    每行是一个 SPY 交易日 t：r_spy 为 t 当天收益，next_csi 为 t 之后第一个
    510300 交易日的收益，same_csi 为 t 当天（若为 510300 交易日）的收益。
    """
    r_spy = log_returns(spy_close)
    r_csi = log_returns(csi_close)
    pos = map_next_day(r_spy.index, r_csi.index)
    valid = pos >= 0
    next_csi = pd.Series(np.nan, index=r_spy.index, dtype=float)
    next_csi.iloc[np.where(valid)[0]] = r_csi.values[pos[valid]]
    d = pd.DataFrame({"r_spy": r_spy, "next_csi": next_csi})
    d["same_csi"] = r_csi.reindex(d.index)
    d = d.dropna(subset=["next_csi"])
    d["crash"] = (d["r_spy"] <= -0.02).astype(float)
    d["surge"] = (d["r_spy"] >= 0.02).astype(float)
    return d, r_csi


def _report(d, col, label):
    ev = d[d[col] == 1]["next_csi"]
    nm = d[d[col] == 0]["next_csi"]
    res = hac_regression(d["next_csi"], d[col])
    t, p = stats.ttest_ind(ev, nm, equal_var=False)
    print(
        f"[{label}] 事件 {len(ev)} 天（{len(ev) / len(d) * 100:.2f}%）| "
        f"次日 A 股均值 {ev.mean() * 100:.4f}% 中位数 {ev.median() * 100:.4f}% "
        f"下跌占比 {(ev < 0).mean() * 100:.1f}%"
    )
    print(
        f"  正常日子次日均值 {nm.mean() * 100:.4f}% | "
        f"HAC 系数={res['coef'] * 100:.4f}% t={res['t']:.2f} p={res['p']:.4f} | "
        f"Welch t={t:.2f} p={p:.4f}"
    )
    return ev, nm, res


def run():
    spy = download_close("SPY", "2012-05-01", "2026-11-01")
    csi = download_close("510300.SS", "2012-05-01", "2026-11-01")
    d, r_csi = make_frame(spy, csi)
    print(f"样本：{d.index[0].date()} ~ {d.index[-1].date()}，n={len(d)}")

    ev_cr, nm_cr, _ = _report(d, "crash", "美股暴跌日 (<=-2%)")
    ev_su, nm_su, _ = _report(d, "surge", "美股大涨日 (>=+2%，安慰剂)")

    # 严重程度分层：跌得越狠，次日跟跌是否单调递增。
    for lbl, mask in [
        ("-2% 至 -3%", (d["r_spy"] <= -0.02) & (d["r_spy"] > -0.03)),
        ("-3% 至 -5%", (d["r_spy"] <= -0.03) & (d["r_spy"] > -0.05)),
        ("<=-5%", d["r_spy"] <= -0.05),
    ]:
        ev = d[mask]["next_csi"]
        print(f"  {lbl}：n={len(ev)} 次日 A 股均值 {ev.mean() * 100:.4f}%")

    # 分时段：三段符号是否一致。
    for lbl, sub in [
        ("2012-2016", d.loc[:"2016-12-31"]),
        ("2017-2021", d.loc["2017-01-01":"2021-12-31"]),
        ("2022-2026", d.loc["2022-01-01":]),
    ]:
        ev = sub[sub["crash"] == 1]["next_csi"]
        print(f"  {lbl}：暴跌日 n={len(ev)} 次日 A 股均值 {ev.mean() * 100:.4f}%")

    # 5 日累计：暴跌日后 5 个 A 股交易日累计收益 vs 正常日子。
    pos = map_next_day(d.index, r_csi.index)
    fwd5 = pd.Series(np.nan, index=d.index, dtype=float)
    for i, p in enumerate(pos):
        if 0 <= p and p + 5 <= len(r_csi):
            fwd5.iloc[i] = r_csi.values[p : p + 5].sum()
    dd5 = pd.DataFrame({"crash": d["crash"], "fwd5": fwd5}).dropna()
    ev5 = dd5[dd5["crash"] == 1]["fwd5"]
    nm5 = dd5[dd5["crash"] == 0]["fwd5"]
    print(
        f"[5 日累计] 暴跌日后 n={len(ev5)} 均值 {ev5.mean() * 100:.4f}% | "
        f"正常日子 {nm5.mean() * 100:.4f}%"
    )

    # 去重：同一 A 股交易日被多个美股暴跌日映射时只计第一次。
    # 注意：先筛出暴跌日再去重——若对全样本去重，暴跌日会因其映射的次日
    # 已被更早的非暴跌日占用而被误删。
    pos_all = map_next_day(d.index, r_csi.index)
    nxt_dates = pd.Series(
        np.where(pos_all >= 0, r_csi.index.values[np.maximum(pos_all, 0)], pd.NaT),
        index=d.index,
    )
    cr = d[d["crash"] == 1].assign(nxt=pd.to_datetime(nxt_dates))
    dedup = cr.drop_duplicates(subset=["nxt"], keep="first")
    ev_dd = dedup["next_csi"]
    print(
        f"[去重] 暴跌日映射到 {dedup[dedup['crash'] == 1]['nxt'].nunique()} 个不同次日，"
        f"均值 {ev_dd.mean() * 100:.4f}% 下跌占比 {(ev_dd < 0).mean() * 100:.1f}%"
    )

    # 同一交易日 sanity check：暴跌日当天的 A 股（不可能反应当晚美股收盘）。
    same = d.dropna(subset=["same_csi"])
    ev_same = same[same["crash"] == 1]["same_csi"]
    nm_same = same[same["crash"] == 0]["same_csi"]
    print(
        f"[同一交易日] 暴跌日当天 A 股均值 {ev_same.mean() * 100:.4f}% | "
        f"正常日子 {nm_same.mean() * 100:.4f}%"
    )

    # 反例：暴跌最狠、次日 A 股却涨得最多的几天。
    worst = d[d["crash"] == 1].nsmallest(3, "r_spy")
    print("暴跌最狠的 3 天与其后次日 A 股：")
    for dt, row in worst.iterrows():
        print(
            f"  {dt.date()}：SPY {row['r_spy'] * 100:.2f}%，"
            f"次日 A 股 {row['next_csi'] * 100:.2f}%"
        )

    verdict = kill_check(
        [
            {
                "name": "K1 暴跌日后次日 A 股均值 >= 0",
                "killed": ev_cr.mean() >= 0,
                "detail": f"暴跌日后次日均值 {ev_cr.mean() * 100:.4f}% vs 正常日子 {nm_cr.mean() * 100:.4f}%",
            },
            {
                "name": "K2 大涨日后次日 A 股同样为负",
                "killed": bool(ev_su.mean() < 0 and abs(ev_su.mean()) >= 0.5 * abs(ev_cr.mean())),
                "detail": f"大涨日后次日均值 {ev_su.mean() * 100:.4f}% vs 暴跌日后 {ev_cr.mean() * 100:.4f}%",
            },
        ]
    )
    print(verdict["summary"])


if __name__ == "__main__":
    run()
