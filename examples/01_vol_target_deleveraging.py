"""案例 01：波动率目标的强制去杠杆，在 A 股是否留下可交易的反弹？

市场：沪深 300（000300.SH），akshare 日线。
机制：波动率目标基金按 10% 年化波动目标调仓，杠杆 0-2 倍；波动上升时被迫减仓
（dlev < 0），减仓压力释放后价格可能反弹。
检验：未来 5 日收益对 dlev 回归，控制 mom20，HAC 标准误（5 阶）。
预注册杀死条件（原文）：
  K1：控制 mom20 后 dlev 系数符号不对或不显著；
  K2：强制减仓事件后 5 日平均收益不显著；
  K3：置换安慰剂不能排除（经验 p >= 0.05）；
  K4：扣 5bps 单边成本后净 alpha 不显著。
原文结论：四条均未触发，净 alpha +0.4642%/事件（t=2.39），但由右尾驱动
（中位数 +0.17%，胜率 53.3% vs 基线 52.4%），不是稳定信号。
研报：https://haimao.app/posts/22
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from anomaly_toolkit import (
    download_akshare_index_close,
    hac_regression,
    kill_check,
    permutation_placebo,
    tstat_1sample,
)


def run():
    px = download_akshare_index_close("sh000300")
    print(f"样本：{px.index[0].date()} ~ {px.index[-1].date()}，{len(px)} 个交易日")
    rv20 = px.pct_change().rolling(20).std() * np.sqrt(252)
    lev = (0.10 / rv20).clip(0, 2)
    dlev = lev.diff()
    df = pd.DataFrame(
        {
            "fwd5": px.pct_change(5).shift(-5),
            "dlev": dlev,
            "mom20": px.pct_change(20),
        }
    ).dropna()
    print(f"回归样本量：{len(df)}")

    res = hac_regression(df["fwd5"], df["dlev"], controls={"mom20": df["mom20"]})
    print(f"dlev 系数={res['coef']:.4f} HAC t={res['t']:.2f} p={res['p']:.4f}")

    neg = df.loc[df["dlev"] < -0.05, "fwd5"]
    pos = df.loc[df["dlev"] > 0.05, "fwd5"]
    neg_t = tstat_1sample(neg)
    print(f"强制减仓事件：n={len(neg)}，后 5 日均值={neg.mean() * 100:.4f}%，t={neg_t:.2f}")
    print(f"强制加仓事件：n={len(pos)}，后 5 日均值={pos.mean() * 100:.4f}%，t={tstat_1sample(pos):.2f}")

    pl = permutation_placebo(df["fwd5"], df["dlev"], controls={"mom20": df["mom20"]})
    print(f"置换安慰剂：真实 t={pl['real_t']:.2f}，经验 p={pl['p_empirical']:.4f}，5% 分位={pl['p5']:.2f}")

    net = neg.mean() - 0.001  # 5bps 单边成本，双边 0.10%
    net_t = tstat_1sample(neg - 0.001)
    print(f"扣成本后净 alpha：{net * 100:.4f}%/事件，t={net_t:.2f}")

    verdict = kill_check(
        [
            {
                "name": "K1 dlev 系数",
                "killed": not (res["coef"] < 0 and res["p"] < 0.05),
                "detail": f"系数 {res['coef']:.4f}，p={res['p']:.4f}",
            },
            {
                "name": "K2 减仓后反弹",
                "killed": not (neg.mean() > 0 and neg_t >= 1.96),
                "detail": f"均值 {neg.mean() * 100:.4f}%，t={neg_t:.2f}",
            },
            {
                "name": "K3 置换安慰剂",
                "killed": pl["p_empirical"] >= 0.05,
                "detail": f"经验 p={pl['p_empirical']:.4f}",
            },
            {
                "name": "K4 扣成本净 alpha",
                "killed": not (net > 0 and net_t >= 1.96),
                "detail": f"净 {net * 100:.4f}%，t={net_t:.2f}",
            },
        ]
    )
    print(verdict["summary"])


if __name__ == "__main__":
    run()
