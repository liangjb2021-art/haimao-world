"""案例 04：美股暴跌日，黄金真的避险吗？（2004—2026）

市场：SPY 与 GLD 调整后日线（yfinance），GLD 自 2004-11-18 上市起。
暴跌日定义：SPY 对数收益 < -2%。
预注册杀死条件：
  K1：暴跌日 GLD 平均收益 <= 0 即杀死“当日避险”假设；
  K2：安慰剂（暴跌日的次日 GLD 收益）均值 >= 真实暴跌日均值即杀死。
原文结论：两条均触发，假设被杀死。暴跌日 GLD 均值 -0.0838%，HAC t=-0.83，p=0.41。
研报：https://haimao.app/posts/45
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from anomaly_toolkit import (
    download_closes,
    hac_regression,
    kill_check,
    log_returns,
)


def run():
    px = download_closes(["SPY", "GLD"], "2004-11-18", "2026-10-24")
    d = pd.DataFrame({"spy": log_returns(px["SPY"]), "gld": log_returns(px["GLD"])}).dropna()
    print(f"样本：{d.index[0].date()} ~ {d.index[-1].date()}，n={len(d)}")
    d["crash"] = (d["spy"] < -0.02).astype(float)
    cr, nm = d[d["crash"] == 1], d[d["crash"] == 0]
    print(f"暴跌日：{len(cr)} 天（{len(cr) / len(d) * 100:.2f}%）")
    print(f"GLD 暴跌日均值：{cr['gld'].mean() * 100:.4f}% | 正常日均值：{nm['gld'].mean() * 100:.4f}%")
    print(f"GLD 暴跌日正收益占比：{(cr['gld'] > 0).mean() * 100:.1f}% | 正常日：{(nm['gld'] > 0).mean() * 100:.1f}%")

    res = hac_regression(d["gld"], d["crash"])
    print(f"GLD ~ 暴跌哑变量：系数={res['coef'] * 100:.4f}% t={res['t']:.2f} p={res['p']:.4f}")
    pl_mean = d["gld"].shift(-1)[d["crash"] == 1].mean()
    print(f"安慰剂（暴跌次日 GLD 均值）：{pl_mean * 100:.4f}%")

    for lbl, sub in [("暴跌日", cr), ("正常日", nm)]:
        b = np.polyfit(sub["spy"], sub["gld"], 1)[0]
        print(f"{lbl}：corr={sub['spy'].corr(sub['gld']):.3f} beta={b:.3f} n={len(sub)}")

    for lo, hi, lbl in [(-0.03, -0.02, "-3%..-2%"), (-0.05, -0.03, "-5%..-3%"), (-1, -0.05, "<-5%")]:
        s = d[(d["spy"] >= lo) & (d["spy"] < hi)]
        print(f"分层 {lbl}：n={len(s)} GLD 均值={s['gld'].mean() * 100:.4f}% 正占比={(s['gld'] > 0).mean() * 100:.0f}%")

    d["gld_fwd5"] = np.log(px["GLD"].shift(-5) / px["GLD"])
    f = d.dropna(subset=["gld_fwd5"])
    fc, fn = f[f["crash"] == 1], f[f["crash"] == 0]
    print(f"GLD 暴跌后 5 日：{fc['gld_fwd5'].mean() * 100:.3f}%（n={len(fc)}）| 正常后 5 日：{fn['gld_fwd5'].mean() * 100:.3f}%")

    w = d.nsmallest(20, "spy")[["spy", "gld"]]
    print(f"最差 20 个 SPY 日的 GLD 均值：{w['gld'].mean() * 100:.3f}% 正占比：{(w['gld'] > 0).mean() * 100:.0f}%")

    verdict = kill_check(
        [
            {
                "name": "K1 暴跌日 GLD 均值",
                "killed": cr["gld"].mean() <= 0,
                "detail": f"均值 {cr['gld'].mean() * 100:.4f}%",
            },
            {
                "name": "K2 安慰剂对照",
                "killed": pl_mean >= cr["gld"].mean(),
                "detail": f"安慰剂 {pl_mean * 100:.4f}% vs 真实 {cr['gld'].mean() * 100:.4f}%",
            },
        ]
    )
    print(verdict["summary"])


if __name__ == "__main__":
    run()
