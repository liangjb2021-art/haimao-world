"""异象复现的核心工具：HAC 回归、事件统计、安慰剂检验、杀死条件裁决。

设计原则（对应研报的方法论）：
1. 事件定义和杀死条件在看结果之前固定；
2. 检验统计量用 Newey-West（HAC）标准误，默认 5 阶滞后；
3. 每个结论必须过安慰剂对照；
4. 描述性数字再好看，也要服从预设的检验。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm


def log_returns(prices: pd.Series) -> pd.Series:
    """对数收益：log(P_t / P_{t-1})，去掉首个 NaN。"""
    return np.log(prices / prices.shift(1)).dropna()


def hac_regression(y, dummy, controls=None, lags=5):
    """y 对事件哑变量（+可选控制变量）做 OLS，Newey-West HAC 标准误。

    y、dummy、controls 按索引对齐后去 NaN。
    返回 dict：coef（哑变量系数）、t、p、n，以及完整模型对象。
    """
    df = pd.DataFrame({"y": y, "dummy": dummy})
    if controls:
        for name, s in controls.items():
            df[name] = s
    df = df.dropna()
    cols = ["dummy"] + (list(controls.keys()) if controls else [])
    X = sm.add_constant(df[cols])
    model = sm.OLS(df["y"], X).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return {
        "coef": float(model.params["dummy"]),
        "t": float(model.tvalues["dummy"]),
        "p": float(model.pvalues["dummy"]),
        "n": int(len(df)),
        "model": model,
    }


def event_stats(y, dummy):
    """事件日 vs 非事件日的描述统计：均值、中位数、正收益占比、样本量。"""
    df = pd.DataFrame({"y": y, "dummy": dummy}).dropna()
    ev = df.loc[df["dummy"] == 1, "y"]
    nm = df.loc[df["dummy"] == 0, "y"]

    def _s(s):
        return {
            "mean": float(s.mean()),
            "median": float(s.median()),
            "positive_share": float((s > 0).mean()),
            "n": int(len(s)),
        }

    return {"event": _s(ev), "normal": _s(nm)}


def placebo_shift(dummy, shift=1):
    """把事件哑变量沿交易日平移 shift 天，作为安慰剂事件。"""
    return dummy.shift(shift).fillna(0)


def permutation_placebo(y, dummy, controls=None, n_perm=500, seed=42):
    """打乱哑变量做置换安慰剂：返回经验 p 值和 5%/95% 分位数。

    对应第一篇（波动率目标）的 placebo 做法：真实 t 是否比打乱后的 t 更极端。
    注意：本实现用 numpy default_rng，与原文 np.random.seed(42) 的随机流不同，
    经验 p 值在蒙特卡洛误差范围内小幅出入属正常。
    """
    real = hac_regression(y, dummy, controls=controls)
    df = pd.DataFrame({"y": y, "dummy": dummy})
    if controls:
        for name, s in controls.items():
            df[name] = s
    df = df.dropna()
    cols = ["dummy"] + (list(controls.keys()) if controls else [])
    rng = np.random.default_rng(seed)
    ts = []
    for _ in range(n_perm):
        d = df.copy()
        d["dummy"] = rng.permutation(d["dummy"].values)
        X = sm.add_constant(d[cols])
        m = sm.OLS(d["y"], X).fit()
        ts.append(float(m.tvalues["dummy"]))
    ts = np.array(ts)
    return {
        "real_t": real["t"],
        "p_empirical": float((np.abs(ts) >= abs(real["t"])).mean()),
        "p5": float(np.percentile(ts, 5)),
        "p95": float(np.percentile(ts, 95)),
    }


def tstat_1sample(x):
    """单样本 t：mean / (std / sqrt(n))。"""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    return float(x.mean() / x.std(ddof=1) * np.sqrt(len(x)))


def kill_check(conditions):
    """裁决预注册杀死条件。

    conditions: list of dict，每项 {"name": str, "killed": bool, "detail": str}。
    任一条件触发即杀死；全部未触发才算活下来。
    返回 {"verdict": "killed" | "survived", "triggered": [...], "summary": str}。
    """
    triggered = [c for c in conditions if c["killed"]]
    verdict = "killed" if triggered else "survived"
    lines = [
        f"{c['name']}: {'触发' if c['killed'] else '未触发'} — {c['detail']}"
        for c in conditions
    ]
    summary = (
        f"裁决：{'假设被杀死' if triggered else '假设活下来'}"
        f"（{len(triggered)}/{len(conditions)} 条触发）\n" + "\n".join(lines)
    )
    return {
        "verdict": verdict,
        "triggered": [c["name"] for c in triggered],
        "summary": summary,
    }
