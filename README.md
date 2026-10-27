# 异象复现工具箱 anomaly-toolkit

用**预注册杀死条件 + HAC 检验 + 安慰剂对照**，复现教科书级的市场异象。
描述性数字再好看，也要服从预设的检验——这正是本工具箱与“回测作坊”的区别。

## 已复现的 5 个案例

| # | 案例 | 市场 | 样本 | 结论 | 研报 |
|---|------|------|------|------|------|
| 01 | 波动率目标的强制去杠杆 | 沪深 300 | 2002—2026 | 活下来，但右尾驱动，不是稳定信号 | [posts/22](https://haimao.app/posts/22) |
| 02 | 月末月初效应 | SPY | 2000—2026 | 被杀死：全样本 t=0.94，2013 后归零 | [posts/27](https://haimao.app/posts/27) |
| 03 | 恒生指数“周一魔咒” | HSI | 2000—2026 | 被杀死：周一平均跌 0.036%，检验不显著 | [专栏](https://haimao.app/users/haimao2016) |
| 04 | 美股暴跌日，黄金避险吗 | SPY vs GLD | 2004—2026 | 被杀死：暴跌日黄金平均还跌 | [posts/45](https://haimao.app/posts/45) |
| 05 | VIX 飙升后 SPY 次日反弹吗 | SPY vs VIX | 1993—2026 | 活下来，但 p=0.034 在边缘，且反弹在盘中 | [posts/51](https://haimao.app/posts/51) |

5 个假设，2 个活下来，3 个被杀死——框架没有被调成只产出阳性结果。

## 快速开始

```bash
pip install -r requirements.txt
# 或
pip install -e .

python examples/01_vol_target_deleveraging.py
python examples/02_turn_of_month_spy.py
python examples/03_hsi_monday.py
python examples/04_gold_hedge_crash.py
python examples/05_vix_spike_rebound.py
```

每个案例都是独立脚本：下载数据（yfinance / akshare）、按事前定义构造事件、
跑检验、输出裁决。跑通不需要你准备数据。

测试（只用合成数据，不联网）：

```bash
python tests/test_core.py
```

## 方法论：为什么是“杀死条件”而不是“显著性”

1. **事前定义**：事件窗口、检验、杀死条件在看结果之前固定，写进脚本 docstring。
2. **HAC 标准误**：收益序列自相关是常态，普通 OLS 的 t 值会虚高；统一用 Newey-West（5 阶滞后）。
3. **安慰剂对照**：每个阳性结论必须过安慰剂——平移事件窗口、或打乱哑变量，看效应是否还成立。
4. **杀死条件裁决**：任一条件触发即杀死；全部未触发才算活下来。阴性结果和阳性结果同等分量。

核心函数都在 `anomaly_toolkit/core.py`：`hac_regression`、`event_stats`、
`placebo_shift`、`permutation_placebo`、`kill_check`。要复现一个新异象，
按 `examples/` 的格式写一个脚本即可。

## 研报专栏

本工具箱配套的研报连载在宽客汇 QuantHub：

- 专栏主页：https://haimao.app/users/haimao2016
- 邀请链接：（发布前填入你的 QuantHub 邀请链接）

每篇研报都按本工具箱的方法论写：机制、硬数字、局限、质询问题。
欢迎订阅专栏，也欢迎用邀请链接注册 QuantHub、自己发一篇复现。

## 免责

本项目用于研究与教育，不构成投资建议。历史复现结果不代表未来收益；
所有案例均未扣足滑点与冲击成本（除 01 注明外），实盘前请自行做成本后检验。

## License

MIT
