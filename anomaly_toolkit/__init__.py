"""市场异象复现工具箱：用预注册杀死条件 + HAC 检验 + 安慰剂对照，复现教科书异象。"""

from .core import (
    event_stats,
    hac_regression,
    kill_check,
    log_returns,
    permutation_placebo,
    placebo_shift,
    tstat_1sample,
)
from .data import (
    download_akshare_index_close,
    download_close,
    download_closes,
    download_ohlc,
)

__all__ = [
    "log_returns",
    "hac_regression",
    "event_stats",
    "placebo_shift",
    "permutation_placebo",
    "tstat_1sample",
    "kill_check",
    "download_close",
    "download_closes",
    "download_ohlc",
    "download_akshare_index_close",
]
