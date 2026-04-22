"""
28个 CTA 因子挖掘脚本（基于英文文献 Moskowitz/Hurst/Baltas）

因子类别：
  TSMOM  — 时序动量 (1M/3M/6M/12M/连续)
  MA交叉  — 均线交叉 (10/40, 50/200, MACD-V)
  突破    — Donchian 通道 (20d/55d), ATR 通道
  波动率  — 波动扩张比, ATR比率, 低波动因子
  均值回归 — RSI, Bollinger %B, Z-Score
  量能    — OBV 动量, VWAP 偏离, 成交量爆量
  趋势质量 — R², 滚动 Sharpe, 回归斜率, 趋势综合

训练/验证/测试: 2018-2023 / 2024 / 2025-今

前提: 已执行 download_update_2025.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = ["Heiti TC", "STHeiti", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False
import warnings
warnings.filterwarnings("ignore")

import sys
sys.path.insert(0, "/Users/wiger-2/Documents/vnpy")

from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import matplotlib.gridspec as gridspec
from scipy.stats import spearmanr

from vnpy.trader.constant import Interval
from vnpy.alpha import AlphaLab, AlphaDataset, Segment
from vnpy.alpha.dataset import (
    process_drop_na,
    process_cs_norm,
    process_fill_na,
)


# ── 配置 ──────────────────────────────────────────────────────────────────────
LAB_PATH     = "/Users/wiger-2/Documents/vnpy/lab/csi300"
INDEX_SYMBOL = "000300.SSE"
DATASET_NAME = "csi300_cta28"
START        = "2018-01-01"
END          = "2026-04-22"
EXTENDED     = 260   # 252日动量需要260天预热

TRAIN_PERIOD = ("2018-01-01", "2023-12-31")   # 6年训练
VALID_PERIOD = ("2024-01-01", "2024-12-31")   # 1年验证
TEST_PERIOD  = ("2025-01-01", "2026-04-22")   # 样本外测试

OUTPUT_DIR   = Path(LAB_PATH) / "output"
PLOT_FILE    = OUTPUT_DIR / "cta_factor_ic.png"


# ── 28个 CTA 因子定义 ─────────────────────────────────────────────────────────
# 参考文献:
#   Moskowitz, Ooi & Pedersen (JFE 2012) — TSMOM
#   Hurst, Ooi & Pedersen (AQR) — Two Centuries of Trend Following
#   Baltas & Kosowski (SSRN 2140091) — 连续 TSMOM
#   Spiroglou (NAAIM 2022) — MACD-V
#   Turtle Trading (Richard Dennis 1983) — Donchian

FACTOR_DEFS: list[tuple[str, str, str]] = [
    # ── TSMOM 时序动量 ─────────────────────────────────────────────────────
    ("tsmom_1m",  "close / (ts_delay(close, 21) + 1e-12) - 1",
                  "1月时序动量 [Moskowitz2012]"),
    ("tsmom_3m",  "close / (ts_delay(close, 63) + 1e-12) - 1",
                  "3月时序动量 [Hurst AQR]"),
    ("tsmom_6m",  "close / (ts_delay(close, 126) + 1e-12) - 1",
                  "6月时序动量 [Hurst AQR]"),
    ("tsmom_12m", "ts_delay(close, 21) / (ts_delay(close, 252) + 1e-12) - 1",
                  "12月动量(跳过1月) [Moskowitz2012 核心信号]"),
    ("tsmom_vol", "(ts_delay(close, 21) / (ts_delay(close, 252) + 1e-12) - 1) / (ts_std(close, 252) / close + 1e-12)",
                  "连续TSMOM(波动率调整) [Baltas&Kosowski]"),
    # ── MA 均线交叉 ────────────────────────────────────────────────────────
    ("ma_10_40",  "(ts_mean(close, 10) - ts_mean(close, 40)) / close",
                  "10/40日MA交叉"),
    ("ma_50_200", "(ts_mean(close, 50) - ts_mean(close, 200)) / close",
                  "50/200日MA交叉(黄金/死叉)"),
    ("macd_v",    "(ts_mean(close, 12) - ts_mean(close, 26)) / (ta_atr(high, low, close, 26) + 1e-12)",
                  "MACD-V(ATR标准化) [Spiroglou NAAIM2022]"),
    # ── Donchian 通道突破 ──────────────────────────────────────────────────
    ("donch_20",  "(close - ts_min(low, 20)) / (ts_max(high, 20) - ts_min(low, 20) + 1e-12)",
                  "20日Donchian通道位置 [Turtle System1]"),
    ("donch_55",  "(close - ts_min(low, 55)) / (ts_max(high, 55) - ts_min(low, 55) + 1e-12)",
                  "55日Donchian通道位置 [Turtle System2]"),
    ("atr_chan",  "(close - ts_mean(close, 20)) / (ta_atr(high, low, close, 20) * 2 + 1e-12)",
                  "ATR通道相对位置(2σ)"),
    # ── 波动率信号 ─────────────────────────────────────────────────────────
    ("rvol_exp",  "ts_std(close, 20) / (ts_std(close, 60) + 1e-12)",
                  "波动率扩张比(短/长) [TSMOM仓位调节]"),
    ("atr_exp",   "ta_atr(high, low, close, 7) / (ta_atr(high, low, close, 30) + 1e-12)",
                  "ATR扩张比(短/长) [趋势加速信号]"),
    ("low_vol",   "ts_std(close, 20) / close * -1",
                  "低波动因子(负波动率)"),
    ("tr_ratio",  "(high - low) / (ta_atr(high, low, close, 14) + 1e-12) * -1",
                  "真实波幅/ATR(负值=安静日=均值回归)"),
    # ── 均值回归信号 ───────────────────────────────────────────────────────
    ("rsi_mr",    "ta_rsi(close, 14) * -1",
                  "RSI均值回归(低RSI=超卖=买信号) [Wilder]"),
    ("bb_mr",     "(close - ts_mean(close, 20)) / (ts_std(close, 20) * 2 + 1e-12) * -1",
                  "Bollinger %B均值回归"),
    ("zscore_mr", "(close - ts_mean(close, 40)) / (ts_std(close, 40) + 1e-12) * -1",
                  "40日Z-Score均值回归"),
    # ── 量能信号 ───────────────────────────────────────────────────────────
    ("obv_mom",   "ts_sum(volume * (close - ts_delay(close, 1)), 20) / (ts_mean(volume * close, 20) + 1e-12)",
                  "OBV动量(带符号成交额) [Goyenko2025]"),
    ("vwap_dev",  "(close - ts_mean(vwap, 20)) / (ts_std(vwap, 20) + 1e-12)",
                  "VWAP偏离度(短期)"),
    ("vol_burst", "ts_mean(volume, 5) / (ts_mean(volume, 60) + 1e-12)",
                  "5日vs60日成交量爆量比"),
    ("vol_sign",  "ts_sum(volume, 5) / (ts_mean(volume, 60) + 1e-12) * (close / (ts_delay(close, 5) + 1e-12) - 1)",
                  "有方向的爆量信号"),
    # ── 趋势质量 ───────────────────────────────────────────────────────────
    ("r2_60",     "ts_rsquare(close, 60)",
                  "60日趋势R² [趋势纯度]"),
    ("sharpe_60", "ts_mean(close / (ts_delay(close, 1) + 1e-12) - 1, 60) / (ts_std(close / (ts_delay(close, 1) + 1e-12) - 1, 60) + 1e-12)",
                  "60日滚动Sharpe [Research Affiliates]"),
    ("slope_20",  "ts_slope(close, 20) / close",
                  "20日回归斜率(短趋势)"),
    ("slope_60",  "ts_slope(close, 60) / close",
                  "60日回归斜率(中趋势)"),
    ("trend_ql",  "ts_slope(close, 60) / close * ts_rsquare(close, 60)",
                  "趋势质量综合(斜率×R²)"),
    ("ma200_dev", "close / (ts_mean(close, 200) + 1e-12) - 1",
                  "价格vs200日MA偏离(长期趋势过滤)"),
]

FACTOR_NAMES = [f[0] for f in FACTOR_DEFS]

FACTOR_CATS = {
    "TSMOM":   ["tsmom_1m","tsmom_3m","tsmom_6m","tsmom_12m","tsmom_vol"],
    "MA交叉":  ["ma_10_40","ma_50_200","macd_v"],
    "突破":    ["donch_20","donch_55","atr_chan"],
    "波动率":  ["rvol_exp","atr_exp","low_vol","tr_ratio"],
    "均值回归":["rsi_mr","bb_mr","zscore_mr"],
    "量能":    ["obv_mom","vwap_dev","vol_burst","vol_sign"],
    "趋势质量":["r2_60","sharpe_60","slope_20","slope_60","trend_ql","ma200_dev"],
}
CAT_COLORS = {
    "TSMOM":"#1565C0","MA交叉":"#00838F","突破":"#E65100",
    "波动率":"#6A1B9A","均值回归":"#C62828","量能":"#2E7D32","趋势质量":"#795548",
}


# ── 数据集类 ──────────────────────────────────────────────────────────────────

class CTA28(AlphaDataset):
    """28个 CTA 因子数据集"""

    def __init__(self, df, train_period, valid_period, test_period):
        super().__init__(df=df, train_period=train_period,
                         valid_period=valid_period, test_period=test_period)
        for name, expr, _ in FACTOR_DEFS:
            self.add_feature(name, expr)
        # 标签：3日远期收益（T+1跳过，T+1→T+3）
        self.set_label("ts_delay(close, -3) / ts_delay(close, -1) - 1")


# ── IC 工具 ───────────────────────────────────────────────────────────────────

def compute_ic_series(raw_df: pl.DataFrame, factor: str) -> pd.Series:
    if factor not in raw_df.columns or "label" not in raw_df.columns:
        return pd.Series(dtype=float)
    sub = (raw_df.select(["datetime","vt_symbol",factor,"label"])
           .drop_nulls().to_pandas()
           .dropna(subset=[factor,"label"]))
    result = {}
    for dt, grp in sub.groupby("datetime"):
        if len(grp) < 10:
            continue
        ic, _ = spearmanr(grp[factor], grp["label"])
        if not np.isnan(ic):
            result[dt] = ic
    return pd.Series(result).sort_index()


def ic_stats(s: pd.Series) -> dict:
    if s.empty:
        return dict(ic=0, icir=0, win=0, n=0)
    m, sd = s.mean(), s.std()
    return dict(ic=m, icir=m / sd if sd > 1e-8 else 0, win=(s > 0).mean(), n=len(s))


# ── 可视化 ─────────────────────────────────────────────────────────────────────

def plot_ic(stats_df: pd.DataFrame, ic_dict: dict) -> None:
    stats_df = stats_df.sort_values("icir", ascending=False)
    name_to_cat = {n: cat for cat, names in FACTOR_CATS.items() for n in names}
    colors = [CAT_COLORS.get(name_to_cat.get(n,""), "#607D8B")
              for n in stats_df["factor"]]

    fig = plt.figure(figsize=(24, 16), facecolor="#F8F9FA")
    fig.suptitle("沪深300  CTA 28因子  IC 综合分析（训练+验证 2018-2024）",
                 fontsize=16, fontweight="bold", y=0.985, color="#212121")
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.50, wspace=0.30,
                           top=0.94, bottom=0.06, left=0.06, right=0.97)

    DARK="#212121"; GRID="#E0E0E0"; GREEN="#2E7D32"; RED="#C62828"

    def sty(ax, title="", ylabel=""):
        ax.set_facecolor("#FAFAFA")
        ax.spines[["top","right"]].set_visible(False)
        ax.spines[["left","bottom"]].set_color(GRID)
        ax.tick_params(colors=DARK, labelsize=8.5)
        ax.grid(axis="y", color=GRID, lw=0.5, zorder=0)
        if title:  ax.set_title(title,  fontsize=11, color=DARK, pad=5, fontweight="bold")
        if ylabel: ax.set_ylabel(ylabel, fontsize=9,  color=DARK)

    # Panel 1: ICIR 柱状图
    ax1 = fig.add_subplot(gs[0, :])
    x = np.arange(len(stats_df))
    ax1.bar(x, stats_df["icir"], color=colors, width=0.7, zorder=2)
    ax1.axhline(0,     color=DARK,  lw=0.8)
    ax1.axhline( 0.5,  color=GREEN, lw=1.2, ls="--", alpha=0.7)
    ax1.axhline(-0.5,  color=RED,   lw=1.2, ls="--", alpha=0.7)
    ax1.set_xticks(x)
    ax1.set_xticklabels(stats_df["factor"], rotation=40, ha="right", fontsize=9)
    for xi, (_, row) in zip(x, stats_df.iterrows()):
        h = row["icir"]
        ax1.text(xi, h+(0.015 if h>=0 else -0.025),
                 f"{row['ic']:.3f}", ha="center",
                 va="bottom" if h>=0 else "top", fontsize=6, color="#424242")
    leg = [plt.Rectangle((0,0),1,1,color=c,label=cat) for cat,c in CAT_COLORS.items()]
    ax1.legend(handles=leg, loc="upper right", fontsize=9, ncol=4, framealpha=0.85)
    sty(ax1, "CTA 28因子 ICIR（2018-2024）", "ICIR")

    # Panel 2: IC均值 vs 胜率
    ax2 = fig.add_subplot(gs[1, 0])
    sc_c = [CAT_COLORS.get(name_to_cat.get(n,""),"#607D8B") for n in stats_df["factor"]]
    ax2.scatter(stats_df["ic"], stats_df["win"], c=sc_c, s=65, zorder=3, alpha=0.9)
    for _, row in stats_df.iterrows():
        ax2.annotate(row["factor"], (row["ic"], row["win"]),
                     fontsize=6.5, xytext=(2,2), textcoords="offset points", color="#555")
    ax2.axvline(0, color=DARK, lw=0.7)
    ax2.axhline(0.5, color="#9E9E9E", lw=0.7, ls="--", alpha=0.6)
    sty(ax2, "IC均值 vs 胜率", "IC胜率")
    ax2.set_xlabel("IC均值", fontsize=9, color=DARK)

    # Panel 3: Top5 IC分布
    ax3 = fig.add_subplot(gs[1, 1])
    for fn in stats_df.head(5)["factor"].tolist():
        s = ic_dict.get(fn, pd.Series())
        if not s.empty:
            ax3.hist(s, bins=35, alpha=0.5, label=fn, density=True)
    ax3.axvline(0, color=DARK, lw=1)
    ax3.axvline( 0.03, color=GREEN, lw=0.8, ls="--", alpha=0.5)
    ax3.axvline(-0.03, color=RED,   lw=0.8, ls="--", alpha=0.5)
    sty(ax3, "Top5因子 IC分布", "频率密度")
    ax3.set_xlabel("IC值", fontsize=9, color=DARK)
    ax3.legend(fontsize=8, loc="upper left")

    # Panel 4: Top7 因子 60日滚动IC
    ax4 = fig.add_subplot(gs[2, :])
    palette = ["#1565C0","#E65100","#2E7D32","#6A1B9A","#C62828","#00838F","#795548"]
    top7 = stats_df.head(7)["factor"].tolist()
    for fn, col in zip(top7, palette):
        s = ic_dict.get(fn, pd.Series())
        if s.empty:
            continue
        roll = s.rolling(60, min_periods=20).mean()
        ax4.plot(roll.index, roll.values, label=fn, color=col, lw=1.8)
    ax4.axhline(0, color=DARK, lw=0.8)
    ax4.axhspan(-0.03, 0.03, alpha=0.07, color="gray")
    sty(ax4, "Top7因子  60日滚动IC", "滚动IC")
    ax4.set_xlabel("日期", fontsize=9, color=DARK)
    ax4.legend(fontsize=8.5, loc="upper right", ncol=4, framealpha=0.85)

    plt.savefig(str(PLOT_FILE), dpi=160, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  IC图表 → {PLOT_FILE}")


# ── 主流程 ─────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    lab = AlphaLab(LAB_PATH)

    # ── 1. 加载行情 ─────────────────────────────────────────────────────────
    print("=" * 60)
    print("[1/4] 加载行情数据")
    print("=" * 60)
    syms = lab.load_component_symbols(INDEX_SYMBOL, START, END)
    print(f"成分股: {len(syms)} 只")

    bar_df = lab.load_bar_df(
        vt_symbols=syms, interval=Interval.DAILY,
        start=START, end=END, extended_days=EXTENDED,
    )
    if bar_df is None or bar_df.is_empty():
        print("错误: 行情数据为空")
        return
    print(f"数据规模: {bar_df.shape[0]:,} 行 × {bar_df.shape[1]} 列")
    print(f"日期范围: {bar_df['datetime'].min()} ~ {bar_df['datetime'].max()}")

    # ── 2. 构建数据集 ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("[2/4] 构建 CTA28 数据集")
    print("=" * 60)
    print(f"训练: {TRAIN_PERIOD[0]} ~ {TRAIN_PERIOD[1]}")
    print(f"验证: {VALID_PERIOD[0]} ~ {VALID_PERIOD[1]}")
    print(f"测试: {TEST_PERIOD[0]}  ~ {TEST_PERIOD[1]}")

    dataset = CTA28(
        df=bar_df,
        train_period=TRAIN_PERIOD,
        valid_period=VALID_PERIOD,
        test_period=TEST_PERIOD,
    )
    dataset.add_processor("learn", partial(process_drop_na,  names=["label"]))
    dataset.add_processor("learn", partial(process_cs_norm,  names=["label"], method="zscore"))
    dataset.add_processor("infer", partial(process_fill_na,  fill_value=0))

    for i, (name, _, desc) in enumerate(FACTOR_DEFS, 1):
        print(f"  {i:2d}. {name:<14} — {desc}")

    # ── 3. 并行计算 ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("[3/4] 并行计算 28 个因子（max_workers=4）")
    print("=" * 60)

    filters = lab.load_component_filters(INDEX_SYMBOL, START, END)
    dataset.prepare_data(filters, max_workers=4)
    dataset.process_data()
    lab.save_dataset(DATASET_NAME, dataset)
    print(f"\n数据集已保存: {DATASET_NAME}")

    for seg_name, seg in [("train", Segment.TRAIN), ("valid", Segment.VALID), ("test", Segment.TEST)]:
        try:
            df = dataset.fetch_infer(seg)
            print(f"  {seg_name:5s}: {df.shape[0]:>8,} 条")
        except Exception:
            pass

    # ── 4. 批量 IC 分析（训练+验证集）────────────────────────────────────────
    print("\n" + "=" * 60)
    print("[4/4] 批量 IC 分析")
    print("=" * 60)

    from vnpy.alpha.dataset.template import query_by_time
    raw_tv = query_by_time(dataset.raw_df, TRAIN_PERIOD[0], VALID_PERIOD[1])

    ic_dict, rows = {}, []
    print(f"\n{'因子':<14} {'IC均值':>8} {'ICIR':>8} {'胜率':>7} {'有效天':>7}  文献类别")
    print("-" * 80)
    for name, _, desc in FACTOR_DEFS:
        s  = compute_ic_series(raw_tv, name)
        st = ic_stats(s)
        ic_dict[name] = s
        rows.append({"factor": name, **st, "desc": desc})
        flag = "★" if abs(st["icir"]) > 0.5 else ("·" if abs(st["ic"]) > 0.03 else " ")
        print(f"{name:<14} {st['ic']:>8.4f} {st['icir']:>8.4f} "
              f"{st['win']:>7.2%} {st['n']:>7d}  {flag}")

    stats_df = pd.DataFrame(rows)
    csv_path = OUTPUT_DIR / "cta_ic_stats.csv"
    stats_df.to_csv(str(csv_path), index=False, encoding="utf-8-sig")
    print(f"\nIC统计表 → {csv_path}")

    print("\n── Top10因子（|ICIR|排序）──")
    top10 = stats_df.reindex(stats_df["icir"].abs().sort_values(ascending=False).index).head(10)
    for _, r in top10.iterrows():
        print(f"  {r['factor']:<14} ICIR={r['icir']:+.4f}  IC={r['ic']:+.4f}  胜率={r['win']:.2%}")

    plot_ic(stats_df, ic_dict)
    print("\n下一步: python cta_backtest.py")


if __name__ == "__main__":
    main()
