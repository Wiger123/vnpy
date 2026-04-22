"""
30个基本面因子挖掘脚本

流程:
  1. 加载中证500行情 + 基本面估值数据，合并为统一 DataFrame
  2. 注册30个因子（价值/规模/动量/波动/流动性/质量）
  3. 并行计算因子值
  4. 批量输出每个因子的 IC / ICIR / 胜率
  5. 生成因子IC总览图（按ICIR排序）

前提: 已执行 download_csi500.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = ["Heiti TC", "STHeiti", "SimHei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False
import warnings
warnings.filterwarnings("ignore")

from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
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
DATASET_NAME = "csi300_fund30"
START        = "2018-01-01"
END          = "2024-12-31"
EXTENDED     = 130   # 保证120日滚动窗口完整预热

TRAIN_PERIOD = ("2018-01-01", "2021-12-31")
VALID_PERIOD = ("2022-01-01", "2022-12-31")
TEST_PERIOD  = ("2023-01-01", "2024-12-31")

FUND_PATH    = Path(LAB_PATH) / "fundamental" / "fundamental.parquet"
OUTPUT_DIR   = Path(LAB_PATH) / "output"

PLOT_FILE    = OUTPUT_DIR / "factor_ic_summary.png"


# ── 30个基本面因子定义 ────────────────────────────────────────────────────────
# 注: df已提前合并衍生列: ep_ttm=1/pe_ttm, bp=1/pb, sp=1/ps_ttm, log_mv=log(total_mv+1)

FACTOR_DEFS: list[tuple[str, str, str]] = [
    # name,               expression,                                                    description
    # ── 价值类 (Value) ─────────────────────────────────────────────────────
    # 注: 引用预计算列 pe_inv/pb_inv/ps_inv/pcf_inv（避免与因子名同名导致原地替换）
    ("ep_ttm",    "pe_inv",                                                        "盈利收益率 1/PE"),
    ("bp",        "pb_inv",                                                        "账面市值比 1/PB"),
    ("sp",        "ps_inv",                                                        "销售收益率 1/PS"),
    ("pcf_yield", "pcf_inv",                                                       "现金流收益率 1/PCF"),
    ("ep_chg20",  "pe_inv - ts_delay(pe_inv, 20)",                                "EP 20日改善"),
    ("bp_chg60",  "pb_inv - ts_delay(pb_inv, 60)",                                "BP 60日改善（估值收缩）"),
    ("val_comp",  "pe_inv + pb_inv + ps_inv",                                     "综合价值评分"),
    ("ep_ma60",   "ts_mean(pe_inv, 60)",                                           "60日平均EP（稳健盈利）"),
    # ── 规模类 (Size) ───────────────────────────────────────────────────────
    ("pb_vs_ep",  "pb_inv - pe_inv",                                              "BP-EP利差（ROE代理）"),
    ("ps_chg60",  "ts_delay(ps_inv, 60) - ps_inv",                               "SP 60日变化（PS收缩）"),
    ("pb_rank",   "ts_rank(pb_inv, 60) * -1",                                     "PB时序排名取负（低PB偏好）"),
    # ── 动量类 (Momentum) ───────────────────────────────────────────────────
    ("mom_20",    "close / (ts_delay(close, 20) + 1e-12) - 1",                   "20日价格动量"),
    ("mom_60",    "close / (ts_delay(close, 60) + 1e-12) - 1",                   "60日价格动量"),
    ("mom_120",   "close / (ts_delay(close, 120) + 1e-12) - 1",                  "120日价格动量"),
    ("rev_5",     "ts_delay(close, 5) / (close + 1e-12) - 1",                    "5日短期反转"),
    ("risk_mom",  "(close / (ts_delay(close, 20) + 1e-12) - 1) / (ts_std(close, 20) / close + 1e-12)",
                                                                                   "风险调整动量"),
    ("earn_mom",  "pe_inv - ts_delay(pe_inv, 60)",                               "60日盈利动量"),
    ("trend",     "ts_slope(close, 20) / close",                                  "价格趋势斜率"),
    # ── 波动类 (Volatility) ─────────────────────────────────────────────────
    ("vol_20",    "ts_std(close, 20) / close",                                    "20日收益率波动"),
    ("vol_60",    "ts_std(close, 60) / close",                                    "60日收益率波动"),
    ("vol_ratio", "ts_std(close, 5) / (ts_std(close, 60) + 1e-12)",             "短/长期波动比"),
    ("down_vol",  "ts_std(low, 20) / close",                                     "下行波动率"),
    # ── 流动性类 (Liquidity) ─────────────────────────────────────────────────
    ("tover",     "turnover / (close + 1e-8)",                                   "标准化成交额（成交额/价格）"),
    ("amihud",    "ts_mean(ts_abs(close / (ts_delay(close, 1) + 1e-12) - 1) / (turnover + 1e-8), 20) * -1",
                                                                                   "Amihud流动性（取负）"),
    ("vol_stab",  "ts_mean(volume, 20) / (ts_std(volume, 20) + 1e-12)",          "成交量稳定性"),
    ("vp_corr",   "ts_corr(volume, close, 20)",                                   "量价相关系数"),
    # ── 质量代理类 (Quality) ─────────────────────────────────────────────────
    ("pe_stab",   "ts_std(pe_ttm, 60) * -1",                                     "PE稳定性（负波动）"),
    ("peak_pct",  "close / (ts_max(high, 120) + 1e-12)",                         "价格相对120日高点"),
    ("earn_cons", "ts_mean(pe_inv, 120) / (ts_std(pe_inv, 60) + 1e-12)",        "盈利一致性"),
    ("vwap_dev",  "(close - ts_mean(vwap, 20)) / (ts_std(vwap, 20) + 1e-12)",   "价格对VWAP偏离度"),
]

FACTOR_NAMES = [f[0] for f in FACTOR_DEFS]
FACTOR_CATS  = {
    "价值":  ["ep_ttm", "bp", "sp", "pcf_yield", "ep_chg20", "bp_chg60", "val_comp", "ep_ma60"],
    "规模":  ["pb_vs_ep", "ps_chg60", "pb_rank"],
    "动量":  ["mom_20", "mom_60", "mom_120", "rev_5", "risk_mom", "earn_mom", "trend"],
    "波动":  ["vol_20", "vol_60", "vol_ratio", "down_vol"],
    "流动性": ["tover", "amihud", "vol_stab", "vp_corr"],
    "质量":  ["pe_stab", "peak_pct", "earn_cons", "vwap_dev"],
}
CAT_COLORS = {
    "价值": "#2196F3", "规模": "#9C27B0", "动量": "#FF9800",
    "波动": "#F44336", "流动性": "#4CAF50", "质量": "#795548",
}


class Fundamental30(AlphaDataset):
    """30个基本面因子数据集"""

    def __init__(self, df: pl.DataFrame, train_period, valid_period, test_period):
        super().__init__(
            df=df,
            train_period=train_period,
            valid_period=valid_period,
            test_period=test_period,
        )
        for name, expr, _ in FACTOR_DEFS:
            self.add_feature(name, expr)

        # 标签: 未来3日收益（跳过T+1，使用T+1到T+3的收益）
        self.set_label("ts_delay(close, -3) / ts_delay(close, -1) - 1")


# ── IC 计算工具 ────────────────────────────────────────────────────────────────

def compute_ic_series(raw_df: pl.DataFrame, factor: str) -> pd.Series:
    """按日计算 rank IC（Spearman）序列"""
    if factor not in raw_df.columns or "label" not in raw_df.columns:
        return pd.Series(dtype=float)

    sub = raw_df.select(["datetime", "vt_symbol", factor, "label"]).drop_nulls()
    pdf = sub.to_pandas()
    pdf = pdf.dropna(subset=[factor, "label"])

    daily_ic: dict = {}
    for dt, grp in pdf.groupby("datetime"):
        if len(grp) < 10:
            continue
        ic, _ = spearmanr(grp[factor], grp["label"])
        if not np.isnan(ic):
            daily_ic[dt] = ic

    return pd.Series(daily_ic).sort_index()


def ic_stats(ic_series: pd.Series) -> dict:
    if ic_series.empty:
        return dict(ic=0, icir=0, win=0, n=0)
    ic_mean = ic_series.mean()
    ic_std  = ic_series.std()
    icir    = ic_mean / ic_std if ic_std > 1e-8 else 0
    win     = (ic_series > 0).mean()
    return dict(ic=ic_mean, icir=icir, win=win, n=len(ic_series))


# ── 可视化 ─────────────────────────────────────────────────────────────────────

def plot_ic_summary(stats_df: pd.DataFrame, ic_dict: dict[str, pd.Series]) -> None:
    """生成2面板IC总览图: 上图=ICIR排序柱状图，下图=Top10因子滚动IC"""
    stats_df = stats_df.sort_values("icir", ascending=False)

    # 为每个因子分配类别颜色
    name_to_cat = {}
    for cat, names in FACTOR_CATS.items():
        for n in names:
            name_to_cat[n] = cat
    colors = [CAT_COLORS.get(name_to_cat.get(n, ""), "#607D8B")
              for n in stats_df["factor"]]

    fig = plt.figure(figsize=(20, 14), facecolor="#F8F9FA")
    fig.suptitle("中证500 基本面30因子 IC 综合分析", fontsize=16, fontweight="bold",
                 y=0.98, color="#212121")

    gs = gridspec.GridSpec(3, 2, figure=fig,
                           hspace=0.45, wspace=0.35,
                           top=0.93, bottom=0.06, left=0.07, right=0.97)

    # ── Panel 1: ICIR 柱状图（全30个因子）─────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    x = np.arange(len(stats_df))
    bars = ax1.bar(x, stats_df["icir"], color=colors, width=0.7, zorder=2)
    ax1.axhline(0,   color="#424242", lw=0.8, ls="-")
    ax1.axhline( 0.5, color="#43A047", lw=1.0, ls="--", alpha=0.7, label="ICIR=0.5")
    ax1.axhline(-0.5, color="#E53935", lw=1.0, ls="--", alpha=0.7, label="ICIR=-0.5")
    ax1.set_xticks(x)
    ax1.set_xticklabels(stats_df["factor"], rotation=45, ha="right", fontsize=8.5)
    ax1.set_ylabel("ICIR", fontsize=10)
    ax1.set_title("30因子 ICIR（训练集+验证集）", fontsize=11, pad=6)
    ax1.grid(axis="y", alpha=0.3, zorder=0)
    ax1.set_facecolor("#FAFAFA")

    # 在柱上标注IC均值
    for bar, (_, row) in zip(bars, stats_df.iterrows()):
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 h + (0.02 if h >= 0 else -0.04),
                 f"{row['ic']:.3f}", ha="center", va="bottom" if h >= 0 else "top",
                 fontsize=6.5, color="#424242")

    # 图例（类别色）
    legend_handles = [plt.Rectangle((0, 0), 1, 1, color=c, label=cat)
                      for cat, c in CAT_COLORS.items()]
    ax1.legend(handles=legend_handles, loc="upper right", fontsize=8,
               ncol=3, framealpha=0.8)

    # ── Panel 2: IC均值 + 胜率 散点 ───────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    sc_colors = [CAT_COLORS.get(name_to_cat.get(n, ""), "#607D8B")
                 for n in stats_df["factor"]]
    ax2.scatter(stats_df["ic"], stats_df["win"], c=sc_colors, s=60, zorder=3, alpha=0.9)
    for _, row in stats_df.iterrows():
        ax2.annotate(row["factor"], (row["ic"], row["win"]),
                     fontsize=6, xytext=(3, 2), textcoords="offset points",
                     color="#555555")
    ax2.axvline(0, color="#424242", lw=0.8)
    ax2.axhline(0.5, color="#9E9E9E", lw=0.8, ls="--", alpha=0.6)
    ax2.set_xlabel("IC均值", fontsize=10)
    ax2.set_ylabel("IC胜率", fontsize=10)
    ax2.set_title("IC均值 vs 胜率分布", fontsize=11, pad=6)
    ax2.grid(alpha=0.3)
    ax2.set_facecolor("#FAFAFA")

    # ── Panel 3: IC分布直方图（Top5 + Bottom5）────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    top5    = stats_df.head(5)["factor"].tolist()
    bottom5 = stats_df.tail(5)["factor"].tolist()
    for fname in top5:
        s = ic_dict.get(fname, pd.Series())
        if not s.empty:
            ax3.hist(s, bins=30, alpha=0.5,
                     label=fname, density=True)
    ax3.axvline(0, color="#212121", lw=1)
    ax3.set_xlabel("IC值", fontsize=10)
    ax3.set_ylabel("频率密度", fontsize=10)
    ax3.set_title("Top5因子 IC分布", fontsize=11, pad=6)
    ax3.legend(fontsize=7, loc="upper left")
    ax3.grid(alpha=0.3)
    ax3.set_facecolor("#FAFAFA")

    # ── Panel 4: Top6因子滚动IC（60日）─────────────────────────────────────
    ax4 = fig.add_subplot(gs[2, :])
    top6 = stats_df.head(6)["factor"].tolist()
    for fname in top6:
        s = ic_dict.get(fname, pd.Series())
        if s.empty:
            continue
        cat = name_to_cat.get(fname, "")
        color = CAT_COLORS.get(cat, "#607D8B")
        rolling = s.rolling(60, min_periods=20).mean()
        ax4.plot(rolling.index, rolling.values, label=fname, color=color, lw=1.5)
    ax4.axhline(0, color="#424242", lw=0.8)
    ax4.axhspan(-0.03, 0.03, alpha=0.07, color="gray")
    ax4.set_xlabel("日期", fontsize=10)
    ax4.set_ylabel("滚动60日IC", fontsize=10)
    ax4.set_title("Top6因子 60日滚动IC", fontsize=11, pad=6)
    ax4.legend(fontsize=8, loc="upper right", ncol=3)
    ax4.grid(alpha=0.3)
    ax4.set_facecolor("#FAFAFA")

    plt.savefig(str(PLOT_FILE), dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  图表已保存: {PLOT_FILE}")


# ── 主流程 ─────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    lab = AlphaLab(LAB_PATH)

    # ── 步骤 1: 加载行情 ────────────────────────────────────────────────────
    print("=" * 60)
    print("[1/5] 加载中证500行情数据")
    print("=" * 60)

    component_symbols = lab.load_component_symbols(INDEX_SYMBOL, START, END)
    print(f"成分股数量: {len(component_symbols)} 只")

    bar_df: pl.DataFrame = lab.load_bar_df(
        vt_symbols    =component_symbols,
        interval      =Interval.DAILY,
        start         =START,
        end           =END,
        extended_days =EXTENDED,
    )
    if bar_df is None or bar_df.is_empty():
        print("错误: 行情数据为空，请先运行 download_csi500.py")
        return
    print(f"行情数据: {bar_df.shape[0]:,} 行 × {bar_df.shape[1]} 列")

    # ── 步骤 2: 合并基本面数据 ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("[2/5] 合并基本面估值数据")
    print("=" * 60)

    if not FUND_PATH.exists():
        print(f"错误: 找不到 {FUND_PATH}，请先运行 download_csi500.py")
        return

    fund_df = pl.read_parquet(str(FUND_PATH))
    # baostock 日期是 00:00:00，bar_df 是 15:00:00，需要加 15 小时对齐
    fund_df = fund_df.with_columns(
        (pl.col("date").cast(pl.Datetime("us"))
         + pl.duration(hours=15))
        .alias("datetime")
    ).drop("date")

    # 合并：左连接，保持 bar_df 所有行
    merged_df = bar_df.join(fund_df, on=["datetime", "vt_symbol"], how="left")

    # 填充基本面数据缺口（上一个有效值前向填充）
    for col in ["pe_ttm", "pb", "ps_ttm", "pcf_ttm"]:
        if col in merged_df.columns:
            merged_df = merged_df.with_columns(
                pl.col(col).forward_fill().over("vt_symbol")
            )

    # 预计算衍生列，避免在表达式中出现 1/x（DataProxy不支持右除）
    # 注: 用 when/then 处理 null 和 0，避免产生 1/1e-8 = 1e8 的异常值
    merged_df = merged_df.with_columns([
        # pe_inv = 1/PE（因子ep_ttm的底层列，避免与因子同名导致原地替换）
        pl.when(pl.col("pe_ttm") > 0)
          .then(1.0 / pl.col("pe_ttm"))
          .otherwise(None)
          .alias("pe_inv"),
        pl.when(pl.col("pb") > 0)
          .then(1.0 / pl.col("pb"))
          .otherwise(None)
          .alias("pb_inv"),
        pl.when(pl.col("ps_ttm") > 0)
          .then(1.0 / pl.col("ps_ttm"))
          .otherwise(None)
          .alias("ps_inv"),
        pl.when(pl.col("pcf_ttm") > 0)
          .then(1.0 / pl.col("pcf_ttm"))
          .otherwise(None)
          .alias("pcf_inv"),
        # pe_ttm 保留原始值（给 ts_std(pe_ttm,...) 等时序函数用）
        pl.col("pe_ttm").alias("pe_ttm"),
    ])

    print(f"合并后数据: {merged_df.shape[0]:,} 行 × {merged_df.shape[1]} 列")
    print(f"新增基本面列: ep_ttm, bp, sp, pcf_yield, pe_ttm")

    # ── 步骤 3: 构建因子数据集 ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("[3/5] 构建 Fundamental30 数据集（30个因子）")
    print("=" * 60)

    for i, (name, expr, desc) in enumerate(FACTOR_DEFS, 1):
        print(f"  {i:2d}. {name:<14} — {desc}")

    dataset = Fundamental30(
        df           =merged_df,
        train_period =TRAIN_PERIOD,
        valid_period =VALID_PERIOD,
        test_period  =TEST_PERIOD,
    )

    dataset.add_processor("learn", partial(process_drop_na,  names=["label"]))
    dataset.add_processor("learn", partial(process_cs_norm,  names=["label"], method="zscore"))
    dataset.add_processor("infer", partial(process_fill_na,  fill_value=0))

    # ── 步骤 4: 并行计算因子 ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("[4/5] 并行计算30个因子（max_workers=4）")
    print("=" * 60)

    filters = lab.load_component_filters(INDEX_SYMBOL, START, END)
    dataset.prepare_data(filters, max_workers=4)
    dataset.process_data()

    lab.save_dataset(DATASET_NAME, dataset)
    print(f"\n因子计算完成 ✓  数据集: {DATASET_NAME}")

    for seg_name, seg in [("train", Segment.TRAIN), ("valid", Segment.VALID), ("test", Segment.TEST)]:
        try:
            df = dataset.fetch_infer(seg)
            print(f"  {seg_name:5s}: {df.shape[0]:>8,} 条样本")
        except Exception:
            pass

    # ── 步骤 5: 批量 IC 分析 ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("[5/5] 批量计算因子 IC / ICIR / 胜率（训练集+验证集）")
    print("=" * 60)

    # 使用训练+验证集的 raw_df 做IC分析（避免未来数据泄露）
    train_start, _ = TRAIN_PERIOD
    _, valid_end   = VALID_PERIOD
    from vnpy.alpha.dataset.template import query_by_time
    raw_tv = query_by_time(dataset.raw_df, train_start, valid_end)

    ic_dict: dict[str, pd.Series] = {}
    rows: list[dict] = []

    print(f"\n{'因子':<14} {'IC均值':>8} {'ICIR':>8} {'胜率':>7} {'有效天':>7}  描述")
    print("-" * 75)

    for name, _, desc in FACTOR_DEFS:
        s = compute_ic_series(raw_tv, name)
        ic_dict[name] = s
        st = ic_stats(s)
        rows.append({"factor": name, **st, "desc": desc})

        flag = ""
        if abs(st["icir"]) > 0.5:
            flag = "★"
        elif abs(st["ic"]) > 0.03:
            flag = "·"

        print(f"{name:<14} {st['ic']:>8.4f} {st['icir']:>8.4f} {st['win']:>7.2%} "
              f"{st['n']:>7d}  {flag} {desc}")

    stats_df = pd.DataFrame(rows)

    # 保存统计表
    csv_path = OUTPUT_DIR / "factor_ic_stats.csv"
    stats_df.to_csv(str(csv_path), index=False, encoding="utf-8-sig")
    print(f"\nIC统计表已保存: {csv_path}")

    # 生成可视化图表
    print("\n生成因子IC总览图...")
    plot_ic_summary(stats_df, ic_dict)

    # 打印Top10
    top10 = stats_df.reindex(stats_df["icir"].abs().sort_values(ascending=False).index).head(10)
    print("\n── Top10因子（按|ICIR|排序）──")
    for _, row in top10.iterrows():
        print(f"  {row['factor']:<14} ICIR={row['icir']:+.4f}  IC={row['ic']:+.4f}  胜率={row['win']:.2%}")

    print("\n下一步: python fundamental_backtest.py")


if __name__ == "__main__":
    main()
