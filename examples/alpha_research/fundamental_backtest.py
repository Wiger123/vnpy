"""
基本面30因子模型训练 + 回测 + 可视化

流程:
  1. 加载 csi500_fund30 数据集
  2. 训练 LightGBM 模型
  3. 测试集生成预测信号，评估IC
  4. EquityDemoStrategy 回测（top_k=50，中证500持仓风格）
  5. 输出综合业绩图（对标参考图排版）

前提: 已执行 fundamental_factors.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = ["Heiti TC", "STHeiti", "SimHei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from datetime import datetime
import sys
sys.path.insert(0, "/Users/wiger-2/Documents/vnpy/examples/alpha_research")
from fundamental_factors import Fundamental30  # noqa: F401  让 pickle 能找到类

import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import spearmanr

from vnpy.trader.constant import Interval
from vnpy.alpha import AlphaLab, Segment, AlphaDataset, AlphaModel
from vnpy.alpha.model.models.lgb_model import LgbModel
from vnpy.alpha.strategy import BacktestingEngine
from vnpy.alpha.strategy.strategies.equity_demo_strategy import EquityDemoStrategy


# ── 配置 ──────────────────────────────────────────────────────────────────────
LAB_PATH     = "/Users/wiger-2/Documents/vnpy/lab/csi300"
INDEX_SYMBOL = "000300.SSE"
DATASET_NAME = "csi300_fund30"
MODEL_NAME   = "csi300_fund30_lgb"
SIGNAL_NAME  = "csi300_fund30_lgb"

TEST_START   = datetime(2023, 1, 1)
TEST_END     = datetime(2024, 12, 31)

OUTPUT_DIR   = Path(LAB_PATH) / "output"
PLOT_FILE    = OUTPUT_DIR / "backtest_performance.png"


# ── 每日IC计算（用于信号IC面板）────────────────────────────────────────────────

def daily_rank_ic(signal_df: pd.DataFrame, label_col: str = "label") -> pd.Series:
    """按日计算 rank IC"""
    result = {}
    for dt, grp in signal_df.groupby("datetime"):
        grp = grp.dropna(subset=["signal", label_col])
        if len(grp) < 10:
            continue
        ic, _ = spearmanr(grp["signal"], grp[label_col])
        if not np.isnan(ic):
            result[dt] = ic
    return pd.Series(result).sort_index()


# ── 综合业绩图 ─────────────────────────────────────────────────────────────────

def plot_performance(
    daily_df:        pd.DataFrame,     # 回测逐日结果（含 balance, return, drawdown 等列）
    benchmark_ret:   pd.Series,        # 基准每日涨跌幅（index=date）
    capital:         float,
    ic_series:       pd.Series,        # 信号IC序列（index=datetime）
    feat_importance: pd.DataFrame,     # 特征重要性（feature, importance）
    stats:           dict,
) -> None:
    """
    6面板研究报告式回测图:
      ┌──────────────────────────────────┬────────────┐
      │ [1] 累计收益曲线（策略/基准/超额） │ [2] 指标表  │
      ├──────────────────────────────────┤            │
      │ [3] 年度收益柱状图                │            │
      ├──────────────────────────────────┴────────────┤
      │ [4] 超额收益回撤                               │
      ├───────────────────────┬────────────────────────┤
      │ [5] 信号IC + 60日ICIR  │ [6] Top20特征重要性    │
      └───────────────────────┴────────────────────────┘
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── 预处理 ──────────────────────────────────────────────────────────────
    df = daily_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()

    # 策略累计收益
    df["cum_ret"] = (df["balance"] / capital - 1) * 100

    # 对齐基准
    bm = benchmark_ret.reindex(df.index, method="ffill").fillna(0)
    df["bm_cum"] = ((1 + bm).cumprod() - 1) * 100

    # 超额收益
    strat_daily = df["return"].fillna(0)
    df["excess_daily"] = strat_daily - bm
    df["excess_cum"]   = ((1 + df["excess_daily"]).cumprod() - 1) * 100
    df["excess_hw"]    = df["excess_cum"].cummax()
    df["excess_dd"]    = df["excess_cum"] - df["excess_hw"]

    # 年度收益
    annual = (df["return"].resample("YE").apply(lambda x: (1 + x).prod() - 1) * 100)
    annual.index = annual.index.year

    # IC
    ic_pd = ic_series.copy()
    ic_pd.index = pd.to_datetime(ic_pd.index)
    ic_rolling = ic_pd.rolling(60, min_periods=20)
    ic_roll_mean = ic_rolling.mean()
    ic_roll_std  = ic_rolling.std()
    icir_rolling = ic_roll_mean / (ic_roll_std + 1e-8)

    # ── 画布布局 ─────────────────────────────────────────────────────────
    DARK   = "#212121"
    LIGHT  = "#F8F9FA"
    GRID   = "#E0E0E0"
    BLUE   = "#1565C0"
    GREEN  = "#2E7D32"
    RED    = "#C62828"
    ORANGE = "#E65100"
    PURPLE = "#6A1B9A"

    fig = plt.figure(figsize=(22, 18), facecolor=LIGHT)
    fig.patch.set_facecolor(LIGHT)

    gs = gridspec.GridSpec(
        4, 2, figure=fig,
        height_ratios=[2.2, 1.0, 1.2, 1.5],
        hspace=0.42, wspace=0.28,
        top=0.93, bottom=0.05, left=0.06, right=0.97
    )

    def style_ax(ax, title="", xlabel="", ylabel=""):
        ax.set_facecolor(LIGHT)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color(GRID)
        ax.tick_params(colors=DARK, labelsize=9)
        ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
        if title:  ax.set_title(title, fontsize=10.5, color=DARK, pad=5, fontweight="bold")
        if xlabel: ax.set_xlabel(xlabel, fontsize=9, color=DARK)
        if ylabel: ax.set_ylabel(ylabel, fontsize=9, color=DARK)

    # ── [1] 累计收益曲线 ─────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(df.index, df["cum_ret"],    color=BLUE,   lw=2.0, label="策略", zorder=3)
    ax1.plot(df.index, df["bm_cum"],     color=DARK,   lw=1.5, ls="--", alpha=0.7, label="中证500", zorder=2)
    ax1.plot(df.index, df["excess_cum"], color=GREEN,  lw=1.5, label="超额收益", zorder=3)
    ax1.fill_between(df.index, df["excess_cum"], 0,
                     where=df["excess_cum"] >= 0, alpha=0.12, color=GREEN)
    ax1.fill_between(df.index, df["excess_cum"], 0,
                     where=df["excess_cum"] < 0,  alpha=0.12, color=RED)
    ax1.axhline(0, color=DARK, lw=0.6)
    ax1.set_ylabel("累计收益 (%)", fontsize=9, color=DARK)
    ax1.legend(fontsize=8.5, loc="upper left", framealpha=0.9)
    style_ax(ax1, title="累计收益（测试集 2023-2024）")

    # ── [2] 指标表 ───────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0:2, 1])
    ax2.axis("off")

    stat_rows = [
        ("回测区间",    f"{stats.get('start_date','')}  →  {stats.get('end_date','')}"),
        ("初始资金",    f"¥ {stats.get('capital', 0):,.0f}"),
        ("结束资金",    f"¥ {stats.get('end_balance', 0):,.0f}"),
        ("总收益率",    f"{stats.get('total_return', 0):+.2f} %"),
        ("年化收益率",  f"{stats.get('annual_return', 0):+.2f} %"),
        ("Sharpe比率",  f"{stats.get('sharpe_ratio', 0):.4f}"),
        ("最大回撤",    f"{stats.get('max_ddpercent', 0):.2f} %"),
        ("最长回撤天数", f"{stats.get('max_drawdown_duration', 0)} 天"),
        ("收益回撤比",  f"{stats.get('return_drawdown_ratio', 0):.2f}"),
        ("日均成交额",  f"¥ {stats.get('daily_turnover', 0):,.0f}"),
        ("总手续费",    f"¥ {stats.get('total_commission', 0):,.0f}"),
        ("盈利天/亏损天", f"{stats.get('profit_days',0)} / {stats.get('loss_days',0)}"),
    ]

    table_data = [[k, v] for k, v in stat_rows]
    tbl = ax2.table(
        cellText   =table_data,
        colWidths  =[0.48, 0.52],
        loc        ="center",
        cellLoc    ="left",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.55)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor(GRID)
        if c == 0:
            cell.set_facecolor("#E3F2FD")
            cell.set_text_props(color=DARK, fontweight="semibold")
        else:
            cell.set_facecolor("#FAFAFA")
            cell.set_text_props(color=BLUE if "%" in cell.get_text().get_text() else DARK)
    ax2.set_title("策略统计指标", fontsize=10.5, color=DARK, fontweight="bold", pad=10)

    # ── [3] 年度收益柱状图 ───────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    bar_colors = [GREEN if v >= 0 else RED for v in annual.values]
    bars = ax3.bar(annual.index.astype(str), annual.values, color=bar_colors,
                   width=0.55, zorder=2)
    for bar in bars:
        h = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width() / 2,
                 h + (0.3 if h >= 0 else -0.8),
                 f"{h:.1f}%", ha="center", va="bottom" if h >= 0 else "top",
                 fontsize=8.5, color=DARK)
    ax3.axhline(0, color=DARK, lw=0.6)
    style_ax(ax3, title="年度收益", ylabel="收益率 (%)")

    # ── [4] 超额收益回撤 ─────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[2, :])
    ax4.fill_between(df.index, df["excess_dd"], 0, color=RED, alpha=0.45, zorder=2)
    ax4.plot(df.index, df["excess_dd"], color=RED, lw=0.8, zorder=3)
    ax4.axhline(0, color=DARK, lw=0.6)
    ax4b = ax4.twinx()
    ax4b.plot(df.index, df["excess_cum"], color=BLUE, lw=1.5, alpha=0.6, label="超额累计")
    ax4b.set_ylabel("超额累计收益 (%)", fontsize=9, color=BLUE)
    ax4b.tick_params(axis="y", labelcolor=BLUE, labelsize=9)
    ax4b.spines[["top"]].set_visible(False)
    style_ax(ax4, title="超额收益回撤", ylabel="回撤幅度 (%)")

    # ── [5] 信号IC + 滚动ICIR ────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[3, 0])
    ax5.bar(ic_pd.index, ic_pd.values, color=np.where(ic_pd.values >= 0, BLUE, RED),
            alpha=0.4, width=1.5, label="日IC")
    ax5.plot(ic_roll_mean.index, ic_roll_mean.values, color=BLUE, lw=1.5,
             label="60日均IC")
    ax5b = ax5.twinx()
    ax5b.plot(icir_rolling.index, icir_rolling.values, color=ORANGE, lw=1.5,
              ls="--", label="60日ICIR")
    ax5b.axhline(0.5,  color=ORANGE, lw=0.7, ls=":", alpha=0.6)
    ax5b.axhline(-0.5, color=ORANGE, lw=0.7, ls=":", alpha=0.6)
    ax5b.set_ylabel("ICIR", fontsize=9, color=ORANGE)
    ax5b.tick_params(axis="y", labelcolor=ORANGE, labelsize=9)
    ax5b.spines[["top"]].set_visible(False)
    ax5.axhline(0, color=DARK, lw=0.6)
    ax5.axhline( 0.03, color=GREEN, lw=0.7, ls="--", alpha=0.5)
    ax5.axhline(-0.03, color=RED,   lw=0.7, ls="--", alpha=0.5)
    lines1, labels1 = ax5.get_legend_handles_labels()
    lines2, labels2 = ax5b.get_legend_handles_labels()
    ax5.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper left")
    style_ax(ax5, title="信号IC  |  60日滚动ICIR", ylabel="IC")

    # ── [6] Top20特征重要性 ──────────────────────────────────────────────
    ax6 = fig.add_subplot(gs[3, 1])
    fi = feat_importance.head(20).sort_values("importance", ascending=True)
    bar_h = ax6.barh(fi["feature"], fi["importance"], color=PURPLE, alpha=0.75)
    ax6.set_xlabel("重要性得分", fontsize=9, color=DARK)
    style_ax(ax6, title="LightGBM Top20特征重要性")
    ax6.tick_params(axis="y", labelsize=8)

    # ── 总标题 ────────────────────────────────────────────────────────────
    fig.suptitle(
        "中证500 基本面30因子策略  ·  样本外回测报告（2023-2024）",
        fontsize=15, fontweight="bold", color=DARK, y=0.97
    )

    plt.savefig(str(PLOT_FILE), dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"\n  业绩图已保存: {PLOT_FILE}")


# ── 主流程 ─────────────────────────────────────────────────────────────────────

def sep(title: str = "") -> None:
    print("\n" + "=" * 60)
    if title:
        print(title)
        print("=" * 60)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    lab = AlphaLab(LAB_PATH)

    # ── 1. 加载数据集 ────────────────────────────────────────────────────
    sep("[1/5] 加载 Fundamental30 数据集")
    dataset: AlphaDataset = lab.load_dataset(DATASET_NAME)
    if dataset is None:
        print("错误: 数据集不存在，请先运行 fundamental_factors.py")
        return

    for seg_name, seg in [("train", Segment.TRAIN), ("valid", Segment.VALID), ("test", Segment.TEST)]:
        df = dataset.fetch_infer(seg)
        print(f"  {seg_name:5s}: {df.shape[0]:>8,} 样本  ×  {df.shape[1]} 列")

    # ── 2. 训练 LightGBM ─────────────────────────────────────────────────
    sep("[2/5] 训练 LightGBM 模型")
    model: AlphaModel = LgbModel(
        learning_rate        =0.05,
        num_leaves           =63,
        num_boost_round      =300,
        early_stopping_rounds=30,
        seed                 =42,
    )
    model.fit(dataset)

    sep_line = "─" * 40
    print(f"\n{sep_line}\n特征重要性 (Top20)\n{sep_line}")
    model.detail()

    lab.save_model(MODEL_NAME, model)
    print(f"\n模型已保存: {MODEL_NAME}")

    # 提取特征重要性 DataFrame
    try:
        lgb_model = model.model          # 底层 LightGBM Booster
        fi_vals   = lgb_model.feature_importance(importance_type="gain")
        fi_names  = lgb_model.feature_name()
        feat_importance = (
            pd.DataFrame({"feature": fi_names, "importance": fi_vals})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )
    except Exception:
        feat_importance = pd.DataFrame({"feature": [], "importance": []})

    # ── 3. 生成测试集信号 ─────────────────────────────────────────────────
    sep("[3/5] 测试集信号生成（2023-01-01 ~ 2024-12-31）")
    pre: np.ndarray      = model.predict(dataset, Segment.TEST)
    test_df: pl.DataFrame = dataset.fetch_infer(Segment.TEST)
    test_df = test_df.with_columns(pl.Series(pre).alias("signal"))
    signal: pl.DataFrame = test_df.select(["datetime", "vt_symbol", "signal"])

    print(f"信号形状: {signal.shape}")
    print(f"信号统计: mean={signal['signal'].mean():.4f}  "
          f"std={signal['signal'].std():.4f}  "
          f"min={signal['signal'].min():.4f}  "
          f"max={signal['signal'].max():.4f}")

    # 计算IC序列（用raw_df的label）
    raw_test_cols = ["datetime", "vt_symbol", "label"] if "label" in dataset.raw_df.columns else []
    ic_series = pd.Series(dtype=float)
    if raw_test_cols:
        from vnpy.alpha.dataset.template import query_by_time
        raw_test = query_by_time(dataset.raw_df, str(TEST_START.date()), str(TEST_END.date()))
        # 合并信号与标签
        sig_pd  = signal.to_pandas()
        raw_pd  = raw_test.select(["datetime", "vt_symbol", "label"]).to_pandas()
        merged  = sig_pd.merge(raw_pd, on=["datetime", "vt_symbol"], how="inner")
        ic_series = daily_rank_ic(merged)
        ic_mean = ic_series.mean()
        ic_std  = ic_series.std()
        icir    = ic_mean / ic_std if ic_std > 1e-8 else 0
        print(f"\n信号 IC: mean={ic_mean:.4f}  ICIR={icir:.4f}  "
              f"胜率={( ic_series > 0).mean():.2%}  有效天={len(ic_series)}")

    lab.save_signal(SIGNAL_NAME, signal)
    print(f"\n信号已保存: {SIGNAL_NAME}")

    # ── 4. 策略回测 ───────────────────────────────────────────────────────
    sep("[4/5] 策略回测（EquityDemoStrategy，top_k=50）")

    component_symbols = lab.load_component_symbols(INDEX_SYMBOL, "2023-01-01", "2024-12-31")
    signal = lab.load_signal(SIGNAL_NAME)

    engine = BacktestingEngine(lab)
    engine.set_parameters(
        vt_symbols =component_symbols,
        interval   =Interval.DAILY,
        start      =TEST_START,
        end        =TEST_END,
        capital    =100_000_000,   # 1亿初始资金
    )
    engine.add_strategy(EquityDemoStrategy, {
        "top_k"      : 50,    # 持有中证500中top50（约10%仓位）
        "n_drop"     : 5,     # 每次换出最多5只
        "hold_thresh": 3,     # 最短持有3天
    }, signal)

    engine.load_data()
    engine.run_backtesting()
    engine.calculate_result()
    stats = engine.calculate_statistics()

    # ── 5. 生成业绩图 ─────────────────────────────────────────────────────
    sep("[5/5] 生成综合业绩图")

    daily_pd = engine.daily_df.to_pandas()
    daily_pd["date"] = pd.to_datetime(daily_pd["date"])

    # 加载基准价格，计算日涨跌幅
    from vnpy.trader.object import BarData
    bm_bars: list[BarData] = lab.load_bar_data(INDEX_SYMBOL, Interval.DAILY, TEST_START, TEST_END)
    bm_prices = pd.Series(
        [b.close_price for b in bm_bars],
        index=pd.to_datetime([b.datetime.date() for b in bm_bars])
    )
    bm_ret = bm_prices.pct_change().fillna(0)

    plot_performance(
        daily_df       =daily_pd,
        benchmark_ret  =bm_ret,
        capital        =engine.capital,
        ic_series      =ic_series,
        feat_importance=feat_importance,
        stats          =stats,
    )

    sep("完成")
    print(f"业绩图: {PLOT_FILE}")
    print(f"模型:   {LAB_PATH}/model/{MODEL_NAME}")
    print(f"信号:   {LAB_PATH}/signal/{SIGNAL_NAME}.parquet")


if __name__ == "__main__":
    main()
