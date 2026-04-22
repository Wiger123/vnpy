"""
CTA 28因子 LightGBM 模型训练 + 2025-2026 样本外回测

前提: 已执行 cta_factors.py（数据集 csi300_cta28 已保存）
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
sys.path.insert(0, "/Users/wiger-2/Documents/vnpy/examples/alpha_research")
from cta_factors import CTA28  # noqa — pickle 反序列化需要类定义

from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import polars as pl
import matplotlib.gridspec as gridspec
from scipy.stats import spearmanr

from vnpy.alpha import AlphaLab, Segment, AlphaDataset, AlphaModel
from vnpy.alpha.model.models.lgb_model import LgbModel
from vnpy.alpha.strategy import BacktestingEngine
from vnpy.alpha.strategy.strategies.equity_demo_strategy import EquityDemoStrategy
from vnpy.trader.constant import Interval
from vnpy.trader.object import BarData
from vnpy.alpha.dataset.template import query_by_time


LAB_PATH     = "/Users/wiger-2/Documents/vnpy/lab/csi300"
INDEX_SYMBOL = "000300.SSE"
DATASET_NAME = "csi300_cta28"
MODEL_NAME   = "csi300_cta28_lgb"
SIGNAL_NAME  = "csi300_cta28_lgb"

TEST_START   = datetime(2025, 1, 1)
TEST_END     = datetime(2026, 4, 20)

OUTPUT_DIR   = Path(LAB_PATH) / "output"
PLOT_FILE    = OUTPUT_DIR / "cta_backtest_2025.png"
CAPITAL      = 100_000_000


# ── 每日 rank IC ──────────────────────────────────────────────────────────────

def daily_ic(signal_pd: pd.DataFrame) -> pd.Series:
    result = {}
    for dt, grp in signal_pd.groupby("datetime"):
        grp = grp.dropna(subset=["signal","label"])
        if len(grp) < 10:
            continue
        from scipy.stats import spearmanr
        ic, _ = spearmanr(grp["signal"], grp["label"])
        if not np.isnan(ic):
            result[dt] = ic
    return pd.Series(result).sort_index()


# ── 综合业绩图（6面板）────────────────────────────────────────────────────────

def plot_performance(daily_pd, bm_ret, capital, ic_series, feat_imp, stats):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = daily_pd.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()

    df["cum_ret"]  = (df["balance"] / capital - 1) * 100
    bm             = bm_ret.reindex(df.index, method="ffill").fillna(0)
    df["bm_cum"]   = ((1 + bm).cumprod() - 1) * 100
    exc_daily      = df["return"].fillna(0) - bm
    df["exc_cum"]  = ((1 + exc_daily).cumprod() - 1) * 100
    df["exc_hw"]   = df["exc_cum"].cummax()
    df["exc_dd"]   = df["exc_cum"] - df["exc_hw"]

    annual = (df["return"].resample("YE")
              .apply(lambda x: (1 + x).prod() - 1) * 100)
    annual.index = annual.index.year

    ic_pd   = ic_series.copy(); ic_pd.index = pd.to_datetime(ic_pd.index)
    ic_rm   = ic_pd.rolling(40, min_periods=15).mean()
    icir_r  = ic_rm / (ic_pd.rolling(40, min_periods=15).std() + 1e-8)

    DARK="#212121"; LIGHT="#F8F9FA"; GRID="#E0E0E0"
    BLUE="#1565C0"; GREEN="#2E7D32"; RED="#C62828"; ORANGE="#E65100"; PURPLE="#6A1B9A"

    def sty(ax, title="", ylabel=""):
        ax.set_facecolor(LIGHT)
        ax.spines[["top","right"]].set_visible(False)
        ax.spines[["left","bottom"]].set_color(GRID)
        ax.tick_params(colors=DARK, labelsize=9)
        ax.grid(axis="y", color=GRID, lw=0.5, zorder=0)
        if title:  ax.set_title(title,  fontsize=11, color=DARK, pad=5, fontweight="bold")
        if ylabel: ax.set_ylabel(ylabel, fontsize=9,  color=DARK)

    fig = plt.figure(figsize=(22, 18), facecolor=LIGHT)
    fig.suptitle("沪深300  CTA 28因子策略  ·  样本外回测报告（2025-2026）",
                 fontsize=15, fontweight="bold", color=DARK, y=0.975)
    gs = gridspec.GridSpec(4, 2, figure=fig,
                           height_ratios=[2.2, 1.0, 1.2, 1.5],
                           hspace=0.45, wspace=0.28,
                           top=0.94, bottom=0.05, left=0.06, right=0.97)

    # [1] 累计收益
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(df.index, df["cum_ret"],  color=BLUE,  lw=2.2, label="策略",      zorder=3)
    ax1.plot(df.index, df["bm_cum"],   color=DARK,  lw=1.5, ls="--", alpha=0.65, label="沪深300", zorder=2)
    ax1.plot(df.index, df["exc_cum"],  color=GREEN, lw=1.5, label="超额收益",  zorder=3)
    ax1.fill_between(df.index, df["exc_cum"], 0, where=df["exc_cum"]>=0, alpha=0.12, color=GREEN)
    ax1.fill_between(df.index, df["exc_cum"], 0, where=df["exc_cum"]< 0, alpha=0.12, color=RED)
    ax1.axhline(0, color=DARK, lw=0.6)
    ax1.legend(fontsize=9, loc="upper left", framealpha=0.9)
    sty(ax1, "累计收益（样本外 2025-2026）", "收益率 (%)")

    # [2] 统计表
    ax2 = fig.add_subplot(gs[0:2, 1])
    ax2.axis("off")
    s = stats
    rows_t = [
        ("回测区间",   f"{s.get('start_date','')} → {s.get('end_date','')}"),
        ("初始资金",   f"¥ {s.get('capital',0):,.0f}"),
        ("结束资金",   f"¥ {s.get('end_balance',0):,.0f}"),
        ("总收益率",   f"{s.get('total_return',0):+.2f} %"),
        ("年化收益率", f"{s.get('annual_return',0):+.2f} %"),
        ("Sharpe比率", f"{s.get('sharpe_ratio',0):.4f}"),
        ("最大回撤",   f"{s.get('max_ddpercent',0):.2f} %"),
        ("最长回撤天", f"{s.get('max_drawdown_duration',0)} 天"),
        ("收益回撤比", f"{s.get('return_drawdown_ratio',0):.2f}"),
        ("信号IC",     f"{ic_series.mean():+.4f}"),
        ("信号ICIR",   f"{ic_series.mean()/(ic_series.std()+1e-8):+.4f}"),
        ("总手续费",   f"¥ {s.get('total_commission',0):,.0f}"),
    ]
    tbl = ax2.table(cellText=[[k, v] for k, v in rows_t],
                    colWidths=[0.48, 0.52], loc="center", cellLoc="left")
    tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1, 1.55)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor(GRID)
        if c == 0:
            cell.set_facecolor("#E3F2FD")
            cell.set_text_props(color=DARK, fontweight="semibold")
        else:
            cell.set_facecolor("#FAFAFA")
            v = cell.get_text().get_text()
            cell.set_text_props(color=BLUE if ("%" in v or "¥" in v) else DARK)
    ax2.set_title("策略统计指标", fontsize=11, color=DARK, fontweight="bold", pad=10)

    # [3] 年度收益
    ax3 = fig.add_subplot(gs[1, 0])
    bar_c = [GREEN if v >= 0 else RED for v in annual.values]
    bars  = ax3.bar(annual.index.astype(str), annual.values, color=bar_c, width=0.55, zorder=2)
    for bar in bars:
        h = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width() / 2,
                 h + (0.4 if h >= 0 else -1.0),
                 f"{h:.1f}%", ha="center",
                 va="bottom" if h >= 0 else "top",
                 fontsize=9.5, color=DARK, fontweight="semibold")
    ax3.axhline(0, color=DARK, lw=0.7)
    sty(ax3, "年度收益", "收益率 (%)")

    # [4] 超额回撤（宽面板）
    ax4 = fig.add_subplot(gs[2, :])
    ax4.fill_between(df.index, df["exc_dd"], 0, color=RED, alpha=0.4, zorder=2, label="超额回撤")
    ax4.plot(df.index, df["exc_dd"], color=RED, lw=0.8, zorder=3)
    ax4.axhline(0, color=DARK, lw=0.6)
    ax4b = ax4.twinx()
    ax4b.plot(df.index, df["exc_cum"], color=BLUE, lw=1.8, alpha=0.7, label="超额累计")
    ax4b.set_ylabel("超额累计 (%)", fontsize=9, color=BLUE)
    ax4b.tick_params(axis="y", labelcolor=BLUE, labelsize=9)
    ax4b.spines[["top"]].set_visible(False)
    l1, lb1 = ax4.get_legend_handles_labels()
    l2, lb2 = ax4b.get_legend_handles_labels()
    ax4.legend(l1+l2, lb1+lb2, fontsize=9, loc="lower left", framealpha=0.85)
    sty(ax4, "超额收益回撤", "回撤 (%)")

    # [5] 信号 IC + 40日滚动 ICIR
    ax5 = fig.add_subplot(gs[3, 0])
    pos  = ic_pd.values >= 0
    ax5.bar(ic_pd.index[pos],  ic_pd.values[pos],  width=1.2, color=BLUE, alpha=0.4)
    ax5.bar(ic_pd.index[~pos], ic_pd.values[~pos], width=1.2, color=RED,  alpha=0.4)
    ax5.plot(ic_rm.index, ic_rm.values, color=BLUE, lw=1.8, label="40日均IC")
    ax5b = ax5.twinx()
    ax5b.plot(icir_r.index, icir_r.values, color=ORANGE, lw=1.5, ls="--", label="40日ICIR")
    ax5b.axhline( 0.5, color=ORANGE, lw=0.7, ls=":", alpha=0.6)
    ax5b.axhline(-0.5, color=ORANGE, lw=0.7, ls=":", alpha=0.6)
    ax5b.set_ylabel("ICIR", fontsize=9, color=ORANGE)
    ax5b.tick_params(axis="y", labelcolor=ORANGE, labelsize=9)
    ax5b.spines[["top"]].set_visible(False)
    ax5.axhline(0, color=DARK, lw=0.6)
    ax5.axhline( 0.03, color=GREEN, lw=0.7, ls="--", alpha=0.4)
    ax5.axhline(-0.03, color=RED,   lw=0.7, ls="--", alpha=0.4)
    l1, lb1 = ax5.get_legend_handles_labels()
    l2, lb2 = ax5b.get_legend_handles_labels()
    ax5.legend(l1+l2, lb1+lb2, fontsize=8.5, loc="upper left", framealpha=0.85)
    sty(ax5, "信号 IC  |  40日滚动 ICIR", "IC")

    # [6] Top20 特征重要性
    ax6 = fig.add_subplot(gs[3, 1])
    fi = feat_imp.head(20).sort_values("importance", ascending=True)
    ax6.barh(fi["feature"], fi["importance"], color=PURPLE, alpha=0.78, zorder=2)
    ax6.tick_params(axis="y", labelsize=8.5)
    ax6.grid(axis="x", color=GRID, lw=0.5, zorder=0)
    ax6.grid(axis="y", visible=False)
    sty(ax6, "LightGBM Top20 特征重要性", "重要性得分")

    plt.savefig(str(PLOT_FILE), dpi=160, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  业绩图 → {PLOT_FILE}")


# ── 主流程 ─────────────────────────────────────────────────────────────────────

def sep(t=""):
    print("\n" + "=" * 60 + ("\n" + t + "\n" + "=" * 60 if t else ""))


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    lab = AlphaLab(LAB_PATH)

    sep("[1/5] 加载 CTA28 数据集")
    dataset: AlphaDataset = lab.load_dataset(DATASET_NAME)
    if dataset is None:
        print("错误: 数据集不存在，请先运行 cta_factors.py")
        return
    for seg_name, seg in [("train", Segment.TRAIN), ("valid", Segment.VALID), ("test", Segment.TEST)]:
        df = dataset.fetch_infer(seg)
        print(f"  {seg_name:5s}: {df.shape[0]:>8,} 样本  × {df.shape[1]} 列")

    sep("[2/5] 训练 LightGBM（early_stopping on 2024 验证集）")
    model: AlphaModel = LgbModel(
        learning_rate        =0.05,
        num_leaves           =63,
        num_boost_round      =400,
        early_stopping_rounds=40,
        seed                 =42,
    )
    model.fit(dataset)
    model.detail()
    lab.save_model(MODEL_NAME, model)
    print(f"\n模型已保存: {MODEL_NAME}")

    # 特征重要性
    try:
        fi_vals  = model.model.feature_importance(importance_type="gain")
        fi_names = model.model.feature_name()
        feat_imp = (pd.DataFrame({"feature": fi_names, "importance": fi_vals})
                    .sort_values("importance", ascending=False)
                    .reset_index(drop=True))
    except Exception:
        feat_imp = pd.DataFrame({"feature":[], "importance":[]})

    sep("[3/5] 生成 2025-2026 测试集信号")
    pre       = model.predict(dataset, Segment.TEST)
    test_df   = dataset.fetch_infer(Segment.TEST)
    test_df   = test_df.with_columns(pl.Series(pre).alias("signal"))
    signal    = test_df.select(["datetime","vt_symbol","signal"])
    print(f"信号: {signal.shape[0]:,} 条  "
          f"mean={signal['signal'].mean():.4f}  std={signal['signal'].std():.4f}")

    # 信号 IC
    raw_test = query_by_time(dataset.raw_df,
                             str(TEST_START.date()), str(TEST_END.date()))
    ic_series = pd.Series(dtype=float)
    if "label" in dataset.raw_df.columns:
        sig_pd  = signal.to_pandas()
        raw_pd  = raw_test.select(["datetime","vt_symbol","label"]).to_pandas()
        merged  = sig_pd.merge(raw_pd, on=["datetime","vt_symbol"]).dropna()
        ic_series = daily_ic(merged)
        if not ic_series.empty:
            print(f"信号 IC: mean={ic_series.mean():.4f}  "
                  f"ICIR={ic_series.mean()/(ic_series.std()+1e-8):.4f}  "
                  f"胜率={(ic_series>0).mean():.2%}  有效天={len(ic_series)}")

    lab.save_signal(SIGNAL_NAME, signal)

    sep("[4/5] 策略回测（EquityDemoStrategy，top_k=30）")
    syms   = lab.load_component_symbols(INDEX_SYMBOL,
                                        str(TEST_START.date()), str(TEST_END.date()))
    signal = lab.load_signal(SIGNAL_NAME)

    engine = BacktestingEngine(lab)
    engine.set_parameters(
        vt_symbols=syms, interval=Interval.DAILY,
        start=TEST_START, end=TEST_END, capital=CAPITAL,
    )
    engine.add_strategy(EquityDemoStrategy, {
        "top_k"      : 30,
        "n_drop"     : 5,
        "hold_thresh": 3,
    }, signal)
    engine.load_data()
    engine.run_backtesting()
    engine.calculate_result()
    stats = engine.calculate_statistics()

    sep("[5/5] 生成业绩图")
    daily_pd = engine.daily_df.to_pandas()
    daily_pd["date"] = pd.to_datetime(daily_pd["date"])

    bm_bars  = lab.load_bar_data(INDEX_SYMBOL, Interval.DAILY, TEST_START, TEST_END)
    bm_prices = pd.Series(
        [b.close_price for b in bm_bars],
        index=pd.to_datetime([b.datetime.date() for b in bm_bars])
    )
    bm_ret = bm_prices.pct_change().fillna(0)

    plot_performance(daily_pd, bm_ret, CAPITAL, ic_series, feat_imp, stats)

    sep("完成")
    print(f"业绩图: {PLOT_FILE}")
    print(f"模型:   {LAB_PATH}/model/{MODEL_NAME}")
    print(f"信号:   {LAB_PATH}/signal/{SIGNAL_NAME}.parquet")


if __name__ == "__main__":
    main()
