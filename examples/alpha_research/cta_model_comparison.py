"""
CTA 28因子 多模型/多参数对比回测

8种配置对比（均在 2025-2026 样本外数据上评估）:
  1. lgb_deep     — LightGBM 63叶 lr=0.05  (原始)
  2. lgb_medium   — LightGBM 31叶 lr=0.03  min_data=50
  3. lgb_shallow  — LightGBM 15叶 lr=0.02  min_data=200
  4. lasso_001    — Lasso α=0.001
  5. lasso_0001   — Lasso α=0.0001
  6. single_tsmom — 单因子 tsmom_12m
  7. single_lvol  — 单因子 low_vol
  8. combo_top3   — 组合 z-score(tsmom_12m + tsmom_vol + low_vol)

前提: 已执行 cta_factors.py（数据集 csi300_cta28 存在）
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
from cta_factors import CTA28  # noqa

from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
from scipy.stats import spearmanr

from vnpy.alpha import AlphaLab, Segment, AlphaDataset
from vnpy.alpha.model.models.lgb_model   import LgbModel
from vnpy.alpha.model.models.lasso_model import LassoModel
from vnpy.alpha.strategy import BacktestingEngine
from vnpy.alpha.strategy.strategies.equity_demo_strategy import EquityDemoStrategy
from vnpy.trader.constant import Interval
from vnpy.alpha.dataset.template import query_by_time


# ── 配置 ──────────────────────────────────────────────────────────────────────
LAB_PATH     = "/Users/wiger-2/Documents/vnpy/lab/csi300"
INDEX_SYMBOL = "000300.SSE"
DATASET_NAME = "csi300_cta28"
TEST_START   = datetime(2025, 1, 1)
TEST_END     = datetime(2026, 4, 20)
CAPITAL      = 100_000_000

OUTPUT_DIR   = Path(LAB_PATH) / "output"
PLOT_FILE    = OUTPUT_DIR / "cta_comparison.png"

# 回测策略参数（所有配置共用）
STRATEGY_SETTING = {"top_k": 30, "n_drop": 5, "hold_thresh": 3}


# ── 8种对比配置 ───────────────────────────────────────────────────────────────
CONFIGS = [
    {
        "id":   "lgb_deep",
        "name": "LGB 63叶 lr=0.05\n（原始）",
        "type": "lgb",
        "lgb_params": dict(learning_rate=0.05, num_leaves=63,
                           num_boost_round=400, early_stopping_rounds=40),
        "color": "#1565C0",
    },
    {
        "id":   "lgb_medium",
        "name": "LGB 31叶 lr=0.03\nmin_data=50",
        "type": "lgb",
        "lgb_params": dict(learning_rate=0.03, num_leaves=31,
                           num_boost_round=500, early_stopping_rounds=50,
                           min_data_in_leaf=50),
        "color": "#00838F",
    },
    {
        "id":   "lgb_shallow",
        "name": "LGB 15叶 lr=0.02\nmin_data=200",
        "type": "lgb",
        "lgb_params": dict(learning_rate=0.02, num_leaves=15,
                           num_boost_round=600, early_stopping_rounds=60,
                           min_data_in_leaf=200),
        "color": "#00BCD4",
    },
    {
        "id":   "lasso_001",
        "name": "Lasso α=0.001\n（强稀疏）",
        "type": "lasso",
        "lasso_alpha": 0.001,
        "color": "#E65100",
    },
    {
        "id":   "lasso_0001",
        "name": "Lasso α=0.0001\n（弱稀疏）",
        "type": "lasso",
        "lasso_alpha": 0.0001,
        "color": "#FF9800",
    },
    {
        "id":   "single_tsmom",
        "name": "单因子\ntsmom_12m",
        "type": "single",
        "factor": "tsmom_12m",
        "color": "#6A1B9A",
    },
    {
        "id":   "single_lvol",
        "name": "单因子\nlow_vol",
        "type": "single",
        "factor": "low_vol",
        "color": "#880E4F",
    },
    {
        "id":   "combo_top3",
        "name": "组合 z-score\ntsmom_12m+vol+lvol",
        "type": "combo",
        "factors": ["tsmom_12m", "tsmom_vol", "low_vol"],
        "color": "#2E7D32",
    },
]


# ── 工具函数 ───────────────────────────────────────────────────────────────────

def daily_rank_ic(sig_pd: pd.DataFrame) -> pd.Series:
    result = {}
    for dt, grp in sig_pd.groupby("datetime"):
        grp = grp.dropna(subset=["signal", "label"])
        if len(grp) < 10:
            continue
        ic, _ = spearmanr(grp["signal"], grp["label"])
        if not np.isnan(ic):
            result[dt] = ic
    return pd.Series(result).sort_index()


def zscore_series(s: pl.Series) -> pl.Series:
    mu, sd = s.mean(), s.std()
    if sd and sd > 1e-8:
        return (s - mu) / sd
    return s * 0.0


def build_signal(cfg: dict, dataset: AlphaDataset, lab: AlphaLab) -> tuple[pl.DataFrame, dict]:
    """训练/构造信号，返回 (signal_df, model_meta)"""
    meta = {}

    if cfg["type"] == "lgb":
        p = cfg["lgb_params"]
        model = LgbModel(
            learning_rate        =p["learning_rate"],
            num_leaves           =p["num_leaves"],
            num_boost_round      =p["num_boost_round"],
            early_stopping_rounds=p["early_stopping_rounds"],
            seed=42,
        )
        # 注入额外 LGB 参数
        for k in ("min_data_in_leaf", "max_depth", "lambda_l1", "lambda_l2"):
            if k in p:
                model.params[k] = p[k]
        model.fit(dataset)
        best_iter = getattr(model.model, "best_iteration", None)
        meta["best_iter"] = best_iter
        # 特征重要性
        try:
            fi = dict(zip(model.model.feature_name(),
                          model.model.feature_importance(importance_type="gain")))
            meta["feature_importance"] = fi
        except Exception:
            pass
        # Lasso 系数（不适用）
        meta["nonzero_features"] = None

        pred = model.predict(dataset, Segment.TEST)
        df_t = dataset.fetch_infer(Segment.TEST).with_columns(pl.Series(pred).alias("signal"))

    elif cfg["type"] == "lasso":
        model = LassoModel(alpha=cfg["lasso_alpha"], max_iter=2000, random_state=42)
        model.fit(dataset)
        coef = dict(zip(model.feature_names, model.model.coef_))
        nonzero = {k: v for k, v in coef.items() if abs(v) > 1e-8}
        meta["nonzero_features"] = nonzero
        meta["feature_importance"] = {k: abs(v) for k, v in nonzero.items()}
        meta["best_iter"] = None

        pred = model.predict(dataset, Segment.TEST)
        df_t = dataset.fetch_infer(Segment.TEST).with_columns(pl.Series(pred).alias("signal"))

    elif cfg["type"] == "single":
        fname = cfg["factor"]
        df_t = dataset.fetch_infer(Segment.TEST)
        if fname in df_t.columns:
            sig = zscore_series(df_t[fname])
        else:
            sig = pl.Series([0.0] * len(df_t))
        df_t = df_t.with_columns(sig.alias("signal"))
        meta["best_iter"] = None
        meta["feature_importance"] = {fname: 1.0}
        meta["nonzero_features"] = None

    elif cfg["type"] == "combo":
        factors = cfg["factors"]
        df_t = dataset.fetch_infer(Segment.TEST)
        parts = []
        for f in factors:
            if f in df_t.columns:
                parts.append(zscore_series(df_t[f]))
        if parts:
            combo = sum(parts) / len(parts)
        else:
            combo = pl.Series([0.0] * len(df_t))
        df_t = df_t.with_columns(combo.alias("signal"))
        meta["best_iter"] = None
        meta["feature_importance"] = {f: 1.0 / len(factors) for f in factors}
        meta["nonzero_features"] = None

    signal = df_t.select(["datetime", "vt_symbol", "signal"])
    return signal, meta


def run_backtest(signal: pl.DataFrame, lab: AlphaLab,
                 syms: list[str]) -> tuple[dict, pd.DataFrame]:
    """运行回测，返回 (stats, daily_df)"""
    engine = BacktestingEngine(lab)
    engine.set_parameters(
        vt_symbols=syms, interval=Interval.DAILY,
        start=TEST_START, end=TEST_END, capital=CAPITAL,
    )
    engine.add_strategy(EquityDemoStrategy, STRATEGY_SETTING, signal)
    engine.load_data()
    engine.run_backtesting()
    engine.calculate_result()
    stats = engine.calculate_statistics()
    daily_pd = engine.daily_df.to_pandas()
    daily_pd["date"] = pd.to_datetime(daily_pd["date"])
    return stats, daily_pd


# ── 可视化 ─────────────────────────────────────────────────────────────────────

def plot_comparison(all_results: list[dict], bm_ret: pd.Series) -> None:
    """
    6面板综合对比图:
      [1] 所有配置累计收益曲线  [2] 指标热力表（Sharpe/回撤/收益）
      [3] 年化收益/最大回撤横向  [4] 信号IC对比柱状图
      [5] 特征重要性（Top5×每模型）[6] 超额收益曲线
    """
    DARK  = "#212121"; LIGHT = "#F8F9FA"; GRID = "#E0E0E0"
    GREEN = "#2E7D32"; RED   = "#C62828"

    n = len(all_results)

    fig = plt.figure(figsize=(24, 20), facecolor=LIGHT)
    fig.suptitle("沪深300  CTA 28因子  多模型/多参数对比  ·  样本外 2025-2026",
                 fontsize=16, fontweight="bold", color=DARK, y=0.975)
    gs = gridspec.GridSpec(3, 2, figure=fig,
                           height_ratios=[1.8, 1.2, 1.4],
                           hspace=0.50, wspace=0.28,
                           top=0.94, bottom=0.05, left=0.06, right=0.97)

    def sty(ax, title="", ylabel="", xgrid=False):
        ax.set_facecolor(LIGHT)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color(GRID)
        ax.tick_params(colors=DARK, labelsize=8.5)
        ax.grid(axis="y", color=GRID, lw=0.5, zorder=0)
        if xgrid:
            ax.grid(axis="x", color=GRID, lw=0.5, zorder=0)
        if title:  ax.set_title(title,  fontsize=11, color=DARK, pad=5, fontweight="bold")
        if ylabel: ax.set_ylabel(ylabel, fontsize=9, color=DARK)

    # ── [1] 累计收益曲线 ──────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    # 基准
    dates_all = all_results[0]["daily_df"]["date"]
    bm = bm_ret.reindex(dates_all, method="ffill").fillna(0)
    bm_cum = ((1 + bm).cumprod() - 1) * 100
    ax1.plot(dates_all, bm_cum, color=DARK, lw=1.5, ls="--", alpha=0.5, label="沪深300基准")

    for r in all_results:
        df = r["daily_df"].set_index("date")
        cum = (df["balance"] / CAPITAL - 1) * 100
        ax1.plot(cum.index, cum.values,
                 color=r["color"], lw=1.8, label=r["id"].replace("_", " "), zorder=3)
    ax1.axhline(0, color=DARK, lw=0.6)
    ax1.legend(fontsize=7.5, loc="upper left", framealpha=0.9, ncol=2)
    sty(ax1, "全配置累计收益曲线（样本外 2025-2026）", "收益率 (%)")

    # ── [2] 指标热力表 ────────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.axis("off")

    metric_keys = ["total_return", "annual_return", "sharpe_ratio",
                   "max_ddpercent", "return_drawdown_ratio"]
    metric_labels = ["总收益%", "年化收益%", "Sharpe", "最大回撤%", "收益回撤比"]

    mat = []
    row_labels = []
    for r in all_results:
        s = r["stats"]
        mat.append([
            s.get("total_return", 0),
            s.get("annual_return", 0),
            s.get("sharpe_ratio", 0),
            abs(s.get("max_ddpercent", 0)),
            s.get("return_drawdown_ratio", 0),
        ])
        row_labels.append(r["name"].replace("\n", " "))

    mat_np = np.array(mat, dtype=float)

    # 各列归一化 → 颜色映射（值越大越好，除最大回撤列）
    col_norm = np.zeros_like(mat_np)
    for j in range(mat_np.shape[1]):
        col = mat_np[:, j].copy()
        if j == 3:  # 最大回撤取反（越小越好）
            col = -col
        rng = col.max() - col.min()
        if rng > 0:
            col_norm[:, j] = (col - col.min()) / rng

    cell_text = []
    for i, row in enumerate(mat):
        cell_text.append([
            f"{row[0]:+.1f}",
            f"{row[1]:+.1f}",
            f"{row[2]:.3f}",
            f"{row[3]:.1f}",
            f"{row[4]:.2f}",
        ])

    tbl = ax2.table(
        cellText=cell_text,
        colLabels=metric_labels,
        rowLabels=row_labels,
        loc="center", cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.05, 1.62)

    cmap = plt.cm.RdYlGn
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor(GRID)
        if r == 0:  # header
            cell.set_facecolor("#E3F2FD")
            cell.set_text_props(fontweight="bold", color=DARK, fontsize=9)
        elif c == -1:  # row label
            cell.set_facecolor("#F5F5F5")
            cell.set_text_props(color=DARK, fontsize=8.5)
        else:
            v = col_norm[r - 1, c]
            rgba = cmap(0.2 + v * 0.6)
            cell.set_facecolor(rgba)
            cell.set_text_props(color="white" if v > 0.6 else DARK, fontsize=9)

    ax2.set_title("性能指标对比（颜色：绿=优，红=差，回撤列取反）",
                  fontsize=11, color=DARK, pad=8, fontweight="bold")

    # ── [3] 年化收益 vs 最大回撤散点 ─────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    for r in all_results:
        s = r["stats"]
        ann = s.get("annual_return", 0)
        mdd = abs(s.get("max_ddpercent", 0))
        sharpe = s.get("sharpe_ratio", 0)
        ax3.scatter(mdd, ann, s=150, color=r["color"], zorder=4, alpha=0.9)
        ax3.annotate(r["id"].replace("_", "\n"),
                     (mdd, ann), xytext=(4, 3),
                     textcoords="offset points", fontsize=7, color=r["color"])
    ax3.axhline(0, color=DARK, lw=0.7)
    ax3.set_xlabel("最大回撤 (%)", fontsize=9, color=DARK)
    sty(ax3, "年化收益 vs 最大回撤（右上角最优）", "年化收益率 (%)")

    # ── [4] 信号 IC 对比（柱状图）────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    names = [r["id"].replace("_", "\n") for r in all_results]
    ics   = [r.get("ic_mean", 0) for r in all_results]
    icirs = [r.get("icir", 0)    for r in all_results]
    x     = np.arange(len(names))
    w     = 0.35
    bars1 = ax4.bar(x - w/2, ics,   width=w, label="IC均值",
                    color=[r["color"] for r in all_results], alpha=0.7, zorder=2)
    ax4b  = ax4.twinx()
    ax4b.bar(x + w/2, icirs, width=w, label="ICIR",
             color=[r["color"] for r in all_results], alpha=0.45, zorder=2, hatch="//")
    ax4b.set_ylabel("ICIR", fontsize=9, color="#555")
    ax4b.tick_params(axis="y", labelsize=8.5)
    ax4b.spines[["top"]].set_visible(False)
    ax4.set_xticks(x)
    ax4.set_xticklabels(names, fontsize=7.5)
    ax4.axhline(0, color=DARK, lw=0.6)
    ax4.axhline( 0.03, color=GREEN, lw=0.7, ls="--", alpha=0.5)
    ax4.axhline(-0.03, color=RED,   lw=0.7, ls="--", alpha=0.5)
    l1, lb1 = ax4.get_legend_handles_labels()
    l2, lb2 = ax4b.get_legend_handles_labels()
    ax4.legend(l1 + l2, lb1 + lb2, fontsize=8, loc="upper left")
    sty(ax4, "各配置信号 IC / ICIR（测试集 2025-2026）", "IC均值")

    # ── [5] 特征重要性 Top6（按各模型前6特征）────────────────────────────────
    ax5 = fig.add_subplot(gs[2, 0])
    all_feats: dict[str, float] = {}
    for r in all_results:
        fi = r.get("feature_importance") or {}
        for k, v in fi.items():
            all_feats[k] = all_feats.get(k, 0) + abs(v)

    top_feats = sorted(all_feats.items(), key=lambda x: x[1], reverse=True)[:12]
    feat_names = [f[0] for f in top_feats]

    y = np.arange(len(feat_names))
    left = np.zeros(len(feat_names))
    for r in all_results:
        fi = r.get("feature_importance") or {}
        vals = [fi.get(f, 0) for f in feat_names]
        total = sum(fi.values()) if fi else 1
        norm_vals = [v / (total + 1e-8) for v in vals]
        ax5.barh(y, norm_vals, left=left,
                 color=r["color"], alpha=0.8, label=r["id"].replace("_", " "))
        left += np.array(norm_vals)

    ax5.set_yticks(y)
    ax5.set_yticklabels(feat_names, fontsize=8.5)
    ax5.set_xlabel("归一化重要性（各模型叠加）", fontsize=9, color=DARK)
    ax5.legend(fontsize=7, loc="lower right", ncol=2, framealpha=0.85)
    sty(ax5, "Top12 因子重要性（各模型叠加堆积）", xgrid=True)
    ax5.grid(axis="y", visible=False)

    # ── [6] 超额收益曲线 ──────────────────────────────────────────────────────
    ax6 = fig.add_subplot(gs[2, 1])
    for r in all_results:
        df = r["daily_df"].set_index("date")
        cum   = (df["balance"] / CAPITAL - 1) * 100
        bm_a  = bm_ret.reindex(df.index, method="ffill").fillna(0)
        bm_c  = ((1 + bm_a).cumprod() - 1) * 100
        strat_daily = df["return"].fillna(0)
        exc   = ((1 + strat_daily - bm_a).cumprod() - 1) * 100
        ax6.plot(exc.index, exc.values,
                 color=r["color"], lw=1.6, label=r["id"].replace("_", " "), zorder=3)
    ax6.axhline(0, color=DARK, lw=0.7)
    ax6.fill_between(exc.index, 0, 0, alpha=0)
    ax6.legend(fontsize=7.5, loc="upper left", ncol=2, framealpha=0.9)
    sty(ax6, "各配置超额收益曲线（vs 沪深300）", "超额收益 (%)")

    plt.savefig(str(PLOT_FILE), dpi=160, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"\n对比图 → {PLOT_FILE}")


# ── 主流程 ─────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    lab = AlphaLab(LAB_PATH)

    print("=" * 65)
    print("加载数据集 & 基准行情")
    print("=" * 65)
    dataset: AlphaDataset = lab.load_dataset(DATASET_NAME)
    if dataset is None:
        print("错误: 数据集不存在，请先运行 cta_factors.py")
        return

    for seg, name in [(Segment.TRAIN, "train"), (Segment.VALID, "valid"),
                      (Segment.TEST, "test")]:
        df = dataset.fetch_infer(seg)
        print(f"  {name:5s}: {df.shape[0]:>8,} 条  ×  {df.shape[1]} 列")

    # 基准日涨跌幅
    from vnpy.trader.object import BarData
    syms = lab.load_component_symbols(INDEX_SYMBOL,
                                      str(TEST_START.date()), str(TEST_END.date()))
    bm_bars = lab.load_bar_data(INDEX_SYMBOL, Interval.DAILY, TEST_START, TEST_END)
    bm_prices = pd.Series(
        [b.close_price for b in bm_bars],
        index=pd.to_datetime([b.datetime.date() for b in bm_bars])
    )
    bm_ret = bm_prices.pct_change().fillna(0)

    # 测试集 label（用于计算信号IC）
    raw_test = query_by_time(dataset.raw_df,
                             str(TEST_START.date()), str(TEST_END.date()))
    raw_pd = raw_test.select(["datetime", "vt_symbol", "label"]).to_pandas()

    # ── 逐配置运行 ────────────────────────────────────────────────────────────
    all_results = []
    header = (f"\n{'配置':<22} {'总收益%':>8} {'年化%':>7} {'Sharpe':>8} "
              f"{'最大回撤%':>10} {'收益/回撤':>9} {'IC均值':>8} {'ICIR':>8} "
              f"{'最佳轮数':>8}")
    print("\n" + "=" * 65)
    print("开始逐配置训练 + 回测")
    print("=" * 65)
    print(header)
    print("-" * len(header))

    for cfg in CONFIGS:
        print(f"\n── {cfg['id']} ──")

        # 构建信号
        signal, meta = build_signal(cfg, dataset, lab)

        # 信号 IC
        sig_pd  = signal.to_pandas()
        merged  = sig_pd.merge(raw_pd, on=["datetime", "vt_symbol"]).dropna()
        ic_s = pd.Series(dtype=float)
        for dt, grp in merged.groupby("datetime"):
            if len(grp) < 10:
                continue
            ic, _ = spearmanr(grp["signal"], grp["label"])
            if not np.isnan(ic):
                ic_s[dt] = ic

        ic_mean = ic_s.mean() if not ic_s.empty else 0
        icir    = ic_mean / (ic_s.std() + 1e-8) if not ic_s.empty else 0

        # 回测
        stats, daily_pd = run_backtest(signal, lab, syms)

        result = {
            "id":               cfg["id"],
            "name":             cfg["name"],
            "color":            cfg["color"],
            "stats":            stats,
            "daily_df":         daily_pd,
            "ic_mean":          ic_mean,
            "icir":             icir,
            "best_iter":        meta.get("best_iter"),
            "feature_importance": meta.get("feature_importance"),
            "nonzero_features": meta.get("nonzero_features"),
        }
        all_results.append(result)

        s = stats
        bi = meta.get("best_iter") or "-"
        print(f"  {cfg['id']:<22} "
              f"{s.get('total_return',0):>8.1f} "
              f"{s.get('annual_return',0):>7.1f} "
              f"{s.get('sharpe_ratio',0):>8.3f} "
              f"{abs(s.get('max_ddpercent',0)):>10.1f} "
              f"{s.get('return_drawdown_ratio',0):>9.2f} "
              f"{ic_mean:>8.4f} "
              f"{icir:>8.4f} "
              f"{str(bi):>8}")

    # ── 汇总表 ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("汇总排名（按 Sharpe 降序）")
    print("=" * 65)
    sorted_r = sorted(all_results, key=lambda x: x["stats"].get("sharpe_ratio", 0), reverse=True)
    print(f"  {'Rank':>4}  {'配置':<22} {'Sharpe':>8} {'年化%':>7} {'最大回撤%':>10}")
    for i, r in enumerate(sorted_r, 1):
        s = r["stats"]
        print(f"  #{i:<3}  {r['id']:<22} "
              f"{s.get('sharpe_ratio',0):>8.3f} "
              f"{s.get('annual_return',0):>7.1f} "
              f"{abs(s.get('max_ddpercent',0)):>10.1f}")

    # Lasso 系数输出
    print("\n── Lasso 选出的非零因子 ──")
    for r in all_results:
        if r.get("nonzero_features"):
            nz = r["nonzero_features"]
            top = sorted(nz.items(), key=lambda x: abs(x[1]), reverse=True)[:8]
            print(f"\n  {r['id']} ({len(nz)} 个非零):")
            for name, coef in top:
                print(f"    {name:<18} {coef:+.6f}")

    # ── 生成对比图 ────────────────────────────────────────────────────────────
    plot_comparison(all_results, bm_ret)

    # ── 保存 CSV ──────────────────────────────────────────────────────────────
    rows = []
    for r in all_results:
        s = r["stats"]
        rows.append({
            "config":          r["id"],
            "total_return":    s.get("total_return", 0),
            "annual_return":   s.get("annual_return", 0),
            "sharpe_ratio":    s.get("sharpe_ratio", 0),
            "max_ddpercent":   s.get("max_ddpercent", 0),
            "return_dd_ratio": s.get("return_drawdown_ratio", 0),
            "ic_mean":         r["ic_mean"],
            "icir":            r["icir"],
            "best_iter":       r["best_iter"],
        })
    csv_path = OUTPUT_DIR / "cta_comparison_stats.csv"
    pd.DataFrame(rows).to_csv(str(csv_path), index=False, encoding="utf-8-sig")
    print(f"\n汇总 CSV → {csv_path}")


if __name__ == "__main__":
    main()
