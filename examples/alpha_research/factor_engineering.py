"""
Alpha158 因子特征工程脚本

基于下载好的 CSI 300 日线数据，计算 158 个量化因子，并输出单因子 IC 分析。

使用方法：
    python factor_engineering.py
（需先运行 download_akshare.py 完成数据下载）
"""

import matplotlib
matplotlib.use("Agg")   # non-interactive backend: no display needed, plots saved to files

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from functools import partial

import polars as pl

from vnpy.trader.constant import Interval
from vnpy.alpha import AlphaLab
from vnpy.alpha.dataset import (
    AlphaDataset,
    process_drop_na,
    process_cs_norm,
    process_fill_na,
)
from vnpy.alpha.dataset.datasets.alpha_158 import Alpha158


# ── 配置 ──────────────────────────────────────────────────────────────────────
LAB_PATH     = "/Users/wiger-2/Documents/vnpy/lab/csi300"
INDEX_SYMBOL = "000300.SSE"
DATASET_NAME = "csi300_alpha158"
START        = "2018-01-01"
END          = "2024-12-31"
INTERVAL     = Interval.DAILY
EXTENDED     = 100      # 向前扩展天数：保证 60 日滚动窗口因子的预热数据完整

# Walk-forward 时间划分（无信息泄露）
TRAIN_PERIOD = ("2018-01-01", "2021-12-31")  # 4 年训练集
VALID_PERIOD = ("2022-01-01", "2022-12-31")  # 1 年验证集（早停用）
TEST_PERIOD  = ("2023-01-01", "2024-12-31")  # 2 年样本外测试集

# 展示 IC 分析的代表性因子
DEMO_FACTORS = [
    "rsv_5",    # 5 日相对强弱值（KDJ 原始值），短期动量
    "roc_20",   # 20 日价格变化率，中期动量
    "std_20",   # 20 日收益率标准差，波动率因子
    "corr_20",  # 20 日量价相关系数，量价背离信号
    "ma_5",     # 5 日均线动量
    "beta_20",  # 20 日线性趋势（回归斜率）
]


def main():
    lab = AlphaLab(LAB_PATH)

    # ── 步骤 1：加载行情数据 ────────────────────────────────────────────────────
    print("=" * 60)
    print("[1/4] 加载行情数据")
    print("=" * 60)

    component_symbols = lab.load_component_symbols(INDEX_SYMBOL, START, END)
    print(f"成分股数量: {len(component_symbols)} 只")

    df: pl.DataFrame = lab.load_bar_df(
        vt_symbols    =component_symbols,
        interval      =INTERVAL,
        start         =START,
        end           =END,
        extended_days =EXTENDED,
    )
    if df is None or df.is_empty():
        print("错误：行情数据为空，请先运行 download_akshare.py")
        return

    print(f"行情数据规模: {df.shape[0]:,} 行 × {df.shape[1]} 列")
    print(f"列名: {df.columns}")
    print(df.head(3))

    # ── 步骤 2：构建 Alpha158 数据集 ───────────────────────────────────────────
    print("\n" + "=" * 60)
    print("[2/4] 构建 Alpha158 数据集（共 158 个量化因子）")
    print("=" * 60)
    print(f"训练集: {TRAIN_PERIOD[0]} ~ {TRAIN_PERIOD[1]}")
    print(f"验证集: {VALID_PERIOD[0]} ~ {VALID_PERIOD[1]}")
    print(f"测试集: {TEST_PERIOD[0]}  ~ {TEST_PERIOD[1]}")

    dataset: AlphaDataset = Alpha158(
        df,
        train_period=TRAIN_PERIOD,
        valid_period=VALID_PERIOD,
        test_period =TEST_PERIOD,
    )

    # 训练集处理管道
    dataset.add_processor("learn", partial(process_drop_na, names=["label"]))
    dataset.add_processor("learn", partial(process_cs_norm, names=["label"], method="zscore"))
    # 推断集处理管道（NaN 补 0，避免因股票上市不足导致因子缺失）
    dataset.add_processor("infer", partial(process_fill_na, fill_value=0))

    feature_names = list(dataset.feature_expressions.keys())
    print(f"\n已注册因子数量: {len(feature_names)}")
    print(f"因子列表（前 20 个）: {feature_names[:20]}")

    # ── 步骤 3：并行计算因子 ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("[3/4] 并行计算 158 个因子（max_workers=4）")
    print("=" * 60)

    # 加载成分过滤器：每个交易日只保留当时的指数成分股，防止幸存者偏差
    filters = lab.load_component_filters(INDEX_SYMBOL, START, END)
    print(f"成分过滤记录数: {len(filters)} 个交易日")

    dataset.prepare_data(filters, max_workers=4)
    dataset.process_data()

    # 立即保存，确保因子计算结果不丢失
    lab.save_dataset(DATASET_NAME, dataset)
    print("\n因子计算完成 ✓  数据集已保存")

    # 打印各时段数据量统计
    from vnpy.alpha import Segment
    for seg_name, seg in [("train", Segment.TRAIN), ("valid", Segment.VALID), ("test", Segment.TEST)]:
        try:
            seg_df = dataset.fetch_infer(seg)
            print(f"  {seg_name:5s}: {seg_df.shape[0]:>8,} 条样本")
        except Exception:
            pass

    # ── 步骤 4：单因子 IC 分析 ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("[4/4] 单因子 IC 信息系数分析")
    print("=" * 60)
    print("IC（信息系数）衡量因子值与未来收益的相关性")
    print("  |IC| > 0.03  — 有统计意义")
    print("  |ICIR| > 0.5 — 信号稳定性较好\n")

    for factor in DEMO_FACTORS:
        print(f"── {factor} " + "─" * (40 - len(factor)))
        try:
            dataset.show_feature_performance(factor)
        except Exception as e:
            print(f"   分析失败: {e}")
        print()

    # ── 保存数据集 ──────────────────────────────────────────────────────────────
    print("=" * 60)
    lab.save_dataset(DATASET_NAME, dataset)
    print(f"✓ 数据集已保存: {DATASET_NAME}")
    print(f"  路径: {LAB_PATH}/dataset/{DATASET_NAME}")
    print()
    print("下一步：训练预测模型")
    print("  Lasso:    from vnpy.alpha.model.models.lasso_model import LassoModel")
    print("  LightGBM: from vnpy.alpha.model.models.lgb_model   import LgbModel")
    print("  MLP:      from vnpy.alpha.model.models.mlp_model   import MlpModel")
    print()
    print("  参考: examples/alpha_research/research_workflow_lgb.ipynb")


if __name__ == "__main__":
    main()
