"""
模型训练、信号生成、策略回测脚本

基于已保存的 csi300_alpha158 数据集，完整跑通：
  1. 训练 LightGBM 模型
  2. 在测试集（2023-2024）生成预测信号
  3. 评估信号 IC 质量
  4. 用 EquityDemoStrategy 进行策略回测

使用方法：
    PYTHONPATH=/Users/wiger-2/Documents/vnpy \\
    python examples/alpha_research/model_training.py

前提：已执行 factor_engineering.py，数据集 csi300_alpha158 已保存。
"""

import matplotlib
matplotlib.use("Agg")   # 无显示器环境，不弹出图形窗口

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import polars as pl
from datetime import datetime

from vnpy.trader.constant import Interval
from vnpy.alpha import AlphaLab, Segment, AlphaDataset, AlphaModel
from vnpy.alpha.model.models.lgb_model import LgbModel
from vnpy.alpha.strategy import BacktestingEngine
from vnpy.alpha.strategy.strategies.equity_demo_strategy import EquityDemoStrategy


# ── 配置 ──────────────────────────────────────────────────────────────────────
LAB_PATH     = "/Users/wiger-2/Documents/vnpy/lab/csi300"
INDEX_SYMBOL = "000300.SSE"
DATASET_NAME = "csi300_alpha158"
MODEL_NAME   = "csi300_lgb"
SIGNAL_NAME  = "csi300_lgb"

# 与 factor_engineering.py 中 TEST_PERIOD 一致
TEST_START   = datetime(2023, 1, 1)
TEST_END     = datetime(2024, 12, 31)


def sep(title: str = "") -> None:
    print("\n" + "=" * 60)
    if title:
        print(title)
        print("=" * 60)


def main() -> None:
    lab = AlphaLab(LAB_PATH)

    # ── 步骤 1：加载数据集 ────────────────────────────────────────────────────
    sep("[1/4] 加载 Alpha158 数据集")
    dataset: AlphaDataset = lab.load_dataset(DATASET_NAME)
    if dataset is None:
        print("错误：数据集不存在，请先运行 factor_engineering.py")
        return

    for seg_name, seg in [("train", Segment.TRAIN), ("valid", Segment.VALID), ("test", Segment.TEST)]:
        df = dataset.fetch_infer(seg)
        print(f"  {seg_name:5s}: {df.shape[0]:>8,} 样本  ×  {df.shape[1]} 列")

    # ── 步骤 2：训练 LightGBM 模型 ────────────────────────────────────────────
    sep("[2/4] 训练 LightGBM 模型")
    model: AlphaModel = LgbModel(
        learning_rate       =0.05,
        num_leaves          =31,
        num_boost_round     =200,
        early_stopping_rounds=20,
        seed                =42,
    )
    model.fit(dataset)
    print("\n── 特征重要性（Top 20）──")
    model.detail()

    lab.save_model(MODEL_NAME, model)
    print(f"\n模型已保存: {MODEL_NAME}")

    # ── 步骤 3：生成预测信号（测试集，样本外）────────────────────────────────
    sep("[3/4] 生成测试集预测信号（2023-01-01 ~ 2024-12-31）")
    pre: np.ndarray          = model.predict(dataset, Segment.TEST)
    df_t: pl.DataFrame       = dataset.fetch_infer(Segment.TEST)
    df_t = df_t.with_columns(pl.Series(pre).alias("signal"))
    signal: pl.DataFrame     = df_t["datetime", "vt_symbol", "signal"]

    print(f"信号形状: {signal.shape}")
    print(f"信号统计: mean={signal['signal'].mean():.4f}  "
          f"std={signal['signal'].std():.4f}  "
          f"min={signal['signal'].min():.4f}  "
          f"max={signal['signal'].max():.4f}")
    print("\n── 信号 IC 分析 ──")
    try:
        dataset.show_signal_performance(signal)
    except Exception as e:
        print(f"  IC 分析失败: {e}")

    lab.save_signal(SIGNAL_NAME, signal)
    print(f"\n信号已保存: {SIGNAL_NAME}")

    # ── 步骤 4：策略回测 ──────────────────────────────────────────────────────
    sep("[4/4] 策略回测（EquityDemoStrategy，top_k=30）")

    component_symbols = lab.load_component_symbols(INDEX_SYMBOL, "2023-01-01", "2024-12-31")
    signal = lab.load_signal(SIGNAL_NAME)

    engine = BacktestingEngine(lab)
    engine.set_parameters(
        vt_symbols =component_symbols,
        interval   =Interval.DAILY,
        start      =TEST_START,
        end        =TEST_END,
        capital    =100_000_000,   # 初始资金 1 亿
    )

    setting = {
        "top_k"      : 30,   # 持有股票数量上限
        "n_drop"     : 3,    # 每次换出最多 3 只
        "hold_thresh": 3,    # 最短持有 3 天
    }
    engine.add_strategy(EquityDemoStrategy, setting, signal)

    engine.load_data()
    engine.run_backtesting()
    engine.calculate_result()
    engine.calculate_statistics()

    print("\n── 对比基准（沪深300指数）──")
    try:
        engine.show_performance(benchmark_symbol=INDEX_SYMBOL)
    except Exception as e:
        print(f"  基准对比失败: {e}")

    sep("完成")
    print("已生成文件：")
    print(f"  模型: {LAB_PATH}/model/{MODEL_NAME}")
    print(f"  信号: {LAB_PATH}/signal/{SIGNAL_NAME}.parquet")
    print("\n后续步骤：")
    print("  - 调整策略参数（top_k / n_drop / hold_thresh）")
    print("  - 尝试 Lasso / MLP 模型对比")
    print("  - 参考 examples/alpha_research/research_workflow_lgb.ipynb")


if __name__ == "__main__":
    main()
