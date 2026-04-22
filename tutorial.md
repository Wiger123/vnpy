# VeighNa A股量化投研教程

本教程基于 VeighNa v4.3.0，完整演示如何利用 `vnpy.alpha` 模块完成 A 股多因子机器学习策略的全链路开发：**数据下载 → 因子计算 → 模型训练 → 信号生成 → 策略回测**。

---

## 目录

1. [环境安装](#1-环境安装)
2. [核心概念](#2-核心概念)
3. [获取 A 股历史数据](#3-获取-a-股历史数据)
4. [因子工程](#4-因子工程)
5. [模型训练](#5-模型训练)
6. [信号生成与评估](#6-信号生成与评估)
7. [策略回测](#7-策略回测)
8. [自定义扩展](#8-自定义扩展)
9. [完整示例脚本](#9-完整示例脚本)

---

## 1. 环境安装

### 1.1 基础依赖

```bash
# 建议使用 Python 3.10–3.13，推荐 conda 管理环境
conda create -n vnpy python=3.11
conda activate vnpy

# 安装 VeighNa 核心
pip install vnpy

# Alpha 研究模块额外依赖
pip install polars lightgbm scikit-learn torch plotly alphalens-reloaded tqdm

# TA-Lib（vnpy.trader 的必要依赖）
pip install TA-Lib
```

> **本地开发模式**：如果 vnpy 是从源码克隆而非 pip 安装，运行脚本时需要设置 PYTHONPATH：
> ```bash
> PYTHONPATH=/path/to/vnpy python your_script.py
> ```

### 1.2 数据服务接入

| 数据源 | 安装命令 | 是否免费 | 特点 |
|--------|----------|----------|------|
| **BaoStock** | `pip install baostock` | ✅ 完全免费 | 无需注册，专为 A 股设计，稳定可靠 |
| AKShare | `pip install akshare` | ✅ 免费 | 数据丰富但部分接口有频率限制 |
| TuShare | `pip install vnpy_tushare` | 积分制 | 免费注册，数据质量高 |
| RQData（米筐） | `pip install vnpy_rqdata` | 付费 | 覆盖全市场，机构首选 |
| 迅投研（XtQuant） | `pip install vnpy_xt` | 需开户 | 国内主流，支持实盘 |

> **推荐入门方案：BaoStock**  
> 无需注册账号，直接安装即可使用，本教程示例脚本均基于 BaoStock。  
> 完整可运行脚本见 [`examples/alpha_research/download_baostock.py`](./examples/alpha_research/download_baostock.py)。

---

## 2. 核心概念

`vnpy.alpha` 模块由四个核心组件构成：

```
AlphaLab          ← 统一管理数据/数据集/模型/信号的研究工作台
  ├── AlphaDataset  ← 因子特征工程（计算 + 预处理）
  ├── AlphaModel    ← 机器学习模型（训练 + 预测）
  └── BacktestingEngine  ← 策略回测引擎
```

数据在磁盘上以如下目录结构组织：

```
lab/csi300/
├── daily/          # 日线 Parquet 文件，每只股票一个文件
├── minute/         # 分钟线（同上）
├── component/      # 指数成分变化记录
├── dataset/        # 序列化的 AlphaDataset 对象
├── model/          # 序列化的 AlphaModel 对象
├── signal/         # 预测信号 Parquet 文件
└── contract.json   # 手续费/合约参数配置
```

---

## 3. 获取 A 股历史数据

> 完整可运行脚本：[`examples/alpha_research/download_baostock.py`](./examples/alpha_research/download_baostock.py)

### 3.1 初始化研究工作台

```python
from vnpy.alpha import AlphaLab

lab = AlphaLab("./lab/csi300")   # 首次运行自动创建目录
```

### 3.2 获取沪深 300 成分股列表

```python
import akshare as ak
from vnpy.trader.constant import Exchange

def infer_exchange(code: str) -> Exchange:
    """根据代码前缀推断 A 股交易所"""
    if code.startswith(("60", "68", "51", "11", "58")):
        return Exchange.SSE
    elif code.startswith(("83", "87", "43", "92")):
        return Exchange.BSE
    return Exchange.SZSE

# AKShare 获取当前沪深300成分股（无需账号）
comp_df = ak.index_stock_cons_csindex(symbol="000300")

component_symbols: list[str] = []
for _, row in comp_df.iterrows():
    code     = str(row["成分券代码"]).zfill(6)
    exchange = infer_exchange(code)
    component_symbols.append(f"{code}.{exchange.value}")

print(f"成分股数量: {len(component_symbols)} 只")   # 300 只

# 以当前快照覆盖全历史（简化处理；生产环境应追踪每次调仓变动以避免幸存者偏差）
index_components: dict[str, list[str]] = {}
for year in range(2018, 2025):
    for month in [1, 4, 7, 10]:
        index_components[f"{year}-{month:02d}-01"] = component_symbols

lab.save_component_data("000300.SSE", index_components)
```

### 3.3 下载历史行情（BaoStock，后复权）

```python
import baostock as bs
from datetime import datetime
from tqdm import tqdm

from vnpy.trader.object import BarData
from vnpy.trader.constant import Interval
from vnpy.trader.database import DB_TZ

def to_bs_code(code: str) -> str:
    """'600000' → 'sh.600000'，'000001' → 'sz.000001'"""
    prefix = "sh" if infer_exchange(code) == Exchange.SSE else "sz"
    return f"{prefix}.{code}"

def download_bars(bs_code: str, symbol: str, exchange: Exchange,
                  start: str = "2018-01-01", end: str = "2024-12-31") -> list[BarData]:
    rs = bs.query_history_k_data_plus(
        bs_code, "date,open,high,low,close,volume,amount",
        start_date=start, end_date=end,
        frequency="d", adjustflag="1",   # 后复权
    )
    bars = []
    while (rs.error_code == "0") and rs.next():
        date_str, open_, high, low, close, volume, amount = rs.get_row_data()
        if not close:
            continue
        dt = datetime.strptime(date_str, "%Y-%m-%d").replace(hour=15, tzinfo=DB_TZ)
        bars.append(BarData(
            symbol=symbol, exchange=exchange, datetime=dt,
            interval=Interval.DAILY,
            open_price=float(open_ or 0), high_price=float(high or 0),
            low_price=float(low or 0),    close_price=float(close or 0),
            volume=float(volume or 0),    # 单位：股（BaoStock 已换算）
            turnover=float(amount or 0),  # 单位：元
            open_interest=0.0, gateway_name="BAOSTOCK",
        ))
    return bars

bs.login()
for vt_symbol in tqdm(component_symbols):
    symbol   = vt_symbol.split(".")[0]
    exchange = infer_exchange(symbol)
    bars = download_bars(to_bs_code(symbol), symbol, exchange)
    if bars:
        lab.save_bar_data(bars)          # Parquet 存储，自动合并去重

# 同时下载沪深300指数本身（基准对比用）
idx_bars = download_bars("sh.000300", "000300", Exchange.SSE)
lab.save_bar_data(idx_bars)
bs.logout()
```

> **说明**  
> - BaoStock 的 `volume` 已是股数（不是"手"），`amount` 已是元，`vwap = amount / volume` 自动正确。  
> - AlphaLab 在 `load_bar_df()` 中计算 `vwap = turnover / volume`，单位匹配。

### 3.4 配置合约参数

```python
for vt_symbol in component_symbols:
    lab.add_contract_setting(
        vt_symbol,
        long_rate  =5  / 10000,   # 买入佣金 0.05‰
        short_rate =10 / 10000,   # 卖出：印花税 1‰ + 佣金 ≈ 0.1%
        size       =1,
        pricetick  =0.01,
    )
```

### 3.5 读取数据验证

```python
import polars as pl
from vnpy.trader.constant import Interval

# 批量加载为 Polars DataFrame（158 因子的计算输入）
df: pl.DataFrame = lab.load_bar_df(
    vt_symbols    =component_symbols[:5],
    interval      =Interval.DAILY,
    start         ="2023-01-01",
    end           ="2024-01-01",
    extended_days =60,       # 向前多取 60 天，保证 60 日滚动窗口完整
)
print(df)
# shape: (N, 10)
# 列: datetime, open, high, low, close, volume, turnover, open_interest, vwap, vt_symbol
```

---

## 4. 因子工程

### 4.1 Alpha158 因子集

VeighNa 内置了来自微软 Qlib 项目的 **158 个量化因子**，涵盖 K 线形态、价格趋势、时序波动、成交量等多个维度：

| 类别 | 代表因子 | 说明 |
|------|----------|------|
| K 线形态 | `kmid`, `klen`, `kup`, `klow` | 实体/影线比例 |
| 价格相对强度 | `open_0`, `high_0`, `vwap_0` | 当日开/高/均价与收盘价之比 |
| 时序动量 | `roc_5/10/20/30/60` | N 日价格变化率 |
| 移动平均 | `ma_5/10/20/30/60` | N 日均线与收盘价之比 |
| 时序波动 | `std_5/10/20/30/60` | N 日收盘价标准差 |
| 线性趋势 | `beta_5/10/20/30/60` | N 日线性回归斜率 |
| 量价相关 | `corr_5/10/20/30/60` | 价格与成交量的相关性 |
| 相对强弱 | `rsv_5/10/20/30/60` | RSV（KDJ 中的原始值）|

### 4.2 构建数据集

```python
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import polars as pl
from functools import partial

from vnpy.trader.constant import Interval
from vnpy.alpha import AlphaLab
from vnpy.alpha.dataset import (
    AlphaDataset,
    process_drop_na,
    process_cs_norm,
    process_fill_na,
)
from vnpy.alpha.dataset.datasets.alpha_158 import Alpha158

lab = AlphaLab("./lab/csi300")

index_symbol = "000300.SSE"
start        = "2008-01-01"
end          = "2023-12-31"
interval     = Interval.DAILY
extended_days = 100            # 为滚动窗口特征预留前置数据

# 加载全量成分股行情
component_symbols = lab.load_component_symbols(index_symbol, start, end)
df = lab.load_bar_df(component_symbols, interval, start, end, extended_days)

# 定义训练/验证/测试时段（Walk-forward 不泄露未来信息）
dataset: AlphaDataset = Alpha158(
    df,
    train_period = ("2008-01-01", "2014-12-31"),   # 训练集：7 年
    valid_period = ("2015-01-01", "2016-12-31"),   # 验证集：2 年（用于早停）
    test_period  = ("2017-01-01", "2020-08-31"),   # 测试集：3.5 年（样本外）
)
```

### 4.3 添加数据预处理器

```python
# 训练时：去除标签缺失行，对标签做截面 z-score 归一化
dataset.add_processor("learn", partial(process_drop_na, names=["label"]))
dataset.add_processor("learn", partial(process_cs_norm, names=["label"], method="zscore"))

# 推断时：特征缺失用 0 填充（避免因股票上市时间不足导致特征为 NaN）
dataset.add_processor("infer", partial(process_fill_na, fill_value=0))
```

### 4.4 计算因子（支持多进程）

```python
# 加载成分过滤器：每个交易日只保留当时的指数成分股
# 防止"幸存者偏差"——历史上已退市/调出指数的股票不会出现在对应日期的训练数据中
filters: dict[str, list[str]] = lab.load_component_filters(index_symbol, start, end)

# 并行计算 158 个因子（max_workers 根据 CPU 核数调整）
dataset.prepare_data(filters, max_workers=4)

# 应用预处理管道
dataset.process_data()
```

### 4.5 查看单因子表现

```python
# 以 RSV_5（5 日相对强弱值）为例，查看其 IC 信息系数
dataset.show_feature_performance("rsv_5")
```

### 4.6 自定义因子

除 Alpha158 外，可在任意 `AlphaDataset` 子类中用表达式语言添加自定义因子：

```python
from vnpy.alpha.dataset import AlphaDataset

class MyDataset(AlphaDataset):
    def __init__(self, df, train_period, valid_period, test_period):
        super().__init__(df, train_period, valid_period, test_period)

        # 内置时序函数：ts_mean / ts_std / ts_max / ts_min / ts_delay
        #              ts_rank / ts_slope / ts_corr / ts_rsquare / ts_resi
        #              ts_sum / ts_quantile / ts_argmax / ts_argmin
        # 内置截面函数：cs_rank / cs_mean / cs_std / cs_scale
        # 内置数学函数：abs / log / sign / pow1 / pow2

        # 5 日成交量加权价格动量
        self.add_feature("vwap_mom_5", "vwap / ts_delay(vwap, 5)")

        # 20 日截面量价相关（值越高说明近期涨价伴随放量）
        self.add_feature("vol_price_corr_20", "ts_corr(close, volume, 20)")

        # 自定义标签：未来 5 日对数收益率（默认标签已在 Alpha158 中定义）
        # self.set_label("ts_delay(close, -5) / close - 1")
```

### 4.7 保存数据集

```python
name = "300_lasso"
lab.save_dataset(name, dataset)  # 序列化到 lab/csi300/dataset/300_lasso
```

---

## 5. 模型训练

VeighNa 提供三种开箱即用的模型，均实现了统一的 `AlphaModel` 接口。

### 5.1 Lasso 回归（线性基准）

```python
import numpy as np
from vnpy.alpha import Segment, AlphaDataset, AlphaModel
from vnpy.alpha.model.models.lasso_model import LassoModel

dataset: AlphaDataset = lab.load_dataset(name)

model: AlphaModel = LassoModel(
    alpha=0.0005,      # L1 正则化系数，越大特征越稀疏
    max_iter=1000,
)
model.fit(dataset)

# 查看非零系数的因子及其权重（特征选择结果）
model.detail()

lab.save_model(name, model)
```

### 5.2 LightGBM（梯度提升树，推荐）

```python
from vnpy.alpha.model.models.lgb_model import LgbModel

model: AlphaModel = LgbModel(
    learning_rate=0.1,
    num_leaves=31,
    num_boost_round=200,
    early_stopping_rounds=20,  # 在验证集上早停，防止过拟合
    seed=42,
)
model.fit(dataset)

# 查看特征重要性排名
model.detail()
```

### 5.3 MLP 神经网络（深度学习）

```python
from vnpy.alpha.model.models.mlp_model import MlpModel

model: AlphaModel = MlpModel(
    hidden_sizes=[256, 128, 64],
    dropout=0.3,
    lr=1e-3,
    max_epochs=100,
    early_stopping_rounds=10,
)
model.fit(dataset)
```

### 5.4 模型对比方法

三个模型的接口完全一致，切换只需替换类名：

```python
# 同一套数据，快速横向对比三个模型
for ModelClass in [LassoModel, LgbModel, MlpModel]:
    model = ModelClass()
    model.fit(dataset)
    print(f"=== {ModelClass.__name__} ===")
    model.detail()
```

---

## 6. 信号生成与评估

### 6.1 生成预测信号

```python
model: AlphaModel = lab.load_model(name)
dataset: AlphaDataset = lab.load_dataset(name)

# 在测试集上预测（样本外，无未来信息泄露）
pre: np.ndarray = model.predict(dataset, Segment.TEST)

# 拼接为含 datetime / vt_symbol / signal 三列的 DataFrame
df_t: pl.DataFrame = dataset.fetch_infer(Segment.TEST)
df_t = df_t.with_columns(pl.Series(pre).alias("signal"))
signal: pl.DataFrame = df_t["datetime", "vt_symbol", "signal"]

print(signal.head())
```

### 6.2 信号质量评估

```python
# 计算 IC / ICIR / 分组收益等经典因子评价指标
dataset.show_signal_performance(signal)
```

### 6.3 保存信号

```python
lab.save_signal(name, signal)    # 存为 Parquet，供回测引擎读取
```

---

## 7. 策略回测

### 7.1 内置示例策略：EquityDemoStrategy

`EquityDemoStrategy` 是一个多空中性偏多头的选股策略，核心逻辑：

1. 每个交易日根据信号值排序，持有排名前 `top_k` 只股票
2. 持仓低于 `min_days` 天的股票不强制卖出
3. 每次最多换出 `n_drop` 只排名垫底的股票，控制换手率

```python
import importlib
from datetime import datetime

from vnpy.alpha.strategy import BacktestingEngine
import vnpy.alpha.strategy.strategies.equity_demo_strategy as equity_demo_strategy

# 热重载策略（Notebook 中修改策略后无需重启内核）
importlib.reload(equity_demo_strategy)
EquityDemoStrategy = equity_demo_strategy.EquityDemoStrategy

# 加载信号
signal = lab.load_signal(name)

# 配置回测引擎
engine = BacktestingEngine(lab)
engine.set_parameters(
    vt_symbols=component_symbols,          # 回测股票池
    interval=Interval.DAILY,
    start=datetime(2017, 1, 1),            # 与 test_period 对齐
    end=datetime(2020, 8, 1),
    capital=100_000_000,                   # 初始资金（1 亿）
)

# 添加策略及参数
setting = {
    "top_k": 30,          # 持有股票数量上限
    "n_drop": 3,          # 每次最多换出 3 只
    "hold_thresh": 3,     # 最短持有 3 天
}
engine.add_strategy(EquityDemoStrategy, setting, signal)
```

### 7.2 运行回测并查看结果

```python
engine.load_data()           # 从 lab 加载 K 线数据
engine.run_backtesting()     # 执行逐 K 线模拟撮合
engine.calculate_result()    # 汇总每日盈亏
engine.calculate_statistics()  # 计算绩效指标（Sharpe、最大回撤等）
engine.show_chart()          # 交互式净值曲线（需要 Plotly）

# 对比基准（沪深 300 指数）
engine.show_performance(benchmark_symbol=index_symbol)
```

回测引擎输出的核心指标包括：

| 指标 | 说明 |
|------|------|
| 总收益率 | 策略期间累计涨跌幅 |
| 年化收益率 | 折算为年化（按 240 交易日） |
| 最大回撤 | 净值从高点到最低点的最大跌幅 |
| Sharpe 比率 | 超额收益 / 波动率（越高越好） |
| 信息比率 | 超额收益 / 主动风险（同基准对比） |
| 胜率 | 盈利交易天数占比 |

### 7.3 解读策略参数影响

| 参数 | 增大效果 | 减小效果 |
|------|---------|---------|
| `top_k` | 分散风险，但 alpha 被稀释 | 集中持仓，收益波动加大 |
| `n_drop` | 换仓更快，追踪信号更及时 | 换手率低，持仓惰性更强 |
| `hold_thresh` | 减少噪声交易，降低换手费用 | 对信号反应更敏感 |

---

## 8. 自定义扩展

### 8.1 自定义策略

继承 `AlphaStrategy`，实现三个回调方法：

```python
from collections import defaultdict
import polars as pl
from vnpy.trader.object import BarData, TradeData
from vnpy.trader.constant import Direction
from vnpy.alpha import AlphaStrategy


class MySelectStrategy(AlphaStrategy):
    """自定义选股策略"""

    top_k: int = 20
    cash_ratio: float = 0.95

    def on_init(self) -> None:
        self.write_log("策略初始化")

    def on_trade(self, trade: TradeData) -> None:
        """成交回调：可在此更新持仓状态"""
        pass

    def on_bars(self, bars: dict[str, BarData]) -> None:
        """每个交易日调用一次，是策略逻辑的核心"""
        # 1. 获取当日最新信号，按分值降序排列
        signal: pl.DataFrame = self.get_signal()
        signal = signal.sort("signal", descending=True)

        # 2. 选取 top_k 只股票
        target_symbols = set(signal["vt_symbol"][:self.top_k])

        # 3. 平仓不在目标池中的持仓
        for vt_symbol, pos in self.pos_data.items():
            if pos and vt_symbol not in target_symbols:
                self.set_target(vt_symbol, 0)

        # 4. 等权买入目标股票
        cash = self.get_cash_available()
        buy_per_stock = cash * self.cash_ratio / self.top_k

        for vt_symbol in target_symbols:
            if self.get_pos(vt_symbol) == 0:
                bar = bars.get(vt_symbol)
                if bar and bar.close_price > 0:
                    volume = int(buy_per_stock / bar.close_price / 100) * 100
                    self.set_target(vt_symbol, volume)

        # 5. 统一执行下单（含滑点和手续费模拟）
        self.execute_trading(bars, price_add=0.05)
```

### 8.2 自定义模型

继承 `AlphaModel`，实现 `fit` / `predict` / `detail` 三个方法：

```python
import numpy as np
from sklearn.linear_model import Ridge
from vnpy.alpha import AlphaModel, AlphaDataset, Segment


class RidgeModel(AlphaModel):
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.model: Ridge | None = None

    def fit(self, dataset: AlphaDataset) -> None:
        X_train, y_train = dataset.fetch_learn("train")   # 获取训练集特征和标签
        X_valid, y_valid = dataset.fetch_learn("valid")

        self.model = Ridge(alpha=self.alpha)
        self.model.fit(X_train, y_train)

    def predict(self, dataset: AlphaDataset, segment: Segment) -> np.ndarray:
        X = dataset.fetch_infer(segment).drop("datetime", "vt_symbol").to_numpy()
        return self.model.predict(X)

    def detail(self) -> None:
        coef = self.model.coef_
        print(f"Ridge 系数范数: {np.linalg.norm(coef):.4f}")
```

---

## 9. 完整示例脚本

以下脚本将上述所有步骤串联为一个可直接运行的 Python 文件，适合在数据已就绪后快速复现完整流程：

```python
"""
vnpy_alpha_quickstart.py

前提：已完成数据下载（Step 1–3），lab 目录已存在
运行：python vnpy_alpha_quickstart.py
"""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import polars as pl
from functools import partial
from datetime import datetime

from vnpy.trader.constant import Interval
from vnpy.alpha import AlphaLab, Segment
from vnpy.alpha.dataset import process_drop_na, process_cs_norm, process_fill_na
from vnpy.alpha.dataset.datasets.alpha_158 import Alpha158
from vnpy.alpha.model.models.lgb_model import LgbModel
from vnpy.alpha.strategy import BacktestingEngine
from vnpy.alpha.strategy.strategies.equity_demo_strategy import EquityDemoStrategy


# ── 0. 配置 ──────────────────────────────────────────────────────────────────
LAB_PATH     = "./lab/csi300"
INDEX_SYMBOL = "000300.SSE"
NAME         = "300_lgb_demo"
START        = "2008-01-01"
END          = "2023-12-31"
INTERVAL     = Interval.DAILY
EXTENDED     = 100

TRAIN = ("2008-01-01", "2014-12-31")
VALID = ("2015-01-01", "2016-12-31")
TEST  = ("2017-01-01", "2020-08-31")


# ── 1. 初始化 lab ────────────────────────────────────────────────────────────
lab = AlphaLab(LAB_PATH)

component_symbols = lab.load_component_symbols(INDEX_SYMBOL, START, END)
print(f"[1] 成分股数量: {len(component_symbols)}")


# ── 2. 加载行情，构建因子数据集 ──────────────────────────────────────────────
df = lab.load_bar_df(component_symbols, INTERVAL, START, END, EXTENDED)
print(f"[2] 行情数据: {df.shape}")

dataset = Alpha158(df, train_period=TRAIN, valid_period=VALID, test_period=TEST)

dataset.add_processor("learn", partial(process_drop_na, names=["label"]))
dataset.add_processor("learn", partial(process_cs_norm, names=["label"], method="zscore"))
dataset.add_processor("infer", partial(process_fill_na, fill_value=0))

filters = lab.load_component_filters(INDEX_SYMBOL, START, END)
dataset.prepare_data(filters, max_workers=4)
dataset.process_data()

lab.save_dataset(NAME, dataset)
print("[2] 数据集已保存")


# ── 3. 训练 LightGBM 模型 ────────────────────────────────────────────────────
dataset = lab.load_dataset(NAME)

model = LgbModel(seed=42)
model.fit(dataset)
model.detail()

lab.save_model(NAME, model)
print("[3] 模型已保存")


# ── 4. 生成样本外预测信号 ─────────────────────────────────────────────────────
model   = lab.load_model(NAME)
dataset = lab.load_dataset(NAME)

pre    = model.predict(dataset, Segment.TEST)
df_t   = dataset.fetch_infer(Segment.TEST)
df_t   = df_t.with_columns(pl.Series(pre).alias("signal"))
signal = df_t["datetime", "vt_symbol", "signal"]

dataset.show_signal_performance(signal)
lab.save_signal(NAME, signal)
print("[4] 信号已保存")


# ── 5. 策略回测 ──────────────────────────────────────────────────────────────
signal = lab.load_signal(NAME)

engine = BacktestingEngine(lab)
engine.set_parameters(
    vt_symbols=component_symbols,
    interval=INTERVAL,
    start=datetime(2017, 1, 1),
    end=datetime(2020, 8, 1),
    capital=100_000_000,
)

setting = {"top_k": 30, "n_drop": 3, "hold_thresh": 3}
engine.add_strategy(EquityDemoStrategy, setting, signal)

engine.load_data()
engine.run_backtesting()
engine.calculate_result()
engine.calculate_statistics()
engine.show_chart()
engine.show_performance(benchmark_symbol=INDEX_SYMBOL)

print("[5] 回测完成")
```

---

## 参考资源

- **官方文档**：[vnpy.com/docs](https://www.vnpy.com/docs/cn/index.html)
- **社区论坛**：[vnpy.com/forum](https://www.vnpy.com/forum/)
- **Jupyter Notebooks**：[examples/alpha_research/](./examples/alpha_research/)
- **Qlib 因子参考**：[microsoft/qlib](https://github.com/microsoft/qlib)

---

*本教程基于 VeighNa v4.3.0 编写，如有 API 变更请以最新源码为准。*
