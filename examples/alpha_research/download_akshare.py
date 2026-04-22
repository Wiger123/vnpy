"""
A 股历史数据下载脚本（基于 AKShare，无需注册）

使用方法：
    cd /Users/wiger-2/Documents/vnpy/examples/alpha_research
    python download_akshare.py
"""

import time
import warnings
warnings.filterwarnings("ignore")

from datetime import datetime

import akshare as ak
import pandas as pd
from tqdm import tqdm

from vnpy.trader.object import BarData
from vnpy.trader.constant import Exchange, Interval
from vnpy.trader.database import DB_TZ
from vnpy.alpha import AlphaLab, logger


# ── 配置 ──────────────────────────────────────────────────────────────────────
LAB_PATH     = "/Users/wiger-2/Documents/vnpy/lab/csi300"
INDEX_SYMBOL = "000300.SSE"
START_DATE   = "20180101"
END_DATE     = "20241231"
START_STR    = "2018-01-01"
SLEEP_SEC    = 0.3          # 每次 API 调用间隔（礼貌性限流）


# ── 工具函数 ───────────────────────────────────────────────────────────────────
def infer_exchange(code: str) -> Exchange:
    """根据代码前缀推断交易所"""
    if code.startswith(("60", "68", "51", "11", "58")):
        return Exchange.SSE
    elif code.startswith(("83", "87", "43", "92")):
        return Exchange.BSE
    else:
        return Exchange.SZSE


def ak_df_to_bars(
    df: pd.DataFrame,
    symbol: str,
    exchange: Exchange,
) -> list[BarData]:
    """将 AKShare DataFrame 转换为 list[BarData]

    注意：
      - AKShare 成交量单位是"手"（100 股），需乘 100 转为股数
      - AlphaLab.load_bar_df 内部计算 vwap = turnover / volume
        所以 volume 必须以"股"为单位，turnover 以"元"为单位
    """
    bars = []
    for _, row in df.iterrows():
        dt = pd.to_datetime(row["日期"]).to_pydatetime()
        dt = dt.replace(hour=15, minute=0, second=0, tzinfo=DB_TZ)

        volume   = float(row.get("成交量", 0) or 0) * 100  # 手 → 股
        turnover = float(row.get("成交额", 0) or 0)

        bar = BarData(
            symbol=symbol,
            exchange=exchange,
            datetime=dt,
            interval=Interval.DAILY,
            open_price =float(row.get("开盘", 0) or 0),
            high_price =float(row.get("最高", 0) or 0),
            low_price  =float(row.get("最低", 0) or 0),
            close_price=float(row.get("收盘", 0) or 0),
            volume     =volume,
            turnover   =turnover,
            open_interest=0.0,
            gateway_name="AKSHARE",
        )
        bars.append(bar)
    return bars


# ── 主流程 ────────────────────────────────────────────────────────────────────
def main():
    lab = AlphaLab(LAB_PATH)

    # ── 步骤 1：获取沪深 300 当前成分股 ────────────────────────────────────────
    print("\n[1/5] 获取沪深 300 成分股列表...")
    comp_df = ak.index_stock_cons_csindex(symbol="000300")
    # 列: 日期, 指数代码, 指数名称, ..., 成分券代码, 成分券名称, ..., 交易所

    component_symbols: list[str] = []
    for _, row in comp_df.iterrows():
        code     = str(row["成分券代码"]).zfill(6)
        exchange = infer_exchange(code)
        component_symbols.append(f"{code}.{exchange.value}")

    print(f"    成分股数量: {len(component_symbols)} 只")

    # ── 步骤 2：保存成分数据 ────────────────────────────────────────────────────
    # 使用当前成分作为历史快照（简化处理，生产环境应追踪每次调仓变动）
    print("\n[2/5] 保存成分数据（当前快照作为历史记录）...")
    index_components: dict[str, list[str]] = {}
    for year in range(2018, 2025):
        for month in [1, 4, 7, 10]:
            key = f"{year}-{month:02d}-01"
            index_components[key] = component_symbols
    lab.save_component_data(INDEX_SYMBOL, index_components)
    print("    完成")

    # ── 步骤 3：下载成分股日线行情（后复权）──────────────────────────────────────
    print(f"\n[3/5] 下载 {len(component_symbols)} 只成分股日线数据...")
    print(f"      时间范围: {START_DATE} ~ {END_DATE}，预计耗时约 {len(component_symbols) * SLEEP_SEC / 60:.1f} 分钟")

    failed: list[str] = []
    for vt_symbol in tqdm(component_symbols, desc="下载进度"):
        symbol   = vt_symbol.split(".")[0]
        exchange = infer_exchange(symbol)
        try:
            df = ak.stock_zh_a_hist(
                symbol=symbol,
                period="daily",
                start_date=START_DATE,
                end_date=END_DATE,
                adjust="hfq",   # 后复权，消除分红/送股对价格序列的影响
            )
            if df is None or df.empty:
                failed.append(vt_symbol)
                continue
            bars = ak_df_to_bars(df, symbol, exchange)
            if bars:
                lab.save_bar_data(bars)
        except Exception as e:
            logger.error(f"  {vt_symbol} 下载失败: {e}")
            failed.append(vt_symbol)

        time.sleep(SLEEP_SEC)

    print(f"\n    下载完成 ✓   成功: {len(component_symbols) - len(failed)} 只  失败: {len(failed)} 只")
    if failed:
        print(f"    失败列表（前10）: {failed[:10]}")

    # ── 步骤 4：下载沪深 300 指数日线（用于回测基准对比）────────────────────────
    print("\n[4/5] 下载沪深 300 指数日线...")
    try:
        idx_df = ak.stock_zh_index_daily(symbol="sz000300")
        # 列名: date, open, high, low, close, volume
        idx_df = idx_df.rename(columns={
            "date": "日期", "open": "开盘", "high": "最高",
            "low":  "最低", "close": "收盘", "volume": "成交量",
        })
        idx_df["成交额"] = 0.0
        idx_df = idx_df[idx_df["日期"].astype(str) >= "2018-01-01"]
        bars = ak_df_to_bars(idx_df, "000300", Exchange.SSE)
        lab.save_bar_data(bars)
        print(f"    完成，共 {len(bars)} 根 K 线")
    except Exception as e:
        print(f"    指数数据下载失败（不影响因子计算）: {e}")

    # ── 步骤 5：配置合约手续费参数 ─────────────────────────────────────────────
    print("\n[5/5] 写入合约参数（手续费率）...")
    for vt_symbol in component_symbols:
        lab.add_contract_setting(
            vt_symbol,
            long_rate  =5 / 10000,   # 买入佣金 0.05‰
            short_rate =10 / 10000,  # 卖出：印花税 1‰ + 佣金 ≈ 0.1%
            size       =1,
            pricetick  =0.01,
        )
    print("    完成")

    print(f"\n✓ 全部完成！Lab 路径: {LAB_PATH}")
    print("  下一步: python factor_engineering.py")


if __name__ == "__main__":
    main()
