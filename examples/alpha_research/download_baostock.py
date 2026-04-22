"""
A 股历史数据下载脚本（基于 BaoStock，免费、无需注册、稳定）

- 成分股列表：AKShare (index_stock_cons_csindex)
- 历史行情：BaoStock (query_history_k_data_plus, 后复权)
- 存储格式：AlphaLab Parquet

使用方法：
    PYTHONPATH=/Users/wiger-2/Documents/vnpy python examples/alpha_research/download_baostock.py
"""

import warnings
warnings.filterwarnings("ignore")

from datetime import datetime

import akshare as ak
import baostock as bs
import pandas as pd
from tqdm import tqdm

from vnpy.trader.object import BarData
from vnpy.trader.constant import Exchange, Interval
from vnpy.trader.database import DB_TZ
from vnpy.alpha import AlphaLab, logger


# ── 配置 ──────────────────────────────────────────────────────────────────────
LAB_PATH     = "/Users/wiger-2/Documents/vnpy/lab/csi300"
INDEX_SYMBOL = "000300.SSE"
START_DATE   = "2018-01-01"
END_DATE     = "2024-12-31"
FIELDS       = "date,open,high,low,close,volume,amount"


# ── 工具函数 ───────────────────────────────────────────────────────────────────
def infer_exchange(code: str) -> Exchange:
    if code.startswith(("60", "68", "51", "11", "58")):
        return Exchange.SSE
    elif code.startswith(("83", "87", "43", "92")):
        return Exchange.BSE
    else:
        return Exchange.SZSE


def to_bs_code(code: str) -> str:
    """'600000' → 'sh.600000'，'000001' → 'sz.000001'"""
    prefix = "sh" if infer_exchange(code) == Exchange.SSE else "sz"
    return f"{prefix}.{code}"


def fetch_bars(bs_code: str, symbol: str, exchange: Exchange) -> list[BarData]:
    """BaoStock 查询单只股票日线并转为 list[BarData]"""
    rs = bs.query_history_k_data_plus(
        bs_code, FIELDS,
        start_date=START_DATE, end_date=END_DATE,
        frequency="d",
        adjustflag="1",   # 后复权
    )

    rows = []
    while (rs.error_code == "0") and rs.next():
        rows.append(rs.get_row_data())

    if not rows:
        return []

    bars: list[BarData] = []
    for row in rows:
        date_str, open_, high, low, close, volume, amount = row

        # BaoStock 在停牌日返回空字符串，过滤掉
        if not close or close == "":
            continue

        try:
            dt = datetime.strptime(date_str, "%Y-%m-%d")
            dt = dt.replace(hour=15, minute=0, second=0, tzinfo=DB_TZ)

            bar = BarData(
                symbol=symbol,
                exchange=exchange,
                datetime=dt,
                interval=Interval.DAILY,
                open_price =float(open_  or 0),
                high_price =float(high   or 0),
                low_price  =float(low    or 0),
                close_price=float(close  or 0),
                volume     =float(volume or 0),   # 已是股数
                turnover   =float(amount or 0),   # 已是元
                open_interest=0.0,
                gateway_name="BAOSTOCK",
            )
            bars.append(bar)
        except (ValueError, TypeError):
            continue

    return bars


# ── 主流程 ────────────────────────────────────────────────────────────────────
def main():
    lab = AlphaLab(LAB_PATH)

    # 1. 获取沪深 300 成分股列表（AKShare，组件列表 API 稳定）
    print("\n[1/5] 获取沪深 300 成分股列表（AKShare）...")
    comp_df = ak.index_stock_cons_csindex(symbol="000300")
    component_symbols: list[str] = []
    for _, row in comp_df.iterrows():
        code     = str(row["成分券代码"]).zfill(6)
        exchange = infer_exchange(code)
        component_symbols.append(f"{code}.{exchange.value}")
    print(f"    成分股数量: {len(component_symbols)} 只")

    # 2. 保存成分数据
    print("\n[2/5] 保存成分数据...")
    index_components: dict[str, list[str]] = {}
    for year in range(2018, 2025):
        for month in [1, 4, 7, 10]:
            key = f"{year}-{month:02d}-01"
            index_components[key] = component_symbols
    lab.save_component_data(INDEX_SYMBOL, index_components)
    print("    完成")

    # 3. 登录 BaoStock，下载成分股日线（后复权）
    print(f"\n[3/5] 登录 BaoStock，下载日线数据 ({START_DATE} ~ {END_DATE})...")
    bs.login()

    failed: list[str] = []
    for vt_symbol in tqdm(component_symbols, desc="下载进度"):
        code     = vt_symbol.split(".")[0]
        exchange = infer_exchange(code)
        bs_code  = to_bs_code(code)

        try:
            bars = fetch_bars(bs_code, code, exchange)
            if bars:
                lab.save_bar_data(bars)
            else:
                failed.append(vt_symbol)
        except Exception as e:
            logger.error(f"  {vt_symbol} 失败: {e}")
            failed.append(vt_symbol)

    # 3b. 下载沪深 300 指数本身（用于回测基准）
    print("\n    下载沪深 300 指数...")
    try:
        idx_bars = fetch_bars("sh.000300", "000300", Exchange.SSE)
        if idx_bars:
            lab.save_bar_data(idx_bars)
            print(f"    指数: {len(idx_bars)} 根 K 线")
        else:
            print("    指数数据为空")
    except Exception as e:
        print(f"    指数下载失败: {e}")

    bs.logout()
    print(f"\n    下载完成 ✓  成功: {len(component_symbols) - len(failed)} 只  失败: {len(failed)} 只")
    if failed:
        print(f"    失败列表（前 10）: {failed[:10]}")

    # 4. 合约参数
    print("\n[4/5] 写入合约参数（手续费率）...")
    for vt_symbol in component_symbols:
        lab.add_contract_setting(
            vt_symbol,
            long_rate  =5  / 10000,
            short_rate =10 / 10000,
            size       =1,
            pricetick  =0.01,
        )
    print("    完成")

    # 5. 验证：打印几只股票的数据量
    print("\n[5/5] 数据验证...")
    import polars as pl
    from pathlib import Path
    daily_path = Path(LAB_PATH) / "daily"
    files = sorted(daily_path.glob("*.parquet"))
    print(f"    Parquet 文件数: {len(files)}")
    for f in files[:3]:
        df = pl.read_parquet(f)
        print(f"    {f.name}: {len(df)} 行，{df['datetime'].min()} ~ {df['datetime'].max()}")

    print(f"\n✓ 全部完成！Lab 路径: {LAB_PATH}")
    print("  下一步: PYTHONPATH=... python examples/alpha_research/factor_engineering.py")


if __name__ == "__main__":
    main()
