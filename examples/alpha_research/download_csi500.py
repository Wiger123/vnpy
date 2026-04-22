"""
中证500数据下载脚本 —— OHLCV行情 + 基本面估值指标

使用：
    python download_csi500.py
"""

import time
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import akshare as ak
import pandas as pd
import polars as pl
from tqdm import tqdm

from vnpy.trader.object import BarData
from vnpy.trader.constant import Exchange, Interval
from vnpy.trader.database import DB_TZ
from vnpy.alpha import AlphaLab, logger


LAB_PATH     = "/Users/wiger-2/Documents/vnpy/lab/csi500"
INDEX_SYMBOL = "000905.SSE"
START_DATE   = "20180101"
END_DATE     = "20241231"
SLEEP_SEC    = 0.3


def infer_exchange(code: str) -> Exchange:
    if code.startswith(("60", "68", "51", "11", "58")):
        return Exchange.SSE
    elif code.startswith(("83", "87", "43", "92")):
        return Exchange.BSE
    return Exchange.SZSE


def ak_df_to_bars(df: pd.DataFrame, symbol: str, exchange: Exchange) -> list[BarData]:
    bars = []
    for _, row in df.iterrows():
        dt = pd.to_datetime(row["日期"]).to_pydatetime()
        dt = dt.replace(hour=15, minute=0, second=0, tzinfo=DB_TZ)
        volume   = float(row.get("成交量", 0) or 0) * 100
        turnover = float(row.get("成交额", 0) or 0)
        bar = BarData(
            symbol=symbol, exchange=exchange, datetime=dt,
            interval=Interval.DAILY,
            open_price =float(row.get("开盘", 0) or 0),
            high_price =float(row.get("最高", 0) or 0),
            low_price  =float(row.get("最低", 0) or 0),
            close_price=float(row.get("收盘", 0) or 0),
            volume=volume, turnover=turnover,
            open_interest=0.0, gateway_name="AKSHARE",
        )
        bars.append(bar)
    return bars


def main():
    lab = AlphaLab(LAB_PATH)
    fund_dir = Path(LAB_PATH) / "fundamental"
    fund_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. 中证500成分股 ──────────────────────────────────────────────────────
    print("\n[1/6] 获取中证500成分股...")
    comp_df = ak.index_stock_cons_csindex(symbol="000905")
    component_symbols: list[str] = []
    for _, row in comp_df.iterrows():
        code = str(row["成分券代码"]).zfill(6)
        component_symbols.append(f"{code}.{infer_exchange(code).value}")
    print(f"  成分股数量: {len(component_symbols)} 只")

    # ── 2. 保存成分快照 ───────────────────────────────────────────────────────
    print("[2/6] 保存成分快照...")
    index_components: dict[str, list[str]] = {}
    for year in range(2018, 2025):
        for month in [1, 4, 7, 10]:
            index_components[f"{year}-{month:02d}-01"] = component_symbols
    lab.save_component_data(INDEX_SYMBOL, index_components)
    print("  完成")

    # ── 3. 下载OHLCV行情（后复权）─────────────────────────────────────────────
    print(f"\n[3/6] 下载 {len(component_symbols)} 只日线行情（后复权）...")
    print(f"  时间范围: {START_DATE} ~ {END_DATE}")
    failed: list[str] = []
    for vt_symbol in tqdm(component_symbols, desc="行情"):
        symbol = vt_symbol.split(".")[0]
        exchange = infer_exchange(symbol)
        try:
            df = ak.stock_zh_a_hist(
                symbol=symbol, period="daily",
                start_date=START_DATE, end_date=END_DATE, adjust="hfq",
            )
            if df is None or df.empty:
                failed.append(vt_symbol)
                continue
            bars = ak_df_to_bars(df, symbol, exchange)
            if bars:
                lab.save_bar_data(bars)
        except Exception as e:
            logger.error(f"{vt_symbol} 行情失败: {e}")
            failed.append(vt_symbol)
        time.sleep(SLEEP_SEC)
    print(f"  成功: {len(component_symbols) - len(failed)}  失败: {len(failed)}")
    if failed:
        print(f"  失败列表(前10): {failed[:10]}")

    # ── 4. 下载基本面估值指标 ─────────────────────────────────────────────────
    # pe_ttm / pb / ps_ttm / dv_ratio(股息率) / total_mv(总市值)
    print(f"\n[4/6] 下载估值指标(PE/PB/PS/股息率/市值) —— 共 {len(component_symbols)} 只...")
    fund_records: list[pd.DataFrame] = []
    fund_failed: list[str] = []

    for vt_symbol in tqdm(component_symbols, desc="估值"):
        symbol = vt_symbol.split(".")[0]
        try:
            df = ak.stock_a_indicator_lg(symbol=symbol)
            if df is None or df.empty:
                fund_failed.append(vt_symbol)
                continue
            df = df.rename(columns={"trade_date": "date"})
            df["date"] = pd.to_datetime(df["date"])
            df = df[
                (df["date"] >= pd.Timestamp("2018-01-01")) &
                (df["date"] <= pd.Timestamp("2024-12-31"))
            ]
            df["vt_symbol"] = vt_symbol
            cols = ["date", "vt_symbol", "pe_ttm", "pb", "ps_ttm", "dv_ratio", "total_mv"]
            available = [c for c in cols if c in df.columns]
            fund_records.append(df[available])
        except Exception as e:
            logger.error(f"{vt_symbol} 估值失败: {e}")
            fund_failed.append(vt_symbol)
        time.sleep(SLEEP_SEC)

    if fund_records:
        fund_df = pd.concat(fund_records, ignore_index=True)
        # 数值列强制转float，避免object列
        for col in ["pe_ttm", "pb", "ps_ttm", "dv_ratio", "total_mv"]:
            if col in fund_df.columns:
                fund_df[col] = pd.to_numeric(fund_df[col], errors="coerce")

        fund_pl = pl.from_pandas(fund_df).with_columns(
            pl.col("date").cast(pl.Date)
        )
        out_path = fund_dir / "fundamental.parquet"
        fund_pl.write_parquet(str(out_path))
        print(f"  基本面数据: {len(fund_pl):,} 行 → {out_path}")
        print(f"  失败: {len(fund_failed)} 只")
    else:
        print("  警告：基本面数据全部下载失败")

    # ── 5. 下载中证500指数行情（基准用）──────────────────────────────────────
    print("\n[5/6] 下载中证500指数日线（基准）...")
    try:
        idx_df = ak.stock_zh_index_daily(symbol="sh000905")
        idx_df = idx_df.rename(columns={
            "date": "日期", "open": "开盘", "high": "最高",
            "low":  "最低", "close": "收盘", "volume": "成交量",
        })
        idx_df["成交额"] = 0.0
        idx_df = idx_df[idx_df["日期"].astype(str) >= "2018-01-01"]
        bars = ak_df_to_bars(idx_df, "000905", Exchange.SSE)
        lab.save_bar_data(bars)
        print(f"  完成，共 {len(bars)} 根K线")
    except Exception as e:
        print(f"  失败（不影响因子计算）: {e}")

    # ── 6. 合约参数 ───────────────────────────────────────────────────────────
    print("\n[6/6] 写入合约参数...")
    for vt_symbol in component_symbols:
        lab.add_contract_setting(
            vt_symbol,
            long_rate  =5  / 10000,
            short_rate =10 / 10000,
            size       =1,
            pricetick  =0.01,
        )
    print("  完成")

    print(f"\n✓ 全部完成！Lab路径: {LAB_PATH}")
    print("  下一步: python fundamental_factors.py")


if __name__ == "__main__":
    main()
