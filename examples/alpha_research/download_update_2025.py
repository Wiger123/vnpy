"""
CSI300 行情增量更新：通过 baostock 追加 2025-01-01 至今的日线数据

使用：
    PYTHONPATH=/Users/wiger-2/Documents/vnpy python download_update_2025.py
"""

import sys
sys.path.insert(0, "/Users/wiger-2/Documents/vnpy")

import warnings
warnings.filterwarnings("ignore")

from datetime import datetime, date
from pathlib import Path

import baostock as bs
import pandas as pd
import polars as pl
from tqdm import tqdm

from vnpy.trader.object import BarData
from vnpy.trader.constant import Exchange, Interval
from vnpy.trader.database import DB_TZ
from vnpy.alpha import AlphaLab, logger


LAB_PATH    = "/Users/wiger-2/Documents/vnpy/lab/csi300"
INDEX_SYM   = "000300.SSE"
NEW_START   = "2025-01-01"
NEW_END     = date.today().strftime("%Y-%m-%d")   # 今天


def to_bs_code(vt_symbol: str) -> str:
    code, exch = vt_symbol.split(".")
    return ("sh" if exch == "SSE" else "sz") + "." + code


def bs_df_to_bars(df: pd.DataFrame, symbol: str, exchange: Exchange) -> list[BarData]:
    bars = []
    for _, row in df.iterrows():
        try:
            dt = datetime.strptime(str(row["date"]), "%Y-%m-%d")
            dt = dt.replace(hour=15, tzinfo=DB_TZ)
            vol  = float(row.get("volume", 0) or 0) * 100   # 手→股
            turn = float(row.get("amount",  0) or 0)
            o = float(row.get("open",  0) or 0)
            h = float(row.get("high",  0) or 0)
            l = float(row.get("low",   0) or 0)
            c = float(row.get("close", 0) or 0)
            if c == 0:
                continue
            bars.append(BarData(
                symbol=symbol, exchange=exchange, datetime=dt,
                interval=Interval.DAILY,
                open_price=o, high_price=h, low_price=l, close_price=c,
                volume=vol, turnover=turn,
                open_interest=0.0, gateway_name="BAOSTOCK",
            ))
        except Exception:
            pass
    return bars


def main():
    lab = AlphaLab(LAB_PATH)

    # ── 1. 成分股列表（沿用已有快照）──────────────────────────────────────────
    syms = lab.load_component_symbols(INDEX_SYM, "2018-01-01", "2024-12-31")
    print(f"成分股: {len(syms)} 只")

    # 补充 2025-2026 成分快照（用当前成分）
    index_components: dict[str, list[str]] = {}
    for year in [2025, 2026]:
        for month in [1, 4, 7, 10]:
            key = f"{year}-{month:02d}-01"
            index_components[key] = syms
    lab.save_component_data(INDEX_SYM, index_components)
    print(f"成分快照已更新到 2026")

    # ── 2. 下载成分股 2025-今 日线（baostock 后复权）─────────────────────────
    print(f"\n[1/3] 下载成分股日线 {NEW_START} ~ {NEW_END}...")
    lg = bs.login()
    print("baostock:", lg.error_msg)

    failed: list[str] = []
    for vt_symbol in tqdm(syms, desc="日线"):
        symbol   = vt_symbol.split(".")[0]
        exchange = Exchange.SSE if vt_symbol.endswith("SSE") else Exchange.SZSE
        bs_code  = to_bs_code(vt_symbol)
        try:
            rs = bs.query_history_k_data_plus(
                bs_code,
                "date,open,high,low,close,volume,amount",
                start_date=NEW_START, end_date=NEW_END,
                frequency="d", adjustflag="2",   # 后复权
            )
            if rs.error_code != "0":
                failed.append(vt_symbol)
                continue
            rows = []
            while rs.next():
                rows.append(rs.get_row_data())
            if not rows:
                failed.append(vt_symbol)
                continue
            df = pd.DataFrame(rows, columns=rs.fields)
            for col in ["open","high","low","close","volume","amount"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            bars = bs_df_to_bars(df, symbol, exchange)
            if bars:
                lab.save_bar_data(bars)
        except Exception as e:
            logger.error(f"{vt_symbol}: {e}")
            failed.append(vt_symbol)

    print(f"成功: {len(syms)-len(failed)}  失败: {len(failed)}")

    # ── 3. 下载 CSI300 指数日线（基准用）─────────────────────────────────────
    print(f"\n[2/3] 下载 CSI300 指数日线 {NEW_START} ~ {NEW_END}...")
    try:
        rs = bs.query_history_k_data_plus(
            "sh.000300",
            "date,open,high,low,close,volume,amount",
            start_date=NEW_START, end_date=NEW_END,
            frequency="d", adjustflag="3",
        )
        rows = []
        while rs.next():
            rows.append(rs.get_row_data())
        df = pd.DataFrame(rows, columns=rs.fields)
        for col in ["open","high","low","close","volume","amount"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        bars = bs_df_to_bars(df, "000300", Exchange.SSE)
        lab.save_bar_data(bars)
        print(f"  指数: {len(bars)} 根K线")
    except Exception as e:
        print(f"  指数下载失败: {e}")

    bs.logout()

    # ── 4. 更新基本面数据（PE/PB/PS）────────────────────────────────────────
    print(f"\n[3/3] 更新基本面估值数据 {NEW_START} ~ {NEW_END}...")
    lg = bs.login()
    fund_dir = Path(LAB_PATH) / "fundamental"
    fund_dir.mkdir(exist_ok=True)

    new_records = []
    fund_failed = []
    for vt_symbol in tqdm(syms, desc="估值"):
        bs_code = to_bs_code(vt_symbol)
        try:
            rs = bs.query_history_k_data_plus(
                bs_code,
                "date,peTTM,pbMRQ,psTTM,pcfNcfTTM",
                start_date=NEW_START, end_date=NEW_END,
                frequency="d", adjustflag="3",
            )
            if rs.error_code != "0":
                fund_failed.append(vt_symbol)
                continue
            rows = []
            while rs.next():
                rows.append(rs.get_row_data())
            if not rows:
                fund_failed.append(vt_symbol)
                continue
            df = pd.DataFrame(rows, columns=rs.fields)
            df["vt_symbol"] = vt_symbol
            df["date"] = pd.to_datetime(df["date"])
            for col in ["peTTM","pbMRQ","psTTM","pcfNcfTTM"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            new_records.append(df[["date","vt_symbol","peTTM","pbMRQ","psTTM","pcfNcfTTM"]])
        except Exception as e:
            fund_failed.append(vt_symbol)

    bs.logout()

    if new_records:
        new_df  = pd.concat(new_records, ignore_index=True)
        new_df  = new_df.rename(columns={"peTTM":"pe_ttm","pbMRQ":"pb",
                                          "psTTM":"ps_ttm","pcfNcfTTM":"pcf_ttm"})
        new_pl  = pl.from_pandas(new_df).with_columns(
            pl.col("date").cast(pl.Datetime("us"))
        )
        old_path = fund_dir / "fundamental.parquet"
        if old_path.exists():
            old_pl = pl.read_parquet(str(old_path)).with_columns(
                pl.col("date").cast(pl.Datetime("us"))
            )
            merged = pl.concat([old_pl, new_pl]).unique(
                ["date","vt_symbol"], keep="last"
            ).sort(["vt_symbol","date"])
        else:
            merged = new_pl
        merged.write_parquet(str(old_path))
        print(f"  基本面: {len(merged):,} 行 → {old_path}")

    print(f"\n✓ 完成！新增日线: {NEW_START} ~ {NEW_END}")
    print("  下一步: python cta_factors.py")


if __name__ == "__main__":
    main()
