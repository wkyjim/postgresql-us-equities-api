import os
from datetime import date
from typing import Optional, List

import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from sqlalchemy import create_engine, text


# ============================================================
# ENV / DATABASE
# ============================================================

load_dotenv()

neon_password = os.getenv("neon_password")

if not neon_password:
    raise ValueError("Missing environment variable: neon_password")

NEON_DATABASE_URL = (
    f"postgresql://neondb_owner:{neon_password}"
    "@ep-aged-moon-ao3o4z0j-pooler.c-2.ap-southeast-1.aws.neon.tech/"
    "neondb?sslmode=require&channel_binding=require"
)

engine = create_engine(
    NEON_DATABASE_URL,
    pool_pre_ping=True,
    pool_recycle=3600,
)

MAX_BATCH_ITEMS = 50


# ============================================================
# EQUITIES RESPONSE MODELS
# ============================================================

class EquityData(BaseModel):
    date: Optional[str] = None
    indicator_date: Optional[str] = None
    ticker: Optional[str] = None
    name: Optional[str] = None
    market: Optional[str] = None

    open: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    close: Optional[float] = None
    change: Optional[float] = None
    pct_chg: Optional[float] = None
    prev_close: Optional[float] = None

    turnover: Optional[float] = None
    volume: Optional[float] = None
    mkt_cap: Optional[float] = None

    ytd_pct_chg: Optional[float] = None
    pe_ttm: Optional[float] = None
    amplitude: Optional[float] = None
    turnover_rate: Optional[float] = None

    ma_5: Optional[float] = None
    ma_20: Optional[float] = None
    ma_50: Optional[float] = None
    ma_100: Optional[float] = None
    ma_200: Optional[float] = None

    ema_12: Optional[float] = None
    ema_26: Optional[float] = None

    rsi_14: Optional[float] = None

    macd: Optional[float] = None
    macd_signal: Optional[float] = None
    macd_hist: Optional[float] = None

    atr_14: Optional[float] = None

    volume_ma_20: Optional[float] = None
    volume_ratio_20: Optional[float] = None

    high_52w: Optional[float] = None
    low_52w: Optional[float] = None

    return_5d: Optional[float] = None
    return_20d: Optional[float] = None
    return_60d: Optional[float] = None

    volatility_20d: Optional[float] = None
    is_live: bool = False
    data_source: str = "close"


class SingleEquityResponse(BaseModel):
    ticker: str
    found: bool
    data: Optional[EquityData] = None
    message: Optional[str] = None


class BatchEquityResponse(BaseModel):
    requested_tickers: List[str]
    count: int
    data: List[EquityData]

# ============================================================
# MACRO RESPONSE MODELS
# ============================================================

class MacroData(BaseModel):
    date: Optional[str] = None
    observed_at: Optional[str] = None
    symbol: Optional[str] = None
    name: Optional[str] = None
    asset_type: Optional[str] = None

    open: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    close: Optional[float] = None
    adj_close: Optional[float] = None
    volume: Optional[float] = None

    prev_close: Optional[float] = None
    change: Optional[float] = None
    pct_chg: Optional[float] = None
    amplitude: Optional[float] = None
    is_live: bool = False
    data_source: str = "close"


class SingleMacroResponse(BaseModel):
    symbol: str
    found: bool
    data: Optional[MacroData] = None
    message: Optional[str] = None


class BatchMacroResponse(BaseModel):
    requested_symbols: List[str]
    count: int
    data: List[MacroData]


class MarketTapeItem(MacroData):
    group: str
    display_name: str
    change_bps: Optional[float] = None


class MarketTapeResponse(BaseModel):
    count: int
    groups: dict[str, List[MarketTapeItem]]
    data: List[MarketTapeItem]


class RootResponse(BaseModel):
    status: str


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="US Equities API",
    version="1.1.0",
    servers=[
        {"url": "https://postgresql-us-equities-api.onrender.com"}
    ],
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://wkyjim.github.io",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ],
    allow_credentials=False,
    allow_methods=["GET"],
    allow_headers=["*"],
)


# ============================================================
# HELPERS
# ============================================================

def normalize_row(row: dict) -> dict:
    """
    Convert pandas/SQL values to JSON-safe values.
    """
    output = {}

    for key, value in row.items():
        if pd.isna(value):
            output[key] = None
        elif key in {"date", "indicator_date", "market_date", "observed_at"}:
            output[key] = str(value)
        else:
            output[key] = value

    return output


def dataframe_to_records(df: pd.DataFrame) -> list[dict]:
    records = []

    for _, row in df.iterrows():
        records.append(normalize_row(row.to_dict()))

    return records


def parse_csv_symbols(value: str, *, uppercase: bool = True) -> list[str]:
    items = [
        item.strip()
        for item in value.split(",")
        if item.strip()
    ]

    if uppercase:
        items = [item.upper() for item in items]

    # Preserve request order while removing duplicates before querying Neon.
    items = list(dict.fromkeys(items))

    if len(items) > MAX_BATCH_ITEMS:
        raise HTTPException(
            status_code=400,
            detail=f"Batch requests are limited to {MAX_BATCH_ITEMS} symbols",
        )

    return items


def normalize_macro_symbol(value: str) -> str:
    """Macro symbols can be case-sensitive in Neon, especially Investing.com codes."""
    return value.strip()


def macro_lookup_values(symbols: list[str]) -> list[str]:
    return [symbol.lower() for symbol in symbols]


def validate_iso_date(value: str) -> str:
    try:
        return date.fromisoformat(value).isoformat()
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail="Date must use YYYY-MM-DD format",
        ) from exc


MARKET_TAPE_GROUPS = {
    "asia_equity_indices": [
        ("^HSI", "Hang Seng"),
        ("^N225", "Nikkei 225"),
        ("^KS11", "KOSPI"),
        ("000001.SS", "SSE Composite"),
    ],
    "index_futures": [
        ("ES=F", "S&P 500 Future"),
        ("NQ=F", "Nasdaq 100 Future"),
        ("YM=F", "Dow Future"),
        ("RTY=F", "Russell 2000 Future"),
        ("HK50", "Hang Seng Future"),
        ("NIY=F", "Nikkei Future"),
        ("KOR200c1", "KOSPI 200 Future"),
        ("CIHc1", "SSE 50 Future"),
    ],
    "ust_yields": [
        ("US2YT=X", "UST 2Y"),
        ("US3YT=X", "UST 3Y"),
        ("US5YT=X", "UST 5Y"),
        ("US7YT=X", "UST 7Y"),
        ("US10YT=X", "UST 10Y"),
        ("US20YT=X", "UST 20Y"),
        ("US30YT=X", "UST 30Y"),
    ],
    "volatility": [
        ("^VIX", "VIX"),
        ("^SKEW", "CBOE SKEW"),
    ],
}


def market_tape_symbols() -> list[str]:
    symbols = []
    for rows in MARKET_TAPE_GROUPS.values():
        symbols.extend(symbol for symbol, _label in rows)
    return symbols


def market_tape_metadata() -> dict[str, tuple[str, str]]:
    metadata = {}
    for group, rows in MARKET_TAPE_GROUPS.items():
        for symbol, label in rows:
            metadata[symbol] = (group, label)
    return metadata


def add_market_tape_fields(row: dict, metadata: dict[str, tuple[str, str]]) -> dict:
    symbol = row.get("symbol")
    group, label = metadata.get(symbol, ("other", symbol or "Unknown"))
    enriched = dict(row)
    enriched["group"] = group
    enriched["display_name"] = label
    enriched["change_bps"] = None
    if group == "ust_yields" and enriched.get("change") is not None:
        enriched["change_bps"] = round(float(enriched["change"]) * 100, 4)
    return enriched


# ============================================================
# GENERAL ROUTES
# ============================================================

@app.get("/privacy", response_class=HTMLResponse)
def privacy_policy():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Privacy Policy</title>
    </head>
    <body>
        <h1>Privacy Policy</h1>

        <p>
            This API is used for financial market data retrieval and analysis.
        </p>

        <p>
            No personal user data is stored, sold, or shared.
        </p>

        <p>
            Requests may be logged temporarily for operational and debugging purposes.
        </p>

        <p>
            Contact: wkyjim@gmail.com
        </p>
    </body>
    </html>
    """


@app.get("/", response_model=RootResponse, operation_id="healthCheck")
def root():
    return {"status": "US Equities API running"}
    
# ============================================================
# EQUITIES ROUTES
# ============================================================

@app.get(
    "/equities/latest/{ticker}",
    operation_id="getLatestEquity",
    response_model=SingleEquityResponse,
)
def get_latest_equity(ticker: str):
    sql = """
    SELECT
        p.date,
        i.date AS indicator_date,
        p.ticker,
        p.name,
        p.market,

        p.open,
        p.high,
        p.low,
        p.close,
        p.change,
        p.pct_chg,
        p.prev_close,
        p.turnover,
        p.volume,
        p.mkt_cap,
        p.ytd_pct_chg,
        p.pe_ttm,
        p.amplitude,
        p.turnover_rate,

        i.ma_5,
        i.ma_20,
        i.ma_50,
        i.ma_100,
        i.ma_200,
        i.ema_12,
        i.ema_26,
        i.rsi_14,
        i.macd,
        i.macd_signal,
        i.macd_hist,
        i.atr_14,
        i.volume_ma_20,
        i.volume_ratio_20,
        i.high_52w,
        i.low_52w,
        i.return_5d,
        i.return_20d,
        i.return_60d,
        i.volatility_20d,
        FALSE AS is_live,
        'close' AS data_source

    FROM public.us_equities p
    LEFT JOIN public.us_equities_indicators i
      ON i.ticker = p.ticker
     AND i.date = p.date
    WHERE p.ticker = :ticker
    ORDER BY p.date DESC
    LIMIT 1
    """

    df = pd.read_sql(
        text(sql),
        engine,
        params={"ticker": ticker.upper()},
    )

    if df.empty:
        return {
            "ticker": ticker.upper(),
            "found": False,
            "data": None,
            "message": "Ticker not found",
        }

    row = normalize_row(df.iloc[0].to_dict())

    return {
        "ticker": ticker.upper(),
        "found": True,
        "data": row,
        "message": None,
    }


@app.get(
    "/equities/batch/latest",
    operation_id="getLatestEquitiesBatch",
    response_model=BatchEquityResponse,
)
def get_latest_equities_batch(
    tickers: str = Query(
        ...,
        description="Comma-separated tickers, e.g. AAPL,NVDA,MSFT",
    )
):
    ticker_list = parse_csv_symbols(tickers)

    sql = """
    WITH latest AS (
        SELECT DISTINCT ON (p.ticker)
            p.date,
            i.date AS indicator_date,
            p.ticker,
            p.name,
            p.market,
            p.open,
            p.high,
            p.low,
            p.close,
            p.change,
            p.pct_chg,
            p.prev_close,
            p.turnover,
            p.volume,
            p.mkt_cap,
            p.ytd_pct_chg,
            p.pe_ttm,
            p.amplitude,
            p.turnover_rate,

            i.ma_5,
            i.ma_20,
            i.ma_50,
            i.ma_100,
            i.ma_200,
            i.ema_12,
            i.ema_26,
            i.rsi_14,
            i.macd,
            i.macd_signal,
            i.macd_hist,
            i.atr_14,
            i.volume_ma_20,
            i.volume_ratio_20,
            i.high_52w,
            i.low_52w,
            i.return_5d,
            i.return_20d,
            i.return_60d,
            i.volatility_20d,
            FALSE AS is_live,
            'close' AS data_source

        FROM public.us_equities p
        LEFT JOIN public.us_equities_indicators i
          ON i.ticker = p.ticker
         AND i.date = p.date
        WHERE p.ticker = ANY(:tickers)
        ORDER BY p.ticker, p.date DESC
    )
    SELECT *
    FROM latest
    ORDER BY ticker;
    """

    df = pd.read_sql(
        text(sql),
        engine,
        params={"tickers": ticker_list},
    )

    return {
        "requested_tickers": ticker_list,
        "count": len(df),
        "data": dataframe_to_records(df),
    }

@app.get(
    "/equities/history/{ticker}",
    operation_id="getEquityHistory",
)
def get_equity_history(
    ticker: str,
    start_date: Optional[str] = Query(None, description="YYYY-MM-DD"),
    end_date: Optional[str] = Query(None, description="YYYY-MM-DD"),
    limit: int = Query(252, ge=1, le=5000),
):
    if start_date:
        start_date = validate_iso_date(start_date)
    if end_date:
        end_date = validate_iso_date(end_date)

    sql = """
    SELECT
        p.date,
        i.date AS indicator_date,
        p.ticker,
        p.name,
        p.market,

        p.open,
        p.high,
        p.low,
        p.close,
        p.change,
        p.pct_chg,
        p.prev_close,
        p.turnover,
        p.volume,
        p.mkt_cap,
        p.ytd_pct_chg,
        p.pe_ttm,
        p.amplitude,
        p.turnover_rate,

        i.ma_5,
        i.ma_20,
        i.ma_50,
        i.ma_100,
        i.ma_200,
        i.ema_12,
        i.ema_26,
        i.rsi_14,
        i.macd,
        i.macd_signal,
        i.macd_hist,
        i.atr_14,
        i.volume_ma_20,
        i.volume_ratio_20,
        i.high_52w,
        i.low_52w,
        i.return_5d,
        i.return_20d,
        i.return_60d,
        i.volatility_20d,
        FALSE AS is_live,
        'close' AS data_source

    FROM public.us_equities p
    LEFT JOIN public.us_equities_indicators i
      ON p.ticker = i.ticker
     AND p.date = i.date
    WHERE p.ticker = :ticker
      AND (:start_date IS NULL OR p.date >= CAST(:start_date AS DATE))
      AND (:end_date IS NULL OR p.date <= CAST(:end_date AS DATE))
    ORDER BY p.date DESC
    LIMIT :limit
    """

    df = pd.read_sql(
        text(sql),
        engine,
        params={
            "ticker": ticker.upper(),
            "start_date": start_date,
            "end_date": end_date,
            "limit": limit,
        },
    )

    return {
        "ticker": ticker.upper(),
        "count": len(df),
        "data": dataframe_to_records(df),
    }


@app.get(
    "/equities/date/{ticker}",
    operation_id="getEquityByDate",
    response_model=SingleEquityResponse,
)
def get_equity_by_date(
    ticker: str,
    date_value: str = Query(..., alias="date", description="YYYY-MM-DD"),
):
    requested_date = validate_iso_date(date_value)
    sql = """
    SELECT
        p.date, i.date AS indicator_date, p.ticker, p.name, p.market,
        p.open, p.high, p.low, p.close, p.change, p.pct_chg, p.prev_close,
        p.turnover, p.volume, p.mkt_cap, p.ytd_pct_chg, p.pe_ttm,
        p.amplitude, p.turnover_rate,
        i.ma_5, i.ma_20, i.ma_50, i.ma_100, i.ma_200,
        i.ema_12, i.ema_26, i.rsi_14,
        i.macd, i.macd_signal, i.macd_hist, i.atr_14,
        i.volume_ma_20, i.volume_ratio_20, i.high_52w, i.low_52w,
        i.return_5d, i.return_20d, i.return_60d, i.volatility_20d,
        FALSE AS is_live,
        'close' AS data_source
    FROM public.us_equities p
    LEFT JOIN public.us_equities_indicators i
      ON i.ticker = p.ticker
     AND i.date = p.date
    WHERE p.ticker = :ticker
      AND p.date = CAST(:date AS DATE)
    LIMIT 1
    """
    normalized_ticker = ticker.upper()
    df = pd.read_sql(
        text(sql),
        engine,
        params={"ticker": normalized_ticker, "date": requested_date},
    )
    if df.empty:
        return {
            "ticker": normalized_ticker,
            "found": False,
            "data": None,
            "message": f"No close data found for {requested_date}",
        }
    return {
        "ticker": normalized_ticker,
        "found": True,
        "data": normalize_row(df.iloc[0].to_dict()),
        "message": None,
    }
    
# ============================================================
# MACRO ROUTES
# ============================================================

@app.get(
    "/macro/latest/{symbol}",
    operation_id="getLatestMacro",
    response_model=SingleMacroResponse,
)
def get_latest_macro(symbol: str):
    requested_symbol = normalize_macro_symbol(symbol)
    sql = """
    WITH live_row AS (
        SELECT
            market_date AS date, observed_at, symbol, name, asset_type,
            open, high, low, close, adj_close, volume,
            prev_close, change, pct_chg, amplitude,
            TRUE AS is_live, 'live' AS data_source
        FROM public.macro_live
        WHERE lower(symbol) = :symbol_lookup
          AND is_market_closed = FALSE
        ORDER BY observed_at DESC
        LIMIT 1
    ),
    close_row AS (
        SELECT
            date, NULL::timestamptz AS observed_at, symbol, name, asset_type,
            open, high, low, close, adj_close, volume,
            prev_close, change, pct_chg, amplitude,
            FALSE AS is_live, 'close' AS data_source
        FROM public.macro
        WHERE lower(symbol) = :symbol_lookup
        ORDER BY date DESC
        LIMIT 1
    )
    SELECT * FROM live_row
    UNION ALL
    SELECT * FROM close_row
    WHERE NOT EXISTS (SELECT 1 FROM live_row)
    LIMIT 1
    """

    df = pd.read_sql(
        text(sql),
        engine,
        params={"symbol_lookup": requested_symbol.lower()},
    )

    if df.empty:
        return {
            "symbol": requested_symbol,
            "found": False,
            "data": None,
            "message": "Macro symbol not found",
        }

    row = normalize_row(df.iloc[0].to_dict())

    return {
        "symbol": row.get("symbol") or requested_symbol,
        "found": True,
        "data": row,
        "message": None,
    }


@app.get(
    "/macro/batch/latest",
    operation_id="getLatestMacrosBatch",
    response_model=BatchMacroResponse,
)
def get_latest_macros_batch(
    symbols: str = Query(
        ...,
        description="Comma-separated symbols, e.g. ^GSPC,^TNX,BTC-USD,EURUSD=X",
    )
):
    symbol_list = parse_csv_symbols(symbols, uppercase=False)
    symbol_lookup = macro_lookup_values(symbol_list)

    sql = """
    WITH live_latest AS (
        SELECT DISTINCT ON (symbol)
            market_date AS date, observed_at, symbol, name, asset_type,
            open, high, low, close, adj_close, volume,
            prev_close, change, pct_chg, amplitude,
            TRUE AS is_live, 'live' AS data_source
        FROM public.macro_live
        WHERE lower(symbol) = ANY(:symbol_lookup)
          AND is_market_closed = FALSE
        ORDER BY symbol, observed_at DESC
    ),
    close_latest AS (
        SELECT DISTINCT ON (symbol)
            date, NULL::timestamptz AS observed_at, symbol, name, asset_type,
            open, high, low, close, adj_close, volume,
            prev_close, change, pct_chg, amplitude,
            FALSE AS is_live, 'close' AS data_source
        FROM public.macro
        WHERE lower(symbol) = ANY(:symbol_lookup)
        ORDER BY symbol, date DESC
    ),
    latest AS (
        SELECT * FROM live_latest
        UNION ALL
        SELECT close_latest.*
        FROM close_latest
        WHERE NOT EXISTS (
            SELECT 1
            FROM live_latest
            WHERE live_latest.symbol = close_latest.symbol
        )
    )
    SELECT *
    FROM latest
    ORDER BY symbol;
    """

    df = pd.read_sql(
        text(sql),
        engine,
        params={"symbol_lookup": symbol_lookup},
    )

    return {
        "requested_symbols": symbol_list,
        "count": len(df),
        "data": dataframe_to_records(df),
    }


@app.get(
    "/market-tape",
    operation_id="getMarketTape",
    response_model=MarketTapeResponse,
)
def get_market_tape():
    symbol_list = market_tape_symbols()
    metadata = market_tape_metadata()
    sql = """
    WITH live_latest AS (
        SELECT DISTINCT ON (symbol)
            market_date AS date, observed_at, symbol, name, asset_type,
            open, high, low, close, adj_close, volume,
            prev_close, change, pct_chg, amplitude,
            TRUE AS is_live, 'live' AS data_source
        FROM public.macro_live
        WHERE symbol = ANY(:symbols)
          AND is_market_closed = FALSE
        ORDER BY symbol, observed_at DESC
    ),
    close_latest AS (
        SELECT DISTINCT ON (symbol)
            date, NULL::timestamptz AS observed_at, symbol, name, asset_type,
            open, high, low, close, adj_close, volume,
            prev_close, change, pct_chg, amplitude,
            FALSE AS is_live, 'close' AS data_source
        FROM public.macro
        WHERE symbol = ANY(:symbols)
        ORDER BY symbol, date DESC
    ),
    latest AS (
        SELECT * FROM live_latest
        UNION ALL
        SELECT close_latest.*
        FROM close_latest
        WHERE NOT EXISTS (
            SELECT 1
            FROM live_latest
            WHERE live_latest.symbol = close_latest.symbol
        )
    )
    SELECT *
    FROM latest
    ORDER BY symbol;
    """

    df = pd.read_sql(text(sql), engine, params={"symbols": symbol_list})
    records = [
        add_market_tape_fields(row, metadata)
        for row in dataframe_to_records(df)
    ]
    record_by_symbol = {row.get("symbol"): row for row in records}
    ordered_records = [
        record_by_symbol[symbol]
        for symbol in symbol_list
        if symbol in record_by_symbol
    ]

    groups = {}
    for group, rows in MARKET_TAPE_GROUPS.items():
        symbols = [symbol for symbol, _label in rows]
        groups[group] = [
            row for row in ordered_records if row.get("symbol") in symbols
        ]

    return {
        "count": len(ordered_records),
        "groups": groups,
        "data": ordered_records,
    }


@app.get(
    "/macro/live/{symbol}",
    operation_id="getLiveMacro",
    response_model=SingleMacroResponse,
)
def get_live_macro(symbol: str):
    normalized_symbol = normalize_macro_symbol(symbol)
    sql = """
    SELECT
        market_date AS date, observed_at, symbol, name, asset_type,
        open, high, low, close, adj_close, volume,
        prev_close, change, pct_chg, amplitude,
        TRUE AS is_live, 'live' AS data_source
    FROM public.macro_live
    WHERE lower(symbol) = :symbol_lookup
      AND is_market_closed = FALSE
    ORDER BY observed_at DESC
    LIMIT 1
    """
    df = pd.read_sql(text(sql), engine, params={"symbol_lookup": normalized_symbol.lower()})
    if df.empty:
        return {
            "symbol": normalized_symbol,
            "found": False,
            "data": None,
            "message": "No active live macro snapshot; use /macro/latest for close fallback",
        }
    row = normalize_row(df.iloc[0].to_dict())
    return {
        "symbol": row.get("symbol") or normalized_symbol,
        "found": True,
        "data": row,
        "message": None,
    }


@app.get(
    "/macro/batch/live",
    operation_id="getLiveMacrosBatch",
    response_model=BatchMacroResponse,
)
def get_live_macros_batch(
    symbols: str = Query(
        ...,
        description="Comma-separated symbols. Returns active live snapshots only.",
    )
):
    symbol_list = parse_csv_symbols(symbols, uppercase=False)
    symbol_lookup = macro_lookup_values(symbol_list)
    sql = """
    SELECT
        market_date AS date, observed_at, symbol, name, asset_type,
        open, high, low, close, adj_close, volume,
        prev_close, change, pct_chg, amplitude,
        TRUE AS is_live, 'live' AS data_source
    FROM public.macro_live
    WHERE lower(symbol) = ANY(:symbol_lookup)
      AND is_market_closed = FALSE
    ORDER BY symbol
    """
    df = pd.read_sql(text(sql), engine, params={"symbol_lookup": symbol_lookup})
    return {
        "requested_symbols": symbol_list,
        "count": len(df),
        "data": dataframe_to_records(df),
    }


@app.get(
    "/macro/date/{symbol}",
    operation_id="getMacroByDate",
    response_model=SingleMacroResponse,
)
def get_macro_by_date(
    symbol: str,
    date_value: str = Query(..., alias="date", description="YYYY-MM-DD"),
):
    requested_date = validate_iso_date(date_value)
    normalized_symbol = normalize_macro_symbol(symbol)
    sql = """
    SELECT
        date, NULL::timestamptz AS observed_at, symbol, name, asset_type,
        open, high, low, close, adj_close, volume,
        prev_close, change, pct_chg, amplitude,
        FALSE AS is_live, 'close' AS data_source
    FROM public.macro
    WHERE lower(symbol) = :symbol_lookup
      AND date = CAST(:date AS DATE)
    LIMIT 1
    """
    df = pd.read_sql(
        text(sql),
        engine,
        params={"symbol_lookup": normalized_symbol.lower(), "date": requested_date},
    )
    if df.empty:
        return {
            "symbol": normalized_symbol,
            "found": False,
            "data": None,
            "message": f"No close data found for {requested_date}",
        }
    row = normalize_row(df.iloc[0].to_dict())
    return {
        "symbol": row.get("symbol") or normalized_symbol,
        "found": True,
        "data": row,
        "message": None,
    }


@app.get(
    "/macro/history/{symbol}",
    operation_id="getMacroHistory",
)
def get_macro_history(
    symbol: str,
    start_date: Optional[str] = Query(None, description="YYYY-MM-DD"),
    end_date: Optional[str] = Query(None, description="YYYY-MM-DD"),
    limit: int = Query(252, ge=1, le=2000),
):
    normalized_symbol = normalize_macro_symbol(symbol)
    if start_date:
        start_date = validate_iso_date(start_date)
    if end_date:
        end_date = validate_iso_date(end_date)

    sql = """
    SELECT
        date,
        NULL::timestamptz AS observed_at,
        symbol,
        name,
        asset_type,
        open,
        high,
        low,
        close,
        adj_close,
        volume,
        prev_close,
        change,
        pct_chg,
        amplitude,
        FALSE AS is_live,
        'close' AS data_source
    FROM public.macro
    WHERE lower(symbol) = :symbol_lookup
      AND (:start_date IS NULL OR date >= CAST(:start_date AS DATE))
      AND (:end_date IS NULL OR date <= CAST(:end_date AS DATE))
    ORDER BY date DESC
    LIMIT :limit
    """

    df = pd.read_sql(
        text(sql),
        engine,
        params={
            "symbol_lookup": normalized_symbol.lower(),
            "start_date": start_date,
            "end_date": end_date,
            "limit": limit,
        },
    )

    return {
        "symbol": normalized_symbol,
        "count": len(df),
        "data": dataframe_to_records(df),
    }
