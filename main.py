import os
from typing import Optional, List

import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, Query
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


# ============================================================
# EQUITIES RESPONSE MODELS
# ============================================================

class EquityData(BaseModel):
    date: Optional[str] = None
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


class SingleMacroResponse(BaseModel):
    symbol: str
    found: bool
    data: Optional[MacroData] = None
    message: Optional[str] = None


class BatchMacroResponse(BaseModel):
    requested_symbols: List[str]
    count: int
    data: List[MacroData]


class RootResponse(BaseModel):
    status: str


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="US Equities API",
    version="1.0.0",
    servers=[
        {"url": "https://postgresql-us-equities-api.onrender.com"}
    ],
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
        elif key == "date":
            output[key] = str(value)
        else:
            output[key] = value

    return output


def dataframe_to_records(df: pd.DataFrame) -> list[dict]:
    records = []

    for _, row in df.iterrows():
        records.append(normalize_row(row.to_dict()))

    return records

    
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
        i.volatility_20d

    FROM public.us_equities p
    LEFT JOIN public.us_equities_indicators i
      ON p.ticker = i.ticker
     AND p.date = i.date
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
    "/equities/latest",
    operation_id="getLatestEquities",
    response_model=BatchEquityResponse,
)
def get_latest_equities(
    tickers: str = Query(
        ...,
        description="Comma-separated tickers, e.g. AAPL,NVDA,MSFT",
    )
):
    ticker_list = [
        t.strip().upper()
        for t in tickers.split(",")
        if t.strip()
    ]

    sql = """
    WITH latest AS (
        SELECT DISTINCT ON (p.ticker)
            p.date,
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
            i.volatility_20d

        FROM public.us_equities p
        LEFT JOIN public.us_equities_indicators i
          ON p.ticker = i.ticker
         AND p.date = i.date
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
    sql = """
    SELECT
        p.date,
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
        i.volatility_20d

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
    
# ============================================================
# MACRO ROUTES
# ============================================================

@app.get(
    "/macro/latest/{symbol}",
    operation_id="getLatestMacro",
    response_model=SingleMacroResponse,
)
def get_latest_macro(symbol: str):
    sql = """
    SELECT
        date,
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
        amplitude
    FROM public.macro
    WHERE symbol = :symbol
    ORDER BY date DESC
    LIMIT 1
    """

    df = pd.read_sql(
        text(sql),
        engine,
        params={"symbol": symbol.upper()},
    )

    if df.empty:
        return {
            "symbol": symbol.upper(),
            "found": False,
            "data": None,
            "message": "Macro symbol not found",
        }

    row = normalize_row(df.iloc[0].to_dict())

    return {
        "symbol": symbol.upper(),
        "found": True,
        "data": row,
        "message": None,
    }


@app.get(
    "/macro/latest",
    operation_id="getLatestMacros",
    response_model=BatchMacroResponse,
)
def get_latest_macros(
    symbols: str = Query(
        ...,
        description="Comma-separated symbols, e.g. ^GSPC,^TNX,BTC-USD,EURUSD=X",
    )
):
    symbol_list = [
        s.strip().upper()
        for s in symbols.split(",")
        if s.strip()
    ]

    sql = """
    WITH latest AS (
        SELECT DISTINCT ON (symbol)
            date,
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
            amplitude
        FROM public.macro
        WHERE symbol = ANY(:symbols)
        ORDER BY symbol, date DESC
    )
    SELECT *
    FROM latest
    ORDER BY symbol;
    """

    df = pd.read_sql(
        text(sql),
        engine,
        params={"symbols": symbol_list},
    )

    return {
        "requested_symbols": symbol_list,
        "count": len(df),
        "data": dataframe_to_records(df),
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
    sql = """
    SELECT
        date,
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
        amplitude
    FROM public.macro
    WHERE symbol = :symbol
      AND (:start_date IS NULL OR date >= CAST(:start_date AS DATE))
      AND (:end_date IS NULL OR date <= CAST(:end_date AS DATE))
    ORDER BY date DESC
    LIMIT :limit
    """

    df = pd.read_sql(
        text(sql),
        engine,
        params={
            "symbol": symbol.upper(),
            "start_date": start_date,
            "end_date": end_date,
            "limit": limit,
        },
    )

    return {
        "symbol": symbol.upper(),
        "count": len(df),
        "data": dataframe_to_records(df),
    }