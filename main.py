import os
from fastapi import FastAPI, Query
from sqlalchemy import create_engine, text
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

neon_password = os.getenv("neon_password")

app = FastAPI(
    title="US Equities API",
    servers=[
        {"url": "https://postgresql-us-equities-api.onrender.com/"}
    ]
)

# Use Render environment variable later
NEON_DATABASE_URL = (
    f"postgresql://neondb_owner:{neon_password}"
    "@ep-aged-moon-ao3o4z0j-pooler.c-2.ap-southeast-1.aws.neon.tech/"
    "neondb?sslmode=require&channel_binding=require"
)

engine = create_engine(
    NEON_DATABASE_URL,
    pool_pre_ping=True,
    pool_recycle=3600
)


@app.get("/")
def root():
    return {"status": "US Equities API running"}


@app.get("/equities/latest/{ticker}")
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
        params={"ticker": ticker.upper()}
    )

    if df.empty:
        return {
            "ticker": ticker.upper(),
            "found": False,
            "message": "Ticker not found"
        }

    row = df.iloc[0].where(pd.notnull(df.iloc[0]), None).to_dict()

    return {
        "ticker": ticker.upper(),
        "found": True,
        "data": row
    }


@app.get("/equities/latest")
def get_latest_equities(
    tickers: str = Query(..., description="Comma-separated tickers, e.g. AAPL,NVDA,MSFT")
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
        params={"tickers": ticker_list}
    )

    df = df.where(pd.notnull(df), None)

    return {
        "requested_tickers": ticker_list,
        "count": len(df),
        "data": df.to_dict(orient="records")
    }