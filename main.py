import os
import json
import math
import urllib.error
import urllib.parse
import urllib.request
from datetime import date
from typing import Optional, List

import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException, Query, Request
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


class TelegramWebhookResponse(BaseModel):
    ok: bool
    handled: bool = False
    command: Optional[str] = None
    message: Optional[str] = None


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
        if isinstance(value, float) and not math.isfinite(value):
            output[key] = None
        elif value is None or (not isinstance(value, (dict, list, tuple)) and pd.isna(value)):
            output[key] = None
        elif key.endswith("_date") or key.endswith("_at") or key in {"date", "indicator_date", "market_date", "observed_at"}:
            output[key] = str(value)
        elif key == "regime_reason_json" and isinstance(value, str):
            try:
                output[key] = json.loads(value)
            except json.JSONDecodeError:
                output[key] = {"unparsed": value}
        else:
            output[key] = value

    return output


def dataframe_to_records(df: pd.DataFrame) -> list[dict]:
    records = []

    for _, row in df.iterrows():
        records.append(normalize_row(row.to_dict()))

    return records


SHORT_ANALYTICS_SORT_COLUMNS = {
    "ticker", "name", "sector", "industry", "analytics_date", "latest_price",
    "short_interest", "days_to_cover", "si_change_pct_latest", "si_slope_6m",
    "si_persistence_12m", "svr_5d", "svr_20d", "svr_z20", "casv_10d",
    "short_activity_score", "short_position_score", "funding_short_score",
    "funding_short_quality_score", "unwind_risk_score", "short_pressure_effectiveness",
    "rsi14", "price_vs_ma50_pct", "price_vs_ma200_pct", "regime_confidence",
}


def _short_sort_column(sort_by: str) -> str:
    if sort_by not in SHORT_ANALYTICS_SORT_COLUMNS:
        raise HTTPException(status_code=400, detail="Unsupported short-analytics sort field")
    return sort_by


def _short_filter_sql(
    *, ticker: str | None, sector: str | None, industry: str | None,
    security_type: str | None, regime: str | None, expected_si_direction: str | None,
    min_funding_short_score: float | None, min_funding_short_quality: float | None,
    min_short_activity_score: float | None, min_short_position_score: float | None,
    min_unwind_risk: float | None,
) -> tuple[str, dict]:
    clauses = []
    params: dict[str, object] = {}
    if security_type is None:
        clauses.append("security_type = 'common_stock'")
    elif security_type.lower() != "all":
        clauses.append("lower(security_type) = :security_type")
        params["security_type"] = security_type.lower()
    for field, value in (("sector", sector), ("industry", industry), ("short_regime", regime), ("expected_si_direction", expected_si_direction)):
        if value:
            clauses.append(f"lower({field}) = :{field}")
            params[field] = value.lower()
    if ticker:
        clauses.append("(ticker ILIKE :ticker OR name ILIKE :ticker)")
        params["ticker"] = f"%{ticker.strip()}%"
    thresholds = {
        "funding_short_score": min_funding_short_score,
        "funding_short_quality_score": min_funding_short_quality,
        "short_activity_score": min_short_activity_score,
        "short_position_score": min_short_position_score,
        "unwind_risk_score": min_unwind_risk,
    }
    for field, value in thresholds.items():
        if value is not None:
            clauses.append(f"{field} >= :{field}")
            params[field] = value
    return (" AND ".join(clauses) if clauses else "TRUE"), params


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
# TELEGRAM HELPERS
# ============================================================

TELEGRAM_MAX_MESSAGE_LENGTH = 3900


def telegram_env(name: str, *aliases: str) -> Optional[str]:
    value = os.getenv(name)
    if value:
        return value
    for alias in aliases:
        value = os.getenv(alias)
        if value:
            return value
    return None


def split_telegram_message(text: str, limit: int = TELEGRAM_MAX_MESSAGE_LENGTH) -> list[str]:
    text = text.strip()
    if not text:
        return ["No message content."]
    if len(text) <= limit:
        return [text]

    chunks: list[str] = []
    remaining = text
    while remaining:
        if len(remaining) <= limit:
            chunks.append(remaining)
            break
        cut = remaining.rfind("\n", 0, limit)
        if cut < int(limit * 0.6):
            cut = limit
        chunks.append(remaining[:cut].strip())
        remaining = remaining[cut:].strip()
    return chunks


def telegram_api_post(method: str, payload: dict) -> dict:
    token = telegram_env("TG_token", "TG_TOKEN")
    if not token:
        raise HTTPException(status_code=503, detail="TG_token is not configured")
    url = f"https://api.telegram.org/bot{token}/{method}"
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise HTTPException(status_code=502, detail=f"Telegram API error: {detail}") from exc
    except urllib.error.URLError as exc:
        raise HTTPException(status_code=502, detail=f"Telegram API connection failed: {exc}") from exc


def send_telegram_text(text: str, *, chat_id: str | None = None) -> int:
    target_chat_id = chat_id or telegram_env("TG_chat_id", "TG_CHAT_ID")
    if not target_chat_id:
        raise HTTPException(status_code=503, detail="TG_chat_id is not configured")
    sent_count = 0
    for chunk in split_telegram_message(text):
        telegram_api_post(
            "sendMessage",
            {
                "chat_id": target_chat_id,
                "text": chunk,
                "disable_web_page_preview": True,
            },
        )
        sent_count += 1
    return sent_count


def clean_markdown_text(text: str) -> str:
    text = text.replace("\r\n", "\n")
    text = text.replace("**", "")
    text = text.replace("`", "")
    return text.strip()


def fetch_text_url(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "market-dashboard-telegram-bot/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            return response.read().decode("utf-8", errors="replace")
    except Exception:
        return ""


def latest_report_markdown() -> str:
    report_url = telegram_env("TG_REPORT_URL", "LATEST_REPORT_URL")
    if report_url:
        return fetch_text_url(report_url)
    return ""


def normalize_heading(value: str) -> str:
    return " ".join(value.strip().lower().split())


def extract_markdown_section(markdown: str, heading: str) -> str:
    wanted = normalize_heading(heading)
    lines = markdown.splitlines()
    start: int | None = None
    start_level = 0
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if not stripped.startswith("##"):
            continue
        level = len(stripped) - len(stripped.lstrip("#"))
        title = stripped[level:].strip()
        if normalize_heading(title) == wanted:
            start = idx + 1
            start_level = level
            break
    if start is None:
        return ""
    end = len(lines)
    for idx in range(start, len(lines)):
        stripped = lines[idx].strip()
        if not stripped.startswith("##"):
            continue
        level = len(stripped) - len(stripped.lstrip("#"))
        if level <= start_level:
            end = idx
            break
    return "\n".join(lines[start:end]).strip()


def markdown_table_rows(section: str) -> list[dict[str, str]]:
    rows = [line.strip() for line in section.splitlines() if line.strip().startswith("|")]
    if len(rows) < 3:
        return []
    headers = [cell.strip() for cell in rows[0].strip("|").split("|")]
    parsed: list[dict[str, str]] = []
    for row in rows[2:]:
        cells = [cell.strip() for cell in row.strip("|").split("|")]
        if len(cells) == len(headers):
            parsed.append(dict(zip(headers, cells)))
    return parsed


def format_value(value, suffix: str = "") -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):,.2f}{suffix}"
    except (TypeError, ValueError):
        return str(value)


def build_report_dashboard(markdown: str) -> str:
    section = extract_markdown_section(markdown, "Executive Dashboard")
    if not section:
        return ""
    lines = [line.strip() for line in clean_markdown_text(section).splitlines() if line.strip()]
    return "Latest Executive Dashboard\n" + "\n".join(lines[:10])


def build_report_sectors(markdown: str) -> str:
    rows = markdown_table_rows(extract_markdown_section(markdown, "Official Sector Strength"))
    if not rows:
        return ""
    top = rows[0]
    bottom = rows[-1]
    lines = [
        "Top Sector Signals",
        f"- Highest score: {top.get('Sector', 'n/a')} ({top.get('Score', 'n/a')})",
        f"- Lowest score: {bottom.get('Sector', 'n/a')} ({bottom.get('Score', 'n/a')})",
        "",
        "Top 5 sectors:",
    ]
    for idx, row in enumerate(rows[:5], start=1):
        lines.append(f"- {idx}. {row.get('Sector', 'n/a')}: {row.get('Score', 'n/a')}")
    return "\n".join(lines)


def build_market_tape_message() -> str:
    response = get_market_tape()
    lines = ["Latest Market Snapshot"]
    for group, rows in response["groups"].items():
        if not rows:
            continue
        lines.append("")
        lines.append(group.replace("_", " ").title())
        for row in rows[:8]:
            change = row.get("change_bps") if group == "ust_yields" else row.get("pct_chg")
            suffix = " bps" if group == "ust_yields" else "%"
            lines.append(
                f"- {row.get('display_name')}: {format_value(row.get('close'))} "
                f"({format_value(change, suffix)}), {row.get('data_source', 'n/a')}"
            )
    return "\n".join(lines)


def build_macro_message() -> str:
    symbols = "^GSPC,^NDX,^DJI,^VIX,US2YT=X,US10YT=X,US30YT=X,GC=F,CL=F,BTC-USD"
    response = get_latest_macros_batch(symbols)
    lines = ["Latest Macro Summary"]
    for row in response["data"]:
        lines.append(
            f"- {row.get('symbol')}: {format_value(row.get('close'))} "
            f"({format_value(row.get('pct_chg'), '%')}), {row.get('data_source', 'n/a')}"
        )
    return "\n".join(lines)


def build_equity_message(ticker: str) -> str:
    if not ticker:
        return "Usage: /equity AAPL"
    response = get_latest_equity(ticker)
    if not response.get("found"):
        return response.get("message") or f"No data found for {ticker.upper()}."
    row = response["data"]
    return "\n".join(
        [
            f"Equity Snapshot: {ticker.upper()}",
            f"- Name: {row.get('name') or 'n/a'}",
            f"- Date: {row.get('date') or 'n/a'}",
            f"- Close: {format_value(row.get('close'))}",
            f"- 1D pct change: {format_value(row.get('pct_chg'), '%')}",
            f"- RSI 14: {format_value(row.get('rsi_14'))}",
            f"- 20D return: {format_value(row.get('return_20d'), '%')}",
            f"- 60D return: {format_value(row.get('return_60d'), '%')}",
            f"- Volume ratio 20D: {format_value(row.get('volume_ratio_20'))}",
        ]
    )


def build_status_message() -> str:
    lines = ["System Status", "- Render API: running"]
    try:
        tape = get_market_tape()
        lines.append(f"- Market tape rows: {tape['count']}")
    except Exception as exc:
        lines.append(f"- Market tape check failed: {type(exc).__name__}")
    report_url = telegram_env("TG_REPORT_URL", "LATEST_REPORT_URL")
    lines.append(f"- TG_REPORT_URL: {'configured' if report_url else 'not configured'}")
    lines.append(f"- TG_token: {'configured' if telegram_env('TG_token', 'TG_TOKEN') else 'not configured'}")
    lines.append(f"- TG_chat_id: {'configured' if telegram_env('TG_chat_id', 'TG_CHAT_ID') else 'not configured'}")
    lines.append(f"- TG_webhook_secret: {'configured' if telegram_env('TG_webhook_secret', 'TG_WEBHOOK_SECRET') else 'not configured'}")
    return "\n".join(lines)


TELEGRAM_START_TEXT = """Market Intelligence Bot

Commands:
/market - Show the latest market snapshot
/dashboard - Show the latest executive dashboard
/report - Show the latest full market report
/signals - Show latest deterministic setup signals
/sectors - Show top and bottom sector scores
/equity AAPL - Look up latest stock data and signals
/macro - Show latest macro summary
/risk - Show current market risk alerts
/status - Show database, API, and data update status
"""


def build_telegram_command_response(text_value: str) -> str:
    command_text = text_value.strip()
    command, _sep, arg_text = command_text.partition(" ")
    command = command.lower()
    markdown = latest_report_markdown()

    if command == "/start":
        return TELEGRAM_START_TEXT.strip()
    if command == "/market":
        return build_market_tape_message()
    if command == "/dashboard":
        return build_report_dashboard(markdown) or build_market_tape_message()
    if command == "/report":
        return clean_markdown_text(markdown) if markdown else "Latest report is unavailable. Configure TG_REPORT_URL on Render."
    if command == "/signals":
        section = extract_markdown_section(markdown, "Three-Month Outperformance Setup")
        rows = markdown_table_rows(section)[:8]
        if not rows:
            return "Signals are unavailable in the latest report."
        lines = ["Latest Rule-Based Setup Signals"]
        for row in rows:
            lines.append(f"- {row.get('Theme', 'n/a')}: {row.get('Classification', 'n/a')} ({row.get('Score', 'n/a')})")
        return "\n".join(lines)
    if command == "/sectors":
        return build_report_sectors(markdown) or "Sector ranking is unavailable in the latest report."
    if command == "/equity":
        return build_equity_message(arg_text.strip().upper())
    if command == "/macro":
        return build_macro_message()
    if command == "/risk":
        section = extract_markdown_section(markdown, "Volatility and Risk Signals")
        if not section:
            section = extract_markdown_section(markdown, "Contradiction / Audit Flags")
        return "Current Market Risk Alerts\n" + (clean_markdown_text(section) if section else "No risk section found in latest report.")
    if command == "/status":
        return build_status_message()
    return "Unknown command. Use /start for available commands."


def verify_telegram_webhook_secret(secret: str, header_secret: str | None) -> None:
    expected_path_secret = telegram_env("TG_webhook_secret", "TG_WEBHOOK_SECRET")
    if not expected_path_secret:
        raise HTTPException(status_code=503, detail="TG_webhook_secret is not configured")
    if secret != expected_path_secret:
        raise HTTPException(status_code=403, detail="Invalid Telegram webhook secret")

    expected_header_secret = telegram_env("TG_webhook_secret_token", "TG_WEBHOOK_SECRET_TOKEN")
    if expected_header_secret and header_secret != expected_header_secret:
        raise HTTPException(status_code=403, detail="Invalid Telegram webhook secret token")


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


@app.get(
    "/telegram/webhook/info",
    response_model=TelegramWebhookResponse,
    operation_id="telegramWebhookInfo",
)
def telegram_webhook_info():
    return {
        "ok": True,
        "handled": False,
        "command": None,
        "message": build_status_message(),
    }


@app.post(
    "/telegram/webhook/register/{secret}",
    response_model=TelegramWebhookResponse,
    operation_id="registerTelegramWebhook",
)
def register_telegram_webhook(
    secret: str,
    x_telegram_bot_api_secret_token: Optional[str] = Header(None),
):
    verify_telegram_webhook_secret(secret, x_telegram_bot_api_secret_token)
    base_url = telegram_env("TG_RENDER_BASE_URL", "RENDER_EXTERNAL_URL")
    if not base_url:
        raise HTTPException(status_code=503, detail="TG_RENDER_BASE_URL is not configured")
    webhook_url = f"{base_url.rstrip('/')}/telegram/webhook/{secret}"
    payload = {
        "url": webhook_url,
        "drop_pending_updates": True,
    }
    header_secret = telegram_env("TG_webhook_secret_token", "TG_WEBHOOK_SECRET_TOKEN")
    if header_secret:
        payload["secret_token"] = header_secret
    result = telegram_api_post("setWebhook", payload)
    return {
        "ok": bool(result.get("ok")),
        "handled": True,
        "command": "setWebhook",
        "message": "Telegram webhook registered." if result.get("ok") else str(result),
    }


@app.post(
    "/telegram/webhook/{secret}",
    response_model=TelegramWebhookResponse,
    operation_id="telegramWebhook",
)
async def telegram_webhook(
    secret: str,
    request: Request,
    x_telegram_bot_api_secret_token: Optional[str] = Header(None),
):
    verify_telegram_webhook_secret(secret, x_telegram_bot_api_secret_token)
    update = await request.json()
    message = update.get("message") or update.get("edited_message") or {}
    text_value = message.get("text")
    chat = message.get("chat") or {}
    chat_id = str(chat.get("id")) if chat.get("id") is not None else None
    if not text_value or not chat_id:
        return {
            "ok": True,
            "handled": False,
            "command": None,
            "message": "No text command found in update.",
        }

    allowed_chat_id = telegram_env("TG_chat_id", "TG_CHAT_ID")
    if allowed_chat_id and chat_id != str(allowed_chat_id):
        return {
            "ok": True,
            "handled": False,
            "command": text_value.split()[0],
            "message": "Ignored update from unauthorized chat.",
        }

    response_text = build_telegram_command_response(text_value)
    send_telegram_text(response_text, chat_id=chat_id)
    return {
        "ok": True,
        "handled": True,
        "command": text_value.split()[0],
        "message": "Command handled.",
    }


# ============================================================
# LATEST SHORT-POSITIONING SNAPSHOT ROUTES
# ============================================================

@app.get("/short-analytics/latest", operation_id="getLatestShortAnalytics")
def get_latest_short_analytics(
    limit: int = Query(50, ge=1, le=250),
    offset: int = Query(0, ge=0),
    ticker: Optional[str] = None,
    sector: Optional[str] = None,
    industry: Optional[str] = None,
    security_type: Optional[str] = None,
    regime: Optional[str] = None,
    expected_si_direction: Optional[str] = None,
    min_funding_short_score: Optional[float] = Query(None, ge=0, le=100),
    min_funding_short_quality: Optional[float] = Query(None, ge=0, le=100),
    min_short_activity_score: Optional[float] = Query(None, ge=0, le=100),
    min_short_position_score: Optional[float] = Query(None, ge=0, le=100),
    min_unwind_risk: Optional[float] = Query(None, ge=0, le=100),
    sort_by: str = "funding_short_score",
    sort_order: str = Query("desc", pattern="^(?i:asc|desc)$"),
):
    sort_column = _short_sort_column(sort_by)
    direction = "ASC" if sort_order.lower() == "asc" else "DESC"
    where, params = _short_filter_sql(
        ticker=ticker, sector=sector, industry=industry, security_type=security_type,
        regime=regime, expected_si_direction=expected_si_direction,
        min_funding_short_score=min_funding_short_score,
        min_funding_short_quality=min_funding_short_quality,
        min_short_activity_score=min_short_activity_score,
        min_short_position_score=min_short_position_score,
        min_unwind_risk=min_unwind_risk,
    )
    params.update({"limit": limit, "offset": offset})
    count = pd.read_sql(
        text(f"SELECT count(*) AS total FROM public.us_equities_short_analytics_latest WHERE {where}"),
        engine, params=params,
    )
    frame = pd.read_sql(
        text(
            f"""
            SELECT * FROM public.us_equities_short_analytics_latest
            WHERE {where}
            ORDER BY {sort_column} {direction} NULLS LAST, ticker ASC
            LIMIT :limit OFFSET :offset
            """
        ),
        engine, params=params,
    )
    return {
        "count": len(frame), "total": int(count.iloc[0]["total"]),
        "limit": limit, "offset": offset, "data": dataframe_to_records(frame),
    }


@app.get("/short-analytics/funding-shorts", operation_id="getTopFundingShorts")
def get_top_funding_shorts(
    limit: int = Query(25, ge=1, le=100),
    sector: Optional[str] = None,
    industry: Optional[str] = None,
    min_score: float = Query(60, ge=0, le=100),
    min_quality: float = Query(50, ge=0, le=100),
):
    return get_latest_short_analytics(
        limit=limit, offset=0, ticker=None, sector=sector, industry=industry,
        security_type="common_stock", regime=None, expected_si_direction=None,
        min_funding_short_score=min_score, min_funding_short_quality=min_quality,
        min_short_activity_score=None, min_short_position_score=None, min_unwind_risk=None,
        sort_by="funding_short_score", sort_order="desc",
    )


@app.get("/short-analytics/latest/{ticker}", operation_id="getLatestShortAnalyticsTicker")
def get_latest_short_analytics_ticker(ticker: str):
    symbol = ticker.strip().upper()
    frame = pd.read_sql(
        text("SELECT * FROM public.us_equities_short_analytics_latest WHERE ticker = :ticker LIMIT 1"),
        engine, params={"ticker": symbol},
    )
    if frame.empty:
        raise HTTPException(status_code=404, detail="Short analytics ticker not found")
    return {"ticker": symbol, "found": True, "data": normalize_row(frame.iloc[0].to_dict())}
    
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
