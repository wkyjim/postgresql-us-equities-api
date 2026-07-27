import os
import sys
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
from fastapi import HTTPException

os.environ.setdefault("neon_password", "test-only")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import main


def test_latest_macro_returns_data_source_from_query_result():
    frame = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2026-07-08"),
                "observed_at": None,
                "symbol": "^GSPC",
                "close": 7000.0,
                "is_live": False,
                "data_source": "close",
            }
        ]
    )
    with patch.object(main.pd, "read_sql", return_value=frame):
        response = main.get_latest_macro("^gspc")

    assert response["found"] is True
    assert response["data"]["date"] == "2026-07-08 00:00:00"
    assert response["data"]["is_live"] is False
    assert response["data"]["data_source"] == "close"


def test_macro_batch_preserves_case_sensitive_investing_symbols():
    frame = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2026-07-24"),
                "observed_at": None,
                "symbol": "KOR200c1",
                "name": "KOSPI 200 Futures",
                "close": 1034.05,
                "is_live": False,
                "data_source": "close",
            },
            {
                "date": pd.Timestamp("2026-07-24"),
                "observed_at": None,
                "symbol": "CIHc1",
                "name": "SSE 50 Futures",
                "close": 2930.2,
                "is_live": False,
                "data_source": "close",
            },
        ]
    )
    with patch.object(main.pd, "read_sql", return_value=frame) as read_sql:
        response = main.get_latest_macros_batch("KOR200c1,CIHc1")

    params = read_sql.call_args.kwargs["params"]
    assert response["requested_symbols"] == ["KOR200c1", "CIHc1"]
    assert params["symbol_lookup"] == ["kor200c1", "cihc1"]
    assert response["data"][0]["symbol"] == "KOR200c1"


def test_live_macro_empty_response_directs_caller_to_fallback():
    with patch.object(main.pd, "read_sql", return_value=pd.DataFrame()):
        response = main.get_live_macro("^GSPC")

    assert response["found"] is False
    assert "/macro/latest" in response["message"]


def test_equity_custom_date_is_explicitly_close_only():
    frame = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2026-07-06"),
                "indicator_date": pd.Timestamp("2026-07-06"),
                "ticker": "SPY",
                "close": 700.0,
                "is_live": False,
                "data_source": "close",
            }
        ]
    )
    with patch.object(main.pd, "read_sql", return_value=frame):
        response = main.get_equity_by_date("spy", "2026-07-06")

    assert response["found"] is True
    assert response["data"]["ticker"] == "SPY"
    assert response["data"]["data_source"] == "close"
    assert response["data"]["indicator_date"] == "2026-07-06 00:00:00"


def test_equity_queries_require_same_date_indicator_join():
    frame = pd.DataFrame(
        [{"date": pd.Timestamp("2026-07-07"), "ticker": "AAPL", "close": 200.0}]
    )
    with patch.object(main.pd, "read_sql", return_value=frame) as read_sql:
        main.get_latest_equity("AAPL")

    query = str(read_sql.call_args.args[0])
    assert "AND i.date = p.date" in query
    assert "indicator.date <= p.date" not in query


def test_invalid_custom_date_returns_http_400():
    with pytest.raises(HTTPException) as exc:
        main.validate_iso_date("07/08/2026")

    assert exc.value.status_code == 400


def test_github_pages_origin_is_allowed_by_cors():
    cors = next(
        middleware
        for middleware in main.app.user_middleware
        if middleware.cls.__name__ == "CORSMiddleware"
    )

    assert "https://wkyjim.github.io" in cors.kwargs["allow_origins"]


def test_market_tape_groups_asia_futures_and_ust_bps():
    frame = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2026-07-08"),
                "observed_at": None,
                "symbol": "^HSI",
                "name": "Hang Seng",
                "close": 24000.0,
                "change": 120.0,
                "pct_chg": 0.5,
                "is_live": False,
                "data_source": "close",
            },
            {
                "date": pd.Timestamp("2026-07-08"),
                "observed_at": None,
                "symbol": "NQ=F",
                "name": "Nasdaq 100 Future",
                "close": 23000.0,
                "change": -50.0,
                "pct_chg": -0.2,
                "is_live": True,
                "data_source": "live",
            },
            {
                "date": pd.Timestamp("2026-07-08"),
                "observed_at": None,
                "symbol": "US10YT=X",
                "name": "United States 10-Year Treasury Yield",
                "close": 4.35,
                "change": 0.0425,
                "pct_chg": 0.99,
                "is_live": False,
                "data_source": "close",
            },
        ]
    )

    with patch.object(main.pd, "read_sql", return_value=frame):
        response = main.get_market_tape()

    assert response["count"] == 3
    assert response["groups"]["asia_equity_indices"][0]["symbol"] == "^HSI"
    assert response["groups"]["index_futures"][0]["symbol"] == "NQ=F"
    ust = response["groups"]["ust_yields"][0]
    assert ust["symbol"] == "US10YT=X"
    assert ust["display_name"] == "UST 10Y"
    assert ust["change_bps"] == 4.25


def test_equity_history_contains_ohlcv_fields_for_candlestick_chart():
    frame = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2026-07-08"),
                "indicator_date": pd.Timestamp("2026-07-08"),
                "ticker": "AAPL",
                "open": 190.0,
                "high": 195.0,
                "low": 188.0,
                "close": 194.0,
                "volume": 1000000,
                "is_live": False,
                "data_source": "close",
            }
        ]
    )

    with patch.object(main.pd, "read_sql", return_value=frame):
        response = main.get_equity_history("AAPL", start_date=None, end_date=None, limit=1)

    row = response["data"][0]
    assert row["open"] == 190.0
    assert row["high"] == 195.0
    assert row["low"] == 188.0
    assert row["close"] == 194.0
    assert row["volume"] == 1000000
