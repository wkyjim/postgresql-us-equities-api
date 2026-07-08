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


def test_invalid_custom_date_returns_http_400():
    with pytest.raises(HTTPException) as exc:
        main.validate_iso_date("07/08/2026")

    assert exc.value.status_code == 400
