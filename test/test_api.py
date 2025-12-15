import pandas as pd
import json
import pytest

from api import (
    process_transactions,
    generate_all_insights,
    generate_custom_insight,
    export_analytics_json,
)


def _write_sample_csv(tmp_path):
    p = tmp_path / "tx.csv"
    pd.DataFrame({
        "timestamp": ["2024-01-01 10:00:00", "2024-01-02 11:00:00"],
        "ticker": ["AAPL", "MSFT"],
        "action": ["BUY", "SELL"],
        "quantity": [10, 5],
        "price": [100.0, 200.0],
        "trader_id": ["T1", "T2"],
    }).to_csv(p, index=False)
    return str(p)


def test_process_transactions(tmp_path):
    csv_path = _write_sample_csv(tmp_path)
    df, analytics = process_transactions(csv_path)

    assert isinstance(df, pd.DataFrame)
    assert analytics["total_transactions"] == 2
    assert analytics["unique_tickers"] == 2
    assert "volume_by_ticker" in analytics


def test_generate_all_insights_local(tmp_path):
    csv_path = _write_sample_csv(tmp_path)
    out = generate_all_insights(csv_path, api_provider="local")

    assert "patterns" in out
    assert "risks" in out
    assert "Local Insights" in out["patterns"]


def test_generate_custom_insight(tmp_path):
    csv_path = _write_sample_csv(tmp_path)
    text = generate_custom_insight(
        csv_path,
        question="Is there concentration risk?",
        api_provider="local",
    )

    assert "Local Insights" in text


def test_generate_custom_insight_empty_question(tmp_path):
    csv_path = _write_sample_csv(tmp_path)
    with pytest.raises(ValueError):
        generate_custom_insight(csv_path, "", api_provider="local")


def test_export_analytics_json(tmp_path):
    csv_path = _write_sample_csv(tmp_path)
    s = export_analytics_json(csv_path)

    obj = json.loads(s)
    assert "daily_volume" in obj
    assert all(isinstance(k, str) for k in obj["daily_volume"].keys())
