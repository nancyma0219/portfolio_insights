import pandas as pd
import plotly.graph_objects as go

from dashboard import (
    create_volume_chart,
    create_position_chart,
    create_trader_activity_chart,
    create_daily_volume_chart,
    process_uploaded_csv,
)


def _sample_csv_bytes() -> bytes:
    csv = """timestamp,ticker,action,quantity,price,trader_id
2024-01-01 10:00:00,AAPL,BUY,10,100,T1
2024-01-01 11:00:00,AAPL,SELL,4,110,T1
2024-01-02 10:00:00,MSFT,BUY,5,200,T2
"""
    return csv.encode("utf-8")


def test_process_uploaded_csv_returns_expected_shapes():
    cleaned_df, analytics = process_uploaded_csv(_sample_csv_bytes())

    assert isinstance(cleaned_df, pd.DataFrame)
    assert len(cleaned_df) == 3

    # Key analytics fields
    assert analytics["total_transactions"] == 3
    assert "volume_by_ticker" in analytics
    assert "net_position" in analytics
    assert "trader_activity" in analytics
    assert "daily_volume" in analytics


def test_chart_builders_return_figures():
    cleaned_df, analytics = process_uploaded_csv(_sample_csv_bytes())

    fig1 = create_volume_chart(analytics, top_k=2)
    fig2 = create_position_chart(analytics, top_k=2)
    fig3 = create_trader_activity_chart(analytics, top_k=2)
    fig4 = create_daily_volume_chart(analytics)

    assert isinstance(fig1, go.Figure)
    assert isinstance(fig2, go.Figure)
    assert isinstance(fig3, go.Figure)
    assert isinstance(fig4, go.Figure)
