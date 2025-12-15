import pandas as pd
from transaction_processor import TransactionProcessor

def test_analytics_basic(tmp_path):
    p = tmp_path / "ok.csv"
    pd.DataFrame({
        "timestamp": ["2024-01-01 10:00:00", "2024-01-01 11:00:00"],
        "ticker": ["AAPL", "AAPL"],
        "action": ["BUY", "SELL"],
        "quantity": [10, 4],
        "price": [100.0, 110.0],
        "trader_id": ["T1", "T1"],
    }).to_csv(p, index=False)

    proc = TransactionProcessor(str(p))
    proc.load_data()
    proc.clean_data()
    a = proc.calculate_analytics()

    assert a["total_transactions"] == 2
    # Volume by ticker is total notional
    # 10*100 + 4*110 = 1000 + 440 = 1440
    assert float(a["volume_by_ticker"].loc["AAPL"]) == 1440.0
    # Net position is shares: buys - sells = 10 - 4 = 6
    assert float(a["net_position"].loc["AAPL"]) == 6

def test_query_apis(tmp_path):
    p = tmp_path / "ok.csv"
    pd.DataFrame({
        "timestamp": ["2024-01-01 10:00:00", "2024-02-01 10:00:00"],
        "ticker": ["AAPL", "MSFT"],
        "action": ["BUY", "BUY"],
        "quantity": [10, 5],
        "price": [100.0, 200.0],
        "trader_id": ["T1", "T2"],
    }).to_csv(p, index=False)

    proc = TransactionProcessor(str(p))
    proc.load_data()
    proc.clean_data()

    assert len(proc.get_transactions_by_ticker("aapl")) == 1
    assert len(proc.get_trader_transactions("t2")) == 1
    assert len(proc.get_transactions_by_timerange("2024-01-01", "2024-01-31")) == 1
