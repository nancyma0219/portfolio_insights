import pandas as pd
from transaction_processor import TransactionProcessor

def test_clean_standardizes_fields(tmp_path):
    p = tmp_path / "dirty.csv"
    pd.DataFrame({
        "timestamp": ["2024-01-01 10:00:00"],
        "ticker": [" aapl "],
        "action": [" buy "],
        "quantity": ["10"],
        "price": ["100.0"],
        "trader_id": [" t1 "],
    }).to_csv(p, index=False)

    proc = TransactionProcessor(str(p))
    proc.load_data()
    cleaned = proc.clean_data()

    assert cleaned.loc[0, "ticker"] == "AAPL"
    assert cleaned.loc[0, "action"] == "BUY"
    assert cleaned.loc[0, "trader_id"] == "T1"
    assert cleaned.loc[0, "quantity"] == 10
    assert cleaned.loc[0, "price"] == 100.0
    assert cleaned.loc[0, "total_value"] == 1000.0

def test_clean_drops_invalid_rows(tmp_path):
    # Includes:
    # - bad timestamp
    # - invalid action
    # - non-positive quantity
    p = tmp_path / "mixed.csv"
    pd.DataFrame({
        "timestamp": ["bad_time", "2024-01-01 10:00:00", "2024-01-01 11:00:00"],
        "ticker": ["AAPL", "AAPL", "MSFT"],
        "action": ["BUY", "HOLD", "SELL"],
        "quantity": [10, 10, 0],
        "price": [100, 100, 200],
        "trader_id": ["T1", "T1", "T2"],
    }).to_csv(p, index=False)

    proc = TransactionProcessor(str(p))
    proc.load_data()
    cleaned = proc.clean_data()

    # Only valid rows should remain (none in this example after filters)
    assert len(cleaned) == 0
