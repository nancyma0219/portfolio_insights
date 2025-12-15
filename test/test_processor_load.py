import pandas as pd
import pytest
from transaction_processor import TransactionProcessor

def test_load_missing_columns(tmp_path):
    # Create a CSV missing required columns
    p = tmp_path / "bad.csv"
    pd.DataFrame({"timestamp": ["2024-01-01"], "ticker": ["AAPL"]}).to_csv(p, index=False)

    proc = TransactionProcessor(str(p))
    with pytest.raises(ValueError):
        proc.load_data()

def test_load_ok(tmp_path):
    p = tmp_path / "ok.csv"
    pd.DataFrame({
        "timestamp": ["2024-01-01 10:00:00"],
        "ticker": ["AAPL"],
        "action": ["BUY"],
        "quantity": [10],
        "price": [100.0],
        "trader_id": ["t1"],
    }).to_csv(p, index=False)

    proc = TransactionProcessor(str(p))
    df = proc.load_data()
    assert len(df) == 1
