"""
Transaction Processor Module

This module implements a stateful data-processing pipeline for financial
transaction data. It is responsible for loading raw transactions from CSV,
validating and cleaning the data, computing aggregated analytics, and exposing
query APIs for downstream consumers (API layer, dashboard, or LLM insights).

Inputs:
- CSV file path (string) pointing to a transactions dataset.
- The CSV must contain the following required columns:
  - timestamp: when the transaction occurred (string, parseable to datetime)
  - ticker: stock symbol (string)
  - action: BUY or SELL (string)
  - quantity: number of shares (numeric, > 0)
  - price: price per share (numeric, > 0)
  - trader_id: trader identifier (string)

Outputs:
- cleaned_df (pd.DataFrame):
  A normalized, validated, and time-sorted DataFrame containing only valid
  transactions. The cleaned data guarantees:
  - Parsed datetime timestamps
  - Standardized string fields (uppercased, stripped)
  - Valid BUY/SELL actions
  - Positive quantity and price values
  - No missing critical fields
  - Derived columns:
      - total_value = quantity * price
      - date = timestamp.date

- analytics (dict):
  A dictionary of aggregated portfolio statistics computed from cleaned_df,
  including (but not limited to):
  - Total transaction count and total traded notional
  - Volume traded per ticker
  - Net position per ticker (total BUY shares minus total SELL shares)
  - Trader activity metrics (transaction count and notional)
  - Time-based aggregates (daily volume)
  - Action distribution (BUY vs SELL)

Design Notes:
- Processing is split into explicit stages (load, clean, analyze) to improve
  testability, debuggability, and reuse.
- The processor maintains internal state and can be queried multiple times
  without reloading the CSV.
- Retrieval APIs are provided to efficiently access subsets of transactions
  by ticker, time range, or trader.
"""


from __future__ import annotations

import logging
from typing import Dict, Any, Optional

import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TransactionProcessor:
    """Process and analyze financial transaction data."""

    REQUIRED_COLS = ["timestamp", "ticker", "action", "quantity", "price", "trader_id"]

    def __init__(self, csv_path: str):
        """Initialize processor with CSV file path."""
        self.csv_path = csv_path
        self.df: Optional[pd.DataFrame] = None
        self.cleaned_df: Optional[pd.DataFrame] = None
        self.analytics: Dict[str, Any] = {}

    # ----------------------------
    # Step 1: Load
    # ----------------------------
    def load_data(self) -> pd.DataFrame:
        """Load CSV data and perform initial validation."""
        try:
            self.df = pd.read_csv(self.csv_path)
            logger.info("Loaded %d transactions", len(self.df))

            # Validate required columns
            missing_cols = [c for c in self.REQUIRED_COLS if c not in self.df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")

            return self.df

        except Exception as e:
            logger.error("Error loading data: %s", str(e))
            raise

    # ----------------------------
    # Step 2: Clean
    # ----------------------------
    def clean_data(self) -> pd.DataFrame:
        """Clean and prepare data for analysis."""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        df = self.df.copy()

        # Parse timestamp
        df["timestamp"] = pd.to_datetime(df["timestamp"],
                                         format="%Y-%m-%d %H:%M:%S",
                                         errors="coerce"
                                         )


        # Standardize string fields early
        df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
        df["action"] = df["action"].astype(str).str.strip().str.upper()
        df["trader_id"] = df["trader_id"].astype(str).str.strip().str.upper()

        # Coerce numeric fields
        df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
        df["price"] = pd.to_numeric(df["price"], errors="coerce")

        # Drop rows with missing critical fields (timestamp included)
        initial_count = len(df)
        df = df.dropna(subset=["timestamp", "ticker", "action", "quantity", "price"])
        dropped_missing = initial_count - len(df)
        if dropped_missing > 0:
            logger.warning("Dropped %d rows with missing critical values", dropped_missing)

        # Validate action values
        valid_actions_mask = df["action"].isin(["BUY", "SELL"])
        invalid_actions = (~valid_actions_mask).sum()
        if invalid_actions > 0:
            logger.warning("Removed %d rows with invalid actions", invalid_actions)
            df = df[valid_actions_mask]

        # Validate numeric constraints (optional but sensible)
        # Quantity and price should be positive for meaningful trades
        nonpositive_mask = (df["quantity"] <= 0) | (df["price"] <= 0)
        nonpositive_count = nonpositive_mask.sum()
        if nonpositive_count > 0:
            logger.warning("Removed %d rows with non-positive quantity/price", nonpositive_count)
            df = df[~nonpositive_mask]

        # Derived columns
        df["total_value"] = df["quantity"] * df["price"]
        df["date"] = df["timestamp"].dt.date

        # Sort by timestamp (helps time-range queries and charts)
        df = df.sort_values("timestamp").reset_index(drop=True)

        self.cleaned_df = df
        logger.info("Data cleaned. Final count: %d transactions", len(self.cleaned_df))

        return self.cleaned_df

    # ----------------------------
    # Step 3: Analytics
    # ----------------------------
    def calculate_analytics(self) -> Dict[str, Any]:
        """Calculate key analytics on the transaction data."""
        if self.cleaned_df is None:
            raise ValueError("Data not cleaned. Call clean_data() first.")

        df = self.cleaned_df

        # Total volume traded per ticker (in $ notional)
        volume_by_ticker = df.groupby("ticker")["total_value"].sum().sort_values(ascending=False)

        # Net position per ticker (shares: buys - sells)
        buys = df[df["action"] == "BUY"].groupby("ticker")["quantity"].sum()
        sells = df[df["action"] == "SELL"].groupby("ticker")["quantity"].sum()
        net_position = buys.sub(sells, fill_value=0).sort_values(ascending=False)

        # Most active traders (count + total $ notional)
        trader_activity = (
            df.groupby("trader_id")
            .agg({"timestamp": "count", "total_value": "sum"})
            .rename(columns={"timestamp": "transaction_count"})
            .sort_values("transaction_count", ascending=False)
        )

        # Time-based analysis
        daily_volume = df.groupby("date")["total_value"].sum().sort_index()

        # Action distribution
        action_counts = df["action"].value_counts()

        # Summary statistics
        total_transactions = len(df)
        total_volume = float(df["total_value"].sum())
        unique_tickers = int(df["ticker"].nunique())
        unique_traders = int(df["trader_id"].nunique())
        date_range = (df["timestamp"].min(), df["timestamp"].max())

        self.analytics = {
            "total_transactions": total_transactions,
            "total_volume": total_volume,
            "unique_tickers": unique_tickers,
            "unique_traders": unique_traders,
            "date_range": date_range,
            "volume_by_ticker": volume_by_ticker,
            "net_position": net_position,
            "trader_activity": trader_activity,
            "daily_volume": daily_volume,
            "action_counts": action_counts,
        }

        logger.info("Analytics calculated successfully")
        return self.analytics

    # ----------------------------
    # Retrieval APIs
    # ----------------------------
    def get_transactions_by_ticker(self, ticker: str) -> pd.DataFrame:
        """Retrieve all transactions for a specific ticker."""
        if self.cleaned_df is None:
            raise ValueError("Data not cleaned. Call clean_data() first.")
        t = str(ticker).strip().upper()
        return self.cleaned_df[self.cleaned_df["ticker"] == t].copy()

    def get_transactions_by_timerange(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Retrieve transactions within a date/time range (inclusive)."""
        if self.cleaned_df is None:
            raise ValueError("Data not cleaned. Call clean_data() first.")

        start = pd.to_datetime(start_date, errors="raise")
        end = pd.to_datetime(end_date, errors="raise")

        mask = (self.cleaned_df["timestamp"] >= start) & (self.cleaned_df["timestamp"] <= end)
        return self.cleaned_df[mask].copy()

    def get_trader_transactions(self, trader_id: str) -> pd.DataFrame:
        """Retrieve all transactions for a specific trader."""
        if self.cleaned_df is None:
            raise ValueError("Data not cleaned. Call clean_data() first.")
        tid = str(trader_id).strip().upper()
        return self.cleaned_df[self.cleaned_df["trader_id"] == tid].copy()

    # ----------------------------
    # Convenience: Summary
    # ----------------------------
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get high-level summary statistics for display."""
        if not self.analytics:
            self.calculate_analytics()

        volume_series = self.analytics.get("volume_by_ticker")
        trader_df = self.analytics.get("trader_activity")

        top_ticker = "N/A"
        if volume_series is not None and len(volume_series) > 0:
            top_ticker = str(volume_series.index[0])

        most_active_trader = "N/A"
        if trader_df is not None and len(trader_df) > 0:
            most_active_trader = str(trader_df.index[0])

        return {
            "total_transactions": self.analytics["total_transactions"],
            # FIXED formatting: use :,.2f
            "total_volume": f"${self.analytics['total_volume']:,.2f}",
            "unique_tickers": self.analytics["unique_tickers"],
            "unique_traders": self.analytics["unique_traders"],
            "date_range": f"{self.analytics['date_range'][0]} to {self.analytics['date_range'][1]}",
            "top_ticker_by_volume": top_ticker,
            "most_active_trader": most_active_trader,
        }


if __name__ == "__main__":
    processor = TransactionProcessor("data/sample_transactions.csv")
    processor.load_data()
    processor.clean_data()
    processor.calculate_analytics()
    print(processor.get_summary_stats())
