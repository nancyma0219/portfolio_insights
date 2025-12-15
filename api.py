"""
Application API layer for Portfolio Insights Generator

This module acts as a thin orchestration layer between:
- raw transaction data (CSV input)
- the core processing pipeline (TransactionProcessor)
- and AI-driven insights generation (InsightsGenerator)

Goals:
- Load, clean, and analyze transaction data
- Expose reusable programmatic APIs for analytics and insights
- Invoke LLM-based insight generation in a controlled, aggregated manner
- Provide JSON-safe exports for downstream consumers (UI, CLI, future services)
- Keep this layer stateless and composable
- Make it callable from Streamlit, CLI, or a future FastAPI service
- Ensure analytics outputs are serialization-safe
"""


from __future__ import annotations

import json
from typing import Dict, Any, Tuple

import pandas as pd

from transaction_processor import TransactionProcessor
from insights_generator import InsightsGenerator


# ----------------------------
# Core processing APIs
# ----------------------------
def process_transactions(csv_path: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Load, clean, and analyze transactions from a CSV file.

    Args:
        csv_path: Path to transaction CSV.

    Returns:
        cleaned_df: Cleaned transaction DataFrame.
        analytics: Analytics dictionary.
    """
    processor = TransactionProcessor(csv_path)
    processor.load_data()
    processor.clean_data()
    processor.calculate_analytics()

    return processor.cleaned_df.copy(), processor.analytics


def generate_all_insights(
    csv_path: str,
    api_provider: str = "anthropic",
) -> Dict[str, str]:
    """
    Generate pattern and risk insights for a given CSV.

    Args:
        csv_path: Path to transaction CSV.
        api_provider: 'anthropic', 'openai', or 'local'.

    Returns:
        Dictionary with 'patterns' and 'risks' text.
    """
    _, analytics = process_transactions(csv_path)
    generator = InsightsGenerator(api_provider=api_provider)

    return {
        "patterns": generator.generate_pattern_insights(analytics),
        "risks": generator.generate_risk_insights(analytics),
    }


def generate_custom_insight(
    csv_path: str,
    question: str,
    api_provider: str = "anthropic",
) -> str:
    """
    Generate a custom insight for a user question.

    Args:
        csv_path: Path to transaction CSV.
        question: Custom question about the data.
        api_provider: 'anthropic', 'openai', or 'local'.

    Returns:
        Generated insight text.
    """
    if not question or not question.strip():
        raise ValueError("Question must be a non-empty string.")

    _, analytics = process_transactions(csv_path)
    generator = InsightsGenerator(api_provider=api_provider)

    return generator.generate_custom_insights(analytics, question)


# ----------------------------
# Export helpers
# ----------------------------
def export_analytics_json(csv_path: str) -> str:
    """
    Process transactions and return analytics as a JSON string.
    Ensures keys are JSON-compatible (e.g., date -> str) and NaN is converted to None.
    """
    import math
    import datetime

    _, analytics = process_transactions(csv_path)

    def _normalize(obj):
        # Pandas objects
        if hasattr(obj, "to_dict"):
            obj = obj.to_dict()

        # Tuples (e.g., date_range)
        if isinstance(obj, tuple):
            return [_normalize(x) for x in obj]

        # Dict: stringify keys + normalize values
        if isinstance(obj, dict):
            out = {}
            for k, v in obj.items():
                ks = _normalize_key(k)
                out[ks] = _normalize(v)
            return out

        # Lists
        if isinstance(obj, list):
            return [_normalize(x) for x in obj]

        # Datetime/date
        if isinstance(obj, (datetime.datetime, datetime.date)):
            return obj.isoformat()

        # Floats: convert NaN/Inf to None for strict JSON
        if isinstance(obj, float):
            if math.isnan(obj) or math.isinf(obj):
                return None
            return obj

        return obj

    def _normalize_key(k):
        import datetime
        # JSON keys must be str/int/float/bool/None; safest is str.
        if isinstance(k, (datetime.datetime, datetime.date)):
            return k.isoformat()
        return str(k)

    jsonable = _normalize(analytics)
    return json.dumps(jsonable, indent=2)



# ----------------------------
# CLI demo (optional)
# ----------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Portfolio Insights API Demo")
    parser.add_argument("csv", help="Path to transactions CSV")
    parser.add_argument("--provider", default="local", help="anthropic | openai | local")
    parser.add_argument("--question", default="", help="Optional custom question")

    args = parser.parse_args()

    print("Processing transactions...")
    df, analytics = process_transactions(args.csv)
    print(f"Cleaned rows: {len(df)}")
    print(f"Unique tickers: {analytics['unique_tickers']}")
    print(f"Total volume: ${float(analytics['total_volume']):,.2f}")

    gen = InsightsGenerator(api_provider=args.provider)

    print("\n=== PATTERN INSIGHTS ===")
    print(gen.generate_pattern_insights(analytics))

    print("\n=== RISK INSIGHTS ===")
    print(gen.generate_risk_insights(analytics))

    if args.question.strip():
        print("\n=== CUSTOM INSIGHT ===")
        print(gen.generate_custom_insights(analytics, args.question))
