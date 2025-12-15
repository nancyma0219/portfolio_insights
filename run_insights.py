# Example usage:
# python run_insights.py \
#   --csv data/sample_transactions.csv \
#   --provider openai \
#   --prompt-type custom \
#   --question "What are the biggest concentration risks in this portfolio, and what should be reviewed next?" \
#   > insights_examples/openai_custom.txt (Optional step to save prompt and response as txt)


from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from typing import Any, Dict, Tuple

from transaction_processor import TransactionProcessor
from insights_generator import InsightsGenerator


def build_analytics(csv_path: str) -> Dict[str, Any]:
    """Run the full transaction pipeline and return analytics."""
    p = TransactionProcessor(csv_path)
    p.load_data()
    p.clean_data()
    p.calculate_analytics()
    return p.analytics


def analytics_summary_for_prompt(analytics: Dict[str, Any], top_n: int = 5) -> str:
    """
    Create a compact, privacy-preserving summary string for logging/prompt examples.
    Avoids dumping raw transactions or large tables.
    """
    lines = []
    lines.append(f"- Total transactions: {int(analytics.get('total_transactions', 0)):,}")
    lines.append(f"- Total notional: ${float(analytics.get('total_volume', 0.0)):,.2f}")
    lines.append(f"- Unique tickers: {int(analytics.get('unique_tickers', 0))}")
    lines.append(f"- Unique traders: {int(analytics.get('unique_traders', 0))}")

    dr = analytics.get("date_range")
    if isinstance(dr, (tuple, list)) and len(dr) == 2:
        lines.append(f"- Date range: {dr[0]} to {dr[1]}")

    # Top tickers by notional
    vbt = analytics.get("volume_by_ticker")
    if hasattr(vbt, "head"):
        top = vbt.head(top_n)
        top_str = ", ".join([f"{idx}=${float(val):,.0f}" for idx, val in top.items()])
        lines.append(f"- Top tickers by volume: {top_str}")

    # BUY/SELL distribution
    ac = analytics.get("action_counts")
    if hasattr(ac, "to_dict"):
        d = ac.to_dict()
        lines.append(f"- BUY/SELL counts: {d}")

    # Most active trader
    ta = analytics.get("trader_activity")
    if hasattr(ta, "head") and len(ta) > 0:
        # Expect columns: transaction_count, total_volume (per your processor)
        first = ta.iloc[0]
        trader_id = ta.index[0]
        tx_count = int(first.get("transaction_count", 0))
        vol = float(first.get("total_volume", 0.0))
        lines.append(f"- Most active trader: {trader_id} (tx={tx_count:,}, notional=${vol:,.0f})")

    return "\n".join(lines)


def make_prompt(prompt_type: str, summary: str, question: str | None = None) -> str:
    """Create a human-readable prompt text to save as an example."""
    if prompt_type == "patterns":
        return (
            "Generate concise pattern insights based on the following portfolio summary.\n\n"
            f"Summary:\n{summary}\n"
        )
    if prompt_type == "risks":
        return (
            "Identify potential risk flags (concentration, imbalance, anomalies) based on the following summary.\n\n"
            f"Summary:\n{summary}\n"
        )
    if prompt_type == "custom":
        q = question or "What are the main risks and what should be checked next?"
        return (
            "Answer the user's question using only the summary below. If information is missing, state what is needed.\n\n"
            f"Question:\n{q}\n\n"
            f"Summary:\n{summary}\n"
        )
    raise ValueError("prompt_type must be one of: patterns, risks, custom")


def save_example_md(
    out_path: str,
    provider: str,
    model: str | None,
    prompt_type: str,
    prompt_text: str,
    response_text: str,
) -> None:
    """Append a prompt/response example to a markdown file."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(out_path, "a", encoding="utf-8") as f:
        f.write("\n---\n")
        f.write(f"## Example ({ts})\n\n")
        f.write(f"- Provider: `{provider}`\n")
        if model:
            f.write(f"- Model: `{model}`\n")
        f.write(f"- Prompt type: `{prompt_type}`\n\n")
        f.write("### Prompt\n\n```text\n")
        f.write(prompt_text.rstrip() + "\n")
        f.write("```\n\n")
        f.write("### Response\n\n```text\n")
        f.write(response_text.rstrip() + "\n")
        f.write("```\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run InsightsGenerator and print/save outputs.")
    parser.add_argument("--csv", default="sample_transactions.csv", help="Path to transactions CSV")
    parser.add_argument(
        "--provider",
        default="local",
        choices=["local", "anthropic", "openai"],
        help="Insights provider",
    )
    parser.add_argument(
        "--prompt-type",
        default="all",
        choices=["patterns", "risks", "custom", "all"],
        help="Which insight prompt to run",
    )
    parser.add_argument("--question", default="", help="Custom question (used for prompt-type=custom)")
    parser.add_argument(
        "--save-md",
        default="",
        help="If set, append prompt/response examples to this markdown file",
    )
    parser.add_argument("--top-n", type=int, default=5, help="Top-N tickers to include in the summary")
    args = parser.parse_args()

    analytics = build_analytics(args.csv)
    summary = analytics_summary_for_prompt(analytics, top_n=args.top_n)

    gen = InsightsGenerator(api_provider=args.provider)
    model = getattr(gen, "model", None)

    def run_one(ptype: str) -> Tuple[str, str]:
        prompt_text = make_prompt(ptype, summary, question=args.question if args.question else None)

        if ptype == "patterns":
            resp = gen.generate_pattern_insights(analytics)
        elif ptype == "risks":
            resp = gen.generate_risk_insights(analytics)
        else:
            q = args.question or "What are the main risks and what should be checked next?"
            resp = gen.generate_custom_insights(analytics, q)

        return prompt_text, resp

    prompt_types = ["patterns", "risks", "custom"] if args.prompt_type == "all" else [args.prompt_type]

    for ptype in prompt_types:
        prompt_text, resp = run_one(ptype)

        print("\n" + "=" * 90)
        print(f"PROVIDER: {args.provider} | PROMPT: {ptype} | MODEL: {model or 'N/A'}")
        print("=" * 90)
        print("\n--- PROMPT (for reference) ---")
        print(prompt_text)
        print("\n--- RESPONSE ---")
        print(resp)

        if args.save_md:
            save_example_md(
                out_path=args.save_md,
                provider=args.provider,
                model=model,
                prompt_type=ptype,
                prompt_text=prompt_text,
                response_text=resp,
            )


if __name__ == "__main__":
    main()
