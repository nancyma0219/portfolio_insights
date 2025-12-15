"""
Insights Generator Module

This module converts aggregated transaction analytics into human-readable
insights. It supports calling external LLM providers (Anthropic/OpenAI) and a
deterministic local fallback when API keys/SDKs are unavailable.

Inputs:
- analytics (dict): Aggregated statistics computed from cleaned transaction data.
  Expected keys may include:
  - total_transactions (int), total_volume (float), unique_tickers (int),
    unique_traders (int), date_range (tuple)
  - volume_by_ticker (pd.Series), net_position (pd.Series)
  - trader_activity (pd.DataFrame), daily_volume (pd.Series)
  - action_counts (pd.Series)

Configuration:
- api_provider: 'anthropic' | 'openai' | 'local'
- Environment variables:
  - ANTHROPIC_API_KEY (if using Anthropic)
  - OPENAI_API_KEY (if using OpenAI)
  - Optional model overrides via ANTHROPIC_MODEL / OPENAI_MODEL
  - "local" mode generates deterministic, rule-based insights without calling any external LLM APIs.


Outputs:
- str (markdown-like text): Natural language insights for one of:
  - Trading patterns
  - Risk flags / red flags
  - Custom user questions

Design Notes:
- Only compact, aggregated summaries are sent to LLM APIs (never raw transactions).
- The system fails gracefully: if provider calls fail, it returns deterministic
  heuristic insights instead of raising errors.
"""


from __future__ import annotations

import os
import logging
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)


class InsightsGenerator:
    """Generate AI-powered insights using LLM APIs or local fallback."""

    def __init__(
        self,
        api_provider: str = "anthropic",
        model: Optional[str] = None,
        max_tokens: int = 900,
        temperature: float = 0.2,
    ):
        """
        Initialize the insights generator.

        Args:
            api_provider: 'anthropic', 'openai', or 'local'
            model: Optional model override. If None, uses provider defaults/env.
            max_tokens: Completion token budget for LLM calls.
            temperature: Sampling temperature for LLM calls.
        """
        self.api_provider = (api_provider or "local").strip().lower()
        if self.api_provider not in {"anthropic", "openai", "local"}:
            raise ValueError("api_provider must be one of: 'anthropic', 'openai', 'local'")

        self.max_tokens = int(max_tokens)
        self.temperature = float(temperature)

        # Keys
        self.api_key = None
        if self.api_provider == "anthropic":
            self.api_key = os.environ.get("ANTHROPIC_API_KEY")
        elif self.api_provider == "openai":
            self.api_key = os.environ.get("OPENAI_API_KEY")

        # Model defaults (can be overridden by env or argument)
        if self.api_provider == "anthropic":
            default_model = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
        elif self.api_provider == "openai":
            default_model = os.environ.get("OPENAI_MODEL", "gpt-4o")
        else:
            default_model = "local-heuristic"

        self.model = model or default_model

    # ----------------------------
    # Public APIs
    # ----------------------------
    def generate_pattern_insights(self, analytics: Dict[str, Any], top_n: int = 5) -> str:
        """Generate insights about trading patterns."""
        data_summary = self._prepare_data_summary(analytics, top_n)

        prompt = f"""You are a buy-side risk analyst. Analyze the trading analytics below and identify key patterns.
        ANALYTICS SUMMARY
        {data_summary}
        
        REQUIREMENTS
        - Use only the information in the summary; do not assume missing facts.
        - Be concise and actionable.
        - Cite numbers from the summary (tickers, volumes, net shares, counts) when making claims.

        OUTPUT FORMAT (markdown)
        ## Key Patterns
        - ...

        ## Concentrations / Imbalances
        - ...

        ## Unusual Activity (if any)
        - ...

        ## Suggested Follow-ups
        - ...
        """

        return self._call_llm_api(prompt, analytics_hint=analytics)

    def generate_risk_insights(self, analytics: Dict[str, Any], top_n: int = 10) -> str:
        """Generate insights about potential risks."""
        data_summary = self._prepare_data_summary(analytics, top_n)

        prompt = f"""You are a risk manager. Review the trading analytics below and identify risks and red flags.

                    ANALYTICS SUMMARY
                    {data_summary}

                    FOCUS AREAS
                    1) Concentration risk (large exposure to a ticker via net shares and/or volume)
                    2) Buy/Sell imbalance
                    3) Unusual spikes in daily volume
                    4) Outlier behavior by traders (very high count or notional)

                    REQUIREMENTS
                    - Be specific and data-backed.
                    - If evidence is insufficient, say so explicitly.

                    OUTPUT FORMAT (markdown)
                    ## Risks
                    - ...

                    ## Supporting Evidence
                    - ...

                    ## Mitigations / Next Checks
                    - ...
                    """
        return self._call_llm_api(prompt, analytics_hint=analytics)

    def generate_custom_insights(self, analytics: Dict[str, Any], custom_prompt: str, top_n: int = 10) -> str:
        """Generate insights based on a custom user prompt."""
        data_summary = self._prepare_data_summary(analytics, top_n)
        user_q = (custom_prompt or "").strip()

        prompt = f"""You are a financial analyst. Answer the user's question using ONLY the analytics summary.

                    ANALYTICS SUMMARY
                    {data_summary}

                    USER QUESTION
                    {user_q}

                    REQUIREMENTS
                    - If the summary does not contain enough evidence, explain what's missing.
                    - Provide a short, structured answer with bullet points.
                    """
        return self._call_llm_api(prompt, analytics_hint=analytics)

    # ----------------------------
    # Summary preparation
    # ----------------------------
    def _prepare_data_summary(self, analytics: Dict[str, Any], top_n: int = 5) -> str:
        """Prepare a compact text summary of analytics for LLM consumption."""
        summary_parts: List[str] = []

        summary_parts.append("OVERALL STATISTICS:")
        summary_parts.append(f"- Total Transactions: {analytics['total_transactions']}")
        summary_parts.append(f"- Total Volume (Notional): ${float(analytics['total_volume']):,.2f}")
        summary_parts.append(f"- Unique Tickers: {analytics['unique_tickers']}")
        summary_parts.append(f"- Unique Traders: {analytics['unique_traders']}")
        summary_parts.append(f"- Date Range: {analytics['date_range'][0]} to {analytics['date_range'][1]}")
        summary_parts.append("")

        summary_parts.append("ACTION DISTRIBUTION:")
        for action, count in analytics["action_counts"].items():
            summary_parts.append(f"- {action}: {int(count)} transactions")
        summary_parts.append("")

        summary_parts.append(f"TOP {top_n} TICKERS BY VOLUME (Notional):")
        for ticker, volume in analytics["volume_by_ticker"].head(top_n).items():
            summary_parts.append(f"- {ticker}: ${float(volume):,.2f}")
        summary_parts.append("")

        summary_parts.append(f"TOP {top_n} NET POSITIONS (Shares):")
        for ticker, position in analytics["net_position"].head(top_n).items():
            summary_parts.append(f"- {ticker}: {float(position):,.0f} shares")
        summary_parts.append("")

        summary_parts.append(f"TOP {top_n} MOST ACTIVE TRADERS:")
        for trader_id, row in analytics["trader_activity"].head(top_n).iterrows():
            summary_parts.append(
                f"- {trader_id}: {int(row['transaction_count'])} tx, ${float(row['total_value']):,.2f} notional"
            )
        summary_parts.append("")

        # Add daily volume hint (top 3 days) to help anomaly detection without huge prompts
        if "daily_volume" in analytics and analytics["daily_volume"] is not None and len(analytics["daily_volume"]) > 0:
            dv = analytics["daily_volume"].sort_values(ascending=False).head(3)
            summary_parts.append("TOP DAILY VOLUME DAYS:")
            for d, v in dv.items():
                summary_parts.append(f"- {d}: ${float(v):,.2f}")
            summary_parts.append("")

        return "\n".join(summary_parts).strip()

    # ----------------------------
    # Provider calling / fallback
    # ----------------------------
    def _call_llm_api(self, prompt: str, analytics_hint: Optional[Dict[str, Any]] = None) -> str:
        """Call the selected provider, or fall back to local heuristic insights."""
        if self.api_provider == "local":
            return self._generate_local_insights(prompt, analytics_hint)

        if not self.api_key:
            logger.warning("No API key found for provider '%s'. Falling back to local insights.", self.api_provider)
            return self._generate_local_insights(prompt, analytics_hint)

        try:
            if self.api_provider == "anthropic":
                return self._call_anthropic(prompt)
            if self.api_provider == "openai":
                return self._call_openai(prompt)
        except ImportError as e:
            logger.error("SDK import error: %s. Falling back to local insights.", str(e))
            return self._generate_local_insights(prompt, analytics_hint)
        except Exception as e:
            logger.error("Provider call failed: %s. Falling back to local insights.", str(e))
            return self._generate_local_insights(prompt, analytics_hint)

        return self._generate_local_insights(prompt, analytics_hint)

    def _call_anthropic(self, prompt: str) -> str:
        """Call Anthropic Messages API."""
        import anthropic  # type: ignore

        client = anthropic.Anthropic(api_key=self.api_key)
        message = client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        # message.content is typically a list of content blocks
        return message.content[0].text if getattr(message, "content", None) else ""

    def _call_openai(self, prompt: str) -> str:
        """Call OpenAI Chat Completions."""
        from openai import OpenAI  # type: ignore

        client = OpenAI(api_key=self.api_key)
        resp = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        return resp.choices[0].message.content or ""

    # ----------------------------
    # Local heuristic fallback
    # ----------------------------
    def _generate_local_insights(self, prompt: str, analytics: Optional[Dict[str, Any]]) -> str:
        """
        Generate deterministic insights without calling external APIs.
        This is used as a fallback when API keys are unavailable.
        """
        if not analytics:
            return (
                "## NO available API keys => Local Insights (Fallback)\n"
                "- No AI analytics provided, so only a generic response is possible.\n"
                "- Please process data first and pass the analytics dictionary.\n"
            )

        # Basic signals
        total_tx = analytics.get("total_transactions", 0)
        total_vol = float(analytics.get("total_volume", 0.0))
        action_counts = analytics.get("action_counts")
        buy_ct = int(action_counts.get("BUY", 0)) if action_counts is not None else 0
        sell_ct = int(action_counts.get("SELL", 0)) if action_counts is not None else 0

        vol_series = analytics.get("volume_by_ticker")
        pos_series = analytics.get("net_position")
        trader_df = analytics.get("trader_activity")
        daily_vol = analytics.get("daily_volume")

        lines: List[str] = []
        lines.append("## Local Insights (Fallback)")
        lines.append("- Note: API keys/SDKs unavailable or provider call failed. Using deterministic heuristics.\n")

        # Key patterns
        lines.append("## Key Patterns")
        lines.append(f"- Total transactions: {total_tx:,}")
        lines.append(f"- Total notional volume: ${total_vol:,.2f}")
        if buy_ct + sell_ct > 0:
            buy_ratio = buy_ct / (buy_ct + sell_ct)
            lines.append(f"- Buy/Sell mix: BUY {buy_ct} vs SELL {sell_ct} (BUY ratio {buy_ratio:.1%})")
        else:
            lines.append("- Buy/Sell mix: insufficient data")

        # Concentrations
        lines.append("\n## Concentrations / Imbalances")
        if vol_series is not None and len(vol_series) > 0:
            top = vol_series.head(3)
            top_share = float(top.sum()) / float(vol_series.sum()) if float(vol_series.sum()) > 0 else 0.0
            lines.append(
                f"- Top 3 tickers by notional: "
                + ", ".join([f"{k} (${float(v):,.2f})" for k, v in top.items()])
            )
            lines.append(f"- Concentration proxy: top3 notional share ≈ {top_share:.1%}")
        else:
            lines.append("- No volume-by-ticker data available.")

        if pos_series is not None and len(pos_series) > 0:
            top_pos = pos_series.head(3)
            lines.append(
                "- Largest net positions (shares): "
                + ", ".join([f"{k} ({float(v):,.0f})" for k, v in top_pos.items()])
            )

        # Unusual activity
        lines.append("\n## Unusual Activity (Heuristic)")
        if daily_vol is not None and len(daily_vol) >= 3:
            dv_sorted = daily_vol.sort_values(ascending=False)
            top_day, top_val = dv_sorted.index[0], float(dv_sorted.iloc[0])
            med_val = float(dv_sorted.median())
            if med_val > 0 and top_val / med_val >= 3.0:
                lines.append(f"- Daily volume spike: {top_day} is {top_val/med_val:.1f}× the median day.")
            else:
                lines.append("- No strong daily volume spikes detected (rule: top day ≥ 3× median).")
        else:
            lines.append("- Not enough daily volume history to assess spikes.")

        if trader_df is not None and len(trader_df) > 0:
            top_tr = trader_df.sort_values("transaction_count", ascending=False).head(1)
            tid = str(top_tr.index[0])
            txc = int(top_tr.iloc[0]["transaction_count"])
            lines.append(f"- Most active trader: {tid} with {txc} transactions.")
        else:
            lines.append("- No trader activity data available.")

        # Suggested follow-ups
        lines.append("\n## Suggested Follow-ups")
        lines.append("- Validate whether large net positions align with risk limits (per-ticker exposure).")
        lines.append("- Review the top trader’s trades for potential concentration or repeated intraday activity.")
        lines.append("- If there are spikes, inspect the underlying tickers and timestamps for that day.")

        return "\n".join(lines).strip()
