import types
import pandas as pd
import pytest

from insights_generator import InsightsGenerator


def _fake_analytics():
    volume_by_ticker = pd.Series({"AAPL": 1000.0, "MSFT": 500.0})
    net_position = pd.Series({"AAPL": 10.0, "MSFT": -5.0})
    trader_activity = pd.DataFrame(
        {"transaction_count": [3, 1], "total_value": [1200.0, 300.0]},
        index=["T1", "T2"],
    )
    action_counts = pd.Series({"BUY": 3, "SELL": 1})
    daily_volume = pd.Series({pd.to_datetime("2024-01-01").date(): 1000.0, pd.to_datetime("2024-01-02").date(): 200.0})

    return {
        "total_transactions": 4,
        "total_volume": 1500.0,
        "unique_tickers": 2,
        "unique_traders": 2,
        "date_range": (pd.to_datetime("2024-01-01"), pd.to_datetime("2024-01-02")),
        "volume_by_ticker": volume_by_ticker,
        "net_position": net_position,
        "trader_activity": trader_activity,
        "action_counts": action_counts,
        "daily_volume": daily_volume,
    }


def test_prepare_data_summary_contains_key_fields():
    gen = InsightsGenerator(api_provider="local")
    summary = gen._prepare_data_summary(_fake_analytics(), top_n=2)

    assert "OVERALL STATISTICS" in summary
    assert "Total Transactions: 4" in summary
    assert "AAPL" in summary
    assert "NET POSITIONS" in summary
    assert "TOP DAILY VOLUME DAYS" in summary


def test_local_fallback_returns_deterministic_insights():
    gen = InsightsGenerator(api_provider="local")
    out = gen.generate_pattern_insights(_fake_analytics(), top_n=2)

    assert "Local Insights (Fallback)" in out
    assert "Top 3 tickers" in out or "Top 3 tickers by notional" in out
    assert "Suggested Follow-ups" in out


def test_openai_call_is_used_when_key_present(monkeypatch):
    # Fake OpenAI module with OpenAI client
    class FakeResp:
        def __init__(self):
            self.choices = [types.SimpleNamespace(message=types.SimpleNamespace(content="OPENAI_OK"))]

    class FakeChatCompletions:
        def create(self, **kwargs):
            return FakeResp()

    class FakeChat:
        def __init__(self):
            self.completions = FakeChatCompletions()

    class FakeClient:
        def __init__(self, api_key=None):
            self.chat = FakeChat()

    fake_openai_module = types.SimpleNamespace(OpenAI=FakeClient)
    monkeypatch.setitem(__import__("sys").modules, "openai", fake_openai_module)
    monkeypatch.setenv("OPENAI_API_KEY", "dummy")

    gen = InsightsGenerator(api_provider="openai", model="fake-model")
    out = gen.generate_risk_insights(_fake_analytics())

    assert out == "OPENAI_OK"


def test_anthropic_call_is_used_when_key_present(monkeypatch):
    # Fake anthropic module with Anthropic client
    class FakeMessage:
        def __init__(self):
            self.content = [types.SimpleNamespace(text="ANTHROPIC_OK")]

    class FakeMessages:
        def create(self, **kwargs):
            return FakeMessage()

    class FakeAnthropicClient:
        def __init__(self, api_key=None):
            self.messages = FakeMessages()

    fake_anthropic_module = types.SimpleNamespace(Anthropic=FakeAnthropicClient)
    monkeypatch.setitem(__import__("sys").modules, "anthropic", fake_anthropic_module)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy")

    gen = InsightsGenerator(api_provider="anthropic", model="fake-model")
    out = gen.generate_pattern_insights(_fake_analytics())

    assert out == "ANTHROPIC_OK"


def test_provider_without_key_falls_back_to_local(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    gen = InsightsGenerator(api_provider="openai")

    out = gen.generate_custom_insights(_fake_analytics(), "What are the risks?")
    assert "Local Insights (Fallback)" in out
