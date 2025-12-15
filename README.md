# Portfolio Insights Generator

A lightweight analytics and insights application for financial transaction data. The system ingests transaction CSVs, performs robust data cleaning and analytics, and generates data-backed insights using either large language models(LLMs) or a deterministic local fallback to rule-based analysis.

---
## Overview

This project is constructed with these main parts:

- **TransactionProcessor** handles data loading, cleaning, analytics, and retrieval

- **InsightsGenerator** converts aggregated analytics into human-readable insights. 

- **dashboard.py** provides an interactive Streamlit UI

- **api.py** exposes reusable, UI-agnostic programmatic APIs (CLI / future services)



---
## Key Capabilities

### Transaction Processing
- **CSV ingestion** with schema validation
- **Robust data cleaning**: handles missing values, invalid actions, and data type conversion
- **Time-series analysis** with automatic sorting and filtering

### Portfolio Analytics
- **Volume analysis**: Total trading volume by ticker and daily trends
- **Position tracking**: Net positions (BUY − SELL) per ticker
- **Trader activity**: Most active traders by transaction count and volume
- **Market composition**: Buy/Sell distribution and concentration metrics

### AI-Powered Insights
- **Pattern detection**: Identifies unusual trading patterns and concentrations
- **Risk assessment**: Flags concentration risk, imbalances, and unusual activity
- **Custom queries**: Ask specific question(s) about your portfolio
- **Multi-provider support**: Anthropic Claude, OpenAI GPT, or local rule-based fallback
- **Privacy-focused**: Only aggregated statistics sent to LLMs, never raw transaction data (Raw transaction data is processed locally to reduce costs, improve efficiency, and prevent hallucinations.)

### Interactive Dashboard
- **Streamlit-based UI** with real-time updates
- **Interactive visualizations** using Plotly
- **One-click insights generation**
- **Data export**: Downloadable cleaned data (CSV) and analytics (JSON)

---

## 📁 Project Structure

```
portfolio_insights/
├── data/
│   └── sample_transactions.csv      # Sample dataset
├── transaction_processor.py         # Data ingestion, cleaning, analytics
├── insights_generator.py            # LLM + local heuristic insights
├── run_insights.py                  # CLI tool to generate and save insights (for validation)
├── dashboard.py                     # Streamlit interactive UI
├── api.py                           # Reusable application/service layer
├── test_api_usage.py                # Quick/Direct API testing script
├── requirements.txt                 # Python dependencies
├── .env                             # API keys (not committed)
└── README.md
├── SOLUTION.md                      # Design decisions & tradeoffs (Problem 2)
├── insights_examples/               # Examples generated from run_insights.py
├── anthropic_custom.txt
│   ├── local_all.txt
│   ├── anthropic_custom.txt
│   ├── anthropic_patterns.txt
│   ├── anthropic_risks.txt   
│   ├── openai_custom.txt
│   ├── openai_patterns.txt
│   └── openai_risks.txt
└── test/                            # Unit tests
    ├── test_processor_load.py
    ├── test_processor_clean.py
    ├── test_processor_analytics.py
    ├── test_insights_generator.py
    ├── test_api.py
    └── test_dashboard.py

```

---

## Setup

### 1. Installation

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate 

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure API Keys (Optional)

Create a `.env` file in the project root:

```env
ANTHROPIC_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here
```

If desired, you may also override the default model selection for each provider:
```env
OPENAI_MODEL=gpt-4o
ANTHROPIC_MODEL=claude-sonnet-4-20250514
```

**Note**: If no API keys are provided, the system automatically uses local rule-based insights.

### Usage
#### 1. Interactive Dashboard (recommended)

```bash
streamlit run dashboard.py
```

Then:

- Upload a transaction CSV

- Explore analytics and visualizations

- Generate pattern, risk, or custom insights (selectable insights provider on the left)



#### 2. CLI / Programmatic API

Run the application from the command line:
```bash
python api.py data/sample_transactions.csv --provider local
```

If API keys are configured, you may use an LLM provider:
```bash
python api.py data/sample_transactions.csv --provider anthropic
python api.py data/sample_transactions.csv --provider openai
```

With a custom question
```bash
python api.py data/sample_transactions.csv \
  --provider local \     # can also replace local with anthropic or openai 
  --question "Are there concentration or outlier risks?"
```

#### 3. Direct Python Usage

Use the core modules directly in Python:
```python
from api import process_transactions, generate_all_insights

# Process transactions
cleaned_df, analytics = process_transactions("data/sample_transactions.csv")

# Generate insights
insights = generate_all_insights(
    csv_path="data/sample_transactions.csv",
    api_provider="local"  # or "anthropic", "openai"
)

print(insights["patterns"])
print(insights["risks"])
```

---

## CSV Data Format

The data(.csv) used for analytics with this application must include these columns:

| Column    | Type     | Description                |
|-----------|----------|----------------------------|
| timestamp | datetime | Transaction timestamp      |
| ticker    | string   | Stock symbol (e.g., AAPL)  |
| action    | string   | BUY or SELL                |
| quantity  | numeric  | Number of shares           |
| price     | numeric  | Price per share            |
| trader_id | string   | Unique trader identifier   |

**Example:**
```csv
timestamp,ticker,action,quantity,price,trader_id
2024-01-15 09:30:00,AAPL,BUY,100,185.50,T001
2024-01-15 10:15:00,GOOGL,SELL,50,142.30,T002
```

---

## Testing

Run all tests:
```bash
python -m pytest -q
```

Tests and Manual Validation together cover:
- Data loading & cleaning
- Analytics correctness
- Insights generation (LLM + local fallback)
- API layer
- Dashboard utilities
