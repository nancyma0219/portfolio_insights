# Problem 2 - Implementation Documentation

**Developer**: Yuhan (Nancy) Ma   
**Actual Time Spent**: ~17 hours

---

## 1. Architecture Overview

#### Overview

The project follows a **layered architecture** with clear separation of concerns:

```
┌─────────────────────────────────────────────────┐
│                   User Interface                │
│  ┌─────────────────────┐  ┌────────────────────┐│
│  │ Streamlit Dashboard │  │     CLI Tools      ││
│  └─────────────────────┘  └────────────────────┘│
└─────────────────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────┐
│       Application Orchestration (api.py)        │
│        - Process transactions                   │
│        - Generate insights                      │
│        - Export analytics                       │
└─────────────────────────────────────────────────┘
                       ↓
┌──────────────────────────────────────────────────┐
│                   Business Logic                 │
│┌───────────────────────┐  ┌─────────────────────┐│
││ Transaction Processor │  │ Insights Generator  ││
││ - Load & clean data   │  │ - LLM integration   ││
││ - Calculate analytics │  │ - Local fallback    ││
││ - Query APIs          │  │ - Prompt engineering││
│└───────────────────────┘  └─────────────────────┘│
└──────────────────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────┐
│                   Data Layer                    │
│┌───────────────────────┐  ┌────────────────────┐│
││ In-Memory DataFrames  │  │     File System    ││
││      (pandas)         │  │     (CSV/JSON)     ││
│└───────────────────────┘  └────────────────────┘│
└─────────────────────────────────────────────────┘
```

**Component responsibilities:**

- transaction_processor.py: Handles CSV ingestion, validation, cleaning, analytics computation, and query utilities.

- insights_generator.py: Translates aggregated analytics into human-readable insights using LLMs or a deterministic local fallback.

- api.py: Acts as a UI-agnostic orchestration layer, enabling reuse across the dashboard, CLI, and future services.

- dashboard.py: Implements a Streamlit-based interactive UI for exploration and visualization.

#### Intentions of Design

1. **Modularity**: Each component has a single, well-defined responsibility
2. **Testability**: All modules can be tested independently
3. **Extensibility**: Easy to add new providers, analytics, or visualizations
4. **User-Friendly**: Both programmatic API and interactive UI options

---

## 2. Data Processing Decisions

Given that the input represents real-world financial transactions, the system prioritizes **data integrity over completeness**, avoiding any form of imputation that could introduce artificial signals while remaining robust to common real-world data quality issues.

---

#### Technology Stack

**Choice:** Pandas and NumPy

**Rationale**:
- Industry-standard for financial data analysis
- Vectorized operations (10-100x faster than pure Python)
- Excellent time-series support
- Sub-second queries on 10,000+ transactions

**Alternative Considered**: Pure Python (dictionaries/lists)
- Faster for single O(1) lookups via hash tables
- But significantly slower for aggregations and filtering
- Pandas vectorization wins for our use case (multiple analytics operations)

**Trade-off**: For datasets >100K rows, custom hash-based indexing or database backend would be faster than Pandas boolean indexing.

---

#### Schema Validation

Input CSV files are validated against a required schema before any processing occurs. If any required field is missing, processing fails early.
```python
REQUIRED_COLS = ["timestamp", "ticker", "action", "quantity", "price", "trader_id"]
```

This implementation(based on problem spec) specifically assumes that input data follows a structure consistent with the provided sample dataset. Failing fast prevents ambiguous behavior and ensures that downstream analytics operate on well-defined inputs.

However, in a production setting, this validation step could be extended to support schema versioning, optional fields, or configurable column mappings to accommodate heterogeneous data sources.

---

#### Cleaning Strategy

The cleaning pipeline applies the following principles to ensure all retained transactions are internally consistent and analytically meaningful:

- **Invalid records are dropped rather than corrected**  
  Transactions with missing or invalid timestamps, quantities, or prices are removed instead of imputed. This avoids introducing fabricated data into financial analytics.

- **Normalization without inference**  
  Tickers, actions, and trader identifiers are standardized (case and whitespace normalization only) without altering semantic meaning.

- **Strict action validation**  
  Only `BUY` and `SELL` actions are retained. Unsupported or ambiguous action values are discarded to preserve analytical correctness.

- **Positive numeric constraints**  
  Quantities and prices must be strictly positive. Non-numeric or non-positive values are treated as invalid.

---

#### Derived Fields

After cleaning, two derived fields are computed:

- **Notional value** (quantity × price) to support volume-based analytics
- **Transaction date** extracted from timestamps to enable time-based aggregation

These fields are deterministic transformations and do not introduce new assumptions.

---

## 3. Analytics Computation

Key analytics computed from the cleaned dataset include:
- Total notional trading volume per ticker
- Net position per ticker (BUY − SELL, shares)
- Buy/sell distribution
- Daily trading volume trends
- Most active traders (by transaction count and notional)

Net positions are computed via aligned buy–sell aggregation with missing sides treated as zero, avoiding NaN artifacts for tickers with only buys or only sells. All metrics are implemented using vectorized pandas operations and run efficiently at the target scale (1,000+ transactions).


### Design Considerations and Alternatives

The analytics layer is intentionally implemented using vectorized pandas operations, which provide O(n) performance and sufficient efficiency for the target scale (1,000+ transactions).

Several alternative approaches were considered but not adopted:

- **Pre-indexed hash tables (e.g., ticker → transactions)**  
  Could reduce lookup latency for repeated queries, but add memory overhead and complexity. Pandas groupby operations are already optimized and sufficient at this scale.

- **Database-backed analytics (SQL with indexes)**  
  Useful for persistence, concurrent access, or very large datasets, but unnecessary for an in-memory, single-user analytical workflow.

- **Incremental or streaming aggregation**  
  Appropriate for real-time ingestion, but full recomputation is simpler and fast enough for batch-style analysis triggered by file upload.

- **Statistical or ML-based analytics**  
  Techniques such as anomaly detection or clustering were intentionally excluded to avoid introducing modeling assumptions beyond the problem scope.

The chosen approach prioritizes clarity, determinism, and maintainability while meeting performance requirements, but other alternatives should be more carefully compared if we scale up even further.

---

## 4. API Design & Data Flow

#### Data Flow

```
CSV File → TransactionProcessor → Analytics Dict → InsightsGenerator → Insights Text
                                        ↓
                                  Dashboard/API
```

#### API Layer Design Rationale

Although the Streamlit dashboard exposes most functionality, an explicit `api.py` layer is still included to:

- Decouple business logic from UI concerns
- Enable CLI and programmatic usage
- Improve testability and reuse
- Allow future extensions (e.g., FastAPI service) without refactoring core logic

---

#### TransactionProcessor
**Responsibility:** Deterministic data processing and analytics

- Load and validate CSV input
- Clean and normalize data
- Compute analytics
- Support common query patterns (by ticker, trader, time range)

**Design Choice:** Separate load → clean → analyze phases  
**Pros:** Easier debugging, inspectability, and reusability without reloading data.

**Alternative not taken:**  
Pre-built hash indexes (ticker → rows) were considered, but pandas groupby operations are already efficient at the target scale(1000+) and avoid additional memory and complexity.

---

#### InsightsGenerator
**Responsibility:** Convert analytics into natural-language insights and support multiple providers via runtime selection:
  - Anthropic
  - OpenAI
  - Deterministic local fallback

LLM responses are expected to be structured markdown suitable for direct rendering, and all API interactions are wrapped with robust error handling. If API keys are missing or calls fail, deterministic heuristics analyze concentration, imbalances, spikes, and trader dominance. This design allows the same analytics pipeline to function with or without external APIs. 


**Prompt Strategy Summary:**

Rather than sending raw transaction data, prompts were constructed around compact analytics summaries that highlight totals, distributions, top contributors, and outliers. Using only compact summary statistics can significantly reduce token usage, minimizes hallucination risk, and avoids exposing sensitive transaction-level data.

Each prompt explicitly instructed the model to reason only from the provided summary, cite numerical evidence when making claims, and avoid assumptions when information was missing. Structured output formats (e.g., sections for patterns, risks, and follow-up actions) were used to guide consistent, scannable responses suitable for direct UI rendering. As development progressed, prompts were refined to reduce verbosity, improve factual grounding, and align outputs with practical risk-analysis workflows.

<br>

**Example:**

PROVIDER: openai | PROMPT: patterns | MODEL: gpt-4o

**PROMPT:**

Generate concise pattern insights based on the following portfolio summary.

Summary:
- Total transactions: 1,030
- Total notional: $21,633,775.10
- Unique tickers: 10
- Unique traders: 16
- Date range: 2024-01-15 19:02:00 to 2024-01-18 01:30:00
- Top tickers by volume: MSFT=$8,896,850, AAPL=$4,797,094, GOOGL=$3,867,913, NVDA=$1,241,259, GS=$932,483
- BUY/SELL counts: {'SELL': 523, 'BUY': 507}
- Most active trader: T001 (tx=344, notional=$0)


**RESPONSE:**

Key Patterns
- The total transactions are almost evenly split between SELL (523) and BUY (507) actions, indicating a balanced trading strategy.
- MSFT dominates the trading volume with $8,896,849.90, significantly higher than the next highest, AAPL, at $4,797,093.65.
- The top three tickers by volume (MSFT, AAPL, GOOGL) account for a substantial portion of the total notional volume, suggesting a focus on these large-cap stocks.

Concentrations / Imbalances
- There is a notable concentration of trading activity in MSFT, which alone accounts for approximately 41% of the total notional volume.
- The net positions reveal a long position in JPM (300 shares) and short positions in GS (-110 shares), NVDA (-195 shares), GOOGL (-225 shares), and BAC (-280 shares), indicating a potential bearish outlook on these stocks.
- Trader T001 is significantly more active than others, with 344 transactions and a notional volume of $7,422,950.30, suggesting a potential concentration of trading risk.

Unusual Activity (if any)
- The high volume of trading on 2024-01-17 ($6,971,558.10) and 2024-01-16 ($6,956,748.25) compared to 2024-01-15 ($5,844,861.05) may indicate unusual market activity or specific events driving increased trading.

Suggested Follow-ups
- Investigate the reasons behind the high concentration of trading in MSFT and the implications of the net short positions in GS, NVDA, GOOGL, and BAC.
- Analyze the trading strategy and risk exposure of Trader T001 due to their outsized activity and notional volume.
- Examine the market conditions or news events on 2024-01-16 and 2024-01-17 that may have led to the spike in trading volume.

*Note: More examples can be found in folder "insights_examples"*


---

#### Unified API (`api.py`)
**Responsibility:** Application orchestration

- One-line transaction processing
- One-call insight generation (patterns, risks, custom)
- JSON export with normalized types

---

#### JSON Export Considerations

All exported analytics are normalized to ensure strict JSON compatibility:
- datetime / date → ISO-8601 strings
- NaN / Infinity → null
- Pandas objects → plain Python dictionaries

This allows safe use in downloads, APIs, and LLM payloads.


---

## 5. Testing

The solution was tested using a combination of unit tests, mocked integration tests, and manual validation. Unit tests cover CSV loading, schema validation, data cleaning edge cases, analytics correctness, and query behavior. LLM provider calls are fully mocked to ensure deterministic, offline-safe testing of routing and fallback logic. Edge cases considered include empty or malformed CSV files, single-transaction datasets, all-BUY or all-SELL scenarios, missing or invalid fields, and environments where no API keys are available. Besides, manual testing and insights response validation were conducted through both the Streamlit dashboard and CLI. 

---

## 6. Challenges & Learnings

The most challenging aspect of the assignment was designing and coordinating the overall system structure while balancing LLM usage with correctness, cost, and privacy. Ensuring that generated insights were meaningful and data-backed without sending raw transactions also required careful aggregation design as well as subjective evaluation of qualitative output quality. 

With more time, I would introduce database-backed storage for larger datasets, add statistical anomaly detection alongside heuristic rules, implement automated benchmarking and CI, and further optimize prompt quality and efficiency. 

Through this project, I gained more hands-on experience designing an end-to-end LLM pipeline and strengthened my familiarity with Streamlit.

---

## 7. AI Tools Used

ChatGPT and Claude were used during development mainly for architecture brainstorming, discussing trade-offs, iterating on prompt structure, refining markdown documentation, and drafting initial ideas for edge cases and test scenarios. All AI suggestions were critically reviewed, adapted, and manually implemented. No AI-generated code was used without verification.



