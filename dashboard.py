"""
Dashboard Module (Streamlit UI)

This module provides an interactive Streamlit dashboard for exploring transaction analytics and AI-generated insights.

Goals:
- Accept CSV uploads from the user
- Invoke the transaction processing pipeline
- Display cleaned transactions, summary metrics, and charts
- Generate and display pattern, risk, and custom insights via InsightsGenerator

Inputs:
- User-uploaded CSV file (transactions)
- User-selected LLM provider (local / openai / anthropic)
- Optional custom question text

Outputs:
- Interactive UI components (tables, charts, metrics, text insights)
- Downloadable cleaned data and analytics summaries

Design Notes:
- This module contains no business logic for data cleaning or analytics.
- All computation is delegated to TransactionProcessor and InsightsGenerator.
- The dashboard acts purely as a presentation and orchestration layer.
"""


from __future__ import annotations

import json
import tempfile
import math
import re
from datetime import date, datetime
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from transaction_processor import TransactionProcessor
from insights_generator import InsightsGenerator

from dotenv import load_dotenv
load_dotenv()


# ----------------------------
# STYLING CONFIGURATION
# Adjust these values to customize spacing and typography
# ----------------------------
SPACING_CONFIG = {
    "section_top_margin": "2rem",      # Space above major sections
    "section_bottom_margin": "1rem",   # Space below major sections
    "subsection_margin": "1.5rem",     # Space for subsections
    "element_gap": "0.5rem",           # Gap between small elements
    "chart_margin": "2rem",            # Space around charts
}

TYPOGRAPHY_CONFIG = {
    "main_header_size": "2.5rem",      # Main title size
    "section_header_size": "2rem",     # Major section headers (h2)
    "subsection_header_size": "1.5rem", # Subsection headers (h3)
    "body_line_height": "1.6",         # Line height for text
}


# ----------------------------
# Page configuration & styling
# ----------------------------
st.set_page_config(
    page_title="Portfolio Insights Generator",
    page_icon="📊",
    layout="wide",
)

st.markdown(
    f"""
    <style>
    /* Main header styling */
    .main-header {{
        font-size: {TYPOGRAPHY_CONFIG['main_header_size']};
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }}
    
    /* Insight box styling */
    .insight-box {{
        background-color: #e8f4f8;
        padding: 1.25rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
        white-space: pre-wrap;
        line-height: {TYPOGRAPHY_CONFIG['body_line_height']};
        font-size: 1rem;
    }}
    
    /* Override all headers inside insight boxes to use consistent styling */
    .insight-box h1,
    .insight-box h2,
    .insight-box h3,
    .insight-box h4 {{
        color: #1976d2 !important;
        font-size: 1.2rem !important;
        font-weight: 600 !important;
        margin-top: 1rem !important;
        margin-bottom: 0.5rem !important;
        border-bottom: none !important;
        padding-bottom: 0 !important;
    }}
    
    /* First heading in insight box should have no top margin */
    .insight-box h1:first-child,
    .insight-box h2:first-child,
    .insight-box h3:first-child,
    .insight-box h4:first-child {{
        margin-top: 0 !important;
    }}
    
    /* Bullet points inside insight boxes */
    .insight-box ul {{
        margin: 0.5rem 0;
        padding-left: 1.5rem;
    }}
    
    .insight-box li {{
        margin: 0.5rem 0;
        line-height: 1.6;
    }}
    
    /* Paragraphs inside insight boxes */
    .insight-box p {{
        margin: 0.5rem 0;
        line-height: 1.6;
    }}
    
    /* Section spacing */
    .stMarkdown {{
        margin-bottom: {SPACING_CONFIG['element_gap']};
    }}
    
    /* Reduce spacing between header and content */
    div[data-testid="stVerticalBlock"] > div:has(div.stMarkdown) {{
        gap: {SPACING_CONFIG['element_gap']};
    }}
    
    /* Consistent spacing for charts */
    div[data-testid="stHorizontalBlock"] {{
        gap: 1.5rem;
        margin-bottom: {SPACING_CONFIG['chart_margin']};
    }}
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 1rem;
        margin-bottom: 1rem;
    }}
    
    /* Metric card styling */
    div[data-testid="stMetricValue"] {{
        font-size: 1.5rem;
        font-weight: 600;
    }}
    
    /* Better button spacing */
    .stButton button {{
        margin-top: 0.5rem;
    }}
    
    /* Streamlit native headers - override for consistency */
    h1, h2, h3 {{
        margin-top: {SPACING_CONFIG['section_top_margin']} !important;
        margin-bottom: {SPACING_CONFIG['section_bottom_margin']} !important;
        font-weight: 600 !important;
    }}
    
    h1 {{
        font-size: {TYPOGRAPHY_CONFIG['main_header_size']} !important;
    }}
    
    h2 {{
        font-size: {TYPOGRAPHY_CONFIG['section_header_size']} !important;
        color: #1f77b4 !important;
        padding-bottom: 0.5rem !important;
        border-bottom: 2px solid #e0e0e0 !important;
    }}
    
    h3 {{
        font-size: {TYPOGRAPHY_CONFIG['subsection_header_size']} !important;
        color: #333333 !important;
        margin-top: {SPACING_CONFIG['subsection_margin']} !important;
    }}
    
    /* Horizontal rule styling */
    hr {{
        margin: {SPACING_CONFIG['section_top_margin']} 0 {SPACING_CONFIG['section_bottom_margin']} 0 !important;
        border: none !important;
        border-top: 1px solid #e0e0e0 !important;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------
# Chart builders (pure functions)
# ----------------------------
def create_volume_chart(analytics: Dict[str, Any], top_k: int = 10) -> go.Figure:
    """Create interactive volume by ticker chart."""
    top = analytics["volume_by_ticker"].head(top_k)

    fig = px.bar(
        x=top.index,
        y=top.values,
        labels={"x": "Ticker", "y": "Total Volume ($)"},
        title=f"Top {top_k} Tickers by Trading Volume",
    )
    fig.update_layout(height=400, showlegend=False)
    return fig


def create_position_chart(analytics: Dict[str, Any], top_k: int = 10) -> go.Figure:
    """Create net position chart (shares)."""
    top = analytics["net_position"].head(top_k)

    colors = ["green" if x > 0 else "red" for x in top.values]
    fig = go.Figure(
        data=[
            go.Bar(
                x=top.index,
                y=top.values,
                marker_color=colors,
                text=[f"{v:,.0f}" for v in top.values],
                textposition="auto",
            )
        ]
    )
    fig.update_layout(
        title=f"Top {top_k} Net Positions (Shares)",
        xaxis_title="Ticker",
        yaxis_title="Net Position (shares)",
        height=400,
    )
    return fig


def create_trader_activity_chart(analytics: Dict[str, Any], top_k: int = 10) -> go.Figure:
    """Create trader activity chart."""
    top = analytics["trader_activity"].head(top_k)

    fig = go.Figure(
        data=[
            go.Bar(
                x=top.index,
                y=top["transaction_count"],
                text=[f"{int(v):,}" for v in top["transaction_count"].values],
                textposition="auto",
            )
        ]
    )
    fig.update_layout(
        title=f"Top {top_k} Most Active Traders",
        xaxis_title="Trader ID",
        yaxis_title="Number of Transactions",
        height=400,
    )
    return fig


def create_daily_volume_chart(analytics: Dict[str, Any]) -> go.Figure:
    """Create daily volume trend chart."""
    daily = analytics["daily_volume"].reset_index()
    daily.columns = ["Date", "Volume"]

    fig = px.line(
        daily,
        x="Date",
        y="Volume",
        title="Daily Trading Volume Trend",
        markers=True,
    )
    fig.update_layout(height=400, xaxis_title="Date", yaxis_title="Total Volume ($)")
    return fig


# ----------------------------
# Data processing (cached)
# ----------------------------
@st.cache_data(show_spinner=False)
def process_uploaded_csv(file_bytes: bytes) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Process uploaded CSV bytes into (cleaned_df, analytics).

    Notes:
    - Uses a temporary file because TransactionProcessor expects a path.
    - Cached by Streamlit based on file_bytes content hash.
    """
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=True) as tmp:
        tmp.write(file_bytes)
        tmp.flush()

        processor = TransactionProcessor(tmp.name)
        processor.load_data()
        processor.clean_data()
        processor.calculate_analytics()

        cleaned_df = processor.cleaned_df.copy()
        analytics = processor.analytics

    return cleaned_df, analytics


def analytics_to_jsonable(analytics: Dict[str, Any]) -> Dict[str, Any]:
    """Convert analytics dict into JSON-serializable types (including dict keys)."""

    def _convert(obj: Any) -> Any:
        # Pandas / numpy containers
        if isinstance(obj, pd.Series):
            # Ensure keys become strings (e.g., datetime.date -> "YYYY-MM-DD")
            return {str(k): _convert(v) for k, v in obj.to_dict().items()}

        if isinstance(obj, pd.DataFrame):
            # Use records for JSON friendliness
            return obj.reset_index().to_dict(orient="records")

        # Dict: convert keys to strings
        if isinstance(obj, dict):
            return {str(k): _convert(v) for k, v in obj.items()}

        # Lists / tuples
        if isinstance(obj, (list, tuple)):
            return [_convert(x) for x in obj]

        # Datetimes
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()

        # Numpy scalars
        if isinstance(obj, (np.integer, np.floating, np.bool_)):
            obj = obj.item()

        # NaN -> None (valid JSON)
        if isinstance(obj, float) and math.isnan(obj):
            return None

        return obj

    return _convert(analytics)


def markdown_to_html(text: str) -> str:
    """
    Convert markdown text to HTML for display in insight boxes.
    Handles headers, bullet points, bold, and basic formatting.
    """
    if not text:
        return ""
    
    html = text
    
    # Convert headers (##, ###, ####)
    html = re.sub(r'^#### (.+)$', r'<h4>\1</h4>', html, flags=re.MULTILINE)
    html = re.sub(r'^### (.+)$', r'<h3>\1</h3>', html, flags=re.MULTILINE)
    html = re.sub(r'^## (.+)$', r'<h2>\1</h2>', html, flags=re.MULTILINE)
    html = re.sub(r'^# (.+)$', r'<h1>\1</h1>', html, flags=re.MULTILINE)
    
    # Convert bold (**text**)
    html = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', html)
    
    # Convert italic (*text*)
    html = re.sub(r'\*(.+?)\*', r'<em>\1</em>', html)
    
    # Convert bullet points (- item)
    lines = html.split('\n')
    in_list = False
    result_lines = []
    
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('- '):
            if not in_list:
                result_lines.append('<ul>')
                in_list = True
            content = stripped[2:].strip()
            result_lines.append(f'<li>{content}</li>')
        else:
            if in_list:
                result_lines.append('</ul>')
                in_list = False
            if stripped:
                # Only wrap non-header lines in paragraphs
                if not (stripped.startswith('<h') or stripped.startswith('</h')):
                    result_lines.append(f'<p>{line}</p>')
                else:
                    result_lines.append(line)
            else:
                result_lines.append('')
    
    if in_list:
        result_lines.append('</ul>')
    
    html = '\n'.join(result_lines)
    
    return html



# ----------------------------
# Main app
# ----------------------------
def main() -> None:
    st.markdown('<div class="main-header">📊 Portfolio Insights Generator</div>', unsafe_allow_html=True)
    st.markdown("Upload your transaction data to generate analytics and AI-powered insights.")

    # Sidebar
    with st.sidebar:
        st.header("Settings")

        uploaded = st.file_uploader(
            "Upload Transaction CSV",
            type=["csv"],
            help="Required columns: timestamp, ticker, action, quantity, price, trader_id",
        )

        st.subheader("AI Insights")
        llm_provider = st.selectbox(
            "LLM Provider",
            ["anthropic", "openai", "local"],
            help="Select which provider to use for insights generation.",
        )

        st.markdown("---")
        st.markdown(
            """
            ### About
            This dashboard:
            - Cleans and analyzes transaction data locally
            - Generates summary analytics and interactive charts
            - Produces AI (or local fallback) insights
            """
        )

    # Initialize session state for insight persistence
    if "pattern_insight" not in st.session_state:
        st.session_state["pattern_insight"] = ""
    if "risk_insight" not in st.session_state:
        st.session_state["risk_insight"] = ""
    if "custom_insight" not in st.session_state:
        st.session_state["custom_insight"] = ""

    if uploaded is None:
        st.info("👈 Upload a CSV file to get started.")
        st.markdown("")  # Add spacing
        st.markdown("### Expected CSV Format")
        st.code(
            """timestamp,ticker,action,quantity,price,trader_id
2024-01-15 09:30:00,AAPL,BUY,100,185.50,T001
2024-01-15 10:15:00,GOOGL,SELL,50,142.30,T002
""",
            language="text",
        )
        return

    # Process data
    try:
        with st.spinner("Processing transaction data..."):
            file_bytes = uploaded.getvalue()
            cleaned_df, analytics = process_uploaded_csv(file_bytes)

        st.success(f"✅ Successfully loaded {analytics['total_transactions']:,} transactions")

        # Summary metrics
        st.markdown("---")  # Add separator
        st.markdown("## 📈 Summary Statistics")
        st.markdown("")  # Add spacing
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Total Transactions", f"{analytics['total_transactions']:,}")
        with c2:
            st.metric("Total Volume", f"${float(analytics['total_volume']):,.2f}")
        with c3:
            st.metric("Unique Tickers", int(analytics["unique_tickers"]))
        with c4:
            st.metric("Unique Traders", int(analytics["unique_traders"]))

        # Buy/sell distribution
        st.markdown("")  # Add spacing
        st.markdown("### Buy/Sell Distribution")
        left, right = st.columns(2)
        with left:
            action_counts = analytics["action_counts"]
            fig_pie = px.pie(values=action_counts.values, names=action_counts.index, title="Transaction Type Distribution")
            st.plotly_chart(fig_pie, use_container_width=True)
        with right:
            st.dataframe(
                pd.DataFrame({"Action": action_counts.index, "Count": action_counts.values}),
                use_container_width=True,
                height=220,
            )

        # Charts
        st.markdown("---")  # Add separator
        st.markdown("## 📊 Analytics Dashboard")
        st.markdown("")  # Add spacing
        r1, r2 = st.columns(2)
        with r1:
            st.plotly_chart(create_volume_chart(analytics), use_container_width=True)
        with r2:
            st.plotly_chart(create_position_chart(analytics), use_container_width=True)

        r3, r4 = st.columns(2)
        with r3:
            st.plotly_chart(create_trader_activity_chart(analytics), use_container_width=True)
        with r4:
            st.plotly_chart(create_daily_volume_chart(analytics), use_container_width=True)

        # Detailed tables + downloads
        st.markdown("---")  # Add separator
        st.markdown("## 📋 Detailed Analytics")
        st.markdown("")  # Add spacing
        t1, t2, t3 = st.tabs(["Volume by Ticker", "Net Positions", "Trader Activity"])

        with t1:
            dfv = pd.DataFrame({"Ticker": analytics["volume_by_ticker"].index, "Total Volume": analytics["volume_by_ticker"].values})
            dfv["Total Volume"] = dfv["Total Volume"].apply(lambda x: f"${float(x):,.2f}")
            st.dataframe(dfv, use_container_width=True)

        with t2:
            dfp = pd.DataFrame({"Ticker": analytics["net_position"].index, "Net Position (shares)": analytics["net_position"].values})
            st.dataframe(dfp, use_container_width=True)

        with t3:
            dft = analytics["trader_activity"].reset_index()
            dft.columns = ["Trader ID", "Transaction Count", "Total Volume"]
            dft["Total Volume"] = dft["Total Volume"].apply(lambda x: f"${float(x):,.2f}")
            st.dataframe(dft, use_container_width=True)

        # Downloads (nice take-home polish)
        st.markdown("")  # Add spacing
        dl1, dl2 = st.columns(2)
        with dl1:
            st.download_button(
                "⬇️ Download Cleaned CSV",
                data=cleaned_df.to_csv(index=False).encode("utf-8"),
                file_name="cleaned_transactions.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with dl2:
            st.download_button(
                "⬇️ Download Analytics JSON",
                data=json.dumps(analytics_to_jsonable(analytics), indent=2).encode("utf-8"),
                file_name="analytics.json",
                mime="application/json",
                use_container_width=True,
            )

        # Insights
        st.markdown("---")  # Add separator
        st.markdown("## 🤖 AI-Generated Insights")
        st.markdown("")  # Add spacing
        gen = InsightsGenerator(api_provider=llm_provider)

        b1, b2 = st.columns(2)
        with b1:
            if st.button("🔍 Generate Pattern Insights", use_container_width=True):
                with st.spinner("Analyzing patterns..."):
                    st.session_state["pattern_insight"] = gen.generate_pattern_insights(analytics)

        with b2:
            if st.button("⚠️ Generate Risk Insights", use_container_width=True):
                with st.spinner("Analyzing risks..."):
                    st.session_state["risk_insight"] = gen.generate_risk_insights(analytics)

        if st.session_state["pattern_insight"]:
            html_content = markdown_to_html(st.session_state["pattern_insight"])
            st.markdown(f'<div class="insight-box">{html_content}</div>', unsafe_allow_html=True)

        if st.session_state["risk_insight"]:
            html_content = markdown_to_html(st.session_state["risk_insight"])
            st.markdown(f'<div class="insight-box">{html_content}</div>', unsafe_allow_html=True)

        st.markdown("")  # Add spacing
        st.markdown("### 💬 Ask a Custom Question")
        q = st.text_area("Enter your question about the data:", placeholder="e.g., What are the main concentration risks in this portfolio?")
        if st.button("Generate Custom Insight"):
            if q.strip():
                with st.spinner("Generating insights..."):
                    st.session_state["custom_insight"] = gen.generate_custom_insights(analytics, q)
            else:
                st.warning("Please enter a question first.")

        if st.session_state["custom_insight"]:
            html_content = markdown_to_html(st.session_state["custom_insight"])
            st.markdown(f'<div class="insight-box">{html_content}</div>', unsafe_allow_html=True)

        # Raw data
        with st.expander("📄 View Cleaned Transaction Data"):
            st.dataframe(cleaned_df, use_container_width=True)

    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        st.exception(e)


if __name__ == "__main__":
    main()