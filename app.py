"""
app.py
======
Streamlit dashboard for the Market Regime Detection application.

Bloomberg Dark Mode UI · Plotly interactive charts · PCA-HMM backend.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

from copy import deepcopy
from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from data_engine import get_market_data
from model_engine import RegimeDetectionModel

# ── Page configuration (must be the very first Streamlit call) ────────────
st.set_page_config(
    page_title="Market Regime Detector",
    page_icon="▪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Design tokens — Bloomberg dark palette
# ---------------------------------------------------------------------------
_BG_MAIN    = "#111111"
_BG_PANEL   = "#161616"
_BG_CARD    = "#1a1a1a"
_BG_SIDEBAR = "#0d0d0d"
_BORDER     = "#1e3a5f"
_TEXT_WHITE = "#e8e8e8"
_TEXT_DIM   = "#888888"
_TEXT_AMBER = "#FFB300"
_BLUE_HI    = "#00BFFF"
_BLUE_ACC   = "#1565C0"

REGIME_COLORS: dict[str, str] = {
    "BULL":    "#00C853",   # vivid green
    "NEUTRAL": "#FFB300",   # amber
    "BEAR":    "#FF3D00",   # vivid red
}
REGIME_SHADE: dict[str, str] = {
    "BULL":    "rgba(0,200,83,0.10)",
    "NEUTRAL": "rgba(255,179,0,0.10)",
    "BEAR":    "rgba(255,61,0,0.10)",
}
REGIME_ICON: dict[str, str] = {
    "BULL": "▲", "NEUTRAL": "◆", "BEAR": "▼",
}

# ---------------------------------------------------------------------------
# Bloomberg CSS injection
# ---------------------------------------------------------------------------
_CSS = f"""
<style>
/* ── Global reset ── */
html, body, [data-testid="stAppViewContainer"],
[data-testid="stMain"], .main {{
    background-color: {_BG_MAIN} !important;
    color: {_TEXT_WHITE};
    font-family: 'Courier New', Courier, monospace;
}}

/* ── Sidebar ── */
[data-testid="stSidebar"] {{
    background-color: {_BG_SIDEBAR} !important;
    border-right: 1px solid {_BORDER};
}}
[data-testid="stSidebar"] * {{
    color: {_TEXT_WHITE} !important;
}}

/* ── Widget labels ── */
[data-testid="stTextInput"] label,
[data-testid="stDateInput"]  label,
[data-testid="stNumberInput"] label {{
    color: {_TEXT_DIM} !important;
    font-size: 0.70rem !important;
    letter-spacing: 0.14em !important;
    text-transform: uppercase !important;
}}

/* ── Inputs ── */
[data-testid="stTextInput"] input,
[data-testid="stDateInput"]  input {{
    background-color: #1e1e1e !important;
    color: {_TEXT_AMBER} !important;
    border: 1px solid {_BORDER} !important;
    border-radius: 3px !important;
    font-family: 'Courier New', monospace !important;
}}

/* ── Primary button ── */
.stButton > button {{
    background-color: {_BLUE_ACC} !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 3px !important;
    font-family: 'Courier New', monospace !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.14em !important;
    text-transform: uppercase !important;
    padding: 10px 16px !important;
    width: 100% !important;
    transition: background-color 0.2s ease !important;
}}
.stButton > button:hover {{
    background-color: #1976D2 !important;
}}

/* ── Expander ── */
[data-testid="stExpander"] {{
    background-color: {_BG_CARD} !important;
    border: 1px solid {_BORDER} !important;
    border-radius: 4px !important;
}}
[data-testid="stExpander"] summary {{
    color: {_TEXT_DIM} !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.14em !important;
    text-transform: uppercase !important;
}}

/* ── Plotly chart container ── */
[data-testid="stPlotlyChart"] {{
    border: 1px solid {_BORDER};
    border-radius: 4px;
    overflow: hidden;
}}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {{
    border: 1px solid {_BORDER} !important;
    border-radius: 4px !important;
}}
[data-testid="stDataFrame"] th {{
    background-color: #0d1b2e !important;
    color: {_TEXT_AMBER} !important;
    font-size: 0.70rem !important;
    letter-spacing: 0.10em !important;
}}

/* ── Alerts / info boxes ── */
[data-testid="stAlert"] {{
    background-color: {_BG_CARD} !important;
    border-left: 3px solid {_BLUE_ACC} !important;
    color: {_TEXT_WHITE} !important;
}}

/* ── Spinner text ── */
[data-testid="stSpinner"] p {{
    color: {_TEXT_AMBER} !important;
    font-family: 'Courier New', monospace !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.12em !important;
}}

/* ── Horizontal rule ── */
hr {{ border-color: {_BORDER} !important; }}

/* ── Custom component classes ── */
.bb-header {{
    background: linear-gradient(90deg, #0a1628 0%, {_BG_MAIN} 70%);
    border-bottom: 2px solid {_BLUE_ACC};
    padding: 10px 24px 9px;
    margin-bottom: 20px;
    letter-spacing: 0.18em;
}}
.bb-header-text {{
    color: {_TEXT_AMBER};
    font-size: 0.78rem;
    font-weight: 700;
    text-transform: uppercase;
    margin: 0;
}}

.regime-card {{
    border: 1px solid {_BORDER};
    border-radius: 4px;
    padding: 20px 24px;
    background: {_BG_CARD};
    text-align: center;
    height: 100%;
}}
.regime-label {{
    font-size: 0.65rem;
    letter-spacing: 0.18em;
    color: {_TEXT_DIM};
    text-transform: uppercase;
    margin-bottom: 8px;
}}
.regime-value {{
    font-size: 2.6rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    line-height: 1.1;
}}
.regime-date {{
    font-size: 0.68rem;
    color: {_TEXT_DIM};
    margin-top: 8px;
    letter-spacing: 0.06em;
}}

.stat-card {{
    border: 1px solid {_BORDER};
    border-radius: 4px;
    padding: 14px 12px;
    background: {_BG_CARD};
    text-align: center;
    height: 100%;
}}
.stat-label {{
    font-size: 0.62rem;
    color: {_TEXT_DIM};
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 6px;
}}
.stat-value {{
    font-size: 1.35rem;
    font-weight: 700;
    line-height: 1.1;
}}

.section-hdr {{
    color: {_TEXT_AMBER};
    font-size: 0.68rem;
    letter-spacing: 0.20em;
    text-transform: uppercase;
    border-bottom: 1px solid {_BORDER};
    padding-bottom: 5px;
    margin: 20px 0 10px 0;
}}
</style>
"""

# ---------------------------------------------------------------------------
# Base Plotly layout (shared across all charts)
# ---------------------------------------------------------------------------
_AXIS_BASE = dict(
    gridcolor="#222222",
    showline=True,
    linecolor=_BORDER,
    tickcolor=_TEXT_DIM,
    tickfont=dict(color=_TEXT_DIM, size=10),
    zeroline=False,
)


def _base_layout(title: str = "", height: int = 340, **extra) -> dict:
    """Return a fresh Bloomberg-themed Plotly layout dict."""
    layout = dict(
        paper_bgcolor=_BG_MAIN,
        plot_bgcolor=_BG_PANEL,
        font=dict(color=_TEXT_WHITE, family="'Courier New', monospace", size=11),
        title=dict(
            text=title,
            font=dict(color=_TEXT_AMBER, size=10, family="'Courier New', monospace"),
            x=0.01,
        ),
        margin=dict(l=52, r=24, t=44, b=40),
        height=height,
        legend=dict(
            bgcolor=_BG_CARD,
            bordercolor=_BORDER,
            borderwidth=1,
            font=dict(color=_TEXT_WHITE, size=10),
        ),
    )
    layout.update(extra)
    return layout


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False, ttl=3_600)
def load_data(ticker: str, start_date: str):
    """
    Fetch price data and engineer features.
    Cached with st.cache_data (serialised, 1-hour TTL).

    Returns (feature_df, scaled_matrix, scaler).
    """
    return get_market_data(ticker, start_date)


@st.cache_resource(show_spinner=False)
def fit_regime_model(ticker: str, start_date: str):
    """
    Fit the PCA-HMM pipeline.
    Cached with st.cache_resource (in-process object, never serialised).
    Cache key = (ticker, start_date) — model is refit when either changes.

    Returns (fitted RegimeDetectionModel, regimes pd.Series).
    """
    feature_df, scaled_matrix, _ = load_data(ticker, start_date)
    model = RegimeDetectionModel(n_states=3, n_pca_components=3, n_iter=200)
    regimes = model.fit_predict(feature_df, scaled_matrix)
    return model, regimes


# ---------------------------------------------------------------------------
# Chart helpers
# ---------------------------------------------------------------------------

def _regime_periods(regimes: pd.Series) -> list[dict]:
    """Split a regime series into contiguous [{start, end, label}] blocks."""
    if regimes.empty:
        return []
    periods, prev_label, prev_start = [], regimes.iloc[0], regimes.index[0]
    for dt, label in regimes.items():
        if label != prev_label:
            periods.append({"start": prev_start, "end": dt, "label": prev_label})
            prev_label, prev_start = label, dt
    periods.append({"start": prev_start, "end": regimes.index[-1], "label": prev_label})
    return periods


def _shade_regimes(fig: go.Figure, regimes: pd.Series,
                   row: int = 1, col: int = 1) -> None:
    """Overlay transparent regime rectangles on a Plotly figure panel."""
    for p in _regime_periods(regimes):
        fig.add_vrect(
            x0=p["start"], x1=p["end"],
            fillcolor=REGIME_SHADE[p["label"]],
            line_width=0,
            row=row, col=col,
        )


# ── Chart 1: Price index with regime shading ─────────────────────────────

def build_price_chart(feature_df: pd.DataFrame, regimes: pd.Series,
                      ticker: str) -> go.Figure:
    """
    Cumulative-return price index with colour-coded regime background.

    Uses exp(cumsum(log_returns)) to reconstruct a normalised price index
    (value = 1.0 at the first observed date), avoiding a second network call.
    """
    price_idx = np.exp(feature_df["Log_Return"].cumsum())
    price_idx /= price_idx.iloc[0]

    fig = go.Figure()
    _shade_regimes(fig, regimes)

    fig.add_trace(go.Scatter(
        x=price_idx.index,
        y=price_idx.values,
        mode="lines",
        line=dict(color=_TEXT_WHITE, width=1.6),
        name=ticker,
        hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Index: %{y:.4f}<extra></extra>",
    ))

    # Regime legend annotations (colour dots)
    for regime, color in REGIME_COLORS.items():
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(color=color, size=8, symbol="square"),
            name=regime, showlegend=True,
        ))

    layout = _base_layout(
        title=f"{ticker} — TOTAL RETURN INDEX  (regime-shaded background)",
        height=330,
        showlegend=True,
    )
    layout["xaxis"] = dict(**_AXIS_BASE, rangeslider=dict(visible=False))
    layout["yaxis"] = dict(**_AXIS_BASE, title="Normalised Index")
    fig.update_layout(**layout)
    return fig


# ── Chart 2: Transition matrix heatmap ───────────────────────────────────

def build_transition_heatmap(transition_df: pd.DataFrame) -> go.Figure:
    """Annotated Plotly heatmap of the learned HMM transition probabilities."""
    labels = list(transition_df.index)
    z = transition_df.values.tolist()
    text = [[f"{v:.3f}" for v in row] for row in transition_df.values]

    colorscale = [
        [0.00, "#060f1e"],
        [0.35, "#0d2744"],
        [0.65, "#1565C0"],
        [1.00, "#00BFFF"],
    ]

    fig = go.Figure(go.Heatmap(
        z=z,
        x=labels,
        y=labels,
        colorscale=colorscale,
        zmin=0.0, zmax=1.0,
        text=text,
        texttemplate="%{text}",
        textfont=dict(size=14, color="white", family="'Courier New', monospace"),
        showscale=True,
        colorbar=dict(
            tickfont=dict(color=_TEXT_DIM, size=9),
            outlinecolor=_BORDER,
            outlinewidth=1,
            thickness=12,
        ),
        hovertemplate=(
            "From: <b>%{y}</b><br>"
            "To:   <b>%{x}</b><br>"
            "Prob: <b>%{z:.4f}</b><extra></extra>"
        ),
    ))

    layout = _base_layout(title="REGIME TRANSITION PROBABILITIES", height=300)
    layout["xaxis"] = dict(**_AXIS_BASE, title="To State",   side="bottom")
    layout["yaxis"] = dict(**_AXIS_BASE, title="From State", autorange="reversed")
    fig.update_layout(**layout)
    return fig


# ── Chart 3: Feature breakdown (credit stress + PCA components) ───────────

def build_feature_chart(feature_df: pd.DataFrame,
                        scaled_matrix: np.ndarray,
                        model: RegimeDetectionModel,
                        regimes: pd.Series) -> go.Figure:
    """
    Two-panel subplot:
      Top    — Credit Stress Proxy (daily Δ log-spread HYG–LQD)
      Bottom — PCA components PC1, PC2, PC3 (already-fitted transform)
    Both panels carry regime background shading.
    """
    pca_data = model.pca.transform(scaled_matrix)
    pca_df = pd.DataFrame(
        pca_data,
        columns=[f"PC{i+1}" for i in range(pca_data.shape[1])],
        index=feature_df.index,
    )
    pca_colors = [_BLUE_HI, "#FF7043", "#69F0AE"]
    subplot_titles = ["CREDIT STRESS PROXY  (Δ log HYG − log LQD)",
                      "PCA COMPONENTS  (PC1 · PC2 · PC3)"]

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.10,
        subplot_titles=subplot_titles,
    )

    # ── Top: Credit Stress ────────────────────────────────────────────────
    _shade_regimes(fig, regimes, row=1, col=1)
    fig.add_trace(go.Scatter(
        x=feature_df.index,
        y=feature_df["Credit_Stress"],
        mode="lines",
        line=dict(color=_TEXT_AMBER, width=1.3),
        name="Credit Stress",
        hovertemplate="%{x|%Y-%m-%d}: %{y:.5f}<extra>Credit Stress</extra>",
    ), row=1, col=1)
    fig.add_hline(
        y=0, line=dict(color="#444444", width=1, dash="dot"),
        row=1, col=1,
    )

    # ── Bottom: PCA components ────────────────────────────────────────────
    _shade_regimes(fig, regimes, row=2, col=1)
    for col_name, color in zip(pca_df.columns, pca_colors):
        fig.add_trace(go.Scatter(
            x=pca_df.index,
            y=pca_df[col_name],
            mode="lines",
            line=dict(color=color, width=1.0),
            name=col_name,
            hovertemplate=f"%{{x|%Y-%m-%d}}: %{{y:.4f}}<extra>{col_name}</extra>",
        ), row=2, col=1)
    fig.add_hline(
        y=0, line=dict(color="#444444", width=1, dash="dot"),
        row=2, col=1,
    )

    # ── Shared layout ─────────────────────────────────────────────────────
    fig.update_layout(
        paper_bgcolor=_BG_MAIN,
        plot_bgcolor=_BG_PANEL,
        font=dict(color=_TEXT_WHITE, family="'Courier New', monospace", size=11),
        height=480,
        margin=dict(l=52, r=24, t=44, b=40),
        showlegend=True,
        legend=dict(
            bgcolor=_BG_CARD, bordercolor=_BORDER, borderwidth=1,
            font=dict(color=_TEXT_WHITE, size=10),
        ),
    )
    for row_n in [1, 2]:
        fig.update_xaxes(**_AXIS_BASE, row=row_n, col=1)
        fig.update_yaxes(**_AXIS_BASE, row=row_n, col=1)
    for ann in fig.layout.annotations:
        ann.font.update(color=_TEXT_AMBER, size=10,
                        family="'Courier New', monospace")
    return fig


# ── Chart 4: EM convergence trace ────────────────────────────────────────

def build_convergence_chart(model: RegimeDetectionModel) -> go.Figure:
    """Log-likelihood curve over Baum-Welch iterations."""
    history = list(model.summary()["hmm_log_prob_history"])
    iters = list(range(1, len(history) + 1))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=iters, y=history,
        mode="lines+markers",
        line=dict(color=_TEXT_AMBER, width=1.5),
        marker=dict(size=4, color=_TEXT_AMBER),
        hovertemplate="Iter %{x}: %{y:.4f}<extra></extra>",
    ))

    layout = _base_layout(
        title="BAUM-WELCH EM — LOG-LIKELIHOOD CONVERGENCE",
        height=220,
        showlegend=False,
    )
    layout["xaxis"] = dict(**_AXIS_BASE, title="Iteration")
    layout["yaxis"] = dict(**_AXIS_BASE, title="Log-Probability")
    fig.update_layout(**layout)
    return fig


# ── Chart 5: PCA explained variance bar ──────────────────────────────────

def build_pca_bar(model: RegimeDetectionModel) -> go.Figure:
    """Bar chart of PCA explained variance per component."""
    evr = model.pca.explained_variance_ratio_
    labels = [f"PC{i+1}" for i in range(len(evr))]
    cumulative = np.cumsum(evr)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=labels, y=evr,
        marker_color=_BLUE_ACC,
        name="Individual",
        hovertemplate="%{x}: %{y:.2%}<extra>Explained Var.</extra>",
    ))
    fig.add_trace(go.Scatter(
        x=labels, y=cumulative,
        mode="lines+markers",
        line=dict(color=_TEXT_AMBER, width=1.5),
        marker=dict(size=6, color=_TEXT_AMBER),
        name="Cumulative",
        yaxis="y2",
        hovertemplate="%{x}: %{y:.2%}<extra>Cumulative</extra>",
    ))

    layout = _base_layout(
        title="PCA EXPLAINED VARIANCE",
        height=220,
        showlegend=True,
    )
    layout["xaxis"] = dict(**_AXIS_BASE)
    layout["yaxis"] = dict(**_AXIS_BASE, title="Variance Explained",
                           tickformat=".0%")
    layout["yaxis2"] = dict(
        **_AXIS_BASE,
        title="Cumulative",
        overlaying="y",
        side="right",
        tickformat=".0%",
        showgrid=False,
    )
    fig.update_layout(**layout)
    return fig


# ---------------------------------------------------------------------------
# HTML component helpers
# ---------------------------------------------------------------------------

def _regime_card(regime: str, as_of: str) -> str:
    color = REGIME_COLORS.get(regime, _TEXT_AMBER)
    icon  = REGIME_ICON.get(regime, "■")
    return (
        f'<div class="regime-card">'
        f'  <div class="regime-label">Current Market Regime</div>'
        f'  <div class="regime-value" style="color:{color};">{icon}&nbsp;{regime}</div>'
        f'  <div class="regime-date">as of {as_of}</div>'
        f'</div>'
    )


def _stat_card(label: str, value: str, color: str = _TEXT_AMBER) -> str:
    return (
        f'<div class="stat-card">'
        f'  <div class="stat-label">{label}</div>'
        f'  <div class="stat-value" style="color:{color};">{value}</div>'
        f'</div>'
    )


def _section(title: str) -> None:
    st.markdown(f'<div class="section-hdr">{title}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Landing screen (shown before first run)
# ---------------------------------------------------------------------------

def _render_landing() -> None:
    st.markdown(f"""
    <div style="text-align:center; padding:70px 20px 40px;">
      <div style="color:{_TEXT_AMBER}; font-size:0.72rem; letter-spacing:0.28em;
                  text-transform:uppercase; margin-bottom:14px;">
        Market Regime Detection Engine
      </div>
      <div style="color:{_TEXT_WHITE}; font-size:1.0rem; margin-bottom:10px;
                  font-family:'Courier New',monospace;">
        Hidden Markov Model &nbsp;·&nbsp; PCA &nbsp;·&nbsp; Gaussian Emission
      </div>
      <div style="color:{_TEXT_DIM}; font-size:0.80rem; max-width:520px;
                  margin:0 auto; line-height:1.9;">
        Select a ticker and start date in the sidebar,<br>
        then press <span style="color:{_TEXT_AMBER}; font-weight:700;">■ DETECT REGIMES</span>
        to run the full PCA-HMM pipeline.
      </div>
      <div style="margin-top:48px; display:flex; justify-content:center; gap:40px;">
        <div>
          <div style="color:#00C853; font-size:1.5rem; font-weight:700;">▲ BULL</div>
          <div style="color:{_TEXT_DIM}; font-size:0.65rem; margin-top:5px;
                      letter-spacing:0.10em;">POSITIVE DRIFT</div>
        </div>
        <div>
          <div style="color:{_TEXT_AMBER}; font-size:1.5rem; font-weight:700;">◆ NEUTRAL</div>
          <div style="color:{_TEXT_DIM}; font-size:0.65rem; margin-top:5px;
                      letter-spacing:0.10em;">SIDEWAYS</div>
        </div>
        <div>
          <div style="color:#FF3D00; font-size:1.5rem; font-weight:700;">▼ BEAR</div>
          <div style="color:{_TEXT_DIM}; font-size:0.65rem; margin-top:5px;
                      letter-spacing:0.10em;">NEGATIVE DRIFT</div>
        </div>
      </div>
      <div style="margin-top:56px; color:{_TEXT_DIM}; font-size:0.65rem;
                  letter-spacing:0.10em; line-height:2.0;">
        DATA: yfinance (Adj Close) &nbsp;·&nbsp; FRED (UNRATE, FEDFUNDS)<br>
        FEATURES: log-return · vol (21d, 63d) · credit stress · macro<br>
        MODEL: GaussianHMM(3) w/ diagonally dominant transmat prior
      </div>
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Stats table formatter
# ---------------------------------------------------------------------------

def _format_stats(state_stats: pd.DataFrame) -> pd.DataFrame:
    """Return a display-formatted copy of the regime statistics DataFrame."""
    df = state_stats.copy()
    df["Obs"]        = df["Obs"].astype(int)
    df["Pct_Time"]   = df["Pct_Time"].apply(lambda x: f"{x:.1f}%")
    df["Mean_Daily"] = df["Mean_Daily"].apply(lambda x: f"{x:.4%}")
    df["Ann_Return"] = df["Ann_Return"].apply(lambda x: f"{x:.2%}")
    df["Daily_Vol"]  = df["Daily_Vol"].apply(lambda x: f"{x:.4%}")
    df["Ann_Vol"]    = df["Ann_Vol"].apply(lambda x: f"{x:.2%}")
    df["Sharpe"]     = df["Sharpe"].apply(
        lambda x: f"{x:.2f}" if pd.notna(x) else "—"
    )
    return df


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------

def main() -> None:
    # ── CSS ───────────────────────────────────────────────────────────────
    st.markdown(_CSS, unsafe_allow_html=True)

    # ── Bloomberg header strip ────────────────────────────────────────────
    st.markdown(
        '<div class="bb-header">'
        '<p class="bb-header-text">'
        '■ MARKET REGIME DETECTION SYSTEM'
        '&nbsp;&nbsp;|&nbsp;&nbsp;'
        'PCA-HMM ENGINE v1.0'
        '&nbsp;&nbsp;|&nbsp;&nbsp;'
        'BULL · NEUTRAL · BEAR'
        '</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── Sidebar ───────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown(
            f'<div style="color:{_TEXT_DIM}; font-size:0.68rem; '
            f'letter-spacing:0.18em; text-transform:uppercase; '
            f'margin-bottom:18px;">CONFIGURATION</div>',
            unsafe_allow_html=True,
        )

        ticker = st.text_input(
            "Ticker Symbol",
            value="SPY",
            max_chars=12,
            help="Any equity or ETF listed on Yahoo Finance (e.g. SPY, QQQ, AAPL).",
        ).upper().strip()

        default_start = date.today() - timedelta(days=365 * 10)
        start_date = st.date_input(
            "Start Date",
            value=default_start,
            min_value=date(1993, 1, 1),
            max_value=date.today() - timedelta(days=200),
            help=(
                "At least 200 days recommended: 63 for rolling vol warm-up "
                "plus sufficient observations for HMM convergence."
            ),
        )

        st.markdown("<hr>", unsafe_allow_html=True)
        run_btn = st.button("■  DETECT REGIMES", use_container_width=True)
        st.markdown("<hr>", unsafe_allow_html=True)

        # Static model spec panel
        st.markdown(
            f'<div style="color:{_TEXT_DIM}; font-size:0.63rem; '
            f'line-height:2.1; letter-spacing:0.06em;">'
            f'<span style="color:{_TEXT_AMBER};">MODEL PARAMETERS</span><br>'
            f'Regimes&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;: 3<br>'
            f'PCA Components : 3<br>'
            f'Covariance&nbsp;&nbsp;&nbsp;: full<br>'
            f'Training&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;: Baum-Welch EM<br>'
            f'Decoding&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;: Viterbi<br>'
            f'Diag Prior&nbsp;&nbsp;&nbsp;: 10.0<br>'
            f'Off-diag Prior: 1.0<br>'
            f'Max Iterations : 200'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── Session state management ──────────────────────────────────────────
    for key, default in [
        ("results_ready", False),
        ("run_ticker", None),
        ("run_start", None),
    ]:
        if key not in st.session_state:
            st.session_state[key] = default

    if run_btn:
        st.session_state.results_ready = True
        st.session_state.run_ticker    = ticker
        st.session_state.run_start     = start_date.strftime("%Y-%m-%d")

    # ── Landing screen ────────────────────────────────────────────────────
    if not st.session_state.results_ready:
        _render_landing()
        return

    run_ticker = st.session_state.run_ticker
    run_start  = st.session_state.run_start

    # ── Data + model pipeline ─────────────────────────────────────────────
    try:
        with st.spinner(f"Fetching market data for {run_ticker} …"):
            feature_df, scaled_matrix, scaler = load_data(run_ticker, run_start)

        with st.spinner("Fitting PCA-HMM model (Baum-Welch + Viterbi) …"):
            model, regimes = fit_regime_model(run_ticker, run_start)

    except ValueError as exc:
        st.error(f"Data error — {exc}")
        st.info("Check the ticker symbol, or extend the start date to give rolling windows time to warm up.")
        st.session_state.results_ready = False
        return
    except RuntimeError as exc:
        st.error(f"FRED / model error — {exc}")
        st.session_state.results_ready = False
        return
    except Exception as exc:
        st.error(f"Unexpected error — {exc}")
        st.session_state.results_ready = False
        return

    # ── Derived outputs ───────────────────────────────────────────────────
    current_regime = str(regimes.iloc[-1])
    current_date   = regimes.index[-1].strftime("%Y-%m-%d")
    transition_df  = model.get_transition_matrix()
    state_stats    = model.get_state_statistics(feature_df, regimes)
    model_info     = model.summary()

    def _stat(label: str, default=0.0):
        return state_stats.loc[label] if label in state_stats.index else None

    bull = _stat("BULL")
    bear = _stat("BEAR")
    neut = _stat("NEUTRAL")

    # ── Section 1: Regime metric + quick-stat cards ───────────────────────
    c_regime, c_bull, c_bear, c_neut, c_pca = st.columns([2.4, 1, 1, 1, 1])

    with c_regime:
        st.markdown(
            _regime_card(current_regime, current_date),
            unsafe_allow_html=True,
        )
    with c_bull:
        pct = f"{bull['Pct_Time']:.1f}%" if bull is not None else "—"
        st.markdown(
            _stat_card("Bull Regime", pct, REGIME_COLORS["BULL"]),
            unsafe_allow_html=True,
        )
    with c_bear:
        pct = f"{bear['Pct_Time']:.1f}%" if bear is not None else "—"
        st.markdown(
            _stat_card("Bear Regime", pct, REGIME_COLORS["BEAR"]),
            unsafe_allow_html=True,
        )
    with c_neut:
        pct = f"{neut['Pct_Time']:.1f}%" if neut is not None else "—"
        st.markdown(
            _stat_card("Neutral", pct, REGIME_COLORS["NEUTRAL"]),
            unsafe_allow_html=True,
        )
    with c_pca:
        pct_var = f"{model_info['pca_cumulative_variance']:.1%}"
        st.markdown(
            _stat_card("PCA Var. Expl.", pct_var, _BLUE_HI),
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Section 2: Price chart ────────────────────────────────────────────
    date_range = (
        f"{feature_df.index[0].strftime('%Y-%m-%d')}  →  {current_date}"
    )
    _section(f"{run_ticker}  ·  PRICE INDEX  ·  {date_range}")
    st.plotly_chart(
        build_price_chart(feature_df, regimes, run_ticker),
        use_container_width=True,
    )

    # ── Section 3: Transition matrix + regime statistics ──────────────────
    col_tm, col_stats = st.columns([1, 1.5])

    with col_tm:
        _section("TRANSITION MATRIX")
        st.plotly_chart(
            build_transition_heatmap(transition_df),
            use_container_width=True,
        )

    with col_stats:
        _section("REGIME STATISTICS")
        display_df = _format_stats(state_stats)
        st.dataframe(
            display_df,
            use_container_width=True,
            height=175,
            column_config={
                "Obs":        st.column_config.TextColumn("Days"),
                "Pct_Time":   st.column_config.TextColumn("% Time"),
                "Mean_Daily": st.column_config.TextColumn("Mean Daily"),
                "Ann_Return": st.column_config.TextColumn("Ann. Return"),
                "Daily_Vol":  st.column_config.TextColumn("Daily Vol"),
                "Ann_Vol":    st.column_config.TextColumn("Ann. Vol"),
                "Sharpe":     st.column_config.TextColumn("Sharpe"),
            },
        )

        # Per-regime Sharpe callout
        st.markdown("<br>", unsafe_allow_html=True)
        sharpe_cols = st.columns(3)
        for idx, (label, col) in enumerate(
            zip(["BULL", "NEUTRAL", "BEAR"], sharpe_cols)
        ):
            row = _stat(label)
            sharpe_val = (
                f"{row['Sharpe']:.2f}" if row is not None and pd.notna(row["Sharpe"])
                else "—"
            )
            with col:
                st.markdown(
                    _stat_card(f"{label} Sharpe", sharpe_val, REGIME_COLORS[label]),
                    unsafe_allow_html=True,
                )

    # ── Section 4: Feature breakdown ──────────────────────────────────────
    _section("FEATURE BREAKDOWN  ·  CREDIT STRESS PROXY & PCA COMPONENTS")
    st.plotly_chart(
        build_feature_chart(feature_df, scaled_matrix, model, regimes),
        use_container_width=True,
    )

    # ── Section 5: Diagnostics expander ──────────────────────────────────
    with st.expander("MODEL DIAGNOSTICS", expanded=False):
        col_conv, col_pca_bar, col_meta = st.columns([1.4, 1, 1])

        with col_conv:
            st.plotly_chart(
                build_convergence_chart(model),
                use_container_width=True,
            )

        with col_pca_bar:
            st.plotly_chart(
                build_pca_bar(model),
                use_container_width=True,
            )

        with col_meta:
            converged    = model_info["hmm_converged"]
            cv_color     = "#00C853" if converged else "#FF3D00"
            cv_text      = "YES" if converged else "NO"
            state_labels = {
                v: k for k, v in model_info["state_label_map"].items()
            }

            st.markdown(
                f'<div style="font-size:0.70rem; color:{_TEXT_DIM}; '
                f'line-height:2.2; font-family: Courier New, monospace;">'
                f'<span style="color:{_TEXT_AMBER};">RUN PARAMETERS</span><br>'
                f'Ticker&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;: '
                f'<span style="color:{_TEXT_AMBER};">{run_ticker}</span><br>'
                f'Start Date&nbsp;&nbsp;: '
                f'<span style="color:{_TEXT_AMBER};">{run_start}</span><br>'
                f'Observations: '
                f'<span style="color:{_TEXT_AMBER};">{len(feature_df):,}</span><br>'
                f'<br>'
                f'<span style="color:{_TEXT_AMBER};">HMM DIAGNOSTICS</span><br>'
                f'Converged&nbsp;&nbsp;&nbsp;: '
                f'<span style="color:{cv_color};">{cv_text}</span><br>'
                f'EM Iterations: '
                f'<span style="color:{_TEXT_AMBER};">{model_info["hmm_iterations"]}</span><br>'
                f'Final LL&nbsp;&nbsp;&nbsp;&nbsp;: '
                f'<span style="color:{_TEXT_AMBER};">'
                f'{model_info["hmm_final_log_prob"]:.4f}</span><br>'
                f'<br>'
                f'<span style="color:{_TEXT_AMBER};">STATE MAP (HMM → Label)</span><br>'
                + "".join(
                    f'State {state_labels[lbl]}&nbsp;→&nbsp;'
                    f'<span style="color:{REGIME_COLORS[lbl]};">{lbl}</span><br>'
                    for lbl in ["BULL", "NEUTRAL", "BEAR"]
                    if lbl in state_labels
                )
                + "</div>",
                unsafe_allow_html=True,
            )


if __name__ == "__main__":
    main()
