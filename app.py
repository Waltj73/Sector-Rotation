"""
Sector Rotation Momentum Dashboard
-----------------------------------
Tracks relative strength and momentum across the 11 S&P 500 sector ETFs
(SPDR Select Sector Funds) vs SPY, to visualize which sectors are leading
or lagging the market.

Run with:
    streamlit run app.py

Dependencies:
    pip install streamlit yfinance pandas numpy plotly
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# -----------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------

st.set_page_config(
    page_title="Sector Rotation Dashboard",
    page_icon="🔄",
    layout="wide",
)

SECTOR_ETFS = {
    "XLK": "Technology",
    "XLF": "Financials",
    "XLV": "Health Care",
    "XLY": "Consumer Discretionary",
    "XLP": "Consumer Staples",
    "XLE": "Energy",
    "XLI": "Industrials",
    "XLB": "Materials",
    "XLU": "Utilities",
    "XLRE": "Real Estate",
    "XLC": "Communication Services",
}

# Sub-industry ETFs tracked alongside (not part of) the 11 GICS sectors above.
# XBI sits inside XLV (Health Care) but is volatile enough to be worth watching
# on its own — equal-weighted, skews small/mid-cap biotech.
EXTRA_INDUSTRIES = {
    "XBI": "Biotech (sub-industry of Health Care)",
}

BENCHMARK = "SPY"

LOOKBACKS = {
    "1 Week": 5,
    "1 Month": 21,
    "3 Months": 63,
    "6 Months": 126,
    "1 Year": 252,
}

# -----------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------

@st.cache_data(ttl=60 * 30, show_spinner=False)
def load_price_data(tickers, period="1y"):
    """Download adjusted close prices for a list of tickers."""
    raw = yf.download(
        tickers,
        period=period,
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=True,
    )

    closes = {}
    if isinstance(raw.columns, pd.MultiIndex):
        for t in tickers:
            try:
                closes[t] = raw[t]["Close"]
            except KeyError:
                continue
    else:
        # Single ticker case
        closes[tickers[0]] = raw["Close"]

    df = pd.DataFrame(closes).dropna(how="all")
    df = df.ffill().dropna()
    return df


def compute_returns(prices: pd.DataFrame, days: int) -> pd.Series:
    """% return over the trailing `days` trading days for each column."""
    if len(prices) <= days:
        days = len(prices) - 1
    if days <= 0:
        return pd.Series(0.0, index=prices.columns)
    return (prices.iloc[-1] / prices.iloc[-1 - days] - 1.0) * 100


def compute_relative_strength(prices: pd.DataFrame, benchmark: str) -> pd.DataFrame:
    """Normalize each sector's price by the benchmark price (RS line)."""
    rs = prices.div(prices[benchmark], axis=0)
    rs = rs / rs.iloc[0] * 100  # index to 100 at start of window
    return rs


def momentum_score(prices: pd.DataFrame, days: int, smooth: int = 5) -> pd.Series:
    """Rate of change of relative strength — used as the RRG-style 'momentum' axis."""
    rs = compute_relative_strength(prices, BENCHMARK)
    rs_smooth = rs.rolling(smooth, min_periods=1).mean()
    if len(rs_smooth) <= days:
        days = len(rs_smooth) - 1
    if days <= 0:
        return pd.Series(0.0, index=prices.columns)
    momentum = (rs_smooth.iloc[-1] / rs_smooth.iloc[-1 - days] - 1.0) * 100
    return momentum


def compute_acceleration(prices: pd.DataFrame, short_days: int, long_days: int) -> pd.DataFrame:
    """
    Compares the recent (short_days) return pace to the longer-term (long_days) return pace,
    both normalized to a 'per short_days window' basis, so they're directly comparable.

    Positive = the move is speeding up relative to its own longer-term trend (accelerating).
    Negative = the move is slowing down relative to its own longer-term trend (decelerating) —
    this applies whether the underlying return is positive (gains losing steam) or negative
    (losses getting worse, i.e. accelerating to the downside).

    Returns a DataFrame with columns: short_return, long_return, long_pace, accel_raw, accel_norm
    accel_norm is squashed to roughly [-100, 100] via tanh scaling for gauge display.
    """
    if long_days <= short_days:
        # Guard against misconfiguration (e.g. UI state where long window <= short window) —
        # fall back to treating the long window as at least 2x the short window.
        long_days = short_days * 2

    short_ret = compute_returns(prices, short_days)
    long_ret = compute_returns(prices, long_days)
    # Pace of the long window, rescaled to the same time basis as the short window
    long_pace = long_ret * (short_days / long_days)
    accel_raw = short_ret - long_pace

    # Squash to a stable gauge range. Scale chosen so a ~10pp pace gap reads as a strong
    # (but not pegged) reading; tanh keeps extreme outliers from clipping the gauge silently.
    accel_norm = np.tanh(accel_raw / 10.0) * 100

    return pd.DataFrame({
        "short_return": short_ret,
        "long_return": long_ret,
        "long_pace": long_pace,
        "accel_raw": accel_raw,
        "accel_norm": accel_norm,
    })


# -----------------------------------------------------------------------
# Sidebar controls
# -----------------------------------------------------------------------

st.sidebar.title("🔄 Sector Rotation")
st.sidebar.markdown("Settings")

history_period = st.sidebar.selectbox(
    "History to download",
    options=["6mo", "1y", "2y", "5y"],
    index=1,
)

rank_lookback_label = st.sidebar.selectbox(
    "Rank sectors by",
    options=list(LOOKBACKS.keys()),
    index=2,  # 3 Months default
)
rank_lookback = LOOKBACKS[rank_lookback_label]

quadrant_lookback_label = st.sidebar.selectbox(
    "Rotation quadrant lookback (momentum)",
    options=list(LOOKBACKS.keys()),
    index=1,  # 1 Month default
)
quadrant_lookback = LOOKBACKS[quadrant_lookback_label]

accel_short_label = st.sidebar.selectbox(
    "Acceleration: short window",
    options=list(LOOKBACKS.keys()),
    index=1,  # 1 Month default
    help="Recent pace. Compared against the long window below to detect speeding up / slowing down.",
)
accel_short = LOOKBACKS[accel_short_label]

_long_options = [l for l in LOOKBACKS.keys() if LOOKBACKS[l] > accel_short]
accel_long_label = st.sidebar.selectbox(
    "Acceleration: long window",
    options=_long_options if _long_options else list(LOOKBACKS.keys()),
    index=0,  # first available window longer than the short window (3 Months by default)
    help="Baseline trend pace. Short window's pace is compared against this one.",
)
accel_long = LOOKBACKS[accel_long_label]

include_extras = st.sidebar.multiselect(
    "Sub-industries to include",
    options=list(EXTRA_INDUSTRIES.keys()),
    default=list(EXTRA_INDUSTRIES.keys()),
    format_func=lambda t: f"{t} — {EXTRA_INDUSTRIES[t]}",
    help="These overlap with one of the 11 sectors above (e.g. XBI is part of XLV) "
         "but are shown separately since they can move very differently.",
)

st.sidebar.markdown("---")
if st.sidebar.button("🔁 Refresh data (clear cache)"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.caption(
    f"Last loaded: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    "Data via Yahoo Finance (yfinance). Free-tier data may be delayed."
)

# -----------------------------------------------------------------------
# Load data
# -----------------------------------------------------------------------

all_tickers = list(SECTOR_ETFS.keys()) + list(include_extras) + [BENCHMARK]

with st.spinner("Downloading price data..."):
    try:
        prices = load_price_data(all_tickers, period=history_period)
    except Exception as e:
        st.error(f"Failed to download data: {e}")
        st.stop()

if prices.empty or BENCHMARK not in prices.columns:
    st.error(
        "No data returned. Check your internet connection or that the "
        "tickers are valid. If you're behind a restrictive network/proxy, "
        "yfinance may be blocked."
    )
    st.stop()

missing = [t for t in all_tickers if t not in prices.columns]
if missing:
    st.warning(f"Could not load data for: {', '.join(missing)}")

available_sectors = {k: v for k, v in SECTOR_ETFS.items() if k in prices.columns}
available_extras = {k: v for k, v in EXTRA_INDUSTRIES.items() if k in prices.columns and k in include_extras}

# -----------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------

st.title("Sector Rotation Momentum Dashboard")
st.caption(
    "Relative strength and momentum across the 11 S&P 500 sectors, "
    f"benchmarked against {BENCHMARK}. Not investment advice."
)

spy_last = prices[BENCHMARK].iloc[-1]
spy_chg_1d = (prices[BENCHMARK].iloc[-1] / prices[BENCHMARK].iloc[-2] - 1) * 100 if len(prices) > 1 else 0
c1, c2, c3 = st.columns(3)
c1.metric("SPY (last close)", f"${spy_last:,.2f}", f"{spy_chg_1d:+.2f}%")
c2.metric("Data range", f"{prices.index[0].date()} → {prices.index[-1].date()}")
c3.metric("Sectors tracked", f"{len(available_sectors)} / 11")

# -----------------------------------------------------------------------
# Acceleration (money in/out): short-window pace vs long-window pace
# -----------------------------------------------------------------------

accel_df = compute_acceleration(prices, accel_short, accel_long)

def accel_glyph(norm_val: float) -> str:
    """Arrow glyph for a normalized acceleration score in roughly [-100, 100]."""
    if norm_val >= 60:
        return "⬆️⬆️"
    elif norm_val >= 20:
        return "⬆️"
    elif norm_val > -20:
        return "➡️"
    elif norm_val > -60:
        return "⬇️"
    else:
        return "⬇️⬇️"

def accel_label(norm_val: float) -> str:
    if norm_val >= 60:
        return "Strongly accelerating"
    elif norm_val >= 20:
        return "Accelerating"
    elif norm_val > -20:
        return "Steady"
    elif norm_val > -60:
        return "Decelerating"
    else:
        return "Strongly decelerating"

# -----------------------------------------------------------------------
# Leaderboard table
# -----------------------------------------------------------------------

st.subheader("📊 Sector Leaderboard")

rows = []
for ticker, name in available_sectors.items():
    row = {"Ticker": ticker, "Sector": name, "Type": "Sector"}
    for label, days in LOOKBACKS.items():
        row[label] = compute_returns(prices[[ticker]], days)[ticker]
    row["Accel"] = accel_glyph(accel_df.loc[ticker, "accel_norm"])
    row["Accel Score"] = accel_df.loc[ticker, "accel_norm"]
    rows.append(row)

leaderboard = pd.DataFrame(rows)
leaderboard = leaderboard.sort_values(rank_lookback_label, ascending=False).reset_index(drop=True)
leaderboard.insert(0, "Rank", range(1, len(leaderboard) + 1))

# Extra sub-industry rows (e.g. XBI) — ranked among themselves, not mixed into
# the 11-sector rank numbers above, since they're not a GICS sector.
extra_rows = []
for ticker, name in available_extras.items():
    row = {"Rank": "—", "Ticker": ticker, "Sector": name, "Type": "Sub-Industry"}
    for label, days in LOOKBACKS.items():
        row[label] = compute_returns(prices[[ticker]], days)[ticker]
    row["Accel"] = accel_glyph(accel_df.loc[ticker, "accel_norm"])
    row["Accel Score"] = accel_df.loc[ticker, "accel_norm"]
    extra_rows.append(row)

# Add SPY row for reference
spy_row = {"Rank": "—", "Ticker": BENCHMARK, "Sector": "S&P 500 (Benchmark)", "Type": "Benchmark"}
for label, days in LOOKBACKS.items():
    spy_row[label] = compute_returns(prices[[BENCHMARK]], days)[BENCHMARK]
spy_row["Accel"] = accel_glyph(accel_df.loc[BENCHMARK, "accel_norm"])
spy_row["Accel Score"] = accel_df.loc[BENCHMARK, "accel_norm"]

leaderboard_display = pd.concat(
    [leaderboard, pd.DataFrame(extra_rows), pd.DataFrame([spy_row])],
    ignore_index=True,
)

def style_returns(val):
    if isinstance(val, (int, float)):
        color = "#1a9850" if val > 0 else "#d73027" if val < 0 else "#888"
        return f"color: {color}; font-weight: 600"
    return ""

fmt_dict = {label: "{:+.2f}%" for label in LOOKBACKS.keys()}
fmt_dict["Accel Score"] = "{:+.0f}"
styler = leaderboard_display.style.format(fmt_dict)
# pandas >= 2.1 renamed Styler.applymap -> Styler.map; pandas 3.x removed applymap entirely.
if hasattr(styler, "map"):
    styled = styler.map(style_returns, subset=list(LOOKBACKS.keys()) + ["Accel Score"])
else:
    styled = styler.applymap(style_returns, subset=list(LOOKBACKS.keys()) + ["Accel Score"])
st.dataframe(styled, use_container_width=True, hide_index=True)
st.caption(
    f"Accel = pace of the last {accel_short_label.lower()} vs. the trend pace implied by the last "
    f"{accel_long_label.lower()}. ⬆️⬆️/⬆️ = speeding up (whether gaining or, for negative returns, "
    f"losing ground faster). ⬇️/⬇️⬇️ = slowing down (cooling off, or losses easing)."
)

leading = leaderboard.iloc[0]
lagging = leaderboard.iloc[-1]
st.markdown(
    f"**Leading sector ({rank_lookback_label}):** {leading['Sector']} ({leading['Ticker']}) "
    f"at {leading[rank_lookback_label]:+.2f}% &nbsp;&nbsp;|&nbsp;&nbsp; "
    f"**Lagging sector:** {lagging['Sector']} ({lagging['Ticker']}) "
    f"at {lagging[rank_lookback_label]:+.2f}%"
)

if len(extra_rows) > 0:
    extra_bits = " &nbsp;&nbsp;|&nbsp;&nbsp; ".join(
        f"**{r['Sector'].split(' (')[0]} ({r['Ticker']}):** {r[rank_lookback_label]:+.2f}%"
        for r in extra_rows
    )
    st.caption(f"Sub-industries (not ranked against sectors above): {extra_bits}")

# -----------------------------------------------------------------------
# Plain-English rotation summary
# -----------------------------------------------------------------------

st.subheader("📝 What's the money doing?")

summary_window_label = st.selectbox(
    "Summarize over",
    options=list(LOOKBACKS.keys()),
    index=2,  # 3 Months default
    key="summary_window",
)
summary_days = LOOKBACKS[summary_window_label]

# Thresholds for calling a sector "unchanged" vs a real mover. Scaled loosely by
# window length so "flat" means something different over 1 week vs 1 year.
flat_threshold = max(1.0, 0.6 * (summary_days / 21))  # ~1% for short windows, scales up for longer ones

# Acceleration for the write-up is tied to the window the user picked here, not the
# sidebar's accel windows — otherwise the "still picking up pace" commentary could
# describe a different timeframe than the one being summarized.
_longer_windows = [d for d in LOOKBACKS.values() if d > summary_days]
summary_long_days = min(_longer_windows) if _longer_windows else summary_days * 3
summary_accel_df = compute_acceleration(prices, summary_days, summary_long_days)

summary_rows = []
for ticker, name in list(available_sectors.items()) + list(available_extras.items()):
    ret = compute_returns(prices[[ticker]], summary_days)[ticker]
    accel = summary_accel_df.loc[ticker, "accel_norm"] if ticker in summary_accel_df.index else 0.0
    is_extra = ticker in available_extras
    summary_rows.append({"Ticker": ticker, "Name": name, "Return": ret, "Accel": accel, "IsExtra": is_extra})

summary_df = pd.DataFrame(summary_rows)

inflow = summary_df[(summary_df["Return"] > flat_threshold)].sort_values("Return", ascending=False)
outflow = summary_df[(summary_df["Return"] < -flat_threshold)].sort_values("Return", ascending=True)
unchanged = summary_df[(summary_df["Return"].abs() <= flat_threshold)].sort_values("Return", ascending=False)

def _fmt_list(df: pd.DataFrame) -> str:
    parts = []
    for _, r in df.iterrows():
        tag = " (sub-industry)" if r["IsExtra"] else ""
        parts.append(f"**{r['Name'].split(' (')[0]}** ({r['Ticker']}{tag}, {r['Return']:+.1f}%)")
    if len(parts) == 0:
        return ""
    if len(parts) == 1:
        return parts[0]
    return ", ".join(parts[:-1]) + (", and " if len(parts) > 2 else " and ") + parts[-1]

summary_lines = []

if not inflow.empty:
    accelerating_in = inflow[inflow["Accel"] >= 20]
    line = f"Over the last **{summary_window_label.lower()}**, money has been moving into "
    line += _fmt_list(inflow) + "."
    summary_lines.append(line)
    if not accelerating_in.empty:
        summary_lines.append(
            f"Of those, {_fmt_list(accelerating_in)} {'is' if len(accelerating_in)==1 else 'are'} "
            f"still picking up pace — the move looks like it's accelerating, not just coasting."
        )

if not outflow.empty:
    line = f"Money has been leaving " + _fmt_list(outflow) + "."
    summary_lines.append(line)
    worsening = outflow[outflow["Accel"] < -20]
    if not worsening.empty:
        summary_lines.append(
            f"{_fmt_list(worsening)} {'is' if len(worsening)==1 else 'are'} decelerating on top of "
            f"already being negative, meaning the outflow itself looks like it's speeding up, not bottoming."
        )

if not unchanged.empty:
    if not inflow.empty or not outflow.empty:
        line = "Meanwhile, "
    else:
        line = f"Over the last **{summary_window_label.lower()}**, "
    line += _fmt_list(unchanged)
    line += f" {'has' if len(unchanged)==1 else 'have'} been roughly flat — no clear rotation in or out."
    summary_lines.append(line)

if summary_lines:
    st.markdown(" ".join(summary_lines))
else:
    st.markdown("No sectors moved enough to call a clear rotation over this window.")

st.caption(
    f"\"Flat\" here means a return within ±{flat_threshold:.1f}% over {summary_window_label.lower()}. "
    "This summary describes what already happened in the price data — it isn't a prediction or a "
    "recommendation to buy or sell anything."
)

# -----------------------------------------------------------------------
# Acceleration gauges (money in/out)
# -----------------------------------------------------------------------

st.subheader("🌡️ Acceleration Gauges — Money In / Out")
st.caption(
    f"Each gauge compares the last **{accel_short_label.lower()}**'s pace of return to the trend "
    f"pace implied by the last **{accel_long_label.lower()}**. Right of center = accelerating "
    "(money flowing in faster, or for a sector already falling, flowing out faster). "
    "Left of center = decelerating (cooling off, or losses easing). The needle position is the "
    "score; it does NOT mean the sector's return is positive or negative — check the leaderboard "
    "for that."
)

gauge_universe = list(available_sectors.items()) + list(available_extras.items())
n_gauges = len(gauge_universe)
n_cols = 4 if n_gauges >= 4 else max(n_gauges, 1)
n_rows = -(-n_gauges // n_cols)  # ceiling division

# Sort so strongest accelerators and decelerators are easy to scan first
gauge_universe_sorted = sorted(
    gauge_universe,
    key=lambda kv: accel_df.loc[kv[0], "accel_norm"],
    reverse=True,
)

fig_gauges = make_subplots(
    rows=n_rows,
    cols=n_cols,
    specs=[[{"type": "indicator"} for _ in range(n_cols)] for _ in range(n_rows)],
    horizontal_spacing=0.05,
    vertical_spacing=0.18 if n_rows > 1 else 0.05,
)

GAUGE_STEPS = [
    {"range": [-100, -60], "color": "#d73027"},
    {"range": [-60, -20], "color": "#fc9272"},
    {"range": [-20, 20], "color": "#e0e0e0"},
    {"range": [20, 60], "color": "#a6d96a"},
    {"range": [60, 100], "color": "#1a9850"},
]

for i, (ticker, name) in enumerate(gauge_universe_sorted):
    r = i // n_cols + 1
    c = i % n_cols + 1
    score = accel_df.loc[ticker, "accel_norm"]
    needle_color = "#1a9850" if score > 20 else "#d73027" if score < -20 else "#888"
    fig_gauges.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=float(score),
            number={"suffix": "", "font": {"size": 18, "color": needle_color}},
            title={"text": f"{ticker}<br><span style='font-size:11px;color:gray'>{accel_label(score)}</span>", "font": {"size": 13}},
            gauge={
                "axis": {"range": [-100, 100], "tickvals": [-100, 0, 100], "ticktext": ["Out", "", "In"], "tickfont": {"size": 9}},
                "bar": {"color": needle_color, "thickness": 0.35},
                "steps": GAUGE_STEPS,
                "threshold": {"line": {"color": "black", "width": 2}, "thickness": 0.8, "value": float(score)},
            },
            domain={"row": r - 1, "column": c - 1},
        ),
        row=r,
        col=c,
    )

fig_gauges.update_layout(
    height=220 * n_rows,
    margin=dict(l=20, r=20, t=40, b=20),
    grid={"rows": n_rows, "columns": n_cols, "pattern": "independent"},
)
st.plotly_chart(fig_gauges, use_container_width=True)

# -----------------------------------------------------------------------
# Bar chart of current ranking
# -----------------------------------------------------------------------

st.subheader(f"📈 Returns by Sector — {rank_lookback_label}")

bar_df = leaderboard[["Sector", "Ticker", rank_lookback_label]].copy()
bar_df["category"] = np.where(bar_df[rank_lookback_label] >= 0, "Positive", "Negative")

if available_extras:
    extra_bar_rows = []
    for ticker, name in available_extras.items():
        ret = compute_returns(prices[[ticker]], rank_lookback)[ticker]
        extra_bar_rows.append({"Sector": name, "Ticker": ticker, rank_lookback_label: ret, "category": "Sub-Industry"})
    bar_df = pd.concat([bar_df, pd.DataFrame(extra_bar_rows)], ignore_index=True)

bar_df = bar_df.sort_values(rank_lookback_label, ascending=True)

fig_bar = px.bar(
    bar_df,
    x=rank_lookback_label,
    y="Sector",
    orientation="h",
    color="category",
    color_discrete_map={"Positive": "#1a9850", "Negative": "#d73027", "Sub-Industry": "#8856a7"},
    text=bar_df[rank_lookback_label].map(lambda x: f"{x:+.2f}%"),
    labels={rank_lookback_label: "Return (%)"},
)
fig_bar.update_traces(textposition="outside")
fig_bar.update_layout(
    showlegend=False,
    height=450,
    margin=dict(l=10, r=10, t=10, b=10),
)
st.plotly_chart(fig_bar, use_container_width=True)

# -----------------------------------------------------------------------
# Relative strength lines vs SPY
# -----------------------------------------------------------------------

st.subheader("📉 Relative Strength vs SPY")
st.caption("Each line = sector price ÷ SPY price, indexed to 100 at the start of the window. Rising = outperforming SPY.")

rs_window_label = st.select_slider(
    "Chart window",
    options=list(LOOKBACKS.keys()),
    value="3 Months",
)
rs_days = LOOKBACKS[rs_window_label]
rs_days = min(rs_days, len(prices) - 1)
rs_prices = prices.iloc[-(rs_days + 1):]
rs_lines = compute_relative_strength(rs_prices, BENCHMARK)

all_rs_options = list(available_sectors.keys()) + list(available_extras.keys())
all_rs_labels = {**available_sectors, **available_extras}

selected_sectors = st.multiselect(
    "Sectors to show",
    options=all_rs_options,
    default=all_rs_options,
    format_func=lambda t: f"{t} — {all_rs_labels[t]}",
)

fig_rs = go.Figure()
for ticker in selected_sectors:
    is_extra = ticker in available_extras
    fig_rs.add_trace(
        go.Scatter(
            x=rs_lines.index,
            y=rs_lines[ticker],
            mode="lines",
            name=f"{ticker} ({all_rs_labels[ticker]})",
            line=dict(dash="dash" if is_extra else "solid", width=3 if is_extra else 2),
        )
    )
fig_rs.add_hline(y=100, line_dash="dash", line_color="gray", opacity=0.6)
fig_rs.update_layout(
    height=500,
    yaxis_title="Relative Strength (indexed to 100)",
    xaxis_title="Date",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    margin=dict(l=10, r=10, t=40, b=10),
)
st.plotly_chart(fig_rs, use_container_width=True)

# -----------------------------------------------------------------------
# Rotation quadrant (RRG-style)
# -----------------------------------------------------------------------

st.subheader("🧭 Rotation Quadrant")
st.caption(
    "X-axis: relative strength vs SPY (RS level, indexed to 100 = in-line with market). "
    "Y-axis: momentum (rate of change of relative strength). "
    "Inspired by Relative Rotation Graphs (RRG) — not an official RRG implementation."
)

quad_days = min(max(LOOKBACKS.values()), len(prices) - 1)
quad_prices = prices.iloc[-(quad_days + 1):]
rs_full = compute_relative_strength(quad_prices, BENCHMARK)
rs_current = rs_full.iloc[-1]
mom_current = momentum_score(quad_prices, quadrant_lookback)

quad_df = pd.DataFrame({
    "Ticker": list(available_sectors.keys()) + list(available_extras.keys()),
    "Sector": [available_sectors[t] for t in available_sectors.keys()] + [available_extras[t] for t in available_extras.keys()],
    "Type": (["Sector"] * len(available_sectors)) + (["Sub-Industry"] * len(available_extras)),
})
quad_df["RS"] = quad_df["Ticker"].map(rs_current)
quad_df["Momentum"] = quad_df["Ticker"].map(mom_current)
quad_df = quad_df.dropna()

def quadrant_label(rs, mom):
    if rs >= 100 and mom >= 0:
        return "Leading"
    elif rs >= 100 and mom < 0:
        return "Weakening"
    elif rs < 100 and mom < 0:
        return "Lagging"
    else:
        return "Improving"

quad_df["Quadrant"] = quad_df.apply(lambda r: quadrant_label(r["RS"], r["Momentum"]), axis=1)

quad_colors = {
    "Leading": "#1a9850",
    "Weakening": "#fee08b",
    "Lagging": "#d73027",
    "Improving": "#4575b4",
}

fig_quad = px.scatter(
    quad_df,
    x="RS",
    y="Momentum",
    color="Quadrant",
    symbol="Type",
    symbol_map={"Sector": "circle", "Sub-Industry": "diamond"},
    color_discrete_map=quad_colors,
    text="Ticker",
    hover_data={"Sector": True, "Type": True, "RS": ":.2f", "Momentum": ":.2f", "Quadrant": True},
)
fig_quad.update_traces(textposition="top center", marker=dict(size=14, line=dict(width=1, color="white")))
fig_quad.add_vline(x=100, line_dash="dash", line_color="gray", opacity=0.5)
fig_quad.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)

x_range = [quad_df["RS"].min() - 2, quad_df["RS"].max() + 2]
y_range = [quad_df["Momentum"].min() - 1, quad_df["Momentum"].max() + 1]
fig_quad.add_annotation(x=x_range[1], y=y_range[1], text="LEADING", showarrow=False, font=dict(color="#1a9850", size=12), xanchor="right", yanchor="top")
fig_quad.add_annotation(x=x_range[0], y=y_range[1], text="IMPROVING", showarrow=False, font=dict(color="#4575b4", size=12), xanchor="left", yanchor="top")
fig_quad.add_annotation(x=x_range[1], y=y_range[0], text="WEAKENING", showarrow=False, font=dict(color="#b8860b", size=12), xanchor="right", yanchor="bottom")
fig_quad.add_annotation(x=x_range[0], y=y_range[0], text="LAGGING", showarrow=False, font=dict(color="#d73027", size=12), xanchor="left", yanchor="bottom")

fig_quad.update_layout(
    height=550,
    xaxis_title="Relative Strength (100 = in-line with SPY)",
    yaxis_title=f"Momentum ({quadrant_lookback_label} RS rate-of-change)",
    margin=dict(l=10, r=10, t=10, b=10),
)
st.plotly_chart(fig_quad, use_container_width=True)

with st.expander("How to read the quadrant"):
    st.markdown(
        """
- **Leading** (top-right): outperforming SPY and still gaining momentum.
- **Weakening** (bottom-right): outperforming SPY but momentum is fading — often rotates into Lagging next.
- **Lagging** (bottom-left): underperforming SPY and still losing momentum.
- **Improving** (top-left): underperforming SPY but momentum is turning up — often rotates into Leading next.

Sectors tend to rotate clockwise through these quadrants over a full market cycle.
        """
    )

# -----------------------------------------------------------------------
# Raw data (optional)
# -----------------------------------------------------------------------

with st.expander("📋 Raw price data"):
    st.dataframe(prices.tail(30).sort_index(ascending=False), use_container_width=True)

st.caption(
    "Data source: Yahoo Finance via yfinance. For informational purposes only — not financial advice."
)
