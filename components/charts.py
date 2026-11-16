"""Reusable Plotly chart builders for cash flow, spending breakdowns, and portfolio visuals."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go

_PALETTE: list[str] = [
    "#3B82F6",
    "#10B981",
    "#F59E0B",
    "#8B5CF6",
    "#EC4899",
    "#06B6D4",
    "#F97316",
    "#14B8A6",
    "#6366F1",
    "#EF4444",
]


def _empty_figure(message: str, height: int = 380) -> go.Figure:
    """Return a clean empty-state Plotly figure without axes clutter."""
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(size=14, color="#94A3B8"),
    )
    fig.update_layout(
        height=height,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    return fig


def _apply_base_layout(fig: go.Figure, title: str, height: int = 420) -> go.Figure:
    """Apply consistent responsive dark/light-compatible Plotly styling with smooth transitions."""
    fig.update_layout(
        title=dict(text=title, font=dict(size=16), x=0.0, xanchor="left"),
        height=height,
        margin=dict(l=16, r=16, t=56, b=24),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        transition=dict(duration=550, easing="cubic-in-out"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1.0,
        ),
        hoverlabel=dict(namelength=-1),
    )
    return fig


def build_cash_flow_chart(
    monthly_df: pd.DataFrame,
    currency_symbol: str = "₹",
) -> go.Figure:
    """Build grouped Income vs. Expense bar chart overlaid with Net Cash Flow line."""
    if monthly_df is None or monthly_df.empty:
        return _empty_figure("No cash flow records match the current filter criteria.")

    required = {"month", "Income", "Expense", "Net Cash Flow"}
    if not required.issubset(monthly_df.columns):
        return _empty_figure("Cash flow columns unavailable.")

    months = monthly_df["month"].astype(str).tolist()
    incomes = pd.to_numeric(monthly_df["Income"], errors="coerce").fillna(0.0).tolist()
    expenses = pd.to_numeric(monthly_df["Expense"], errors="coerce").fillna(0.0).tolist()
    net_flow = pd.to_numeric(monthly_df["Net Cash Flow"], errors="coerce").fillna(0.0).tolist()

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="Income",
            x=months,
            y=incomes,
            marker=dict(color="#10B981", opacity=0.88),
            hovertemplate=f"Income: {currency_symbol}%{{y:,.2f}}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            name="Expense",
            x=months,
            y=expenses,
            marker=dict(color="#EF4444", opacity=0.88),
            hovertemplate=f"Expense: {currency_symbol}%{{y:,.2f}}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            name="Net Cash Flow",
            x=months,
            y=net_flow,
            mode="lines+markers",
            line=dict(color="#3B82F6", width=3),
            marker=dict(size=8, color="#3B82F6", line=dict(width=1.5, color="#FFFFFF")),
            hovertemplate=f"Net Cash Flow: {currency_symbol}%{{y:,.2f}}<extra></extra>",
        )
    )

    _apply_base_layout(fig, "Monthly Income vs. Expense & Net Cash Flow", height=430)
    fig.update_layout(
        barmode="group",
        bargap=0.22,
        bargroupgap=0.08,
        hovermode="x unified",
        xaxis=dict(title="Month", type="category", showgrid=False),
        yaxis=dict(
            title=f"Amount ({currency_symbol})",
            showgrid=True,
            gridcolor="rgba(148, 163, 184, 0.18)",
            zeroline=True,
            zerolinecolor="rgba(148, 163, 184, 0.35)",
        ),
    )
    return fig


def build_spending_donut_chart(
    breakdown_df: pd.DataFrame,
    currency_symbol: str = "₹",
) -> go.Figure:
    """Build a Plotly Donut Chart illustrating expense allocation by category."""
    if breakdown_df is None or not isinstance(breakdown_df, pd.DataFrame) or breakdown_df.empty:
        return _empty_figure("No expense transactions found for the selected filters.")

    work = breakdown_df.copy()
    if "amount" not in work.columns or "category" not in work.columns:
        return _empty_figure("No expense category data available.")

    work["amount"] = pd.to_numeric(work["amount"], errors="coerce").fillna(0.0)
    work = work.loc[work["amount"] > 0]
    if work.empty:
        return _empty_figure("No positive expense amounts available to visualize.")

    total_spend = float(work["amount"].sum())
    colors = (_PALETTE * ((len(work) // len(_PALETTE)) + 1))[: len(work)]
    fig = go.Figure(
        data=[
            go.Pie(
                labels=work["category"].astype(str).tolist(),
                values=work["amount"].tolist(),
                hole=0.58,
                marker=dict(colors=colors),
                textinfo="percent",
                textposition="inside",
                insidetextorientation="radial",
                hovertemplate=(
                    "<b>%{label}</b><br>"
                    f"Spend: {currency_symbol}%{{value:,.2f}}<br>"
                    "Share: %{percent}<extra></extra>"
                ),
            )
        ]
    )
    fig.add_annotation(
        text=f"<b>Total Spend</b><br>{currency_symbol}{total_spend:,.0f}",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=13),
    )
    _apply_base_layout(fig, "Expense Sector Allocation", height=410)
    fig.update_layout(
        legend=dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.02,
        )
    )
    return fig


def build_category_bar_chart(
    breakdown_df: pd.DataFrame,
    currency_symbol: str = "₹",
) -> go.Figure:
    """Build a horizontal bar chart ranking expense categories by total spend."""
    if breakdown_df is None or not isinstance(breakdown_df, pd.DataFrame) or breakdown_df.empty:
        return _empty_figure("No category spending data available.")

    work = breakdown_df.copy()
    if "amount" not in work.columns or "category" not in work.columns:
        return _empty_figure("No category spending data available.")

    work["amount"] = pd.to_numeric(work["amount"], errors="coerce").fillna(0.0)
    work = work.loc[work["amount"] > 0].copy()
    if work.empty:
        return _empty_figure("No positive expense amounts available.")

    if "percentage" in work.columns:
        work["percentage"] = pd.to_numeric(work["percentage"], errors="coerce").fillna(0.0)
    else:
        total_amt = float(work["amount"].sum())
        work["percentage"] = (
            np.round((work["amount"] / total_amt) * 100.0, 2) if total_amt > 0 else 0.0
        )
    work = work.head(10)

    fig = go.Figure(
        data=[
            go.Bar(
                x=work["amount"].tolist(),
                y=work["category"].astype(str).tolist(),
                orientation="h",
                marker=dict(color="#3B82F6", opacity=0.88),
                text=[
                    f"{currency_symbol}{amt:,.0f} ({pct:.1f}%)"
                    for amt, pct in zip(work["amount"], work["percentage"])
                ],
                textposition="auto",
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    f"Amount: {currency_symbol}%{{x:,.2f}}<extra></extra>"
                ),
            )
        ]
    )
    _apply_base_layout(fig, "Top Spending Categories", height=410)
    fig.update_layout(
        showlegend=False,
        xaxis=dict(
            title=f"Amount ({currency_symbol})",
            showgrid=True,
            gridcolor="rgba(148, 163, 184, 0.18)",
        ),
        yaxis=dict(autorange="reversed", showgrid=False),
    )
    return fig


def build_portfolio_treemap(
    portfolio_df: pd.DataFrame,
    currency_symbol: str = "₹",
) -> go.Figure:
    """Build a Plotly Treemap of capital deployed per asset ticker."""
    if portfolio_df is None or not isinstance(portfolio_df, pd.DataFrame) or portfolio_df.empty:
        return _empty_figure("No investment transactions recorded for this period.")

    work = portfolio_df.copy()
    if "total_invested" not in work.columns:
        return _empty_figure("No investment allocations to display.")

    if "asset_ticker" not in work.columns:
        work["asset_ticker"] = "Asset"
    work["total_invested"] = pd.to_numeric(work["total_invested"], errors="coerce").fillna(0.0)
    work["total_units"] = (
        pd.to_numeric(work["total_units"], errors="coerce").fillna(0.0)
        if "total_units" in work.columns
        else 0.0
    )
    work["avg_cost_basis"] = (
        pd.to_numeric(work["avg_cost_basis"], errors="coerce").fillna(0.0)
        if "avg_cost_basis" in work.columns
        else 0.0
    )
    work = work.loc[work["total_invested"] > 0].copy()

    if work.empty:
        return _empty_figure("No positive investment allocations to display.")

    total_cap = float(work["total_invested"].sum())
    if "allocation_pct" in work.columns:
        work["allocation_pct"] = pd.to_numeric(work["allocation_pct"], errors="coerce").fillna(0.0)
    else:
        work["allocation_pct"] = np.round((work["total_invested"] / total_cap) * 100.0, 2)

    tickers = work["asset_ticker"].fillna("UNSPECIFIED").astype(str).tolist()
    invested = work["total_invested"].tolist()
    alloc_pct = work["allocation_pct"].tolist()
    custom_data = np.column_stack(
        [
            work["total_units"].to_numpy(dtype=float),
            work["avg_cost_basis"].to_numpy(dtype=float),
            work["allocation_pct"].to_numpy(dtype=float),
        ]
    )

    fig = go.Figure(
        go.Treemap(
            ids=[f"{i}_{t}" for i, t in enumerate(tickers)],
            labels=tickers,
            parents=[""] * len(tickers),
            values=invested,
            customdata=custom_data,
            marker=dict(
                colors=alloc_pct,
                colorscale="Tealgrn",
                showscale=False,
            ),
            texttemplate=(
                "<b>%{label}</b><br>"
                f"{currency_symbol}%{{value:,.0f}}<br>"
                "%{customdata[2]:.1f}%"
            ),
            hovertemplate=(
                "<b>%{label}</b><br>"
                f"Capital Deployed: {currency_symbol}%{{value:,.2f}}<br>"
                "Portfolio Share: %{customdata[2]:.2f}%<br>"
                "Total Units: %{customdata[0]:,.4f}<br>"
                f"Avg Cost Basis: {currency_symbol}%{{customdata[1]:,.2f}}"
                "<extra></extra>"
            ),
        )
    )
    _apply_base_layout(fig, "Capital Deployed by Asset Ticker", height=430)
    fig.update_layout(margin=dict(l=12, r=12, t=52, b=16))
    return fig


def build_3d_spending_landscape(
    df: pd.DataFrame,
    currency_symbol: str = "₹",
) -> go.Figure:
    """Build an interactive 3D rotatable spending landscape (Month × Category × Spend)."""
    if df is None or not isinstance(df, pd.DataFrame) or df.empty or "date" not in df.columns:
        return _empty_figure("Add expense records across categories to explore the 3D spending cube.", height=450)

    work = df.copy()
    if "type" not in work.columns:
        work["type"] = "Expense"
    if "category" not in work.columns:
        work["category"] = "Other"

    # pyrefly: ignore[no-matching-overload]
    work["amount"] = pd.to_numeric(work.get("amount"), errors="coerce").fillna(0.0)
    work = work.loc[(work["type"] == "Expense") & (work["amount"] > 0)].copy()
    if work.empty:
        return _empty_figure("No positive expense data available for 3D rendering.", height=450)

    parsed_dates = pd.to_datetime(work["date"], format="mixed", errors="coerce")
    work = work.loc[parsed_dates.notna()].copy()
    if work.empty:
        return _empty_figure("No valid dated expenses available for 3D rendering.", height=450)

    work["month"] = parsed_dates.loc[work.index].dt.to_period("M").astype(str)
    work["category"] = work["category"].fillna("Other").astype(str)

    agg = (
        work.groupby(["month", "category"], as_index=False)[["amount"]]
        .sum()
        .sort_values(by=["month", "amount"], ascending=[True, False])
    )
    months_sorted = sorted(agg["month"].unique().tolist())
    cats_sorted = (
        agg.groupby("category")["amount"].sum().sort_values(ascending=False).index.tolist()
    )

    if agg.empty or not months_sorted or not cats_sorted:
        return _empty_figure("No positive expense data available for 3D rendering.", height=450)

    month_idx_map = {m: i for i, m in enumerate(months_sorted)}
    cat_idx_map = {c: i for i, c in enumerate(cats_sorted)}

    x_vals = [month_idx_map[m] for m in agg["month"]]
    y_vals = [cat_idx_map[c] for c in agg["category"]]
    z_vals = agg["amount"].tolist()

    max_amt = max(z_vals) if z_vals else 1.0
    sizes = [max(10.0, min(32.0, 10.0 + (amt / max_amt) * 22.0)) for amt in z_vals]

    fig = go.Figure()

    # Vertical 3D stem lines from floor (z=0) to each marker for clear spatial depth
    stem_x: list[Optional[float]] = []
    stem_y: list[Optional[float]] = []
    stem_z: list[Optional[float]] = []
    for xv, yv, zv in zip(x_vals, y_vals, z_vals):
        stem_x.extend([xv, xv, None])
        stem_y.extend([yv, yv, None])
        stem_z.extend([0.0, zv, None])

    fig.add_trace(
        go.Scatter3d(
            x=stem_x,
            y=stem_y,
            z=stem_z,
            mode="lines",
            line=dict(color="rgba(56, 189, 248, 0.45)", width=4),
            hoverinfo="skip",
            showlegend=False,
        )
    )

    fig.add_trace(
        go.Scatter3d(
            x=x_vals,
            y=y_vals,
            z=z_vals,
            mode="markers",
            marker=dict(
                size=sizes,
                color=z_vals,
                colorscale="Turbo",
                opacity=0.92,
                line=dict(width=1.2, color="rgba(255,255,255,0.65)"),
            ),
            customdata=np.column_stack([agg["month"].to_numpy(), agg["category"].to_numpy()]),
            hovertemplate=(
                "<b>%{customdata[1]}</b><br>"
                "Month: %{customdata[0]}<br>"
                f"Spend: {currency_symbol}%{{z:,.2f}}<extra></extra>"
            ),
            name="Category Spend",
            showlegend=False,
        )
    )

    _apply_base_layout(fig, "3D Interactive Spending Cube (Drag to Rotate)", height=470)
    fig.update_layout(
        margin=dict(l=0, r=0, t=52, b=0),
        scene=dict(
            bgcolor="rgba(0,0,0,0)",
            xaxis=dict(
                title="Month",
                tickvals=list(range(len(months_sorted))),
                ticktext=months_sorted,
                gridcolor="rgba(148, 163, 184, 0.2)",
                backgroundcolor="rgba(15, 23, 42, 0.12)",
            ),
            yaxis=dict(
                title="Category",
                tickvals=list(range(len(cats_sorted))),
                ticktext=cats_sorted,
                gridcolor="rgba(148, 163, 184, 0.2)",
                backgroundcolor="rgba(15, 23, 42, 0.12)",
            ),
            zaxis=dict(
                title=f"Spend ({currency_symbol})",
                gridcolor="rgba(148, 163, 184, 0.25)",
                backgroundcolor="rgba(15, 23, 42, 0.18)",
            ),
            camera=dict(eye=dict(x=1.55, y=-1.55, z=1.15)),
        ),
    )
    return fig


def build_health_score_gauge(
    score: Any,
    label: str = "Stable",
    grade: str = "B",
) -> go.Figure:
    """Render a 0-100 radial speedometer gauge for financial health score."""
    try:
        numeric = int(float(score if score is not None else 50))
    except (TypeError, ValueError):
        numeric = 50
    clean_score = max(0, min(100, numeric))
    bar_color = "#10B981" if clean_score >= 75 else ("#F59E0B" if clean_score >= 50 else "#EF4444")
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=clean_score,
            number={"suffix": "/100", "font": {"size": 28, "color": "#F8FAFC"}},
            title={"text": f"<b>Financial Health: {label} (Grade {grade})</b>", "font": {"size": 15, "color": "#94A3B8"}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#475569"},
                "bar": {"color": bar_color, "thickness": 0.28},
                "bgcolor": "rgba(15, 23, 42, 0.5)",
                "borderwidth": 1,
                "bordercolor": "rgba(255, 255, 255, 0.12)",
                "steps": [
                    {"range": [0, 50], "color": "rgba(239, 68, 68, 0.25)"},
                    {"range": [50, 75], "color": "rgba(245, 158, 11, 0.25)"},
                    {"range": [75, 100], "color": "rgba(16, 185, 129, 0.28)"},
                ],
            },
        )
    )
    fig.update_layout(
        height=240,
        margin=dict(l=24, r=24, t=48, b=16),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def build_50_30_20_chart(
    alloc: Optional[dict[str, Any]] = None,
    currency_symbol: str = "₹",
) -> go.Figure:
    """Build a stacked horizontal bar comparison of actual allocation vs 50/30/20 target."""
    safe_alloc = alloc if isinstance(alloc, dict) else {}
    try:
        needs_pct = float(safe_alloc.get("needs_pct", 0.0))
    except (TypeError, ValueError):
        needs_pct = 0.0
    try:
        wants_pct = float(safe_alloc.get("wants_pct", 0.0))
    except (TypeError, ValueError):
        wants_pct = 0.0
    try:
        savings_pct = float(safe_alloc.get("savings_pct", 0.0))
    except (TypeError, ValueError):
        savings_pct = 0.0

    fig = go.Figure()
    # Needs
    fig.add_trace(
        go.Bar(
            name="Needs (Target 50%)",
            y=["Your Allocation", "50/30/20 Benchmark"],
            x=[needs_pct, 50.0],
            orientation="h",
            marker=dict(color="#3B82F6", opacity=0.88),
            text=[f"{needs_pct:.1f}%", "50.0%"],
            textposition="auto",
        )
    )
    # Wants
    fig.add_trace(
        go.Bar(
            name="Wants (Target 30%)",
            y=["Your Allocation", "50/30/20 Benchmark"],
            x=[wants_pct, 30.0],
            orientation="h",
            marker=dict(color="#EC4899", opacity=0.88),
            text=[f"{wants_pct:.1f}%", "30.0%"],
            textposition="auto",
        )
    )
    # Savings
    fig.add_trace(
        go.Bar(
            name="Savings & Invest (Target 20%)",
            y=["Your Allocation", "50/30/20 Benchmark"],
            x=[savings_pct, 20.0],
            orientation="h",
            marker=dict(color="#10B981", opacity=0.88),
            text=[f"{savings_pct:.1f}%", "20.0%"],
            textposition="auto",
        )
    )

    fig.update_layout(
        barmode="stack",
        height=220,
        title=dict(text="50/30/20 Budget Rule Benchmark", font=dict(size=15), x=0.0),
        margin=dict(l=16, r=16, t=44, b=16),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="right", x=1.0),
        xaxis=dict(range=[0, 100], ticksuffix="%", showgrid=False),
        yaxis=dict(showgrid=False),
    )
    return fig
