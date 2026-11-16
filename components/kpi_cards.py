"""Streamlit KPI metric cards for top-row financial telemetry."""

from __future__ import annotations

import inspect
import math
from typing import Any, Literal, Mapping, Optional

import pandas as pd
import streamlit as st

_METRIC_PARAMS = inspect.signature(st.metric).parameters
_SUPPORTS_BORDER = "border" in _METRIC_PARAMS
_SUPPORTS_SPARKLINE = "chart_data" in _METRIC_PARAMS


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Convert any scalar safely into a finite float."""
    if value is None:
        return default
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(numeric):
        return default
    return numeric


def format_currency(value: Any, symbol: str = "₹", decimals: Optional[int] = None) -> str:
    """Format a numeric currency value with sign handling and readable thousands separators."""
    numeric = _safe_float(value, default=0.0)
    sign = "-" if numeric < 0 else ""
    abs_val = abs(numeric)
    if decimals is not None:
        return f"{sign}{symbol}{abs_val:,.{decimals}f}"
    if abs(abs_val - round(abs_val)) < 1e-4:
        return f"{sign}{symbol}{abs_val:,.0f}"
    return f"{sign}{symbol}{abs_val:,.2f}"


def _clean_sparkline(df: Optional[pd.DataFrame], column: str) -> Optional[list[float]]:
    """Extract a finite float list from a DataFrame column for sparklines."""
    if df is None or df.empty or len(df) < 2 or column not in df.columns:
        return None
    series = pd.to_numeric(df[column], errors="coerce").fillna(0.0)
    return [float(v) if math.isfinite(float(v)) else 0.0 for v in series.tolist()]


def _render_metric(
    col: Any,
    *,
    label: str,
    value: str,
    delta: Optional[str] = None,
    delta_color: Literal["normal", "inverse", "off"] = "normal",
    help_text: Optional[str] = None,
    sparkline: Optional[list[float]] = None,
    chart_type: Literal["line", "bar", "area"] = "area",
) -> None:
    kwargs: dict[str, Any] = {
        "label": label,
        "value": value,
        "delta": delta,
        "delta_color": delta_color,
        "help": help_text,
    }
    if _SUPPORTS_BORDER:
        kwargs["border"] = True
    if _SUPPORTS_SPARKLINE and sparkline and len(sparkline) >= 2:
        kwargs["chart_data"] = sparkline
        kwargs["chart_type"] = chart_type
    with col:
        st.metric(**kwargs)


def render_kpi_cards(
    kpis: Mapping[str, Any],
    monthly_df: Optional[pd.DataFrame] = None,
    currency_symbol: str = "₹",
) -> None:
    """Render the 4 top-row KPI cards: Total Inflow, Total Expenses, Investments, Savings Rate."""
    inflow = _safe_float(kpis.get("total_inflow", 0.0))
    outflow = _safe_float(kpis.get("total_outflow", 0.0))
    invested = _safe_float(kpis.get("capital_deployed", 0.0))
    net_savings = _safe_float(kpis.get("net_savings", 0.0))
    savings_rate = _safe_float(kpis.get("savings_rate", 0.0))
    tx_count = int(_safe_float(kpis.get("transaction_count", 0)))

    inflow_series = _clean_sparkline(monthly_df, "Income")
    expense_series = _clean_sparkline(monthly_df, "Expense")
    invest_series = _clean_sparkline(monthly_df, "Investment")
    net_series = _clean_sparkline(monthly_df, "Net Cash Flow")

    expense_share = f"{(outflow / inflow) * 100.0:.1f}% of inflow" if inflow > 0 else "No inflow logged"
    invest_share = f"{(invested / inflow) * 100.0:.1f}% of inflow" if inflow > 0 else "Capital deployed"

    cols = st.columns(4, gap="medium")

    _render_metric(
        cols[0],
        label="Total Inflow",
        value=format_currency(inflow, currency_symbol),
        delta=f"{tx_count} transactions",
        delta_color="off",
        help_text="Total verified Income inflows in the selected timeframe.",
        sparkline=inflow_series,
        chart_type="area",
    )

    _render_metric(
        cols[1],
        label="Total Expenses",
        value=format_currency(outflow, currency_symbol),
        delta=expense_share,
        delta_color="off",
        help_text="Total operating outflows (excluding investment capital deployments).",
        sparkline=expense_series,
        chart_type="bar",
    )

    _render_metric(
        cols[2],
        label="Investments",
        value=format_currency(invested, currency_symbol),
        delta=invest_share,
        delta_color="off",
        help_text="Total capital deployed into equities, ETFs, funds, and digital assets.",
        sparkline=invest_series,
        chart_type="bar",
    )

    _render_metric(
        cols[3],
        label="Savings Rate",
        value=f"{savings_rate:.1f}%",
        delta=f"Net {format_currency(net_savings, currency_symbol)}",
        delta_color="normal",
        help_text="Net Savings (Total Inflow - Total Expenses) divided by Total Inflow.",
        sparkline=net_series,
        chart_type="line",
    )
