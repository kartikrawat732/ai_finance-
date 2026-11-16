"""Reusable Streamlit UI components for auth, KPI cards, charts, and data entry forms."""

from components.auth_ui import render_auth_portal
from components.charts import (
    build_3d_spending_landscape,
    build_50_30_20_chart,
    build_cash_flow_chart,
    build_category_bar_chart,
    build_health_score_gauge,
    build_portfolio_treemap,
    build_spending_donut_chart,
)
from components.forms import (
    render_csv_uploader,
    render_manual_transaction_form,
    render_receipt_scanner,
)
from components.kpi_cards import format_currency, render_kpi_cards
from components.spatial_3d import get_spatial_css, render_spatial_canvas

__all__ = [
    "build_3d_spending_landscape",
    "build_50_30_20_chart",
    "build_cash_flow_chart",
    "build_category_bar_chart",
    "build_health_score_gauge",
    "build_portfolio_treemap",
    "build_spending_donut_chart",
    "format_currency",
    "get_spatial_css",
    "render_auth_portal",
    "render_csv_uploader",
    "render_kpi_cards",
    "render_manual_transaction_form",
    "render_receipt_scanner",
    "render_spatial_canvas",
]
