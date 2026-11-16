"""Main Streamlit entrypoint with 3D animated UI, email/phone auth, and financial dashboard."""

from __future__ import annotations

import datetime as dt
import hashlib
import html
import re
import sys
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import streamlit as st

_APP_DIR = Path(__file__).resolve().parent
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

try:
    from ai_engine import (
        chat_financial_copilot,
        generate_financial_insights,
        is_gemini_configured,
    )
except ImportError:
    import importlib
    import ai_engine
    importlib.reload(ai_engine)
    from ai_engine import (
        chat_financial_copilot,
        generate_financial_insights,
        is_gemini_configured,
    )
from analytics import (
    build_ai_summary_payload,
    compute_50_30_20,
    compute_budget_alerts,
    compute_category_breakdown,
    compute_financial_health_score,
    compute_kpis,
    compute_monthly_cash_flow,
    compute_portfolio_allocation,
    compute_top_expenses,
    detect_recurring_subscriptions,
)
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
from database import (
    ALLOWED_TRANSACTION_TYPES,
    delete_transactions,
    get_db_target_info,
    get_filter_metadata,
    get_transactions,
    init_db,
    seed_sample_data,
    update_transaction,
)

st.set_page_config(
    page_title="AI-Powered Personal Finance & Investment Tracker",
    page_icon="💹",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _inject_custom_styles(*, authenticated: bool) -> None:
    """Apply high-end 3D spatial glassmorphic styling and ambient lighting."""
    st.markdown(get_spatial_css(authenticated=authenticated), unsafe_allow_html=True)


def _resolve_date_range(
    preset: str,
    metadata: dict[str, Any],
) -> tuple[Optional[dt.date], Optional[dt.date]]:
    """Compute start and end dates from sidebar preset or custom range picker."""
    today = dt.date.today()

    if preset == "Current Month":
        return today.replace(day=1), today

    if preset == "YTD":
        return dt.date(today.year, 1, 1), today

    if preset == "Last 90 Days":
        return today - dt.timedelta(days=90), today

    if preset == "All Time":
        min_str = metadata.get("min_date")
        max_str = metadata.get("max_date")
        if min_str and max_str:
            return dt.date.fromisoformat(min_str), dt.date.fromisoformat(max_str)
        return today - dt.timedelta(days=365), today

    # Custom Range
    default_start = today - dt.timedelta(days=90)
    picked = st.sidebar.date_input(
        "Custom Date Window",
        value=(default_start, today),
        max_value=today + dt.timedelta(days=365),
        key="custom_date_range_input",
    )
    if isinstance(picked, (tuple, list)):
        if len(picked) == 2:
            s_dt, e_dt = picked
            return (min(s_dt, e_dt), max(s_dt, e_dt))
        if len(picked) == 1:
            return picked[0], picked[0]
    if isinstance(picked, dt.date):
        return picked, picked
    return default_start, today


def _dataframe_fingerprint(df: pd.DataFrame) -> str:
    """Create a deterministic hash of filtered dataframe contents for AI insight caching."""
    if df.empty:
        return "empty"
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    return hashlib.sha256(csv_bytes).hexdigest()


def _render_cash_flow_tab(monthly_df: pd.DataFrame, currency_symbol: str) -> None:
    st.plotly_chart(
        build_cash_flow_chart(monthly_df, currency_symbol=currency_symbol),
        width="stretch",
    )

    st.subheader("Monthly Cash Flow Matrix")
    if monthly_df.empty:
        st.info("No monthly cash flow data available for the active filter window.")
        return

    st.dataframe(
        monthly_df,
        width="stretch",
        hide_index=True,
        column_config={
            "month": st.column_config.TextColumn("Month"),
            "Income": st.column_config.NumberColumn("Income", format=f"{currency_symbol}%,.2f"),
            "Expense": st.column_config.NumberColumn("Expenses", format=f"{currency_symbol}%,.2f"),
            "Investment": st.column_config.NumberColumn(
                "Investments", format=f"{currency_symbol}%,.2f"
            ),
            "Net Cash Flow": st.column_config.NumberColumn(
                "Net Cash Flow", format=f"{currency_symbol}%,.2f"
            ),
            "Investable Surplus": st.column_config.NumberColumn(
                "Post-Invest Surplus", format=f"{currency_symbol}%,.2f"
            ),
        },
    )


def _render_spending_tab(df: pd.DataFrame, currency_symbol: str) -> None:
    # 1. Active Budget Warnings & Surge Alerts
    alerts_df = compute_budget_alerts(df)
    if not alerts_df.empty:
        surges = alerts_df[alerts_df["severity"].isin(["critical", "warning"])]
        if not surges.empty:
            with st.expander(f"⚠️ Active Budget & Spending Alerts ({len(surges)} Categories Exceeding Baseline)", expanded=True):
                for _, row in surges.iterrows():
                    icon = "🚨" if row["severity"] == "critical" else "⚠️"
                    badge_style = "color:#f87171;" if row["severity"] == "critical" else "color:#fbbf24;"
                    st.markdown(
                        f"""
                        <div style="padding: 8px 12px; margin-bottom: 6px; border-radius: 8px; background: rgba(15, 23, 42, 0.6); border: 1px solid rgba(255,255,255,0.08);">
                            {icon} <b style="{badge_style}">{html.escape(str(row['category']))}</b>: 
                            Current month spend is <b>{currency_symbol}{row['current_spend']:,.2f}</b> 
                            (<span style="{badge_style} font-weight:bold;">{row['pct_of_average']:.1f}%</span> of historical average of {currency_symbol}{row['monthly_average']:,.2f}).
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

    breakdown_df = compute_category_breakdown(df, trans_type="Expense")
    top_expenses_df = compute_top_expenses(df, top_n=25)

    col_left, col_right = st.columns(2, gap="medium")
    with col_left:
        st.plotly_chart(
            build_spending_donut_chart(breakdown_df, currency_symbol=currency_symbol),
            width="stretch",
        )
    with col_right:
        st.plotly_chart(
            build_category_bar_chart(breakdown_df, currency_symbol=currency_symbol),
            width="stretch",
        )

    st.plotly_chart(
        build_3d_spending_landscape(df, currency_symbol=currency_symbol),
        width="stretch",
    )

    tbl_col1, tbl_col2 = st.columns([1, 1.35], gap="medium")
    with tbl_col1:
        st.subheader("Category Allocation")
        if breakdown_df.empty:
            st.info("No expense categories recorded in this period.")
        else:
            st.dataframe(
                breakdown_df,
                width="stretch",
                hide_index=True,
                column_config={
                    "category": st.column_config.TextColumn("Category"),
                    "amount": st.column_config.NumberColumn(
                        "Total Spend", format=f"{currency_symbol}%,.2f"
                    ),
                    "percentage": st.column_config.ProgressColumn(
                        "Share (%)", min_value=0.0, max_value=100.0, format="%.1f%%"
                    ),
                    "transactions": st.column_config.NumberColumn("Txns", format="%d"),
                },
            )

    with tbl_col2:
        st.subheader("Top Expenses")
        if top_expenses_df.empty:
            st.info("No individual expenses recorded in this period.")
        else:
            st.dataframe(
                top_expenses_df,
                width="stretch",
                hide_index=True,
                column_config={
                    "date": st.column_config.TextColumn("Date"),
                    "category": st.column_config.TextColumn("Category"),
                    "description": st.column_config.TextColumn("Description"),
                    "amount": st.column_config.NumberColumn(
                        "Amount", format=f"{currency_symbol}%,.2f"
                    ),
                    "share_pct": st.column_config.NumberColumn("Share %", format="%.2f%%"),
                    "is_ai_tagged": st.column_config.CheckboxColumn("AI Tagged"),
                },
            )

    subs_df = detect_recurring_subscriptions(df)
    if not subs_df.empty:
        total_sub_annual = float(subs_df["annualized_cost"].sum())
        with st.expander(
            f"💳 Detected Subscriptions & Annualized Burn ({currency_symbol}{total_sub_annual:,.2f}/yr)",
            expanded=False,
        ):
            st.caption("Identified regular recurring expenses and active memberships with projected annual drain.")
            st.dataframe(
                subs_df,
                width="stretch",
                hide_index=True,
                column_config={
                    "service_name": st.column_config.TextColumn("Subscription / Service"),
                    "category": st.column_config.TextColumn("Category"),
                    "frequency": st.column_config.TextColumn("Cadence"),
                    "avg_amount": st.column_config.NumberColumn("Avg Cost", format=f"{currency_symbol}%,.2f"),
                    "annualized_cost": st.column_config.NumberColumn("Annual Burn", format=f"{currency_symbol}%,.2f"),
                    "occurrences": st.column_config.NumberColumn("Recorded Bills", format="%d"),
                },
            )


def _render_portfolio_tab(df: pd.DataFrame, currency_symbol: str) -> None:
    portfolio_df = compute_portfolio_allocation(df, fetch_live_prices=True)

    if not portfolio_df.empty:
        total_cap = float(portfolio_df["total_invested"].sum())
        cur_val = float(portfolio_df["current_value"].sum()) if "current_value" in portfolio_df.columns else total_cap
        tot_pnl = float(portfolio_df["unrealized_pnl"].sum()) if "unrealized_pnl" in portfolio_df.columns else 0.0
        ret_pct = (tot_pnl / total_cap * 100.0) if total_cap > 0 else 0.0

        p_cols = st.columns(4, gap="medium")
        p_cols[0].metric("Capital Deployed", format_currency(total_cap, currency_symbol))
        p_cols[1].metric("Current Market Value", format_currency(cur_val, currency_symbol))
        p_cols[2].metric(
            "Unrealized Gain/Loss",
            format_currency(tot_pnl, currency_symbol),
            delta=f"{ret_pct:+.2f}%",
        )
        p_cols[3].metric("Active Holdings", f"{len(portfolio_df)}")

    st.plotly_chart(
        build_portfolio_treemap(portfolio_df, currency_symbol=currency_symbol),
        width="stretch",
    )

    st.subheader("Live Holdings & Performance Matrix")
    if portfolio_df.empty:
        st.info("No investment transactions found for the selected timeframe.")
        return

    st.dataframe(
        portfolio_df,
        width="stretch",
        hide_index=True,
        column_config={
            "asset_ticker": st.column_config.TextColumn("Ticker"),
            "total_invested": st.column_config.NumberColumn(
                "Capital Deployed", format=f"{currency_symbol}%,.2f"
            ),
            "total_units": st.column_config.NumberColumn("Total Units", format="%.4f"),
            "avg_cost_basis": st.column_config.NumberColumn(
                "Avg Cost Basis", format=f"{currency_symbol}%,.2f"
            ),
            "current_price": st.column_config.NumberColumn(
                "Live Market Price", format=f"{currency_symbol}%,.2f"
            ),
            "current_value": st.column_config.NumberColumn(
                "Market Value", format=f"{currency_symbol}%,.2f"
            ),
            "unrealized_pnl": st.column_config.NumberColumn(
                "Unrealized P&L", format=f"{currency_symbol}%,.2f"
            ),
            "return_pct": st.column_config.NumberColumn(
                "Return (%)", format="%+.2f%%"
            ),
            "allocation_pct": st.column_config.ProgressColumn(
                "Allocation (%)", min_value=0.0, max_value=100.0, format="%.1f%%"
            ),
            "transactions": st.column_config.NumberColumn("Buys", format="%d"),
        },
    )


def _rows_differ(orig_row: pd.Series, new_row: pd.Series, cols: list[str]) -> bool:
    """Compare original and edited ledger rows safely across text, date, and numeric types."""
    for col in cols:
        old_val = orig_row[col]
        new_val = new_row[col]
        old_na = pd.isna(old_val)
        new_na = pd.isna(new_val)
        if old_na and new_na:
            continue
        if col in {"amount", "quantity", "unit_price"}:
            old_num = 0.0 if old_na else float(old_val)
            new_num = 0.0 if new_na else float(new_val)
            if abs(old_num - new_num) > 1e-6:
                return True
        else:
            old_str = "" if old_na else str(old_val).strip()
            new_str = "" if new_na else str(new_val).strip()
            if old_str != new_str:
                return True
    return False


def _render_ledger_tab(
    df: pd.DataFrame,
    currency_symbol: str,
    categories: list[str],
    user_id: int,
) -> None:
    st.subheader("Interactive Transaction Ledger")

    f_col1, f_col2, f_col3 = st.columns([2, 1, 1.2], gap="medium")
    with f_col1:
        search_query = st.text_input(
            "Search description, category, or ticker",
            placeholder="Filter transactions...",
            key="ledger_search_input",
        )
    with f_col2:
        type_filter = st.multiselect(
            "Filter Type",
            options=list(ALLOWED_TRANSACTION_TYPES),
            default=[],
            key="ledger_type_filter",
        )
    with f_col3:
        cat_filter = st.multiselect(
            "Filter Category",
            options=categories,
            default=[],
            key="ledger_category_filter",
        )

    view_df = df.copy()
    if not view_df.empty:
        if type_filter:
            view_df = view_df.loc[view_df["type"].isin(type_filter)]
        if cat_filter:
            view_df = view_df.loc[view_df["category"].isin(cat_filter)]
        if search_query.strip():
            q = search_query.strip().lower()
            desc_match = (
                view_df["description"].fillna("").astype(str).str.lower().str.contains(q, regex=False)
            )
            cat_match = (
                view_df["category"].fillna("").astype(str).str.lower().str.contains(q, regex=False)
            )
            tkr_match = (
                view_df["asset_ticker"].fillna("").astype(str).str.lower().str.contains(q, regex=False)
            )
            view_df = view_df.loc[desc_match | cat_match | tkr_match]

    if view_df.empty:
        st.info("No transactions match the current date range and ledger filters.")
        return

    editor_df = view_df.copy().reset_index(drop=True)
    editor_df.insert(0, "delete", False)
    editor_df["date"] = pd.to_datetime(editor_df["date"], errors="coerce").dt.date
    editor_df["is_ai_tagged"] = editor_df["is_ai_tagged"].astype(bool)
    editor_df["asset_ticker"] = editor_df["asset_ticker"].fillna("")

    editor_key = f"ledger_data_editor_{_dataframe_fingerprint(view_df)[:12]}"
    edited_df = st.data_editor(
        editor_df,
        width="stretch",
        hide_index=True,
        num_rows="fixed",
        disabled=["id", "is_ai_tagged"],
        key=editor_key,
        column_config={
            "delete": st.column_config.CheckboxColumn("Delete", default=False),
            "id": st.column_config.NumberColumn("ID", format="%d"),
            "date": st.column_config.DateColumn("Date", format="YYYY-MM-DD", required=True),
            "type": st.column_config.SelectboxColumn(
                "Type",
                options=list(ALLOWED_TRANSACTION_TYPES),
                required=True,
            ),
            "category": st.column_config.TextColumn("Category", required=True),
            "amount": st.column_config.NumberColumn(
                f"Amount ({currency_symbol})",
                min_value=0.01,
                step=10.0,
                format=f"{currency_symbol}%,.2f",
                required=True,
            ),
            "description": st.column_config.TextColumn("Description"),
            "asset_ticker": st.column_config.TextColumn("Ticker"),
            "quantity": st.column_config.NumberColumn("Qty", min_value=0.0, format="%.4f"),
            "unit_price": st.column_config.NumberColumn(
                "Unit Price", min_value=0.0, format=f"{currency_symbol}%,.2f"
            ),
            "is_ai_tagged": st.column_config.CheckboxColumn("AI Tagged"),
        },
    )

    act_col1, act_col2, act_col3 = st.columns([1.2, 1.2, 1.2], gap="medium")

    with act_col1:
        if st.button("Save Table Edits", type="primary", width="stretch"):
            compare_cols = [
                "date",
                "type",
                "category",
                "amount",
                "description",
                "asset_ticker",
                "quantity",
                "unit_price",
            ]
            updated_count = 0
            errors: list[str] = []

            orig_indexed = editor_df.set_index("id")
            edit_indexed = edited_df.set_index("id")

            for tx_id in edit_indexed.index:
                if bool(edit_indexed.loc[tx_id, "delete"]):
                    continue
                orig_row = orig_indexed.loc[tx_id]
                new_row = edit_indexed.loc[tx_id]
                if not _rows_differ(orig_row, new_row, compare_cols):
                    continue
                try:
                    raw_amt = new_row["amount"]
                    if pd.isna(raw_amt):
                        raise ValueError("Amount cannot be empty.")
                    raw_type = new_row["type"]
                    raw_cat = new_row["category"]
                    raw_date = new_row["date"]
                    tx_date = orig_row["date"] if pd.isna(raw_date) else raw_date
                    ok = update_transaction(
                        transaction_id=int(tx_id),
                        date=tx_date,
                        trans_type="" if pd.isna(raw_type) else str(raw_type),
                        category="" if pd.isna(raw_cat) else str(raw_cat),
                        amount=float(raw_amt),
                        description=str(new_row["description"])
                        if pd.notna(new_row["description"])
                        else None,
                        asset_ticker=str(new_row["asset_ticker"])
                        if pd.notna(new_row["asset_ticker"])
                        else None,
                        quantity=float(new_row["quantity"])
                        if pd.notna(new_row["quantity"])
                        else 0.0,
                        unit_price=float(new_row["unit_price"])
                        if pd.notna(new_row["unit_price"])
                        else 0.0,
                        user_id=user_id,
                    )
                    if ok:
                        updated_count += 1
                except (ValueError, TypeError) as exc:
                    errors.append(f"Row ID {tx_id}: {exc}")

            if errors:
                for err in errors:
                    st.error(err)
            elif updated_count > 0:
                st.session_state["last_action_toast"] = f"Updated {updated_count} transaction(s)."
                st.rerun()
            else:
                st.info("No modified rows to save.")

    with act_col2:
        delete_mask = edited_df["delete"].fillna(False).astype(bool)
        selected_ids = edited_df.loc[delete_mask, "id"].astype(int).tolist()
        if st.button(
            f"Delete Selected ({len(selected_ids)})",
            disabled=len(selected_ids) == 0,
            width="stretch",
        ):
            deleted_count = delete_transactions(selected_ids, user_id=user_id)
            st.session_state["last_action_toast"] = f"Deleted {deleted_count} transaction(s)."
            st.rerun()

    with act_col3:
        export_csv = view_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Export Filtered CSV",
            data=export_csv,
            file_name=f"finance_ledger_{dt.date.today().isoformat()}.csv",
            mime="text/csv",
            width="stretch",
        )


def _render_copilot_tab(
    df: pd.DataFrame,
    active_api_key: Optional[str],
    currency_symbol: str,
) -> None:
    """Render the interactive Conversational AI Financial Copilot chat interface."""
    st.subheader("💬 AI Financial Copilot")
    st.caption("Ask natural-language questions about your spending patterns, portfolio performance, and budget optimizations.")

    summary_payload = build_ai_summary_payload(df)

    if "copilot_chat_history" not in st.session_state:
        st.session_state["copilot_chat_history"] = [
            {
                "role": "assistant",
                "content": "Hello! I am your AI Financial Copilot. Ask me anything about your cash flow, top expenses, or investment allocations in this timeframe.",
            }
        ]

    # Quick prompt suggestion pills
    s_col1, s_col2, s_col3, s_col4 = st.columns(4)
    quick_query: Optional[str] = None
    with s_col1:
        if st.button("Top spending category?", key="qp_top_cat", width="stretch"):
            quick_query = "What is my highest spending category and how much did I spend?"
    with s_col2:
        if st.button("Savings rate check", key="qp_savings", width="stretch"):
            quick_query = "What is my current savings rate and is it healthy?"
    with s_col3:
        if st.button("Portfolio summary", key="qp_portfolio", width="stretch"):
            quick_query = "Give me a summary of my active investment portfolio holdings."
    with s_col4:
        if st.button(f"Where can I cut {currency_symbol}5,000?", key="qp_cut", width="stretch"):
            quick_query = f"Where can I cut {currency_symbol}5,000 from discretionary spending?"

    # Display chat history
    for msg in st.session_state["copilot_chat_history"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_query = st.chat_input("Ask a question about your finances, budget, or investments...")
    effective_query = user_query or quick_query

    if effective_query:
        st.session_state["copilot_chat_history"].append({"role": "user", "content": effective_query})
        with st.chat_message("user"):
            st.markdown(effective_query)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing your ledger..."):
                reply = chat_financial_copilot(
                    effective_query,
                    history=st.session_state["copilot_chat_history"],
                    summary_payload=summary_payload,
                    api_key=active_api_key,
                    currency_symbol=currency_symbol,
                )
                st.markdown(reply)
                st.session_state["copilot_chat_history"].append({"role": "assistant", "content": reply})


def _format_user_contact(user: dict[str, Any]) -> str:
    contacts = [c for c in (user.get("email"), user.get("phone")) if c]
    return " · ".join(contacts) if contacts else "Authenticated Session"


def main() -> None:
    """Orchestrate auth portal, 3D sidebar controls, KPI metrics, AI insights, and tabs."""
    try:
        init_db()
    except Exception as exc:
        st.warning(f"Database initialization alert: {exc}. Using fallback storage.")

    toast_msg = st.session_state.pop("last_action_toast", None)
    if toast_msg:
        st.toast(toast_msg)

    current_user = st.session_state.get("auth_user")
    if not isinstance(current_user, dict) or not current_user.get("id"):
        _inject_custom_styles(authenticated=False)
        render_auth_portal()
        return

    _inject_custom_styles(authenticated=True)

    render_spatial_canvas(mode="dashboard")

    user_id = int(current_user["id"])
    metadata = get_filter_metadata(user_id=user_id)

    with st.sidebar:
        safe_name = html.escape(str(current_user.get("full_name", "User")))
        safe_contact = html.escape(_format_user_contact(current_user))
        db_info = get_db_target_info()
        db_badge_color = "#34d399" if db_info.get("is_cloud") else "#94a3b8"
        st.markdown(
            f"""
            <div class="user-badge-card">
                <div class="user-badge-name">👤 {safe_name}</div>
                <div class="user-badge-contact">{safe_contact}</div>
                <div style="font-size:0.75rem; color:{db_badge_color}; margin-top:6px; font-weight:600;">
                    {html.escape(db_info.get('display', ''))}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Sign Out", key="sidebar_logout_btn", width="stretch"):
            for k in ("auth_user", "demo_otp_preview", "ai_insights_markdown", "ai_insights_fingerprint"):
                st.session_state.pop(k, None)
            st.session_state["last_action_toast"] = "Signed out cleanly."
            st.rerun()

        st.divider()
        st.title("Control Panel")

        with st.expander("Settings & AI Engine", expanded=False):
            currency_symbol = st.selectbox(
                "Display Currency",
                options=["₹", "$", "€", "£"],
                index=0,
                key="currency_symbol_select",
            )
            st.markdown(f"**Database:** `{db_info.get('provider', 'SQLite')}`")
            if db_info.get("is_cloud"):
                st.caption(f"Host: `{db_info.get('target', '')}`")
            else:
                st.caption("Local SQLite mode. Set `DATABASE_URL` in `.env` to connect Supabase.")

            session_api_key = st.text_input(
                "Gemini API Key (session override)",
                type="password",
                placeholder="Reads GEMINI_API_KEY env if blank",
                key="session_gemini_api_key",
            )
            active_api_key = session_api_key.strip() or None
            if is_gemini_configured(active_api_key):
                st.success("Gemini AI active (Structured JSON Mode)")
            else:
                st.caption("No API key detected — running deterministic offline fallback.")
        currency_symbol = st.session_state.get("currency_symbol_select", "₹")
        active_api_key = (st.session_state.get("session_gemini_api_key", "") or "").strip() or None

        date_preset = st.selectbox(
            "Date Range",
            options=["Last 90 Days", "Current Month", "YTD", "All Time", "Custom Range"],
            index=0,
            key="sidebar_date_preset",
        )
        start_date, end_date = _resolve_date_range(date_preset, metadata)

        st.divider()
        render_manual_transaction_form(
            currency_symbol=currency_symbol,
            api_key=active_api_key,
            user_id=user_id,
        )

        st.divider()
        render_csv_uploader(api_key=active_api_key, user_id=user_id)

        st.divider()
        render_receipt_scanner(user_id=user_id, api_key=active_api_key, currency_symbol=currency_symbol)

        st.divider()
        with st.expander("Demo Data & Workspace", expanded=False):
            st.caption(f"Your ledger records: **{metadata['total_count']}**")
            if st.button("Load Sample Dataset", width="stretch"):
                inserted = seed_sample_data(user_id=user_id)
                st.session_state["last_action_toast"] = (
                    f"Loaded {inserted} sample multi-month transactions."
                )
                st.rerun()

    # Main 3D Animated Dashboard Header
    filtered_df = get_transactions(start_date=start_date, end_date=end_date, user_id=user_id)
    window_label = (
        f"{start_date.isoformat()} to {end_date.isoformat()}"
        if start_date and end_date
        else "All Dates"
    )
    ai_badge = (
        "Gemini AI Active"
        if is_gemini_configured(active_api_key)
        else "Heuristic Fallback Mode (Set GEMINI_API_KEY for live Gemini)"
    )
    st.markdown(
        f"""
        <div class="hero-banner-3d">
            <div class="cube-stage-sm" aria-hidden="true">
                <div class="cube-3d">
                    <div class="cube-face cube-front">₹</div>
                    <div class="cube-face cube-back">$</div>
                    <div class="cube-face cube-right">AI</div>
                    <div class="cube-face cube-left">₿</div>
                    <div class="cube-face cube-top">▲</div>
                    <div class="cube-face cube-bottom">◆</div>
                </div>
            </div>
            <div>
                <h1 class="hero-banner-title">AI-Powered Personal Finance &amp; Investment Tracker</h1>
                <div class="finance-subtitle">
                    Window: <b>{window_label}</b> · <b>{len(filtered_df)}</b> records in view · {ai_badge}
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if metadata["total_count"] == 0:
        empty_col1, empty_col2 = st.columns([3, 1], vertical_alignment="center")
        with empty_col1:
            st.info(
                "Your personal ledger is empty. Add a transaction from the sidebar, upload a CSV bank statement, or load sample multi-month data to explore all 3D charts and analytics."
            )
        with empty_col2:
            if st.button("Load Sample Dataset", type="primary", key="hero_seed_btn", width="stretch"):
                inserted = seed_sample_data(user_id=user_id)
                st.session_state["last_action_toast"] = (
                    f"Loaded {inserted} sample multi-month transactions."
                )
                st.rerun()

    # Row 1: Top 3D KPI Metric Cards
    kpis = compute_kpis(filtered_df)
    monthly_df = compute_monthly_cash_flow(filtered_df)
    render_kpi_cards(kpis, monthly_df=monthly_df, currency_symbol=currency_symbol)

    st.write("")

    # Row 2: On-Demand AI Financial Insights Expander
    with st.expander("AI Financial Insights & Anomaly Detection", expanded=False):
        ins_col1, ins_col2 = st.columns([1, 3], vertical_alignment="center")
        with ins_col1:
            analyze_clicked = st.button(
                "Analyze Portfolio",
                type="primary",
                width="stretch",
                key="analyze_portfolio_btn",
            )
        with ins_col2:
            st.caption(
                "Sends strictly aggregated, PII-free cash flow & allocation statistics to Gemini to flag spending spikes and recommend optimizations."
            )

        fingerprint = _dataframe_fingerprint(filtered_df)
        cached_fp = st.session_state.get("ai_insights_fingerprint")
        cached_md = st.session_state.get("ai_insights_markdown")

        if analyze_clicked:
            with st.spinner("Generating structured financial insights..."):
                insights_md = generate_financial_insights(
                    filtered_df,
                    api_key=active_api_key,
                    currency_symbol=currency_symbol,
                )
            st.session_state["ai_insights_fingerprint"] = fingerprint
            st.session_state["ai_insights_markdown"] = insights_md
            cached_fp = fingerprint
            cached_md = insights_md

        if cached_md and cached_fp == fingerprint:
            safe_md = re.sub(
                r"(`[^`]*`)|(?<!\\)\$",
                lambda m: m.group(1) if m.group(1) is not None else r"\$",
                cached_md,
            )
            st.markdown(safe_md, unsafe_allow_html=True)
        elif cached_md and cached_fp != fingerprint:
            st.warning(
                "Active data or filters changed since the last analysis. Click **Analyze Portfolio** to refresh insights."
            )

    # Row 3: 3D Financial Health Score & 50/30/20 Benchmark
    health = compute_financial_health_score(filtered_df)
    alloc_50_30_20 = compute_50_30_20(filtered_df)
    h_col1, h_col2 = st.columns([1, 1.45], gap="medium")
    with h_col1:
        st.plotly_chart(
            build_health_score_gauge(health["score"], label=health["label"], grade=health["grade"]),
            width="stretch",
        )
    with h_col2:
        st.plotly_chart(
            build_50_30_20_chart(alloc_50_30_20, currency_symbol=currency_symbol),
            width="stretch",
        )

    # Tabs: 5 Interactive Analytical Views
    tab_cash_flow, tab_spending, tab_portfolio, tab_ledger, tab_copilot = st.tabs(
        [
            "Cash Flow & Trend Analysis",
            "Spending Breakdown",
            "Portfolio & Asset Allocation",
            "Transaction Ledger",
            "💬 AI Financial Copilot",
        ]
    )

    with tab_cash_flow:
        _render_cash_flow_tab(monthly_df, currency_symbol)

    with tab_spending:
        _render_spending_tab(filtered_df, currency_symbol)

    with tab_portfolio:
        _render_portfolio_tab(filtered_df, currency_symbol)

    with tab_ledger:
        _render_ledger_tab(
            filtered_df,
            currency_symbol=currency_symbol,
            categories=metadata.get("categories", []),
            user_id=user_id,
        )

    with tab_copilot:
        _render_copilot_tab(filtered_df, active_api_key, currency_symbol)


if __name__ == "__main__":
    main()
