"""Sidebar manual transaction entry forms and intelligent CSV bank statement importer."""

from __future__ import annotations

import datetime as dt
import hashlib
import io
import re
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
import streamlit as st

from ai_engine import STANDARD_CATEGORIES, categorize_transaction, extract_receipt_data
from database import ALLOWED_TRANSACTION_TYPES, add_transaction, bulk_insert_transactions

_CURRENCY_CLEAN_RE = re.compile(r"[₹$€£,\s]")

_COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "date": ("date", "transaction date", "txn date", "value date", "posting date", "posted date"),
    "type": ("type", "trans_type", "transaction type", "txn type"),
    "category": ("category", "cat", "tag", "sector"),
    "amount": ("amount", "amt", "transaction amount", "txn amount", "value"),
    "debit": ("debit", "withdrawal", "withdrawals", "dr", "outflow", "debit amount"),
    "credit": ("credit", "deposit", "deposits", "cr", "inflow", "credit amount"),
    "description": (
        "description",
        "narration",
        "particulars",
        "memo",
        "merchant",
        "details",
        "remarks",
    ),
    "asset_ticker": ("asset_ticker", "ticker", "symbol", "asset", "scrip"),
    "quantity": ("quantity", "qty", "units", "shares"),
    "unit_price": ("unit_price", "price", "nav", "unit cost", "avg_price"),
}

_EXPENSE_CATEGORY_CHOICES: list[str] = [
    "Auto (AI Categorize)",
    "Food & Dining",
    "Transportation",
    "Utilities",
    "Housing & Rent",
    "Shopping & Lifestyle",
    "Healthcare",
    "Entertainment & Subscriptions",
    "Travel",
    "Education",
    "Other",
]

_INCOME_CATEGORY_CHOICES: list[str] = [
    "Salary",
    "Freelance & Business",
    "Auto (AI Categorize)",
    "Other",
]

_INVESTMENT_CATEGORY_CHOICES: list[str] = [
    "Investment",
    "Equity",
    "Index Fund / ETF",
    "Crypto",
    "Mutual Fund",
    "Fixed Income",
]


def _clean_numeric_series(series: pd.Series) -> pd.Series:
    """Parse currency-formatted numeric strings (including parenthesized negatives) into float."""
    text = series.fillna("").astype(str).str.strip()
    is_paren_neg = text.str.startswith("(") & text.str.endswith(")")
    cleaned = text.str.replace(r"[()]", "", regex=True)
    cleaned = cleaned.str.replace(_CURRENCY_CLEAN_RE, "", regex=True)
    numeric = pd.to_numeric(cleaned, errors="coerce")
    return pd.Series(
        np.where(is_paren_neg & numeric.notna(), -np.abs(numeric), numeric),
        index=series.index,
        dtype="float64",
    )


def _map_csv_columns(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize arbitrary bank CSV column headers into canonical internal names."""
    normalized_lookup = {
        # pyrefly: ignore[unnecessary-type-conversion]
        str(col).strip().lower().replace("_", " "): col for col in raw_df.columns
    }
    # pyrefly: ignore[unnecessary-type-conversion]
    raw_lower_lookup = {str(col).strip().lower(): col for col in raw_df.columns}

    rename_map: dict[Any, str] = {}
    assigned_targets: set[str] = set()
    for canonical, aliases in _COLUMN_ALIASES.items():
        if canonical in assigned_targets:
            continue
        for alias in aliases:
            if alias in raw_lower_lookup and raw_lower_lookup[alias] not in rename_map:
                rename_map[raw_lower_lookup[alias]] = canonical
                assigned_targets.add(canonical)
                break
            if alias in normalized_lookup and normalized_lookup[alias] not in rename_map:
                rename_map[normalized_lookup[alias]] = canonical
                assigned_targets.add(canonical)
                break

    renamed = raw_df.rename(columns=rename_map).copy()
    return renamed.loc[:, ~renamed.columns.duplicated()].copy()


def parse_and_enrich_csv(
    raw_bytes: bytes,
    *,
    categorizer: Callable[[str, float], str] = categorize_transaction,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Parse bank/transaction CSV bytes, AI-categorize missing categories, and normalize schema."""
    decoded: Optional[str] = None
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            decoded = raw_bytes.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    if decoded is None:
        raise ValueError("Unable to decode CSV file using UTF-8 or Latin-1.")

    try:
        raw_df = pd.read_csv(io.StringIO(decoded))
    except pd.errors.EmptyDataError:
        return pd.DataFrame(), {"parsed_rows": 0, "skipped_rows": 0, "ai_tagged_rows": 0}
    if raw_df.empty:
        return pd.DataFrame(), {"parsed_rows": 0, "skipped_rows": 0, "ai_tagged_rows": 0}

    df = _map_csv_columns(raw_df)

    if "date" not in df.columns:
        raise ValueError("CSV must contain a 'date' (or 'Transaction Date') column.")

    parsed_dates = pd.to_datetime(df["date"], format="mixed", errors="coerce")
    unparsed_mask = parsed_dates.isna() & df["date"].notna() & (df["date"].astype(str).str.strip() != "")
    if unparsed_mask.any():
        parsed_dates = parsed_dates.fillna(
            pd.to_datetime(df.loc[unparsed_mask, "date"], format="mixed", dayfirst=True, errors="coerce")
        )
    valid_date_mask = parsed_dates.notna()

    has_amount = "amount" in df.columns
    has_dr_cr = ("debit" in df.columns) or ("credit" in df.columns)
    if not has_amount and not has_dr_cr:
        raise ValueError(
            "CSV must include either an 'amount' column or 'debit'/'credit' columns."
        )

    explicit_type_col = "type" in df.columns
    if explicit_type_col:
        raw_types = (
            df["type"]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.title()
            .replace(
                {
                    "Dr": "Expense",
                    "Debit": "Expense",
                    "Withdrawal": "Expense",
                    "Cr": "Income",
                    "Credit": "Income",
                    "Deposit": "Income",
                }
            )
        )
    else:
        raw_types = pd.Series("", index=df.index, dtype="object")

    if has_amount:
        signed_amounts = _clean_numeric_series(df["amount"])
    else:
        debit_vals = (
            _clean_numeric_series(df["debit"]).fillna(0.0).abs()
            if "debit" in df.columns
            else pd.Series(0.0, index=df.index)
        )
        credit_vals = (
            _clean_numeric_series(df["credit"]).fillna(0.0).abs()
            if "credit" in df.columns
            else pd.Series(0.0, index=df.index)
        )
        signed_amounts = np.where(credit_vals > 0, credit_vals, -debit_vals)
        signed_amounts = pd.Series(signed_amounts, index=df.index, dtype="float64")

    abs_amounts = signed_amounts.abs()
    valid_amt_mask = abs_amounts.notna() & np.isfinite(abs_amounts) & (abs_amounts > 0)

    valid_mask = valid_date_mask & valid_amt_mask
    # pyrefly: ignore[unnecessary-type-conversion]
    skipped_rows = int((~valid_mask).sum())
    if not valid_mask.any():
        return pd.DataFrame(), {"parsed_rows": 0, "skipped_rows": skipped_rows, "ai_tagged_rows": 0}

    work = df.loc[valid_mask].copy()
    work["date"] = parsed_dates.loc[valid_mask].dt.strftime("%Y-%m-%d")
    work["amount"] = abs_amounts.loc[valid_mask].round(4)

    signed_valid = signed_amounts.loc[valid_mask]
    raw_types_valid = raw_types.loc[valid_mask]

    inferred_from_sign = np.where(signed_valid < 0, "Expense", "Income")
    type_is_valid = raw_types_valid.isin(ALLOWED_TRANSACTION_TYPES)
    work["type"] = np.where(type_is_valid, raw_types_valid, inferred_from_sign)
    work["_type_was_inferred"] = ~type_is_valid

    if "description" in work.columns:
        work["description"] = work["description"].fillna("").astype(str).str.strip()
    else:
        work["description"] = ""

    if "category" in work.columns:
        work["category"] = work["category"].fillna("").astype(str).str.strip()
    else:
        work["category"] = ""

    missing_cat_mask = work["category"].eq("") | work["category"].str.lower().isin(
        {"nan", "none", "null", "uncategorized", "na", "n/a"}
    )

    work["is_ai_tagged"] = 0
    ai_tagged_rows = 0

    if missing_cat_mask.any():
        cache: dict[tuple[str, float], str] = {}
        assigned_cats: dict[Any, str] = {}
        for idx, desc, amt in zip(
            work.index[missing_cat_mask],
            work.loc[missing_cat_mask, "description"],
            work.loc[missing_cat_mask, "amount"],
        ):
            cache_key = (str(desc).strip().lower(), round(float(amt), 2))
            if cache_key not in cache:
                cache[cache_key] = categorizer(str(desc), float(amt))
            assigned_cats[idx] = cache[cache_key]

        work.loc[missing_cat_mask, "category"] = pd.Series(assigned_cats)
        work.loc[missing_cat_mask, "is_ai_tagged"] = 1
        # pyrefly: ignore[unnecessary-type-conversion]
        ai_tagged_rows = int(missing_cat_mask.sum())

    # If transaction type was inferred (not explicitly provided) and AI tagged as Investment/Salary
    promote_investment = work["_type_was_inferred"] & (work["category"] == "Investment")
    work["type"] = np.where(promote_investment, "Investment", work["type"])
    promote_income = work["_type_was_inferred"] & work["category"].isin(
        ["Salary", "Freelance & Business"]
    )
    work["type"] = np.where(promote_income, "Income", work["type"])

    if "asset_ticker" in work.columns:
        # pyrefly: ignore[bad-argument-type]
        raw_tickers = work["asset_ticker"].where(work["asset_ticker"].notna(), None)
        work["asset_ticker"] = raw_tickers.map(
            lambda x: None
            if x is None or str(x).strip().upper() in {"", "NAN", "NONE", "NULL", "<NA>"}
            else str(x).strip().upper()
        )
    else:
        work["asset_ticker"] = None

    for num_col in ("quantity", "unit_price"):
        if num_col in work.columns:
            work[num_col] = _clean_numeric_series(work[num_col]).fillna(0.0).abs()
        else:
            work[num_col] = 0.0

    output_cols = [
        "date",
        "type",
        "category",
        "amount",
        "description",
        "asset_ticker",
        "quantity",
        "unit_price",
        "is_ai_tagged",
    ]
    cleaned_df = work[output_cols].reset_index(drop=True)
    return cleaned_df, {
        # pyrefly: ignore[unnecessary-type-conversion]
        "parsed_rows": int(len(cleaned_df)),
        "skipped_rows": skipped_rows,
        "ai_tagged_rows": ai_tagged_rows,
    }


def render_manual_transaction_form(
    *,
    currency_symbol: str = "₹",
    api_key: Optional[str] = None,
    user_id: Optional[int] = None,
) -> bool:
    """Render sidebar quick manual entry form supporting Income, Expense, and Investment."""
    st.subheader("Quick Add Transaction")

    tx_type = st.radio(
        "Transaction Type",
        options=list(ALLOWED_TRANSACTION_TYPES),
        horizontal=True,
        key="manual_tx_type_selector",
    )

    if tx_type == "Income":
        category_options = _INCOME_CATEGORY_CHOICES
    elif tx_type == "Investment":
        category_options = _INVESTMENT_CATEGORY_CHOICES
    else:
        category_options = _EXPENSE_CATEGORY_CHOICES

    form_rev = int(st.session_state.get("manual_form_rev", 0))
    submitted_success = False
    with st.form(f"manual_transaction_form_{form_rev}", clear_on_submit=False):
        tx_date = st.date_input("Date", value=dt.date.today())

        selected_cat = st.selectbox("Category", options=category_options, index=0)
        custom_cat = st.text_input(
            "Custom Category (optional)",
            placeholder="Custom name (optional)",
        )
        description = st.text_input(
            "Description / Merchant",
            placeholder="e.g., Swiggy, Zerodha SIP, Salary",
        )

        asset_ticker: Optional[str] = None
        quantity = 0.0
        unit_price = 0.0

        if tx_type == "Investment":
            col_t1, col_t2 = st.columns(2)
            with col_t1:
                asset_ticker = st.text_input("Asset Ticker", placeholder="NIFTY50, TCS, BTC")
            with col_t2:
                quantity = st.number_input("Units / Qty", min_value=0.0, step=0.01, format="%.4f")

            col_p1, col_p2 = st.columns(2)
            with col_p1:
                unit_price = st.number_input(
                    f"Unit Price ({currency_symbol})",
                    min_value=0.0,
                    step=10.0,
                    format="%.2f",
                )
            with col_p2:
                amount = st.number_input(
                    f"Total Amount ({currency_symbol})",
                    min_value=0.0,
                    step=100.0,
                    format="%.2f",
                    help="Leave 0 to auto-compute from Units × Unit Price.",
                )
        else:
            amount = st.number_input(
                f"Amount ({currency_symbol})",
                min_value=0.0,
                step=50.0,
                format="%.2f",
            )

        submitted = st.form_submit_button("Add Transaction", type="primary", width="stretch")

        if submitted:
            effective_amount = float(amount)
            if tx_type == "Investment" and effective_amount <= 0 and quantity > 0 and unit_price > 0:
                effective_amount = round(float(quantity) * float(unit_price), 2)

            if effective_amount <= 0:
                st.error("Amount must be greater than 0.")
                return False

            if tx_type == "Investment" and not (asset_ticker and asset_ticker.strip()):
                st.error("Please enter an Asset Ticker for investment entries (e.g., NIFTY50).")
                return False

            final_category = custom_cat.strip() if custom_cat.strip() else selected_cat
            is_ai = 0
            if final_category == "Auto (AI Categorize)":
                if not description.strip():
                    final_category = "Other"
                else:
                    with st.spinner("Categorizing with AI..."):
                        final_category = categorize_transaction(
                            description.strip(),
                            effective_amount,
                            api_key=api_key,
                        )
                    is_ai = 1

            try:
                add_transaction(
                    date=tx_date.isoformat(),
                    trans_type=tx_type,
                    category=final_category,
                    amount=effective_amount,
                    description=description.strip() or None,
                    asset_ticker=asset_ticker.strip().upper() if asset_ticker else None,
                    quantity=float(quantity),
                    unit_price=float(unit_price),
                    is_ai_tagged=is_ai,
                    user_id=user_id,
                )
                st.session_state["last_action_toast"] = (
                    f"Added {tx_type}: {final_category} ({currency_symbol}{effective_amount:,.2f})"
                )
                st.session_state["manual_form_rev"] = form_rev + 1
                submitted_success = True
            except ValueError as exc:
                st.error(f"Validation error: {exc}")

    if submitted_success:
        st.rerun()
    return submitted_success


def render_csv_uploader(
    *,
    api_key: Optional[str] = None,
    user_id: Optional[int] = None,
) -> bool:
    """Render sidebar CSV bank statement uploader with AI auto-categorization."""
    st.subheader("Import Bank Statement (CSV)")

    sample_template = (
        "date,description,amount,type,category,asset_ticker,quantity,unit_price\n"
        "2026-10-01,Monthly Salary Credit,125000,Income,Salary,,,\n"
        "2026-10-03,Swiggy Bangalore,-640,Expense,,,,\n"
        "2026-10-05,Uber Airport Trip,-820,Expense,,,,\n"
        "2026-10-10,Zerodha SIP NIFTY50,20000,Investment,Investment,NIFTY50,80,250\n"
    )
    st.download_button(
        label="Download Sample CSV Template",
        data=sample_template,
        file_name="sample_transactions_template.csv",
        mime="text/csv",
        width="stretch",
    )

    uploaded_file = st.file_uploader(
        "Drag & drop CSV bank export",
        type=["csv"],
        key="csv_statement_uploader",
        help="Supports standard schema CSVs or raw bank statements (Date, Description, Amount / Debit-Credit). Missing categories are tagged via Gemini AI.",
    )

    if uploaded_file is None:
        return False

    raw_bytes = uploaded_file.getvalue()
    file_hash = hashlib.sha256(raw_bytes).hexdigest()
    hashes_key = f"imported_csv_hashes_{user_id if user_id is not None else 'global'}"
    imported_hashes: set[str] = st.session_state.setdefault(hashes_key, set())

    if file_hash in imported_hashes:
        st.info(f"`{uploaded_file.name}` has already been imported in this session.")
        return False

    if st.button("Parse, AI-Categorize & Import CSV", type="primary", width="stretch"):
        try:
            with st.spinner("Parsing statement & invoking AI categorization..."):
                parsed_df, stats = parse_and_enrich_csv(
                    raw_bytes,
                    categorizer=lambda desc, amt: categorize_transaction(
                        desc, amt, api_key=api_key
                    ),
                )
            if parsed_df.empty:
                st.warning("No valid transaction rows were detected in the uploaded CSV.")
                return False

            inserted = bulk_insert_transactions(parsed_df, user_id=user_id)
            imported_hashes.add(file_hash)
            st.session_state["last_action_toast"] = (
                f"Imported {inserted} transactions "
                f"({stats['ai_tagged_rows']} AI-categorized, {stats['skipped_rows']} skipped)."
            )
            st.rerun()
        except Exception as exc:
            st.error(f"CSV import failed: {exc}")
            return False

    return False


def render_receipt_scanner(
    *,
    user_id: int,
    api_key: Optional[str] = None,
    currency_symbol: str = "₹",
) -> None:
    """Render the multimodal Gemini Vision receipt and invoice scanner."""
    is_receipt_active = bool(st.session_state.get("receipt_image_uploader"))
    with st.expander("📷 Scan Receipt / Invoice (Gemini Vision)", expanded=is_receipt_active):
        st.caption("Upload a receipt photo or invoice image to extract transaction details automatically via Gemini Vision.")
        uploaded_image = st.file_uploader(
            "Upload Receipt Image",
            type=["png", "jpg", "jpeg", "webp"],
            key="receipt_image_uploader",
        )

        if not uploaded_image:
            return

        img_bytes = uploaded_image.getvalue()
        mime_type = uploaded_image.type or "image/jpeg"
        st.image(img_bytes, caption="Receipt Preview", width=220)

        img_hash = hashlib.sha256(img_bytes).hexdigest()
        scanned_data = st.session_state.get(f"scanned_receipt_{img_hash}")

        if st.button("Scan Receipt with Gemini Vision", type="primary", key="scan_receipt_btn", width="stretch"):
            with st.spinner("Extracting merchant, items, date, and amount via Gemini Vision..."):
                extracted = extract_receipt_data(
                    img_bytes,
                    mime_type=mime_type,
                    api_key=api_key,
                )
                scanned_data = extracted.model_dump()
                st.session_state[f"scanned_receipt_{img_hash}"] = scanned_data
                st.success(f"Extracted: {scanned_data.get('merchant_name')} ({currency_symbol}{scanned_data.get('total_amount', 0):,.2f})")

        if isinstance(scanned_data, dict):
            with st.form("confirm_scanned_receipt_form"):
                default_date_str = scanned_data.get("date")
                try:
                    default_date = dt.date.fromisoformat(default_date_str) if default_date_str else dt.date.today()
                except ValueError:
                    default_date = dt.date.today()

                col_d, col_amt = st.columns(2)
                with col_d:
                    r_date = st.date_input("Date", value=default_date, key="scanned_date_input")
                with col_amt:
                    raw_scanned_amt = scanned_data.get("total_amount")
                    try:
                        safe_init_amt = max(0.01, float(raw_scanned_amt)) if raw_scanned_amt else 10.0
                    except (TypeError, ValueError):
                        safe_init_amt = 10.0
                    r_amount = st.number_input(
                        f"Total Amount ({currency_symbol})",
                        min_value=0.01,
                        value=safe_init_amt,
                        step=10.0,
                        key="scanned_amt_input",
                    )

                extracted_cat = str(scanned_data.get("category", "Food & Dining"))
                cat_options = list(STANDARD_CATEGORIES)
                cat_idx = cat_options.index(extracted_cat) if extracted_cat in cat_options else 0
                r_cat = st.selectbox("Category", options=cat_options, index=cat_idx, key="scanned_cat_select")

                r_desc = st.text_input(
                    "Merchant / Description",
                    value=scanned_data.get("merchant_name", "Merchant"),
                    key="scanned_merchant_input",
                )
                r_notes = st.text_area(
                    "Notes / Line Items",
                    value=scanned_data.get("notes", ""),
                    key="scanned_notes_input",
                )

                submit_scanned = st.form_submit_button("Add Scanned Receipt to Ledger", type="primary", width="stretch")
                if submit_scanned:
                    full_desc = f"{r_desc} ({r_notes})" if r_notes.strip() else r_desc
                    new_id = add_transaction(
                        date=r_date,
                        trans_type="Expense",
                        category=r_cat,
                        amount=float(r_amount),
                        description=full_desc,
                        is_ai_tagged=True,
                        user_id=user_id,
                    )
                    st.session_state["last_action_toast"] = (
                        f"Added receipt expense #{new_id} ({currency_symbol}{r_amount:,.2f}) to ledger!"
                    )
                    st.rerun()
