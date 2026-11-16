"""Vectorized financial KPI, cash-flow, category, and portfolio analytics engine."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

_CASH_FLOW_COLUMNS: list[str] = [
    "month",
    "Income",
    "Expense",
    "Investment",
    "Net Cash Flow",
    "Investable Surplus",
]

_BREAKDOWN_COLUMNS: list[str] = [
    "category",
    "amount",
    "percentage",
    "transactions",
]

_PORTFOLIO_COLUMNS: list[str] = [
    "asset_ticker",
    "total_invested",
    "total_units",
    "avg_cost_basis",
    "current_price",
    "current_value",
    "unrealized_pnl",
    "return_pct",
    "allocation_pct",
    "transactions",
]

_SPIKE_COLUMNS: list[str] = [
    "category",
    "month",
    "amount",
    "baseline_mean",
    "pct_above_baseline",
    "z_score",
]

_BUDGET_ALERT_COLUMNS: list[str] = [
    "category",
    "current_month",
    "current_spend",
    "monthly_average",
    "pct_of_average",
    "status",
    "severity",
    "message",
]

_LIVE_PRICE_CACHE: dict[str, tuple[float, float]] = {}
_TICKER_MAP: dict[str, str] = {
    "NIFTY50": "^NSEI",
    "NIFTY": "^NSEI",
    "SENSEX": "^BSESN",
    "TCS": "TCS.NS",
    "INFY": "INFY.NS",
    "RELIANCE": "RELIANCE.NS",
    "HDFCBANK": "HDFCBANK.NS",
    "ICICIBANK": "ICICIBANK.NS",
    "BTC": "BTC-USD",
    "ETH": "ETH-USD",
    "SOL": "SOL-USD",
}


def fetch_live_price(ticker_str: str) -> Optional[float]:
    """Fetch current market price using yfinance with fast in-memory caching."""
    import time
    clean = (ticker_str or "").strip().upper()
    if not clean or clean in {"UNSPECIFIED", "NONE", "NAN", "NULL"}:
        return None

    now = time.time()
    if clean in _LIVE_PRICE_CACHE:
        price, ts = _LIVE_PRICE_CACHE[clean]
        if now - ts < 600:
            return price

    try:
        import yfinance as yf
        sym = _TICKER_MAP.get(clean, clean)
        t = yf.Ticker(sym)
        price = getattr(t.fast_info, "last_price", None)
        if price is None and not sym.endswith(".NS") and "-" not in sym and "^" not in sym:
            price = getattr(yf.Ticker(f"{sym}.NS").fast_info, "last_price", None)
        if price is not None and np.isfinite(price) and price > 0:
            val = float(round(price, 2))
            _LIVE_PRICE_CACHE[clean] = (val, now)
            return val
    except Exception:
        pass
    return None


def _prepare_numeric_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure canonical numeric and string columns exist safely without mutating caller."""
    if df is None or df.empty:
        return pd.DataFrame(
            columns=[
                "id",
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
        )

    work = df.copy()
    for col, default_val in (
        ("amount", 0.0),
        ("quantity", 0.0),
        ("unit_price", 0.0),
        ("is_ai_tagged", 0),
    ):
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce").fillna(default_val).astype("float64")
        else:
            work[col] = float(default_val)

    for col in ("type", "category", "description", "asset_ticker"):
        if col not in work.columns:
            work[col] = None

    work["type"] = work["type"].fillna("").astype(str).str.strip()
    work["category"] = work["category"].fillna("Uncategorized").astype(str).str.strip()
    return work


def compute_kpis(df: pd.DataFrame) -> dict[str, float | int]:
    """Compute vectorized inflow, outflow, investment, net savings, and savings rate KPIs."""
    work = _prepare_numeric_frame(df)
    if work.empty:
        return {
            "total_inflow": 0.0,
            "total_outflow": 0.0,
            "capital_deployed": 0.0,
            "net_savings": 0.0,
            "savings_rate": 0.0,
            "free_cash_flow": 0.0,
            "transaction_count": 0,
        }

    amounts = work["amount"].to_numpy(dtype=np.float64)
    types = work["type"].to_numpy()

    total_inflow = float(np.sum(amounts[types == "Income"]))
    total_outflow = float(np.sum(amounts[types == "Expense"]))
    capital_deployed = float(np.sum(amounts[types == "Investment"]))

    net_savings = total_inflow - total_outflow
    savings_rate = float((net_savings / total_inflow) * 100.0) if total_inflow > 0.0 else 0.0
    free_cash_flow = total_inflow - total_outflow - capital_deployed

    return {
        "total_inflow": round(total_inflow, 2),
        "total_outflow": round(total_outflow, 2),
        "capital_deployed": round(capital_deployed, 2),
        "net_savings": round(net_savings, 2),
        "savings_rate": round(savings_rate, 2),
        "free_cash_flow": round(free_cash_flow, 2),
        "transaction_count": len(work),
    }


def compute_monthly_cash_flow(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate monthly inflows, outflows, investments, and net savings delta."""
    work = _prepare_numeric_frame(df)
    if work.empty or "date" not in work.columns:
        return pd.DataFrame(columns=_CASH_FLOW_COLUMNS)

    parsed_dates = pd.to_datetime(work["date"], format="mixed", errors="coerce")
    valid_mask = parsed_dates.notna() & work["type"].isin(["Income", "Expense", "Investment"])
    if not valid_mask.any():
        return pd.DataFrame(columns=_CASH_FLOW_COLUMNS)

    valid = work.loc[valid_mask, ["type", "amount"]].copy()
    valid["month"] = parsed_dates.loc[valid_mask].dt.to_period("M").astype(str)

    pivoted = (
        valid.groupby(["month", "type"], sort=True)["amount"]
        .sum()
        # pyrefly: ignore[bad-argument-type]
        .unstack(fill_value=0.0)
        .reindex(columns=["Income", "Expense", "Investment"], fill_value=0.0)
    )
    pivoted.columns.name = None
    pivoted = pivoted.reset_index().sort_values(by="month", ascending=True).reset_index(drop=True)

    pivoted["Income"] = pivoted["Income"].astype("float64").round(2)
    pivoted["Expense"] = pivoted["Expense"].astype("float64").round(2)
    pivoted["Investment"] = pivoted["Investment"].astype("float64").round(2)
    pivoted["Net Cash Flow"] = (pivoted["Income"] - pivoted["Expense"]).round(2)
    pivoted["Investable Surplus"] = (
        pivoted["Income"] - pivoted["Expense"] - pivoted["Investment"]
    ).round(2)

    return pivoted[_CASH_FLOW_COLUMNS]


def compute_category_breakdown(
    df: pd.DataFrame,
    trans_type: str = "Expense",
) -> pd.DataFrame:
    """Compute vectorized category spend totals and percentage share for a transaction type."""
    work = _prepare_numeric_frame(df)
    if work.empty:
        return pd.DataFrame(columns=_BREAKDOWN_COLUMNS)

    filtered = work.loc[work["type"] == trans_type]
    if filtered.empty:
        return pd.DataFrame(columns=_BREAKDOWN_COLUMNS)

    grouped = (
        filtered.groupby("category", as_index=False)
        .agg(
            amount=("amount", "sum"),
            transactions=("amount", "size"),
        )
        .sort_values(by=["amount", "category"], ascending=[False, True])
        .reset_index(drop=True)
    )

    total_amount = float(grouped["amount"].sum())
    grouped["amount"] = grouped["amount"].round(2)
    grouped["percentage"] = np.where(
        total_amount > 0.0,
        np.round((grouped["amount"] / total_amount) * 100.0, 2),
        0.0,
    )
    grouped["transactions"] = grouped["transactions"].astype("int64")
    return grouped[_BREAKDOWN_COLUMNS]


def compute_top_expenses(df: pd.DataFrame, top_n: int = 25) -> pd.DataFrame:
    """Return sorted individual expense records with share of total expense spend."""
    cols = ["date", "category", "description", "amount", "share_pct", "is_ai_tagged"]
    work = _prepare_numeric_frame(df)
    if work.empty:
        return pd.DataFrame(columns=cols)

    expenses = work.loc[work["type"] == "Expense"].copy()
    if expenses.empty:
        return pd.DataFrame(columns=cols)

    total_expense = float(expenses["amount"].sum())
    expenses["description"] = expenses["description"].fillna("").astype(str)
    expenses["share_pct"] = np.where(
        total_expense > 0.0,
        np.round((expenses["amount"] / total_expense) * 100.0, 2),
        0.0,
    )
    expenses["is_ai_tagged"] = expenses["is_ai_tagged"].astype("int64")
    expenses["amount"] = expenses["amount"].round(2)

    sorted_expenses = (
        expenses.sort_values(by=["amount", "date"], ascending=[False, False])
        .head(max(1, top_n))
        .reset_index(drop=True)
    )
    return sorted_expenses[cols]


def compute_portfolio_allocation(
    df: pd.DataFrame,
    fetch_live_prices: bool = True,
) -> pd.DataFrame:
    """Aggregate investment holdings by asset_ticker with weighted average cost basis and live P&L."""
    work = _prepare_numeric_frame(df)
    if work.empty:
        return pd.DataFrame(columns=_PORTFOLIO_COLUMNS)

    investments = work.loc[work["type"] == "Investment"].copy()
    if investments.empty:
        return pd.DataFrame(columns=_PORTFOLIO_COLUMNS)

    ticker_clean = (
        investments["asset_ticker"]
        .fillna("UNSPECIFIED")
        .astype(str)
        .str.strip()
        .str.upper()
    )
    investments["asset_ticker"] = ticker_clean.replace(
        {"": "UNSPECIFIED", "NONE": "UNSPECIFIED", "NAN": "UNSPECIFIED", "NULL": "UNSPECIFIED"}
    )

    portfolio = (
        investments.groupby("asset_ticker", as_index=False)
        .agg(
            total_invested=("amount", "sum"),
            total_units=("quantity", "sum"),
            transactions=("amount", "size"),
        )
        .sort_values(by=["total_invested", "asset_ticker"], ascending=[False, True])
        .reset_index(drop=True)
    )

    total_capital = float(portfolio["total_invested"].sum())
    units_arr = portfolio["total_units"].to_numpy(dtype=np.float64)
    invested_arr = portfolio["total_invested"].to_numpy(dtype=np.float64)

    portfolio["avg_cost_basis"] = np.where(
        units_arr > 0.0,
        np.round(invested_arr / np.where(units_arr > 0.0, units_arr, 1.0), 4),
        0.0,
    )
    portfolio["allocation_pct"] = np.where(
        total_capital > 0.0,
        np.round((invested_arr / total_capital) * 100.0, 2),
        0.0,
    )
    portfolio["total_invested"] = portfolio["total_invested"].round(2)
    portfolio["total_units"] = portfolio["total_units"].round(6)
    portfolio["transactions"] = portfolio["transactions"].astype("int64")

    # Vectorized live price computation
    cur_prices: list[float] = []
    for ticker, avg_cost in zip(portfolio["asset_ticker"], portfolio["avg_cost_basis"]):
        lp = fetch_live_price(str(ticker)) if fetch_live_prices else None
        cur_prices.append(float(lp) if lp is not None else float(avg_cost))

    portfolio["current_price"] = np.round(cur_prices, 2)
    portfolio["current_value"] = np.round(portfolio["total_units"] * portfolio["current_price"], 2)
    portfolio["unrealized_pnl"] = np.round(portfolio["current_value"] - portfolio["total_invested"], 2)

    inv_arr = portfolio["total_invested"].to_numpy(dtype=np.float64)
    pnl_arr = portfolio["unrealized_pnl"].to_numpy(dtype=np.float64)
    portfolio["return_pct"] = np.where(
        inv_arr > 0.0,
        np.round((pnl_arr / inv_arr) * 100.0, 2),
        0.0,
    )

    return portfolio[_PORTFOLIO_COLUMNS]


def compute_budget_alerts(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze monthly expense trends per category and emit warnings for spending surges."""
    work = _prepare_numeric_frame(df)
    if work.empty or "date" not in work.columns:
        return pd.DataFrame(columns=_BUDGET_ALERT_COLUMNS)

    expenses = work.loc[work["type"] == "Expense"].copy()
    if expenses.empty:
        return pd.DataFrame(columns=_BUDGET_ALERT_COLUMNS)

    expenses["month"] = expenses["date"].astype(str).str.slice(0, 7)
    months = sorted(expenses["month"].unique())
    if not months:
        return pd.DataFrame(columns=_BUDGET_ALERT_COLUMNS)

    latest_month = months[-1]

    # Category monthly aggregate
    cat_month = (
        expenses.groupby(["category", "month"], as_index=False)["amount"]
        .sum()
    )

    alerts = []
    for cat, grp in cat_month.groupby("category"):
        latest_row = grp[grp["month"] == latest_month]
        if latest_row.empty:
            continue
        current_spend = float(latest_row["amount"].iloc[0])
        hist_rows = grp[grp["month"] != latest_month]
        if hist_rows.empty:
            avg_spend = current_spend
        else:
            avg_spend = float(hist_rows["amount"].mean())

        if avg_spend <= 0:
            pct = 100.0
        else:
            pct = round((current_spend / avg_spend) * 100.0, 1)

        if pct >= 125.0:
            status = "Spike Over Budget"
            severity = "critical"
            msg = f"{cat} spending surged to {pct}% of baseline ({current_spend:,.0f} vs avg {avg_spend:,.0f})."
        elif pct >= 85.0:
            status = "Near Budget Limit"
            severity = "warning"
            msg = f"{cat} is nearing budget at {pct}% of baseline ({current_spend:,.0f} vs avg {avg_spend:,.0f})."
        else:
            status = "Healthy"
            severity = "normal"
            msg = f"{cat} is within safe baseline at {pct}% ({current_spend:,.0f} vs avg {avg_spend:,.0f})."

        alerts.append({
            "category": str(cat),
            "current_month": str(latest_month),
            "current_spend": round(current_spend, 2),
            "monthly_average": round(avg_spend, 2),
            "pct_of_average": pct,
            "status": status,
            "severity": severity,
            "message": msg,
        })

    if not alerts:
        return pd.DataFrame(columns=_BUDGET_ALERT_COLUMNS)

    alert_df = pd.DataFrame(alerts).sort_values(by=["pct_of_average", "current_spend"], ascending=[False, False])
    return alert_df.reset_index(drop=True)[_BUDGET_ALERT_COLUMNS]


def detect_spending_spikes(
    df: pd.DataFrame,
    z_threshold: float = 1.15,
    pct_jump_threshold: float = 35.0,
) -> pd.DataFrame:
    """Detect statistically unusual monthly category expense surges using vectorized NumPy/Pandas."""
    work = _prepare_numeric_frame(df)
    if work.empty or "date" not in work.columns:
        return pd.DataFrame(columns=_SPIKE_COLUMNS)

    expenses = work.loc[work["type"] == "Expense"].copy()
    if expenses.empty:
        return pd.DataFrame(columns=_SPIKE_COLUMNS)

    parsed_dates = pd.to_datetime(expenses["date"], format="mixed", errors="coerce")
    valid_mask = parsed_dates.notna()
    if not valid_mask.any():
        return pd.DataFrame(columns=_SPIKE_COLUMNS)

    expenses = expenses.loc[valid_mask].copy()
    expenses["month"] = parsed_dates.loc[valid_mask].dt.to_period("M").astype(str)

    cat_month = (
        expenses.groupby(["category", "month"], as_index=False)[["amount"]]
        .sum()
        .sort_values(by=["category", "month"])
    )

    month_counts = cat_month.groupby("category")["month"].transform("count")
    cat_month = cat_month.loc[month_counts >= 2].copy()
    if cat_month.empty:
        return pd.DataFrame(columns=_SPIKE_COLUMNS)

    # Leave-one-out / historical baseline mean & std via vectorized group sums
    grp_sum = cat_month.groupby("category")["amount"].transform("sum")
    grp_cnt = cat_month.groupby("category")["amount"].transform("count")
    baseline_mean = (grp_sum - cat_month["amount"]) / (grp_cnt - 1)

    grp_std = cat_month.groupby("category")["amount"].transform("std").fillna(0.0)
    z_score = np.where(
        grp_std > 0.0,
        (cat_month["amount"] - cat_month.groupby("category")["amount"].transform("mean")) / grp_std,
        0.0,
    )
    pct_above = np.where(
        baseline_mean > 0.0,
        ((cat_month["amount"] - baseline_mean) / baseline_mean) * 100.0,
        0.0,
    )

    cat_month["baseline_mean"] = np.round(baseline_mean, 2)
    cat_month["pct_above_baseline"] = np.round(pct_above, 2)
    cat_month["z_score"] = np.round(z_score, 2)
    cat_month["amount"] = np.round(cat_month["amount"], 2)

    flagged = cat_month.loc[
        (cat_month["pct_above_baseline"] >= pct_jump_threshold)
        | (cat_month["z_score"] >= z_threshold)
    ].sort_values(by=["pct_above_baseline", "amount"], ascending=[False, False])

    if flagged.empty:
        return pd.DataFrame(columns=_SPIKE_COLUMNS)

    return flagged[_SPIKE_COLUMNS].reset_index(drop=True)


def build_ai_summary_payload(df: pd.DataFrame) -> dict[str, Any]:
    """Construct a PII-free aggregate summary dictionary suitable for LLM financial analysis."""
    work = _prepare_numeric_frame(df)
    kpis = compute_kpis(work)
    monthly = compute_monthly_cash_flow(work)
    expense_breakdown = compute_category_breakdown(work, trans_type="Expense")
    portfolio = compute_portfolio_allocation(work)
    spikes = detect_spending_spikes(work)

    if not work.empty and "date" in work.columns:
        parsed_dates = pd.to_datetime(work["date"], format="mixed", errors="coerce").dropna()
        start_str = parsed_dates.min().strftime("%Y-%m-%d") if not parsed_dates.empty else None
        end_str = parsed_dates.max().strftime("%Y-%m-%d") if not parsed_dates.empty else None
    else:
        start_str = None
        end_str = None

    return {
        "period": {
            "start_date": start_str,
            "end_date": end_str,
            "months_covered": len(monthly),
            "transaction_count": int(kpis["transaction_count"]),
        },
        "kpis": kpis,
        "monthly_cash_flow": monthly.to_dict(orient="records"),
        "expense_categories": expense_breakdown.head(10).to_dict(orient="records"),
        "portfolio_allocation": portfolio[
            ["asset_ticker", "total_invested", "allocation_pct", "avg_cost_basis"]
        ].to_dict(orient="records"),
        "statistical_spending_spikes": spikes.head(5).to_dict(orient="records"),
    }


def compute_financial_health_score(df: pd.DataFrame) -> dict[str, Any]:
    """Calculate a holistic 0-100 financial health score with grade and breakdown."""
    work = _prepare_numeric_frame(df)
    if work.empty:
        return {
            "score": 50,
            "grade": "C",
            "label": "Neutral / Awaiting Data",
            "breakdown": {
                "savings_pts": 15.0,
                "investment_pts": 10.0,
                "stability_pts": 15.0,
                "discipline_pts": 10.0,
            },
            "metrics": {
                "savings_rate_pct": 0.0,
                "investment_ratio_pct": 0.0,
                "positive_months_pct": 0.0,
            },
        }

    kpis = compute_kpis(work)
    income = float(kpis.get("total_inflow", kpis.get("total_income", 0.0)))
    expense = float(kpis.get("total_outflow", kpis.get("total_expenses", 0.0)))
    investment = float(kpis.get("capital_deployed", kpis.get("total_investments", 0.0)))

    # 1. Savings Rate Points (Max 35 pts)
    savings_rate = ((income - expense) / income) if income > 0 else 0.0
    if savings_rate >= 0.20:
        savings_pts = 35.0
    elif savings_rate > 0:
        savings_pts = (savings_rate / 0.20) * 35.0
    else:
        savings_pts = 0.0

    # 2. Investment Ratio Points (Max 25 pts)
    invest_ratio = (investment / income) if income > 0 else 0.0
    if invest_ratio >= 0.15:
        invest_pts = 25.0
    elif invest_ratio > 0:
        invest_pts = (invest_ratio / 0.15) * 25.0
    else:
        invest_pts = 0.0

    # 3. Cash Flow Stability Points (Max 25 pts)
    monthly = compute_monthly_cash_flow(work)
    if not monthly.empty and "Net Cash Flow" in monthly.columns:
        net_flows = pd.to_numeric(monthly["Net Cash Flow"], errors="coerce").fillna(0.0)
        pos_count = (net_flows > 0).sum()
        pos_ratio = float(pos_count) / len(net_flows)
        stability_pts = 25.0 * pos_ratio
    else:
        pos_ratio = 0.5
        stability_pts = 12.5

    # 4. Expense Discipline Points (Max 15 pts)
    spikes = detect_spending_spikes(work)
    spike_count = len(spikes)
    if spike_count == 0:
        discipline_pts = 15.0
    elif spike_count <= 2:
        discipline_pts = 10.0
    else:
        discipline_pts = 5.0

    total_score = round(min(100.0, max(0.0, savings_pts + invest_pts + stability_pts + discipline_pts)))

    if total_score >= 80:
        grade = "A"
        label = "Thriving"
    elif total_score >= 65:
        grade = "B"
        label = "Stable"
    elif total_score >= 50:
        grade = "C"
        label = "Caution"
    else:
        grade = "D"
        label = "Vulnerable"

    return {
        "score": total_score,
        "grade": grade,
        "label": label,
        "breakdown": {
            "savings_pts": round(savings_pts, 1),
            "investment_pts": round(invest_pts, 1),
            "stability_pts": round(stability_pts, 1),
            "discipline_pts": round(discipline_pts, 1),
        },
        "metrics": {
            "savings_rate_pct": round(savings_rate * 100.0, 1),
            "investment_ratio_pct": round(invest_ratio * 100.0, 1),
            "positive_months_pct": round(pos_ratio * 100.0, 1),
        },
    }


def compute_50_30_20(df: pd.DataFrame) -> dict[str, Any]:
    """Classify cash outflows into Needs (50%), Wants (30%), and Savings/Investments (20%)."""
    work = _prepare_numeric_frame(df)
    if work.empty:
        return {
            "needs_amount": 0.0,
            "wants_amount": 0.0,
            "savings_amount": 0.0,
            "needs_pct": 0.0,
            "wants_pct": 0.0,
            "savings_pct": 0.0,
            "total_base": 0.0,
        }

    kpis = compute_kpis(work)
    total_income = float(kpis.get("total_inflow", kpis.get("total_income", 0.0)))

    needs_cats = {
        "Housing & Rent",
        "Utilities",
        "Food & Dining",
        "Transportation",
        "Healthcare",
        "Education",
    }

    expenses = work.loc[work["type"] == "Expense"]
    needs_amt = 0.0
    wants_amt = 0.0

    if not expenses.empty:
        for _, row in expenses.iterrows():
            cat = str(row.get("category", "")).strip()
            amt = float(row.get("amount", 0.0))
            if cat in needs_cats:
                needs_amt += amt
            else:
                wants_amt += amt

    savings_amt = float(kpis.get("capital_deployed", kpis.get("total_investments", 0.0))) + max(
        0.0, float(kpis.get("net_savings", 0.0))
    )
    total_base = max(total_income, (needs_amt + wants_amt + savings_amt))
    if total_base <= 0:
        total_base = 1.0

    return {
        "needs_amount": round(needs_amt, 2),
        "wants_amount": round(wants_amt, 2),
        "savings_amount": round(savings_amt, 2),
        "needs_pct": round((needs_amt / total_base) * 100.0, 1),
        "wants_pct": round((wants_amt / total_base) * 100.0, 1),
        "savings_pct": round((savings_amt / total_base) * 100.0, 1),
        "total_base": round(total_base, 2),
    }


def detect_recurring_subscriptions(df: pd.DataFrame) -> pd.DataFrame:
    """Identify recurring subscriptions and periodic charges from expense transactions."""
    work = _prepare_numeric_frame(df)
    cols = ["service_name", "category", "frequency", "avg_amount", "annualized_cost", "occurrences"]
    if work.empty:
        return pd.DataFrame(columns=cols)

    expenses = work.loc[work["type"] == "Expense"].copy()
    if expenses.empty:
        return pd.DataFrame(columns=cols)

    sub_keywords = (
        "netflix", "spotify", "prime", "amazon", "apple", "icloud", "hotstar",
        "youtube", "gym", "broadband", "wifi", "airtel", "jio", "chatgpt",
        "openai", "cursor", "github", "adobe", "cloud", "aws", "disney",
        "membership", "subscription",
    )

    rows: list[dict[str, Any]] = []
    expenses["desc_clean"] = expenses["description"].fillna("").astype(str).str.strip()

    # Match by known recurring keywords or repeated merchant descriptions
    for raw_desc, group in expenses.groupby("desc_clean"):
        desc = str(raw_desc).strip()
        if not desc or len(desc) < 3:
            continue
        d_lower = desc.lower()
        is_keyword_sub = any(kw in d_lower for kw in sub_keywords)
        is_repeated = len(group) >= 2

        if is_keyword_sub or is_repeated:
            avg_amt = float(group["amount"].mean())
            occurrences = len(group)
            annualized = avg_amt * (12.0 if occurrences >= 2 else 12.0)
            service_name = desc.title() if len(desc) < 30 else desc[:28] + "..."
            rows.append({
                "service_name": service_name,
                "category": str(group["category"].iloc[0]) if "category" in group.columns else "Subscriptions",
                "frequency": "Monthly",
                "avg_amount": round(avg_amt, 2),
                "annualized_cost": round(annualized, 2),
                "occurrences": occurrences,
            })

    if not rows:
        return pd.DataFrame(columns=cols)

    res_df = pd.DataFrame(rows).sort_values(by="annualized_cost", ascending=False).reset_index(drop=True)
    return res_df
