"""Gemini AI integration for structured transaction categorization and PII-safe insights."""

from __future__ import annotations

import json
import logging
import math
import os
import re
from enum import Enum
from pathlib import Path
from typing import Any, Optional, TypeVar

import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field, ValidationError

from analytics import build_ai_summary_payload

_ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(_ENV_PATH)
load_dotenv()

logger = logging.getLogger(__name__)

DEFAULT_GEMINI_MODEL = (os.getenv("GEMINI_MODEL") or "gemini-2.5-flash").strip()
TModel = TypeVar("TModel", bound=BaseModel)


class StandardCategory(str, Enum):
    """Canonical financial transaction categories."""

    FOOD_DINING = "Food & Dining"
    TRANSPORTATION = "Transportation"
    UTILITIES = "Utilities"
    HOUSING_RENT = "Housing & Rent"
    SHOPPING_LIFESTYLE = "Shopping & Lifestyle"
    HEALTHCARE = "Healthcare"
    ENTERTAINMENT_SUBSCRIPTIONS = "Entertainment & Subscriptions"
    TRAVEL = "Travel"
    EDUCATION = "Education"
    INVESTMENT = "Investment"
    SALARY = "Salary"
    FREELANCE_BUSINESS = "Freelance & Business"
    OTHER = "Other"


STANDARD_CATEGORIES: tuple[str, ...] = tuple(cat.value for cat in StandardCategory)


class TransactionCategorizationSchema(BaseModel):
    """Structured JSON output schema for zero-shot transaction categorization."""

    category: StandardCategory = Field(
        description="Best matching canonical personal finance category."
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score between 0.0 and 1.0.",
    )


class FinancialInsightsSchema(BaseModel):
    """Structured JSON output schema for financial optimization and anomaly flags."""

    summary_headline: str = Field(
        description="One concise sentence summarizing current financial posture."
    )
    optimization_suggestions: list[str] = Field(
        min_length=3,
        max_length=3,
        description="Exactly 3 concise, actionable financial optimization suggestions.",
    )
    unusual_spending_flags: list[str] = Field(
        default_factory=list,
        description="Concise flags highlighting any unusual spending spikes or risk concentrations.",
    )


class ReceiptExtractionSchema(BaseModel):
    """Structured schema for multimodal OCR receipt and invoice extraction."""

    merchant_name: str = Field(
        default="Scanned Merchant",
        description="Name of the merchant, store, vendor, or billing entity.",
    )
    date: Optional[str] = Field(
        default=None,
        description="Transaction date in YYYY-MM-DD ISO format if identified on receipt.",
    )
    total_amount: float = Field(
        default=0.0,
        ge=0.0,
        description="Total grand sum or final amount paid/due.",
    )
    category: StandardCategory = Field(
        default=StandardCategory.OTHER,
        description="Best matching personal finance category.",
    )
    tax_amount: Optional[float] = Field(
        default=0.0,
        description="Tax, GST, or VAT amount identified on the receipt.",
    )
    notes: Optional[str] = Field(
        default="",
        description="Summary of key line items or purchase description.",
    )
    confidence: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Confidence score between 0.0 and 1.0.",
    )


_KEYWORD_RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(
            r"\b(zerodha|groww|upstox|kite|kuvera|coin|sip|mutual\s*fund|nifty|sensex|etf|btc|bitcoin|eth|crypto|stock|equity|demat|brokerage)\b",
            re.IGNORECASE,
        ),
        StandardCategory.INVESTMENT.value,
    ),
    (
        re.compile(
            r"\b(salary|payroll|stipend|paycheck|direct\s*deposit|bonus|dividend)\b",
            re.IGNORECASE,
        ),
        StandardCategory.SALARY.value,
    ),
    (
        re.compile(
            r"\b(freelance|consulting|advisory|retainer|invoice|client\s*payment)\b",
            re.IGNORECASE,
        ),
        StandardCategory.FREELANCE_BUSINESS.value,
    ),
    (
        re.compile(
            r"\b(swiggy|zomato|ubereats|doordash|restaurant|cafe|coffee|starbucks|dining|bakery|grocer|supermarket|blinkit|zepto|instamart|bigbasket|dmart)\b",
            re.IGNORECASE,
        ),
        StandardCategory.FOOD_DINING.value,
    ),
    (
        re.compile(
            r"\b(uber|ola|lyft|rapido|metro|bus|train|rail|fuel|petrol|diesel|shell|hpcl|bpcl|toll|parking|cab|commute)\b",
            re.IGNORECASE,
        ),
        StandardCategory.TRANSPORTATION.value,
    ),
    (
        re.compile(
            r"\b(bescom|electricity|power|water|gas|broadband|fiber|fibernet|wifi|internet|jio|airtel|verizon|att|utility|utilities|mobile\s*bill|recharge)\b",
            re.IGNORECASE,
        ),
        StandardCategory.UTILITIES.value,
    ),
    (
        re.compile(
            r"\b(rent|landlord|lease|apartment|housing|maintenance|hoa|society)\b",
            re.IGNORECASE,
        ),
        StandardCategory.HOUSING_RENT.value,
    ),
    (
        re.compile(
            r"\b(hospital|clinic|pharmacy|apollo|medplus|1mg|pharmeasy|doctor|dental|health|medical|lab|diagnostic)\b",
            re.IGNORECASE,
        ),
        StandardCategory.HEALTHCARE.value,
    ),
    (
        re.compile(
            r"\b(netflix|spotify|prime\s*video|hotstar|disney|youtube|apple\s*one|icloud|google\s*one|subscription|cinema|movie|pvr|inox|gaming)\b",
            re.IGNORECASE,
        ),
        StandardCategory.ENTERTAINMENT_SUBSCRIPTIONS.value,
    ),
    (
        re.compile(
            r"\b(flight|indigo|air\s*india|vistara|hotel|airbnb|makemytrip|goibibo|booking\.com|resort|travel)\b",
            re.IGNORECASE,
        ),
        StandardCategory.TRAVEL.value,
    ),
    (
        re.compile(
            r"\b(udemy|coursera|edx|tuition|university|school|book|books|exam|certification)\b",
            re.IGNORECASE,
        ),
        StandardCategory.EDUCATION.value,
    ),
    (
        re.compile(
            r"\b(amazon|flipkart|myntra|ajio|nykaa|ikea|decathlon|apple\s*store|electronics|apparel|shopping|mall)\b",
            re.IGNORECASE,
        ),
        StandardCategory.SHOPPING_LIFESTYLE.value,
    ),
)


_PLACEHOLDER_KEYS: frozenset[str] = frozenset(
    {
        "your_gemini_api_key_here",
        "your_api_key_here",
        "replace_me",
        "changeme",
    }
)


def _resolve_api_key(api_key: Optional[str] = None) -> Optional[str]:
    candidate = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not candidate:
        try:
            import streamlit as st
            if hasattr(st, "secrets"):
                candidate = st.secrets.get("GEMINI_API_KEY") or st.secrets.get("GOOGLE_API_KEY")
        except Exception:
            pass
    if candidate:
        cleaned = candidate.strip()
        if cleaned and cleaned.lower() not in _PLACEHOLDER_KEYS:
            return cleaned
    return None


def is_gemini_configured(api_key: Optional[str] = None) -> bool:
    """Return True if a non-empty Gemini API key is available."""
    return _resolve_api_key(api_key) is not None


def _coerce_amount(amount: Any) -> float:
    """Convert numeric or string amount inputs safely into a finite float."""
    try:
        numeric = float(0.0 if amount is None else amount)
    except (TypeError, ValueError):
        return 0.0
    return numeric if math.isfinite(numeric) else 0.0


def _heuristic_categorize(description: str, amount: float) -> str:
    """Deterministic fallback classifier used when Gemini API is unavailable."""
    text = (description or "").strip()
    safe_amount = _coerce_amount(amount)
    for pattern, category in _KEYWORD_RULES:
        if pattern.search(text):
            return category
    if safe_amount >= 50000 and any(tok in text.lower() for tok in ("credit", "neft", "imps", "ach")):
        return StandardCategory.SALARY.value
    return StandardCategory.OTHER.value


def _strip_json_fences(raw_text: str) -> str:
    """Remove optional markdown ```json fences from raw model output."""
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def _parse_structured_response(response: Any, schema_cls: type[TModel]) -> Optional[TModel]:
    """Extract and validate a Pydantic model from a google-genai response."""
    parsed = getattr(response, "parsed", None)
    if parsed is not None:
        if isinstance(parsed, schema_cls):
            return parsed
        if isinstance(parsed, dict):
            try:
                return schema_cls.model_validate(parsed)
            except ValidationError:
                return None

    text = getattr(response, "text", None)
    if isinstance(text, str) and text.strip():
        try:
            return schema_cls.model_validate_json(_strip_json_fences(text))
        except ValidationError:
            return None
    return None


def categorize_transaction(
    description: str,
    amount: float,
    *,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
) -> str:
    """Classify an ambiguous transaction description into a standard category via Gemini JSON mode."""
    clean_desc = (description or "").strip()
    if not clean_desc:
        return StandardCategory.OTHER.value

    safe_amount = _coerce_amount(amount)
    resolved_key = _resolve_api_key(api_key)
    if not resolved_key:
        return _heuristic_categorize(clean_desc, safe_amount)

    resolved_model = (model or os.getenv("GEMINI_MODEL") or DEFAULT_GEMINI_MODEL).strip() or "gemini-2.5-flash"
    prompt = (
        "Classify this personal finance transaction into exactly one standard category.\n"
        f"Description: {clean_desc}\n"
        f"Absolute Amount: {safe_amount:.2f}\n"
        f"Allowed Categories: {', '.join(STANDARD_CATEGORIES)}"
    )

    try:
        client = genai.Client(api_key=resolved_key)
        response = client.models.generate_content(
            model=resolved_model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=(
                    "You are a precise financial transaction classifier. "
                    "Map merchant or bank ledger descriptions into the closest canonical category."
                ),
                temperature=0.1,
                response_mime_type="application/json",
                response_schema=TransactionCategorizationSchema,
            ),
        )
        structured = _parse_structured_response(response, TransactionCategorizationSchema)
        if structured is not None:
            cat_val = (
                structured.category.value
                if isinstance(structured.category, StandardCategory)
                else str(structured.category)
            )
            if cat_val in STANDARD_CATEGORIES:
                return cat_val
    except Exception as exc:
        logger.warning("Gemini categorization fallback triggered: %s", exc)

    return _heuristic_categorize(clean_desc, safe_amount)


def _format_insights_markdown(
    insights: FinancialInsightsSchema,
    *,
    source_label: str,
) -> str:
    """Format structured insights into clean Markdown for dashboard rendering."""
    lines = [
        f"**{insights.summary_headline.strip()}**",
        "",
        "### Top 3 Optimization Suggestions",
    ]
    for suggestion in insights.optimization_suggestions[:3]:
        lines.append(f"- {suggestion.strip()}")

    lines.extend(["", "### Spending Spikes & Risk Flags"])
    if insights.unusual_spending_flags:
        for flag in insights.unusual_spending_flags:
            lines.append(f"- {flag.strip()}")
    else:
        lines.append("- No unusual spending spikes detected across the analyzed timeframe.")

    lines.extend(["", f"<sub>Analysis engine: {source_label}</sub>"])
    return "\n".join(lines)


def _deterministic_insights_fallback(
    payload: dict[str, Any],
    *,
    currency_symbol: str = "₹",
    note: str = "Offline heuristic insights (configure GEMINI_API_KEY for Gemini AI analysis)",
) -> str:
    """Produce structured insights purely from aggregate stats when Gemini is unavailable."""
    kpis = payload.get("kpis", {})
    inflow = float(kpis.get("total_inflow", 0.0))
    outflow = float(kpis.get("total_outflow", 0.0))
    invested = float(kpis.get("capital_deployed", 0.0))
    savings_rate = float(kpis.get("savings_rate", 0.0))
    categories = payload.get("expense_categories", [])
    portfolio = payload.get("portfolio_allocation", [])
    spikes = payload.get("statistical_spending_spikes", [])

    top_cat = categories[0] if categories else None
    top_cat_name = top_cat["category"] if top_cat else "discretionary categories"
    top_cat_pct = float(top_cat["percentage"]) if top_cat else 0.0

    suggestions: list[str] = []
    if savings_rate < 20.0 and inflow > 0:
        suggestions.append(
            f"Savings rate is currently {savings_rate:.1f}%; target lifting it above 20% by trimming {top_cat_name} ({top_cat_pct:.1f}% of spend)."
        )
    elif inflow <= 0 and outflow > 0:
        suggestions.append(
            f"No inflow recorded in this period against {currency_symbol}{outflow:,.0f} in expenses; verify income entries or expand the date range."
        )
    else:
        suggestions.append(
            f"Maintain your {savings_rate:.1f}% savings rate and channel surplus cash flow above living expenses into automated recurring investments."
        )

    if top_cat:
        suggestions.append(
            f"{top_cat['category']} accounts for {top_cat['percentage']:.1f}% ({currency_symbol}{top_cat['amount']:,.0f}) of total expenses—set a monthly cap 10% below current run-rate."
        )
    else:
        suggestions.append("Track expenses consistently with clear categories to unlock category-level budget guardrails.")

    if portfolio:
        top_holding = portfolio[0]
        top_alloc = float(top_holding["allocation_pct"])
        if top_alloc > 55.0:
            suggestions.append(
                f"Portfolio concentration alert: {top_holding['asset_ticker']} represents {top_alloc:.1f}% of deployed capital; rebalance new contributions toward diversified assets."
            )
        else:
            suggestions.append(
                f"Capital deployed ({currency_symbol}{invested:,.0f}) is led by {top_holding['asset_ticker']} ({top_alloc:.1f}%); continue periodic cost-averaging."
            )
    else:
        suggestions.append(
            "Zero investment deployments detected in this window—consider automating a fixed monthly allocation."
        )

    flags: list[str] = []
    for sp in spikes[:3]:
        flags.append(
            f"**{sp['category']}** spiked in `{sp['month']}` to {currency_symbol}{sp['amount']:,.0f} "
            f"(+{sp['pct_above_baseline']:.1f}% above its {currency_symbol}{sp['baseline_mean']:,.0f} baseline)."
        )

    headline = (
        f"Analyzed {payload['period']['transaction_count']} transactions: "
        f"{currency_symbol}{inflow:,.0f} inflow, {currency_symbol}{outflow:,.0f} outflow, "
        f"{currency_symbol}{invested:,.0f} invested ({savings_rate:.1f}% savings rate)."
    )

    schema = FinancialInsightsSchema(
        summary_headline=headline,
        optimization_suggestions=suggestions[:3],
        unusual_spending_flags=flags,
    )
    return _format_insights_markdown(schema, source_label=note)


def generate_financial_insights(
    df: pd.DataFrame,
    *,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    currency_symbol: str = "₹",
) -> str:
    """Analyze PII-free aggregate spending/portfolio stats via Gemini and return Markdown insights."""
    if df is None or df.empty:
        return (
            "No transactions available in the active filter window. "
            "Add transactions or widen your date range to generate AI financial insights."
        )

    payload = build_ai_summary_payload(df)
    resolved_key = _resolve_api_key(api_key)
    if not resolved_key:
        return _deterministic_insights_fallback(payload, currency_symbol=currency_symbol)

    resolved_model = (model or os.getenv("GEMINI_MODEL") or DEFAULT_GEMINI_MODEL).strip() or "gemini-2.5-flash"
    prompt = (
        "Analyze the following anonymized personal finance and portfolio summary statistics.\n"
        f"Currency Symbol: {currency_symbol}\n"
        "Return:\n"
        "1. A crisp one-sentence headline summarizing overall financial health.\n"
        "2. Exactly 3 concise, high-impact, quantified financial optimization suggestions.\n"
        "3. Specific flags on any unusual spending spikes (or concentration risks) supported by the metrics.\n\n"
        f"Aggregated Metrics JSON:\n{json.dumps(payload, indent=2)}"
    )

    try:
        client = genai.Client(api_key=resolved_key)
        response = client.models.generate_content(
            model=resolved_model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=(
                    "You are a fiduciary personal wealth strategist and financial data analyst. "
                    "Never invent numbers not present in the JSON payload. Keep recommendations concise, specific, and practical."
                ),
                temperature=0.2,
                response_mime_type="application/json",
                response_schema=FinancialInsightsSchema,
            ),
        )
        structured = _parse_structured_response(response, FinancialInsightsSchema)
        if isinstance(structured, FinancialInsightsSchema):
            return _format_insights_markdown(
                structured,
                source_label=f"Google Gemini ({resolved_model}) • Structured JSON Mode",
            )
    except Exception as exc:
        logger.warning("Gemini insight generation failed, using deterministic fallback: %s", exc)
        return _deterministic_insights_fallback(
            payload,
            currency_symbol=currency_symbol,
            note=f"Deterministic fallback (Gemini request failed: {type(exc).__name__})",
        )

    return _deterministic_insights_fallback(payload, currency_symbol=currency_symbol)


def extract_receipt_data(
    image_bytes: bytes,
    mime_type: str = "image/jpeg",
    *,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
) -> ReceiptExtractionSchema:
    """Extract structured transaction data from receipt or invoice image using Gemini Vision."""
    resolved_key = _resolve_api_key(api_key)
    if not resolved_key or not image_bytes:
        return ReceiptExtractionSchema(
            merchant_name="Scanned Receipt",
            total_amount=0.0,
            category=StandardCategory.FOOD_DINING,
            confidence=0.5,
            notes="Offline mode: verified receipt placeholder. Configure GEMINI_API_KEY for live vision OCR.",
        )

    resolved_model = (model or os.getenv("GEMINI_MODEL") or DEFAULT_GEMINI_MODEL).strip() or "gemini-2.5-flash"
    prompt = (
        "Analyze this receipt or bill image. Extract:\n"
        "1. Merchant or vendor name.\n"
        "2. Date of transaction in YYYY-MM-DD ISO format.\n"
        "3. Total final amount paid or due.\n"
        "4. Canonical category from standard personal finance categories.\n"
        "5. Tax or GST amount if explicitly present.\n"
        "6. Concise notes summarizing line items."
    )

    try:
        client = genai.Client(api_key=resolved_key)
        response = client.models.generate_content(
            model=resolved_model,
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                prompt,
            ],
            config=types.GenerateContentConfig(
                system_instruction="You are an expert financial OCR parser extracting accurate data from receipts and invoices.",
                temperature=0.1,
                response_mime_type="application/json",
                response_schema=ReceiptExtractionSchema,
            ),
        )
        parsed = _parse_structured_response(response, ReceiptExtractionSchema)
        if isinstance(parsed, ReceiptExtractionSchema):
            return parsed
    except Exception as exc:
        logger.warning("Receipt vision OCR extraction failed, using safe fallback: %s", exc)

    return ReceiptExtractionSchema(
        merchant_name="Receipt (Scanned)",
        total_amount=0.0,
        category=StandardCategory.FOOD_DINING,
        confidence=0.5,
        notes="OCR parsing encountered an issue. Please adjust values manually before saving.",
    )


def _deterministic_copilot_fallback(
    query: str,
    summary_payload: Optional[dict[str, Any]],
    currency_symbol: str = "₹",
) -> str:
    """Provide rule-based answers from ledger payload when Gemini is unreachable."""
    q_lower = query.lower()
    if "highest" in q_lower or "spend" in q_lower or "spent" in q_lower or "most" in q_lower:
        cats = (summary_payload or {}).get("expense_categories", [])
        if cats:
            top = cats[0]
            return (
                f"Based on your ledger, your largest spending category is **{top.get('category')}** "
                f"with a total of **{currency_symbol}{top.get('amount', 0):,.2f}** "
                f"({top.get('percentage', 0):.1f}% of total outflow)."
            )
        return "No expenses recorded in the active filter timeframe."

    if "invest" in q_lower or "portfolio" in q_lower:
        port = (summary_payload or {}).get("portfolio_allocation", [])
        if port:
            total_cap = sum(p.get("total_invested", 0) for p in port)
            return (
                f"You currently hold **{len(port)} asset(s)** with a total deployed capital of "
                f"**{currency_symbol}{total_cap:,.2f}**."
            )
        return "No investment holdings found in this period."

    kpis = (summary_payload or {}).get("kpis", {})
    net = kpis.get("net_savings", 0.0)
    return (
        f"Here is your financial snapshot: Net cash flow is **{currency_symbol}{net:,.2f}** "
        f"with a savings rate of **{kpis.get('savings_rate_pct', 0.0):.1f}%**."
    )


def chat_financial_copilot(
    query: str,
    *,
    history: Optional[list[dict[str, str]]] = None,
    summary_payload: Optional[dict[str, Any]] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    currency_symbol: str = "₹",
) -> str:
    """Conversational financial copilot answering user queries based on their personal financial ledger."""
    clean_query = (query or "").strip()
    if not clean_query:
        return "Please ask a question about your finances, budget, or investments."

    resolved_key = _resolve_api_key(api_key)
    if not resolved_key:
        return _deterministic_copilot_fallback(clean_query, summary_payload, currency_symbol)

    payload_str = json.dumps(summary_payload or {}, indent=2)
    resolved_model = (model or os.getenv("GEMINI_MODEL") or DEFAULT_GEMINI_MODEL).strip() or "gemini-2.5-flash"
    system_instruction = (
        "You are an elite, fiduciary personal finance copilot and wealth strategist. "
        "You have access to the user's aggregated, anonymized financial metrics and KPIs provided below. "
        f"Currency Symbol is {currency_symbol}. "
        "Answer the user's question directly, accurately, and quantitatively using ONLY the facts provided in the financial context. "
        "Keep answers concise, actionable, and encouraging. Never invent numbers.\n\n"
        f"FINANCIAL CONTEXT JSON:\n{payload_str}"
    )

    contents: list[str] = []
    if history:
        for msg in history[-6:]:  # Keep recent context
            role = "User" if msg.get("role") == "user" else "Assistant"
            contents.append(f"{role}: {msg.get('content', '')}")
    contents.append(f"User: {clean_query}\nAssistant:")

    try:
        client = genai.Client(api_key=resolved_key)
        response = client.models.generate_content(
            model=resolved_model,
            contents="\n\n".join(contents),
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.3,
            ),
        )
        return (response.text or "").strip() or "I analyzed your financial metrics, but could not produce a response."
    except Exception as exc:
        logger.warning("Gemini copilot chat failed, falling back to deterministic: %s", exc)
        return _deterministic_copilot_fallback(clean_query, summary_payload, currency_symbol)


__all__ = [
    "DEFAULT_GEMINI_MODEL",
    "FinancialInsightsSchema",
    "ReceiptExtractionSchema",
    "StandardCategory",
    "STANDARD_CATEGORIES",
    "TransactionCategorizationSchema",
    "categorize_transaction",
    "chat_financial_copilot",
    "extract_receipt_data",
    "generate_financial_insights",
    "is_gemini_configured",
]
