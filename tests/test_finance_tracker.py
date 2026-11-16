"""Comprehensive unit and integration test suite for finance_tracker."""

from __future__ import annotations

import datetime as dt
import json
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd

PACKAGE_DIR = Path(__file__).resolve().parent.parent
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

import ai_engine  # noqa: E402
import analytics  # noqa: E402
import auth  # noqa: E402
import database  # noqa: E402
from components import charts, forms, kpi_cards  # noqa: E402


class DatabaseTests(unittest.TestCase):
    """Tests for SQLite pooling, migrations, validation, CRUD, and concurrency."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmpdir.name) / "test_finance.db"
        database.init_db(self.db_path)

    def tearDown(self) -> None:
        database.close_pools(self.db_path)
        self.tmpdir.cleanup()

    def test_idempotent_init_and_empty_query(self) -> None:
        database.init_db(self.db_path)
        df = database.get_transactions(db_path=self.db_path)
        self.assertTrue(df.empty)
        self.assertEqual(list(df.columns), database.TRANSACTION_COLUMNS)

    def test_add_get_update_delete_lifecycle(self) -> None:
        tx_id = database.add_transaction(
            date="2026-10-05",
            trans_type="Expense",
            category="Food & Dining",
            amount=640.50,
            description="Swiggy Bangalore",
            asset_ticker="SHOULD_BE_NULL",
            quantity=5.0,
            is_ai_tagged=True,
            db_path=self.db_path,
        )
        self.assertGreater(tx_id, 0)

        df = database.get_transactions(db_path=self.db_path)
        self.assertEqual(len(df), 1)
        row = df.iloc[0]
        self.assertEqual(row["type"], "Expense")
        self.assertIsNone(row["asset_ticker"])
        self.assertEqual(row["quantity"], 0.0)
        self.assertEqual(row["is_ai_tagged"], 1)

        updated = database.update_transaction(
            tx_id,
            date="2026-10-06",
            trans_type="Expense",
            category="Utilities",
            amount=900.0,
            description="BESCOM Power",
            db_path=self.db_path,
        )
        self.assertTrue(updated)
        df_after = database.get_transactions(category="Utilities", db_path=self.db_path)
        self.assertEqual(len(df_after), 1)
        self.assertEqual(df_after.iloc[0]["amount"], 900.0)

        deleted = database.delete_transaction(tx_id, db_path=self.db_path)
        self.assertTrue(deleted)
        self.assertTrue(database.get_transactions(db_path=self.db_path).empty)

    def test_validation_rejects_invalid_records(self) -> None:
        with self.assertRaises(ValueError):
            database.add_transaction(
                "2026-10-01", "Expense", "Food", -100.0, db_path=self.db_path
            )
        with self.assertRaises(ValueError):
            database.add_transaction(
                "2026-10-01", "Transfer", "Other", 100.0, db_path=self.db_path
            )
        with self.assertRaises(ValueError):
            database.add_transaction(
                "not-a-date", "Income", "Salary", 1000.0, db_path=self.db_path
            )

    def test_bulk_insert_and_filters(self) -> None:
        sample = pd.DataFrame(
            [
                {
                    "date": "2026-09-01",
                    "type": "Income",
                    "category": "Salary",
                    "amount": 100000.0,
                    "description": "Payroll",
                },
                {
                    "date": "2026-09-10",
                    "type": "Investment",
                    "category": "Investment",
                    "amount": 25000.0,
                    "description": "Zerodha SIP",
                    "asset_ticker": "nifty50",
                    "quantity": 100.0,
                    "unit_price": 0.0,
                    "is_ai_tagged": 1,
                },
            ]
        )
        inserted = database.bulk_insert_transactions(sample, db_path=self.db_path)
        self.assertEqual(inserted, 2)
        self.assertEqual(database.bulk_insert_transactions(pd.DataFrame(), db_path=self.db_path), 0)

        inv_df = database.get_transactions(
            start_date="2026-09-05",
            end_date="2026-09-30",
            trans_type="Investment",
            ai_tagged_only=True,
            db_path=self.db_path,
        )
        self.assertEqual(len(inv_df), 1)
        self.assertEqual(inv_df.iloc[0]["asset_ticker"], "NIFTY50")
        self.assertEqual(int(inv_df.iloc[0]["is_ai_tagged"]), 1)
        self.assertAlmostEqual(inv_df.iloc[0]["unit_price"], 250.0, places=4)

    def test_concurrent_thread_pool_writes_and_reads(self) -> None:
        def _worker(idx: int) -> int:
            tx_id = database.add_transaction(
                date="2026-10-15",
                trans_type="Expense",
                category="Food & Dining",
                amount=100.0 + idx,
                description=f"Concurrent Tx {idx}",
                db_path=self.db_path,
            )
            _ = database.get_transactions(db_path=self.db_path)
            return tx_id

        with ThreadPoolExecutor(max_workers=6) as executor:
            ids = list(executor.map(_worker, range(24)))

        self.assertEqual(len(set(ids)), 24)
        all_df = database.get_transactions(db_path=self.db_path)
        self.assertEqual(len(all_df), 24)

    def test_seed_sample_data(self) -> None:
        count = database.seed_sample_data(
            reference_date=dt.date(2026, 11, 10),
            db_path=self.db_path,
        )
        self.assertGreater(count, 20)
        df = database.get_transactions(db_path=self.db_path)
        self.assertEqual(len(df), count)


class AnalyticsTests(unittest.TestCase):
    """Tests for vectorized KPI, cash flow, category, portfolio, and spike logic."""

    def test_empty_dataframe_safety(self) -> None:
        empty = pd.DataFrame(columns=database.TRANSACTION_COLUMNS)
        kpis = analytics.compute_kpis(empty)
        self.assertEqual(kpis["total_inflow"], 0.0)
        self.assertEqual(kpis["savings_rate"], 0.0)
        self.assertTrue(analytics.compute_monthly_cash_flow(empty).empty)
        self.assertTrue(analytics.compute_category_breakdown(empty).empty)
        self.assertTrue(analytics.compute_portfolio_allocation(empty).empty)
        self.assertTrue(analytics.detect_spending_spikes(empty).empty)

    def test_zero_income_savings_rate_safe(self) -> None:
        df = pd.DataFrame(
            [
                {
                    "date": "2026-10-01",
                    "type": "Expense",
                    "category": "Food & Dining",
                    "amount": 1200.0,
                }
            ]
        )
        kpis = analytics.compute_kpis(df)
        self.assertEqual(kpis["total_inflow"], 0.0)
        self.assertEqual(kpis["total_outflow"], 1200.0)
        self.assertEqual(kpis["net_savings"], -1200.0)
        self.assertEqual(kpis["savings_rate"], 0.0)

    def test_vectorized_kpis_portfolio_and_pii_free_payload(self) -> None:
        df = pd.DataFrame(
            [
                {
                    "id": 1,
                    "date": "2026-09-01",
                    "type": "Income",
                    "category": "Salary",
                    "amount": 100000.0,
                    "description": "SECRET_EMP_ID_99911 Payroll",
                    "asset_ticker": None,
                    "quantity": 0.0,
                    "unit_price": 0.0,
                    "is_ai_tagged": 0,
                },
                {
                    "id": 2,
                    "date": "2026-09-05",
                    "type": "Expense",
                    "category": "Food & Dining",
                    "amount": 5000.0,
                    "description": "PRIVATE_ORDER_ADDR_42 Swiggy",
                    "asset_ticker": None,
                    "quantity": 0.0,
                    "unit_price": 0.0,
                    "is_ai_tagged": 1,
                },
                {
                    "id": 3,
                    "date": "2026-10-05",
                    "type": "Expense",
                    "category": "Food & Dining",
                    "amount": 15000.0,
                    "description": "PRIVATE_DINNER_99",
                    "asset_ticker": None,
                    "quantity": 0.0,
                    "unit_price": 0.0,
                    "is_ai_tagged": 1,
                },
                {
                    "id": 4,
                    "date": "2026-09-10",
                    "type": "Investment",
                    "category": "Investment",
                    "amount": 10000.0,
                    "description": "Buy NIFTY50",
                    "asset_ticker": "NIFTY50",
                    "quantity": 40.0,
                    "unit_price": 250.0,
                    "is_ai_tagged": 0,
                },
                {
                    "id": 5,
                    "date": "2026-10-10",
                    "type": "Investment",
                    "category": "Investment",
                    "amount": 12000.0,
                    "description": "Buy NIFTY50",
                    "asset_ticker": "NIFTY50",
                    "quantity": 40.0,
                    "unit_price": 300.0,
                    "is_ai_tagged": 0,
                },
            ]
        )

        kpis = analytics.compute_kpis(df)
        self.assertEqual(kpis["total_inflow"], 100000.0)
        self.assertEqual(kpis["total_outflow"], 20000.0)
        self.assertEqual(kpis["capital_deployed"], 22000.0)
        self.assertEqual(kpis["net_savings"], 80000.0)
        self.assertEqual(kpis["savings_rate"], 80.0)

        monthly = analytics.compute_monthly_cash_flow(df)
        self.assertEqual(list(monthly["month"]), ["2026-09", "2026-10"])
        self.assertEqual(float(monthly.loc[0, "Net Cash Flow"]), 95000.0)
        self.assertEqual(float(monthly.loc[1, "Net Cash Flow"]), -15000.0)

        with patch("analytics.fetch_live_price", return_value=300.0):
            portfolio = analytics.compute_portfolio_allocation(df)
        self.assertEqual(len(portfolio), 1)
        self.assertEqual(portfolio.iloc[0]["asset_ticker"], "NIFTY50")
        self.assertEqual(portfolio.iloc[0]["total_invested"], 22000.0)
        self.assertEqual(portfolio.iloc[0]["total_units"], 80.0)
        self.assertAlmostEqual(portfolio.iloc[0]["avg_cost_basis"], 275.0, places=4)
        self.assertIn("current_price", portfolio.columns)
        self.assertIn("current_value", portfolio.columns)
        self.assertIn("unrealized_pnl", portfolio.columns)
        self.assertIn("return_pct", portfolio.columns)

        budget_alerts = analytics.compute_budget_alerts(df)
        self.assertFalse(budget_alerts.empty)
        food_alert = budget_alerts[budget_alerts["category"] == "Food & Dining"]
        self.assertFalse(food_alert.empty)
        self.assertIn("Spike", food_alert.iloc[0]["status"])

        spikes = analytics.detect_spending_spikes(df)
        self.assertFalse(spikes.empty)
        self.assertEqual(spikes.iloc[0]["month"], "2026-10")

        payload = analytics.build_ai_summary_payload(df)
        serialized = json.dumps(payload)
        self.assertNotIn("SECRET_EMP_ID_99911", serialized)
        self.assertNotIn("PRIVATE_ORDER_ADDR_42", serialized)


class AIEngineTests(unittest.TestCase):
    """Tests for Gemini structured output categorization, insights, and offline fallback."""

    def test_heuristic_categorization_without_api_key(self) -> None:
        with patch.dict("os.environ", {"GEMINI_API_KEY": "", "GOOGLE_API_KEY": ""}, clear=False):
            self.assertFalse(ai_engine.is_gemini_configured("your_gemini_api_key_here"))
            self.assertEqual(
                ai_engine.categorize_transaction("Swiggy Bangalore", 450.0),
                "Food & Dining",
            )
            self.assertEqual(
                ai_engine.categorize_transaction("Uber trip to airport", 780.0),
                "Transportation",
            )
            self.assertEqual(
                ai_engine.categorize_transaction("Zerodha fund add", 25000.0),
                "Investment",
            )

    def test_mocked_gemini_structured_categorization_and_insights(self) -> None:
        mock_cat_resp = SimpleNamespace(
            parsed=ai_engine.TransactionCategorizationSchema(
                category=ai_engine.StandardCategory.FOOD_DINING,
                confidence=0.96,
            ),
            text=None,
        )
        mock_ins_resp = SimpleNamespace(
            parsed=ai_engine.FinancialInsightsSchema(
                summary_headline="Strong 65% savings rate with dining surge in October.",
                optimization_suggestions=[
                    "Cap Food & Dining to 10% of monthly inflow.",
                    "Automate surplus into NIFTY50 SIP.",
                    "Consolidate recurring subscriptions.",
                ],
                unusual_spending_flags=["Food & Dining spiked +120% in 2026-10."],
            ),
            text=None,
        )

        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = [mock_cat_resp, mock_ins_resp]

        with patch("ai_engine.genai.Client", return_value=mock_client):
            cat = ai_engine.categorize_transaction(
                "Ambiguous Merchant XYZ", 500.0, api_key="fake-test-key"
            )
            self.assertEqual(cat, "Food & Dining")

            sample_df = pd.DataFrame(
                [
                    {
                        "date": "2026-10-01",
                        "type": "Income",
                        "category": "Salary",
                        "amount": 100000.0,
                        "description": "DO_NOT_SEND_PII_NAME",
                    },
                    {
                        "date": "2026-10-05",
                        "type": "Expense",
                        "category": "Food & Dining",
                        "amount": 12000.0,
                        "description": "DO_NOT_SEND_HOME_ADDR",
                    },
                ]
            )
            insights_md = ai_engine.generate_financial_insights(
                sample_df, api_key="fake-test-key"
            )
            self.assertIn("Top 3 Optimization Suggestions", insights_md)
            self.assertIn("Food & Dining spiked", insights_md)

            # Verify no raw PII leaked into prompt
            ins_call_kwargs = mock_client.models.generate_content.call_args_list[1].kwargs
            self.assertNotIn("DO_NOT_SEND_PII_NAME", ins_call_kwargs["contents"])
            self.assertNotIn("DO_NOT_SEND_HOME_ADDR", ins_call_kwargs["contents"])
            self.assertEqual(
                ins_call_kwargs["config"].response_mime_type,
                "application/json",
            )

    def test_fenced_json_and_nan_amount_safety(self) -> None:
        fenced_resp = SimpleNamespace(
            parsed=None,
            text='```json\n{"category": "Utilities", "confidence": 0.91}\n```',
        )
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = fenced_resp

        with patch.dict("os.environ", {"GEMINI_MODEL": ""}, clear=False):
            with patch("ai_engine.genai.Client", return_value=mock_client):
                cat = ai_engine.categorize_transaction(
                    "BESCOM Power Bill", float("nan"), api_key="fake-test-key"
                )
                self.assertEqual(cat, "Utilities")
                call_kwargs = mock_client.models.generate_content.call_args.kwargs
                self.assertEqual(call_kwargs["model"], "gemini-2.5-flash")


class ChartsAndCSVTests(unittest.TestCase):
    """Tests for Plotly chart builders (empty & populated) and CSV parser."""

    def test_all_charts_handle_empty_and_populated_frames(self) -> None:
        empty_cf = analytics.compute_monthly_cash_flow(pd.DataFrame())
        empty_bd = analytics.compute_category_breakdown(pd.DataFrame())
        empty_pf = analytics.compute_portfolio_allocation(pd.DataFrame())

        self.assertIsNotNone(charts.build_cash_flow_chart(empty_cf))
        self.assertIsNotNone(charts.build_spending_donut_chart(empty_bd))
        self.assertIsNotNone(charts.build_category_bar_chart(empty_bd))
        self.assertIsNotNone(charts.build_portfolio_treemap(empty_pf))
        self.assertIsNotNone(charts.build_3d_spending_landscape(pd.DataFrame()))
        self.assertIsNotNone(charts.build_health_score_gauge(None))
        self.assertIsNotNone(charts.build_50_30_20_chart(None))

        df = pd.DataFrame(
            [
                {"date": "2026-10-01", "type": "Income", "category": "Salary", "amount": 90000.0},
                {"date": "2026-10-04", "type": "Expense", "category": "Food & Dining", "amount": 5000.0},
                {
                    "date": "2026-10-08",
                    "type": "Investment",
                    "category": "Investment",
                    "amount": 20000.0,
                    "asset_ticker": "NIFTY50",
                    "quantity": 80.0,
                },
            ]
        )
        cf_fig = charts.build_cash_flow_chart(analytics.compute_monthly_cash_flow(df))
        donut_fig = charts.build_spending_donut_chart(analytics.compute_category_breakdown(df))
        bar_fig = charts.build_category_bar_chart(analytics.compute_category_breakdown(df))
        tree_fig = charts.build_portfolio_treemap(analytics.compute_portfolio_allocation(df, fetch_live_prices=False))
        cube_fig = charts.build_3d_spending_landscape(df)

        self.assertEqual(len(cf_fig.data), 3)
        self.assertEqual(len(donut_fig.data), 1)
        self.assertEqual(len(bar_fig.data), 1)
        self.assertEqual(len(tree_fig.data), 1)
        self.assertEqual(len(cube_fig.data), 2)
        self.assertNotIn("(?)", str(tree_fig.data[0].customdata))

        # Bar chart resilient when percentage column is omitted
        no_pct_bd = analytics.compute_category_breakdown(df).drop(columns=["percentage"])
        self.assertEqual(len(charts.build_category_bar_chart(no_pct_bd).data), 1)

        # Health score & 50/30/20 rule charts
        health = analytics.compute_financial_health_score(df)
        self.assertIn("score", health)
        self.assertIn("grade", health)
        gauge_fig = charts.build_health_score_gauge(health["score"], health["label"], health["grade"])
        self.assertEqual(len(gauge_fig.data), 1)

        alloc = analytics.compute_50_30_20(df)
        self.assertIn("needs_pct", alloc)
        self.assertIn("wants_pct", alloc)
        alloc_fig = charts.build_50_30_20_chart(alloc)
        self.assertEqual(len(alloc_fig.data), 3)

    def test_recurring_subscriptions_and_copilot_chat(self) -> None:
        sub_df = pd.DataFrame(
            [
                {"date": "2026-09-01", "type": "Expense", "category": "Entertainment", "amount": 649.0, "description": "Netflix Subscription"},
                {"date": "2026-10-01", "type": "Expense", "category": "Entertainment", "amount": 649.0, "description": "Netflix Subscription"},
                {"date": "2026-09-15", "type": "Expense", "category": "Utilities", "amount": 999.0, "description": "Airtel Broadband"},
            ]
        )
        detected = analytics.detect_recurring_subscriptions(sub_df)
        self.assertFalse(detected.empty)
        self.assertTrue(any("Netflix" in name for name in detected["service_name"]))

        # Copilot chat deterministic offline fallback check
        summary = analytics.build_ai_summary_payload(sub_df)
        reply = ai_engine.chat_financial_copilot("What is my highest spend?", summary_payload=summary)
        self.assertIn("Entertainment", reply)

        # Receipt extraction schema verification
        rec = ai_engine.extract_receipt_data(b"fake-image-bytes", api_key=None)
        self.assertIsInstance(rec, ai_engine.ReceiptExtractionSchema)
        self.assertEqual(rec.confidence, 0.5)

    def test_kpi_format_currency_edge_cases(self) -> None:
        self.assertEqual(kpi_cards.format_currency(12500.0, "$"), "$12,500")
        self.assertEqual(kpi_cards.format_currency(-420.5, "₹"), "-₹420.50")
        self.assertEqual(kpi_cards.format_currency(float("inf"), "₹"), "₹0")
        self.assertEqual(kpi_cards.format_currency(float("nan"), "₹"), "₹0")
        self.assertEqual(kpi_cards.format_currency(None, "$"), "$0")

    def test_csv_empty_and_duplicate_alias_headers(self) -> None:
        empty_df, empty_stats = forms.parse_and_enrich_csv(b"")
        self.assertTrue(empty_df.empty)
        self.assertEqual(empty_stats["parsed_rows"], 0)

        dup_csv = (
            "Date,Description,Amount,Value,Type\n"
            "2026-10-01,Swiggy Bangalore,500,ignored,Dr\n"
            "2026-10-02,Bonus Credit,10000,ignored,Cr\n"
        )
        parsed_df, stats = forms.parse_and_enrich_csv(
            dup_csv.encode("utf-8"),
            categorizer=lambda desc, amt: ai_engine._heuristic_categorize(desc, amt),
        )
        self.assertEqual(stats["parsed_rows"], 2)
        self.assertEqual(list(parsed_df["type"]), ["Expense", "Income"])

    def test_csv_bank_statement_parser_with_missing_categories(self) -> None:
        csv_text = (
            "Transaction Date,Narration,Debit,Credit,Category\n"
            "2026-10-01,Monthly Payroll,,\"₹1,20,000.00\",Salary\n"
            "02/10/2026,Swiggy Bangalore,\"₹650.50\",,\n"
            "15-10-2026,Uber Airport Trip,(920.00),,\n"
            "bad-row,Invalid,0,,\n"
        )
        parsed_df, stats = forms.parse_and_enrich_csv(
            csv_text.encode("utf-8"),
            categorizer=lambda desc, amt: ai_engine._heuristic_categorize(desc, amt),
        )
        self.assertEqual(stats["parsed_rows"], 3)
        self.assertEqual(stats["skipped_rows"], 1)
        self.assertEqual(stats["ai_tagged_rows"], 2)
        self.assertEqual(list(parsed_df["type"]), ["Income", "Expense", "Expense"])
        self.assertEqual(list(parsed_df["category"]), ["Salary", "Food & Dining", "Transportation"])
        self.assertAlmostEqual(float(parsed_df.loc[0, "amount"]), 120000.0, places=2)
        self.assertEqual(parsed_df.loc[2, "date"], "2026-10-15")


class AuthTests(unittest.TestCase):
    """Tests for email/phone auth, PBKDF2 hashing, OTP verification, and user data isolation."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmpdir.name) / "auth_test.db"
        database.init_db(self.db_path)

    def tearDown(self) -> None:
        database.close_pools(self.db_path)
        self.tmpdir.cleanup()

    def test_identifier_normalization_and_validation(self) -> None:
        self.assertEqual(auth.normalize_identifier("  Alex@Example.COM "), ("alex@example.com", "email"))
        self.assertEqual(auth.normalize_identifier("+91 98765-43210"), ("+919876543210", "phone"))
        self.assertEqual(auth.normalize_identifier("9876543210"), ("9876543210", "phone"))
        with self.assertRaises(ValueError):
            auth.normalize_identifier("invalid-contact")

    def test_smtp_and_email_dispatch_helpers(self) -> None:
        with patch.dict("os.environ", {"SMTP_HOST": "", "SMTP_USER": "", "SMTP_PASSWORD": ""}):
            self.assertFalse(auth.is_smtp_configured())
            self.assertFalse(auth.send_otp_email("test@example.com", "123456"))

        with patch.dict("os.environ", {"SMTP_HOST": "smtp.example.com", "SMTP_USER": "u", "SMTP_PASSWORD": "p"}):
            self.assertTrue(auth.is_smtp_configured())

    def test_register_password_otp_and_user_isolation(self) -> None:
        user_a = auth.register_user(
            "Aarav Sharma",
            "aarav@example.com",
            "secret123",
            secondary_identifier="+919876543210",
            db_path=self.db_path,
        )
        user_b = auth.register_user(
            "Maya Verma",
            "9811122233",
            "secret456",
            db_path=self.db_path,
        )

        # Authenticate via email OR secondary phone
        by_email = auth.authenticate_with_password("aarav@example.com", "secret123", db_path=self.db_path)
        by_phone = auth.authenticate_with_password("+91 98765 43210", "secret123", db_path=self.db_path)
        self.assertEqual(by_email["id"], user_a["id"])
        self.assertEqual(by_phone["id"], user_a["id"])

        with self.assertRaises(ValueError):
            auth.authenticate_with_password("aarav@example.com", "wrong-pass", db_path=self.db_path)

        # Duplicate registration rejected
        with self.assertRaises(ValueError):
            auth.register_user("Dup", "+919876543210", "abcdef", db_path=self.db_path)

        # OTP login works once and rejects reuse
        otp, _ = auth.request_otp("9811122233", db_path=self.db_path)
        self.assertEqual(len(otp), 6)
        otp_user = auth.verify_otp_login("9811122233", otp, db_path=self.db_path)
        self.assertEqual(otp_user["id"], user_b["id"])
        with self.assertRaises(ValueError):
            auth.verify_otp_login("9811122233", otp, db_path=self.db_path)

        # Per-user transaction isolation
        tx_a = database.add_transaction(
            "2026-10-10", "Expense", "Food & Dining", 500.0, user_id=user_a["id"], db_path=self.db_path
        )
        database.add_transaction(
            "2026-10-10", "Income", "Salary", 80000.0, user_id=user_b["id"], db_path=self.db_path
        )
        self.assertEqual(len(database.get_transactions(user_id=user_a["id"], db_path=self.db_path)), 1)
        self.assertEqual(len(database.get_transactions(user_id=user_b["id"], db_path=self.db_path)), 1)
        self.assertFalse(database.delete_transaction(tx_a, user_id=user_b["id"], db_path=self.db_path))
        self.assertTrue(database.delete_transaction(tx_a, user_id=user_a["id"], db_path=self.db_path))


class PostgresAndSupabaseConfigTests(unittest.TestCase):
    """Test PostgreSQL & Supabase URL detection, masking, and proxy query adaptations."""

    def test_backend_detection_and_masking(self) -> None:
        from database import _mask_db_url, get_database_backend, get_db_target_info

        # SQLite path detection
        with tempfile.NamedTemporaryFile(suffix=".db") as f:
            self.assertEqual(get_database_backend(f.name), "sqlite")
            info = get_db_target_info(f.name)
            self.assertEqual(info["backend"], "sqlite")
            self.assertFalse(info["is_cloud"])

        # Postgres URL detection
        pg_url = "postgresql://postgres:secretpassword@db.xyz.supabase.co:5432/postgres"
        self.assertEqual(get_database_backend(pg_url), "postgresql")
        pg_info = get_db_target_info(pg_url)
        self.assertEqual(pg_info["backend"], "postgresql")
        self.assertTrue(pg_info["is_cloud"])
        self.assertIn("Supabase", pg_info["provider"])
        self.assertNotIn("secretpassword", pg_info["target"])
        self.assertIn(":***", pg_info["target"])

        # Masking helper
        masked = _mask_db_url("postgresql://user:mypass@host:5432/db")
        self.assertEqual(masked, "postgresql://user:***@host:5432/db")

    def test_postgres_proxy_adapts_placeholders_and_returning(self) -> None:
        from unittest.mock import MagicMock
        from database import PostgresConnectionProxy

        mock_raw_conn = MagicMock()
        mock_cur = MagicMock()
        mock_cur.fetchone.return_value = {"id": 42}
        mock_raw_conn.execute.return_value = mock_cur

        proxy = PostgresConnectionProxy(mock_raw_conn)

        # 1. Adapt ? to %s and add RETURNING id on insert
        res = proxy.execute("INSERT INTO transactions (amount, type) VALUES (?, ?);", (500.0, "Expense"))
        mock_raw_conn.execute.assert_called_once_with(
            "INSERT INTO transactions (amount, type) VALUES (%s, %s) RETURNING id;",
            (500.0, "Expense"),
        )
        self.assertEqual(res.lastrowid, 42)

        # 2. Select query adapts ? without appending RETURNING
        mock_raw_conn.reset_mock()
        proxy.execute("SELECT * FROM transactions WHERE date >= ? AND type = ?", ("2026-01-01", "Income"))
        mock_raw_conn.execute.assert_called_once_with(
            "SELECT * FROM transactions WHERE date >= %s AND type = %s",
            ("2026-01-01", "Income"),
        )


class StreamlitAppSmokeTest(unittest.TestCase):
    """Headless Streamlit AppTest verification for auth portal and 3D dashboard."""

    def test_streamlit_app_auth_and_dashboard_without_exceptions(self) -> None:
        from streamlit.testing.v1 import AppTest

        with tempfile.TemporaryDirectory() as tmpdir:
            test_db = Path(tmpdir) / "app_smoke.db"
            try:
                with patch.dict("os.environ", {"FINANCE_DB_PATH": str(test_db), "DATABASE_URL": "", "SUPABASE_DB_URL": ""}):
                    with patch("analytics.fetch_live_price", return_value=300.0):
                        # 1. Unauthenticated view renders 3D auth portal cleanly
                        at = AppTest.from_file(str(PACKAGE_DIR / "app.py"), default_timeout=20)
                        at.run()
                        self.assertFalse(at.exception, f"Auth portal exception: {at.exception}")
                        demo_btn = [b for b in at.button if b.key == "auth_demo_login_btn"]
                        self.assertTrue(demo_btn, "Expected Demo Account login button on auth screen")

                        # 2. Sign in with Demo Account (seeds data & enters 3D dashboard)
                        demo_btn[0].click().run()
                        self.assertFalse(at.exception, f"Demo login exception: {at.exception}")
                        self.assertGreaterEqual(len(at.metric), 4)

                        # 3. Trigger on-demand AI Portfolio Analysis
                        analyze_btn = [b for b in at.button if b.key == "analyze_portfolio_btn"][0]
                        analyze_btn.click().run()
                        self.assertFalse(at.exception, f"Analyze exception: {at.exception}")
                        self.assertTrue(
                            any("Top 3 Optimization Suggestions" in m.value for m in at.markdown)
                        )

                        # 4. Switch Date Range preset & manual transaction type radio
                        date_sel = [
                            s for s in at.sidebar.selectbox if s.key == "sidebar_date_preset"
                        ][0]
                        date_sel.set_value("YTD").run()
                        self.assertFalse(at.exception, f"YTD exception: {at.exception}")

                        tx_radio = [
                            r for r in at.sidebar.radio if r.key == "manual_tx_type_selector"
                        ][0]
                        tx_radio.set_value("Investment").run()
                        self.assertFalse(at.exception, f"Investment toggle exception: {at.exception}")

                        # 5. Sign Out returns cleanly to auth portal
                        logout_btn = [b for b in at.sidebar.button if b.key == "sidebar_logout_btn"][0]
                        logout_btn.click().run()
                        self.assertFalse(at.exception, f"Logout exception: {at.exception}")
                        self.assertTrue(
                            any(b.key == "auth_demo_login_btn" for b in at.button),
                            "Expected return to auth portal after logout",
                        )
            finally:
                database.close_pools(test_db)


if __name__ == "__main__":
    unittest.main()
