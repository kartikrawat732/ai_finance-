# 🌌 AI-Powered 3D Personal Finance & Investment Vault

A high-performance, single-tenant personal financial intelligence system built with **Python 3.11+**, **Streamlit**, **Three.js & GSAP (3D Spatial Layer)**, **Pandas/NumPy**, **Plotly**, **SQLite3 / Supabase PostgreSQL**, and the **Google GenAI SDK (Gemini 2.5 Flash)**.

## Architecture & Project Structure

```text
finance_tracker/
├── app.py                  # Streamlit 3D spatial layout, auth routing, 5 analytical views, session state
├── auth.py                 # Email/phone normalization, PBKDF2 password hashing, OTP verification
├── database.py             # Dual engine: SQLite WAL connection pool & Supabase PostgreSQL with RLS
├── analytics.py            # Vectorized KPIs, cash flow, 50/30/20 rule, 0-100 health score & subscriptions
├── ai_engine.py            # Gemini Vision OCR receipt scanner, conversational copilot & structured insights
├── components/
│   ├── __init__.py
│   ├── spatial_3d.py       # Three.js WebGL & GSAP 3D spatial engine, cursor parallax & card tilt
│   ├── auth_ui.py          # 3D animated glassmorphic Sign In / Create Account portal
│   ├── kpi_cards.py        # 4-card top-row KPI metrics with sparklines
│   ├── charts.py           # Plotly builders (Health Gauge, 50/30/20 Bar, Cash Flow, Donut, 3D Spending Cube)
│   └── forms.py            # Dynamic manual entry form, bank CSV parser & Gemini Vision receipt scanner
├── tests/
│   └── test_finance_tracker.py
├── .env.example
├── requirements.txt
└── README.md
```

### Core Engineering & Feature Highlights
- **Interactive 3D Spatial Experience (Three.js & GSAP)**: Real-time cursor parallax, state-reactive focus acceleration, crimson error shake & surge, and camera warp dive transitions.
- **Deep Obsidian Glassmorphism**: High-contrast `#06060A` aesthetic, multi-layer ambient glows, noise-grain tactile overlay, and dynamic specular cursor glare on all cards.
- **Multimodal Receipt & Invoice Scanner (Gemini Vision)**: Upload receipt photos (PNG/JPG) to extract merchant, date, total amount, category, and line items directly into the ledger.
- **Conversational AI Financial Copilot**: Interrogate your ledger with natural language ("Where did I spend the most?", "Can I afford ₹30,000?") using Gemini 2.5 Flash with deterministic offline fallbacks.
- **0–100 Financial Health Score & 50/30/20 Rule**: Radial gauge analyzing savings rate, investment habit, cash flow buffer, and expense stability against the 50/30/20 benchmark.
- **Zombie Subscription & Recurring Charge Hunter**: Automatically surfaces recurring services (Netflix, Spotify, broadband, cloud) and computes annualized outflow.
- **Strict Separation of Concerns**: Zero UI logic in `database.py`, `analytics.py`, or `ai_engine.py`; zero raw SQL in `app.py`.
- **Thread-Safe Dual Persistence**: SQLite WAL connection pool + Supabase PostgreSQL support.
- **Vectorized Analytics & Live Market Valuation**: Real-time stock (`TCS.NS`, `RELIANCE.NS`) and crypto (`BTC-USD`, `ETH-USD`) valuations via `yfinance`.
- **Privacy-First AI**: Aggregated, PII-free statistics transmitted with Gemini Structured Outputs. Deterministic fallbacks guarantee zero downtime if offline.

---

## Quickstart

### 1. Prerequisites
- Python **3.11+**
- Optional: A Google Gemini API key from Google AI Studio (`GEMINI_API_KEY`)

### 2. Install Dependencies
```bash
cd finance_tracker
python -m venv .venv

# Windows (PowerShell)
.venv\Scripts\Activate.ps1
# macOS / Linux
# source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Configure Database (Supabase PostgreSQL or Local SQLite)
Create a `.env` file inside `finance_tracker/`:

```bash
cp .env.example .env
```

- **For Supabase (PostgreSQL - Recommended)**:
  1. Open your [Supabase Dashboard](https://supabase.com/dashboard) $\rightarrow$ **Project Settings** $\rightarrow$ **Database** $\rightarrow$ **Connection string** $\rightarrow$ **URI**.
  2. Select **Connection Pooling** (recommended, port `6543`) or **Direct** (port `5432`).
  3. Paste it into your `finance_tracker/.env` file:
     ```env
     DATABASE_URL=postgresql://postgres.[PROJECT-REF]:[YOUR-PASSWORD]@aws-0-[REGION].pooler.supabase.com:6543/postgres?sslmode=require
     ```
  4. Once configured, every user registration and transaction syncs live to Supabase. You can view, search, and edit your tables anytime in the **Supabase Table Editor** (`users`, `transactions`, `otp_challenges`) in your browser!

- **For Local SQLite**:
  Leave `DATABASE_URL` empty. The application automatically initializes and manages `finance_tracker.db` locally.

### 4. Configure Gemini AI (Optional)
In your `finance_tracker/.env`:
```env
GEMINI_API_KEY=your_actual_gemini_api_key
```

### 5. Launch the Dashboard
```bash
streamlit run app.py
```

Once open in your browser:
1. Sign up or sign in using an **Email address** or **Phone number** (Password or 6-digit OTP mode), or click **Launch Instant Demo Account**.
2. Click **Load Sample Dataset** on the welcome banner (or in the sidebar **Demo Data & Workspace** expander) to populate multi-month Income, Expense, and Investment (`NIFTY50`, `TCS`, `BTC`) records.
3. Or upload your own bank CSV export via **Import Bank Statement (CSV)** — missing categories are automatically tagged via `categorize_transaction(...)`.
4. Explore the interactive **3D Spending Cube** in **Spending Breakdown** and click **Analyze Portfolio** inside **AI Financial Insights & Anomaly Detection** for on-demand recommendations.

---

## 🚀 Deploying Live to Streamlit Community Cloud

### Step 1: Push Repository to GitHub
1. Create a new repository on [GitHub](https://github.com/new) (e.g. `ai-finance-tracker`).
2. Link your local repo and push:
   ```bash
   git remote add origin https://github.com/<your-username>/ai-finance-tracker.git
   git branch -M main
   git push -u origin main
   ```

### Step 2: Deploy on Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io/) and sign in with GitHub.
2. Click **New app**.
3. Select your repository: `<your-username>/ai-finance-tracker`.
4. Branch: `main`.
5. Main file path: `app.py`.
6. Click **Advanced settings** $\rightarrow$ **Secrets**, and paste your environment variables:
   ```toml
   DATABASE_URL = "postgresql://postgres.[PROJECT-REF]:[PASSWORD]@aws-0-[REGION].pooler.supabase.com:6543/postgres?sslmode=require"
   GEMINI_API_KEY = "your_gemini_api_key_here"
   GEMINI_MODEL = "gemini-2.5-flash"
   ```
7. Click **Deploy!** Your high-end 3D AI personal finance app is immediately live with HTTPS!

---

## Running Automated Tests
```bash
cd finance_tracker
python -m unittest discover -s tests -v
```
