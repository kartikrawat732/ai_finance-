"""Persistence layer with PostgreSQL (Supabase) and SQLite connection pooling, migrations, and CRUD."""

from __future__ import annotations

import datetime as dt
import logging
import math
import os
import queue
import re
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional, Sequence, Union

import numpy as np
import pandas as pd
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Attempt importing psycopg and psycopg_pool for PostgreSQL / Supabase
try:
    import psycopg
    from psycopg.rows import dict_row
    from psycopg_pool import ConnectionPool as PsycopgConnectionPool
    HAS_PSYCOPG = True
    PSYCOPG_INTEGRITY_ERRORS: tuple[type[Exception], ...] = (psycopg.IntegrityError,)
except ImportError:
    HAS_PSYCOPG = False
    PSYCOPG_INTEGRITY_ERRORS = ()

_ENV_LOADED = False


def _ensure_env_loaded() -> None:
    """Ensure environment variables are loaded from finance_tracker/.env or root .env."""
    global _ENV_LOADED
    if not _ENV_LOADED:
        base_dir = Path(__file__).resolve().parent
        for path in (base_dir / ".env", base_dir.parent / ".env"):
            if path.is_file():
                load_dotenv(path, override=False)
        try:
            import streamlit as st
            if hasattr(st, "secrets"):
                for k, v in st.secrets.items():
                    if isinstance(v, str) and k not in os.environ:
                        os.environ[k] = v
        except Exception:
            pass
        _ENV_LOADED = True


DEFAULT_DB_PATH: Path = Path(__file__).resolve().parent / "finance_tracker.db"
ALLOWED_TRANSACTION_TYPES: tuple[str, ...] = ("Income", "Expense", "Investment")

TRANSACTION_COLUMNS: list[str] = [
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

MIGRATIONS: list[tuple[int, str, tuple[str, ...]]] = [
    (
        1,
        "initial_transactions_schema_and_indexes",
        (
            """
            CREATE TABLE IF NOT EXISTS transactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                type TEXT NOT NULL CHECK(type IN ('Income', 'Expense', 'Investment')),
                category TEXT NOT NULL,
                amount REAL NOT NULL CHECK(amount > 0),
                description TEXT,
                asset_ticker TEXT,
                quantity REAL DEFAULT 0.0,
                unit_price REAL DEFAULT 0.0,
                is_ai_tagged INTEGER DEFAULT 0
            );
            """,
            "CREATE INDEX IF NOT EXISTS idx_transactions_date ON transactions(date);",
            "CREATE INDEX IF NOT EXISTS idx_transactions_type_date ON transactions(type, date);",
            "CREATE INDEX IF NOT EXISTS idx_transactions_category ON transactions(category);",
            "CREATE INDEX IF NOT EXISTS idx_transactions_ticker ON transactions(asset_ticker);",
        ),
    ),
    (
        2,
        "composite_category_date_index",
        (
            "CREATE INDEX IF NOT EXISTS idx_transactions_cat_date ON transactions(category, date);",
        ),
    ),
    (
        3,
        "users_otp_and_user_scoped_transactions",
        (
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                full_name TEXT NOT NULL,
                email TEXT UNIQUE,
                phone TEXT UNIQUE,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            """,
            "CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);",
            "CREATE INDEX IF NOT EXISTS idx_users_phone ON users(phone);",
            """
            CREATE TABLE IF NOT EXISTS otp_challenges (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                identifier TEXT NOT NULL,
                otp_hash TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                consumed INTEGER DEFAULT 0,
                created_at TEXT NOT NULL
            );
            """,
            "CREATE INDEX IF NOT EXISTS idx_otp_identifier ON otp_challenges(identifier);",
        ),
    ),
]

POSTGRES_MIGRATIONS: list[tuple[int, str, tuple[str, ...]]] = [
    (
        1,
        "initial_transactions_schema_and_indexes",
        (
            """
            CREATE TABLE IF NOT EXISTS transactions (
                id SERIAL PRIMARY KEY,
                date VARCHAR(32) NOT NULL,
                type VARCHAR(32) NOT NULL CHECK(type IN ('Income', 'Expense', 'Investment')),
                category VARCHAR(128) NOT NULL,
                amount DOUBLE PRECISION NOT NULL CHECK(amount > 0),
                description TEXT,
                asset_ticker VARCHAR(64),
                quantity DOUBLE PRECISION DEFAULT 0.0,
                unit_price DOUBLE PRECISION DEFAULT 0.0,
                is_ai_tagged INTEGER DEFAULT 0
            );
            """,
            "CREATE INDEX IF NOT EXISTS idx_transactions_date ON transactions(date);",
            "CREATE INDEX IF NOT EXISTS idx_transactions_type_date ON transactions(type, date);",
            "CREATE INDEX IF NOT EXISTS idx_transactions_category ON transactions(category);",
            "CREATE INDEX IF NOT EXISTS idx_transactions_ticker ON transactions(asset_ticker);",
        ),
    ),
    (
        2,
        "composite_category_date_index",
        (
            "CREATE INDEX IF NOT EXISTS idx_transactions_cat_date ON transactions(category, date);",
        ),
    ),
    (
        3,
        "users_otp_and_user_scoped_transactions",
        (
            """
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                full_name VARCHAR(255) NOT NULL,
                email VARCHAR(255) UNIQUE,
                phone VARCHAR(64) UNIQUE,
                password_hash VARCHAR(255) NOT NULL,
                created_at VARCHAR(64) NOT NULL
            );
            """,
            "CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);",
            "CREATE INDEX IF NOT EXISTS idx_users_phone ON users(phone);",
            """
            CREATE TABLE IF NOT EXISTS otp_challenges (
                id SERIAL PRIMARY KEY,
                identifier VARCHAR(255) NOT NULL,
                otp_hash VARCHAR(255) NOT NULL,
                expires_at VARCHAR(64) NOT NULL,
                consumed INTEGER DEFAULT 0,
                created_at VARCHAR(64) NOT NULL
            );
            """,
            "CREATE INDEX IF NOT EXISTS idx_otp_identifier ON otp_challenges(identifier);",
            "ALTER TABLE transactions ADD COLUMN IF NOT EXISTS user_id INTEGER REFERENCES users(id) ON DELETE CASCADE;",
            "CREATE INDEX IF NOT EXISTS idx_transactions_user_date ON transactions(user_id, date);",
        ),
    ),
    (
        4,
        "enable_supabase_row_level_security",
        (
            "ALTER TABLE users ENABLE ROW LEVEL SECURITY;",
            "ALTER TABLE transactions ENABLE ROW LEVEL SECURITY;",
            "ALTER TABLE otp_challenges ENABLE ROW LEVEL SECURITY;",
            "DROP POLICY IF EXISTS allow_all_backend_users ON users;",
            "CREATE POLICY allow_all_backend_users ON users FOR ALL USING (true) WITH CHECK (true);",
            "DROP POLICY IF EXISTS allow_all_backend_tx ON transactions;",
            "CREATE POLICY allow_all_backend_tx ON transactions FOR ALL USING (true) WITH CHECK (true);",
            "DROP POLICY IF EXISTS allow_all_backend_otp ON otp_challenges;",
            "CREATE POLICY allow_all_backend_otp ON otp_challenges FOR ALL USING (true) WITH CHECK (true);",
        ),
    ),
]


class SQLiteConnectionPool:
    """Thread-safe SQLite connection pool with serialized write locking and WAL support."""

    def __init__(self, db_path: Path, pool_size: int = 8) -> None:
        self.db_path = db_path
        self.pool_size = max(1, pool_size)
        self._pool: queue.LifoQueue[sqlite3.Connection] = queue.LifoQueue(maxsize=self.pool_size)
        self._write_lock = threading.RLock()
        self._closed = False
        self._state_lock = threading.Lock()

    def _create_connection(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(
            str(self.db_path),
            timeout=30.0,
            check_same_thread=False,
            isolation_level=None,
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        conn.execute("PRAGMA busy_timeout=5000;")
        return conn

    def _checkout(self) -> sqlite3.Connection:
        with self._state_lock:
            if self._closed:
                raise RuntimeError("SQLiteConnectionPool is closed.")
        try:
            conn = self._pool.get_nowait()
            try:
                conn.execute("SELECT 1;")
                return conn
            except sqlite3.Error:
                try:
                    conn.close()
                except sqlite3.Error:
                    pass
        except queue.Empty:
            pass
        return self._create_connection()

    def _checkin(self, conn: sqlite3.Connection) -> None:
        with self._state_lock:
            if self._closed:
                try:
                    conn.close()
                except sqlite3.Error:
                    pass
                return
        try:
            self._pool.put_nowait(conn)
        except queue.Full:
            try:
                conn.close()
            except sqlite3.Error:
                pass

    @contextmanager
    def connection(self, write: bool = False) -> Iterator[sqlite3.Connection]:
        """Yield a pooled SQLite connection; write mode serializes commits cleanly."""
        if write:
            self._write_lock.acquire()
        conn: Optional[sqlite3.Connection] = None
        broken = False
        try:
            conn = self._checkout()
            if write:
                conn.execute("BEGIN IMMEDIATE;")
            yield conn
            if write and conn.in_transaction:
                conn.commit()
        except Exception:
            if conn is not None and conn.in_transaction:
                try:
                    conn.rollback()
                except sqlite3.Error:
                    broken = True
            raise
        finally:
            if conn is not None:
                if broken:
                    try:
                        conn.close()
                    except sqlite3.Error:
                        pass
                else:
                    self._checkin(conn)
            if write:
                self._write_lock.release()

    def close_all(self) -> None:
        """Close all pooled connections cleanly."""
        with self._state_lock:
            self._closed = True
        while True:
            try:
                conn = self._pool.get_nowait()
                try:
                    conn.close()
                except sqlite3.Error:
                    pass
            except queue.Empty:
                break


class PostgresCursorProxy:
    """Adapter wrapping a psycopg cursor to mirror SQLite row semantics and lastrowid."""

    def __init__(self, cursor: Any, lastrowid: Optional[int] = None) -> None:
        self._cursor = cursor
        self.lastrowid = lastrowid

    def fetchone(self) -> Any:
        return self._cursor.fetchone()

    def fetchall(self) -> list[Any]:
        return self._cursor.fetchall()

    @property
    def rowcount(self) -> int:
        return self._cursor.rowcount

    def __iter__(self) -> Any:
        return iter(self._cursor)


class PostgresConnectionProxy:
    """Transparent connection proxy converting SQLite '?' placeholders and tracking lastrowid."""

    def __init__(self, conn: Any) -> None:
        self._conn = conn

    def execute(self, sql: str, params: Any = None) -> PostgresCursorProxy:
        adapted_sql = sql.replace("?", "%s")
        upper = adapted_sql.strip().upper()
        auto_returning = False
        if upper.startswith("INSERT INTO") and "RETURNING" not in upper:
            if "TRANSACTIONS" in upper or "USERS" in upper:
                stripped = adapted_sql.rstrip().rstrip(";")
                adapted_sql = f"{stripped} RETURNING id;"
                auto_returning = True

        if params is not None:
            clean_params = tuple(params) if isinstance(params, (list, tuple)) else params
            cur = self._conn.execute(adapted_sql, clean_params)
        else:
            cur = self._conn.execute(adapted_sql)

        lastrowid = None
        if auto_returning:
            row = cur.fetchone()
            if row is not None:
                lastrowid = int(row["id"] if isinstance(row, dict) else row[0])

        return PostgresCursorProxy(cur, lastrowid=lastrowid)

    def executemany(self, sql: str, seq_of_params: Sequence[Any]) -> PostgresCursorProxy:
        adapted_sql = sql.replace("?", "%s")
        cur = self._conn.cursor()
        cur.executemany(adapted_sql, seq_of_params)
        return PostgresCursorProxy(cur)

    def commit(self) -> None:
        self._conn.commit()

    def rollback(self) -> None:
        self._conn.rollback()


class PostgresConnectionPool:
    """Thread-safe PostgreSQL connection pool with SSL support for Supabase."""

    def __init__(self, conninfo: str, min_size: int = 1, max_size: int = 10) -> None:
        if not HAS_PSYCOPG:
            raise RuntimeError(
                "psycopg is not installed. Run `pip install 'psycopg[binary]' psycopg_pool` to use PostgreSQL / Supabase."
            )
        self.conninfo = self._normalize_conninfo(conninfo)
        self.min_size = max(1, min_size)
        self.max_size = max(self.min_size, max_size)
        self._pool = PsycopgConnectionPool(
            conninfo=self.conninfo,
            min_size=self.min_size,
            max_size=self.max_size,
            open=False,
            kwargs={"row_factory": dict_row, "autocommit": False, "connect_timeout": 8},
        )
        self._pool.open(wait=True, timeout=10.0)
        self._closed = False
        self._state_lock = threading.Lock()

    @staticmethod
    def _normalize_conninfo(url: str) -> str:
        clean = url.strip()
        if (clean.startswith('"') and clean.endswith('"')) or (clean.startswith("'") and clean.endswith("'")):
            clean = clean[1:-1].strip()
        if clean.startswith("postgres://"):
            clean = "postgresql://" + clean[len("postgres://"):]

        # Supabase direct endpoints (db.<ref>.supabase.co) only resolve to IPv6 addresses.
        # Streamlit Community Cloud and IPv4-only hosts cannot route IPv6 packets, causing PoolTimeout.
        # Auto-rewrite direct Supabase endpoints to the Supavisor IPv4 connection pooler.
        supabase_match = re.match(
            r"^(postgres(?:ql)?://)([^:]+):([^@]+)@db\.([a-z0-9_-]+)\.supabase\.co(?::\d+)?(/[^?]+)?(\?.*)?$",
            clean,
        )
        if supabase_match:
            scheme, user, pwd, project_ref, path, query = supabase_match.groups()
            path = path or "/postgres"
            query = query or "?sslmode=require"
            region = os.getenv("SUPABASE_REGION")
            if not region and project_ref == "brjqlccckhqykhgsyfzr":
                region = "ap-northeast-1"
            if region:
                pooler_user = f"{user}.{project_ref}" if f".{project_ref}" not in user else user
                clean = f"{scheme}{pooler_user}:{pwd}@aws-0-{region}.pooler.supabase.com:5432{path}{query}"

        if ("supabase.co" in clean.lower() or "pooler.supabase.com" in clean.lower()) and "sslmode" not in clean.lower():
            separator = "&" if "?" in clean else "?"
            clean = f"{clean}{separator}sslmode=require"
        return clean

    @contextmanager
    def connection(self, write: bool = False) -> Iterator[PostgresConnectionProxy]:
        with self._pool.connection() as raw_conn:
            proxy = PostgresConnectionProxy(raw_conn)
            try:
                yield proxy
                if write:
                    raw_conn.commit()
            except Exception:
                if write:
                    try:
                        raw_conn.rollback()
                    except Exception:
                        pass
                raise

    def close_all(self) -> None:
        with self._state_lock:
            if not self._closed:
                self._closed = True
                try:
                    self._pool.close()
                except Exception:
                    pass


_SQLITE_POOL_REGISTRY: dict[Path, SQLiteConnectionPool] = {}
_POSTGRES_POOL_REGISTRY: dict[str, PostgresConnectionPool] = {}
_REGISTRY_LOCK = threading.Lock()
_POSTGRES_FAILED: bool = False
_LAST_POSTGRES_ERROR: Optional[str] = None


def _resolve_database_url(db_path: Optional[Union[str, Path]] = None) -> Optional[str]:
    """Dynamically resolve the PostgreSQL / Supabase connection URL.

    Checks:
    1. Direct `db_path` parameter:
       - if postgresql:// or postgres:// -> returns URL string
       - if non-None local file path -> returns None (explicit SQLite requested)
    2. `os.environ` (DATABASE_URL, SUPABASE_DB_URL, POSTGRES_URL)
    3. `st.secrets` (DATABASE_URL, SUPABASE_DB_URL, POSTGRES_URL)
    """
    if db_path is not None:
        s = str(db_path).strip()
        if s.startswith(("postgresql://", "postgres://")):
            return PostgresConnectionPool._normalize_conninfo(s)
        return None

    _ensure_env_loaded()

    # 1. Environment variables
    for k in ("DATABASE_URL", "SUPABASE_DB_URL", "POSTGRES_URL"):
        val = os.getenv(k)
        if val:
            val = val.strip()
            if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
                val = val[1:-1].strip()
            if val.startswith(("postgresql://", "postgres://")):
                norm_val = PostgresConnectionPool._normalize_conninfo(val)
                os.environ[k] = norm_val
                return norm_val

    # 2. Streamlit Cloud Secrets (st.secrets) - always evaluated dynamically
    try:
        import streamlit as st
        if hasattr(st, "secrets"):
            if "SUPABASE_REGION" in st.secrets:
                os.environ["SUPABASE_REGION"] = str(st.secrets["SUPABASE_REGION"]).strip()
            for k in ("DATABASE_URL", "SUPABASE_DB_URL", "POSTGRES_URL"):
                if k in st.secrets:
                    raw_val = st.secrets[k]
                    if raw_val:
                        val = str(raw_val).strip()
                        if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
                            val = val[1:-1].strip()
                        if val.startswith(("postgresql://", "postgres://")):
                            norm_val = PostgresConnectionPool._normalize_conninfo(val)
                            os.environ[k] = norm_val
                            return norm_val
    except Exception:
        pass

    return None


def get_database_backend(db_path: Optional[Union[str, Path]] = None) -> str:
    """Return 'postgresql' or 'sqlite' based on parameters and environment variables."""
    global _POSTGRES_FAILED
    if _POSTGRES_FAILED:
        return "sqlite"
    url = _resolve_database_url(db_path)
    return "postgresql" if url else "sqlite"


def _mask_db_url(url: str) -> str:
    """Mask password from database URL for safe display in UI."""
    if not url:
        return ""
    try:
        import urllib.parse
        parsed = urllib.parse.urlsplit(url)
        netloc = parsed.netloc
        if "@" in netloc:
            auth, host_port = netloc.split("@", 1)
            user = auth.split(":", 1)[0]
            masked_netloc = f"{user}:***@{host_port}"
        else:
            masked_netloc = netloc
        return f"{parsed.scheme}://{masked_netloc}{parsed.path}"
    except Exception:
        return "postgresql://***"


def get_db_target_info(db_path: Optional[Union[str, Path]] = None) -> dict[str, Any]:
    """Return metadata about active database engine for UI and logging."""
    global _LAST_POSTGRES_ERROR
    if _LAST_POSTGRES_ERROR:
        resolved = resolve_db_path(db_path)
        return {
            "backend": "sqlite",
            "provider": "SQLite (Supabase Connection Timeout)",
            "display": "⚠️ SQLite Fallback (Supabase Unreachable)",
            "target": str(resolved.name),
            "is_cloud": False,
            "error": _LAST_POSTGRES_ERROR,
        }
    url = _resolve_database_url(db_path)
    if url:
        is_supabase = "supabase" in url.lower()
        provider = "Supabase (PostgreSQL)" if is_supabase else "PostgreSQL"
        masked = _mask_db_url(url)
        return {
            "backend": "postgresql",
            "provider": provider,
            "display": f"🟢 {provider}",
            "target": masked,
            "is_cloud": True,
        }
    resolved = resolve_db_path(db_path)
    return {
        "backend": "sqlite",
        "provider": "SQLite",
        "display": "📁 SQLite (Local)",
        "target": str(resolved.name),
        "is_cloud": False,
    }


def resolve_db_path(db_path: Optional[Union[str, Path]] = None) -> Path:
    """Resolve the SQLite database path from argument, environment variable, or default."""
    if db_path is not None:
        return Path(db_path).expanduser().resolve()
    env_path = os.getenv("FINANCE_DB_PATH")
    if env_path:
        return Path(env_path).expanduser().resolve()
    return DEFAULT_DB_PATH.resolve()


def _get_pool(
    db_path: Optional[Union[str, Path]] = None,
) -> Union[SQLiteConnectionPool, PostgresConnectionPool]:
    global _POSTGRES_FAILED, _LAST_POSTGRES_ERROR
    url = _resolve_database_url(db_path)
    with _REGISTRY_LOCK:
        if url and not _POSTGRES_FAILED:
            pool = _POSTGRES_POOL_REGISTRY.get(url)
            if pool is None or pool._closed:
                try:
                    pool = PostgresConnectionPool(url)
                    _POSTGRES_POOL_REGISTRY[url] = pool
                    _POSTGRES_FAILED = False
                    _LAST_POSTGRES_ERROR = None
                except Exception as exc:
                    import logging
                    logging.getLogger(__name__).warning(
                        f"PostgreSQL connection to Supabase failed: {exc}. Falling back to SQLite."
                    )
                    _POSTGRES_FAILED = True
                    _LAST_POSTGRES_ERROR = str(exc)
                    resolved = resolve_db_path(None)
                    sq_pool = _SQLITE_POOL_REGISTRY.get(resolved)
                    if sq_pool is None or sq_pool._closed:
                        sq_pool = SQLiteConnectionPool(resolved)
                        _SQLITE_POOL_REGISTRY[resolved] = sq_pool
                    return sq_pool
            return pool

        resolved = resolve_db_path(db_path)
        sq_pool = _SQLITE_POOL_REGISTRY.get(resolved)
        if sq_pool is None or sq_pool._closed:
            sq_pool = SQLiteConnectionPool(resolved)
            _SQLITE_POOL_REGISTRY[resolved] = sq_pool
        return sq_pool


def close_pools(db_path: Optional[Union[str, Path]] = None) -> None:
    """Close pooled connections for a specific path/URL or all tracked pools."""
    with _REGISTRY_LOCK:
        if db_path is not None:
            backend = get_database_backend(db_path)
            if backend == "postgresql":
                url = str(db_path).strip()
                p = _POSTGRES_POOL_REGISTRY.pop(url, None)
                if p:
                    p.close_all()
            else:
                resolved = resolve_db_path(db_path)
                sp = _SQLITE_POOL_REGISTRY.pop(resolved, None)
                if sp:
                    sp.close_all()
            return
        for p in _POSTGRES_POOL_REGISTRY.values():
            p.close_all()
        _POSTGRES_POOL_REGISTRY.clear()
        for sp in _SQLITE_POOL_REGISTRY.values():
            sp.close_all()
        _SQLITE_POOL_REGISTRY.clear()


def _normalize_bool_flag(value: Any) -> int:
    """Convert numeric, string, or boolean inputs into an integer 0/1 flag."""
    if value is None:
        return 0
    try:
        if pd.isna(value):
            return 0
    except (TypeError, ValueError):
        pass
    if isinstance(value, bool):
        return 1 if value else 0
    if isinstance(value, str):
        cleaned = value.strip().lower()
        if cleaned in {"", "0", "false", "no", "n", "off"}:
            return 0
        if cleaned in {"1", "true", "yes", "y", "on"}:
            return 1
        try:
            return 1 if float(cleaned) != 0 else 0
        except ValueError:
            return 0
    try:
        return 1 if int(float(value)) != 0 else 0
    except (TypeError, ValueError):
        return 1 if bool(value) else 0


def _normalize_iso_date(value: Any) -> str:
    """Validate and normalize a date-like value to ISO-8601 YYYY-MM-DD."""
    if value is None:
        raise ValueError("Transaction date cannot be empty.")
    try:
        if pd.isna(value):
            raise ValueError("Transaction date cannot be empty.")
    except (TypeError, ValueError) as exc:
        if isinstance(exc, ValueError) and "cannot be empty" in str(exc):
            raise
    if isinstance(value, dt.datetime):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, dt.date):
        return value.strftime("%Y-%m-%d")

    text = str(value).strip()
    if not text:
        raise ValueError("Transaction date cannot be empty.")
    if len(text) == 10:
        try:
            return dt.date.fromisoformat(text).isoformat()
        except ValueError:
            pass

    parsed_ts = pd.to_datetime(text, errors="coerce")
    if pd.isna(parsed_ts):
        raise ValueError(f"Invalid date format '{value}'. Expected YYYY-MM-DD.")
    return parsed_ts.strftime("%Y-%m-%d")


def _validate_and_normalize_record(
    tx_date: Any,
    trans_type: Any,
    category: Any,
    amount: Any,
    description: Optional[Any] = None,
    asset_ticker: Optional[Any] = None,
    quantity: Any = 0.0,
    unit_price: Any = 0.0,
    is_ai_tagged: Any = 0,
) -> tuple[str, str, str, float, Optional[str], Optional[str], float, float, int]:
    """Validate and normalize a single transaction tuple for SQLite persistence."""
    norm_date = _normalize_iso_date(tx_date)

    norm_type = str(trans_type).strip() if trans_type is not None else ""
    if norm_type not in ALLOWED_TRANSACTION_TYPES:
        raise ValueError(
            f"Invalid transaction type '{trans_type}'. Allowed: {', '.join(ALLOWED_TRANSACTION_TYPES)}."
        )

    norm_category = str(category).strip() if category is not None else ""
    if not norm_category or norm_category.lower() in {"nan", "none", "<na>"}:
        raise ValueError("Transaction category cannot be empty.")

    try:
        norm_amount = float(amount)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid amount '{amount}'.") from exc
    if not math.isfinite(norm_amount) or norm_amount <= 0:
        raise ValueError(f"Transaction amount must be greater than 0 (got {amount}).")

    if description is None or (isinstance(description, float) and math.isnan(description)):
        norm_description: Optional[str] = None
    else:
        desc_str = str(description).strip()
        norm_description = desc_str if desc_str and desc_str.lower() not in {"nan", "none", "<na>"} else None

    try:
        norm_quantity = 0.0 if quantity is None or pd.isna(quantity) else float(quantity)
        norm_unit_price = 0.0 if unit_price is None or pd.isna(unit_price) else float(unit_price)
    except (TypeError, ValueError) as exc:
        raise ValueError("Quantity and unit price must be valid numbers.") from exc

    if not math.isfinite(norm_quantity) or norm_quantity < 0:
        raise ValueError(f"Quantity cannot be negative (got {quantity}).")
    if not math.isfinite(norm_unit_price) or norm_unit_price < 0:
        raise ValueError(f"Unit price cannot be negative (got {unit_price}).")

    if norm_type == "Investment":
        if asset_ticker is None or (isinstance(asset_ticker, float) and math.isnan(asset_ticker)):
            norm_ticker: Optional[str] = None
        else:
            ticker_str = str(asset_ticker).strip().upper()
            norm_ticker = (
                ticker_str
                if ticker_str and ticker_str not in {"NAN", "NONE", "NULL", "<NA>"}
                else None
            )
        if norm_quantity > 0 and norm_unit_price == 0.0:
            norm_unit_price = round(norm_amount / norm_quantity, 6)
    else:
        norm_ticker = None
        norm_quantity = 0.0
        norm_unit_price = 0.0

    norm_ai_flag = _normalize_bool_flag(is_ai_tagged)

    return (
        norm_date,
        norm_type,
        norm_category,
        round(norm_amount, 6),
        norm_description,
        norm_ticker,
        round(norm_quantity, 8),
        round(norm_unit_price, 6),
        norm_ai_flag,
    )


def init_db(db_path: Optional[Union[str, Path]] = None) -> None:
    """Idempotently initialize SQLite or PostgreSQL (Supabase) tables, indexes, and schema migrations."""
    pool = _get_pool(db_path)
    backend = get_database_backend(db_path)

    if backend == "postgresql":
        with pool.connection(write=True) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version INTEGER PRIMARY KEY,
                    name VARCHAR(255) NOT NULL,
                    applied_at VARCHAR(64) NOT NULL
                );
                """
            )
            applied_rows = conn.execute("SELECT version FROM schema_migrations;").fetchall()
            applied_versions = {int(row["version"]) for row in applied_rows}

            for version, name, statements in POSTGRES_MIGRATIONS:
                if version in applied_versions:
                    continue
                for stmt in statements:
                    conn.execute(stmt)
                conn.execute(
                    """
                    INSERT INTO schema_migrations (version, name, applied_at)
                    VALUES (?, ?, ?)
                    ON CONFLICT (version) DO NOTHING;
                    """,
                    (version, name, dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")),
                )
        return

    with pool.connection(write=True) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                applied_at TEXT NOT NULL
            );
            """
        )
        applied_rows = conn.execute("SELECT version FROM schema_migrations;").fetchall()
        applied_versions = {int(row["version"]) for row in applied_rows}

        for version, name, statements in MIGRATIONS:
            if version in applied_versions:
                continue
            for stmt in statements:
                conn.execute(stmt)
            if version == 3:
                cols = {
                    str(r["name"])
                    for r in conn.execute("PRAGMA table_info(transactions);").fetchall()
                }
                if "user_id" not in cols:
                    conn.execute("ALTER TABLE transactions ADD COLUMN user_id INTEGER;")
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_transactions_user_date ON transactions(user_id, date);"
                )
            conn.execute(
                "INSERT OR IGNORE INTO schema_migrations (version, name, applied_at) VALUES (?, ?, ?);",
                (version, name, dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")),
            )


def add_transaction(
    date: Union[str, dt.date, dt.datetime],
    trans_type: Optional[str] = None,
    category: str = "",
    amount: float = 0.0,
    description: Optional[str] = None,
    asset_ticker: Optional[str] = None,
    quantity: float = 0.0,
    unit_price: float = 0.0,
    is_ai_tagged: Union[bool, int] = 0,
    *,
    type: Optional[str] = None,
    user_id: Optional[int] = None,
    db_path: Optional[Union[str, Path]] = None,
) -> int:
    """Validate and insert a single transaction record, returning its row ID."""
    init_db(db_path)
    resolved_type = type if type is not None else trans_type
    record = _validate_and_normalize_record(
        tx_date=date,
        trans_type=resolved_type,
        category=category,
        amount=amount,
        description=description,
        asset_ticker=asset_ticker,
        quantity=quantity,
        unit_price=unit_price,
        is_ai_tagged=is_ai_tagged,
    )
    resolved_uid = int(user_id) if user_id is not None else None
    pool = _get_pool(db_path)
    with pool.connection(write=True) as conn:
        cursor = conn.execute(
            """
            INSERT INTO transactions (
                date, type, category, amount, description,
                asset_ticker, quantity, unit_price, is_ai_tagged, user_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (*record, resolved_uid),
        )
        return int(cursor.lastrowid or 0)


def bulk_insert_transactions(
    df: pd.DataFrame,
    db_path: Optional[Union[str, Path]] = None,
    *,
    user_id: Optional[int] = None,
) -> int:
    """High-performance vectorized validation and atomic batch insertion from a DataFrame."""
    init_db(db_path)
    if df is None or df.empty:
        return 0

    work = df.copy()
    if "type" not in work.columns and "trans_type" in work.columns:
        work["type"] = work["trans_type"]

    required_cols = {"date", "type", "category", "amount"}
    missing = required_cols - set(work.columns)
    if missing:
        raise ValueError(f"Missing required columns for bulk insert: {sorted(missing)}")

    parsed_dates = pd.to_datetime(work["date"], errors="coerce")
    if parsed_dates.isna().any():
        bad_idx = work.index[parsed_dates.isna()].tolist()[:5]
        raise ValueError(f"Invalid date values found at rows: {bad_idx}")
    work["date"] = parsed_dates.dt.strftime("%Y-%m-%d")

    work["type"] = work["type"].fillna("").astype(str).str.strip()
    invalid_types = ~work["type"].isin(ALLOWED_TRANSACTION_TYPES)
    if invalid_types.any():
        bad_vals = work.loc[invalid_types, "type"].unique().tolist()[:5]
        raise ValueError(
            f"Invalid transaction type(s) {bad_vals}. Allowed: {', '.join(ALLOWED_TRANSACTION_TYPES)}."
        )

    work["category"] = work["category"].fillna("").astype(str).str.strip()
    invalid_cat = (work["category"] == "") | work["category"].str.lower().isin({"nan", "none", "<na>"})
    if invalid_cat.any():
        raise ValueError("Transaction category cannot be empty during bulk insert.")

    work["amount"] = pd.to_numeric(work["amount"], errors="coerce")
    invalid_amt = work["amount"].isna() | ~np.isfinite(work["amount"]) | (work["amount"] <= 0)
    if invalid_amt.any():
        bad_idx = work.index[invalid_amt].tolist()[:5]
        raise ValueError(f"Transaction amounts must be finite numbers > 0 (rows: {bad_idx}).")
    work["amount"] = work["amount"].round(6)

    if "description" in work.columns:
        # pyrefly: ignore[bad-argument-type]
        desc_series = work["description"].where(work["description"].notna(), None)
        cleaned_desc = desc_series.map(
            lambda x: None
            if x is None or str(x).strip().lower() in {"", "nan", "none", "<na>"}
            else str(x).strip()
        )
        work["description"] = cleaned_desc
    else:
        work["description"] = None

    is_inv = work["type"] == "Investment"

    if "asset_ticker" in work.columns:
        # pyrefly: ignore[bad-argument-type]
        ticker_series = work["asset_ticker"].where(is_inv & work["asset_ticker"].notna(), None)
        work["asset_ticker"] = ticker_series.map(
            lambda x: None
            if x is None or str(x).strip().upper() in {"", "NAN", "NONE", "NULL", "<NA>"}
            else str(x).strip().upper()
        )
    else:
        work["asset_ticker"] = None

    if "quantity" in work.columns:
        qty = pd.to_numeric(work["quantity"], errors="coerce").fillna(0.0)
        if ((qty < 0) | ~np.isfinite(qty)).any():
            raise ValueError("Quantity values must be finite and >= 0.")
        work["quantity"] = np.where(is_inv, np.round(qty, 8), 0.0)
    else:
        work["quantity"] = 0.0

    if "unit_price" in work.columns:
        price = pd.to_numeric(work["unit_price"], errors="coerce").fillna(0.0)
        if ((price < 0) | ~np.isfinite(price)).any():
            raise ValueError("Unit price values must be finite and >= 0.")
        work["unit_price"] = np.where(is_inv, np.round(price, 6), 0.0)
    else:
        work["unit_price"] = 0.0

    needs_price = is_inv & (work["quantity"] > 0) & (work["unit_price"] == 0.0)
    safe_qty = np.where(work["quantity"] > 0, work["quantity"], 1.0)
    work["unit_price"] = np.where(
        needs_price,
        np.round(work["amount"] / safe_qty, 6),
        work["unit_price"],
    )

    if "is_ai_tagged" in work.columns:
        work["is_ai_tagged"] = work["is_ai_tagged"].map(_normalize_bool_flag).astype("int64")
    else:
        work["is_ai_tagged"] = 0

    resolved_uid = int(user_id) if user_id is not None else None
    records = list(
        zip(
            work["date"].astype(str).tolist(),
            work["type"].astype(str).tolist(),
            work["category"].astype(str).tolist(),
            [float(v) for v in work["amount"].tolist()],
            work["description"].tolist(),
            work["asset_ticker"].tolist(),
            [float(v) for v in work["quantity"].tolist()],
            [float(v) for v in work["unit_price"].tolist()],
            [int(v) for v in work["is_ai_tagged"].tolist()],
            [resolved_uid] * len(work),
        )
    )

    pool = _get_pool(db_path)
    with pool.connection(write=True) as conn:
        conn.executemany(
            """
            INSERT INTO transactions (
                date, type, category, amount, description,
                asset_ticker, quantity, unit_price, is_ai_tagged, user_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            records,
        )
    return len(records)


def _normalize_filter_values(value: Optional[Union[str, Sequence[str]]]) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned or cleaned.lower() == "all":
            return []
        return [cleaned]
    return [str(v).strip() for v in value if str(v).strip() and str(v).strip().lower() != "all"]


def _empty_transactions_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": pd.Series(dtype="int64"),
            "date": pd.Series(dtype="object"),
            "type": pd.Series(dtype="object"),
            "category": pd.Series(dtype="object"),
            "amount": pd.Series(dtype="float64"),
            "description": pd.Series(dtype="object"),
            "asset_ticker": pd.Series(dtype="object"),
            "quantity": pd.Series(dtype="float64"),
            "unit_price": pd.Series(dtype="float64"),
            "is_ai_tagged": pd.Series(dtype="int64"),
        }
    )


def get_transactions(
    start_date: Optional[Union[str, dt.date, dt.datetime]] = None,
    end_date: Optional[Union[str, dt.date, dt.datetime]] = None,
    category: Optional[Union[str, Sequence[str]]] = None,
    trans_type: Optional[Union[str, Sequence[str]]] = None,
    search_query: Optional[str] = None,
    ai_tagged_only: bool = False,
    user_id: Optional[int] = None,
    db_path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """Fetch filtered transaction records as a strongly typed Pandas DataFrame."""
    init_db(db_path)
    clauses: list[str] = []
    params: list[Any] = []

    if user_id is not None:
        clauses.append("user_id = ?")
        params.append(int(user_id))

    if start_date is not None:
        clauses.append("date >= ?")
        params.append(_normalize_iso_date(start_date))

    if end_date is not None:
        clauses.append("date <= ?")
        params.append(_normalize_iso_date(end_date))

    categories = _normalize_filter_values(category)
    if categories:
        placeholders = ", ".join("?" for _ in categories)
        clauses.append(f"category IN ({placeholders})")
        params.extend(categories)

    types = _normalize_filter_values(trans_type)
    if types:
        placeholders = ", ".join("?" for _ in types)
        clauses.append(f"type IN ({placeholders})")
        params.extend(types)

    if search_query and search_query.strip():
        like_term = f"%{search_query.strip()}%"
        clauses.append(
            "(COALESCE(description, '') LIKE ? OR category LIKE ? OR COALESCE(asset_ticker, '') LIKE ?)"
        )
        params.extend([like_term, like_term, like_term])

    if ai_tagged_only:
        clauses.append("is_ai_tagged = 1")

    where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    sql = f"""
        SELECT
            id, date, type, category, amount, description,
            asset_ticker, quantity, unit_price, is_ai_tagged
        FROM transactions
        {where_sql}
        ORDER BY date DESC, id DESC;
    """

    pool = _get_pool(db_path)
    with pool.connection(write=False) as conn:
        rows = conn.execute(sql, params).fetchall()

    if not rows:
        return _empty_transactions_df()

    df = pd.DataFrame([dict(row) for row in rows], columns=TRANSACTION_COLUMNS)
    df["id"] = pd.to_numeric(df["id"], errors="coerce").astype("int64")
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0).astype("float64")
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce").fillna(0.0).astype("float64")
    df["unit_price"] = pd.to_numeric(df["unit_price"], errors="coerce").fillna(0.0).astype("float64")
    df["is_ai_tagged"] = pd.to_numeric(df["is_ai_tagged"], errors="coerce").fillna(0).astype("int64")
    df["description"] = df["description"].fillna("").astype(str)
    return df


def update_transaction(
    transaction_id: int,
    date: Union[str, dt.date, dt.datetime],
    trans_type: str,
    category: str,
    amount: float,
    description: Optional[str] = None,
    asset_ticker: Optional[str] = None,
    quantity: float = 0.0,
    unit_price: float = 0.0,
    is_ai_tagged: Optional[Union[bool, int]] = None,
    user_id: Optional[int] = None,
    db_path: Optional[Union[str, Path]] = None,
) -> bool:
    """Update an existing transaction row with full validation."""
    init_db(db_path)
    pool = _get_pool(db_path)
    with pool.connection(write=True) as conn:
        if user_id is not None:
            existing = conn.execute(
                "SELECT is_ai_tagged FROM transactions WHERE id = ? AND user_id = ?;",
                (int(transaction_id), int(user_id)),
            ).fetchone()
        else:
            existing = conn.execute(
                "SELECT is_ai_tagged FROM transactions WHERE id = ?;",
                (int(transaction_id),),
            ).fetchone()
        if existing is None:
            return False

        effective_ai_flag = existing["is_ai_tagged"] if is_ai_tagged is None else is_ai_tagged
        record = _validate_and_normalize_record(
            tx_date=date,
            trans_type=trans_type,
            category=category,
            amount=amount,
            description=description,
            asset_ticker=asset_ticker,
            quantity=quantity,
            unit_price=unit_price,
            is_ai_tagged=effective_ai_flag,
        )
        if user_id is not None:
            cursor = conn.execute(
                """
                UPDATE transactions
                SET date = ?, type = ?, category = ?, amount = ?, description = ?,
                    asset_ticker = ?, quantity = ?, unit_price = ?, is_ai_tagged = ?
                WHERE id = ? AND user_id = ?;
                """,
                (*record, int(transaction_id), int(user_id)),
            )
        else:
            cursor = conn.execute(
                """
                UPDATE transactions
                SET date = ?, type = ?, category = ?, amount = ?, description = ?,
                    asset_ticker = ?, quantity = ?, unit_price = ?, is_ai_tagged = ?
                WHERE id = ?;
                """,
                (*record, int(transaction_id)),
            )
        return cursor.rowcount > 0


def delete_transaction(
    transaction_id: int,
    user_id: Optional[int] = None,
    db_path: Optional[Union[str, Path]] = None,
) -> bool:
    """Delete a single transaction record by its ID."""
    init_db(db_path)
    pool = _get_pool(db_path)
    with pool.connection(write=True) as conn:
        if user_id is not None:
            cursor = conn.execute(
                "DELETE FROM transactions WHERE id = ? AND user_id = ?;",
                (int(transaction_id), int(user_id)),
            )
        else:
            cursor = conn.execute(
                "DELETE FROM transactions WHERE id = ?;",
                (int(transaction_id),),
            )
        return cursor.rowcount > 0


def delete_transactions(
    transaction_ids: Iterable[int],
    user_id: Optional[int] = None,
    db_path: Optional[Union[str, Path]] = None,
) -> int:
    """Delete multiple transactions by ID inside a single atomic transaction."""
    init_db(db_path)
    clean_ids = sorted({int(tx_id) for tx_id in transaction_ids})
    if not clean_ids:
        return 0

    placeholders = ", ".join("?" for _ in clean_ids)
    pool = _get_pool(db_path)
    with pool.connection(write=True) as conn:
        if user_id is not None:
            cursor = conn.execute(
                f"DELETE FROM transactions WHERE id IN ({placeholders}) AND user_id = ?;",
                [*clean_ids, int(user_id)],
            )
        else:
            cursor = conn.execute(
                f"DELETE FROM transactions WHERE id IN ({placeholders});",
                clean_ids,
            )
        return int(cursor.rowcount)


def get_filter_metadata(
    user_id: Optional[int] = None,
    db_path: Optional[Union[str, Path]] = None,
) -> dict[str, Any]:
    """Return available date bounds, distinct categories, and total row count."""
    init_db(db_path)
    pool = _get_pool(db_path)
    with pool.connection(write=False) as conn:
        if user_id is not None:
            bounds = conn.execute(
                "SELECT MIN(date) AS min_date, MAX(date) AS max_date, COUNT(*) AS total_count "
                "FROM transactions WHERE user_id = ?;",
                (int(user_id),),
            ).fetchone()
            cat_rows = conn.execute(
                "SELECT DISTINCT category FROM transactions WHERE user_id = ? ORDER BY category ASC;",
                (int(user_id),),
            ).fetchall()
        else:
            bounds = conn.execute(
                "SELECT MIN(date) AS min_date, MAX(date) AS max_date, COUNT(*) AS total_count FROM transactions;"
            ).fetchone()
            cat_rows = conn.execute(
                "SELECT DISTINCT category FROM transactions ORDER BY category ASC;"
            ).fetchall()

    return {
        "min_date": bounds["min_date"] if bounds else None,
        "max_date": bounds["max_date"] if bounds else None,
        "total_count": int(bounds["total_count"]) if bounds else 0,
        "categories": [str(row["category"]) for row in cat_rows],
    }


def seed_sample_data(
    reference_date: Optional[dt.date] = None,
    db_path: Optional[Union[str, Path]] = None,
    user_id: Optional[int] = None,
) -> int:
    """Insert a realistic multi-month personal finance & investment sample dataset."""
    init_db(db_path)
    anchor = reference_date or dt.date.today()
    month_starts: list[dt.date] = []
    cur_year, cur_month = anchor.year, anchor.month
    for offset in range(3, -1, -1):
        m = cur_month - offset
        y = cur_year
        while m <= 0:
            m += 12
            y -= 1
        month_starts.append(dt.date(y, m, 1))

    rows: list[dict[str, Any]] = []
    for idx, m_start in enumerate(month_starts):
        def _dt(day: int) -> str:
            candidate = m_start + dt.timedelta(days=min(day - 1, 27))
            if candidate > anchor:
                candidate = anchor
            return candidate.strftime("%Y-%m-%d")

        # Income
        rows.append(
            {
                "date": _dt(1),
                "type": "Income",
                "category": "Salary",
                "amount": 125000.0,
                "description": "Monthly Tech Payroll Credit",
                "asset_ticker": None,
                "quantity": 0.0,
                "unit_price": 0.0,
                "is_ai_tagged": 0,
            }
        )
        if idx in (1, 3):
            rows.append(
                {
                    "date": _dt(18),
                    "type": "Income",
                    "category": "Freelance & Business",
                    "amount": 18500.0 if idx == 1 else 24000.0,
                    "description": "Architecture Advisory Retainer",
                    "asset_ticker": None,
                    "quantity": 0.0,
                    "unit_price": 0.0,
                    "is_ai_tagged": 0,
                }
            )

        # Recurring Expenses
        rows.extend(
            [
                {
                    "date": _dt(3),
                    "type": "Expense",
                    "category": "Housing & Rent",
                    "amount": 28000.0,
                    "description": "Apartment Monthly Rent NEFT",
                    "asset_ticker": None,
                    "quantity": 0.0,
                    "unit_price": 0.0,
                    "is_ai_tagged": 0,
                },
                {
                    "date": _dt(6),
                    "type": "Expense",
                    "category": "Utilities",
                    "amount": 3250.0 + idx * 180.0,
                    "description": "BESCOM Power & ACT Fibernet",
                    "asset_ticker": None,
                    "quantity": 0.0,
                    "unit_price": 0.0,
                    "is_ai_tagged": 1,
                },
                {
                    "date": _dt(9),
                    "type": "Expense",
                    "category": "Food & Dining",
                    "amount": 6400.0 if idx < 3 else 14800.0,
                    "description": "Swiggy Bangalore & Dining",
                    "asset_ticker": None,
                    "quantity": 0.0,
                    "unit_price": 0.0,
                    "is_ai_tagged": 1,
                },
                {
                    "date": _dt(12),
                    "type": "Expense",
                    "category": "Transportation",
                    "amount": 2900.0 + idx * 250.0,
                    "description": "Uber Trip & Metro Pass",
                    "asset_ticker": None,
                    "quantity": 0.0,
                    "unit_price": 0.0,
                    "is_ai_tagged": 1,
                },
                {
                    "date": _dt(16),
                    "type": "Expense",
                    "category": "Entertainment & Subscriptions",
                    "amount": 1499.0,
                    "description": "Cloud Storage & Streaming Bundle",
                    "asset_ticker": None,
                    "quantity": 0.0,
                    "unit_price": 0.0,
                    "is_ai_tagged": 1,
                },
            ]
        )

        if idx == 2:
            rows.append(
                {
                    "date": _dt(21),
                    "type": "Expense",
                    "category": "Healthcare",
                    "amount": 5200.0,
                    "description": "Annual Preventive Health Checkup",
                    "asset_ticker": None,
                    "quantity": 0.0,
                    "unit_price": 0.0,
                    "is_ai_tagged": 0,
                }
            )
        if idx == 3:
            rows.append(
                {
                    "date": _dt(14),
                    "type": "Expense",
                    "category": "Shopping & Lifestyle",
                    "amount": 19500.0,
                    "description": "Workstation Monitor Upgrade",
                    "asset_ticker": None,
                    "quantity": 0.0,
                    "unit_price": 0.0,
                    "is_ai_tagged": 1,
                }
            )

        # Investments
        nifty_price = 245.0 + idx * 4.5
        tcs_price = 3850.0 + idx * 45.0
        btc_price = 6200000.0 + idx * 150000.0
        rows.extend(
            [
                {
                    "date": _dt(5),
                    "type": "Investment",
                    "category": "Investment",
                    "amount": round(80.0 * nifty_price, 2),
                    "description": "Zerodha SIP NIFTYBEES ETF",
                    "asset_ticker": "NIFTY50",
                    "quantity": 80.0,
                    "unit_price": nifty_price,
                    "is_ai_tagged": 0,
                },
                {
                    "date": _dt(11),
                    "type": "Investment",
                    "category": "Investment",
                    "amount": round(4.0 * tcs_price, 2),
                    "description": "Equity Accumulation TCS",
                    "asset_ticker": "TCS",
                    "quantity": 4.0,
                    "unit_price": tcs_price,
                    "is_ai_tagged": 0,
                },
                {
                    "date": _dt(20),
                    "type": "Investment",
                    "category": "Investment",
                    "amount": round(0.002 * btc_price, 2),
                    "description": "Bitcoin DCA Allocation",
                    "asset_ticker": "BTC",
                    "quantity": 0.002,
                    "unit_price": btc_price,
                    "is_ai_tagged": 0,
                },
            ]
        )

    sample_df = pd.DataFrame(rows)
    return bulk_insert_transactions(sample_df, db_path=db_path, user_id=user_id)


def create_user(
    *,
    full_name: str,
    email: Optional[str],
    phone: Optional[str],
    password_hash: str,
    db_path: Optional[Union[str, Path]] = None,
) -> dict[str, Any]:
    """Persist a new user account with email and/or phone number."""
    init_db(db_path)
    clean_name = full_name.strip()
    if not clean_name:
        raise ValueError("Full name cannot be empty.")
    if not email and not phone:
        raise ValueError("Either email or phone number is required.")

    created_at = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    pool = _get_pool(db_path)
    with pool.connection(write=True) as conn:
        try:
            cursor = conn.execute(
                """
                INSERT INTO users (full_name, email, phone, password_hash, created_at)
                VALUES (?, ?, ?, ?, ?);
                """,
                (clean_name, email, phone, password_hash, created_at),
            )
        except (sqlite3.IntegrityError, *PSYCOPG_INTEGRITY_ERRORS) as exc:
            raise ValueError("An account with that email or phone number already exists.") from exc
        new_id = int(cursor.lastrowid or 0)
        row = conn.execute(
            "SELECT id, full_name, email, phone, password_hash, created_at FROM users WHERE id = ?;",
            (new_id,),
        ).fetchone()
    return dict(row)


def get_user_by_identifier(
    identifier: str,
    db_path: Optional[Union[str, Path]] = None,
) -> Optional[dict[str, Any]]:
    """Fetch a user record matching either normalized email or normalized phone."""
    init_db(db_path)
    pool = _get_pool(db_path)
    with pool.connection(write=False) as conn:
        row = conn.execute(
            """
            SELECT id, full_name, email, phone, password_hash, created_at
            FROM users
            WHERE email = ? OR phone = ?
            LIMIT 1;
            """,
            (identifier, identifier),
        ).fetchone()
    return dict(row) if row else None


def get_user_by_id(
    user_id: int,
    db_path: Optional[Union[str, Path]] = None,
) -> Optional[dict[str, Any]]:
    """Fetch a user record by primary key ID."""
    init_db(db_path)
    pool = _get_pool(db_path)
    with pool.connection(write=False) as conn:
        row = conn.execute(
            "SELECT id, full_name, email, phone, password_hash, created_at FROM users WHERE id = ?;",
            (int(user_id),),
        ).fetchone()
    return dict(row) if row else None


def store_otp_challenge(
    identifier: str,
    otp_hash: str,
    expires_at: str,
    db_path: Optional[Union[str, Path]] = None,
) -> None:
    """Invalidate prior unconsumed OTPs for identifier and store a fresh hashed OTP challenge."""
    init_db(db_path)
    created_at = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    pool = _get_pool(db_path)
    with pool.connection(write=True) as conn:
        conn.execute(
            "UPDATE otp_challenges SET consumed = 1 WHERE identifier = ? AND consumed = 0;",
            (identifier,),
        )
        conn.execute(
            """
            INSERT INTO otp_challenges (identifier, otp_hash, expires_at, consumed, created_at)
            VALUES (?, ?, ?, 0, ?);
            """,
            (identifier, otp_hash, expires_at, created_at),
        )


def consume_otp_challenge(
    identifier: str,
    otp_hash: str,
    now_iso: str,
    db_path: Optional[Union[str, Path]] = None,
) -> bool:
    """Verify and atomically consume an active, unexpired OTP challenge."""
    init_db(db_path)
    pool = _get_pool(db_path)
    with pool.connection(write=True) as conn:
        row = conn.execute(
            """
            SELECT id, otp_hash, expires_at
            FROM otp_challenges
            WHERE identifier = ? AND consumed = 0
            ORDER BY id DESC
            LIMIT 1;
            """,
            (identifier,),
        ).fetchone()
        if row is None:
            return False
        if str(row["expires_at"]) < now_iso:
            conn.execute("UPDATE otp_challenges SET consumed = 1 WHERE id = ?;", (int(row["id"]),))
            return False
        if str(row["otp_hash"]) != otp_hash:
            return False
        conn.execute("UPDATE otp_challenges SET consumed = 1 WHERE id = ?;", (int(row["id"]),))
        return True
