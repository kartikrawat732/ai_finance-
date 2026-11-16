"""Migration script to sync existing local SQLite data into Supabase (PostgreSQL)."""

import os
import sqlite3
from pathlib import Path
from dotenv import load_dotenv
import psycopg
from psycopg.rows import dict_row

def run_migration() -> None:
    base_dir = Path(__file__).resolve().parent
    load_dotenv(base_dir / ".env")

    pg_url = os.getenv("DATABASE_URL")
    if not pg_url:
        print("ERROR: DATABASE_URL not found in .env!")
        return

    sqlite_path = base_dir / "finance_tracker.db"
    if not sqlite_path.is_file():
        print(f"No local SQLite database found at {sqlite_path}.")
        return

    print(f"Connecting to SQLite: {sqlite_path}")
    s_conn = sqlite3.connect(str(sqlite_path))
    s_conn.row_factory = sqlite3.Row

    print("Connecting to Supabase PostgreSQL...")
    p_conn = psycopg.connect(pg_url, row_factory=dict_row)

    with p_conn.cursor() as p_cur:
        # 1. Migrate Users
        users = [dict(r) for r in s_conn.execute("SELECT * FROM users;").fetchall()]
        print(f"Migrating {len(users)} users...")
        for u in users:
            p_cur.execute(
                """
                INSERT INTO users (id, full_name, email, phone, password_hash, created_at)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    full_name = EXCLUDED.full_name,
                    email = EXCLUDED.email,
                    phone = EXCLUDED.phone,
                    password_hash = EXCLUDED.password_hash;
                """,
                (u["id"], u["full_name"], u["email"], u["phone"], u["password_hash"], u["created_at"]),
            )
        if users:
            p_cur.execute("SELECT setval('users_id_seq', (SELECT COALESCE(MAX(id), 1) FROM users));")

        # 2. Migrate Transactions
        txs = [dict(r) for r in s_conn.execute("SELECT * FROM transactions;").fetchall()]
        print(f"Migrating {len(txs)} transactions...")
        for t in txs:
            p_cur.execute(
                """
                INSERT INTO transactions (
                    id, date, type, category, amount, description,
                    asset_ticker, quantity, unit_price, is_ai_tagged, user_id
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    date = EXCLUDED.date,
                    type = EXCLUDED.type,
                    category = EXCLUDED.category,
                    amount = EXCLUDED.amount,
                    description = EXCLUDED.description,
                    asset_ticker = EXCLUDED.asset_ticker,
                    quantity = EXCLUDED.quantity,
                    unit_price = EXCLUDED.unit_price,
                    is_ai_tagged = EXCLUDED.is_ai_tagged,
                    user_id = EXCLUDED.user_id;
                """,
                (
                    t["id"],
                    t["date"],
                    t["type"],
                    t["category"],
                    t["amount"],
                    t["description"],
                    t["asset_ticker"],
                    t["quantity"],
                    t["unit_price"],
                    t["is_ai_tagged"],
                    t["user_id"] if "user_id" in t.keys() else None,
                ),
            )
        if txs:
            p_cur.execute("SELECT setval('transactions_id_seq', (SELECT COALESCE(MAX(id), 1) FROM transactions));")

        # 3. Migrate OTP challenges if any
        try:
            otps = [dict(r) for r in s_conn.execute("SELECT * FROM otp_challenges;").fetchall()]
            for o in otps:
                p_cur.execute(
                    """
                    INSERT INTO otp_challenges (id, identifier, otp_hash, expires_at, consumed, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (id) DO NOTHING;
                    """,
                    (o["id"], o["identifier"], o["otp_hash"], o["expires_at"], o["consumed"], o["created_at"]),
                )
            if otps:
                p_cur.execute("SELECT setval('otp_challenges_id_seq', (SELECT COALESCE(MAX(id), 1) FROM otp_challenges));")
            print(f"Migrated {len(otps)} OTP challenges.")
        except Exception as e:
            print("OTP migration skipped:", e)

    p_conn.commit()
    print("ALL DATA COMMITTED TO SUPABASE SUCCESSFULLY!")

    # Verify counts in Supabase
    with p_conn.cursor() as p_cur:
        u_count = p_cur.execute("SELECT COUNT(*) as count FROM users;").fetchone()["count"]
        t_count = p_cur.execute("SELECT COUNT(*) as count FROM transactions;").fetchone()["count"]
        print(f"Supabase Table Verification -> Users: {u_count}, Transactions: {t_count}")

    s_conn.close()
    p_conn.close()

if __name__ == "__main__":
    run_migration()
