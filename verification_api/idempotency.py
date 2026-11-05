import hashlib
import sqlite3
import time

from verification_api.config import get_idempotency_database_path


IDEMPOTENCY_TTL_SECONDS = 24 * 60 * 60


def claim_idempotency_key(key: str) -> bool:
    database_path = get_idempotency_database_path()
    database_path.parent.mkdir(parents=True, exist_ok=True)
    key_digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    now = int(time.time())
    connection = sqlite3.connect(database_path, timeout=5)

    try:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS idempotency_keys (
                key_digest TEXT PRIMARY KEY,
                created_at INTEGER NOT NULL
            )
            """
        )
        connection.execute("BEGIN IMMEDIATE")
        connection.execute(
            "DELETE FROM idempotency_keys WHERE created_at < ?",
            (now - IDEMPOTENCY_TTL_SECONDS,),
        )
        try:
            connection.execute(
                "INSERT INTO idempotency_keys (key_digest, created_at) VALUES (?, ?)",
                (key_digest, now),
            )
        except sqlite3.IntegrityError:
            connection.rollback()
            return False
        connection.commit()
        return True
    finally:
        connection.close()
