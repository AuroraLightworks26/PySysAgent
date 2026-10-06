import sqlite3
from typing import List, Dict, Any, Optional


class DatabaseStore:
    def __init__(self, db_path: str = "sysagent.db"):
        self.db_path = db_path
        self._init_db()

    def _get_connection(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            # Active System Rules Schema
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS system_rules (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    context_key TEXT,
                    description TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                );
            """)
            # Active System Profile Schema
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS system_profile (
                    category TEXT,
                    attribute TEXT,
                    value TEXT,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (category, attribute)
                );
            """)
            conn.commit()

    def get_system_rules(self) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT context_key, description FROM system_rules ORDER BY id DESC")
            return [dict(row) for row in cursor.fetchall()]

    def save_system_rule(self, context_key: str, description: str) -> bool:
        clean_rule = description.strip()
        if not clean_rule:
            return False

        with self._get_connection() as conn:
            cursor = conn.cursor()
            # Case-insensitive exact deduplication check
            cursor.execute(
                "SELECT COUNT(1) FROM system_rules WHERE LOWER(description) = LOWER(?)",
                (clean_rule,)
            )
            if cursor.fetchone()[0] > 0:
                return False  # Rule already exists in memory

            cursor.execute(
                "INSERT INTO system_rules (context_key, description) VALUES (?, ?)",
                (context_key, clean_rule)
            )
            conn.commit()
            return True

    def get_system_profile(self) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT category, attribute, value FROM system_profile")
            return [dict(row) for row in cursor.fetchall()]