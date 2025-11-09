import sqlite3
from pathlib import Path
from typing import List, Tuple

DB_PATH = Path(__file__).resolve().parent / "chat_memory.db"


def get_connection():
    """Create a connection and make sure table exists."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            message TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    return conn


def save_message(session_id: str, role: str, message: str):
    """Save a single message (either user or assistant)."""
    conn = get_connection()
    conn.execute(
        "INSERT INTO chat_history (session_id, role, message) VALUES (?, ?, ?);",
        (session_id, role, message),
    )
    conn.commit()
    conn.close()


def load_history(session_id: str) -> List[Tuple[str, str]]:
    """
    Load full conversation history for one session.
    Returns a list of (role, message).
    """
    conn = get_connection()
    cur = conn.execute(
        "SELECT role, message FROM chat_history WHERE session_id = ? ORDER BY id;",
        (session_id,),
    )
    rows = cur.fetchall()
    conn.close()
    return rows
