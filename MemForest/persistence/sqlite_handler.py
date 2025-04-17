import sqlite3
import json
import os
from datetime import datetime
from typing import List, Dict, Optional, Iterable

from MemForest.memory.memory_unit import MemoryUnit
from MemForest.memory.session_memory import SessionMemory
from MemForest.memory.long_term_memory import LongTermMemory

DEFAULT_DB_FOLDER = "memory_storage"  # Folder where DB files will be stored


def _get_db_path(chatbot_id: str, base_path: str = DEFAULT_DB_FOLDER) -> str:
    """Gets the path to the SQLite DB file for a specific chatbot."""
    chatbot_dir = os.path.join(base_path, chatbot_id)
    os.makedirs(chatbot_dir, exist_ok=True)
    return os.path.join(chatbot_dir, f"{chatbot_id}.db")


def _to_isoformat(dt: Optional[datetime]) -> Optional[str]:
    """Converts aware datetime to ISO format string."""
    if dt:
        return dt.isoformat()
    return None


def _from_isoformat(iso_str: Optional[str]) -> Optional[datetime]:
    """Converts ISO format string back to aware datetime."""
    if iso_str:
        try:
            dt = datetime.fromisoformat(iso_str)
            return dt
        except ValueError:
            print(f"Warning: Could not parse ISO format string: {iso_str}")
            return None
    return None


def initialize_db(chatbot_id: str, base_path: str = DEFAULT_DB_FOLDER):
    """Initializes the SQLite database and creates tables if they don't exist."""
    db_path = _get_db_path(chatbot_id, base_path)
    conn = None
    try:
        conn = sqlite3.connect(db_path, check_same_thread=False)  # Allow multi-thread access if needed
        cursor = conn.cursor()

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS memory_units (
            id TEXT PRIMARY KEY,
            parent_id TEXT,
            content TEXT NOT NULL,
            creation_time TEXT, -- Store as ISO 8601 string
            end_time TEXT,     -- Store as ISO 8601 string
            source TEXT,
            metadata TEXT,     -- Store as JSON string
            last_visit INTEGER DEFAULT 0,
            visit_count INTEGER DEFAULT 0,
            never_delete INTEGER DEFAULT 0, -- Boolean as 0 or 1
            children_ids TEXT, -- Store as JSON string
            rank INTEGER DEFAULT 0,
            pre_id TEXT,
            next_id TEXT,
            group_id TEXT
        )
        """)
        # Add indices for common lookups
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_mu_parent_id ON memory_units (parent_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_mu_group_id ON memory_units (group_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_mu_creation_time ON memory_units (creation_time)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_mu_rank ON memory_units (rank)")

        # Session Memories Table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS session_memories (
            id TEXT PRIMARY KEY,
            memory_unit_ids TEXT NOT NULL, -- Store as JSON string
            creation_time TEXT,
            end_time TEXT
        )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_sm_creation_time ON session_memories (creation_time)")

        # Long Term Memories Table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS long_term_memories (
            id TEXT PRIMARY KEY,
            chatbot_id TEXT NOT NULL,
            visit_count INTEGER DEFAULT 0,
            session_ids TEXT NOT NULL, -- Store as JSON string
            summary_unit_ids TEXT,    -- Store as JSON string
            last_session_id TEXT,
            creation_time TEXT,
            end_time TEXT
        )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_ltm_chatbot_id ON long_term_memories (chatbot_id)")

        conn.commit()
        print(f"SQLite database initialized/verified at: {db_path}")
    except sqlite3.Error as e:
        print(f"Error initializing SQLite database '{db_path}': {e}")
    finally:
        if conn:
            conn.close()


def get_connection(chatbot_id: str, base_path: str = DEFAULT_DB_FOLDER) -> Optional[sqlite3.Connection]:
    """Gets a connection object to the chatbot's database."""
    db_path = _get_db_path(chatbot_id, base_path)
    try:
        if not os.path.exists(db_path):
            initialize_db(chatbot_id, base_path)
        elif os.path.getsize(db_path) == 0:  # Handle empty file case
            initialize_db(chatbot_id, base_path)

        conn = sqlite3.connect(db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row  # Return rows as dictionary-like objects
        return conn
    except sqlite3.Error as e:
        print(f"Error connecting to SQLite database '{db_path}': {e}")
        return None


# --- MemoryUnit Operations ---

def upsert_memory_units(conn: sqlite3.Connection, units: Iterable[MemoryUnit]):
    """Inserts or replaces multiple MemoryUnit objects in the database."""
    sql = """
    INSERT OR REPLACE INTO memory_units
    (id, parent_id, content, creation_time, end_time, source, metadata,
     last_visit, visit_count, never_delete, children_ids, rank, pre_id, next_id, group_id)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    data_to_insert = []
    for unit in units:
        data_to_insert.append((
            unit.id, unit.parent_id, unit.content,
            _to_isoformat(unit.creation_time), _to_isoformat(unit.end_time),
            unit.source, json.dumps(unit.metadata or {}), unit.last_visit,
            unit.visit_count, int(unit.never_delete), json.dumps(unit.children_ids or []),
            unit.rank, unit.pre_id, unit.next_id, unit.group_id
        ))
    try:
        with conn:
            conn.executemany(sql, data_to_insert)
    except sqlite3.Error as e:
        print(f"Error upserting memory units: {e}")
        raise


def load_memory_unit(conn: sqlite3.Connection, unit_id: str) -> Optional[MemoryUnit]:
    """Loads a single MemoryUnit by its ID."""
    sql = "SELECT * FROM memory_units WHERE id = ?"
    try:
        cursor = conn.cursor()
        cursor.execute(sql, (unit_id,))
        row = cursor.fetchone()
        if row:
            return _row_to_memory_unit(row)
        return None
    except sqlite3.Error as e:
        print(f"Error loading memory unit {unit_id}: {e}")
        return None


def load_memory_units(conn: sqlite3.Connection, unit_ids: List[str]) -> Dict[str, MemoryUnit]:
    """Loads multiple MemoryUnits by their IDs."""
    if not unit_ids:
        return {}
    placeholders = ', '.join('?' for _ in unit_ids)
    sql = f"SELECT * FROM memory_units WHERE id IN ({placeholders})"
    units = {}
    try:
        cursor = conn.cursor()
        cursor.execute(sql, tuple(unit_ids))
        rows = cursor.fetchall()
        for row in rows:
            unit = _row_to_memory_unit(row)
            if unit:
                units[unit.id] = unit
        return units
    except sqlite3.Error as e:
        print(f"Error loading memory units {unit_ids}: {e}")
        return {}


def load_memory_units_for_session(conn: sqlite3.Connection, session_id: str) -> Dict[str, MemoryUnit]:
    """Loads all memory units associated with a given session ID."""
    if not session_id:
        return {}
    sql = f"SELECT * FROM memory_units WHERE group_id = ?"
    units = {}
    try:
        cursor = conn.cursor()
        cursor.execute(sql, (session_id,))
        rows = cursor.fetchall()
        for row in rows:
            unit = _row_to_memory_unit(row)
            if unit:
                units[unit.id] = unit
        return units
    except sqlite3.Error as e:
        print(f"Error loading units for session memory {session_id}: {e}")
        return {}


def load_all_memory_units(conn: sqlite3.Connection) -> Dict[str, MemoryUnit]:
    """Loads ALL memory units from the database. Use with caution on large DBs."""
    sql = "SELECT * FROM memory_units"
    units = {}
    try:
        cursor = conn.cursor()
        cursor.execute(sql)
        rows = cursor.fetchall()
        for row in rows:
            unit = _row_to_memory_unit(row)
            if unit:
                units[unit.id] = unit
        return units
    except sqlite3.Error as e:
        print(f"Error loading all memory units: {e}")
        return {}


def delete_memory_units(conn: sqlite3.Connection, unit_ids: List[str]):
    """Deletes multiple MemoryUnits by their IDs."""
    if not unit_ids:
        return 0
    placeholders = ', '.join('?' * len(unit_ids))
    sql = f"DELETE FROM memory_units WHERE id IN ({placeholders})"
    try:
        with conn:
            cursor = conn.cursor()
            cursor.execute(sql, tuple(unit_ids))
            return cursor.rowcount
    except sqlite3.Error as e:
        print(f"Error deleting memory units {unit_ids}: {e}")
        return 0


def _row_to_memory_unit(row: sqlite3.Row) -> Optional[MemoryUnit]:
    """Converts a database row into a MemoryUnit object."""
    try:
        metadata_str = row['metadata']
        children_ids_str = row['children_ids']
        return MemoryUnit(
            memory_id=row['id'],
            parent_id=row['parent_id'],
            content=row['content'],
            creation_time=_from_isoformat(row['creation_time']),
            end_time=_from_isoformat(row['end_time']),
            source=row['source'],
            metadata=json.loads(metadata_str) if metadata_str else {},
            last_visit=row['last_visit'],
            visit_count=row['visit_count'],
            never_delete=bool(row['never_delete']),
            children_ids=json.loads(children_ids_str) if children_ids_str else [],
            rank=row['rank'],
            pre_id=row['pre_id'],
            next_id=row['next_id'],
            group_id=row['group_id']
        )
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        print(f"Error converting row to MemoryUnit (ID: {row['id']}): {e}")
        return None


# --- SessionMemory Operations ---

def upsert_session_memories(conn: sqlite3.Connection, sessions: Iterable[SessionMemory]):
    """Inserts or replaces multiple SessionMemory objects."""
    sql = """
    INSERT OR REPLACE INTO session_memories
    (id, memory_unit_ids, creation_time, end_time)
    VALUES (?, ?, ?, ?)
    """
    data_to_insert = [(
        s.id, json.dumps(s.memory_unit_ids or []),
        _to_isoformat(s.creation_time), _to_isoformat(s.end_time)
    ) for s in sessions]
    try:
        with conn:
            conn.executemany(sql, data_to_insert)
    except sqlite3.Error as e:
        print(f"Error upserting session memories: {e}")
        raise


def load_session_memory(conn: sqlite3.Connection, session_id: str) -> Optional[SessionMemory]:
    """Loads a single SessionMemory by its ID."""
    sql = "SELECT * FROM session_memories WHERE id = ?"
    try:
        cursor = conn.cursor()
        cursor.execute(sql, (session_id,))
        row = cursor.fetchone()
        if row:
            return _row_to_session_memory(row)
        return None
    except sqlite3.Error as e:
        print(f"Error loading session memory {session_id}: {e}")
        return None


def load_session_memories(conn: sqlite3.Connection, session_ids: List[str]) -> Dict[str, SessionMemory]:
    """Loads multiple SessionMemories by their IDs."""
    if not session_ids:
        return {}
    placeholders = ', '.join('?' * len(session_ids))
    sql = f"SELECT * FROM session_memories WHERE id IN ({placeholders})"
    sessions = {}
    try:
        cursor = conn.cursor()
        cursor.execute(sql, tuple(session_ids))
        rows = cursor.fetchall()
        for row in rows:
            session = _row_to_session_memory(row)
            if session:
                sessions[session.id] = session
        return sessions
    except sqlite3.Error as e:
        print(f"Error loading session memories {session_ids}: {e}")
        return {}


def load_all_session_memories(conn: sqlite3.Connection) -> Dict[str, SessionMemory]:
    """Loads ALL session memories from the database."""
    sql = "SELECT * FROM session_memories"
    sessions = {}
    try:
        cursor = conn.cursor()
        cursor.execute(sql)
        rows = cursor.fetchall()
        for row in rows:
            session = _row_to_session_memory(row)
            if session:
                sessions[session.id] = session
        return sessions
    except sqlite3.Error as e:
        print(f"Error loading all session memories: {e}")
        return {}


def delete_session_memories(conn: sqlite3.Connection, session_ids: List[str]):
    """Deletes multiple SessionMemories by their IDs."""
    if not session_ids:
        return 0
    placeholders = ', '.join('?' * len(session_ids))
    sql = f"DELETE FROM session_memories WHERE id IN ({placeholders})"
    try:
        with conn:
            cursor = conn.cursor()
            cursor.execute(sql, tuple(session_ids))
            return cursor.rowcount
    except sqlite3.Error as e:
        print(f"Error deleting session memories {session_ids}: {e}")
        return 0


def _row_to_session_memory(row: sqlite3.Row) -> Optional[SessionMemory]:
    """Converts a database row into a SessionMemory object."""
    try:
        mu_ids_str = row['memory_unit_ids']
        return SessionMemory(
            session_id=row['id'],
            memory_unit_ids=json.loads(mu_ids_str) if mu_ids_str else [],
            creation_time=_from_isoformat(row['creation_time']),
            end_time=_from_isoformat(row['end_time'])
        )
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        print(f"Error converting row to SessionMemory (ID: {row['id']}): {e}")
        return None


# --- LongTermMemory Operations ---

def upsert_long_term_memories(conn: sqlite3.Connection, ltms: Iterable[LongTermMemory]):
    """Inserts or replaces multiple LongTermMemory objects."""
    sql = """
    INSERT OR REPLACE INTO long_term_memories
    (id, chatbot_id, visit_count, session_ids, summary_unit_ids, last_session_id, creation_time, end_time)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """
    data_to_insert = [(
        ltm.id, ltm.chatbot_id, ltm.visit_count,
        json.dumps(ltm.session_ids or []),
        json.dumps(ltm.summary_unit_ids or []),
        ltm.last_session_id,
        _to_isoformat(ltm.creation_time), _to_isoformat(ltm.end_time)
    ) for ltm in ltms]
    try:
        with conn:
            conn.executemany(sql, data_to_insert)
    except sqlite3.Error as e:
        print(f"Error upserting long term memories: {e}")
        raise


def load_long_term_memory(conn: sqlite3.Connection, ltm_id: str, chatbot_id: str) -> Optional[LongTermMemory]:
    """Loads a single LongTermMemory by its ID and chatbot ID."""
    sql = "SELECT * FROM long_term_memories WHERE id = ? AND chatbot_id = ?"
    try:
        cursor = conn.cursor()
        cursor.execute(sql, (ltm_id, chatbot_id))
        row = cursor.fetchone()
        if row:
            return _row_to_long_term_memory(row)
        return None
    except sqlite3.Error as e:
        print(f"Error loading long term memory {ltm_id} for chatbot {chatbot_id}: {e}")
        return None


def load_all_long_term_memories_for_chatbot(conn: sqlite3.Connection, chatbot_id: str) -> Dict[str, LongTermMemory]:
    """Loads all LongTermMemory objects for a specific chatbot."""
    sql = "SELECT * FROM long_term_memories WHERE chatbot_id = ?"
    ltms = {}
    try:
        cursor = conn.cursor()
        cursor.execute(sql, (chatbot_id,))
        rows = cursor.fetchall()
        for row in rows:
            ltm = _row_to_long_term_memory(row)
            if ltm:
                ltms[ltm.id] = ltm
        return ltms
    except sqlite3.Error as e:
        print(f"Error loading all long term memories for chatbot {chatbot_id}: {e}")
        return {}


def delete_long_term_memories(conn: sqlite3.Connection, ltm_ids: List[str]):
    """Deletes multiple LongTermMemories by their IDs."""
    if not ltm_ids:
        return 0
    placeholders = ', '.join('?' * len(ltm_ids))
    sql = f"DELETE FROM long_term_memories WHERE id IN ({placeholders})"
    try:
        with conn:
            cursor = conn.cursor()
            cursor.execute(sql, tuple(ltm_ids))
            return cursor.rowcount
    except sqlite3.Error as e:
        print(f"Error deleting long term memories {ltm_ids}: {e}")
        return 0


def _row_to_long_term_memory(row: sqlite3.Row) -> Optional[LongTermMemory]:
    """Converts a database row into a LongTermMemory object."""
    try:
        session_ids_str = row['session_ids']
        summary_unit_ids_str = row['summary_unit_ids']
        return LongTermMemory(
            ltm_id=row['id'],
            chatbot_id=row['chatbot_id'],
            visit_count=row['visit_count'],
            session_ids=json.loads(session_ids_str) if session_ids_str else [],
            summary_unit_ids=json.loads(summary_unit_ids_str) if summary_unit_ids_str else [],
            last_session_id=row['last_session_id'],
            creation_time=_from_isoformat(row['creation_time']),
            end_time=_from_isoformat(row['end_time'])
        )
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        print(f"Error converting row to LongTermMemory (ID: {row['id']}): {e}")
        return None
