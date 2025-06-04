import aiosqlite
import json
import os
from datetime import datetime
from typing import List, Dict, Optional, Iterable, Any, Tuple, Union, TypeVar
import struct
import asyncio

try:
    import sqlite_vec

    SQLITE_VEC_AVAILABLE = True
except ImportError:
    print("sqlite-vec Python package not installed. sqlite-vec support will be disabled.")
    SQLITE_VEC_AVAILABLE = False


def serialize_f32(vector: List[float]) -> bytes:
    """serializes a list of floats into a compact "raw bytes" format"""
    return struct.pack("%sf" % len(vector), *vector)


def deserialize_f32(vector_bytes: bytes) -> List[float]:
    """deserializes a compact "raw bytes" format into a list of floats"""
    if not vector_bytes:
        return []
    num_floats = len(vector_bytes) // struct.calcsize('f')
    return list(struct.unpack(f"{num_floats}f", vector_bytes))

from MemForest.memory.memory_unit import MemoryUnit
from MemForest.memory.session_memory import SessionMemory
from MemForest.memory.long_term_memory import LongTermMemory
# --- Configuration ---
DEFAULT_DB_FOLDER = "memory_storage"
T = TypeVar('T')


# --- Helper Functions ---
def _get_db_path(chatbot_id: str, base_path: str = DEFAULT_DB_FOLDER) -> str:
    chatbot_dir = base_path
    os.makedirs(chatbot_dir, exist_ok=True)
    return os.path.join(chatbot_dir, f"{chatbot_id}.db")


def _to_iso_datetime_str(dt: Optional[datetime]) -> Optional[str]:
    if dt is None:
        return None
    return dt.isoformat()


def _to_timestamp(cur_time: Any) -> Optional[datetime]:
    if isinstance(cur_time, str):
        try:
            return datetime.fromisoformat(cur_time)
        except ValueError:
            try:
                return datetime.fromtimestamp(float(cur_time))
            except ValueError:
                return None
    elif isinstance(cur_time, (float, int)):
        try:
            return datetime.fromtimestamp(cur_time)
        except ValueError:
            return None
    elif isinstance(cur_time, datetime):
        return cur_time
    return None


class AsyncSQLiteHandler:

    def __init__(self):
        self.chatbot_id: Optional[str] = None
        self.db_path: Optional[str] = None
        self.use_sqlite_vec: bool = False
        self.embedding_dim: Optional[int] = None
        self._conn: Optional[aiosqlite.Connection] = None
        self._lock = asyncio.Lock()
        self._transaction_depth: int = 0

    def initialize(self, chatbot_id: str, base_path: str = DEFAULT_DB_FOLDER, use_sqlite_vec: bool = False,
                   embedding_dim: Optional[int] = None):
        self.chatbot_id = chatbot_id
        self.db_path = _get_db_path(chatbot_id, base_path)
        self.use_sqlite_vec = use_sqlite_vec and SQLITE_VEC_AVAILABLE
        self.embedding_dim = embedding_dim

        if self.use_sqlite_vec and (not embedding_dim or not isinstance(embedding_dim, int) or embedding_dim <= 0):
            raise ValueError("embedding_dim must be a positive integer when use_sqlite_vec is True")
        if use_sqlite_vec and not SQLITE_VEC_AVAILABLE:
            print("Warning: use_sqlite_vec requested, but sqlite-vec package not found. Disabling sqlite-vec.")
            self.use_sqlite_vec = False



    async def begin(self):
        """Starts a new transaction."""
        conn = await self._get_connection()
        if self._transaction_depth == 0:
            await conn.execute("BEGIN;")
        self._transaction_depth += 1

    async def commit(self):
        """Commits the current transaction."""
        if self._transaction_depth > 0:
            self._transaction_depth -= 1
            if self._transaction_depth == 0:
                conn = await self._get_connection()
                await conn.commit()
        else:
            print(f"SQLiteHandler WARNING: Commit called with transaction depth {self._transaction_depth} for {self.db_path}.")

    async def rollback(self):
        """Rolls back the current transaction."""
        if self._transaction_depth > 0:
            conn = await self._get_connection()
            try:
                # Rollback should be attempted if depth > 0, as an error might have occurred.
                await conn.rollback()
            except aiosqlite.OperationalError as e:
                # This can happen if the transaction was already closed (e.g., by a previous implicit commit or error)
                # but our depth counter was not yet zero.
                if "cannot rollback - no transaction is active" in str(e):
                    print(f"SQLiteHandler WARNING: Rollback attempted for {self.db_path} but no active transaction in SQLite, though depth was {self._transaction_depth}.")
                else:
                    raise # Re-raise other operational errors
            finally:
                self._transaction_depth = 0 # Rollback always ends the entire transaction and resets depth.
            # print(f"SQLiteHandler: ROLLBACK called, depth now {self._transaction_depth} for {self.db_path}") # Optional debug
        else:
            # This case (calling rollback when depth is already 0) can happen if an error occurs outside a managed transaction.
            # It's generally not an issue unless a transaction was expected.
            print(f"SQLiteHandler WARNING: Rollback called with transaction depth {self._transaction_depth} for {self.db_path}.")
            # Ensure depth is 0 if multiple rollbacks are called without intervening begins.
            self._transaction_depth = 0

    async def connect(self):
        if not self.db_path:
            raise ConnectionError("Handler not initialized. Call initialize() first.")
        if self._conn:
            try:
                async with self._conn.execute("SELECT 1") as cursor:
                    await cursor.fetchone()
                print(f"Already connected to {self.db_path} and connection is live.")
                return
            except aiosqlite.Error:
                print("Existing connection object found but it's dead. Attempting to close and reconnect.")
                try:
                    await self._conn.close()
                except aiosqlite.Error:
                    pass
                self._conn = None
        try:
            new_conn  = await aiosqlite.connect(self.db_path, timeout=15.0)
            new_conn .row_factory = aiosqlite.Row
            # await self._conn.execute("PRAGMA foreign_keys = ON;")
            await new_conn.execute("PRAGMA foreign_keys = ON;")
            await new_conn.execute("PRAGMA journal_mode=WAL;")
            await new_conn.commit()

            if self.use_sqlite_vec:
                await new_conn.enable_load_extension(True)
                try:
                    await asyncio.to_thread(sqlite_vec.load, new_conn)  # type: ignore
                    print("sqlite-vec extension loaded successfully.")
                except aiosqlite.Error as e:
                    print(f"Error loading sqlite-vec extension: {e}.")
                    await new_conn.close()
                    raise ConnectionError("Failed to load sqlite-vec extension.") from e
                finally:
                    await new_conn.enable_load_extension(False)
            self._conn = new_conn
            print(f"Async SQLite connection established to {self.db_path}")
        except aiosqlite.Error as e:
            print(f"Error connecting to SQLite database '{self.db_path}': {e}")
            self._conn = None
            raise ConnectionError(f"Could not connect to SQLite DB: {e}") from e

    async def close(self):
        async with self._lock:
            if self._conn:
                conn_to_close = self._conn
                self._conn = None
                try:
                    await conn_to_close.close()
                    print(f"Async SQLite connection closed for {self.db_path}")
                except aiosqlite.Error as e:
                    print(f"Error closing SQLite connection: {e}")
                finally:
                    self._conn = None
            else:
                print("No active SQLite connection to close.")

    async def _get_connection(self) -> aiosqlite.Connection:
        async with self._lock:
            if self._conn is None:
                # print("No connection object found, attempting to connect.") # Debug
                await self.connect()
            else:
                try:
                    async with self._conn.execute("SELECT 1") as cursor:
                        await cursor.fetchone()
                    # print("Existing connection is live.")
                except aiosqlite.Error:
                    print("Connection found but it's dead. Attempting to reconnect.")
                    try:
                        await self._conn.close()
                    except aiosqlite.Error:
                        pass
                    self._conn = None
                    await self.connect()
            if self._conn is None:
                raise ConnectionError("SQLite connection is not available or failed to establish.")
            return self._conn

    async def initialize_db(self):
        conn = await self._get_connection()
        async with conn.cursor() as cursor:
            await cursor.execute("PRAGMA foreign_keys = ON;")
            await cursor.execute("""
            CREATE TABLE IF NOT EXISTS memory_units (
                id TEXT PRIMARY KEY, content TEXT NOT NULL, creation_time TEXT, end_time TEXT,
                source TEXT, metadata TEXT, last_visit INTEGER DEFAULT 0, visit_count INTEGER DEFAULT 0,
                never_delete INTEGER DEFAULT 0, rank INTEGER DEFAULT 0
            )""")
            await cursor.execute("CREATE INDEX IF NOT EXISTS idx_mu_creation_time ON memory_units (creation_time)")
            await cursor.execute("CREATE INDEX IF NOT EXISTS idx_mu_rank ON memory_units (rank)")

            await cursor.execute("""
            CREATE TABLE IF NOT EXISTS session_memories (
                id TEXT PRIMARY KEY, creation_time TEXT, end_time TEXT
            )""")
            await cursor.execute("CREATE INDEX IF NOT EXISTS idx_sm_creation_time ON session_memories (creation_time)")

            await cursor.execute("""
            CREATE TABLE IF NOT EXISTS long_term_memories (
                id TEXT PRIMARY KEY, chatbot_id TEXT NOT NULL, visit_count INTEGER DEFAULT 0,
                last_session_id TEXT, creation_time TEXT, end_time TEXT
            )""")
            await cursor.execute("CREATE INDEX IF NOT EXISTS idx_ltm_chatbot_id ON long_term_memories (chatbot_id)")

            await cursor.execute("""
            CREATE TABLE IF NOT EXISTS mu_hierarchy_edges (
                child_unit_id TEXT PRIMARY KEY, parent_unit_id TEXT NOT NULL,
                FOREIGN KEY (child_unit_id) REFERENCES memory_units(id) ON DELETE CASCADE,
                FOREIGN KEY (parent_unit_id) REFERENCES memory_units(id) ON DELETE CASCADE
            )""")
            await cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_mu_hierarchy_parent ON mu_hierarchy_edges (parent_unit_id)")

            await cursor.execute("""
            CREATE TABLE IF NOT EXISTS mu_sequence_edges (
                unit_id TEXT PRIMARY KEY, pre_unit_id TEXT UNIQUE, next_unit_id TEXT UNIQUE,
                FOREIGN KEY (unit_id) REFERENCES memory_units(id) ON DELETE CASCADE,
                FOREIGN KEY (pre_unit_id) REFERENCES memory_units(id) ON DELETE SET NULL,
                FOREIGN KEY (next_unit_id) REFERENCES memory_units(id) ON DELETE SET NULL
            )""")

            await cursor.execute("""
            CREATE TABLE IF NOT EXISTS mu_session_group_membership (
                unit_id TEXT NOT NULL, session_id TEXT NOT NULL, PRIMARY KEY (unit_id, session_id),
                FOREIGN KEY (unit_id) REFERENCES memory_units(id) ON DELETE CASCADE,
                FOREIGN KEY (session_id) REFERENCES session_memories(id) ON DELETE CASCADE
            )""")
            await cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_musgm_session ON mu_session_group_membership (session_id)")

            await cursor.execute("""
            CREATE TABLE IF NOT EXISTS mu_ltm_group_membership (
                unit_id TEXT NOT NULL, ltm_id TEXT NOT NULL, PRIMARY KEY (unit_id, ltm_id),
                FOREIGN KEY (unit_id) REFERENCES memory_units(id) ON DELETE CASCADE,
                FOREIGN KEY (ltm_id) REFERENCES long_term_memories(id) ON DELETE CASCADE
            )""")
            await cursor.execute("CREATE INDEX IF NOT EXISTS idx_mulgm_ltm ON mu_ltm_group_membership (ltm_id)")

            await cursor.execute("""
            CREATE TABLE IF NOT EXISTS session_unit_membership (
                session_id TEXT NOT NULL, unit_id TEXT NOT NULL, PRIMARY KEY (session_id, unit_id),
                FOREIGN KEY (session_id) REFERENCES session_memories(id) ON DELETE CASCADE,
                FOREIGN KEY (unit_id) REFERENCES memory_units(id) ON DELETE CASCADE
            )""")
            await cursor.execute("CREATE INDEX IF NOT EXISTS idx_sum_unit ON session_unit_membership (unit_id)")

            await cursor.execute("""
            CREATE TABLE IF NOT EXISTS ltm_session_membership (
                ltm_id TEXT NOT NULL, session_id TEXT NOT NULL, PRIMARY KEY (ltm_id, session_id),
                FOREIGN KEY (ltm_id) REFERENCES long_term_memories(id) ON DELETE CASCADE,
                FOREIGN KEY (session_id) REFERENCES session_memories(id) ON DELETE CASCADE
            )""")
            await cursor.execute("CREATE INDEX IF NOT EXISTS idx_lsn_session ON ltm_session_membership (session_id)")

            await cursor.execute("""
            CREATE TABLE IF NOT EXISTS ltm_summary_unit_membership (
                ltm_id TEXT NOT NULL, summary_unit_id TEXT NOT NULL, PRIMARY KEY (ltm_id, summary_unit_id),
                FOREIGN KEY (ltm_id) REFERENCES long_term_memories(id) ON DELETE CASCADE,
                FOREIGN KEY (summary_unit_id) REFERENCES memory_units(id) ON DELETE CASCADE
            )""")
            await cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_lsum_summary_unit ON ltm_summary_unit_membership (summary_unit_id)")

            if self.use_sqlite_vec:
                async with cursor.execute(
                        "SELECT name FROM sqlite_master WHERE type='virtual table' AND name='memory_units_vss'") as cursor:
                    vss_exists = await self._fetch_one_as_dict(cursor)
                if not vss_exists:
                    if not self.embedding_dim: raise ValueError("embedding_dim required for VSS table.")
                    await cursor.execute(
                        f"CREATE VIRTUAL TABLE memory_units_vss USING vec0(embedding float[{self.embedding_dim}])")
                    print("VSS virtual table 'memory_units_vss' created.")
                else:
                    print("VSS virtual table 'memory_units_vss' already exists.")
        print(f"Async SQLite database initialized/verified at: {self.db_path}")

    async def begin_transaction(self):
        conn = await self._get_connection()
        await conn.execute("BEGIN;")

    async def commit_transaction(self):
        conn = await self._get_connection()
        await conn.commit()

    async def rollback_transaction(self):
        conn = await self._get_connection()
        await conn.rollback()

    async def set_foreign_keys(self, enable: bool):
        """
        Enables or disables foreign key constraints for the current connection.
        """
        conn = await self._get_connection()
        state = "ON" if enable else "OFF"
        try:
            await conn.execute(f"PRAGMA foreign_keys = {state};")
            await conn.commit()
            print(f"SQLite foreign keys set to {state}.")
        except aiosqlite.Error as e:
            print(f"Error setting foreign keys to {state}: {e}")

    async def check_foreign_keys(self) -> List[Dict[str, Any]]:
        """
        Checks for foreign key violations in the database.
        Returns a list of violations, or an empty list if none are found.
        """
        conn = await self._get_connection()
        results = []
        try:
            async with conn.execute("PRAGMA foreign_key_check;") as cursor:
                async for row in cursor:
                    results.append(dict(row))
        except aiosqlite.Error as e:
            print(f"Error checking foreign keys: {e}")
        return results

    # --- MemoryUnit New CRUD ---
    async def insert_memory_unit(self, unit_data: Dict[str, Any], embedding: Optional[List[float]] = None) -> str:
        conn = await self._get_connection()
        unit_id = unit_data["id"]
        async with conn.cursor() as cursor:
            sql_main_unit = """
            INSERT INTO memory_units
            (id, content, creation_time, end_time, source, metadata,
             last_visit, visit_count, never_delete, rank)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            values_main = (
                unit_id, unit_data["content"],
                _to_iso_datetime_str(_to_timestamp(unit_data.get("creation_time"))),
                _to_iso_datetime_str(_to_timestamp(unit_data.get("end_time"))),
                unit_data.get("source"), json.dumps(unit_data.get("metadata") or {}),
                unit_data.get("last_visit", 0), unit_data.get("visit_count", 0),
                int(unit_data.get("never_delete", False)), unit_data.get("rank", 0)
            )
            await cursor.execute(sql_main_unit, values_main)
            if self.use_sqlite_vec and embedding is not None:
                async with conn.execute("SELECT rowid FROM memory_units WHERE id = ?", (unit_id,)) as cursor:
                    rowid_row = await self._fetch_one_as_dict(cursor)
                if rowid_row:
                    await cursor.execute(
                        "INSERT INTO memory_units_vss (rowid, embedding) VALUES (?, ?)",
                        (rowid_row['rowid'], serialize_f32(embedding))
                    )
            if "parent_id" in unit_data and unit_data["parent_id"] is not None:
                await self._update_mu_hierarchy_edge_internal(unit_id, unit_data["parent_id"], overwrite=True,
                                                              conn_or_cursor=cursor)
            if "children_ids" in unit_data and unit_data["children_ids"]:
                await self._update_mu_children_edges_internal(unit_id, unit_data["children_ids"], overwrite=True,
                                                              conn_or_cursor=cursor)
            if "pre_id" in unit_data or "next_id" in unit_data:
                await self._update_mu_sequence_edge_internal(unit_id, unit_data.get("pre_id"), unit_data.get("next_id"),
                                                             overwrite=True, conn_or_cursor=cursor)
            if "group_id" in unit_data:
                await self._update_mu_group_membership_edges_internal(unit_id, unit_data.get("group_id"),
                                                                      unit_data.get("rank", 0), overwrite=True,
                                                                      conn_or_cursor=cursor)
        return unit_id

    async def update_memory_unit_data(self, unit_id: str, data_to_update: Dict[str, Any],
                                      embedding: Optional[List[float]] = None) -> bool:
        conn = await self._get_connection()
        if not data_to_update and embedding is None: return True
        async with conn.cursor() as cursor:
            fields, values = [], []
            for key, value in data_to_update.items():
                if key in ["id", "parent_id", "children_ids", "pre_id", "next_id", "group_id"]: continue
                if key in ["creation_time", "end_time"]:
                    values.append(_to_iso_datetime_str(_to_timestamp(value)))
                elif key == "metadata":
                    values.append(json.dumps(value or {}))
                elif key == "never_delete":
                    values.append(int(value))
                else:
                    values.append(value)
                fields.append(f"{key} = ?")

            if fields:
                sql = f"UPDATE memory_units SET {', '.join(fields)} WHERE id = ?"
                values.append(unit_id)
                cursor = await conn.execute(sql, tuple(values))
                if cursor.rowcount == 0: print(f"Warning: MU data update for {unit_id} affected 0 rows.")

            if self.use_sqlite_vec and embedding is not None:
                async with conn.execute("SELECT rowid FROM memory_units WHERE id = ?", (unit_id,)) as c:
                    rowid_row = await self._fetch_one_as_dict(c)
                if rowid_row:
                    await cursor.execute("INSERT OR REPLACE INTO memory_units_vss (rowid, embedding) VALUES (?, ?)",
                                       (rowid_row['rowid'], serialize_f32(embedding)))
                elif fields:
                    print(f"Warning: Cannot update embedding for non-existent MU {unit_id}.")
        return True

    async def _update_mu_hierarchy_edge_internal(self, child_unit_id: str, parent_unit_id: Optional[str],
                                                 overwrite: bool, conn_or_cursor: aiosqlite.Connection):
        if parent_unit_id is None:
            await conn_or_cursor.execute("DELETE FROM mu_hierarchy_edges WHERE child_unit_id = ?", (child_unit_id,))
            return

        if overwrite:
            await conn_or_cursor.execute(
                "INSERT OR REPLACE INTO mu_hierarchy_edges (child_unit_id, parent_unit_id) VALUES (?, ?)",
                (child_unit_id, parent_unit_id))
        else:
            await conn_or_cursor.execute(
                "INSERT OR IGNORE INTO mu_hierarchy_edges (child_unit_id, parent_unit_id) VALUES (?, ?)",
                (child_unit_id, parent_unit_id))

    async def update_mu_hierarchy_edge(self, child_unit_id: str, parent_unit_id: Optional[str], overwrite: bool = True):
        conn = await self._get_connection()
        async with conn.cursor() as cursor:
            await self._update_mu_hierarchy_edge_internal(child_unit_id, parent_unit_id, overwrite, cursor)

    async def _update_mu_children_edges_internal(self, parent_unit_id: str, children_ids: List[str], overwrite: bool,
                                                 conn_or_cursor: aiosqlite.Connection):
        if overwrite:
            await conn_or_cursor.execute("DELETE FROM mu_hierarchy_edges WHERE parent_unit_id = ?", (parent_unit_id,))

        if children_ids:
            await conn_or_cursor.executemany(
                "INSERT OR REPLACE INTO mu_hierarchy_edges (child_unit_id, parent_unit_id) VALUES (?, ?)",
                [(child_id, parent_unit_id) for child_id in children_ids]
            )

    async def update_mu_children_edges(self, parent_unit_id: str, children_ids: List[str], overwrite: bool = True):
        conn = await self._get_connection()
        async with conn.cursor() as cursor:
            await self._update_mu_children_edges_internal(parent_unit_id, children_ids, overwrite, cursor)

    async def _update_mu_sequence_edge_internal(self, unit_id: str, pre_unit_id: Optional[str],
                                                next_unit_id: Optional[str], overwrite: bool,
                                                conn_or_cursor: aiosqlite.Connection):
        pre_id_to_store = pre_unit_id if pre_unit_id else None
        next_id_to_store = next_unit_id if next_unit_id else None
        if overwrite:
            await conn_or_cursor.execute(
                "INSERT OR REPLACE INTO mu_sequence_edges (unit_id, pre_unit_id, next_unit_id) VALUES (?, ?, ?)",
                (unit_id, pre_id_to_store, next_id_to_store))
        else:
            await conn_or_cursor.execute(
                "INSERT OR IGNORE INTO mu_sequence_edges (unit_id, pre_unit_id, next_unit_id) VALUES (?, ?, ?)",
                (unit_id, pre_id_to_store, next_id_to_store))

    async def update_mu_sequence_edge(self, unit_id: str, pre_unit_id: Optional[str], next_unit_id: Optional[str],
                                      overwrite: bool = True):
        conn = await self._get_connection()
        async with conn.cursor() as cursor:
            await self._update_mu_sequence_edge_internal(unit_id, pre_unit_id, next_unit_id, overwrite, cursor)

    async def _update_mu_group_membership_edges_internal(self, unit_id: str,
                                                         group_id_val: Optional[Union[str, List[str]]], rank: int,
                                                         overwrite: bool, conn_or_cursor: aiosqlite.Connection,
                                                         table_name: Optional[str] = None,
                                                         id_col_name: Optional[str] = None,
                                                         ):
        if table_name and id_col_name:
            pass
        else:
            table_name = "mu_session_group_membership" if rank == 0 else "mu_ltm_group_membership"
            id_col_name = "session_id" if rank == 0 else "ltm_id"
        if overwrite:
            await conn_or_cursor.execute(f"DELETE FROM {table_name} WHERE unit_id = ?", (unit_id,))
        if group_id_val is None:
            if not overwrite: # If not overwriting all, but specifically setting to no group
                 await conn_or_cursor.execute(f"DELETE FROM {table_name} WHERE unit_id = ?", (unit_id,))
            return

        group_ids_to_add = [group_id_val] if isinstance(group_id_val, str) else (
            [g_id for g_id in group_id_val if isinstance(g_id, str)] if isinstance(group_id_val, list) else [])

        if group_ids_to_add:
            await conn_or_cursor.executemany(
                f"INSERT OR IGNORE INTO {table_name} (unit_id, {id_col_name}) VALUES (?, ?)",
                [(unit_id, g_id) for g_id in group_ids_to_add]
            )

    async def update_mu_group_membership_edges(self, unit_id: str,
                                                         group_id_val: Optional[Union[str, List[str]]], rank: int,
                                                         overwrite: bool):
        conn = await self._get_connection()
        async with conn.cursor() as cursor:
            await self._update_mu_group_membership_edges_internal(unit_id, group_id_val, rank, overwrite, cursor)

    async def update_mu_session_group_membership_edges(self, unit_id: str, session_ids: Union[str, List[str]],
                                                       overwrite: bool = True):
        conn = await self._get_connection()
        async with conn.cursor() as cursor:
            await self._update_mu_group_membership_edges_internal(unit_id, session_ids, 0, overwrite, cursor)

    async def update_mu_ltm_group_membership_edges(self, unit_id: str, ltm_ids: Union[str, List[str]],
                                                   overwrite: bool = True):
        conn = await self._get_connection()
        async with conn.cursor() as cursor:
            await self._update_mu_group_membership_edges_internal(unit_id, ltm_ids, 1, overwrite,
                                                                  cursor)

    async def _fetch_one_as_dict(self, cursor: aiosqlite.Cursor) -> Optional[Dict[str, Any]]:
        """Fetches a single row and returns it as a dictionary."""
        row_tuple = await cursor.fetchone()
        if not row_tuple:
            return None
        column_names = [desc[0] for desc in cursor.description]
        return dict(zip(column_names, row_tuple))

    async def _fetch_all_as_dicts(self, cursor: aiosqlite.Cursor) -> List[Dict[str, Any]]:
        """Fetches all rows and returns them as a list of dictionaries."""
        rows_tuple = await cursor.fetchall()
        if not rows_tuple:
            return []
        column_names = [desc[0] for desc in cursor.description]
        return [dict(zip(column_names, row)) for row in rows_tuple]

    async def _get_group_id_internal(self, unit_id: str, rank: int, conn_or_cursor: aiosqlite.Connection) -> Optional[Union[str, List[str]]]:
        """Internal helper to fetch group ID(s) for a unit."""
        group_ids_list = []
        table, col = ("mu_session_group_membership", "session_id") if int(rank) == 0 else ("mu_ltm_group_membership", "ltm_id")

        async with conn_or_cursor.execute(f"SELECT {col} FROM {table} WHERE unit_id = ?", (unit_id,)) as c:
            async for g_row in c:
                group_ids_list.append(g_row[col])

        if not group_ids_list:
            return None
        if int(rank) == 0 or int(rank) >= 2:
            return group_ids_list[0]
        else:
            return group_ids_list[0] if len(group_ids_list) == 1 and int(
                        rank) == 0 else group_ids_list

    async def get_group_id(self, unit_id: str) -> Optional[Union[str, List[str]]]:
        """Fetches the current group ID(s) for a specific MemoryUnit."""
        conn = await self._get_connection()
        async with conn.execute("SELECT rank FROM memory_units WHERE id = ?", (unit_id,)) as cursor:
            row = await self._fetch_one_as_dict(cursor)

        if not row:
            print(f"Warning: Unit {unit_id} not found when fetching group_id.")
            return None

        rank = row['rank']
        return await self._get_group_id_internal(unit_id, rank, conn)

    async def _row_to_memory_unit(self, row: aiosqlite.Row, conn_for_edges: aiosqlite.Connection,
                                  include_edges: bool = True) -> Optional[MemoryUnit]:
        if row is None: return None
        try:
            unit_id = row['id']
            rank = row['rank']
            unit_dict = {
                "id": unit_id, "content": row['content'],
                "creation_time": row['creation_time'], "end_time": row['end_time'],
                "source": row['source'], "metadata": json.loads(row['metadata']) if row['metadata'] else {},
                "last_visit": row['last_visit'], "visit_count": row['visit_count'],
                "never_delete": bool(row['never_delete']), "rank": rank,
                "parent_id": None, "children_ids": [], "pre_id": None, "next_id": None, "group_id": None
            }

            if include_edges:
                async with conn_for_edges.execute(
                        "SELECT parent_unit_id FROM mu_hierarchy_edges WHERE child_unit_id = ?", (unit_id,)) as c:
                    parent_row = await self._fetch_one_as_dict(c)
                    if parent_row: unit_dict["parent_id"] = parent_row["parent_unit_id"]
                children = []
                async with conn_for_edges.execute(
                        "SELECT child_unit_id FROM mu_hierarchy_edges WHERE parent_unit_id = ?", (unit_id,)) as c:
                    async for child_row in c: children.append(child_row["child_unit_id"])
                unit_dict["children_ids"] = children
                async with conn_for_edges.execute(
                        "SELECT pre_unit_id, next_unit_id FROM mu_sequence_edges WHERE unit_id = ?", (unit_id,)) as c:
                    seq_row = await self._fetch_one_as_dict(c)
                    if seq_row:
                        unit_dict["pre_id"] = seq_row["pre_unit_id"]
                        unit_dict["next_id"] = seq_row["next_unit_id"]
                group_id_val = await self._get_group_id_internal(unit_id, rank, conn_for_edges) if include_edges else None
                unit_dict['group_id'] = group_id_val
            return MemoryUnit.from_dict(unit_dict)
        except Exception as e:
            print(
                f"Error converting row to MemoryUnit (ID: {row['id'] if row else 'N/A'}, IncludeEdges: {include_edges}): {e}")
            return None

    async def delete_long_term_memory_record(self, ltm_id: str) -> int:
        """Deletes a single LongTermMemory record by its ID."""
        conn = await self._get_connection()
        async with conn.cursor() as cursor:
            cursor = await cursor.execute("DELETE FROM long_term_memories WHERE id = ?", (ltm_id,))
            await conn.commit()
        return cursor.rowcount

    async def load_memory_unit(self, unit_id: str, include_edges: bool = True) -> Optional[MemoryUnit]:
        conn = await self._get_connection()
        async with conn.execute("SELECT * FROM memory_units WHERE id = ?", (unit_id,)) as cursor:
            row = await self._fetch_one_as_dict(cursor)
        if row: return await self._row_to_memory_unit(row, conn, include_edges)
        return None

    async def load_memory_units(self, unit_ids: List[str], include_edges: bool = True) -> Dict[str, MemoryUnit]:
        conn = await self._get_connection()
        if not unit_ids: return {}
        placeholders = ', '.join('?' for _ in unit_ids)
        units = {}
        async with conn.execute(f"SELECT * FROM memory_units WHERE id IN ({placeholders})", tuple(unit_ids)) as cursor:
            rows = await self._fetch_all_as_dicts(cursor)
        for row in rows:
            unit = await self._row_to_memory_unit(row, conn, include_edges)
            if unit and unit.id: units[unit.id] = unit
        return units

    async def load_all_memory_units(self, include_edges: bool = True) -> Dict[str, MemoryUnit]:
        conn = await self._get_connection()
        units = {}
        async with conn.execute("SELECT * FROM memory_units") as cursor:
            rows = await self._fetch_all_as_dicts(cursor)
        for row in rows:
            unit = await self._row_to_memory_unit(row, conn, include_edges)
            if unit and unit.id: units[unit.id] = unit
        return units

    # --- SessionMemory New CRUD ---
    async def insert_session_memory(self, session_data: Dict[str, Any]) -> str:
        conn = await self._get_connection()
        session_id = session_data["id"]
        async with conn.cursor() as cursor:
            await cursor.execute("INSERT INTO session_memories (id, creation_time, end_time) VALUES (?, ?, ?)",
                               (session_id, _to_iso_datetime_str(_to_timestamp(session_data.get("creation_time"))),
                                _to_iso_datetime_str(_to_timestamp(session_data.get("end_time")))))
            if "memory_unit_ids" in session_data:
                await self._update_session_unit_membership_edges_internal(session_id, session_data["memory_unit_ids"],
                                                                          True, cursor)
        return session_id

    async def update_session_memory_data(self, session_id: str, data_to_update: Dict[str, Any]) -> bool:
        conn = await self._get_connection()
        if not data_to_update: return True
        fields, values = [], []
        for key, value in data_to_update.items():
            if key == "id" or key == "memory_unit_ids": continue
            if key in ["creation_time", "end_time"]:
                values.append(_to_iso_datetime_str(_to_timestamp(value)))
            else:
                values.append(value)
            fields.append(f"{key} = ?")
        if not fields: return True

        async with conn.cursor() as cursor:
            sql = f"UPDATE session_memories SET {', '.join(fields)} WHERE id = ?"
            values.append(session_id)
            cursor = await cursor.execute(sql, tuple(values))
            if cursor.rowcount == 0: print(f"Warning: Session data update for {session_id} affected 0 rows.")
        return True

    async def _update_session_unit_membership_edges_internal(self, session_id: str, unit_ids: Union[str, List[str]],
                                                             overwrite: bool, conn_or_cursor: aiosqlite.Connection):
        if overwrite:
            await conn_or_cursor.execute("DELETE FROM session_unit_membership WHERE session_id = ?", (session_id,))
        ids_to_add = [unit_ids] if isinstance(unit_ids, str) else (unit_ids or [])
        if ids_to_add:
            await conn_or_cursor.executemany(
                "INSERT OR IGNORE INTO session_unit_membership (session_id, unit_id) VALUES (?, ?)",
                [(session_id, u_id) for u_id in ids_to_add]
            )

    async def update_session_unit_membership_edges(self, session_id: str, unit_ids: Union[str, List[str]],
                                                   overwrite: bool = False):
        conn = await self._get_connection()
        async with conn.cursor() as cursor:
            await self._update_session_unit_membership_edges_internal(session_id, unit_ids, overwrite, cursor)

    async def _row_to_session_memory(self, row: aiosqlite.Row, conn_for_edges: aiosqlite.Connection,
                                     include_edges: bool = True) -> Optional[SessionMemory]:
        if row is None: return None
        session_id = row['id']
        session_dict = {"id": session_id, "creation_time": row['creation_time'], "end_time": row['end_time'],
                        "memory_unit_ids": []}
        if include_edges:
            try:
                ids = []
                async with conn_for_edges.execute("SELECT unit_id FROM session_unit_membership WHERE session_id = ?",
                                                  (session_id,)) as c:
                    async for mu_row in c: ids.append(mu_row["unit_id"])
                session_dict["memory_unit_ids"] = ids
            except Exception as e:
                print(f"Error fetching session unit memberships for {session_id}: {e}")
        try:
            return SessionMemory.from_dict(session_dict)
        except Exception as e:
            print(f"Error creating SessionMemory from dict for {session_id}: {e}"); return None

    async def load_session_memory(self, session_id: str, include_edges: bool = True) -> Optional[SessionMemory]:
        conn = await self._get_connection()
        async with conn.execute("SELECT * FROM session_memories WHERE id = ?", (session_id,)) as cursor:
            row = await self._fetch_one_as_dict(cursor)
        if row: return await self._row_to_session_memory(row, conn, include_edges)
        return None

    async def load_session_memories(self, session_ids: List[str], include_edges: bool = True) -> Dict[
        str, SessionMemory]:
        conn = await self._get_connection()
        if not session_ids: return {}
        placeholders = ', '.join('?' for _ in session_ids)
        sessions = {}
        async with conn.execute(f"SELECT * FROM session_memories WHERE id IN ({placeholders})",
                                tuple(session_ids)) as cursor:
            rows = await self._fetch_all_as_dicts(cursor)
        for row in rows:
            session = await self._row_to_session_memory(row, conn, include_edges)
            if session and session.id: sessions[session.id] = session
        return sessions

    async def load_all_session_memories(self, include_edges: bool = True) -> Dict[str, SessionMemory]:
        conn = await self._get_connection()
        sessions = {}
        async with conn.execute("SELECT * FROM session_memories") as cursor:
            rows = await self._fetch_all_as_dicts(cursor)
        for row in rows:
            session = await self._row_to_session_memory(row, conn, include_edges)
            if session and session.id: sessions[session.id] = session
        return sessions

    # --- LongTermMemory New CRUD ---
    async def insert_long_term_memory(self, ltm_data: Dict[str, Any]) -> str:
        conn = await self._get_connection()
        ltm_id = ltm_data["id"]
        async with conn.cursor() as cursor:
            await cursor.execute("""
                INSERT INTO long_term_memories (id, chatbot_id, visit_count, last_session_id, creation_time, end_time)
                VALUES (?, ?, ?, ?, ?, ?)""",
                               (ltm_id, ltm_data["chatbot_id"], ltm_data.get("visit_count", 0),
                                ltm_data.get("last_session_id"),
                                _to_iso_datetime_str(_to_timestamp(ltm_data.get("creation_time"))),
                                _to_iso_datetime_str(_to_timestamp(ltm_data.get("end_time")))))
            if "session_ids" in ltm_data:
                await self._update_ltm_session_membership_edges_internal(ltm_id, ltm_data["session_ids"], True, cursor)
            if "summary_unit_ids" in ltm_data:
                await self._update_ltm_summary_unit_membership_edges_internal(ltm_id, ltm_data["summary_unit_ids"],
                                                                              True, cursor)
        return ltm_id

    async def update_long_term_memory_data(self, ltm_id: str, data_to_update: Dict[str, Any]) -> bool:
        conn = await self._get_connection()
        if not data_to_update: return True
        fields, values = [], []
        for key, value in data_to_update.items():
            if key in ["id", "session_ids", "summary_unit_ids"]: continue
            if key in ["creation_time", "end_time"]:
                values.append(_to_iso_datetime_str(_to_timestamp(value)))
            else:
                values.append(value)
            fields.append(f"{key} = ?")
        if not fields: return True

        async with conn.cursor() as cursor:
            sql = f"UPDATE long_term_memories SET {', '.join(fields)} WHERE id = ?"
            values.append(ltm_id)
            cursor = await cursor.execute(sql, tuple(values))
            if cursor.rowcount == 0: print(f"Warning: LTM data update for {ltm_id} affected 0 rows.")
        return True

    async def _update_ltm_session_membership_edges_internal(self, ltm_id: str, session_ids: Union[str, List[str]],
                                                            overwrite: bool, conn_or_cursor: aiosqlite.Connection):
        if overwrite:
            await conn_or_cursor.execute("DELETE FROM ltm_session_membership WHERE ltm_id = ?", (ltm_id,))
        ids_to_add = [session_ids] if isinstance(session_ids, str) else (session_ids or [])
        if ids_to_add:
            await conn_or_cursor.executemany(
                "INSERT OR IGNORE INTO ltm_session_membership (ltm_id, session_id) VALUES (?, ?)",
                [(ltm_id, s_id) for s_id in ids_to_add])

    async def update_ltm_session_membership_edges(self, ltm_id: str, session_ids: Union[str, List[str]],
                                                  overwrite: bool = False):
        conn = await self._get_connection()
        async with conn.cursor() as cursor:
            await self._update_ltm_session_membership_edges_internal(ltm_id, session_ids, overwrite, cursor)

    async def _update_ltm_summary_unit_membership_edges_internal(self, ltm_id: str,
                                                                 summary_unit_ids: Union[str, List[str]],
                                                                 overwrite: bool, conn_or_cursor: aiosqlite.Connection):
        if overwrite:
            await conn_or_cursor.execute("DELETE FROM ltm_summary_unit_membership WHERE ltm_id = ?", (ltm_id,))
        ids_to_add = [summary_unit_ids] if isinstance(summary_unit_ids, str) else (summary_unit_ids or [])
        if ids_to_add:
            await conn_or_cursor.executemany(
                "INSERT OR IGNORE INTO ltm_summary_unit_membership (ltm_id, summary_unit_id) VALUES (?, ?)",
                [(ltm_id, su_id) for su_id in ids_to_add])

    async def update_ltm_summary_unit_membership_edges(self, ltm_id: str, summary_unit_ids: Union[str, List[str]],
                                                       overwrite: bool = False):
        conn = await self._get_connection()
        async with conn.cursor() as cursor:
            await self._update_ltm_summary_unit_membership_edges_internal(ltm_id, summary_unit_ids, overwrite, cursor)

    async def _row_to_long_term_memory(self, row: aiosqlite.Row, conn_for_edges: aiosqlite.Connection,
                                       include_edges: bool = True) -> Optional[LongTermMemory]:
        if row is None: return None
        print(row)
        ltm_id = row['id']
        ltm_dict = {"id": ltm_id, "chatbot_id": row["chatbot_id"], "visit_count": row["visit_count"],
                    "last_session_id": row["last_session_id"], "creation_time": row["creation_time"],
                    "end_time": row["end_time"], "session_ids": [], "summary_unit_ids": []}
        if include_edges:
            try:
                s_ids, su_ids = [], []
                async with conn_for_edges.execute("SELECT session_id FROM ltm_session_membership WHERE ltm_id = ?",
                                                  (ltm_id,)) as c:
                    async for s_row in c: s_ids.append(s_row["session_id"])
                ltm_dict["session_ids"] = s_ids
                async with conn_for_edges.execute(
                        "SELECT summary_unit_id FROM ltm_summary_unit_membership WHERE ltm_id = ?", (ltm_id,)) as c:
                    async for su_row in c: su_ids.append(su_row["summary_unit_id"])
                ltm_dict["summary_unit_ids"] = su_ids
            except Exception as e:
                print(f"Error fetching LTM relations for {ltm_id}: {e}")
        try:
            return LongTermMemory.from_dict(ltm_dict)
        except Exception as e:
            print(f"Error creating LongTermMemory from dict for {ltm_id}: {e}"); return None

    async def load_long_term_memory(self, ltm_id: str, chatbot_id: Optional[str] = None, include_edges: bool = True) -> \
    Optional[LongTermMemory]:
        conn = await self._get_connection()
        sql = "SELECT * FROM long_term_memories WHERE id = ?"
        params: List[Any] = [ltm_id]
        if chatbot_id: sql += " AND chatbot_id = ?"; params.append(chatbot_id)
        async with conn.execute(sql, tuple(params)) as cursor:
            row = await self._fetch_one_as_dict(cursor)
        if row: return await self._row_to_long_term_memory(row, conn, include_edges)
        return None

    async def load_all_long_term_memories_for_chatbot(self, chatbot_id: str, include_edges: bool = True) -> Dict[
        str, LongTermMemory]:
        conn = await self._get_connection()
        ltms = {}
        async with conn.execute("SELECT * FROM long_term_memories WHERE chatbot_id = ?", (chatbot_id,)) as cursor:
            rows = await self._fetch_all_as_dicts(cursor)
        for row in rows:
            ltm = await self._row_to_long_term_memory(row, conn, include_edges)
            if ltm and ltm.id: ltms[ltm.id] = ltm
        return ltms

    # --- Upsert Methods ---
    async def upsert_memory_units(self, units_with_embeddings: Iterable[Dict[str, Any]], commit: bool = True):
        conn = await self._get_connection()
        for unit_dict in units_with_embeddings:
            unit_id = unit_dict["id"]
            embedding = unit_dict.get('embedding') if self.use_sqlite_vec else None

            await conn.execute("""
                INSERT OR REPLACE INTO memory_units
                (id, content, creation_time, end_time, source, metadata,
                 last_visit, visit_count, never_delete, rank)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                               (unit_id, unit_dict["content"],
                                _to_iso_datetime_str(_to_timestamp(unit_dict.get("creation_time"))),
                                _to_iso_datetime_str(_to_timestamp(unit_dict.get("end_time"))),
                                unit_dict.get("source"), json.dumps(unit_dict.get("metadata") or {}),
                                unit_dict.get("last_visit", 0), unit_dict.get("visit_count", 0),
                                int(unit_dict.get("never_delete", False)), unit_dict.get("rank", 0))
                               )

            if self.use_sqlite_vec and embedding is not None:
                async with conn.execute("SELECT rowid FROM memory_units WHERE id = ?", (unit_id,)) as c:
                    rowid_row = await self._fetch_one_as_dict(c)
                if rowid_row:
                    await conn.execute("INSERT OR REPLACE INTO memory_units_vss (rowid, embedding) VALUES (?, ?)",
                                       (rowid_row['rowid'], serialize_f32(embedding)))

        # for unit_dict in units_with_embeddings:
        #     unit_id = unit_dict["id"]
        #     if "parent_id" in unit_dict:
        #         await self._update_mu_hierarchy_edge_internal(unit_id, unit_dict["parent_id"], True, conn)
        #     if "children_ids" in unit_dict:
        #         await self._update_mu_children_edges_internal(unit_id, unit_dict["children_ids"], True,
        #                                                       conn)
        #     if "pre_id" in unit_dict or "next_id" in unit_dict:
        #         await self._update_mu_sequence_edge_internal(unit_id, unit_dict.get("pre_id"),
        #                                                      unit_dict.get("next_id"), True, conn)
            # if "group_id" in unit_dict and "rank" in unit_dict:
            #     await self._update_mu_group_membership_edges_internal(unit_id, unit_dict.get("group_id"),
            #                                                           unit_dict.get("rank", 0), True, conn)
        if commit:
             await conn.commit()

    async def upsert_session_memories(self, sessions_data: Iterable[Dict[str, Any]], commit: bool = True):
        conn = await self._get_connection()
        for session_dict in sessions_data:
            session_id = session_dict["id"]
            await conn.execute(
                "INSERT OR REPLACE INTO session_memories (id, creation_time, end_time) VALUES (?, ?, ?)",
                (session_id, _to_iso_datetime_str(_to_timestamp(session_dict.get("creation_time"))),
                 _to_iso_datetime_str(_to_timestamp(session_dict.get("end_time")))))
            await self._update_session_unit_membership_edges_internal(session_id,
                                                                      session_dict.get("memory_unit_ids", []), True,
                                                                      conn)
        if commit:
             await conn.commit()

    async def upsert_long_term_memories(self, ltms_data: Iterable[Dict[str, Any]], commit: bool = True):
        conn = await self._get_connection()
        for ltm_dict in ltms_data:
            ltm_id = ltm_dict["id"]
            await conn.execute("""
                INSERT OR REPLACE INTO long_term_memories
                (id, chatbot_id, visit_count, last_session_id, creation_time, end_time)
                VALUES (?, ?, ?, ?, ?, ?)""",
                               (ltm_id, ltm_dict["chatbot_id"], ltm_dict.get("visit_count", 0),
                                ltm_dict.get("last_session_id"),
                                _to_iso_datetime_str(_to_timestamp(ltm_dict.get("creation_time"))),
                                _to_iso_datetime_str(_to_timestamp(ltm_dict.get("end_time")))))
            await self._update_ltm_session_membership_edges_internal(ltm_id, ltm_dict.get("session_ids", []), True,
                                                                     conn)
            await self._update_ltm_summary_unit_membership_edges_internal(ltm_id,
                                                                          ltm_dict.get("summary_unit_ids", []),
                                                                          True, conn)
        if commit:
             await conn.commit()

    # --- Deletion Methods ---
    async def delete_memory_units(self, unit_ids: List[str]) -> int:
        conn = await self._get_connection()
        if not unit_ids: return 0
        placeholders = ', '.join('?' * len(unit_ids))
        async with conn.cursor() as cursor:
            if self.use_sqlite_vec:
                async with conn.execute(f"SELECT rowid FROM memory_units WHERE id IN ({placeholders})",
                                        tuple(unit_ids)) as c:
                    rowids_to_delete_vss = [row[0] for row in await self._fetch_all_as_dicts(c)]
                if rowids_to_delete_vss:
                    vss_placeholders = ', '.join('?' * len(rowids_to_delete_vss))
                    await conn.execute(f"DELETE FROM memory_units_vss WHERE rowid IN ({vss_placeholders})",
                                       tuple(rowids_to_delete_vss))
            cursor = await conn.execute(f"DELETE FROM memory_units WHERE id IN ({placeholders})", tuple(unit_ids))
            deleted_count = cursor.rowcount
        return deleted_count

    async def delete_session_memories(self, session_ids: List[str]) -> int:
        conn = await self._get_connection()
        if not session_ids: return 0
        placeholders = ', '.join('?' * len(session_ids))
        async with conn.cursor() as cursor:
            cursor = await cursor.execute(f"DELETE FROM session_memories WHERE id IN ({placeholders})",
                                        tuple(session_ids))
        return cursor.rowcount

    async def delete_long_term_memories(self, ltm_ids: List[str]) -> int:
        conn = await self._get_connection()
        if not ltm_ids: return 0
        placeholders = ', '.join('?' * len(ltm_ids))
        async with conn.cursor() as cursor:
            cursor = await cursor.execute(f"DELETE FROM long_term_memories WHERE id IN ({placeholders})", tuple(ltm_ids))
        return cursor.rowcount

    # --- Vector Search and Utility ---
    async def search_vectors_sqlite_vec(self, query_vector: List[float], k: int, expr: Optional[str] = None,
                                        search_range: Optional[Tuple[Optional[float], Optional[float]]] = None) -> List[
        Tuple[str, float]]:
        if not self.use_sqlite_vec: return []
        if not self.embedding_dim or len(query_vector) != self.embedding_dim:
            raise ValueError(f"Query vector dim ({len(query_vector)}) != expected dim ({self.embedding_dim})")
        conn = await self._get_connection()
        search_sql = f"""
            SELECT mu.id, vss.distance
            FROM memory_units_vss vss JOIN memory_units mu ON vss.rowid = mu.rowid
            WHERE vss_search(vss.embedding, ?)
        """  # This seems to be the pattern from sqlite-vec examples
        params: List[Any] = [serialize_f32(query_vector)]
        if expr: search_sql += f" AND ({expr})"
        search_sql += f" ORDER BY vss.distance ASC LIMIT ?"
        params.append(k)
        results = []
        min_score_threshold, max_score_threshold = search_range if search_range else (None, None)
        async with conn.execute(search_sql, tuple(params)) as cursor:
            rows = await self._fetch_all_as_dicts(cursor)
        for row in rows:
            distance = row['distance']
            similarity_score = 1.0 - (distance * distance) / 2.0 if distance is not None else 0.0
            if (min_score_threshold is None or similarity_score >= min_score_threshold) and \
                    (max_score_threshold is None or similarity_score <= max_score_threshold):
                results.append((row['id'], similarity_score))
        return results

    async def load_all_memory_unit_ids(self) -> List[str]:
        conn = await self._get_connection()
        ids = []
        async with conn.execute("SELECT id FROM memory_units") as cursor:
            async for row in cursor: ids.append(row['id'])
        return ids

    async def load_all_memory_units_with_embeddings(self, include_edges: bool = True) -> List[
        Tuple[MemoryUnit, List[float]]]:
        if not self.use_sqlite_vec: return []
        conn = await self._get_connection()
        units_with_embeddings = []
        async with conn.execute(
                "SELECT mu.*, vss.embedding FROM memory_units mu JOIN memory_units_vss vss ON mu.rowid = vss.rowid") as cursor:
            rows = await self._fetch_all_as_dicts(cursor)
        for row in rows:
            unit = await self._row_to_memory_unit(row, conn, include_edges)
            if unit:
                embedding = deserialize_f32(row['embedding']) if row['embedding'] else []
                units_with_embeddings.append((unit, embedding))
        return units_with_embeddings

    async def get_embedding_for_unit(self, unit_id: str) -> Optional[List[float]]:
        if not self.use_sqlite_vec: return None
        conn = await self._get_connection()
        async with conn.execute(
                "SELECT vss.embedding FROM memory_units mu JOIN memory_units_vss vss ON mu.rowid = vss.rowid WHERE mu.id = ?",
                (unit_id,)) as cursor:
            row = await self._fetch_one_as_dict(cursor)
        if row and row['embedding']: return deserialize_f32(row['embedding'])
        return None

    async def get_embeddings_for_units(self, unit_ids: List[str]) -> Dict[str, List[float]]:
        if not self.use_sqlite_vec or not unit_ids: return {}
        conn = await self._get_connection()
        placeholders = ', '.join('?' * len(unit_ids))
        embeddings_map = {}
        async with conn.execute(
                f"SELECT mu.id, vss.embedding FROM memory_units mu JOIN memory_units_vss vss ON mu.rowid = vss.rowid WHERE mu.id IN ({placeholders})",
                tuple(unit_ids)) as cursor:
            rows = await self._fetch_all_as_dicts(cursor)
        for row in rows:
            if row['embedding']: embeddings_map[row['id']] = deserialize_f32(row['embedding'])
        return embeddings_map