import os
import datetime
import itertools
import operator
import re
import sqlite3
from collections import defaultdict, deque
from typing import (Any, Dict, Iterable, List, Optional, Tuple, Union, Set)
from uuid import uuid4
import numpy as np
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
import json 

from MemForest.memory.long_term_memory import LongTermMemory
from MemForest.memory.memory_unit import MemoryUnit
from MemForest.memory.session_memory import SessionMemory
from MemForest.manager.forgetting import forget_memories
from MemForest.manager.summarizing import (
    summarize_long_term_memory, summarize_session)
from MemForest.utils.embedding_handler import EmbeddingHandler
from MemForest.persistence.vector_store_handler import VectorStoreHandler
from MemForest.utils.helper import link_memory_units
from MemForest.persistence import sqlite_handler
from MemForest.persistence import json_handler

# --- Constants ---

DEFAULT_VISIT_UPDATE_INTERVAL = 20
DEFAULT_SAVING_INTERVAL = 20
DEFAULT_MAX_MILVUS_ENTITIES = 20000  # Default threshold for auto-forgetting
DEFAULT_FORGET_PERCENTAGE = 0.10
DEFAULT_MAX_GROUP_SIZE = 60
DEFAULT_MAX_TOKEN_COUNT = 2000

class MemorySystem:
    """
    Manages chatbot memory using SQLite or Partitioned JSON for primary persistence,
    and Milvus for vectors. Includes caching, auto-forgetting, summarization.
    Operates on data relevant to the initialized ltm_id.
    """

    def __init__(self,
                 chatbot_id: str,
                 ltm_id: str,
                 embedding_handler: EmbeddingHandler,
                 llm: Optional[BaseChatModel] = None,
                 milvus_config: Optional[Dict[str, Any]] = None,
                 base_path: Optional[str] = None,
                 persistence_mode: str = 'sqlite',  # 'sqlite' or 'json'
                 max_context_length: int = 10,
                 visit_update_interval: int = DEFAULT_VISIT_UPDATE_INTERVAL,
                 saving_interval: int = DEFAULT_SAVING_INTERVAL,
                 max_milvus_entities: int = DEFAULT_MAX_MILVUS_ENTITIES,
                 forget_percentage: float = DEFAULT_FORGET_PERCENTAGE,
                 max_group_size: int = DEFAULT_MAX_GROUP_SIZE,
                 max_token_count: int = DEFAULT_MAX_TOKEN_COUNT
                 ):
        self.chatbot_id: str = chatbot_id
        self.ltm_id: str = ltm_id  # Primary LTM this instance manages
        self.llm: Optional[BaseChatModel] = llm
        self.embedding_handler: EmbeddingHandler = embedding_handler
        self.milvus_config: Optional[Dict[str, Any]] = milvus_config
        self.storage_base_path: str = base_path if base_path is not None else os.path.join(os.getcwd(), "memory_storage")
        self.persistence_mode: str = persistence_mode.lower()
        self.vector_store: Optional[VectorStoreHandler] = None

        self.visit_update_interval: int = visit_update_interval
        self.saving_interval: int = saving_interval
        self.max_milvus_entities: int = max_milvus_entities
        self.forget_percentage: float = forget_percentage
        self.max_group_size: int = max_group_size
        self.max_token_count: int = max_token_count

        # --- Tracking Changes for Batch Saving (Scoped to current LTM) ---
        self._updated_memory_unit_ids: set[str] = set()
        self._updated_session_memory_ids: set[str] = set()
        self._updated_ltm_ids: set[str] = set()  # Should usually just contain self.ltm_id
        self._deleted_memory_unit_ids: set[str] = set()
        self._deleted_session_memory_ids: set[str] = set()

        # Validate persistence mode
        if self.persistence_mode not in ['sqlite', 'json']:
            raise ValueError(f"Invalid persistence_mode: {persistence_mode}. Choose 'sqlite' or 'json'.")


        # --- Persistence Setup ---
        self.sql_conn: Optional[sqlite3.Connection] = None
        if self.persistence_mode == 'sqlite':
            self._initialize_sqlite(self.storage_base_path)
        else:  # json mode
            os.makedirs(json_handler._get_chatbot_path(self.chatbot_id, self.storage_base_path), exist_ok=True)

        # --- Milvus Setup ---
        if milvus_config:
            try:
                # Ensure collection name includes primary LTM ID for isolation if needed
                self.vector_store = VectorStoreHandler(
                    chatbot_id=self.chatbot_id,
                    long_term_memory_id=self.ltm_id,  # Link Milvus store to primary LTM
                    **milvus_config
                )
                print(f"Milvus connection established for collection: {self.vector_store.collection_name}")
            except Exception as e:
                print(f"Warning: Failed to initialize VectorStoreHandler. Error: {e}")
                self.vector_store = None
        else:
            print(f"Warning: No Milvus config provided.")

        # --- In-Memory Caches and State (Scoped to current LTM) ---
        # Caches hold data *only* relevant to self.ltm_id
        self.memory_units_cache: Dict[str, MemoryUnit] = {}
        self.session_memories_cache: Dict[str, SessionMemory] = {}
        self.long_term_memory: Optional[LongTermMemory] = None  # The primary LTM object
        # Map for JSON mode (only units belonging to sessions in self.long_term_memory)
        # self.unit_id_to_session_id_map: Dict[str, str] = {}

        # Load initial state relevant to self.ltm_id
        self._load_initial_state()

        # --- Internal Counters and Queues ---
        self._internal_visit_counter: int = 0
        self._staged_updates_count: int = 0
        self._stm_enabled: bool = False
        self._stm_capacity: int = 0
        self._short_term_memory_ids: deque[str] = deque()
        self._short_term_memory_embeddings: Dict[str, np.ndarray] = {}
        self._short_term_memory_units: Dict[str, MemoryUnit] = {}  # STM units live only in RAM
        self._stm_id_set: set[str] = set()
        self._max_context_length: int = max_context_length
        self.context: deque[MemoryUnit] = deque()  # Context units live only in RAM

        self.current_round: int = self.long_term_memory.visit_count if self.long_term_memory else 0

        # External connection state (Unchanged conceptually)
        self.external_chatbot_id: str = ""
        self.external_ltm_id: str = ""
        self.external_vector_store: Optional[VectorStoreHandler] = None
        self.external_long_term_memory: Optional[LongTermMemory] = None
        self.external_session_memories_cache: Dict[str, SessionMemory] = {} 
        self.external_memory_units_cache: Dict[str, MemoryUnit] = {}  



        # JSON mode specific staging (cleared after flush)
        # self._staged_units_to_upsert_json: Dict[str, MemoryUnit] = {} # unit_id -> unit_obj
        # self._staged_units_to_delete_json: Dict[str, MemoryUnit] = {}

        # Visit tracking (Unchanged conceptually)
        self._visited_unit_counts: Dict[str, int] = defaultdict(int)
        self._unit_last_visit_round: Dict[str, int] = defaultdict(int)

        # Session management
        self._current_session_id: str = ""
        self._start_session()  # Start or resume last session from loaded LTM
        # Ensure LTM is marked updated if we assigned a new last_session_id

        print(f"MemorySystem for {self.chatbot_id}/{self.ltm_id} initialized.")
        print(f"Using Persistence Mode: {self.persistence_mode}")
        print(f"Milvus Enabled: {self.vector_store is not None}")
        print(f"Auto-forget threshold: {self.max_milvus_entities if self.vector_store else 'N/A'}")

    def _initialize_sqlite(self, storage_base_path: str):
        """Initializes SQLite DB and connection."""
        if self.persistence_mode != 'sqlite': return
        try:
            # Use handler functions
            sqlite_handler.initialize_db(self.chatbot_id, storage_base_path)
            self.sql_conn = sqlite_handler.get_connection(self.chatbot_id, storage_base_path)
            if not self.sql_conn:
                raise ConnectionError(f"Failed to establish SQLite connection for chatbot {self.chatbot_id}")
            print(f"SQLite connection established for chatbot {self.chatbot_id}.")
        except Exception as e:
            print(f"FATAL: Could not initialize SQLite: {e}")
            raise

    def _load_initial_state(self):
        """Loads the primary LTM and its associated session metadata into cache."""
        print(f"Loading initial state for LTM {self.ltm_id} ({self.persistence_mode} mode)...")
        self.memory_units_cache.clear()  # Ensure caches are clean before load
        self.session_memories_cache.clear()
        # self.unit_id_to_session_id_map.clear()
        self.long_term_memory = None

        if self.persistence_mode == 'sqlite':
            if not self.sql_conn:
                self._initialize_sqlite(self.storage_base_path)
                if not self.sql_conn:
                    raise ConnectionError("SQLite connection required but failed to initialize.")
            try:
                self.long_term_memory = sqlite_handler.load_long_term_memory(self.sql_conn, self.ltm_id,
                                                                             self.chatbot_id)
                if self.long_term_memory and self.long_term_memory.session_ids:
                    self.session_memories_cache = sqlite_handler.load_session_memories(self.sql_conn,
                                                                                       self.long_term_memory.session_ids)
                if self.session_memories_cache:
                    self.memory_units_cache = self._load_all_units_for_ltm()
            except Exception as e:
                raise ValueError(f"Error loading initial state from SQLite: {e}")

        elif self.persistence_mode == 'json':
            try:
                ltm_data_dict = json_handler.load_long_term_memories_json(self.chatbot_id, self.storage_base_path)
                ltm_data = ltm_data_dict.get(self.ltm_id)
                if ltm_data:
                    self.long_term_memory = LongTermMemory.from_dict(ltm_data)

                if self.long_term_memory and self.long_term_memory.session_ids:
                    sm_data_dict = json_handler.load_session_memories_json(self.chatbot_id, self.storage_base_path)
                    self.session_memories_cache = {
                        sm_id: SessionMemory.from_dict(data)
                        for sm_id, data in sm_data_dict.items()
                        if sm_id in self.long_term_memory.session_ids  # Only load sessions in this LTM
                    }
                if self.session_memories_cache:
                    self.memory_units_cache = self._load_all_units_for_ltm()
            except Exception as e:
                raise ValueError(f"Error loading initial state from JSON: {e}")

        # --- Post Loading Steps ---
        if not self.long_term_memory:
            # print(f"LTM {self.ltm_id} not found or failed to load. Creating new LTM state.")
            self.long_term_memory = LongTermMemory(chatbot_id=self.chatbot_id, ltm_id=self.ltm_id)
            self._updated_ltm_ids.add(self.long_term_memory.id)  # Mark new LTM for saving
            self.session_memories_cache = {}  # No sessions in a new LTM
        # else:
        #      print(f"Loaded LTM {self.ltm_id} with {len(self.long_term_memory.session_ids)} sessions.")
        #      print(f"Loaded {len(self.session_memories_cache)} session objects into cache.")

        # # Build unit->session map if using JSON and sessions were loaded
        # if self.persistence_mode == 'json' and self.session_memories_cache:
        #     for sm_id, sm_obj in self.session_memories_cache.items():
        #          for unit_id in sm_obj.memory_unit_ids:
        #               # if unit_id in self.unit_id_to_session_id_map and self.unit_id_to_session_id_map[unit_id] != sm_id:
        #               #      print(f"Warning: Unit {unit_id} found in multiple sessions ({self.unit_id_to_session_id_map[unit_id]} and {sm_id}). Using {sm_id}.")
        #               self.unit_id_to_session_id_map[unit_id] = sm_id
        #     # print(f"Built UnitID->SessionID map with {len(self.unit_id_to_session_id_map)} entries.")

    def _close_sqlite(self):
        """Closes the SQLite connection."""
        if self.persistence_mode == 'sqlite' and self.sql_conn:
            try:
                self.sql_conn.commit()
                self.sql_conn.close()
                self.sql_conn = None
                print("SQLite connection closed.")
            except sqlite3.Error as e:
                print(f"Error closing SQLite connection: {e}")

    # def _load_from_json_or_create_new(self):
    #     """Loads state from JSON files or creates new state if files don't exist."""
    #     try:
    #         print("Attempting to load state from JSON files...")
    #         mus = self._load_memory_units_from_sql()
    #         sms = self._load_session_memories_from_sql()
    #         ltm = self._load_ltm_from_sql()
    #         # Check if loading actually found data
    #         if not ltm and not sms and not mus:
    #             print("No existing SQL data found. Creating new state.")
    #             self.memory_units_cache = {}
    #             self.session_memories_cache = {}
    #             self.long_term_memory = LongTermMemory(chatbot_id=self.chatbot_id, ltm_id=self.ltm_id)
    #         else:
    #             print("Loaded state from SQL.")
    #             self.long_term_memory = ltm
    #             # Filter session memories to only those in the current LTM
    #             self.session_memories_cache = {sid: sm for sid, sm in sms.items() if sid in self.long_term_memory.session_ids}
    #             mus_in_ltm = []
    #             for _,sm in self.session_memories_cache.items():
    #                 mus_in_ltm.extend(sm.memory_unit_ids)
    #                 mus_in_ltm.append(sm.id)
    #             mus_in_ltm.append(ltm.id)
    #             mus_in_ltm.extend(ltm.summary_unit_ids)
    #             mus_in_ltm = set(mus_in_ltm)
    #             self.memory_units_cache = {mid: mu for mid,mu in mus.items() if mid in mus_in_ltm}
    #
    #     except Exception as e:
    #         print(f"Error loading from SQL: {e}. Creating new state.")
    #         self.memory_units_cache = {}
    #         self.session_memories_cache = {}
    #         self.long_term_memory = LongTermMemory(chatbot_id=self.chatbot_id, ltm_id=self.ltm_id)

    # def _load_from_sql_or_create_new(self):
    #     """Loads state from JSON files or creates new state if files don't exist."""
    #     try:
    #         print("Attempting to load state from JSON files...")
    #         mus = self._load_all_memory_units_from_json()
    #         sms = self._load_all_session_memories_from_json()
    #         ltm = self._load_or_create_ltm_from_json()
    #
    #         # Check if loading actually found data
    #         if not ltm and not sms and not mus:
    #             print("No existing JSON data found. Creating new state.")
    #             self.memory_units_cache = {}
    #             self.session_memories_cache = {}
    #             self.long_term_memory = LongTermMemory(chatbot_id=self.chatbot_id, ltm_id=self.ltm_id)
    #         else:
    #             print("Loaded state from JSON.")
    #             self.long_term_memory = ltm
    #             # Filter session memories to only those in the current LTM
    #             self.session_memories_cache = {sid: sm for sid, sm in sms.items() if sid in self.long_term_memory.session_ids}
    #             mus_in_ltm = []
    #             for _,sm in self.session_memories_cache.items():
    #                 mus_in_ltm.extend(sm.memory_unit_ids)
    #             mus_in_ltm = set(mus_in_ltm)
    #             self.memory_units_cache = {mid: mu for mid,mu in mus.items() if mid in mus_in_ltm}
    #
    #     except Exception as e:
    #         print(f"Error loading from JSON: {e}. Creating new state.")
    #         self.memory_units_cache = {}
    #         self.session_memories_cache = {}
    #         self.long_term_memory = LongTermMemory(chatbot_id=self.chatbot_id, ltm_id=self.ltm_id)

    # --- SQLite Helper Methods ---

    # def _sqlite_db_has_data(self) -> bool:
    #     """Checks if the LTM table in SQLite has an entry for the current LTM."""
    #     if not self.sql_conn:
    #         return False
    #     try:
    #         cursor = self.sql_conn.cursor()
    #         cursor.execute("SELECT 1 FROM long_term_memories")
    #         return cursor.fetchone() is not None
    #     except sqlite3.Error as e:
    #         print(f"Error checking SQLite data: {e}")
    #         return False

    # def _load_memory_units_from_sql(self) -> Dict[str, MemoryUnit]:
    #     """Loads all memory units for the current chatbot/LTM from SQLite."""
    #     if not self.sql_conn: return {}
    #     units = {}
    #     try:
    #         cursor = self.sql_conn.cursor()
    #         cursor.execute("SELECT * FROM memory_units")
    #         rows = cursor.fetchall()
    #         colnames = [desc[0] for desc in cursor.description]
    #         for row in rows:
    #             data = dict(zip(colnames, row))
    #             # Convert JSON strings back to dict/list
    #             data['metadata'] = json.loads(data.get('metadata', '{}') or '{}')
    #             data['children_ids'] = json.loads(data.get('children_ids', '[]') or '[]')
    #             # Convert ISO strings back to datetime
    #             data['creation_time'] = from_isoformat(data.get('creation_time'))
    #             data['end_time'] = from_isoformat(data.get('end_time'))
    #             # Convert integer bool back
    #             data['never_delete'] = bool(data.get('never_delete', 0))
    #             # Create object (remove chatbot_id/ltm_id if not in MemoryUnit constructor)
    #             mu_id = data['id']
    #             units[mu_id] = MemoryUnit.from_dict(data)
    #     except sqlite3.Error as e:
    #         print(f"Error loading memory units from SQLite: {e}")
    #     except json.JSONDecodeError as e:
    #         print(f"Error decoding JSON from SQLite memory_units table: {e}")
    #     return units

    # def _load_session_memories_from_sql(self) -> Dict[str, SessionMemory]:
    #     """Loads all session memories for the current chatbot/LTM from SQLite."""
    #     if not self.sql_conn: return {}
    #     sessions = {}
    #     try:
    #         cursor = self.sql_conn.cursor()
    #         cursor.execute("SELECT * FROM session_memories")
    #         rows = cursor.fetchall()
    #         colnames = [desc[0] for desc in cursor.description]
    #         for row in rows:
    #             data = dict(zip(colnames, row))
    #             data['memory_unit_ids'] = json.loads(data.get('memory_unit_ids', '[]') or '[]')
    #             data['creation_time'] = from_isoformat(data.get('creation_time'))
    #             data['end_time'] = from_isoformat(data.get('end_time'))
    #             sm_id = data['id']
    #             sessions[sm_id] = SessionMemory.from_dict(data)
    #     except sqlite3.Error as e:
    #         print(f"Error loading session memories from SQLite: {e}")
    #     except json.JSONDecodeError as e:
    #         print(f"Error decoding JSON from SQLite session_memories table: {e}")
    #     return sessions
    #
    # def _load_ltm_from_sql(self) -> Optional[LongTermMemory]:
    #     """Loads the specific LTM object for the current chatbot/LTM from SQLite."""
    #     if not self.sql_conn: return None
    #     try:
    #         cursor = self.sql_conn.cursor()
    #         cursor.execute("SELECT * FROM long_term_memories WHERE id = ? AND chatbot_id = ?",
    #                        (self.ltm_id, self.chatbot_id))
    #         row = cursor.fetchone()
    #         if row:
    #             colnames = [desc[0] for desc in cursor.description]
    #             data = dict(zip(colnames, row))
    #             data['session_ids'] = json.loads(data.get('session_ids', '[]') or '[]')
    #             data['creation_time'] = from_isoformat(data.get('creation_time'))
    #             data['end_time'] = from_isoformat(data.get('end_time'))
    #             ltm_id = data.pop('id')
    #             return LongTermMemory.from_dict(data)
    #         else:
    #             return None  # Not found
    #     except sqlite3.Error as e:
    #         print(f"Error loading LTM from SQLite: {e}")
    #     except json.JSONDecodeError as e:
    #         print(f"Error decoding JSON from SQLite long_term_memories table: {e}")
    #     return None
    #
    # def _upsert_memory_unit_to_sql(self, unit: MemoryUnit):
    #     """Inserts or updates a MemoryUnit in the SQLite table."""
    #     if not self.sql_conn: return
    #     try:
    #         cursor = self.sql_conn.cursor()
    #         data = (
    #             unit.id, unit.parent_id, unit.content,
    #             to_isoformat(unit.creation_time), to_isoformat(unit.end_time),
    #             unit.source, json.dumps(unit.metadata or {}), unit.last_visit,
    #             unit.visit_count, int(unit.never_delete), json.dumps(unit.children_ids or []),
    #             unit.rank, unit.pre_id, unit.next_id
    #         )
    #         # Use INSERT OR REPLACE for upsert behavior
    #         cursor.execute("""
    #         INSERT OR REPLACE INTO memory_units
    #         (id, parent_id, content, creation_time, end_time, source, metadata,
    #          last_visit, visit_count, never_delete, children_ids, rank, pre_id, next_id)
    #         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    #         """, data)
    #         # Commit happens in _flush_cache
    #     except (sqlite3.Error, TypeError, json.JSONDecodeError) as e:
    #         print(f"Error upserting MemoryUnit {unit.id} to SQLite: {e}")
    #
    # def _upsert_session_memory_to_sql(self, session: SessionMemory):
    #     """Inserts or updates a SessionMemory in the SQLite table."""
    #     if not self.sql_conn: return
    #     try:
    #         cursor = self.sql_conn.cursor()
    #         data = (
    #             session.id, json.dumps(session.memory_unit_ids or []),
    #             to_isoformat(session.creation_time), to_isoformat(session.end_time)
    #         )
    #         cursor.execute("""
    #         INSERT OR REPLACE INTO session_memories
    #         (id, memory_unit_ids, creation_time, end_time)
    #         VALUES (?, ?, ?, ?)
    #         """, data)
    #     except (sqlite3.Error, TypeError, json.JSONDecodeError) as e:
    #         print(f"Error upserting SessionMemory {session.id} to SQLite: {e}")
    #
    # def _upsert_ltm_to_sql(self, ltm: LongTermMemory):
    #     """Inserts or updates the LongTermMemory object in the SQLite table."""
    #     if not self.sql_conn: return
    #     try:
    #         cursor = self.sql_conn.cursor()
    #         data = (
    #             ltm.id, ltm.chatbot_id, ltm.visit_count, json.dumps(ltm.session_ids or []),
    #             ltm.last_session_id, to_isoformat(ltm.creation_time), to_isoformat(ltm.end_time)
    #         )
    #         cursor.execute("""
    #         INSERT OR REPLACE INTO long_term_memories
    #         (id, chatbot_id, visit_count, session_ids, last_session_id, creation_time, end_time)
    #         VALUES (?, ?, ?, ?, ?, ?, ?)
    #         """, data)
    #     except (sqlite3.Error, TypeError, json.JSONDecodeError) as e:
    #         print(f"Error upserting LongTermMemory {ltm.id} to SQLite: {e}")

    # def _delete_memory_units_from_sql(self, unit_ids: List[str]):
    #     """Deletes multiple MemoryUnits from SQLite by ID."""
    #     if not self.sql_conn or not unit_ids: return
    #     try:
    #         cursor = self.sql_conn.cursor()
    #         # Prepare placeholders for parameterized query
    #         placeholders = ', '.join('?' for _ in unit_ids)
    #         sql = f"DELETE FROM memory_units WHERE id IN ({placeholders})"
    #         params = tuple(unit_ids)
    #         cursor.execute(sql, params)
    #         print(f"Deleted {cursor.rowcount} memory units from SQLite.")
    #     except sqlite3.Error as e:
    #         print(f"Error deleting memory units {unit_ids} from SQLite: {e}")
    #
    # def _delete_session_from_sql(self, session_id: str):
    #     """Deletes a SessionMemory from SQLite by ID."""
    #     if not self.sql_conn: return
    #     try:
    #         cursor = self.sql_conn.cursor()
    #         cursor.execute("DELETE FROM session_memories WHERE id = ?",
    #                        (session_id))
    #         print(f"Deleted session {session_id} from SQLite session_memories table (if existed).")
    #     except sqlite3.Error as e:
    #         print(f"Error deleting session {session_id} from SQLite: {e}")

    # --- JSON Loading Wrappers (used in fallback) ---
    # def _load_all_memory_units_from_json(self) -> Dict[str, MemoryUnit]:
    #      mu_data = json_handler.load_mu_json(self.chatbot_id, self.storage_base_path) # Use renamed import
    #      return {mu_id: MemoryUnit.from_dict(data) for mu_id, data in mu_data.items()}
    #
    # def _load_all_session_memories_from_json(self) -> Dict[str, SessionMemory]:
    #      sms_data = json_handler.load_sm_json(self.chatbot_id, self.storage_base_path) # Use renamed import
    #      return {sm_id: SessionMemory.from_dict(data) for sm_id, data in sms_data.items()}
    #
    # def _load_or_create_ltm_from_json(self) -> LongTermMemory:
    #      ltms_data = json_handler.load_ltm_json(self.chatbot_id, self.storage_base_path) # Use renamed import
    #      ltm_data = ltms_data.get(self.ltm_id)
    #      if ltm_data:
    #          return LongTermMemory.from_dict(ltm_data)
    #      else:
    #          # Return a new one if not found in JSON
    #          return LongTermMemory(chatbot_id=self.chatbot_id, ltm_id=self.ltm_id)

    # --- Core Methods Modified for SQLite ---

    def _flush_cache(self, force: bool = False):
        """Saves all staged changes based on persistence mode."""
        # Combine checks for pending changes
        has_db_updates = bool(
            self._updated_memory_unit_ids or self._updated_session_memory_ids or
            self._updated_ltm_ids or self._deleted_memory_unit_ids or
            self._deleted_session_memory_ids
        )
        has_json_staging = bool(
            self.persistence_mode == 'json' and (self._updated_memory_unit_ids or self._updated_session_memory_ids or
                                                 self._updated_ltm_ids or self._deleted_memory_unit_ids or
                                                 self._deleted_session_memory_ids)
        )

        if not force and self._staged_updates_count == 0 and not has_db_updates and not has_json_staging:
            return

        print(f"Flushing cache ({self.persistence_mode} mode)...")
        persistence_successful = False

        # --- SQLite Mode ---
        if self.persistence_mode == 'sqlite':
            if not self.sql_conn: print("Error: Cannot flush, SQLite connection lost."); return
            try:
                with self.sql_conn:  # Transaction
                    # Deletions
                    if self._deleted_memory_unit_ids:
                        sqlite_handler.delete_memory_units(self.sql_conn, list(self._deleted_memory_unit_ids))
                    if self._deleted_session_memory_ids:
                        sqlite_handler.delete_session_memories(self.sql_conn, list(self._deleted_session_memory_ids))
                    # Upserts
                    units_to_upsert = [self.memory_units_cache[mu_id] for mu_id in self._updated_memory_unit_ids if
                                       mu_id in self.memory_units_cache]
                    if units_to_upsert:
                        sqlite_handler.upsert_memory_units(self.sql_conn, units_to_upsert)
                    sessions_to_upsert = [self.session_memories_cache[sm_id] for sm_id in
                                          self._updated_session_memory_ids if sm_id in self.session_memories_cache]
                    if sessions_to_upsert:
                        sqlite_handler.upsert_session_memories(self.sql_conn, sessions_to_upsert)
                    ltms_to_upsert = [
                        self.long_term_memory] if self.long_term_memory and self.long_term_memory.id in self._updated_ltm_ids else []
                    if ltms_to_upsert:
                        sqlite_handler.upsert_long_term_memories(self.sql_conn, ltms_to_upsert)
                    self.sql_conn.commit()
                print("SQLite commit successful.")
                persistence_successful = True
            except sqlite3.Error as e:
                print(f"Error during SQLite transaction: {e}. Transaction rolled back.")
                persistence_successful = False

        # --- JSON Partitioned Mode ---
        elif self.persistence_mode == 'json':
            units_to_upsert = []
            units_to_delete = []
            sessions_for_delete_lookup = {}  # unit_id -> session_id for deletes
            try:
                # 1. Handle Memory Unit Partitions (Upserts and Deletes)
                for unit_id in self._updated_memory_unit_ids:  # Use the general updated set
                    if unit_id in self.memory_units_cache:  # Check if staged this cycle
                        unit_obj = self.memory_units_cache[unit_id]
                        units_to_upsert.append(unit_obj)

                for unit_id in self._deleted_memory_unit_ids:  # Use the general deleted set
                    if unit_id in self.memory_units_cache:  # Check if staged this cycle
                        units_to_delete.append(unit_id)

                # Call incremental save/delete handler
                if units_to_upsert or units_to_delete:
                    json_handler.save_memory_units_incremental_json(
                        units_to_upsert, units_to_delete,
                        self.chatbot_id, self.storage_base_path
                    )

                # 2. Save Session Metadata File (Incremental)
                sessions_to_update_meta = [self.session_memories_cache[sm_id] for sm_id in
                                           self._updated_session_memory_ids if sm_id in self.session_memories_cache]
                sessions_to_delete_meta_ids = list(self._deleted_session_memory_ids)
                if sessions_to_update_meta or sessions_to_delete_meta_ids:
                    json_handler.save_session_memories_incremental_json(
                        sessions_to_update_meta, sessions_to_delete_meta_ids,
                        self.chatbot_id, self.storage_base_path
                    )

                # 3. Save LTM File (Incremental)
                if self.long_term_memory and self.long_term_memory.id in self._updated_ltm_ids:
                    json_handler.save_single_long_term_memory_json(
                        self.long_term_memory, self.chatbot_id, self.storage_base_path
                    )

                print("JSON save operations completed.")
                persistence_successful = True
            except Exception as e:
                print(f"Error saving data to JSON: {e}")
                persistence_successful = False

        # --- Post-Persistence Cleanup ---
        if persistence_successful:
            self._updated_memory_unit_ids.clear()
            self._updated_session_memory_ids.clear()
            self._updated_ltm_ids.clear()
            self._deleted_memory_unit_ids.clear()
            self._deleted_session_memory_ids.clear()
        else:
            print("Persistence failed. Update sets not cleared.")
            return

        # --- Milvus Flush ---
        milvus_flushed = False
        if self.vector_store:
            try:
                self.vector_store.flush()
                milvus_flushed = True
                print("Milvus flush successful.")
            except Exception as e:
                print(f"Error during Milvus flush: {e}")
                milvus_flushed = False
        else:
            milvus_flushed = True

        # --- Auto-forgetting ---
        # Needs updated list of deleted IDs from primary store if JSON mode
        if persistence_successful and milvus_flushed:
            self._check_and_forget_memory()  # Pass deleted IDs if needed

        # Reset staged operation counter
        self._staged_updates_count = 0
        # end_time = time.time()
        # print(f"Cache flush complete in {end_time - start_time:.3f} seconds.")

    def _stage_memory_unit_update(self,
                                  memory_unit: MemoryUnit,
                                  operation: str = 'add',  # 'add', 'update'
                                  embedding: Optional[np.ndarray] = None,
                                  update_session_metadata: bool = True):
        """Stages a MemoryUnit for saving, handling persistence mode."""
        if not memory_unit or not memory_unit.id:
            return
        # Don't update the deleted unit
        if operation == "update" and memory_unit.id in self._deleted_memory_unit_ids:
            return

        # Determine session ID
        group_id = memory_unit.group_id
        # Don't update the deleted session
        if operation == "update" and group_id in self._deleted_session_memory_ids:
            return

        # --- Cache Update (Common) ---
        self.memory_units_cache[memory_unit.id] = memory_unit
        self._updated_memory_unit_ids.add(memory_unit.id)  # Mark ID as dirty
        self._deleted_memory_unit_ids.discard(memory_unit.id)  # Unmark from deletion if added

        # --- Session Metadata Update (Common) ---
        if memory_unit.rank == 0 and update_session_metadata and group_id:
            current_session = self._get_session_memory(group_id)  # Use helper to get/load session
            if current_session:
                if memory_unit.id not in current_session.memory_unit_ids:
                    current_session.memory_unit_ids.append(memory_unit.id)
                current_session.update_timestamps([memory_unit])
                self.session_memories_cache[group_id] = current_session  # Update cache
                self._updated_session_memory_ids.add(group_id)  # Mark session meta for saving
                self._deleted_session_memory_ids.discard(group_id)  # Unmark delete

        # --- Stage for Milvus (Common) ---
        if self.vector_store:
            final_embedding = embedding
            if operation == 'add' and final_embedding is None:
                try:
                    content_to_embed = self._get_formatted_content_with_history(memory_unit, 1)
                    final_embedding = self.embedding_handler.get_embedding(content_to_embed)
                except Exception as e:
                    final_embedding = None
            elif operation == 'update' and final_embedding is None:
                search_result = self.vector_store.get(memory_unit.id,output_fields=["embedding"])
                if search_result:
                    final_embedding = search_result.get("embedding")
                else:
                    try:
                        content_to_embed = self._get_formatted_content_with_history(memory_unit, 1)
                        final_embedding = self.embedding_handler.get_embedding(content_to_embed)
                    except Exception as e:
                        final_embedding = None

            milvus_data = None
            if final_embedding is not None:
                milvus_data = {
                    "id": memory_unit.id, "embedding": final_embedding.tolist(),
                    "parent_id": memory_unit.parent_id, "content": memory_unit.content,
                    "creation_time": memory_unit.creation_time.timestamp() if memory_unit.creation_time else None,
                    "end_time": memory_unit.end_time.timestamp() if memory_unit.end_time else None,
                    "source": memory_unit.source, "metadata": memory_unit.metadata or {},
                    "last_visit": memory_unit.last_visit, "visit_count": memory_unit.visit_count,
                    "never_delete": memory_unit.never_delete, "children_ids": memory_unit.children_ids or [],
                    'rank': memory_unit.rank, 'pre_id': memory_unit.pre_id, 'next_id': memory_unit.next_id
                }
            if milvus_data:
                try:
                    if operation == "update":
                        self.vector_store.upsert([milvus_data], flush=False)
                    elif operation == 'add':
                        self.vector_store.insert([milvus_data], flush=False)
                except Exception as e:
                    print(f"Error staging unit {memory_unit.id} to Milvus ({operation}): {e}")
            else:
                print(f"Fail to insert or update memory unit:{memory_unit.id}")

        # --- Increment Counter ---
        self._staged_updates_count += 1
        if self._staged_updates_count >= self.saving_interval:
            self._flush_cache()

    def _stage_memory_unit_deletion(self, unit_id: str):
        """Stages a MemoryUnit for deletion from primary store and Milvus."""
        print(f"Staging deletion for unit: {unit_id}")  
        # Mark ID for deletion in primary store
        self._deleted_memory_unit_ids.add(unit_id)
        self._updated_memory_unit_ids.discard(unit_id)  # Unmark from updates

        # Remove from cache
        self.memory_units_cache.pop(unit_id, None)

        # Stage deletion for Milvus (delete happens on flush)
        if self.vector_store:
            try:
                self.vector_store.delete(expr=f"id == \"{unit_id}\"", flush=False)
            except Exception as e:
                print(f"Error staging unit {unit_id} for Milvus deletion: {e}")

        # Increment staged counter? Maybe count deletions separately?
        self._staged_updates_count += 1
        if self._staged_updates_count >= self.saving_interval:
            self._flush_cache()

    def _get_memory_unit(self, unit_id: str, use_cache: bool = True) -> Optional[MemoryUnit]:
        """Retrieves a MemoryUnit, checking cache first, then primary store."""
        if use_cache and unit_id in self.memory_units_cache:
            return self.memory_units_cache[unit_id]

        unit: Optional[MemoryUnit] = None
        if self.persistence_mode == 'sqlite':
            if not self.sql_conn:
                return None
            unit = sqlite_handler.load_memory_unit(self.sql_conn, unit_id)
        elif self.persistence_mode == 'json':
            return None
        if unit and use_cache:
            self.memory_units_cache[unit_id] = unit

        return unit

    def _get_session_memory(self, session_id: str, use_cache: bool = True) -> Optional[SessionMemory]:
        """Retrieves a SessionMemory, checking cache first, then primary store."""
        if use_cache and session_id in self.session_memories_cache:
            return self.session_memories_cache[session_id]

        session: Optional[SessionMemory] = None
        if self.persistence_mode == 'sqlite':
            if not self.sql_conn:
                return None
            session = sqlite_handler.load_session_memory(self.sql_conn, session_id)
        elif self.persistence_mode == 'json':
            session_dict = json_handler.load_session_memories_json(self.chatbot_id, self.storage_base_path).get(
                session_id)
            session = SessionMemory.from_dict(session_dict)

        if session and use_cache:
            self.session_memories_cache[session_id] = session

        return session

    def _get_long_term_memory(self, ltm_id: str) -> Optional[LongTermMemory]:
        """Retrieves a specific LTM object (primarily for external use)."""
        if ltm_id == self.ltm_id:
            return self.long_term_memory

        if ltm_id == self.external_ltm_id and self.external_long_term_memory:
            return self.external_long_term_memory

        if self.persistence_mode == 'sqlite' and self.sql_conn:
            return sqlite_handler.load_long_term_memory(self.sql_conn, ltm_id, self.chatbot_id)
        elif self.persistence_mode == 'json':
            ltm_data_dict = json_handler.load_long_term_memories_json(self.chatbot_id, self.storage_base_path)
            ltm_data = ltm_data_dict.get(ltm_id)
            if ltm_data:
                return LongTermMemory.from_dict(ltm_data)

        return None

    def _load_memory_units(self, unit_ids: List[str], use_cache: bool = True) -> Dict[str, MemoryUnit]:
        """Loads multiple MemoryUnits, checking cache first."""
        loaded_units: Dict[str, MemoryUnit] = {}
        ids_to_load_from_store: List[str] = []

        if use_cache:
            for unit_id in unit_ids:
                if unit_id in self.memory_units_cache:
                    loaded_units[unit_id] = self.memory_units_cache[unit_id]
                else:
                    ids_to_load_from_store.append(unit_id)
        else:
            ids_to_load_from_store = unit_ids

        if not ids_to_load_from_store:
            return loaded_units

        # print(f"Loading {len(ids_to_load_from_store)} units from primary store ({self.persistence_mode})...") 
        store_units: Dict[str, MemoryUnit] = {}
        if self.persistence_mode == 'sqlite':
            if not self.sql_conn:
                return loaded_units
            store_units = sqlite_handler.load_memory_units(self.sql_conn, ids_to_load_from_store)
        elif self.persistence_mode == 'json':
            store_units_dict = json_handler.load_memory_units_json(self.chatbot_id, self.storage_base_path)
            for unit_id in unit_ids:
                if unit_id in store_units_dict:
                    store_units[unit_id] = MemoryUnit.from_dict(store_units_dict[unit_id])

        # Add loaded units to cache and final result
        if use_cache:
            self.memory_units_cache.update(store_units)
        loaded_units.update(store_units)
        print(f"Finished loading units. Total loaded/cached: {len(loaded_units)}")  
        return loaded_units

    def _load_sessions(self, session_ids: List[str], use_cache: bool = True) -> Dict[str, SessionMemory]:
        """Loads multiple SessionMemories, checking cache first."""
        loaded_sessions: Dict[str, SessionMemory] = {}
        ids_to_load_from_store: List[str] = []

        if use_cache:
            for session_id in session_ids:
                if session_id in self.session_memories_cache:
                    loaded_sessions[session_id] = self.session_memories_cache[session_id]
                else:
                    ids_to_load_from_store.append(session_id)
        else:
            ids_to_load_from_store = session_ids

        if not ids_to_load_from_store:
            return loaded_sessions

        store_sessions: Dict[str, SessionMemory] = {}
        if self.persistence_mode == 'sqlite':
            if not self.sql_conn:
                return loaded_sessions
            store_sessions = sqlite_handler.load_session_memories(self.sql_conn, ids_to_load_from_store)
        elif self.persistence_mode == 'json':
            session_memories_dict = json_handler.load_session_memories_json(self.chatbot_id, self.storage_base_path)
            for session_id in ids_to_load_from_store:
                if session_id in session_memories_dict:  # Should exist if valid ID
                    store_sessions[session_id] = SessionMemory.from_dict(session_memories_dict[session_id])

        if use_cache:
            self.session_memories_cache.update(store_sessions)
        loaded_sessions.update(store_sessions)
        return loaded_sessions

    def _load_units_for_session(self, session_id: str) -> Dict[str, MemoryUnit]:
        """Helper to load all units listed in a session object."""
        session = self._get_session_memory(session_id)
        if session and session.memory_unit_ids:
            return self._load_memory_units(session.memory_unit_ids)
        return {}

    def _load_all_units_for_ltm(self) -> Dict[str, MemoryUnit]:
        """Loads ALL units belonging to sessions within the current LTM."""
        # print(f"Warning: Loading ALL units for LTM {self.ltm_id}. This may consume significant memory.")
        if not self.long_term_memory or not self.session_memories_cache:
            return {}

        all_unit_ids: Set[str] = set()
        all_unit_ids.add(self.ltm_id)
        all_unit_ids.update(self.long_term_memory.session_ids)
        all_unit_ids.update(self.long_term_memory.summary_unit_ids)
        for session_id in self.long_term_memory.session_ids:
            session = self.session_memories_cache.get(session_id)
            if session:
                all_unit_ids.update(session.memory_unit_ids)

        return self._load_memory_units(list(all_unit_ids))

    def _get_current_session(self) -> Optional[SessionMemory]:
        """Gets the current session object, loading from DB if not in cache."""
        if not self._current_session_id:
            return None
        return self._get_session_memory(self._current_session_id)

    def _update_children_for_removed_unit(self, unit: Optional[MemoryUnit]):
        if not unit:
            return
        for unit_id in unit.children_ids:
            unit = self._get_memory_unit(unit_id)
            if unit:
                unit.parent_id = None
                self._stage_memory_unit_update(unit, operation='update', embedding=None)

    def remove_session(self, session_id: str):
        """Removes a session based on persistence mode."""
        print(f"Removing session: {session_id} ({self.persistence_mode} mode)")

        # 1. Get session object (needed to find units in JSON mode)
        # Always load session metadata first
        session_memory = self._get_session_memory(session_id, use_cache=True)

        # 2. Update LTM object (consistent for both modes)
        if self.long_term_memory and session_id in self.long_term_memory.session_ids:
            self.long_term_memory.session_ids.remove(session_id)
            if self.long_term_memory.last_session_id == session_id:
                self.long_term_memory.last_session_id = self.long_term_memory.session_ids[
                    -1] if self.long_term_memory.session_ids else None

        # 3. Delete summary units for session in LTM if exists.
        unit_ids_to_remove_set: set[str] = set()
        session_summary_unit = self._get_memory_unit(session_id)
        if session_summary_unit.parent_id:
            session_summary_unit = self.memory_units_cache.get(session_summary_unit.parent_id, None)
        while (session_summary_unit is not None):
            unit_ids_to_remove_set.add(session_summary_unit.id)
            self._update_children_for_removed_unit(session_summary_unit)
            parent_unit = None
            if session_summary_unit.parent_id:
                parent_unit = self.memory_units_cache.get(session_summary_unit.parent_id, None)
            session_summary_unit = parent_unit
        if self.long_term_memory:
            self.long_term_memory.summary_unit_ids = [id for id in self.long_term_memory.summary_unit_ids if
                                                      id not in unit_ids_to_remove_set]
            self._updated_ltm_ids.add(self.long_term_memory.id)

        # 4. Remove session object from cache and mark for DB/JSON deletion
        self.session_memories_cache.pop(session_id, None)
        self._updated_session_memory_ids.discard(session_id)
        self._deleted_session_memory_ids.add(session_id)  # Mark session metadata for deletion

        # 5. Identify associated memory units
        if session_memory:
            unit_ids_to_remove_set.update(session_memory.memory_unit_ids)
        unit_ids_to_remove_set.add(session_id)
        unit_ids_to_remove = list(unit_ids_to_remove_set)

        # 6. Remove units from cache and mark for deletion
        # removed_unit_count = 0
        if unit_ids_to_remove:
            # print(f"Removing {len(unit_ids_to_remove)} memory units associated with session {session_id}...")
            for unit_id in unit_ids_to_remove:
                # if self.memory_units_cache.pop(unit_id, None): removed_unit_count += 1
                self._updated_memory_unit_ids.discard(unit_id)
                self._deleted_memory_unit_ids.add(unit_id)
                self._visited_unit_counts.pop(unit_id, None)
                self._unit_last_visit_round.pop(unit_id, None)
                if unit_id in self._stm_id_set:
                    self._remove_unit_from_stm(unit_id)
                # Mark for primary store deletion

            # 6. Remove units from Milvus (consistent for both modes)
            if self.vector_store:
                try:
                    expr = f"id in {json.dumps(unit_ids_to_remove)}"
                    print(f"Deleting units from Milvus with expr: {expr}")
                    self.vector_store.delete(expr, flush=False)  # Defer flush
                except Exception as e:
                    print(f"Error deleting session units from Milvus: {e}")

        # 7. Trigger flush
        # print(f"Session {session_id} removal staged. Flushing cache.")
        self._flush_cache(force=True)
        # print(f"Session {session_id} removal process complete.")

    def _start_session(self, session_id: Optional[str] = None):
        """Starts/resumes session, handles loading/creation based on mode."""
        if session_id:
            # Check cache first (covers both modes as session metadata is cached)
            session_obj = self.session_memories_cache.get(session_id)
            if not session_obj and self.persistence_mode == 'sqlite':
                # If not in cache and using SQLite, try loading from DB
                session_obj = self._get_session_memory(session_id, use_cache=False)

            if session_obj:
                self._current_session_id = session_id
                print(f"Resuming session: {session_id} ({self.persistence_mode} mode)")
                self._restore_stm_from_session(self._current_session_id)
                return
            else:
                print(f"Warning: Requested session {session_id} not found. Starting a new session.")

        # Create new session
        new_id = str(uuid4())
        self._current_session_id = new_id
        print(f"Starting new session: {new_id} ({self.persistence_mode} mode)")
        creation_time = datetime.datetime.now()
        new_sm = SessionMemory(session_id=new_id, creation_time=creation_time, end_time=creation_time)
        self.session_memories_cache[new_id] = new_sm
        self._updated_session_memory_ids.add(new_id)  # Mark for saving

        if self.long_term_memory:
            if new_id not in self.long_term_memory.session_ids:
                self.long_term_memory.session_ids.append(new_id)
                self.long_term_memory.last_session_id = new_id
                self._updated_ltm_ids.add(self.long_term_memory.id)

    def _get_formatted_content_with_history(self, memory_unit: MemoryUnit, history_length: int = 1) -> str:
        """
        Retrieves the content of a memory unit, optionally prepended with history.

        Args:
            memory_unit: The target MemoryUnit.
            history_length: The number of preceding messages to include (0 for none).

        Returns:
            A formatted string combining history and the unit's content.
        """
        history_units: List[MemoryUnit] = []
        current_unit_id = memory_unit.pre_id

        # Traverse backwards, loading from cache or DB
        while current_unit_id and len(history_units) < history_length:
            # Try cache first, then DB
            unit = self._get_memory_unit(current_unit_id, use_cache=True)
            if unit:
                history_units.append(unit)
                current_unit_id = unit.pre_id  # Move to the next predecessor
            else:
                # print(f"Warning: Could not find predecessor unit {current_unit_id} in cache or DB.") 
                break  # Stop traversal if a unit is missing

        # Combine content (logic remains the same)
        contents: List[str] = []
        all_units = history_units[::-1] + [memory_unit]  # Reverse history + current

        for unit in all_units:
            action = unit.metadata.get('action') if unit.metadata else None
            if action and action in ["summary", "mind"]:
                contents.append(f"{unit.source}: ({unit.content})")
            elif action == "speak":
                contents.append(f"{unit.source}: {unit.content}")
            else:
                contents.append(unit.content)
        return "\n".join(contents)

    def enable_external_connection(self,
                                   chatbot_id: str,
                                   ltm_id: str,
                                   milvus_config: Optional[Dict[str, Any]] = None):
        """
        Enables connection to an external chatbot's memory stored in Milvus (and optionally JSON).

        Args:
            chatbot_id: ID of the external chatbot.
            ltm_id: ID of the external LTM instance.
            milvus_config: Milvus connection details for the external store.

        Raises:
            ValueError: If the external Milvus collection doesn't exist or JSON files are missing when requested.
        """
        if not milvus_config:
            raise ValueError("Either Milvus config or load_json_data=True must be provided for external connection.")

        ext_vector_store = None
        if milvus_config:
            try:
                ext_vector_store = VectorStoreHandler(chatbot_id, ltm_id, **milvus_config)
                if not ext_vector_store.has_collection():
                    raise ValueError(f"External Milvus collection for {chatbot_id}/{ltm_id} not found.")
            except Exception as e:
                raise ValueError(f"Failed to connect to external Milvus for {chatbot_id}/{ltm_id}. Error: {e}") from e

        self.external_chatbot_id = chatbot_id
        self.external_ltm_id = ltm_id
        self.external_vector_store = ext_vector_store
        print(f"External connection enabled to {chatbot_id}/{ltm_id}.")

    def _restore_stm_from_session(self, session_id: str):
        """
        Loads MemoryUnits from a specified session into the STM.

        Fetches embeddings from Milvus. Requires STM and Milvus to be enabled.

        Args:
            session_id: The ID of the session to restore.
        """
        if not self._stm_enabled:
            print("Warning: Cannot restore STM, it is disabled.")
            return
        if not self.vector_store:
            print("Warning: Cannot restore STM from session without Milvus vector store.")
            return

        session_memory = self._get_session_memory(session_id)
        if not session_memory:
            print(f"Warning: Session {session_id} not found for STM restoration.")
            return

        memory_unit_ids = session_memory.memory_unit_ids
        if not memory_unit_ids:
            print(f"Warning: Session {session_id} has no memory units to restore.")
            return

        # Fetch units and embeddings from Milvus
        try:
            expr = f"id in {memory_unit_ids}"  # Milvus list query format
            results = self.vector_store.query(expr=expr,
                                              output_fields=self.vector_store.output_fields,
                                              # Get all fields + embedding
                                              top_k=len(memory_unit_ids) * 2)
        except Exception as e:
            print(f"Error querying Milvus for session {session_id} units: {e}")
            return

        units_to_add_to_stm: List[Tuple[MemoryUnit, np.ndarray]] = []
        found_ids = set()

        # Use local memory_units if available, augment with Milvus data if needed
        result_dict = {entity.get("id"): entity for entity in results}

        units = self._load_memory_units(memory_unit_ids)
        for unit_id in memory_unit_ids:
            unit = units.get(unit_id, None)
            entity = result_dict.get(unit_id, None)
            if unit and entity and 'embedding' in entity:
                embedding = np.array(entity['embedding'])
                units_to_add_to_stm.append((unit, embedding))
                found_ids.add(unit_id)
            elif entity:  # Unit not in local cache, reconstruct from Milvus data
                try:
                    unit = MemoryUnit.from_dict(entity)
                    embedding = np.array(entity['embedding'])
                    units_to_add_to_stm.append((unit, embedding))
                    self.memory_units_cache[unit_id] = unit  # Add to local cache
                    found_ids.add(unit_id)
                except Exception as e:
                    print(f"Warning: Could not reconstruct MemoryUnit {unit_id} from Milvus data. Error: {e}")
            else:
                print(f"Warning: Memory unit {unit_id} from session {session_id} not found locally or in Milvus.")
        units_to_add_to_stm.sort(key=lambda item: item[0].creation_time or datetime.datetime.min)

        print(f"Adding {len(units_to_add_to_stm)} units from session {session_id} to STM.")
        self._add_units_to_stm(units_to_add_to_stm)

    def enable_stm(self, capacity: int = 200, restore_session_id: Optional[str] = None):
        """
        Enables Short-Term Memory (STM) with a given capacity.

        Optionally restores the state from the last session or a specific session.

        Args:
            capacity: Maximum number of items to store in STM.
            restore_session_id: If provided, restore STM from this session ID. 
                               # If None, attempts to restore from the last known session.
                               If " LATEST", explicitly uses last_session_id from LTM.
                               If None, don't restore..
        """
        if self._stm_enabled:
            # print("STM is already enabled.")
            if self._stm_capacity != capacity:
                # print(f"Updating STM capacity from {self._stm_capacity} to {capacity}.")
                self._stm_capacity = capacity
                # Trim if new capacity is smaller
                while len(self._short_term_memory_ids) > self._stm_capacity:
                    self._remove_oldest_from_stm()
            return

        # print(f"Enabling STM with capacity {capacity}.")
        self._stm_enabled = True
        self._stm_capacity = capacity
        self._short_term_memory_ids = deque()
        self._short_term_memory_embeddings.clear()
        self._short_term_memory_units.clear()
        self._stm_id_set.clear()

        # Restore STM content
        target_session_id = None
        if restore_session_id == "LATEST":
            target_session_id = self.long_term_memory.last_session_id
        elif restore_session_id:
            target_session_id = restore_session_id

        if target_session_id:
            # print(f"Attempting to restore STM from session: {target_session_id}")
            self._restore_stm_from_session(target_session_id)
        # elif restore_session_id is None and self.long_term_memory.last_session_id:
        #     # Default behavior: restore from the last session if none specified
        #     print(f"Attempting to restore STM from last known session: {self.long_term_memory.last_session_id}")
        #     self._restore_stm_from_session(self.long_term_memory.last_session_id)

    def disable_stm(self):
        """Disables and clears the Short-Term Memory."""
        if not self._stm_enabled:
            print("STM is already disabled.")
            return

        print("Disabling and clearing STM.")
        self._short_term_memory_ids.clear()
        self._short_term_memory_embeddings.clear()
        self._short_term_memory_units.clear()
        self._stm_id_set.clear()
        self._stm_enabled = False
        self._stm_capacity = 0
        self._sqlite_conn = None  # Clear placeholder connection

    def _check_and_forget_memory(self):
        """Checks Milvus count and triggers forgetting mechanism."""
        if not self.vector_store or self.max_milvus_entities <= 0:
            return

        try:
            current_count = self.vector_store.count_entities()
            if current_count <= self.max_milvus_entities:
                return

            print(f"Milvus count {current_count} > {self.max_milvus_entities}. Triggering forgetting...")

            # Call refactored forgetting function
            deleted_unit_ids, updated_parent_ids, updated_session_ids = forget_memories(
                memory_system=self,
                ltm_id=self.ltm_id,
                delete_percentage=self.forget_percentage,
            )

            if not deleted_unit_ids:
                print("Forgetting identified no units to delete.")
                return

            print(f"Forgetting identified {len(deleted_unit_ids)} units. Staging changes...")

            # Stage deletions for primary store and Milvus
            for unit_id in deleted_unit_ids:
                self._stage_memory_unit_deletion(unit_id)
                # Update unit->session map if JSON mode
                if self.persistence_mode == 'json':
                    self.memory_units_cache.pop(unit_id, None)

            # Stage updates for parents
            parents_to_update = self._load_memory_units(list(updated_parent_ids))  # Load parents
            for parent_id, parent_unit in parents_to_update.items():
                original_children = set(parent_unit.children_ids or [])
                deleted_in_this_parent = original_children.intersection(deleted_unit_ids)
                if deleted_in_this_parent:
                    parent_unit.children_ids = [id for id in parent_unit.children_ids if
                                                id not in deleted_in_this_parent]
                    self._stage_memory_unit_update(parent_unit, operation='update', update_session_metadata=False)

            # Stage updates for sessions
            sessions_to_update = self._load_sessions(list(updated_session_ids))  # Load sessions
            for session_id, session_unit in sessions_to_update.items():
                # Remove deleted units from session's list
                original_units = set(session_unit.memory_unit_ids or [])
                deleted_in_this_session = original_units.intersection(deleted_unit_ids)
                if deleted_in_this_session:
                    session_unit.memory_unit_ids = [id for id in session_unit.memory_unit_ids if
                                                    id not in deleted_in_this_session]
                    self._stage_session_memory_update(session_unit)

            # Force flush to commit forgetting changes
            self._flush_cache(force=True)
            print(f"Forgetting cleanup complete.")

        except Exception as e:
            print(f"Error during auto-forgetting check or execution: {e}")

    def convert_sql_to_json(self, output_dir: Optional[str] = None):
        """Loads all data from SQLite and saves it to JSON files."""
        if self.persistence_mode != "sqlite":
            print("Error: SQLite is not enabled. Cannot convert from SQL.")
            return
        if not self.sql_conn:
            print("Error: SQLite connection not available.")
            return

        output_base_path = output_dir if output_dir else self.storage_base_path # Use provided or default path
        target_chatbot_id_dir = os.path.join(output_base_path, self.chatbot_id)
        os.makedirs(target_chatbot_id_dir, exist_ok=True)

        print(f"Converting SQLite data for {self.chatbot_id}/{self.ltm_id} to JSON in {output_base_path}...")

        try:
            # Load all data from SQL
            sql_units = sqlite_handler.load_all_memory_units(self.sql_conn)
            sql_sessions = sqlite_handler.load_all_session_memories(self.sql_conn)
            sql_ltm = sqlite_handler.load_all_long_term_memories_for_chatbot(self.sql_conn, chatbot_id=self.chatbot_id)

            # Save to JSON using existing helpers
            if sql_units:
                memory_units_dict = {k: v.to_dict() for k, v in sql_units.items()}
                json_handler.save_memory_units_json(memory_units_dict, self.chatbot_id, output_base_path)
                print(f"Saved {len(sql_units)} memory units to JSON.")
            if sql_sessions:
                session_memories_dict = {k: v.to_dict() for k, v in sql_sessions.items()}
                json_handler.save_session_memories_json(session_memories_dict, self.chatbot_id, output_base_path)
                print(f"Saved {len(sql_sessions)} session memories to JSON.")
            if sql_ltm:
                long_term_memories_dict = {k: v.to_dict() for k, v in sql_ltm.items()}
                json_handler.save_long_term_memories_json(long_term_memories_dict, self.chatbot_id,
                                                          output_base_path)
                print(f"Saved LTM object to JSON.")

            print("SQLite to JSON conversion complete.")

        except Exception as e:
            print(f"Error during SQLite to JSON conversion: {e}")

    def convert_sql_to_milvus(self, embedding_history_length: int = 1):
        """Loads MemoryUnits from SQLite, generates embeddings, and upserts into Milvus."""
        if self.persistence_mode != "sqlite":
            print("Error: SQLite is not enabled. Cannot convert from SQL.")
            return
        if not self.sql_conn:
            print("Error: SQLite connection not available.")
            return
        if not self.vector_store:
            print("Error: Milvus vector store is not configured.")
            return

        print(f"Converting SQLite memory units for {self.chatbot_id}/{self.ltm_id} to Milvus...")

        sql_units = sqlite_handler.load_all_memory_units(self.sql_conn)
        print(f"Loaded {len(sql_units)} units from SQLite.")
        if not sql_units:
            return

        units_to_upsert = []
        processed_count = 0
        error_count = 0

        for unit_id, unit in sql_units.items():
            try:
                content_to_embed = self._get_formatted_content_with_history(unit,
                                                                            history_length=embedding_history_length)
                embedding = self.embedding_handler.get_embedding(content_to_embed)
                # print(content_to_embed)
                milvus_data = {
                    "id": unit.id, "parent_id": unit.parent_id, "content": unit.content,
                    "creation_time": unit.creation_time.timestamp() if unit.creation_time else None,
                    "end_time": unit.end_time.timestamp() if unit.end_time else None,
                    "source": unit.source, "metadata": unit.metadata,
                    "last_visit": unit.last_visit, "visit_count": unit.visit_count,
                    "never_delete": unit.never_delete, "children_ids": unit.children_ids,
                    "embedding": embedding.tolist(), 'rank': unit.rank,
                    'pre_id': unit.pre_id, 'next_id': unit.next_id, "group_id": unit.group_id}

                units_to_upsert.append(milvus_data)
                processed_count += 1
            except Exception as e:
                error_count += 1
                print(f"Error processing unit {unit_id} for Milvus conversion: {e}")

            # Batch insert (optional, as before)

        # Insert final batch
        if units_to_upsert:
            print(f"Upserting final batch of {len(units_to_upsert)} units to Milvus...")
            try:
                self.vector_store.upsert(units_to_upsert)
                self.vector_store.flush()
            except Exception as e:
                print(f"Error upserting final batch to Milvus: {e}")
                error_count += len(units_to_upsert)

        print(f"SQLite to Milvus conversion finished.")
        print(f"Successfully processed: {processed_count - error_count} units.")
        print(f"Errors encountered: {error_count} units.")

    def _add_to_context(self, memory_unit: MemoryUnit, embedding_history_length: int = 1):
        """Adds to context deque, handles linking, eviction, and staging."""
        if not memory_unit or not memory_unit.id: return

        # Link to previous context unit
        if self.context:
            last_context_unit = self.context[-1]
            if last_context_unit.id != memory_unit.id:  # Avoid self-linking
                last_context_unit.next_id = memory_unit.id
                memory_unit.pre_id = last_context_unit.id
                # Stage update for *previous* unit (linking change)
                self._stage_memory_unit_update(last_context_unit, operation='update',
                                               update_session_metadata=False) 

        self.context.append(memory_unit)
        # Add to cache immediately so _get_formatted_content works
        self.memory_units_cache[memory_unit.id] = memory_unit
        # Also stage the *new* unit itself (as 'add' initially)
        # Embedding will be generated/checked during staging
        self._stage_memory_unit_update(memory_unit, operation='add')

        # Evict oldest if max length exceeded
        if len(self.context) > self._max_context_length:
            evicted_unit = self.context.popleft()
            print(f"Context full. Evicting unit {evicted_unit.id}")
            # Prepare evicted unit for long-term store / STM
            evicted_embedding = None
            try:  # Regenerate embedding with history before eviction
                content_to_embed = self._get_formatted_content_with_history(evicted_unit, embedding_history_length)
                evicted_embedding = self.embedding_handler.get_embedding(content_to_embed)
            except Exception as e:
                print(f"Error embedding evicted {evicted_unit.id}: {e}")

            # Stage evicted unit again (now confirmed 'add' to persistent store)
            # Pass embedding if available
            self._stage_memory_unit_update(evicted_unit, operation='add', embedding=evicted_embedding)

            # Add to STM if enabled
            if self._stm_enabled and evicted_embedding is not None:
                self._add_units_to_stm([(evicted_unit, evicted_embedding)])
            elif self._stm_enabled:
                print(f"Warn: Cannot add evicted {evicted_unit.id} to STM, embedding missing.")

    def add_memory(self, message: Union[str, BaseMessage], source: Optional[str] = None,
                   metadata: Optional[Dict[str, Any]] = None, creation_time: Optional[datetime.datetime] = None,
                   memory_unit_id: Optional[str] = None) -> Optional[MemoryUnit]:
        """Adds a new message memory to context and stages for persistence."""
        if not self._current_session_id: print("Error: No active session."); return None
        now = datetime.datetime.now(datetime.timezone.utc)
        creation_time = creation_time or now
        metadata = metadata if metadata is not None else {"action": "speak"}

        try:  # Create unit
            if isinstance(message, str):
                unit = MemoryUnit(content=message, source=source, metadata=metadata, creation_time=creation_time,
                                  end_time=creation_time, memory_id=memory_unit_id)
            elif isinstance(message, BaseMessage):
                unit = MemoryUnit.from_langchain_message(message=message, source=source, metadata=metadata,
                                                         creation_time=creation_time, end_time=creation_time,
                                                         memory_id=memory_unit_id)
            else:
                print(f"Error: Unsupported message type: {type(message)}")
                return None
        except Exception as e:
            print(f"Error creating MemoryUnit: {e}")
            return None

        self._add_to_context(unit)  # Add to context deque
        return unit

    def get_context(self, length: Optional[int] = None) -> List[MemoryUnit]:
        """
        Returns the current context messages.

        Args:
            length: The maximum number of most recent messages to return. 
                    If None, returns the entire current context.

        Returns:
            A list of MemoryUnit objects from the context.
        """
        if length is None:
            return list(self.context)
        else:
            actual_length = min(length, len(self.context))
            # Return the last 'actual_length' items
            return list(self.context)[-actual_length:]

    def clear_context(self):
        """Clears the current conversation context."""
        self.context.clear()

    def _flush_context(self, embedding_history_length: int = 1):
        """Flush the current conversation context to stm and clear."""
        if not self._stm_enabled:
            return
        # print(f"Clearing context ({len(self.context)} items).")
        while self.context:
            evicted_unit = self.context.popleft()
            try:
                content_to_embed = self._get_formatted_content_with_history(evicted_unit, embedding_history_length)
                evicted_embedding = self.embedding_handler.get_embedding(content_to_embed)
                # Add back to STM cache if STM is enabled
                self._short_term_memory_embeddings[evicted_unit.id] = evicted_embedding
            except Exception as e:
                print(f"Error regenerating embedding for evicted unit {evicted_unit.id}: {e}. Cannot save vector.")
                evicted_embedding = None  # Ensure it's None

            # Stage the evicted unit for persistence (as 'add' because it's now "long term")
            self._stage_memory_unit_update(evicted_unit, operation='add', embedding=evicted_embedding)

            if evicted_embedding is not None:
                self._add_units_to_stm([(evicted_unit, evicted_embedding)])
            else:
                print(f"Warning: Embedding for cleared context unit {evicted_unit.id} missing.")
        self._flush_cache(force=True)
        # print("Context cleared and items processed.")

    def _remove_oldest_from_stm(self):
        """Removes the least recently used item from STM."""
        if not self._short_term_memory_ids:
            return
        removed_id = self._short_term_memory_ids.popleft()
        self._short_term_memory_embeddings.pop(removed_id, None)
        self._short_term_memory_units.pop(removed_id, None)
        try:
            self._stm_id_set.remove(removed_id)
        except ValueError:
            pass
        # print(f"Removed oldest unit {removed_id} from STM.")

    def _remove_unit_from_stm(self, unit_id: str):
        """Removes a specific unit from STM structures."""
        if unit_id in self._stm_id_set:
            try:
                self._short_term_memory_ids.remove(unit_id)
            except ValueError:
                pass
            self._short_term_memory_embeddings.pop(unit_id, None)
            self._short_term_memory_units.pop(unit_id, None)
            self._stm_id_set.remove(unit_id)
            # print(f"Removed specific unit {unit_id} from STM.") 

    def _add_units_to_stm(self, memory_units_with_embeddings: List[Tuple[MemoryUnit, np.ndarray]]):
        """Adds or updates memory units in the STM, managing capacity."""
        if not self._stm_enabled:
            # print("Warning: Attempted to add to STM while disabled.")
            return

        for memory_unit, embedding in memory_units_with_embeddings:
            if not memory_unit or not memory_unit.id:
                print(f"Warning: Skipping invalid unit or unit with no embedding: {getattr(memory_unit, 'id', 'N/A')}")
                continue

            unit_id = memory_unit.id

            if unit_id in self._stm_id_set:
                # Already exists, move to end (most recently used)
                try:
                    self._short_term_memory_ids.remove(unit_id)
                    self._short_term_memory_ids.append(unit_id)
                except ValueError:
                    print(f"Error: Unit {unit_id} in _stm_id_set but not in deque. Re-adding.")
                    self._stm_id_set.remove(unit_id)

            if unit_id not in self._stm_id_set and embedding is not None:
                # Check capacity *before* adding
                if len(self._short_term_memory_ids) >= self._stm_capacity:
                    self._remove_oldest_from_stm()

                # Add the new unit
                self._short_term_memory_ids.append(unit_id)
                self._short_term_memory_units[unit_id] = memory_unit
                self._short_term_memory_embeddings[unit_id] = embedding
                self._stm_id_set.add(unit_id)
                # print(f"Added new unit {unit_id} to STM.") 
            else:
                print(
                    f"Warning: Skipping invalid unit or unit with no embedding: {getattr(memory_unit, 'id', 'N/A')}")
                continue

    def _query_stm(self,
                   query_vector: Optional[List[float]] = None,
                   filters: Optional[List[Tuple[str, str, Any]]] = None,
                   k_limit: int = 5,
                   search_range: Optional[Tuple[Optional[float], Optional[float]]] = (0.75, None)
                   ) -> List[Tuple[MemoryUnit, float]]:
        """
        Queries the Short-Term Memory.

        Performs vector similarity search and/or filtering directly on units in STM.
        Updates the LRU order for accessed units.

        Args:
            query_vector: The vector to search for similarity.
            filters: List of filter conditions (attribute, operator, value).
            k_limit: Maximum number of results to return.
            search_range: Tuple (min_similarity, max_similarity) for range search (optional).
                         Uses 'radius' and 'range_filter' in Milvus search params.

        Returns:
            A list of tuples, each containing a matching MemoryUnit and its similarity score (if vector query).
            Sorted by similarity score (descending) or STM order (descending MRU) if no vector.
        """
        if not self._stm_enabled:
            # raise ValueError("Short-term memory is disabled")
            print("Warning: STM query attempted while disabled.")
            return []

        candidate_ids = list(self._short_term_memory_ids)  # Query based on current order
        results: List[Tuple[MemoryUnit, float]] = []
        query_np = np.array(query_vector) if query_vector else None

        # Iterate through STM items (MRU to LRU order if needed, but deque is LRU->MRU)
        for unit_id in reversed(candidate_ids):

            unit = self._short_term_memory_units.get(unit_id)
            if not unit: continue

            # Apply filters first
            passes_filter = True
            if filters:
                try:
                    passes_filter = all(
                        self._evaluate_python_filter(unit, attr, op, val) for (attr, op, val) in filters)
                except Exception as e:
                    print(f"Warning: Error applying filter to STM unit {unit_id}: {e}")
                    passes_filter = False  # Exclude on filter error

            if not passes_filter:
                continue

            # Apply vector similarity if query_vector exists
            score = 0.0  # Default score if no vector query
            if query_np is not None:
                embedding = self._short_term_memory_embeddings.get(unit_id)
                if embedding is not None:
                    score = np.dot(query_np, embedding)
                    if search_range is not None:
                        min_threshold, max_threshold = search_range
                        if (min_threshold and score < min_threshold) or (max_threshold and score > max_threshold):
                            continue
                else:
                    print(f"Warning: Embedding missing for STM unit {unit_id}, cannot score.")
                    continue

            # If filters passed and similarity passed (or no vector query), add to results
            results.append((unit, score))

        # Sort results: by score (desc) if vector query, otherwise keep MRU order (already reversed)
        if query_np is not None:
            results.sort(key=lambda x: x[1], reverse=True)

        # Update LRU order for the returned items (move them to the end)
        results_ids = [unit.id for unit, score in results[:k_limit]]
        accessed_ids = set(results_ids)
        if accessed_ids:
            new_deque = deque()
            for existing_id in self._short_term_memory_ids:
                if existing_id not in accessed_ids:
                    new_deque.append(existing_id)
            new_deque.extend(results_ids)
            self._short_term_memory_ids = new_deque

        return results[:k_limit]

    def _build_milvus_filter_expression(self, filters: Optional[List[Tuple[str, str, Any]]]) -> Optional[str]:
        """Constructs a filter expression string for Milvus queries."""
        if not filters:
            return None

        expressions = []
        for (var, op, value) in filters:
            try:
                # Basic type checking and quoting
                # Ensure var name is valid Milvus field name (alphanumeric + underscore)
                if not re.fullmatch(r"[a-zA-Z0-9_]+", var):
                    print(f"Warning: Invalid character in filter variable name: {var}. Skipping filter.")
                    continue

                op = op.strip().lower()
                supported_ops = ['==', '!=', '>', '>=', '<', '<=', 'in', 'not in', 'like']  # Extend as needed

                if op not in supported_ops:
                    # Handle potential variations like '=' -> '=='
                    if op == '=':
                        op = '=='
                    elif op == '<>':
                        op = '!='
                    else:
                        print(f"Warning: Unsupported filter operator: {op}. Skipping filter.")
                        continue

                # Value formatting
                if isinstance(value, str):
                    # Escape quotes within the string if necessary
                    escaped_value = value.replace('"', '\\"')
                    formatted_value = f'"{escaped_value}"'
                elif isinstance(value, (list, tuple)) and op in ['in', 'not in']:
                    # Format list/tuple for 'in'/'not in' operator
                    # Ensure elements are correctly formatted (e.g., strings quoted)
                    formatted_elements = []
                    for item in value:
                        if isinstance(item, str):
                            escaped_item = item.replace('"', '\\"')
                            formatted_elements.append(f'"{escaped_item}"')
                        elif isinstance(item, (int, float, bool)):
                            formatted_elements.append(str(item))
                        else:
                            print(f"Warning: Unsupported type in 'in'/'not in' list: {type(item)}. Skipping filter.")
                            formatted_elements = None  # Mark as failed
                            break
                    if formatted_elements is None: continue  # Skip filter if list formatting failed
                    formatted_value = f"[{', '.join(formatted_elements)}]"
                elif isinstance(value, (int, float, bool)):
                    formatted_value = str(value)
                else:
                    print(f"Warning: Unsupported value type for filtering: {type(value)}. Skipping filter.")
                    continue

                # Construct expression part
                expressions.append(f"{var} {op} {formatted_value}")

            except Exception as e:
                print(f"Error building filter part for ({var}, {op}, {value}): {e}. Skipping filter.")
                continue

        if not expressions:
            return None

        expr_str = " and ".join(expressions)
        # print(f"Built Milvus filter expression: {expr_str}") 
        return expr_str

    def _query_ltm(self,
                   vector_store: VectorStoreHandler,
                   query_vector: Optional[List[float]] = None,
                   filters: Optional[List[Tuple[str, str, Any]]] = None,
                   k_limit: int = 5,
                   search_range: Optional[Tuple[Optional[float], Optional[float]]] = (0.75, None)
                   ) -> List[Tuple[MemoryUnit, float, np.ndarray]]:
        """
        Queries a specific Milvus vector store (internal or external LTM).

        Args:
            vector_store: The VectorStoreHandler instance to query.
            query_vector: The vector for similarity search.
            filters: Filter conditions.
            k_limit: Max results.
            search_range: Tuple (min_similarity, max_similarity) for range search (optional).
                         Uses 'radius' and 'range_filter' in Milvus search params.

        Returns:
            List of tuples: (MemoryUnit, similarity_score, embedding_vector). Score is 0.0 if no vector query.
        """
        if not vector_store:
            print("Warning: Attempted to query LTM with no vector store handler.")
            return []

        milvus_results = []
        filter_expr = self._build_milvus_filter_expression(filters)
        output_fields = ["*", "embedding"]  # Request all fields + embedding

        try:
            if query_vector is not None:
                search_params = {"metric_type": "IP", "params": {"nprobe": 10}}  # Example params
                if search_range:
                    min_sim, max_sim = search_range
                    # Milvus uses 'radius' for min similarity in IP (higher is better)
                    # and 'range_filter' for max similarity.
                    if min_sim is not None:
                        search_params['radius'] = min_sim
                    if max_sim is not None:
                        search_params['range_filter'] = max_sim  # Check if this is correct for IP max limit

                # Perform vector search
                search_results = vector_store.search(vectors=[query_vector],
                                                     top_k=k_limit,
                                                     expr=filter_expr,
                                                     search_params=search_params,
                                                     output_fields=output_fields)

                if search_results and search_results[0]:  # Results are list of lists of hits
                    for hit in search_results[0]:
                        entity_data = hit.entity  # Adjust based on actual return type
                        score = hit.distance
                        try:
                            unit = MemoryUnit(
                                memory_id=entity_data.get('id'),
                                parent_id=entity_data.get('parent_id'),
                                content=entity_data.get('content'),
                                creation_time=datetime.datetime.fromtimestamp(
                                    entity_data.get('creation_time')) if entity_data.get('creation_time') else None,
                                end_time=datetime.datetime.fromtimestamp(
                                    entity_data.get('end_time')) if entity_data.get(
                                    'end_time') else None,
                                source=entity_data.get('source'),
                                metadata=entity_data.get('metadata'),
                                last_visit=entity_data.get('last_visit'),
                                visit_count=entity_data.get('visit_count'),
                                never_delete=entity_data.get('never_delete'),
                                children_ids=entity_data.get('children_ids'),
                                rank=entity_data.get('rank'),
                                pre_id=entity_data.get("pre_id"),
                                next_id=entity_data.get("next_id")
                            )
                            embedding = np.array(entity_data.get('embedding', []))
                            if len(embedding) > 0:
                                milvus_results.append((unit, score, embedding))
                            else:
                                print(f"Warning: Unit {unit.id} from Milvus search missing embedding.")
                        except Exception as e:
                            print(
                                f"Error converting Milvus hit to MemoryUnit (ID: {entity_data.get('id', 'N/A')}): {e}")

            elif filter_expr:  # Only filtering, no vector search
                # Use query method for filtering
                query_results = vector_store.query(expr=filter_expr,
                                                   output_fields=output_fields,
                                                   top_k=k_limit)  # top_k might limit results even without vector

                for entity_data in query_results:
                    try:
                        unit = MemoryUnit(
                            memory_id=entity_data.get('id'),
                            parent_id=entity_data.get('parent_id'),
                            content=entity_data.get('content'),
                            creation_time=datetime.datetime.fromtimestamp(
                                entity_data.get('creation_time')) if entity_data.get('creation_time') else None,
                            end_time=datetime.datetime.fromtimestamp(entity_data.get('end_time')) if entity_data.get(
                                'end_time') else None,
                            source=entity_data.get('source'),
                            metadata=entity_data.get('metadata'),
                            last_visit=entity_data.get('last_visit'),
                            visit_count=entity_data.get('visit_count'),
                            never_delete=entity_data.get('never_delete'),
                            children_ids=entity_data.get('children_ids'),
                            rank=entity_data.get('rank'),
                            pre_id=entity_data.get("pre_id"),
                            next_id=entity_data.get("next_id")
                        )
                        embedding = np.array(entity_data.get('embedding', []))
                        if len(embedding) > 0:
                            milvus_results.append((unit, 0.0, embedding))  # Score is 0 for filter-only query
                        else:
                            print(f"Warning: Unit {unit.id} from Milvus query missing embedding.")
                    except Exception as e:
                        print(
                            f"Error converting Milvus query result to MemoryUnit (ID: {entity_data.get('id', 'N/A')}): {e}")
            else:
                # No query vector and no filters - invalid query? Or return random units?
                print("Warning: LTM query called with no query vector and no filters.")
                return []

        except Exception as e:
            print(f"Error during Milvus query/search: {e}")
            return []  # Return empty list on error

        # Sort by score if vector query, otherwise keep Milvus order (or sort by time?)
        if query_vector is not None:
            milvus_results.sort(key=lambda x: x[1], reverse=True)  # Higher score (IP) is better

        return milvus_results

    def _fetch_neighboring_units(self,
                                 unit: MemoryUnit,
                                 source: str = 'local_or_milvus',  # 'stm', 'local_or_milvus', 'external_milvus'
                                 find_pre: bool = True
                                 ) -> Tuple[Optional[MemoryUnit], Optional[MemoryUnit]]:
        """
        Fetches the immediate preceding and succeeding units for a given unit.

        Args:
             unit: The unit whose neighbors are needed.
             source: Where to look for neighbors ('stm', 'local_or_milvus', 'external_milvus').

        Returns:
             A tuple (preceding_unit, succeeding_unit). Units are None if not found.
        """
        pre_unit: Optional[MemoryUnit] = None
        next_unit: Optional[MemoryUnit] = None

        pre_id = unit.pre_id
        next_id = unit.next_id

        # --- Fetch Predecessor ---
        if pre_id and find_pre:
            if source == 'stm' and pre_id in self._stm_id_set:
                pre_unit = self._short_term_memory_units.get(pre_id, None)
            elif source == 'local_or_milvus':
                pre_unit = self._get_memory_unit(pre_id)  # Check local cache first
                if not pre_unit and self.vector_store:  # Try Milvus if not local
                    try:
                        results = self.vector_store.get(pre_id)
                        if results:
                            pre_unit = results
                            self.memory_units_cache[pre_id] = pre_unit  # Cache it
                    except Exception as e:
                        print(f"Error fetching neighbor {pre_id} from LTM: {e}")
            elif source == 'external_milvus' and self.external_vector_store:
                # Check local cache of external units first? Maybe not loaded. Query directly.
                try:
                    results = self.external_vector_store.get(pre_id)
                    if results:
                        pre_unit = results
                except Exception as e:
                    print(f"Error fetching neighbor {pre_id} from external LTM: {e}")

        # --- Fetch Successor ---
        if next_id and not find_pre:
            if source == 'stm' and next_id in self._stm_id_set:
                next_unit = self._short_term_memory_units.get(next_id)
            elif source == 'local_or_milvus':
                next_unit = self._get_memory_unit(next_id)  # Check local cache first
                if not next_unit and self.vector_store:  # Try Milvus if not local
                    try:
                        results = self.vector_store.get(next_id)
                        if results:
                            next_unit = results
                            self.memory_units_cache[next_id] = next_unit  # Cache it
                    except Exception as e:
                        print(f"Error fetching neighbor {next_id} from LTM: {e}")
            elif source == 'external_milvus' and self.external_vector_store:
                try:
                    results = self.external_vector_store.get(next_id)
                    if results:
                        next_unit = results
                        # self.external_memory_units_cache[next_id] = next_unit
                except Exception as e:
                    print(f"Error fetching neighbor {next_id} from external LTM: {e}")

        return pre_unit, next_unit

    def query(self,
              query_vector: Optional[List[float]] = None,
              filters: Optional[List[Tuple[str, str, Any]]] = None,
              k_limit: int = 5,
              search_range: Optional[Tuple[Optional[float], Optional[float]]] = (0.75, None),
              recall_context: bool = True,
              add_ltm_to_stm: bool = True,
              query_external_first: bool = True,
              short_term_only: bool = False,
              long_term_only: bool = False,
              external_only: bool = False,
              threshold_for_stm_before_stm: float = 0.8
              ) -> List[List[MemoryUnit]]:
        """
        Queries the memory system (STM, LTM, External LTM).

        Can query by text (generating vector), vector, and/or filters.
        Handles context recall and updating STM/visit counts.

        Args:
            query_vector: Pre-computed vector for similarity search. Overrides query_text.
            filters: List of filter conditions (attribute, operator, value).
            k_limit: Maximum number of primary results (chains or units) to return.
            search_range: Tuple (min_similarity, max_similarity) for range search (optional).
                         Uses 'radius' and 'range_filter' in Milvus search params.
            recall_context: If True, attempts to retrieve neighbors for each found unit, forming chains.
            long_term_only: If False, queries STM first. If results are found, LTM might be skipped.
            add_ltm_to_stm: If True, adds results found in LTM to STM.
            query_external_first: If True, query external LTM before internal LTM.
            short_term_only: If True, query short-term memory only.
            external_only: If True and external connection exists, only query external LTM.
            threshold_for_stm_before_stm: The higher threshold of short-term memory compared to long-term memory when using mixed memory

        Returns:
            A list of lists of MemoryUnits. Each inner list represents a chain of contextually linked units.
        """

        final_results: List[List[MemoryUnit]] = []
        processed_unit_ids: set[str] = set()  # Track units already included in results
        k_limit = max(k_limit // 3, 1) if recall_context else k_limit
        # --- 1. Query STM (Optional) ---
        if long_term_only and short_term_only:
            raise ValueError("short_term_only and long_term_only are contradictory to each other.")

        if short_term_only and not self._stm_enabled:
            raise ValueError("The user wishes to use only short-term memory, but short-term memory is not enabled.")
        if self._stm_enabled and (not long_term_only or short_term_only):
            stm_results = self._query_stm(query_vector=query_vector,
                                          filters=filters,
                                          k_limit=k_limit,
                                          search_range=search_range if short_term_only else (
                                              threshold_for_stm_before_stm, None))

            if stm_results:
                # print(f"Found {len(stm_results)} potential results in STM.")
                stm_units_with_scores = [(unit, score) for unit, score in stm_results]

                if stm_units_with_scores:
                    # Process STM results (linking, context recall)
                    core_stm_units = [unit for unit, score in stm_units_with_scores]
                    linked_stm_chains = self._process_query_results(core_stm_units,
                                                                    recall_context=recall_context,
                                                                    source='stm')
                    final_results.extend(linked_stm_chains)

                if len(final_results) > 0:
                    self._increment_interaction_round()
                    return final_results[:k_limit]  # Return up to k_limit chains
                elif short_term_only:
                    return []

        # --- 2. Query LTM (Internal and/or External) ---
        # print("Querying Long-Term Memory (LTM)...")
        ltm_results: List[Tuple[MemoryUnit, float, np.ndarray]] = []
        source_map: Dict[str, str] = {}  # Map unit ID to 'internal' or 'external'

        # Determine query order
        query_stores = []
        if external_only:
            if self.external_vector_store:
                query_stores.append(('external', self.external_vector_store))
            else:
                raise ValueError("External connection is not provided.")
        if self.vector_store and not external_only:
            query_stores.append(('internal', self.vector_store))

        if query_external_first and len(query_stores) > 1:
            query_stores.reverse()  # Put external first

        # Execute queries
        for source_name, store_handler in query_stores:
            # print(f"Querying {source_name} LTM...")
            current_store_results = self._query_ltm(vector_store=store_handler,
                                                    query_vector=query_vector,
                                                    filters=filters,
                                                    k_limit=k_limit,  # Fetch more for merging/linking
                                                    search_range=search_range)
            ltm_results.extend(current_store_results)
            for unit, _, _ in current_store_results:
                source_map[unit.id] = source_name

        # --- 3. Merge and Rank LTM Results ---
        if ltm_results:
            # print(f"Found {len(ltm_results)} potential results in LTM(s). Merging and ranking...")
            # Sort merged results by score (desc) if vector query, otherwise keep order (or sort by time?)
            if query_vector is not None:
                ltm_results.sort(key=lambda x: x[1], reverse=True)

            # Select top k unique units from LTM results
            selected_ltm_units: List[MemoryUnit] = []
            embeddings_for_stm: List[Tuple[MemoryUnit, np.ndarray]] = []
            visited_ltm_ids: set[str] = set()  # Track units selected from LTM for visit counting

            for unit, score, embedding in ltm_results:
                if len(selected_ltm_units) >= k_limit:
                    break
                if unit.id not in processed_unit_ids:
                    selected_ltm_units.append(unit)
                    processed_unit_ids.add(unit.id)  # Mark as processed
                    embeddings_for_stm.append((unit, embedding))
                    if source_map.get(unit.id) == 'internal':
                        visited_ltm_ids.add(unit.id)  # Mark internal LTM unit for visit update

            # --- 4. Process Selected LTM Results (Linking, Context Recall) ---
            if selected_ltm_units:
                linked_ltm_chains = self._process_query_results(selected_ltm_units,
                                                                recall_context=recall_context,
                                                                source='ltm',  # Special source for mixed LTM
                                                                source_map=source_map)

                # Need to handle visit counts and STM addition for *all* units in the final chains
                all_units_in_ltm_chains = list(itertools.chain(*linked_ltm_chains))

                # Refetch embeddings for neighbors if not already fetched in _process_query_results
                final_results.extend(linked_ltm_chains)

                # Update visit counts for internal LTM units involved
                self._update_visit_counts(visited_ltm_ids)  # Update for initially selected core units
                # Also update for neighbors if they were fetched from internal LTM
                neighbor_visits = set()
                for unit in all_units_in_ltm_chains:
                    if unit.id not in visited_ltm_ids and source_map.get(unit.id) == 'internal':
                        # This neighbor was likely fetched, count its visit
                        neighbor_visits.add(unit.id)
                self._update_visit_counts(neighbor_visits)

                # Add results to STM if enabled
                if self._stm_enabled and add_ltm_to_stm:
                    # print(f"Adding {len(embeddings_for_stm)} LTM results (core units) to STM.")
                    self._add_units_to_stm(embeddings_for_stm)

        # --- 5. Finalize and Return ---
        self._increment_interaction_round()  # Increment round counter after query processing

        # Ensure final results don't exceed k_limit chains
        return final_results[:k_limit]

    def _process_query_results(self,
                               core_units: List[MemoryUnit],
                               recall_context: bool,
                               source: str,  # 'stm', 'ltm' (indicates mixed internal/external possible)
                               source_map: Optional[Dict[str, str]] = None  # Needed if source='ltm'
                               ) -> List[List[MemoryUnit]]:
        """
        Helper to link units, optionally recall context, and form final chains.

        Args:
             core_units: The list of primary units found by the query.
             recall_context: Whether to fetch neighbors.
             source: The source of the units ('stm' or 'ltm').
             source_map: Required if source is 'ltm', maps unit ID to 'internal' or 'external'.

        Returns:
             List of memory unit chains.
        """
        if not core_units:
            return []

        final_chains = []
        processed_in_this_call: set[str] = set()  # Track units added to chains here

        # Link the core units first to see initial structure
        linked_core_chains = link_memory_units(core_units)

        for core_chain in linked_core_chains:
            if not core_chain:  # Skip if start already processed
                continue

            current_full_chain = list(core_chain)  # Start with the core linked units

            # --- Recall Context (Neighbors) ---
            if recall_context:
                # Fetch neighbors for the first unit in the core chain
                first_unit = core_chain[0]
                neighbor_source = 'stm' if source == 'stm' else \
                    ('external_milvus' if source_map and source_map.get(
                        first_unit.id) == 'external' else 'local_or_milvus')

                pre_unit, _ = self._fetch_neighboring_units(first_unit, source=neighbor_source, find_pre=True)
                if pre_unit and pre_unit.id not in processed_in_this_call:
                    current_full_chain.insert(0, pre_unit)

                # Fetch neighbors for the last unit in the core chain
                last_unit = core_chain[-1]
                neighbor_source = 'stm' if source == 'stm' else \
                    ('external_milvus' if source_map and source_map.get(
                        last_unit.id) == 'external' else 'local_or_milvus')

                _, next_unit = self._fetch_neighboring_units(last_unit, source=neighbor_source, find_pre=False)
                if next_unit and next_unit.id not in processed_in_this_call:
                    current_full_chain.append(next_unit)

            # Add the final chain and mark units as processed
            if current_full_chain:
                final_chains.append(current_full_chain)
                processed_in_this_call.update(unit.id for unit in current_full_chain)

        final_chains = link_memory_units(list(itertools.chain(*final_chains)))

        return final_chains

    def _update_visit_counts(self, visited_ids: set[str]):
        """Updates visit counts in cache and marks units for DB/JSON update."""
        if not visited_ids: return
        current_round = self.current_round
        units_marked_for_update = set()

        for unit_id in visited_ids:
            unit = self._get_memory_unit(unit_id, use_cache=True)  # Load if not cached
            if unit:
                self._visited_unit_counts[unit_id] += 1
                self._unit_last_visit_round[unit_id] = current_round
                units_marked_for_update.add(unit_id)  # Mark for interval flush check
            # else: print(f"Warn: Cannot update visit count for unknown unit {unit_id}")

        self._internal_visit_counter += len(units_marked_for_update)

        if self._internal_visit_counter >= self.visit_update_interval:
            print(f"Visit update interval reached. Staging {len(units_marked_for_update)} units for persistence.")
            for unit_id in units_marked_for_update:
                unit = self.memory_units_cache.get(unit_id)
                if unit:
                    # Apply cumulative counts/round from trackers
                    unit.visit_count += self._visited_unit_counts.get(unit_id, 0)
                    unit.last_visit = self._unit_last_visit_round.get(unit_id, unit.last_visit)
                    # Stage the update (embedding usually not needed for just visit count)
                    self._stage_memory_unit_update(unit, operation='update', update_session_metadata=False)
            # Reset trackers
            self._visited_unit_counts.clear()
            self._unit_last_visit_round.clear()
            self._internal_visit_counter = 0
            # Trigger flush (might include more than just visit updates)
            self._flush_cache()

    def _increment_interaction_round(self):
        """Increments the global interaction round counter."""
        self.current_round += 1
        self.long_term_memory.visit_count = self.current_round  # Keep LTM object updated

    def _evaluate_python_filter(self, obj: Any, attr_expr: str, op: str, value: Any) -> bool:
        """
        Evaluates a filter condition on an object attribute using Python operators.

        Supports nested attribute access like 'metadata.key' or 'metadata["key"]'.
        Uses restricted evaluation for safety but relies on correct operator mapping.

        Args:
            obj: The object to evaluate the filter on (e.g., a MemoryUnit).
            attr_expr: The attribute name or expression (e.g., 'source', 'metadata.action').
            op: The comparison operator (e.g., '==', '>', 'in').
            value: The value to compare against.

        Returns:
            True if the condition is met, False otherwise or on error.

        Raises:
             AttributeError: If the attribute expression is invalid for the object.
             ValueError: If the operator is unsupported or comparison fails.
        """

        def get_nested_attr(target_obj, expression):
            """Safely gets nested attributes or dict/list items."""
            parts = expression.split('.')
            current_val = target_obj
            for part in parts:
                if '[' in part and part.endswith(']'):  # Handle dict/list access like 'key["subkey"]' or 'list[0]'
                    attr_name, index_part = part.split('[', 1)
                    index_str = index_part[:-1].strip('"\'')  # Get content inside []

                    if attr_name:  # Access attribute first, then index
                        if not hasattr(current_val, attr_name):
                            raise AttributeError(f"Object has no attribute '{attr_name}' in expression '{expression}'")
                        current_val = getattr(current_val, attr_name)

                    # Now access using index/key
                    try:
                        # Try as string key first
                        current_val = current_val[index_str]
                    except (TypeError, KeyError):
                        try:  # Try as integer index
                            current_val = current_val[int(index_str)]
                        except (ValueError, IndexError, KeyError, TypeError):
                            raise AttributeError(f"Cannot access index/key '{index_str}' in expression '{expression}'")
                else:  # Simple attribute access
                    if not hasattr(current_val, part):
                        raise AttributeError(f"Object has no attribute '{part}' in expression '{expression}'")
                    current_val = getattr(current_val, part)
            return current_val

        try:
            attribute_value = get_nested_attr(obj, attr_expr)

            # Get the comparison function from the operator module
            op_func = {
                '==': operator.eq, '=': operator.eq,  # Allow = as alias for ==
                '!=': operator.ne, '<>': operator.ne,  # Allow <> as alias for !=
                '>': operator.gt,
                '>=': operator.ge,
                '<': operator.lt,
                '<=': operator.le,
                'in': lambda a, b: a in b,
                'not in': lambda a, b: a not in b,
                'contains': lambda a, b: b in a if isinstance(a, (str, list, tuple, dict)) else False
            }.get(op.strip().lower())

            if op_func is None:
                raise ValueError(f"Unsupported filter operator: '{op}'")

            # Perform the comparison
            return op_func(attribute_value, value)

        except (AttributeError, ValueError, TypeError, IndexError, KeyError) as e:
            # print(f"Filter evaluation error: {e} for ({attr_expr} {op} {value})") 
            return False  # Condition fails on error

    # --- Summarization ---
    def _stage_long_term_memory_update(self, ltm: LongTermMemory):
        if ltm.id == self.ltm_id:
            self.long_term_memory = ltm  # Update cache
            self._updated_ltm_ids.add(ltm.id)  # Mark for save

    def _stage_session_memory_update(self, session: SessionMemory):
        if session.id in self.session_memories_cache or session.id == self._current_session_id:  # Only stage if relevant
            self.session_memories_cache[session.id] = session  # Update cache
            self._updated_session_memory_ids.add(session.id)  # Mark for save
            self._deleted_session_memory_ids.discard(session.id)  # Unmark delete

    def _stage_memory_units_update(self, units: Iterable[MemoryUnit], operation: str = 'add'):
        """Helper to stage multiple units, useful after summarization."""
        # print(f"Staging {len(list(units))} units for operation: {operation}")  
        for unit in units:
            self._stage_memory_unit_update(unit, operation=operation, embedding=None, update_session_metadata=False)

    def summarize_long_term_memory(self, use_external_summary: bool = True, role: str = "ai"):
        """Orchestrates summarizing the current LTM."""
        if not self.llm or not self.long_term_memory:
            print("Warning: LLM or LTM object missing.")
            return

        history_summary_unit: Optional[MemoryUnit] = None
        if use_external_summary and self.external_vector_store and self.external_ltm_id:
            try:
                results = self.external_vector_store.get(self.external_ltm_id)
                if results:
                    history_summary_unit = results
            except Exception as e:
                # print(f"Could not fetch external history summary: {e}")
                pass
        try:
            updated_ltm, new_units, updated_units = summarize_long_term_memory(
                memory_system=self,
                ltm_id=self.ltm_id,
                llm=self.llm,
                history_memory=history_summary_unit,
                role=role,
                max_group_size=self.max_group_size,
                max_token_count=self.max_token_count
            )
            # Stage results
            if updated_ltm: self._stage_long_term_memory_update(updated_ltm)
            if new_units: self._stage_memory_units_update(new_units, operation='add')
            if updated_units: self._stage_memory_units_update(updated_units, operation='update')

            if new_units or updated_units: self._flush_cache(force=True)

        except Exception as e:
            print(f"Error during LTM {self.ltm_id} summarization orchestration: {e}")

    def summarize_session(self, session_id: str, role: str = "ai"):
        """Orchestrates summarizing a specific session."""
        if not self.llm: print("Warning: LLM not configured."); return
        try:
            # Call refactored function, passing self
            updated_session, new_units, updated_units = summarize_session(
                memory_system=self,
                session_id=session_id,
                llm=self.llm,
                role=role,
                max_group_size=self.max_group_size,
                max_token_count=self.max_token_count
            )
            # Stage results for persistence
            if updated_session: self._stage_session_memory_update(updated_session)
            if new_units: self._stage_memory_units_update(new_units, operation='add')  # Stage new summaries
            if updated_units: self._stage_memory_units_update(updated_units,
                                                              operation='update')

            if new_units or updated_units: self._flush_cache(force=True)

        except Exception as e:
            print(f"Error during session {session_id} summarization orchestration: {e}")

    # --- Persistence Management ---

    def convert_json_to_sqlite(self):
        """Loads data from JSON files and saves it into the SQLite database."""

        if not self.persistence_mode == "json":
            print("Error: JSON files are not available for conversion.")
            return

        try:
            # Load from JSON using original handlers
            print("Loading from JSON...")
            json_units = json_handler.load_memory_units_json(self.chatbot_id,self.storage_base_path)
            mus = [MemoryUnit.from_dict(mu_dict) for mu_dict in json_units.values()]
            json_sessions = json_handler.load_session_memories_json(self.chatbot_id,self.storage_base_path)
            sms = [SessionMemory.from_dict(sm_dict) for sm_dict in json_sessions.values()]
            # Load all LTMs for the chatbot from JSON
            json_ltms = json_handler.load_long_term_memories_json(self.chatbot_id,self.storage_base_path)
            ltms = [LongTermMemory.from_dict(ltm_dict) for ltm_dict in json_ltms.values()]

            print(f"Loaded {len(json_units)} units, {len(json_sessions)} sessions, {len(json_ltms)} LTMs from JSON.")

            # Save to SQLite using new handlers
            print("Saving to SQLite...")
            sqlite_handler.initialize_db(chatbot_id=self.chatbot_id,base_path=self.storage_base_path)
            sql_conn = sqlite_handler.get_connection(self.chatbot_id,self.storage_base_path)
            if json_units:
                sqlite_handler.upsert_memory_units(sql_conn, mus)
            if json_sessions:
                sqlite_handler.upsert_session_memories(sql_conn, sms)
            if json_ltms:
                sqlite_handler.upsert_long_term_memories(sql_conn, ltms)
            sql_conn.commit()
            print("JSON to SQLite conversion complete.")

            # Reload current state from DB after conversion
            print("Reloading current state from SQLite after conversion...")

        except Exception as e:
            print(f"Error during JSON to SQLite conversion: {e}")

    def convert_json_to_milvus(self, embedding_history_length: int = 1):
        """
        Loads all memory units from JSON and inserts/updates them into Milvus.

        Use with caution, as it re-embeds and inserts all known units.

        Args:
            embedding_history_length: Context length used for generating embeddings.
        """
        if not self.vector_store:
            print("Error: Milvus vector store is not configured. Cannot convert.")
            return

        if not self.persistence_mode == "json":
            print("Error: JSON files are not configured. Cannot convert.")
            return

        print("Starting conversion of all JSON memory units to Milvus...")
        json_units = json_handler.load_memory_units_json(self.chatbot_id, self.storage_base_path)
        all_units = [MemoryUnit.from_dict(mu_dict) for mu_dict in json_units.values()]

        print(f"Loaded {len(all_units)} units from JSON.")
        if not all_units:
            return

        units_to_upsert = []
        processed_count = 0
        error_count = 0

        for unit in all_units:
            if not unit:
                continue
            try:
                content_to_embed = self._get_formatted_content_with_history(unit,
                                                                            history_length=embedding_history_length)
                embedding = self.embedding_handler.get_embedding(content_to_embed)
                milvus_data = {
                    "id": unit.id,
                    "parent_id": unit.parent_id,
                    "content": unit.content,
                    "creation_time": unit.creation_time.timestamp() if unit.creation_time else None,
                    "end_time": unit.end_time.timestamp() if unit.end_time else None,
                    "source": unit.source,
                    "metadata": unit.metadata,
                    "last_visit": unit.last_visit,
                    "visit_count": unit.visit_count,
                    "never_delete": unit.never_delete,
                    "children_ids": unit.children_ids,
                    "embedding": embedding.tolist(),
                    'rank': unit.rank,
                    'pre_id': unit.pre_id,
                    'next_id': unit.next_id
                }
                units_to_upsert.append(milvus_data)
                processed_count += 1
            except Exception as e:
                error_count += 1
                print(f"Error processing unit {unit.id} for Milvus conversion: {e}")

            # Insert in batches to avoid large memory usage
            # if len(units_to_upsert) >= 100:
            #      print(f"Inserting batch of {len(units_to_upsert)} units to Milvus...")
            #      try:
            #           self.vector_store.upsert(units_to_upsert) # Assuming upsert exists
            #           self.vector_store.flush()
            #      except Exception as e:
            #           print(f"Error inserting batch to Milvus: {e}")
            #           error_count += len(units_to_upsert) # Count batch as errors
            #      units_to_upsert = []

        # Insert any remaining units
        if units_to_upsert:
            print(f"Inserting final batch of {len(units_to_upsert)} units to Milvus...")
            try:
                self.vector_store.upsert(units_to_upsert)
                self.vector_store.flush()
            except Exception as e:
                print(f"Error inserting final batch to Milvus: {e}")
                error_count += len(units_to_upsert)

        print(f"JSON to Milvus conversion finished.")
        print(f"Successfully processed: {processed_count - error_count} units.")
        print(f"Errors encountered: {error_count} units.")

    def close(self, auto_summarize: bool = False, role: str = "ai"):
        """Flushes changes, optionally summarizes, closes connections."""
        print(f"Closing MemorySystem ({self.persistence_mode} mode)...")
        # ... (Stage final visit counts logic - unchanged conceptually) ...
        if self._visited_unit_counts:
            for unit_id, count_increase in self._visited_unit_counts.items():
                unit = self.memory_units_cache.get(unit_id)
                if unit:
                    unit.visit_count += count_increase
                    unit.last_visit = self._unit_last_visit_round.get(unit_id, unit.last_visit)
                    self._stage_memory_unit_update(unit, operation='update', update_session_metadata=False)
            self._visited_unit_counts.clear()
            self._unit_last_visit_round.clear()
            self._internal_visit_counter = 0

        # Flush final data
        print("Flushing final cache before close...")
        self._flush_cache(force=True)

        # Auto-summarize
        if auto_summarize and self.llm:
            print("Performing final auto-summarization...")
            try:
                self.summarize_long_term_memory(use_external_summary=True, role=role)
                print("Flushing cache after final summarization...")
                self._flush_cache(force=True)  # Flush again after summarization
            except Exception as e:
                print(f"Error during final auto-summarization: {e}")

        # Close connections
        self._close_sqlite()
        if self.vector_store:
            try:
                self.vector_store.close()
            except Exception as e:
                print(f"Error closing vector store: {e}")
        if self.external_vector_store is not None:
            self.external_vector_store.close(flush=False)

        print("MemorySystem closed.")
