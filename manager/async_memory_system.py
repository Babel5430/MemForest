import asyncio
import os
import datetime
import operator
import re
from collections import defaultdict, deque
from typing import (Any, Dict, Iterable, List, Optional, Tuple, Union, Set)
from uuid import uuid4
import numpy as np
from langchain_core.messages import BaseMessage

from MemForest.memory.long_term_memory import LongTermMemory
from MemForest.memory.memory_unit import MemoryUnit
from MemForest.memory.session_memory import SessionMemory
from MemForest.manager.forgetting import forget_memories
from MemForest.manager.summarizing import (
    summarize_long_term_memory, summarize_session
)
from MemForest.persistence.filter_types import BaseFilter, FieldFilter, LogicalFilter, FilterOperator
from MemForest.utils.embedding_handler import EmbeddingHandler
from MemForest.persistence.vector_store_handler import VectorStoreHandler
from MemForest.persistence.sqlite_handler import AsyncSQLiteHandler
from MemForest.utils.helper import link_memory_units
from MemForest.persistence import json_handler

# --- Constants ---

DEFAULT_VISIT_UPDATE_INTERVAL = 40
DEFAULT_SAVING_INTERVAL = 40
DEFAULT_MAX_VECTOR_ENTITIES = 20000
DEFAULT_FORGET_PERCENTAGE = 0.10
DEFAULT_MAX_GROUP_SIZE = 60
DEFAULT_MAX_TOKEN_COUNT = 2000
DEFAULT_EMBEDDING_DIM = 512

def _parse_filter_json(filter_json: Optional[Dict[str, Any]]) -> Optional[BaseFilter]:
    """
    Chroma-style format:
       - Field filter: {"field_name": {"$operator": "value"}} (e.g., {"rank": {"$gte": 0}})
       - Logical filter: {"$operator": [list of filters]} (e.g., {"$and": [...]})
       - Logical filter (NOT): {"$not": {single filter}}
    """
    if filter_json is None:
        return None

    if not isinstance(filter_json, dict):
        print(f"Error parsing filter JSON: Input is not a dictionary: {filter_json}")
        return None

    if "$and" in filter_json:
        sub_filters_json = filter_json.get("$and")
        if not isinstance(sub_filters_json, list) or not sub_filters_json:
            print(f"Error parsing filter JSON: '$and' filter requires a non-empty list.")
            return None
        parsed_sub_filters = [_parse_filter_json(f_json) for f_json in sub_filters_json]
        valid_sub_filters = [f for f in parsed_sub_filters if f]
        if len(valid_sub_filters) != len(sub_filters_json):
             print(f"Warning: Some sub-filters in '$and' filter were invalid and skipped.")
             if not valid_sub_filters:
                  return None
        try:
            return LogicalFilter("AND", *valid_sub_filters)
        except ValueError as e:
            print(f"Error creating LogicalFilter (AND): {e}")
            return None

    elif "$or" in filter_json:
        sub_filters_json = filter_json.get("$or")
        if not isinstance(sub_filters_json, list) or not sub_filters_json:
            print(f"Error parsing filter JSON: '$or' filter requires a non-empty list.")
            return None
        parsed_sub_filters = [_parse_filter_json(f_json) for f_json in sub_filters_json]
        valid_sub_filters = [f for f in parsed_sub_filters if f]
        if len(valid_sub_filters) != len(sub_filters_json):
             print(f"Warning: Some sub-filters in '$or' filter were invalid and skipped.")
             if not valid_sub_filters:
                  return None
        try:
            return LogicalFilter("OR", *valid_sub_filters)
        except ValueError as e:
            print(f"Error creating LogicalFilter (OR): {e}")
            return None

    elif "$not" in filter_json:
        sub_filter_json = filter_json.get("$not")
        if not isinstance(sub_filter_json, dict):
            print(f"Error parsing filter JSON: '$not' filter requires a single dictionary.")
            return None
        parsed_sub_filter = _parse_filter_json(sub_filter_json)
        if not parsed_sub_filter:
            print(f"Error parsing filter JSON: Sub-filter for '$not' is invalid.")
            return None
        try:
            return LogicalFilter("NOT", parsed_sub_filter)
        except ValueError as e:
            print(f"Error creating LogicalFilter (NOT): {e}")
            return None

    operator_str_original = filter_json.get("operator")
    if operator_str_original in ["AND", "OR"]:
         sub_filters_json = filter_json.get("filters")
         if not isinstance(sub_filters_json, list) or not sub_filters_json:
             print(f"Error parsing filter JSON: '{operator_str_original}' filter requires a non-empty list of 'filters'.")
             return None
         parsed_sub_filters = [_parse_filter_json(f_json) for f_json in sub_filters_json]
         valid_sub_filters = [f for f in parsed_sub_filters if f]
         if len(valid_sub_filters) != len(sub_filters_json):
              print(f"Warning: Some sub-filters in '{operator_str_original}' filter were invalid and skipped.")
              if not valid_sub_filters:
                   return None # If no valid sub-filters, the logical filter is invalid
         try:
             return LogicalFilter(operator_str_original, *valid_sub_filters)
         except ValueError as e:
             print(f"Error creating LogicalFilter: {e}")
             return None

    elif operator_str_original == "NOT":
         sub_filter_json = filter_json.get("filter")
         if not isinstance(sub_filter_json, dict):
             print(f"Error parsing filter JSON: 'NOT' filter requires a single 'filter' dictionary.")
             return None
         parsed_sub_filter = _parse_filter_json(sub_filter_json)
         if not parsed_sub_filter:
             print(f"Error parsing filter JSON: Sub-filter for 'NOT' is invalid.")
             return None
         try:
             return LogicalFilter(operator_str_original, parsed_sub_filter)
         except ValueError as e:
             print(f"Error creating LogicalFilter: {e}")
             return None

    if len(filter_json) == 1:
        field_name = list(filter_json.keys())[0]
        operator_value_dict = filter_json.get(field_name)
        if isinstance(operator_value_dict, dict) and len(operator_value_dict) == 1:
            chroma_operator = list(operator_value_dict.keys())[0]
            value = list(operator_value_dict.values())[0]
            if field_name.startswith('$'):
                 print(f"Error parsing filter JSON: Expected field name, found logical operator '{field_name}'.")
                 return None

            # Map Chroma operators to FilterOperator enum
            operator_map = {
                "$eq": FilterOperator.EQ,
                "$ne": FilterOperator.NE,
                "$gt": FilterOperator.GT,
                "$gte": FilterOperator.GTE,
                "$lt": FilterOperator.LT,
                "$lte": FilterOperator.LTE,
                "$in": FilterOperator.IN,
                "$nin": FilterOperator.NOT_IN,
                "$contains": FilterOperator.CONTAINS,
                "$like": FilterOperator.LIKE,
            }
            operator_enum = operator_map.get(chroma_operator.lower())

            if operator_enum is None:
                 print(f"Error parsing filter JSON: Invalid Chroma operator '{chroma_operator}' for field '{field_name}'.")
                 return None

            try:
                return FieldFilter(field_name, operator_enum, value)
            except ValueError as e:
                print(f"Error creating FieldFilter from Chroma style: {e} for {filter_json}")
                return None

    if "field" in filter_json and "operator" in filter_json and "value" in filter_json:
        field_name = filter_json["field"]
        operator_val = filter_json["operator"]
        value = filter_json["value"]

        if not isinstance(field_name, str) or not field_name:
            print(f"Error parsing filter JSON: Field name must be a non-empty string: {filter_json}")
            return None
        try:
            operator_enum = FilterOperator[str(operator_val).upper()]
        except KeyError:
            try:
                operator_enum = FilterOperator(str(operator_val).lower())
            except ValueError:
                print(f"Error parsing filter JSON: Invalid operator value '{operator_val}' for field '{field_name}'.")
                return None

        try:
            return FieldFilter(field_name, operator_enum, value)
        except ValueError as e:
            print(f"Error creating FieldFilter (original style): {e} for {filter_json}")
            return None
    else:
        print(f"Error parsing filter JSON: Unknown filter structure: {filter_json}")
        return None

class AsyncMemorySystem:
    """
    Manages chatbot memory using AsyncSQLiteHandler for primary persistence
    and VectorStoreHandler for vector operations (supporting Milvus, Chroma, sqlite-vec).
    Includes caching, auto-forgetting, summarization, and asynchronous operations.
    Operates on data relevant to the initialized ltm_id.
    """

    def __init__(self,
                 chatbot_id: str,
                 ltm_id: str,
                 embedding_handler: EmbeddingHandler,
                 llm: Optional['BaseChatModel'] = None,
                 vector_store_config: Optional[Dict[str, Any]] = None,
                 base_path: Optional[str] = None,
                 # persistence_mode: str = 'sqlite', # Remove now
                 max_context_length: int = 12,
                 visit_update_interval: int = DEFAULT_VISIT_UPDATE_INTERVAL,
                 saving_interval: int = DEFAULT_SAVING_INTERVAL,
                 max_vector_entities: int = DEFAULT_MAX_VECTOR_ENTITIES,
                 forget_percentage: float = DEFAULT_FORGET_PERCENTAGE,
                 max_group_size: int = DEFAULT_MAX_GROUP_SIZE,
                 max_token_count: int = DEFAULT_MAX_TOKEN_COUNT
                 ):
        self.chatbot_id: str = chatbot_id
        self.ltm_id: str = ltm_id
        self.llm: Optional['BaseChatModel'] = llm
        self.embedding_handler: EmbeddingHandler = embedding_handler
        self.vector_store_config: Dict[str, Any] = vector_store_config if vector_store_config else {
            "type": "qdrant"}  # Default to sqlite-vec if no config
        self.storage_base_path: str = base_path if base_path is not None else os.path.join(os.getcwd(),
                                                                                           "memory_storage")

        # --- Ensure embedding_dim is available ---
        self.embedding_dim = self.vector_store_config.get("embedding_dim", self.embedding_handler.dimension if hasattr(
            self.embedding_handler, 'dimension') else DEFAULT_EMBEDDING_DIM)
        if not self.embedding_dim or not isinstance(self.embedding_dim, int) or self.embedding_dim <= 0:
            raise ValueError(f"Invalid or missing 'embedding_dim' ({self.embedding_dim}). Must be a positive integer.")
        if "embedding_dim" not in self.vector_store_config:
            self.vector_store_config["embedding_dim"] = self.embedding_dim

        self.visit_update_interval: int = visit_update_interval
        self.saving_interval: int = saving_interval
        self.max_vector_entities: int = max_vector_entities
        self.forget_percentage: float = forget_percentage
        self.max_group_size: int = max_group_size
        self.max_token_count: int = max_token_count

        # --- Handlers (Initialized in async _initialize) ---
        self.sqlite_handler: Optional[AsyncSQLiteHandler] = None
        self.vector_store: Optional[VectorStoreHandler] = None

        # --- Tracking Changes (remains conceptually same) ---
        self._mu_new_ids: set[str] = set()
        self._mu_content_updated_ids: set[str] = set()
        self._mu_core_data_updated_ids: set[str] = set()
        self._mu_parent_updated_links: Dict[str, Optional[str]] = {} # child_id -> new_parent_id
        self._mu_children_updated_links: Dict[str, List[str]] = {} # parent_id -> new_children_ids
        self._mu_sequence_updated_links: Dict[str, Tuple[Optional[str], Optional[str]]] = {} # unit_id -> (pre_id, next_id)
        self._mu_group_updated_links: Dict[str, Tuple[Union[str, List[str]], int]] = {} # unit_id -> (group_id(s), rank)
        self._deleted_memory_unit_ids: set[str] = set()

        self._updated_session_memory_ids: set[str] = set()
        self._updated_ltm_ids: set[str] = set()
        self._deleted_session_memory_ids: set[str] = set()
        self._deleted_ltm_ids: set[str] = set()

        # --- In-Memory Caches (Scoped to current LTM) ---
        self.memory_units_cache: Dict[str, MemoryUnit] = {}
        self.session_memories_cache: Dict[str, SessionMemory] = {}
        self.long_term_memory: Optional[LongTermMemory] = None
        self._ltm_cache: Optional[Dict[str, LongTermMemory]] = {}

        # --- Internal Counters and Queues ---
        self._internal_visit_counter: int = 0
        self._staged_updates_count: int = 0
        self._stm_enabled: bool = False
        self._stm_capacity: int = 0
        self._short_term_memory_ids: deque[str] = deque()
        self._short_term_memory_embeddings: Dict[str, np.ndarray] = {}  # Stores numpy arrays
        self._short_term_memory_units: Dict[str, MemoryUnit] = {}
        self._stm_id_set: set[str] = set()
        self._max_context_length: int = max(1,max_context_length)
        self.context: deque[MemoryUnit] = deque()
        self._context_ids: set[str] = set()
        self._last_context_unit: Optional[MemoryUnit] = None
        self.current_round: int = 0  # Initialized after loading state

        # --- External Connection State ---
        self.external_chatbot_id: Optional[str] = None
        self.external_ltm_id: Optional[str] = None
        self.external_vector_store: Optional[VectorStoreHandler] = None
        self.external_sqlite_handler: Optional[AsyncSQLiteHandler] = None

        # --- Visit tracking ---
        self._visited_unit_counts: Dict[str, int] = defaultdict(int)
        self._unit_last_visit_round: Dict[str, int] = defaultdict(int)
        self._current_session_id: Optional[str] = None
        self._is_initialized: bool = False
        self._initialization_lock = None

    async def _async_initialize(self):
        """Asynchronous initialization logic."""
        if self._initialization_lock is None:
            self._initialization_lock = asyncio.Lock()
        async with self._initialization_lock:
            if self._is_initialized:
                return
            print("Initializing MemorySystem asynchronously...")
            self.sqlite_handler = AsyncSQLiteHandler()
            sqlite_config = self.vector_store_config.get('sqlite', {})
            sqlite_base_path = sqlite_config.get('base_path', self.storage_base_path)
            db_type = self.vector_store_config.get('type', 'qdrant')
            self.sqlite_handler.initialize(
                chatbot_id=self.chatbot_id,
                base_path=sqlite_base_path,
                embedding_dim=self.embedding_dim,
                use_sqlite_vec=(db_type == 'sqlite-vec')
            )
            await self.sqlite_handler.initialize_db()
            print("AsyncSQLiteHandler (new version) initialized.")
            try:
                self.vector_store = VectorStoreHandler(
                    chatbot_id=self.chatbot_id,
                    long_term_memory_id=self.ltm_id,
                    config=self.vector_store_config
                )
                if self.vector_store.vector_db_type == 'sqlite-vec':
                    self.vector_store._sqlite_handler = self.sqlite_handler
                await self.vector_store.initialize()
                print(f"VectorStoreHandler initialized with type: {self.vector_store.vector_db_type}")
            except Exception as e:
                print(f"Warning: Failed to initialize VectorStoreHandler. Error: {e}")
                self.vector_store = None
            await self._load_initial_state()
            # await self.start_session()
            self.current_round = self.long_term_memory.visit_count if self.long_term_memory else 0
            self._is_initialized = True
            print(f"MemorySystem for {self.chatbot_id}/{self.ltm_id} initialized.")

    async def ensure_initialized(self):
        """Ensures the system is initialized before proceeding."""
        if not self._is_initialized:
            await self._async_initialize()

    def get_current_sesssion_id(self):
        return self._current_session_id

    async def _load_initial_state(self):
        """Loads the primary LTM and its associated session metadata using AsyncSQLiteHandler."""
        if not self.sqlite_handler:
            raise RuntimeError("SQLite handler not initialized before loading state.")

        print(f"Loading initial state for LTM {self.ltm_id} (async)...")
        self.memory_units_cache.clear()
        self.session_memories_cache.clear()
        self.long_term_memory = None

        try:
            self.long_term_memory = await self.sqlite_handler.load_long_term_memory(self.ltm_id, self.chatbot_id, include_edges=True)
            if self.long_term_memory:
                if self.long_term_memory.session_ids:
                    self.session_memories_cache = await self.sqlite_handler.load_session_memories(
                        self.long_term_memory.session_ids, include_edges=True)

                # Load all necessary memory units (summaries, session units) for this LTM
                unit_ids_to_load = set(self.long_term_memory.summary_unit_ids)
                unit_ids_to_load.add(self.ltm_id)
                for sm_id, sm in self.session_memories_cache.items():
                    unit_ids_to_load.add(sm_id)  # Session summary unit
                    unit_ids_to_load.update(sm.memory_unit_ids)  # Units within sessions

                if unit_ids_to_load:
                    self.memory_units_cache = await self.sqlite_handler.load_memory_units(list(unit_ids_to_load), include_edges=True)
            else:
                # LTM doesn't exist, create a new one in memory
                print(f"LTM {self.ltm_id} not found in SQLite. Creating new LTM state.")
                self.long_term_memory = LongTermMemory(chatbot_id=self.chatbot_id, ltm_id=self.ltm_id)
                self._updated_ltm_ids.add(self.long_term_memory.id)  # Mark for saving
                self._ltm_cache[self.ltm_id] = self.long_term_memory

            print(f"Loaded LTM {self.ltm_id} with {len(self.long_term_memory.session_ids)} sessions.")
            print(f"Loaded {len(self.session_memories_cache)} session objects into cache.")
            print(f"Loaded {len(self.memory_units_cache)} memory units into cache.")

        except Exception as e:
            print(f"FATAL: Error loading initial state from SQLite: {e}")
            raise ConnectionError(f"Failed to load initial state from SQLite: {e}") from e

    async def start_session(self, session_id: Optional[str] = None):
        """Starts/resumes session, using async SQLite handler."""
        # await self.ensure_initialized()
        if not self.sqlite_handler or not self.long_term_memory:
            print("Error: Cannot start session, system not properly initialized.")
            return

        target_session_id = session_id if session_id is not None else str(uuid4())

        if target_session_id == 'LATEST':
            target_session_id = self.long_term_memory.last_session_id

        if target_session_id:
            if session_id:
                # Check cache first
                session_obj = self.session_memories_cache.get(target_session_id)
                if not session_obj:
                    # If not in cache, try loading from SQLite DB
                    session_obj = await self.sqlite_handler.load_session_memory(target_session_id)
                    if session_obj:
                        self.session_memories_cache[target_session_id] = session_obj  # Add to cache

                if session_obj:
                    self._current_session_id = target_session_id
                    print(f"Resuming session: {target_session_id}")
                    await self._restore_session(self._current_session_id)
                    session_summary_unit = await self._get_memory_unit(self._current_session_id)
                    if session_summary_unit:
                        is_linked_to_ltm = False
                        current_group_id = session_summary_unit.group_id
                        if isinstance(current_group_id, str) and current_group_id == self.ltm_id:
                            is_linked_to_ltm = True
                        elif isinstance(current_group_id, list) and self.ltm_id in current_group_id:
                            is_linked_to_ltm = True

                        if not is_linked_to_ltm:
                            new_group_id_val = self.ltm_id
                            if isinstance(current_group_id, list):
                                new_group_id_val = list(set(current_group_id + [self.ltm_id]))
                            elif isinstance(current_group_id,str) and current_group_id:
                                new_group_id_val = [current_group_id, self.ltm_id] if current_group_id != self.ltm_id else current_group_id

                            if session_summary_unit.group_id != new_group_id_val:
                                await self._stage_memory_unit_update(
                                    memory_unit=session_summary_unit,
                                    unit_id=session_summary_unit.id,
                                    operation="edge_update",
                                    update_type="group",
                                    update_details=(new_group_id_val, session_summary_unit.rank)
                                )
                    return
                else:
                    print(f"Warning: Requested/Last session {target_session_id} not found. Starting new.")
            new_id = str(uuid4())
            self._current_session_id = new_id
            print(f"Starting new session: {new_id}")
            creation_time = datetime.datetime.now()
            new_sm = SessionMemory(session_id=new_id, creation_time=creation_time, end_time=creation_time)
            self.session_memories_cache[new_id] = new_sm
            await self._stage_session_memory_update(new_sm)
            if new_id not in self.long_term_memory.session_ids:
                self.long_term_memory.session_ids.append(new_id)
            self.long_term_memory.last_session_id = new_id
            await self._stage_long_term_memory_update(self.long_term_memory)

    # --- Core Methods (Async) ---

    async def _flush_cache(self, force: bool = False):
        """Saves all staged changes to SQLite and syncs the vector store."""
        await self.ensure_initialized()
        if not self.sqlite_handler:
            print("Error: Cannot flush, SQLite handler not available.")
            return

        has_any_updates = bool(
            self._mu_new_ids or self._mu_content_updated_ids or self._mu_core_data_updated_ids or
            self._mu_parent_updated_links or self._mu_children_updated_links or
            self._mu_sequence_updated_links or self._mu_group_updated_links or
            self._updated_session_memory_ids or self._updated_ltm_ids or
            self._deleted_memory_unit_ids or self._deleted_session_memory_ids or self._deleted_ltm_ids
        )

        if not force and self._staged_updates_count < self.saving_interval and not has_any_updates:
            return
        if not has_any_updates and self._staged_updates_count == 0:
            return

        print(f"Flushing cache (Forced: {force})... Staged general count: {self._staged_updates_count}")

        # --- Prepare Data for Vector Store (External Only) ---
        units_requiring_embedding_update_for_external_vs: Dict[str, MemoryUnit] = {}
        embedding_cache_for_flush: Dict[str, np.ndarray] = {}  # unit_id -> embedding_np
        ids_for_embedding_generation = self._mu_new_ids.union(self._mu_content_updated_ids)
        if self.vector_store and self.vector_store.vector_db_type != 'sqlite-vec':
            for unit_id in ids_for_embedding_generation:
                if unit_id in self.memory_units_cache:
                    units_requiring_embedding_update_for_external_vs[unit_id] = self.memory_units_cache[unit_id]

        if ids_for_embedding_generation:
            generation_tasks = []
            units_for_generation_list = []
            for unit_id, unit_obj in units_requiring_embedding_update_for_external_vs.items():
                if unit_id in self._short_term_memory_embeddings:
                    embedding_cache_for_flush[unit_id] = self._short_term_memory_embeddings[unit_id]
                else:
                    units_for_generation_list.append(unit_obj)
                    generation_tasks.append(self._generate_embedding_for_unit(unit_obj, history_length=1))

            if generation_tasks:
                embedding_results = await asyncio.gather(*generation_tasks, return_exceptions=True)
                for i, unit_obj in enumerate(units_for_generation_list):
                    result = embedding_results[i]
                    if isinstance(result, np.ndarray):
                        embedding_cache_for_flush[unit_obj.id] = result
                    else:
                        print(f"Warning: Failed to generate embedding for {unit_obj.id} during flush: {result}")

        # --- SQLite Operations ---
        try:
            # async with self.sqlite_handler._get_connection() as conn:  # Use a single transaction for all SQLite ops
            await self.sqlite_handler.begin()
            # 1. Deletions
            if self._deleted_memory_unit_ids:
                await self.sqlite_handler.delete_memory_units(list(self._deleted_memory_unit_ids))
            if self._deleted_session_memory_ids:
                await self.sqlite_handler.delete_session_memories(list(self._deleted_session_memory_ids))
            if self._deleted_ltm_ids:
                for ltm_id in self._deleted_ltm_ids:
                   await self.sqlite_handler.delete_long_term_memory_record(ltm_id)


            # 2. New Memory Units
            all_mu_ids_to_persist = self._mu_new_ids.union(self._mu_content_updated_ids).union(
                self._mu_core_data_updated_ids)
            mu_upsert_dicts = []
            for unit_id in all_mu_ids_to_persist:
                if unit_id in self.memory_units_cache:
                    unit = self.memory_units_cache[unit_id]
                    unit_dict_for_upsert = unit.to_dict()  # Base data from MemoryUnit object

                    # Add embedding if available (for sqlite-vec, which upsert_memory_units handles)
                    if self.sqlite_handler.use_sqlite_vec and unit_id in embedding_cache_for_flush:
                        unit_dict_for_upsert['embedding'] = embedding_cache_for_flush[unit_id].flatten().tolist()
                    elif 'embedding' in unit_dict_for_upsert and not self.sqlite_handler.use_sqlite_vec:
                        del unit_dict_for_upsert['embedding']

                    mu_upsert_dicts.append(unit_dict_for_upsert)

            if mu_upsert_dicts:
                # Assuming sqlite_handler.upsert_memory_units correctly handles the 'embedding' field
                # for sqlite-vec VSS table population, and ignores it otherwise for the main table.
                await self.sqlite_handler.upsert_memory_units(mu_upsert_dicts, commit=False)

            # 3. Session and LTM Upserts
            sessions_to_upsert_sqlite_dicts = []
            for sm_id in self._updated_session_memory_ids:
                if sm_id in self.session_memories_cache:
                    session = self.session_memories_cache[sm_id]
                    session_dict = session.to_dict()
                    sessions_to_upsert_sqlite_dicts.append(session_dict)
            if sessions_to_upsert_sqlite_dicts:
                await self.sqlite_handler.upsert_session_memories(sessions_to_upsert_sqlite_dicts, commit=False)

            ltms_to_upsert_sqlite_dicts = []
            for ltm_id_ in self._updated_ltm_ids:
                if ltm_id_ in self._ltm_cache:
                    ltm_dict = self._ltm_cache[ltm_id_].to_dict()
                    ltms_to_upsert_sqlite_dicts.append(ltm_dict)
                elif self.long_term_memory and self.long_term_memory.id == ltm_id_ :
                     ltm_dict = self.long_term_memory.to_dict()
                     ltms_to_upsert_sqlite_dicts.append(ltm_dict)
            if ltms_to_upsert_sqlite_dicts:
                await self.sqlite_handler.upsert_long_term_memories(ltms_to_upsert_sqlite_dicts, commit=False)

            # 4. Edge Updates
            for child_id, parent_id in self._mu_parent_updated_links.items():
                await self.sqlite_handler.update_mu_hierarchy_edge(child_id, parent_id, overwrite=True)
            for parent_id, children_ids in self._mu_children_updated_links.items():
                await self.sqlite_handler.update_mu_children_edges(parent_id, children_ids,overwrite=True)
            for unit_id, (pre_id, next_id) in self._mu_sequence_updated_links.items():
                await self.sqlite_handler.update_mu_sequence_edge(unit_id, pre_id, next_id,overwrite=True)
            for unit_id, (group_id_val, rank) in self._mu_group_updated_links.items():
                await self.sqlite_handler.update_mu_group_membership_edges(unit_id, group_id_val, rank,
                                                                                     overwrite=True)

            await self.sqlite_handler.commit()
            sqlite_success = True
            print("SQLite operations successful within transaction.")

        except Exception as e:
            print(f"Error during SQLite operations with new handler: {e}.")
            await self.sqlite_handler.rollback()
            sqlite_success = False

        # --- Vector Store Operations (External Only) ---
        vector_store_success = True  # Default true if no external VS or sqlite-vec
        if sqlite_success and self.vector_store and self.vector_store.vector_db_type != 'sqlite-vec':
            vector_store_success = False
            # try:
            if self._deleted_memory_unit_ids:
                await self.vector_store.delete(ids=list(self._deleted_memory_unit_ids))
            ext_upsert_dicts = []
            for unit_id in self._mu_new_ids.union(self._mu_content_updated_ids):
                if unit_id in self.memory_units_cache and unit_id in embedding_cache_for_flush:
                    unit, emb_np = self.memory_units_cache[unit_id], embedding_cache_for_flush[unit_id]
                    udv = unit.to_dict()
                    udv['embedding'] = emb_np.flatten().tolist()
                    udv["creation_time"] = unit.creation_time.timestamp() if unit.creation_time else None
                    udv["end_time"] = unit.end_time.timestamp() if unit.end_time else None
                    ext_upsert_dicts.append(udv)
                elif unit_id in self.memory_units_cache: print(f"Warn: Embedding missing for {unit_id} for vector store upsert.")
            if ext_upsert_dicts:
                await self.vector_store.upsert(ext_upsert_dicts)
            await self.vector_store.flush()
            print("External vector store operations successful.")
            vector_store_success = True
            # except Exception as e:
            #     print(f"Error during external vector store update: {e}")
            #     vector_store_success = False

        # --- Post-Persistence Cleanup ---
        if sqlite_success and vector_store_success:
            self._mu_new_ids.clear()
            self._mu_content_updated_ids.clear()
            self._mu_core_data_updated_ids.clear()
            self._mu_parent_updated_links.clear()
            self._mu_children_updated_links.clear()
            self._mu_sequence_updated_links.clear()
            self._mu_group_updated_links.clear()
            self._updated_session_memory_ids.clear()
            self._updated_ltm_ids.clear()
            self._deleted_memory_unit_ids.clear()
            self._deleted_session_memory_ids.clear()
            self._deleted_ltm_ids.clear()
            self._staged_updates_count = 0
            # print("All staged update sets cleared.")
        else:
            print("Persistence failed for SQLite or Vector Store. Update sets not cleared. Staged count remains.")

        if sqlite_success and vector_store_success and self.vector_store:
            await self._check_and_forget_memory()
        # print("Cache flush complete.")

    async def _get_unit_group_id(self, unit_id: str) -> Optional[Union[str, List[str]]]:
        """Fetches the group ID for a unit using the SQLite handler."""
        if not self.sqlite_handler: return None
        return await self.sqlite_handler.get_group_id(unit_id)

    async def _stage_memory_unit_update(self,
                                        memory_unit: Optional[MemoryUnit],
                                        unit_id: Optional[str] = None,
                                        operation: str = 'add', # 'add', 'content_update', 'core_data_update', 'edge_update'
                                        update_type: Optional[str] = None, # e.g. 'parent', 'sequence', 'group'
                                        update_details: Optional[Any] = None, # e.g. new_parent_id, (pre,next), (group,rank)
                                        update_session_metadata: bool = True):
        """
        Stages a MemoryUnit change.
        `operation` indicates the primary reason for staging.
        `update_type` and `update_details` are for more granular edge/specific updates.
        """
        await self.ensure_initialized()
        current_unit_id: Optional[str] = memory_unit.id if memory_unit else unit_id
        if not current_unit_id:
            print("Warning: _stage_memory_unit_update called without unit identifier.")
            return

        full_unit_available = memory_unit is not None
        rank_of_unit_for_group_update: Optional[int] = None

        if operation == "add" and full_unit_available:
            self._mu_new_ids.add(current_unit_id)
            self._mu_content_updated_ids.add(current_unit_id)
            self._mu_core_data_updated_ids.add(current_unit_id)

            self._mu_group_updated_links[current_unit_id] = (memory_unit.group_id, memory_unit.rank)
            self._mu_parent_updated_links[current_unit_id] = memory_unit.parent_id
            print("parent id:", memory_unit.parent_id)
            self._mu_sequence_updated_links[current_unit_id] = (memory_unit.pre_id,memory_unit.next_id)
        elif operation == "content_update" and full_unit_available:
            self._mu_content_updated_ids.add(current_unit_id)
            self._mu_core_data_updated_ids.add(current_unit_id)
        elif operation == "core_data_update" and full_unit_available:
            self._mu_core_data_updated_ids.add(current_unit_id)
        elif operation == "edge_update" and update_type and current_unit_id:
            if update_type == "parent":  # update_details = new_parent_id
                self._mu_parent_updated_links[current_unit_id] = update_details
            elif update_type == "children":  # update_details = (parent_id_for_this_op, new_children_list)
                parent_id_val, new_children_list_val = update_details
                self._mu_children_updated_links[parent_id_val] = new_children_list_val
            elif update_type == "sequence":  # update_details = (pre_id, next_id)
                self._mu_sequence_updated_links[current_unit_id] = update_details
            elif update_type == "group":  # update_details = (group_id_val, rank_val)
                new_group_val, rank_val = update_details
                old_group_id = await self._get_unit_group_id(current_unit_id)
                await self._handle_group_change_side_effects(current_unit_id, rank_val, old_group_id, new_group_val)
                self._mu_group_updated_links[current_unit_id] = (new_group_val, rank_val)
        elif full_unit_available and memory_unit:
            self._mu_content_updated_ids.add(current_unit_id)
            self._mu_core_data_updated_ids.add(current_unit_id)
            self._mu_sequence_updated_links[current_unit_id] = (memory_unit.pre_id, memory_unit.next_id)
            self._mu_parent_updated_links[current_unit_id] = memory_unit.parent_id
            self._mu_group_updated_links[current_unit_id] = (memory_unit.group_id, memory_unit.rank)
        elif not full_unit_available and operation != "edge_update":
            print(
                f"Warning: Full MemoryUnit object not provided for non-edge update '{operation}' on unit {current_unit_id}. Staging as core data update attempt.")
            self._mu_core_data_updated_ids.add(current_unit_id)

        if full_unit_available and memory_unit:
            self.memory_units_cache[current_unit_id] = memory_unit

        self._deleted_memory_unit_ids.discard(current_unit_id)

        # Session metadata update, needs rank and group_id
        unit_for_session_logic = memory_unit
        if not unit_for_session_logic and current_unit_id in self.memory_units_cache:
            unit_for_session_logic = self.memory_units_cache[current_unit_id]

        # Determine rank for session logic: from passed unit, from group update details, or from cached unit
        effective_rank_for_session_logic = None
        if unit_for_session_logic:
            effective_rank_for_session_logic = unit_for_session_logic.rank
        elif rank_of_unit_for_group_update is not None:
            effective_rank_for_session_logic = rank_of_unit_for_group_update

        if effective_rank_for_session_logic == 0 and update_session_metadata:
            session_id_for_meta = None
            group_id_for_meta = None
            if unit_for_session_logic:
                group_id_for_meta = unit_for_session_logic.group_id
            elif operation == "edge_update" and update_type == "group" and update_details:
                group_id_for_meta = update_details[0]

            if isinstance(group_id_for_meta, str):
                session_id_for_meta = group_id_for_meta
            elif isinstance(group_id_for_meta, list) and group_id_for_meta:
                session_id_for_meta = self._current_session_id if self._current_session_id in group_id_for_meta else \
                group_id_for_meta[0]
            # elif not group_id_for_meta and self._current_session_id:
            #     session_id_for_meta = self._current_session_id
            #     if unit_for_session_logic and unit_for_session_logic.group_id != self._current_session_id:
            #         unit_for_session_logic.group_id = self._current_session_id
            #         if current_unit_id not in self._mu_group_updated_links:
            #             self._mu_group_updated_links[current_unit_id] = (
            #             unit_for_session_logic.group_id, unit_for_session_logic.rank)

            if session_id_for_meta:
                unit_to_pass_to_session = unit_for_session_logic
                if not unit_to_pass_to_session:
                    unit_to_pass_to_session = await self._get_memory_unit(current_unit_id, use_cache=True)

                if unit_to_pass_to_session:
                    current_session_obj = await self._get_session_memory(session_id_for_meta)
                    if current_session_obj:
                        if current_unit_id not in current_session_obj.memory_unit_ids:
                            current_session_obj.memory_unit_ids.append(current_unit_id)
                        current_session_obj.update_timestamps([unit_to_pass_to_session])
                        # self.session_memories_cache[session_id_for_meta] = current_session_obj
                        await self._stage_session_memory_update(current_session_obj)
                else:
                    print(
                        f"Warning: Could not obtain unit {current_unit_id} to update session {session_id_for_meta} metadata.")
            elif effective_rank_for_session_logic == 0:  # Rank 0 unit but no session ID determinable
                print(f"Warning: Rank 0 unit {current_unit_id} has no determinable session ID for metadata update.")
        self._staged_updates_count += 1
        if self._staged_updates_count >= self.saving_interval:
            await self._flush_cache(force=True)

    async def _handle_group_change_side_effects(self, unit_id: str, rank: int, old_group: Optional[Union[str, List[str]]], new_group: Optional[Union[str, List[str]]]):
        """Updates Session/LTM membership lists when a unit's group changes."""
        old_ids = set(old_group) if isinstance(old_group, list) else ({old_group} if old_group else set())
        new_ids = set(new_group) if isinstance(new_group, list) else ({new_group} if new_group else set())

        removed_from = old_ids - new_ids
        added_to = new_ids - old_ids

        if rank == 0:
            for session_id in removed_from:
                if session_id is None: continue
                session = await self._get_session_memory(session_id)
                if session and unit_id in session.memory_unit_ids:
                    session.memory_unit_ids.remove(unit_id)
                    await self._stage_session_memory_update(session)
            for session_id in added_to:
                if session_id is None: continue
                session = await self._get_session_memory(session_id)
                if session and unit_id not in session.memory_unit_ids:
                    session.memory_unit_ids.append(unit_id)
                    await self._stage_session_memory_update(session)
        elif rank == 1:
            for ltm_id in removed_from:
                ltm = await self._get_long_term_memory(ltm_id)
                if ltm:
                    if unit_id in ltm.session_ids: ltm.session_ids.remove(unit_id)
                    if unit_id in ltm.summary_unit_ids: ltm.summary_unit_ids.remove(unit_id)
                    await self._stage_long_term_memory_update(ltm)
            for ltm_id in added_to:
                ltm = await self._get_long_term_memory(ltm_id)
                if ltm:
                    if unit_id not in ltm.session_ids: ltm.session_ids.append(unit_id)
                    await self._stage_long_term_memory_update(ltm)
        elif rank >= 2:
            for ltm_id in removed_from:
                ltm = await self._get_long_term_memory(ltm_id)
                if ltm and unit_id in ltm.summary_unit_ids:
                    ltm.summary_unit_ids.remove(unit_id)
                    await self._stage_long_term_memory_update(ltm)
            for ltm_id in added_to:
                ltm = await self._get_long_term_memory(ltm_id)
                if ltm and unit_id not in ltm.summary_unit_ids:
                    ltm.summary_unit_ids.append(unit_id)
                    await self._stage_long_term_memory_update(ltm)

    async def _stage_memory_unit_deletion(self, unit_id: str):
        await self.ensure_initialized()
        print(f"Staging deletion for unit: {unit_id}")
        self._deleted_memory_unit_ids.add(unit_id)
        self._mu_new_ids.discard(unit_id)
        self._mu_content_updated_ids.discard(unit_id)
        self._mu_core_data_updated_ids.discard(unit_id)


        self._mu_parent_updated_links.pop(unit_id, None)
        keys_to_pop_children = [k for k, v_list in self._mu_children_updated_links.items() if unit_id in v_list]
        for k_child in keys_to_pop_children:
            self._mu_children_updated_links.pop(k_child,None)
        self._mu_children_updated_links.pop(unit_id, None)
        self._mu_sequence_updated_links.pop(unit_id, None)
        self._mu_group_updated_links.pop(unit_id, None)
        self.memory_units_cache.pop(unit_id, None)
        self._visited_unit_counts.pop(unit_id, None)
        self._unit_last_visit_round.pop(unit_id, None)
        if unit_id in self._stm_id_set:
            await self._remove_unit_from_stm(unit_id)
        self._staged_updates_count += 1
        if self._staged_updates_count >= self.saving_interval:
            await self._flush_cache(force=True)

    # async def _stage_memory_units_deletion(self, unit_ids: List[str]):
    #     await self.ensure_initialized()
    #     print(f"Staging deletion for unit.")
    #     self._deleted_memory_unit_ids.update(unit_ids)
    #     unit_ids_set = set(unit_ids)
    #     self._mu_new_ids = self._mu_new_ids - unit_ids_set
    #     self._mu_content_updated_ids = self._mu_content_updated_ids - unit_ids_set
    #     self._mu_core_data_updated_ids = self._mu_core_data_updated_ids - unit_ids_set
    #     keys_to_pop_children = []
    #     for unit_id in unit_ids_set:
    #         self._mu_parent_updated_links.pop(unit_id, None)
    #     for k, v_list in self._mu_children_updated_links.items():
    #         if set(v_child)
    #
    #     for k_child in keys_to_pop_children: self._mu_children_updated_links.pop(k_child,None)
    #     self._mu_children_updated_links.pop(unit_id, None)
    #     self._mu_sequence_updated_links.pop(unit_id, None)
    #     self._mu_group_updated_links.pop(unit_id, None)
    #     self.memory_units_cache.pop(unit_id, None)
    #     self._visited_unit_counts.pop(unit_id, None)
    #     self._unit_last_visit_round.pop(unit_id, None)
    #     if unit_id in self._stm_id_set:
    #         await self._remove_unit_from_stm(unit_id)
    #     self._staged_updates_count += 1
    #     if self._staged_updates_count >= self.saving_interval:
    #         await self._flush_cache(force=True)

    async def _stage_session_memory_update(self, session: SessionMemory):
        """Stages a SessionMemory change in cache and tracking sets."""
        await self.ensure_initialized()
        if not session or not session.id: return
        if session.id in self._deleted_session_memory_ids:
            return
        self.session_memories_cache[session.id] = session
        self._updated_session_memory_ids.add(session.id)
        self._deleted_session_memory_ids.discard(session.id)
        self._staged_updates_count += 1
        if self._staged_updates_count >= self.saving_interval:
            await self._flush_cache(force=True)

    async def _stage_session_memory_deletion(self, session_id: str):
        """Stages a SessionMemory for deletion."""
        await self.ensure_initialized()
        self._deleted_session_memory_ids.add(session_id)
        self._updated_session_memory_ids.discard(session_id)
        self.session_memories_cache.pop(session_id, None)
        self._staged_updates_count += 1

    async def _stage_ltm_deletion(self, ltm_id: str):
        """Stages a LongTermMemory for deletion."""
        await self.ensure_initialized()
        self._deleted_ltm_ids.add(ltm_id)
        self._updated_ltm_ids.discard(ltm_id)
        self._ltm_cache.pop(ltm_id, None)
        if self.long_term_memory and self.long_term_memory.id == ltm_id:
            self.long_term_memory = None
        self._staged_updates_count += 1

    async def _stage_long_term_memory_update(self, ltm: LongTermMemory):
        """Stages the primary LongTermMemory change."""
        await self.ensure_initialized()
        if not ltm or ltm.id != self.ltm_id: return  # Only stage the primary LTM

        self.long_term_memory = ltm  # Update cache (the single LTM object)
        self._updated_ltm_ids.add(ltm.id)
        self._ltm_cache[ltm.id] = ltm

        # LTM updates also count towards flush interval
        self._staged_updates_count += 1
        if self._staged_updates_count >= self.saving_interval:
            # print(f"Saving interval reached ({self._staged_updates_count}) after staging LTM update. Triggering flush.")
            await self._flush_cache(force=True)

    async def _stage_memory_units_update(self, units: Iterable[MemoryUnit], operation: str = 'add'):
        """Helper to stage multiple units, useful after summarization."""
        for unit in units:
            await self._stage_memory_unit_update(unit, operation=operation, update_session_metadata=False)
        # print(f"Staged {count} units for operation: {operation}")

    async def _get_memory_unit(self, unit_id: str, use_cache: bool = True) -> Optional[MemoryUnit]:
        """Retrieves a MemoryUnit async, checking cache first, then SQLite."""
        await self.ensure_initialized()
        if use_cache and unit_id in self.memory_units_cache:
            return self.memory_units_cache[unit_id]

        if not self.sqlite_handler: return None

        # print(f"Cache miss for unit {unit_id}. Loading from SQLite...")
        unit = await self.sqlite_handler.load_memory_unit(unit_id, include_edges=True)

        if unit and use_cache:
            self.memory_units_cache[unit_id] = unit
        return unit

    async def _get_session_memory(self, session_id: str, use_cache: bool = True) -> Optional[SessionMemory]:
        """Retrieves a SessionMemory async, checking cache first, then SQLite."""
        await self.ensure_initialized()
        if use_cache and session_id in self.session_memories_cache:
            return self.session_memories_cache[session_id]

        if not self.sqlite_handler: return None

        # print(f"Cache miss for session {session_id}. Loading from SQLite...")
        session = await self.sqlite_handler.load_session_memory(session_id)

        if session and use_cache:
            self.session_memories_cache[session_id] = session

        return session

    async def _get_long_term_memory(self, ltm_id: str) -> Optional[LongTermMemory]:
        """Retrieves a specific LTM object (primarily for external use). Async version."""
        await self.ensure_initialized()
        if ltm_id == self.ltm_id:
            return self.long_term_memory

        if self.sqlite_handler:
            return await self.sqlite_handler.load_long_term_memory(ltm_id, self.chatbot_id)

        return None

    async def _load_memory_units(self, unit_ids: List[str], use_cache: bool = True) -> Dict[str, MemoryUnit]:
        """Loads multiple MemoryUnits async, checking cache first."""
        await self.ensure_initialized()
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

        if not self.sqlite_handler: return loaded_units

        # print(f"Loading {len(ids_to_load_from_store)} units from SQLite...")
        store_units = await self.sqlite_handler.load_memory_units(ids_to_load_from_store)

        if use_cache:
            self.memory_units_cache.update(store_units)
        loaded_units.update(store_units)
        # print(f"Finished loading units. Total loaded/cached: {len(loaded_units)}")
        return loaded_units

    async def _load_sessions(self, session_ids: List[str], use_cache: bool = True) -> Dict[str, SessionMemory]:
        """Loads multiple SessionMemories async, checking cache first."""
        await self.ensure_initialized()
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

        if not self.sqlite_handler: return loaded_sessions

        store_sessions = await self.sqlite_handler.load_session_memories(ids_to_load_from_store)

        if use_cache:
            self.session_memories_cache.update(store_sessions)
        loaded_sessions.update(store_sessions)
        return loaded_sessions

    async def _load_units_for_session(self, session_id: str) -> Dict[str, MemoryUnit]:
        """Helper to load all units listed in a session object (async)."""
        session = await self._get_session_memory(session_id)
        if session and session.memory_unit_ids:
            return await self._load_memory_units(session.memory_unit_ids)
        return {}

    async def _get_current_session(self) -> Optional[SessionMemory]:
        """Gets the current session object async."""
        await self.ensure_initialized()
        if not self._current_session_id:
            return None
        return await self._get_session_memory(self._current_session_id)

    async def remove_session(self, session_id: str):
        """
        Removes a session, its content units, its summary unit, and recursively
        """
        await self.ensure_initialized()
        print(f"Processing removal of session: {session_id}")
        is_current_session_being_removed = (session_id == self._current_session_id)
        if is_current_session_being_removed:
            self.context.clear()
            self._last_context_unit = None
        if not self.sqlite_handler or not self.long_term_memory:
            print("Error: System not properly initialized for session removal.")
            return
        await self._flush_cache(force=True)
        # 1. Mark SessionMemory object for deletion from its table
        session_memory_obj = await self._get_session_memory(session_id, use_cache=True)
        await self._stage_session_memory_deletion(session_id)

        # 2. Update LTM's list of session_ids and last_session_id
        if session_id in self.long_term_memory.session_ids:
            self.long_term_memory.session_ids.remove(session_id)
            if self.long_term_memory.last_session_id == session_id:
                self.long_term_memory.last_session_id = self.long_term_memory.session_ids[
                    -1] if self.long_term_memory.session_ids else None
        if session_id in self.long_term_memory.summary_unit_ids:
            self.long_term_memory.summary_unit_ids.remove(session_id)

        # if is_current_session_being_removed:
        #     # print(f"Current session {session_id} is being removed. Updating context units' group IDs.") # User requested less printing
        #     for unit_in_context in self.context:  # Iterate directly, modifies objects in deque
        #         update_group_for_this_unit = False
        #         if isinstance(unit_in_context.group_id, str) and unit_in_context.group_id == session_id:
        #             unit_in_context.group_id = None
        #             update_group_for_this_unit = True
        #         elif isinstance(unit_in_context.group_id, list):
        #             original_list_len = len(unit_in_context.group_id)
        #             unit_in_context.group_id = [gid for gid in unit_in_context.group_id if gid != session_id]
        #             if len(unit_in_context.group_id) != original_list_len:
        #                 update_group_for_this_unit = True
        #             if not unit_in_context.group_id:  # List became empty
        #                 unit_in_context.group_id = None
        #             elif len(unit_in_context.group_id) == 1:  # Simplify to string
        #                 unit_in_context.group_id = unit_in_context.group_id[0]

                # if update_group_for_this_unit: # User requested less printing
                # print(f"Unit {unit_in_context.id} in context had group_id related to removed session {session_id}. New group_id: {unit_in_context.group_id}")

            # if self._last_context_unit:  # Also update _last_context_unit if it was tied to this session
            #     if isinstance(self._last_context_unit.group_id, str) and self._last_context_unit.group_id == session_id:
            #         self._last_context_unit.group_id = None
            #     elif isinstance(self._last_context_unit.group_id, list):
            #         self._last_context_unit.group_id = [gid for gid in self._last_context_unit.group_id if
            #                                             gid != session_id]
            #         if not self._last_context_unit.group_id:
            #             self._last_context_unit.group_id = None
            #         elif len(self._last_context_unit.group_id) == 1:
            #             self._last_context_unit.group_id = self._last_context_unit.group_id[0]
            #
            # self._current_session_id = None  # Clear current session ID

        # 3. Collect all units to be deleted
        unit_ids_to_delete_fully: set[str] = set()
        if session_memory_obj and session_memory_obj.memory_unit_ids:
            unit_ids_to_delete_fully.update(session_memory_obj.memory_unit_ids)

        current_summary_id_in_chain = session_id
        while current_summary_id_in_chain:
            # print(f"  Considering summary unit {current_summary_id_in_chain} for deletion chain.")
            unit_ids_to_delete_fully.add(current_summary_id_in_chain)
            summary_unit_obj = await self._get_memory_unit(current_summary_id_in_chain, use_cache=True)
            if not summary_unit_obj:
                print(f"    Warning: Summary unit {current_summary_id_in_chain} not found during deletion.")
                break

            if current_summary_id_in_chain in self.long_term_memory.summary_unit_ids:
                self.long_term_memory.summary_unit_ids.remove(current_summary_id_in_chain)

            parent_id_of_current_summary = summary_unit_obj.parent_id
            if parent_id_of_current_summary:
                parent_summary_unit = await self._get_memory_unit(parent_id_of_current_summary, use_cache=True)
                if parent_summary_unit and parent_summary_unit.rank >= 1:
                    # print(
                    #     f"    Summary unit {current_summary_id_in_chain} has parent summary {parent_id_of_current_summary}. As per rule, parent will also be deleted.")
                    current_summary_id_in_chain = parent_id_of_current_summary
                else:
                    current_summary_id_in_chain = None
            else:
                current_summary_id_in_chain = None
        await self._stage_long_term_memory_update(self.long_term_memory)
        # 4. Stage all collected MemoryUnits for deletion
        if unit_ids_to_delete_fully:
            print(f"Staging deletion for {len(unit_ids_to_delete_fully)} units in total for session {session_id}.")
            for unit_id_to_remove in unit_ids_to_delete_fully:
                await self._stage_memory_unit_deletion(unit_id_to_remove)

        # 5. Flush all changes
        await self._flush_cache(force=True)
        print(f"Session {session_id} removal process (with corrected recursive summary deletion) complete.")

    async def delete_ltm(self, ltm_id: Optional[str] = None):
        target_ltm_id = ltm_id if ltm_id else self.ltm_id
        if not target_ltm_id:
            print("Error: No LTM ID specified or initialized for deletion.")
            return

        collection_name = f"chatbot_{self.chatbot_id.replace('-', '_')}_ltm_{target_ltm_id.replace('-', '_')}"
        if self.vector_store:
            try:
                await self.vector_store.delete_collection(collection_name)
                print(f"Deleted vector store collection: {collection_name}")
            except Exception as e:
                print(f"Warning: Failed to delete vector store collection {collection_name}: {e}")

        await self._stage_ltm_deletion(target_ltm_id)
        await self._flush_cache(force=True)
        print(f"LTM {target_ltm_id} SQLite record staged for deletion and flushed.")

    async def _get_formatted_content_with_history(self, memory_unit: MemoryUnit, history_length: int = 1) -> str:
        """Async version: Retrieves content with history, loading units asynchronously."""
        await self.ensure_initialized()
        history_units: List[MemoryUnit] = []
        current_unit_id = memory_unit.pre_id

        while current_unit_id and len(history_units) < history_length:
            # Use async getter
            unit = await self._get_memory_unit(current_unit_id, use_cache=True)
            if unit:
                history_units.append(unit)
                current_unit_id = unit.pre_id
            else:
                # print(f"Warning: Could not find predecessor unit {current_unit_id}.")
                break

        # Combine content (logic remains the same)
        contents: List[str] = []
        all_units = history_units[::-1] + [memory_unit]

        for unit in all_units:
            action = unit.metadata.get('action') if unit.metadata else None
            if action and action in ["summary", "mind"]:
                contents.append(f"{unit.source}想: ({unit.content})")
            elif action == "speak":
                contents.append(f"{unit.source}说: {unit.content}")
            else:
                contents.append(unit.content)  # Default case
        return "\n".join(contents)

    async def enable_external_connection(self,
                                         external_chatbot_id: str,
                                         external_ltm_id: str,
                                         external_vector_store_config: Dict[str, Any],
                                         external_sqlite_base_path: Optional[str] = None
                                         ):
        """Async: Enables connection to an external chatbot's memory. Must provide the external sqlite db path"""
        print(f"Attempting to enable external connection to {external_chatbot_id}/{external_ltm_id}...")
        effective_external_sqlite_base_path = external_sqlite_base_path if external_sqlite_base_path is not None else self.storage_base_path
        ext_vector_store, ext_sqlite_handler = None, None
        try:
            if "embedding_dim" not in external_vector_store_config:
                external_vector_store_config["embedding_dim"] = self.embedding_dim
            ext_vector_store = VectorStoreHandler(chatbot_id=external_chatbot_id, long_term_memory_id=external_ltm_id, config=external_vector_store_config)
            await ext_vector_store.initialize()
            if not await ext_vector_store.has_collection():
                await ext_vector_store.close()
                raise ValueError(f"External vector store resource for {external_chatbot_id}/{external_ltm_id} not found.")
            print(f"External VectorStoreHandler enabled for {external_chatbot_id}/{external_ltm_id} ({ext_vector_store.vector_db_type}).")
            ext_sqlite_handler = AsyncSQLiteHandler()
            ext_sqlite_handler.initialize(chatbot_id=external_chatbot_id, base_path=effective_external_sqlite_base_path, embedding_dim=self.embedding_dim, use_sqlite_vec=False)
            await ext_sqlite_handler.connect()
            await ext_sqlite_handler.close()
            print(f"External AsyncSQLiteHandler configured for {external_chatbot_id} at {effective_external_sqlite_base_path}.")
            self.external_chatbot_id, self.external_ltm_id, self.external_vector_store, self.external_sqlite_handler = external_chatbot_id, external_ltm_id, ext_vector_store, ext_sqlite_handler
            print(f"External connection fully enabled to {external_chatbot_id}/{external_ltm_id}.")
        except Exception as e:
            print(f"Error enabling external connection: {e}")
            if ext_vector_store: await ext_vector_store.close()
            if ext_sqlite_handler: await ext_sqlite_handler.close()
            self.external_chatbot_id, self.external_ltm_id, self.external_vector_store, self.external_sqlite_handler = None, None, None, None
            raise ValueError(f"Failed to enable external connection. Error: {e}") from e

    async def _restore_session(self, session_id: str):
        """Async: Loads MemoryUnits from a session into STM, fetching embeddings."""
        await self.ensure_initialized()
        if not self._stm_enabled: return
        if not self.vector_store:
            print("Warning: Cannot restore STM from session without VectorStoreHandler.")
            return

        session_memory = await self._get_session_memory(session_id)
        if not session_memory or not session_memory.memory_unit_ids:
            print(f"Warning: Session {session_id} not found or empty for STM restoration.")
            return

        memory_unit_ids = session_memory.memory_unit_ids
        units_to_add_to_stm: List[Tuple[MemoryUnit, np.ndarray]] = []
        found_ids = set()

        # 1. Load units from SQLite first (source of truth for metadata)
        units_dict = await self._load_memory_units(memory_unit_ids)

        # 2. Fetch corresponding embeddings from vector store (batch if possible)
        embeddings_map = {}
        try:
            for unit_id in memory_unit_ids:
                vec_data = await self.vector_store.get(unit_id, output_fields=['embedding'])
                if vec_data and vec_data.get('embedding'):
                    embeddings_map[unit_id] = np.array(vec_data['embedding'])

        except Exception as e:
            print(f"Error querying vector store for session {session_id} embeddings: {e}")
            # Continue without embeddings for units where fetch failed? Or stop?

        # 3. Combine units and embeddings
        for unit_id in memory_unit_ids:  # Iterate in session order
            unit = units_dict.get(unit_id)
            embedding = embeddings_map.get(unit_id)

            if unit and embedding is not None:
                units_to_add_to_stm.append((unit, embedding))
                found_ids.add(unit_id)
            elif unit:
                print(f"Warning: Embedding not found in vector store for unit {unit_id} in session {session_id}.")
            else:
                print(f"Warning: Memory unit {unit_id} from session {session_id} not found in SQLite.")

        units_to_add_to_stm = sorted(
            units_to_add_to_stm,
            key=lambda x: x[0].creation_time if x[0].creation_time else datetime.datetime.min
        )
        for unit, _ in units_to_add_to_stm:
            self.context.append(unit)
            if len(self.context) > self._max_context_length:
                evicted_unit = self.context.popleft()
                if evicted_unit.id in self._context_ids:
                    self._context_ids.remove(evicted_unit.id)

        print(f"Adding {len(units_to_add_to_stm)} units from session {session_id} to STM.")
        await self._add_units_to_stm(units_to_add_to_stm)  # Use the async version

    async def enable_stm(self, capacity: int = 200, restore_session_id: Optional[str] = None):
        """Async: Enables STM, optionally restores from session."""
        await self.ensure_initialized()
        if self._stm_enabled:
            if self._stm_capacity != capacity:
                print(f"Updating STM capacity from {self._stm_capacity} to {capacity}.")
                self._stm_capacity = capacity
                while len(self._short_term_memory_ids) > self._stm_capacity:
                    await self._remove_oldest_from_stm()  # Use async version
            return

        print(f"Enabling STM with capacity {capacity}.")
        self._stm_enabled = True
        self._stm_capacity = capacity
        self._short_term_memory_ids = deque()
        self._short_term_memory_embeddings.clear()
        self._short_term_memory_units.clear()
        self._stm_id_set.clear()

        target_session_id = None
        if restore_session_id == "LATEST":
            target_session_id = self.long_term_memory.last_session_id if self.long_term_memory else None
        elif restore_session_id:
            target_session_id = restore_session_id

        if target_session_id:
            print(f"Attempting to restore STM from session: {target_session_id}")
            await self._restore_session(target_session_id)
        else:
            print("STM enabled without restoring from a session.")

    async def disable_stm(self):
        """Async: Disables and clears STM."""
        if not self._stm_enabled: return
        print("Disabling and clearing STM.")
        self._short_term_memory_ids.clear()
        self._short_term_memory_embeddings.clear()
        self._short_term_memory_units.clear()
        self._stm_id_set.clear()
        self._stm_enabled = False
        self._stm_capacity = 0

    async def _check_and_forget_memory(self):
        """Async: Checks vector store count and triggers forgetting."""
        await self.ensure_initialized()
        if not self.vector_store or self.max_vector_entities <= 0:
            return
        try:
            current_count = await self.vector_store.count_entities(consistently=True)
            if current_count <= self.max_vector_entities:
                return

            print(f"Vector Store count {current_count} > {self.max_vector_entities}. Triggering forgetting...")

            deleted_unit_ids, updated_parent_ids, updated_session_ids = await forget_memories(
                memory_system=self,
                ltm_id=self.ltm_id,
                delete_percentage=self.forget_percentage,
            )

            if not deleted_unit_ids:
                print("Forgetting identified no units to delete.")
                return

            print(f"Forgetting identified {len(deleted_unit_ids)} units. Staging changes...")
            for unit_id in deleted_unit_ids:
                await self._stage_memory_unit_deletion(unit_id)

            deleted_unit_ids_set = set(deleted_unit_ids)
            # Stage updates for parents (children lists)
            parents_to_update = await self._load_memory_units(list(updated_parent_ids))
            for parent_id, parent_unit in parents_to_update.items():
                parent_unit.children_ids = [id for id in parent_unit.children_ids if id not in deleted_unit_ids_set]
                await self._stage_memory_unit_update(parent_unit, operation='edge_update', update_session_metadata=False)

            # Stage updates for sessions (unit lists)
            sessions_to_update = await self._load_sessions(list(updated_session_ids))
            for session_id, session_unit in sessions_to_update.items():
                session_unit.memory_unit_ids = [id for id in session_unit.memory_unit_ids if
                                                id not in deleted_unit_ids_set]
                await self._stage_session_memory_update(session_unit)
            await self._flush_cache(force=True)
            print("Forgetting cleanup complete.")

        except Exception as e:
            print(f"Error during auto-forgetting check or execution: {e}")

    async def _add_to_context(self, memory_unit: MemoryUnit, embedding_history_length: int = 1):
        """
        Async: Adds to context deque, handles linking, eviction, staging.
        In my design, context have no intersection with memories. This may lead to loss of context.
        """
        await self.ensure_initialized()
        if not memory_unit or not memory_unit.id: return
        if not self._current_session_id:
            print("Error: Cannot add to context, no active session.")
            return

        unit_id = memory_unit.id
        # Link to previous context unit
        if self._last_context_unit is not None and self._last_context_unit.id != unit_id:
            last_context_unit = self._last_context_unit
            if last_context_unit.next_id != unit_id:
                last_context_unit.next_id = unit_id
                # Stage update for the previous unit's next_id if it was already persisted
                if self._last_context_unit.id not in self._context_ids:
                    await self._stage_memory_unit_update(memory_unit=None, unit_id=last_context_unit.id,
                                                         operation="edge_update", update_type="sequence",
                                                         update_details=(last_context_unit.pre_id, unit_id),
                                                         update_session_metadata=False)
            if memory_unit.pre_id != last_context_unit.id:
                memory_unit.pre_id = last_context_unit.id

        await self._stage_memory_unit_update(memory_unit=memory_unit, operation="add", update_session_metadata=False)

        # Add to context deque and cache
        self.context.append(memory_unit)
        self.memory_units_cache[unit_id] = memory_unit
        self._last_context_unit = memory_unit
        self._context_ids.add(memory_unit.id)

        current_session = await self._get_current_session()
        if current_session and unit_id not in current_session.memory_unit_ids:
            current_session.memory_unit_ids.append(unit_id)
            current_session.update_timestamps([memory_unit])
            await self._stage_session_memory_update(current_session)

        if len(self.context) > self._max_context_length:
            evicted_unit = self.context.popleft()
            if evicted_unit.id in self._context_ids:
                self._context_ids.remove(evicted_unit.id)
            print(f"Context full. Evicting unit {evicted_unit.id}")
            evicted_embedding_np = await self._generate_embedding_for_unit(evicted_unit, embedding_history_length)
            if evicted_embedding_np is not None:
                await self._add_units_to_stm([(evicted_unit, evicted_embedding_np)])
            # await self._stage_memory_unit_update(memory_unit=evicted_unit, operation="add")
            # if evicted_unit.id not in self._mu_new_ids and evicted_unit.id not in self._mu_content_updated_ids:
            #     await self._stage_memory_unit_update(memory_unit=evicted_unit, operation="core_data_update")

    async def add_memory(self, message: Union[str, BaseMessage], source: Optional[str] = None,
                         metadata: Optional[Dict[str, Any]] = None, creation_time: Optional[datetime.datetime] = None,
                         memory_unit_id: Optional[str] = None) -> Optional[MemoryUnit]:
        """Async: Adds a new message memory to context and stages for persistence."""
        await self.ensure_initialized()
        if not self._current_session_id: return None
        now = datetime.datetime.now()
        creation_time = creation_time or now
        metadata = metadata if metadata is not None else {}
        if 'action' not in metadata: metadata['action'] = 'speak'
        unit_id_to_use = memory_unit_id if memory_unit_id else str(uuid4())
        initial_group_id = self._current_session_id
        try:
            if isinstance(message, str):
                unit = MemoryUnit(content=message, source=source, metadata=metadata, creation_time=creation_time,
                                  end_time=creation_time, memory_id=unit_id_to_use, group_id=initial_group_id, rank=0)
            elif isinstance(message, BaseMessage):
                unit = MemoryUnit.from_langchain_message(message=message, source=source, metadata=metadata,
                                                         creation_time=creation_time, end_time=creation_time,
                                                         memory_id=unit_id_to_use, group_id=initial_group_id, rank=0)
            else:
                return None
        except Exception as e:
            print(f"Error creating MemoryUnit: {e}")
            return None
        # await self._stage_memory_unit_update(memory_unit=unit, operation="add")
        await self._add_to_context(unit)
        return unit

    async def get_context(self, length: Optional[int] = None) -> List[MemoryUnit]:
        """Async: Returns the current context messages."""
        await self.ensure_initialized()
        if length is None:
            return list(self.context)
        else:
            actual_length = min(length, len(self.context))
            return list(self.context)[-actual_length:]

    async def clear_all(self):
        """Async: Clears the current conversation context and STM."""
        # await self.ensure_initialized()
        self.context.clear()
        await self.disable_stm()  # Disables and clears STM
        print("Context and STM cleared.")

    async def flush_context(self, embedding_history_length: int = 1):
        """Async: Flush context items to STM/persistent store."""
        await self.ensure_initialized()
        if not self.context: return
        # print(f"Flushing context ({len(self.context)} items)...")

        items_to_process = list(self.context)
        self.context.clear()

        units_to_add_stm = []

        for evicted_unit in items_to_process:
            evicted_embedding_np = None
            try:
                # Ensure embedding exists before adding to STM/final persistence
                content_to_embed = await self._get_formatted_content_with_history(evicted_unit,
                                                                                  embedding_history_length)
                evicted_embedding_np = self.embedding_handler.get_embedding(content_to_embed)
            except Exception as e:
                print(f"Error embedding evicted {evicted_unit.id}: {e}")

            # Collect units with embeddings to add to STM
            if self._stm_enabled and evicted_embedding_np is not None:
                units_to_add_stm.append((evicted_unit, evicted_embedding_np))
            elif self._stm_enabled:
                print(f"Warn: Cannot add evicted {evicted_unit.id} to STM, embedding missing.")

            # Stage evicted unit again (confirming add to persistent store)
            # The embedding will be included during the _flush_cache logic when saving
            # await self._stage_memory_unit_update(evicted_unit, operation='add')

        # Add collected units to STM in one go
        if units_to_add_stm:
            await self._add_units_to_stm(units_to_add_stm)

        # Flush staged changes (including the flushed context units)
        # await self._flush_cache(force=True)
        # print("Context flushed and items processed.")

    async def _remove_oldest_from_stm(self):
        """Async: Removes the least recently used item from STM."""
        if not self._short_term_memory_ids:
            return
        try:
            removed_id = self._short_term_memory_ids.popleft()
            self._short_term_memory_embeddings.pop(removed_id, None)
            self._short_term_memory_units.pop(removed_id, None)
            self._stm_id_set.discard(removed_id)
            # print(f"Removed oldest unit {removed_id} from STM.")
        except IndexError:
            print("Warning: Tried to pop from empty STM deque.")

    async def _remove_unit_from_stm(self, unit_id: str):
        """Async: Removes a specific unit from STM structures."""
        if unit_id in self._stm_id_set:
            try:
                self._short_term_memory_ids.remove(unit_id)
            except ValueError:
                print(f"Warning: Unit {unit_id} was in _stm_id_set but not found in deque.")
                pass
            self._short_term_memory_embeddings.pop(unit_id, None)
            self._short_term_memory_units.pop(unit_id, None)
            self._stm_id_set.discard(unit_id)
            # print(f"Removed specific unit {unit_id} from STM.")

    async def _add_units_to_stm(self, memory_units_with_embeddings: List[Tuple[MemoryUnit, np.ndarray]]):
        """Async: Adds or updates memory units in the STM, managing capacity."""
        await self.ensure_initialized()
        if not self._stm_enabled: return

        for memory_unit, embedding_np in memory_units_with_embeddings:
            if not memory_unit or not memory_unit.id or embedding_np is None:
                print(
                    f"Warning: Skipping invalid unit or unit with no embedding for STM: {getattr(memory_unit, 'id', 'N/A')}")
                continue

            unit_id = memory_unit.id

            if unit_id in self._stm_id_set:
                # Already exists, move to end (most recently used)
                try:
                    self._short_term_memory_ids.remove(unit_id)
                except ValueError:
                    # print(f"Error: Unit {unit_id} in _stm_id_set but not in deque. Re-adding.")
                    # self._stm_id_set.remove(unit_id)
                    pass
                self._short_term_memory_ids.append(unit_id)  # Move/add to end
                # Update data just in case
                self._short_term_memory_units[unit_id] = memory_unit
                self._short_term_memory_embeddings[unit_id] = embedding_np


            else:
                while len(self._short_term_memory_ids) >= self._stm_capacity:
                    await self._remove_oldest_from_stm()

                # Add the new unit
                self._short_term_memory_ids.append(unit_id)
                self._short_term_memory_units[unit_id] = memory_unit
            self._short_term_memory_embeddings[unit_id] = embedding_np
            self._stm_id_set.add(unit_id)
                # print(f"Added new unit {unit_id} to STM.")

        # --- Querying Methods (Async) ---

    async def _query_stm(self,
                         query_vector: Optional[List[float]] = None,
                         filters: Optional[BaseFilter] = None,
                         k_limit: int = 5,
                         search_range: Optional[Tuple[Optional[float], Optional[float]]] = (0.75, None)
                         ) -> List[Tuple[MemoryUnit, float]]:
        """
        Async: Queries the Short-Term Memory.
        Performs vector similarity search and/or filtering directly on units in STM.
        Updates the LRU order for accessed units.
        """
        await self.ensure_initialized()
        if not self._stm_enabled:
            print("Warning: STM query attempted while disabled.")
            return []

        candidate_ids = list(self._short_term_memory_ids)  # Query based on current order
        results: List[Tuple[MemoryUnit, float]] = []
        query_np = np.array(query_vector, dtype=np.float32) if query_vector is not None else None

        for unit_id in reversed(candidate_ids):  # Check MRU first
            unit = self._short_term_memory_units.get(unit_id)
            if not unit: continue

            passes_filter = True
            if filters:
                try:
                    passes_filter = self._evaluate_filter_object(unit, filters)
                except Exception as e:
                    print(f"Warning: Error applying filter to STM unit {unit_id}: {e}")
                    passes_filter = False

            if not passes_filter: continue

            score = 0.0
            if query_np is not None:
                embedding = self._short_term_memory_embeddings.get(unit_id)
                if embedding is not None:
                    embedding_np = np.array(embedding, dtype=np.float32)
                    score = np.dot(query_np, embedding_np)

                    if search_range is not None:
                        min_threshold, max_threshold = search_range
                        if (min_threshold is not None and score < min_threshold) or \
                                (max_threshold is not None and score > max_threshold):
                            continue  # Skip if outside range
                else:
                    # print(f"Warning: Embedding missing for STM unit {unit_id}, cannot score.")
                    continue

            results.append((unit, float(score)))

        if query_np is not None:
            results.sort(key=lambda x: x[1], reverse=True)  # Higher score is better (cosine sim)

        # Limit results
        limited_results = results[:k_limit]

        results_ids = [unit.id for unit, score in results[:k_limit]]
        accessed_ids = set(results_ids)
        if accessed_ids:
            new_deque = deque(item for item in self._short_term_memory_ids if item not in accessed_ids)
            new_deque.extend(results_ids)  # Add accessed to end
            self._short_term_memory_ids = new_deque

        return limited_results

    async def _query_ltm(self,
                         target_vector_store: VectorStoreHandler,
                         target_sqlite_handler: Optional[AsyncSQLiteHandler] = None,
                         query_vector: Optional[List[float]] = None,
                         filters: Optional[BaseFilter] = None,
                         k_limit: int = 5,
                         search_range: Optional[Tuple[Optional[float],Optional[float]]] = (0.75, None)
                         ) -> List[Tuple[MemoryUnit, float, np.ndarray]]:
        """
        Async: Queries a specific vector store (internal or external LTM).
        Returns: List of tuples: (MemoryUnit, similarity_score, embedding_vector).
        """
        await self.ensure_initialized()
        results_with_embeddings: List[Tuple[MemoryUnit, float, np.ndarray]] = []
        filter_arg = None
        if filters:
            if target_vector_store.vector_db_type in ["milvus", "milvus-lite"]: filter_arg = filters.to_milvus_expr()
            elif target_vector_store.vector_db_type == "chroma": filter_arg = filters.to_chroma_filter()
            elif target_vector_store.vector_db_type == "qdrant": filter_arg = filters.to_qdrant_filter()
            elif target_vector_store.vector_db_type == "sqlite-vec": filter_arg = filters.to_sqlite_where()
        search_params = target_vector_store.config.get("search_params", {"metric_type": "IP", "params": {"nprobe": 10}})
        if search_range is not None: search_params['search_range'] = search_range
        try:
            vector_query_results = []
            vs_output_fields = ["id", "embedding"]
            if query_vector is not None:
                vector_query_results = await target_vector_store.search(vectors=[query_vector], top_k=k_limit, expr=filter_arg, search_params=search_params, output_fields=vs_output_fields)
            elif filter_arg:
                vector_query_results = await target_vector_store.query(expr=filter_arg, output_fields=vs_output_fields, top_k=k_limit)
            else: return []
            unit_ids_from_vs, scores_map, embeddings_from_vs_map = [], {}, {}
            for hit_data in vector_query_results:
                uid = hit_data.get('id')
                if not uid:
                    continue
                unit_ids_from_vs.append(uid)
                scores_map[uid] = float(hit_data.get('distance', 0.0))
                if hit_data.get('embedding') is not None:
                    embeddings_from_vs_map[uid] = np.array(hit_data['embedding'])
            if not unit_ids_from_vs:
                return []
            full_memory_units_dict: Dict[str, MemoryUnit] = {}
            current_sql_handler = target_sqlite_handler if target_sqlite_handler else self.sqlite_handler
            if current_sql_handler:
                if not await current_sql_handler._get_connection():
                    raise
                full_memory_units_dict = await current_sql_handler.load_memory_units(unit_ids_from_vs, include_edges=True)
            for unit_id in unit_ids_from_vs:
                unit = full_memory_units_dict.get(unit_id)
                if unit:
                    score = scores_map.get(unit_id, 0.0)
                    embedding_np = embeddings_from_vs_map.get(unit_id)
                    if embedding_np is None and target_vector_store.vector_db_type == 'sqlite-vec' and current_sql_handler:
                        emb_list = await self._generate_embedding_for_unit(unit,history_length=1)
                        if emb_list:
                            embedding_np = np.array(emb_list)
                    if embedding_np is not None:
                        results_with_embeddings.append((unit, score, embedding_np))
        except Exception as e: print(f"Error during LTM query: {e}"); return []
        if query_vector is not None: results_with_embeddings.sort(key=lambda x: x[1], reverse=True)
        return results_with_embeddings[:k_limit]

    async def _fetch_neighboring_units(self,
                                       unit: MemoryUnit,
                                       source_type_for_lookup: str,
                                       find_pre: bool = True) -> Tuple[Optional[MemoryUnit], Optional[MemoryUnit]]:
        """
        Async: Fetches immediate preceding and succeeding units.
        """
        await self.ensure_initialized()
        pre_unit, next_unit = None, None
        pre_id_to_find, next_id_to_find = unit.pre_id, unit.next_id
        active_sqlite_handler, cache_to_check = None, None
        if source_type_for_lookup == 'stm':
            cache_to_check = self._short_term_memory_units
        elif source_type_for_lookup == 'local_sqlite':
            active_sqlite_handler, cache_to_check = self.sqlite_handler, self.memory_units_cache
        elif source_type_for_lookup == 'external_sqlite':
            active_sqlite_handler = self.external_sqlite_handler
        else:
            return None, None
        if pre_id_to_find and find_pre:
            if cache_to_check and pre_id_to_find in cache_to_check:
                pre_unit = cache_to_check[pre_id_to_find]
            elif active_sqlite_handler:
                if not await active_sqlite_handler._get_connection():
                    raise
                pre_unit = await active_sqlite_handler.load_memory_unit(pre_id_to_find, include_edges=True)
                if pre_unit and source_type_for_lookup == 'local_sqlite' and cache_to_check:
                    cache_to_check[pre_id_to_find] = pre_unit
        if next_id_to_find and not find_pre:
            if cache_to_check and next_id_to_find in cache_to_check:
                next_unit = cache_to_check[next_id_to_find]
            elif active_sqlite_handler:
                if not await active_sqlite_handler._get_connection():
                    raise
                next_unit = await active_sqlite_handler.load_memory_unit(next_id_to_find, include_edges=True)
                if next_unit and source_type_for_lookup == 'local_sqlite' and cache_to_check:
                    cache_to_check[next_id_to_find] = next_unit
        return pre_unit, next_unit

    async def query(self,
                    query_vector: Optional[List[float]] = None,
                    filters: Optional[Dict[str, Any]] = None,
                    k_limit: int = 5,
                    search_range: Optional[Tuple[Optional[float], Optional[float]]] = (0.75, None),
                    recall_context: bool = True,
                    add_ltm_to_stm: bool = True,
                    query_external_first: bool = True,
                    short_term_only: bool = False,
                    long_term_only: bool = False,
                    external_only: bool = False,
                    threshold_for_stm_before_ltm: float = 0.8
                    ) -> List[List[MemoryUnit]]:
        """
        Async: Queries the memory system (STM, LTM, External LTM).
        """
        await self.ensure_initialized()
        final_results: List[List[MemoryUnit]] = []
        processed_unit_ids: set[str] = set()
        effective_k = max(k_limit // 3, 1) if recall_context else k_limit

        parsed_filters: Optional[BaseFilter] = _parse_filter_json(filters)
        if filters is not None and parsed_filters is None:
             print("Warning: Provided filters was invalid and ignored.") # Log if parsing failed

        # --- Validate Flags ---
        if short_term_only and long_term_only:
            raise ValueError("short_term_only and long_term_only cannot both be True.")
        if short_term_only and external_only:
            raise ValueError("short_term_only and external_only cannot both be True.")
        if long_term_only and external_only:
            print("Warning: long_term_only=True and external_only=True. Querying only external LTM.")
            long_term_only = False  # external_only takes precedence

        if short_term_only and not self._stm_enabled:
            raise ValueError("short_term_only=True but STM is not enabled.")
        if external_only and not self.external_vector_store:
            raise ValueError("external_only=True but external connection is not enabled.")

        # --- 1. Query STM (if applicable) ---
        if self._stm_enabled and not long_term_only and not external_only:
            # print("Querying Short-Term Memory (STM)...")
            stm_range = search_range if short_term_only else (threshold_for_stm_before_ltm, None)
            stm_query_results = await self._query_stm(query_vector=query_vector,
                                                filters=parsed_filters,
                                                k_limit=effective_k,
                                                search_range=stm_range)
            stm_results = [unit for unit, score in stm_query_results if unit.id not in self._context_ids]
            if stm_results:
                # print(f"Found {len(stm_results)} potential results in STM.")
                core_stm_units = [unit for unit, score in stm_results]
                linked_stm_chains = await self._process_query_results(core_stm_units,
                                                                      recall_context=recall_context,
                                                                      source='stm')
                # Update processed IDs
                for chain in linked_stm_chains:
                    if chain and chain[0].id not in processed_unit_ids:
                        final_results.append(chain)
                        processed_unit_ids.update(u.id for u in chain)

                # print(f"Added {len(linked_stm_chains)} chains from STM.")

            # If only STM was requested, return now
            if short_term_only:
                await self._increment_interaction_round()
                return final_results[:k_limit]  # Return up to k_limit chains

        # --- 2. Query LTMs (Internal and/or External, if applicable) ---
        if not short_term_only:
            if len(final_results) >= k_limit:
                return final_results[:k_limit]
            ltm_tuples: List[Tuple[MemoryUnit, float, np.ndarray]] = []
            source_map: Dict[str, str] = {}
            stores_info: List[Tuple[str, Optional[VectorStoreHandler], Optional[AsyncSQLiteHandler]]] = []
            if external_only:
                if self.external_vector_store and self.external_sqlite_handler:
                    stores_info.append(('external_sqlite', self.external_vector_store, self.external_sqlite_handler))
            elif long_term_only:
                if self.vector_store and self.sqlite_handler:
                    stores_info.append(('local_sqlite', self.vector_store, self.sqlite_handler))
            else:
                order = [('external_sqlite', self.external_vector_store, self.external_sqlite_handler),
                         ('local_sqlite', self.vector_store, self.sqlite_handler)]
                if not query_external_first:
                    order.reverse()
                for name, vs, sql in order:
                    if vs and sql:
                        stores_info.append((name, vs, sql))
            for name, vs_handler, sql_handler in stores_info:
                if vs_handler and sql_handler:
                    res_from_one_store = await self._query_ltm(
                        target_vector_store=vs_handler,
                        target_sqlite_handler=sql_handler,
                        query_vector=query_vector,
                        filters=parsed_filters,
                        k_limit=effective_k,
                        search_range=search_range
                    )
                    for unit, score, emb in res_from_one_store:
                        if unit.id not in self._context_ids and unit.id not in processed_unit_ids:
                            if unit.id not in source_map:
                                ltm_tuples.append((unit, score, emb))
                                source_map[unit.id] = name

            if ltm_tuples:
                if query_vector is not None: ltm_tuples.sort(key=lambda x: x[1], reverse=True)
                sel_units, stm_add_units, visited_ids = [], [], set()
                for unit, score, emb in ltm_tuples:
                    if len(sel_units) >= effective_k:
                        break
                    if unit.id not in processed_unit_ids:
                        sel_units.append(unit)
                        if self._stm_enabled and add_ltm_to_stm and emb is not None:
                            stm_add_units.append((unit, emb))
                        if source_map.get(unit.id) == 'local_sqlite':
                            visited_ids.add(unit.id)
                if sel_units:
                    chains = await self._process_query_results(sel_units, recall_context, 'use_source_map', source_map)
                    for chain in chains:
                        if chain and chain[0].id not in processed_unit_ids:
                            final_results.append(chain)
                            chain_ids = {u.id for u in chain}
                            processed_unit_ids.update(chain_ids)
                            visited_ids.update(uid for uid in chain_ids if source_map.get(uid) == 'local_sqlite')
                if visited_ids:
                    await self._update_visit_counts(visited_ids)
                if stm_add_units:
                    await self._add_units_to_stm(stm_add_units)
        await self._increment_interaction_round()
        return final_results[:k_limit]

    async def _process_query_results(self,
                                     core_units: List[MemoryUnit],
                                     recall_context: bool,
                                     source: str,
                                     source_map: Optional[Dict[str, str]] = None
                                     ) -> List[List[MemoryUnit]]:
        """
        Async Helper: Links units, recalls context, forms final chains.
        """
        await self.ensure_initialized()
        if not core_units: return []

        final_chains_dict: Dict[str, List[MemoryUnit]] = {}  # Maps start_id to chain
        processed_in_this_call: set[str] = set()

        # Link the core units first using the synchronous helper
        linked_core_chains = link_memory_units(core_units)

        for core_chain in linked_core_chains:
            if not core_chain or core_chain[0].id in processed_in_this_call:
                continue  # Skip if chain is empty or start already processed
            current_full_chain = list(core_chain)  # Start with the core linked units
            start_unit_id = current_full_chain[0].id

            # --- Recall Context (Neighbors) ---
            if recall_context:
                # Determine source for neighbor lookup
                first_unit = core_chain[0]
                last_unit = core_chain[-1]

                first_unit_source_type = source
                if source == 'use_source_map' and source_map:
                    first_unit_source_type = source_map.get(first_unit.id, 'local_sqlite')

                last_unit_source_type = source
                if source == 'use_source_map' and source_map:
                    last_unit_source_type = source_map.get(last_unit.id, 'local_sqlite')

                # Fetch preceding for the first unit
                pre_unit, _ = await self._fetch_neighboring_units(first_unit, source_type_for_lookup=first_unit_source_type,find_pre=True)
                if pre_unit and pre_unit.id not in self._context_ids and pre_unit.id not in processed_in_this_call:
                    current_full_chain.insert(0, pre_unit)
                    start_unit_id = pre_unit.id
                    processed_in_this_call.add(pre_unit.id)

                # Fetch succeeding for the last unit
                _, next_unit = await self._fetch_neighboring_units(last_unit, source_type_for_lookup=last_unit_source_type,find_pre=False)
                if next_unit and next_unit.id not in self._context_ids and next_unit.id not in processed_in_this_call:
                    current_full_chain.append(next_unit)
                    processed_in_this_call.add(next_unit.id)

            # Add the potentially expanded chain to the dict, handling overlaps
            if start_unit_id not in final_chains_dict or len(current_full_chain) > len(final_chains_dict[start_unit_id]):
                final_chains_dict[start_unit_id] = current_full_chain
            # Mark all units in this chain as processed for this call
            processed_in_this_call.update(unit.id for unit in current_full_chain)

        # Convert dict back to list of chains
        final_chains = list(final_chains_dict.values())

        return final_chains

    async def _update_visit_counts(self, visited_ids: set[str]):
        """Async: Updates visit counts and stages units if interval reached."""
        await self.ensure_initialized()
        if not visited_ids: return
        current_round = self.current_round
        units_marked_for_update = set()

        for unit_id in visited_ids:
            unit = await self._get_memory_unit(unit_id, use_cache=True)
            if unit:
                # Update trackers immediately
                self._visited_unit_counts[unit_id] += 1
                self._unit_last_visit_round[unit_id] = current_round
                units_marked_for_update.add(unit_id)
            # else: print(f"Warn: Cannot update visit count for unknown unit {unit_id}")

        self._internal_visit_counter += len(units_marked_for_update)

        if self._internal_visit_counter >= self.visit_update_interval:
            print(
                f"Visit update interval reached ({self._internal_visit_counter}). Staging {len(units_marked_for_update)} units for persistence.")
            staged_count = 0
            for unit_id in units_marked_for_update:
                unit = self.memory_units_cache.get(unit_id)
                if unit:
                    # Apply cumulative counts/round from trackers before staging
                    unit.visit_count += self._visited_unit_counts.pop(unit_id, 0)  # Pop after applying
                    unit.last_visit = self._unit_last_visit_round.pop(unit_id, unit.last_visit)  # Pop after applying
                    await self._stage_memory_unit_update(unit, operation='core_data_update', update_session_metadata=False)
                    staged_count += 1

            # Reset counter
            self._internal_visit_counter = 0
            # Clear any remaining entries in trackers (should be empty if pop worked)
            self._visited_unit_counts.clear()
            self._unit_last_visit_round.clear()

            # Trigger flush (will save the staged visit count updates)
            if staged_count > 0:
                await self._flush_cache(force=True)

    async def _increment_interaction_round(self):
        """Async: Increments interaction round and stages LTM update."""
        await self.ensure_initialized()
        self.current_round += 1
        if self.long_term_memory:
            self.long_term_memory.visit_count = self.current_round
            await self._stage_long_term_memory_update(self.long_term_memory)  # Stage the LTM update

    def _evaluate_python_filter(self, obj: Any, attr_expr: str, op: Union[str, FilterOperator], value: Any) -> bool:
        """Sync Helper: Evaluates a filter condition on an object attribute."""
        def get_nested_attr(target_obj, expression):
             parts = expression.split('.')
             current_val = target_obj
             for part in parts:
                 match = re.match(r"([a-zA-Z0-9_]+)\[(.+)\]", part)
                 if match:
                     attr_name, index_part = match.groups()
                     try:
                         index = eval(index_part, {}, {}) # Use restricted eval
                         if not isinstance(index, (str, int)):
                             raise ValueError("Index must be string or integer literal")
                     except Exception:
                          raise ValueError(f"Invalid index/key format: '{index_part}'")


                     if not hasattr(current_val, attr_name):
                         raise AttributeError(f"Object has no attribute '{attr_name}'")
                     current_val = getattr(current_val, attr_name)

                     try:
                         current_val = current_val[index]
                     except (TypeError, KeyError, IndexError):
                         raise AttributeError(f"Cannot access index/key '{index}'")

                 elif hasattr(current_val, part):
                     current_val = getattr(current_val, part)
                 elif isinstance(current_val, dict) and part in current_val:
                     current_val = current_val[part]
                 else:
                     raise AttributeError(f"Object or dict has no attribute/key '{part}'")
             return current_val

        try:
            attribute_value = get_nested_attr(obj, attr_expr)
            op_str = op.value if isinstance(op, FilterOperator) else op.strip().lower()
            op_func = {
                '==': operator.eq, '=': operator.eq,
                '!=': operator.ne, '<>': operator.ne,
                '>': operator.gt, '>=': operator.ge,
                '<': operator.lt, '<=': operator.le,
                'in': lambda a, b: a in b if isinstance(b, (list, tuple, set)) else False,
                'not in': lambda a, b: a not in b if isinstance(b, (list, tuple, set)) else True,
                'contains': lambda a, b: b in a if isinstance(a, (str, list, tuple, dict)) else False
            }.get(op_str)

            if op_func is None:
                raise ValueError(f"Unsupported filter operator: '{op}'")

            if op_str in ['contains', 'like']:
                 if isinstance(attribute_value, str) and isinstance(value, str):
                      return value in attribute_value
                 elif isinstance(attribute_value, (list, tuple)) and value is not None:
                       return value in attribute_value
                 elif isinstance(attribute_value, dict) and value is not None:
                       return value in attribute_value.values() if hasattr(attribute_value, 'values') else False
                 else:
                      return False

            return op_func(attribute_value, value)
        except (AttributeError, ValueError, TypeError) as e:
            # print(f"Filter evaluation error: {e} for ({attr_expr} {op} {value})")
            return False

    def _evaluate_filter_object(self, obj: Any, filter_obj: BaseFilter) -> bool:
        """Sync Helper: Recursively evaluates a BaseFilter object against a given object."""
        if isinstance(filter_obj, FieldFilter):
            return self._evaluate_python_filter(obj, filter_obj.field, filter_obj.operator, filter_obj.value)
        elif isinstance(filter_obj, LogicalFilter):
            if filter_obj.operator == 'AND':
                return all(self._evaluate_filter_object(obj, f) for f in filter_obj.filters)
            elif filter_obj.operator == 'OR':
                return any(self._evaluate_filter_object(obj, f) for f in filter_obj.filters)
            elif filter_obj.operator == 'NOT':
                if filter_obj.filters:
                    return not self._evaluate_filter_object(obj, filter_obj.filters[0])
                return False
        return False

        # --- Summarization (Async) ---

    async def summarize_long_term_memory(self, use_external_summary: bool = False, role: str = "ai", system_message: Optional[str] = None):
        """Async: Orchestrates summarizing the current LTM."""
        await self.ensure_initialized()
        if not self.llm or not self.long_term_memory:
            print("Warning: LLM or LTM object missing for summarization.")
            return
        if not hasattr(self.llm, 'ainvoke'):
            print("Warning: LLM does not support async invocation ('ainvoke'). Summarization might block.")

        # await self.flush_context()  # Flush context before summarizing
        await self._flush_cache(force=True)  # Ensure latest state is persisted

        history_summary_unit: Optional[MemoryUnit] = None
        if use_external_summary and self.external_sqlite_handler and self.external_ltm_id:
            try:
                history_summary_unit = await self.external_sqlite_handler.load_memory_unit(self.external_ltm_id)

            except Exception as e:
                print(f"Could not fetch external history summary: {e}")
        # try:
        updated_ltm, new_units, updated_units = await summarize_long_term_memory(
            memory_system=self,
            ltm_id=self.ltm_id,
            llm=self.llm,
            history_memory=history_summary_unit,
            role=role,
            max_group_size=self.max_group_size,
            max_token_count=self.max_token_count,
            system_message=system_message
        )
        # Stage results
        if updated_ltm: await self._stage_long_term_memory_update(updated_ltm)
        if new_units: await self._stage_memory_units_update(new_units, operation='add')
        if updated_units: await self._stage_memory_units_update(updated_units, operation='edge_update')

        if new_units or updated_units or updated_ltm:
            await self._flush_cache(force=True)  # Flush changes from summarization

        # except Exception as e:
        #     print(f"Error during LTM {self.ltm_id} summarization orchestration: {e}")

    async def summarize_session(self, session_id: str, role: str = "ai", system_message: Optional[str] = None):
        """Async: Orchestrates summarizing a specific session."""
        await self.ensure_initialized()
        # if not self.llm: print("Warning: LLM not configured."); return
        # if not hasattr(self.llm, 'ainvoke'): print("Warning: LLM not async.")

        # Flush context/cache if summarizing the current session
        # if session_id == self._current_session_id:
        #     await self.flush_context()
        await self._flush_cache(force=True)  # Ensure data is saved before summary
        try:
            updated_session, new_units, updated_units = await summarize_session(
                memory_system=self,
                session_id=session_id,
                llm=self.llm,
                role=role,
                max_group_size=self.max_group_size,
                max_token_count=self.max_token_count,
                system_message=system_message
            )
            # Stage results for persistence
            if updated_session: await self._stage_session_memory_update(updated_session)
            if new_units:
                await self._stage_memory_units_update(new_units, operation='add')
            if updated_units: await self._stage_memory_units_update(updated_units, operation='edge_update')

            if updated_session or new_units or updated_units:
                await self._flush_cache(force=True)
                print("successfully flush")
        except Exception as e:
            print(f"Error during session {session_id} summarization orchestration: {e}")

        # --- Synchronization ---

    async def _get_current_ltm_unit_ids_from_sqlite(self) -> Set[str]:
        """
        Helper to get all MemoryUnit IDs belonging to the current LTM from SQLite.
        This defines the scope of units that should be in any synchronized vector store
        for this LTM.
        """
        await self.ensure_initialized()
        if not self.long_term_memory:
            print("Warning: LongTermMemory object not loaded. Attempting to load for LTM unit ID retrieval.")
            await self._load_initial_state()
            if not self.long_term_memory:
                print(f"Error: Failed to load LongTermMemory for LTM ID '{self.ltm_id}'. Cannot get LTM unit IDs.")
                return set()

        ltm_unit_ids: Set[str] = set()
        ltm_unit_ids.add(self.long_term_memory.id)
        ltm_unit_ids.update(self.long_term_memory.summary_unit_ids)

        # Collect unit IDs from all sessions associated with this LTM
        sessions_to_load_ids = [sid for sid in self.long_term_memory.session_ids if
                                sid not in self.session_memories_cache]

        if sessions_to_load_ids and self.sqlite_handler:
            loaded_sessions = await self.sqlite_handler.load_session_memories(sessions_to_load_ids)
            self.session_memories_cache.update(loaded_sessions)

        for session_id in self.long_term_memory.session_ids:
            ltm_unit_ids.add(session_id)
            session_memory = self.session_memories_cache.get(session_id)
            if session_memory:
                ltm_unit_ids.update(session_memory.memory_unit_ids)
            else:
                print(
                    f"Warning: Session memory for ID '{session_id}' (part of LTM '{self.ltm_id}') not found in cache or DB.")

        return ltm_unit_ids

    async def _get_current_ltm_units_from_sqlite(self) -> Dict[str, MemoryUnit]:
        """
        Helper to get all MemoryUnit objects belonging to the current LTM from SQLite.
        Loads units into cache if not already present.
        """
        ltm_unit_ids = await self._get_current_ltm_unit_ids_from_sqlite()
        if not ltm_unit_ids:
            return {}

        await self._load_memory_units(list(ltm_unit_ids), use_cache=True)
        loaded_ltm_units: Dict[str, MemoryUnit] = {}
        missing_ids_in_cache_after_load: Set[str] = set()

        for unit_id in ltm_unit_ids:
            if unit_id in self.memory_units_cache:
                loaded_ltm_units[unit_id] = self.memory_units_cache[unit_id]
            else:
                missing_ids_in_cache_after_load.add(unit_id)

        if missing_ids_in_cache_after_load:
            print(
                f"Warning: The following LTM-specific unit IDs were not found in cache even after load attempt: {missing_ids_in_cache_after_load}")

        return loaded_ltm_units

    async def synchronize_vector_store(self, batch_size: int = 100, embedding_history_length: int = 1):
        """
        Async: Ensures the configured vector store (external like Milvus/Chroma/Qdrant, or internal
        sqlite-vec VSS table) is consistent with the primary SQLite store for the current LTM.

        For external stores:
        1. Deletes units from the vector store's LTM-specific collection that are no longer in SQLite for this LTM.
        2. Upserts all units from SQLite for this LTM into the vector store, generating embeddings.

        For sqlite-vec (internal VSS):
        1. Ensures all units from SQLite for this LTM have up-to-date embeddings in the VSS table.
        2. (Ideal cleanup of orphaned/non-LTM VSS entries would require additional sqlite_handler methods).

        Args:
            batch_size: The number of items to process in each batch for deletions and upserts.
            embedding_history_length: The history length to use when generating embeddings for new/updated units.
        """
        await self.ensure_initialized()
        if not self.sqlite_handler:
            print("Error: SQLite handler (primary data source) not available. Cannot synchronize.")
            return
        if not self.vector_store:
            print("Error: Target vector store handler not available. Cannot synchronize.")
            return
        if not self.long_term_memory:
            await self._load_initial_state()
            if not self.long_term_memory:
                print(
                    f"Error: LongTermMemory object for LTM ID '{self.ltm_id}' not loaded. Cannot determine scope for synchronization.")
                return

        print(
            f"Starting synchronization for LTM '{self.ltm_id}' with target vector store type: '{self.vector_store.vector_db_type}'...")
        ltm_sqlite_units_map = await self._get_current_ltm_units_from_sqlite()
        ltm_sqlite_unit_ids_set = set(ltm_sqlite_units_map.keys())
        if not ltm_sqlite_unit_ids_set:
            print(f"LTM '{self.ltm_id}' contains no units in SQLite.")
        else:
            print(f"Found {len(ltm_sqlite_unit_ids_set)} units in SQLite belonging to LTM '{self.ltm_id}'.")
        if self.vector_store.vector_db_type != 'sqlite-vec':
            target_collection_name = self.vector_store.collection_name
            print(f"Synchronizing with external vector store collection: '{target_collection_name}'")
            if not hasattr(self.vector_store, 'get_all_unit_ids'):
                print(
                    f"Error: VectorStoreHandler for type '{self.vector_store.vector_db_type}' does not have 'get_all_unit_ids' method. Cannot synchronize.")
                return

            external_store_unit_ids_list = await self.vector_store.get_all_unit_ids()
            external_store_unit_ids_set = set(external_store_unit_ids_list)
            print(
                f"Found {len(external_store_unit_ids_set)} units in external vector store collection '{target_collection_name}'.")

            ids_to_delete_from_external_store = list(external_store_unit_ids_set - ltm_sqlite_unit_ids_set)
            if ids_to_delete_from_external_store:
                print(
                    f"Identified {len(ids_to_delete_from_external_store)} units to delete from '{target_collection_name}'.")
                for i in range(0, len(ids_to_delete_from_external_store), batch_size):
                    batch_delete_ids = ids_to_delete_from_external_store[i:i + batch_size]
                    print(f"Deleting batch of {len(batch_delete_ids)} units from '{target_collection_name}'...")
                    try:
                        await self.vector_store.delete(ids=batch_delete_ids)
                    except Exception as e:
                        print(f"Error deleting batch from external vector store '{target_collection_name}': {e}")
            else:
                print(f"No units to delete from external vector store collection '{target_collection_name}'.")

            units_for_vector_upsert_data = []
            if not ltm_sqlite_units_map:
                print(
                    f"No units in SQLite for LTM '{self.ltm_id}' to upsert to external vector store '{target_collection_name}'.")
            else:
                print(
                    f"Preparing to upsert {len(ltm_sqlite_units_map)} units from LTM '{self.ltm_id}' to '{target_collection_name}'...")

                all_ltm_sqlite_units_list = list(ltm_sqlite_units_map.values())
                embedding_generation_tasks = [
                    self._generate_embedding_for_unit(unit, history_length=embedding_history_length)
                    for unit in all_ltm_sqlite_units_list
                ]

                print(f"Generating embeddings for {len(all_ltm_sqlite_units_list)} LTM SQLite units...")
                embedding_results = await asyncio.gather(*embedding_generation_tasks, return_exceptions=True)

                units_for_vector_upsert_data = []
                for i, unit in enumerate(all_ltm_sqlite_units_list):
                    embedding_np = embedding_results[i]
                    if isinstance(embedding_np, Exception) or embedding_np is None:
                        print(
                            f"Error generating embedding for LTM unit {unit.id}, skipping upsert to external store: {embedding_np}")
                        continue

                    unit_dict_from_obj = unit.to_dict()  # Get base dict from MemoryUnit object
                    unit_dict_vec = {
                        "id": unit.id, "content": unit.content,
                        "source": unit.source, "metadata": unit_dict_from_obj.get("metadata", {}),
                        "last_visit": unit.last_visit, "visit_count": unit.visit_count,
                        "never_delete": unit.never_delete,
                        "rank": unit.rank,
                        "embedding": embedding_np.flatten().tolist(),
                        "creation_time": unit.creation_time.timestamp() if unit.creation_time else None,
                        "end_time": unit.end_time.timestamp() if unit.end_time else None,
                    }
                    units_for_vector_upsert_data.append(unit_dict_vec)

                if units_for_vector_upsert_data:
                    print(f"Upserting {len(units_for_vector_upsert_data)} LTM units to '{target_collection_name}'...")
                    for i in range(0, len(units_for_vector_upsert_data), batch_size):
                        batch_upsert_data = units_for_vector_upsert_data[i:i + batch_size]
                        print(f"Upserting batch of {len(batch_upsert_data)} units to '{target_collection_name}'...")
                        try:
                            await self.vector_store.upsert(data=batch_upsert_data)
                        except Exception as e:
                            print(f"Error upserting batch to external vector store '{target_collection_name}': {e}")
                else:
                    print(
                        f"No valid LTM units with embeddings to upsert to external vector store '{target_collection_name}'.")

            print(f"Flushing external vector store '{target_collection_name}'...")
            await self.vector_store.flush()

        # --- Case 2: Target is sqlite-vec (internal VSS table synchronization for the current LTM) ---
        elif self.vector_store.vector_db_type == 'sqlite-vec':
            print(f"Synchronizing internal sqlite-vec VSS table for LTM '{self.ltm_id}'.")
            if not ltm_sqlite_units_map:
                print(f"No units in SQLite for LTM '{self.ltm_id}' to process for internal VSS synchronization.")
            else:
                print(
                    f"Ensuring/updating {len(ltm_sqlite_units_map)} units from LTM '{self.ltm_id}' in internal VSS table...")
                all_ltm_sqlite_units_list = list(ltm_sqlite_units_map.values())
                embedding_generation_tasks = [
                    self._generate_embedding_for_unit(unit, history_length=embedding_history_length)
                    for unit in all_ltm_sqlite_units_list
                ]

                print(f"Generating embeddings for {len(all_ltm_sqlite_units_list)} LTM SQLite units for VSS update...")
                embedding_results = await asyncio.gather(*embedding_generation_tasks, return_exceptions=True)
                units_for_sqlite_upsert_data = []
                for i, unit in enumerate(all_ltm_sqlite_units_list):
                    embedding_np = embedding_results[i]
                    unit_dict_for_sqlite = unit.to_dict()  # Start with MemoryUnit's dict

                    if isinstance(embedding_np, Exception) or embedding_np is None:
                        print(
                            f"Error generating embedding for LTM unit {unit.id} for VSS, embedding will be null in DB: {embedding_np}")
                        unit_dict_for_sqlite['embedding'] = None  # Store None if embedding failed
                    else:
                        unit_dict_for_sqlite['embedding'] = embedding_np.flatten().tolist()

                    # Ensure datetimes are ISO strings for sqlite_handler's expected format
                    unit_dict_for_sqlite[
                        "creation_time"] = unit.creation_time.isoformat() if unit.creation_time else None
                    unit_dict_for_sqlite["end_time"] = unit.end_time.isoformat() if unit.end_time else None

                    units_for_sqlite_upsert_data.append(unit_dict_for_sqlite)

                if units_for_sqlite_upsert_data:
                    print(
                        f"Upserting/updating {len(units_for_sqlite_upsert_data)} LTM units in SQLite (main data + VSS)...")
                    for i in range(0, len(units_for_sqlite_upsert_data), batch_size):
                        batch_upsert_data = units_for_sqlite_upsert_data[i:i + batch_size]
                        print(f"Upserting batch of {len(batch_upsert_data)} units to SQLite (main data + VSS)...")
                        try:
                            await self.sqlite_handler.upsert_memory_units(batch_upsert_data)
                        except Exception as e:
                            print(f"Error upserting batch to SQLite (main data + VSS): {e}")
                else:
                    print("No valid LTM units to upsert/update in SQLite (main data + VSS).")
            print(
                "Note: Full cleanup of internal VSS table for non-LTM/orphaned entries is not performed in this step and relies on specific handler methods.")

        print(
            f"Synchronization for LTM '{self.ltm_id}' with target type '{self.vector_store.vector_db_type}' is complete.")

    async def convert_sql_to_json(self, output_dir: Optional[str] = None):
        """Async: Loads all data from SQLite and saves it to JSON files."""
        await self.ensure_initialized()
        if not self.sqlite_handler:
            print("Error: SQLite handler not available for conversion.")
            return

        output_base_path = output_dir if output_dir else self.storage_base_path
        print(f"Converting SQLite data for {self.chatbot_id} to JSON in {output_base_path}...")

        try:
            # Load all data using the async handler
            sql_units = await self.sqlite_handler.load_all_memory_units()
            sql_sessions = await self.sqlite_handler.load_all_session_memories()
            sql_ltm = await self.sqlite_handler.load_all_long_term_memories_for_chatbot(self.chatbot_id)

            # Save to JSON using json_handler (assuming it's synchronous file IO)
            # Wrap sync file IO in asyncio.to_thread if it blocks significantly
            if sql_units:
                memory_units_dict = {k: v.to_dict() for k, v in sql_units.items()}
                await asyncio.to_thread(json_handler.save_memory_units_json, memory_units_dict, self.chatbot_id,
                                        output_base_path)
                print(f"Saved {len(sql_units)} memory units to JSON.")
            if sql_sessions:
                session_memories_dict = {k: v.to_dict() for k, v in sql_sessions.items()}
                await asyncio.to_thread(json_handler.save_session_memories_json, session_memories_dict, self.chatbot_id,
                                        output_base_path)
                print(f"Saved {len(sql_sessions)} session memories to JSON.")
            if sql_ltm:
                long_term_memories_dict = {k: v.to_dict() for k, v in sql_ltm.items()}
                await asyncio.to_thread(json_handler.save_long_term_memories_json, long_term_memories_dict,
                                        self.chatbot_id, output_base_path)
                print(f"Saved {len(sql_ltm)} LTM object(s) to JSON.")

            print("SQLite to JSON conversion complete.")

        except Exception as e:
            print(f"Error during SQLite to JSON conversion: {e}")

    async def convert_json_to_sqlite(self, input_dir: Optional[str] = None):
        """Async: Loads data from JSON files and saves it into the SQLite database."""
        await self.ensure_initialized()
        if not self.sqlite_handler:
            print("Error: SQLite handler not available for conversion.")
            return

        input_base_path = input_dir if input_dir else self.storage_base_path
        print(f"Converting JSON data for {self.chatbot_id} from {input_base_path} to SQLite...")

        # conn = await self.sqlite_handler._get_connection()

        try:
            # Load from JSON (wrap sync file IO)
            json_units_dict = await asyncio.to_thread(json_handler.load_memory_units_json, self.chatbot_id,
                                                      input_base_path)
            mus = [MemoryUnit.from_dict(mu_dict) for mu_dict in json_units_dict.values()]

            json_sessions_dict = await asyncio.to_thread(json_handler.load_session_memories_json, self.chatbot_id,
                                                         input_base_path)
            sms = [SessionMemory.from_dict(sm_dict) for sm_dict in json_sessions_dict.values()]

            json_ltms_dict = await asyncio.to_thread(json_handler.load_long_term_memories_json, self.chatbot_id,
                                                     input_base_path)
            ltms = [LongTermMemory.from_dict(ltm_dict) for ltm_dict in json_ltms_dict.values()]

            print(f"Loaded {len(mus)} units, {len(sms)} sessions, {len(ltms)} LTMs from JSON.")

            # Prepare data for async SQLite handler (expects dicts with specific formats)
            units_to_insert = [mu.to_dict() for mu in mus]
            for unit_dict in units_to_insert:  # Convert datetimes
                unit_dict["creation_time"] = unit_dict["creation_time"] if unit_dict["creation_time"] else None
                unit_dict["end_time"] = unit_dict["end_time"] if unit_dict["end_time"] else None
                if self.vector_store and self.vector_store.vector_db_type == 'sqlite-vec':
                    unit_dict['embedding'] = None  # Will be generated if needed on query/sync

            sessions_to_insert = [sm.to_dict() for sm in sms]
            for session_dict in sessions_to_insert:  # Convert datetimes
                session_dict["creation_time"] = session_dict["creation_time"] if session_dict["creation_time"] else None
                session_dict["end_time"] = session_dict["end_time"] if session_dict["end_time"] else None

            ltms_to_insert = [ltm.to_dict() for ltm in ltms]
            for ltm_dict in ltms_to_insert:  # Convert datetimes
                ltm_dict["creation_time"] = ltm_dict["creation_time"] if ltm_dict["creation_time"] else None
                ltm_dict["end_time"] = ltm_dict["end_time"] if ltm_dict["end_time"] else None

            # Save to SQLite using async handlers
            await self.sqlite_handler.set_foreign_keys(False)
            await self.sqlite_handler.begin()
            try:
                print("Saving to SQLite...")
                if ltms_to_insert:
                    await self.sqlite_handler.upsert_long_term_memories(ltms_to_insert)
                if sessions_to_insert:
                    await self.sqlite_handler.upsert_session_memories(sessions_to_insert)
                if units_to_insert:
                    await self.sqlite_handler.upsert_memory_units(units_to_insert)

                await self.sqlite_handler.commit()
                print("Transaction committed.")
            except Exception as e:
                print(f"Error during transaction, rolling back: {e}")
                await self.sqlite_handler.rollback() # Rollback on error
                raise

            finally:
                print("Re-enabling foreign keys...")
                await self.sqlite_handler.set_foreign_keys(True)
            print("Checking foreign key integrity after conversion...")
            fk_errors = await self.sqlite_handler.check_foreign_keys()
            if fk_errors:
                print(f"WARNING: Foreign key check found {len(fk_errors)} issues after conversion:")
                for error in fk_errors:
                    print(
                        f"  - Table: {error.get('table')}, RowID: {error.get('rowid')}, Parent: {error.get('parent')}, FK_ID: {error.get('fkid')}")
            else:
                print("Foreign key integrity check passed.")

            print("JSON to SQLite conversion complete.")
            print("Reloading current state from SQLite...")
            await self._load_initial_state()

        # Commit should be handled within the insert methods by the handler

        except FileNotFoundError:
            print(f"Error: JSON files not found in {input_base_path} for chatbot {self.chatbot_id}.")
        except Exception as e:
            print(f"Error during JSON to SQLite conversion: {e}")

    async def convert_json_to_db(self, input_dir: Optional[str] = None, embedding_history_length: int = 1,
                                     batch_size: int = 100):
        """Async: Loads units from JSON, generates embeddings, and upserts into Vector Store."""
        await self.ensure_initialized()
        if not self.vector_store:
            print("Error: External vector store not configured. Cannot convert.")
            return

        input_base_path = input_dir if input_dir else self.storage_base_path
        print(f"Converting JSON memory units from {input_base_path} for {self.chatbot_id} to Vector Store...")

        try:
            json_units_dict = await asyncio.to_thread(json_handler.load_memory_units_json, self.chatbot_id,
                                                      input_base_path)
            all_units = [MemoryUnit.from_dict(mu_dict) for mu_dict in json_units_dict.values()]
            print(f"Loaded {len(all_units)} units from JSON.")
            if not all_units: return

            processed_count = 0
            error_count = 0

            for i in range(0, len(all_units), batch_size):
                batch_units = all_units[i:i + batch_size]
                print(f"Processing batch {i // batch_size + 1}/{(len(all_units) + batch_size - 1) // batch_size}...")
                batch_to_upsert = []

                # Process batch units concurrently? Use asyncio.gather for embedding generation?
                embedding_tasks = []
                unit_map = {}
                for unit in batch_units:
                    if unit:
                        unit_map[unit.id] = unit
                        embedding_tasks.append(self._generate_embedding_for_unit(unit, embedding_history_length))

                embeddings_results = await asyncio.gather(*embedding_tasks, return_exceptions=True)

                for unit_id, result in zip(unit_map.keys(), embeddings_results):
                    unit = unit_map[unit_id]
                    if isinstance(result, Exception) or result is None:
                        print(
                            f"Error generating embedding for unit {unit_id}: {result if isinstance(result, Exception) else 'None returned'}")
                        error_count += 1
                        continue

                    embedding = result.flatten().tolist()  # result should be numpy array
                    # Prepare data dict
                    unit_dict_vec = unit.to_dict()
                    unit_dict_vec['embedding'] = embedding
                    unit_dict_vec["creation_time"] = unit.creation_time.timestamp() if unit.creation_time else None
                    unit_dict_vec["end_time"] = unit.end_time.timestamp() if unit.end_time else None
                    batch_to_upsert.append(unit_dict_vec)
                    print(unit_id)

                # Upsert batch
                if batch_to_upsert:
                    try:
                        await self.vector_store.upsert(batch_to_upsert)
                        processed_count += len(batch_to_upsert)
                    except Exception as e:
                        print(f"Error upserting batch {i // batch_size + 1} to vector store: {e}")
                        error_count += len(batch_to_upsert)  # Count failed batch as errors

            await self.vector_store.flush()  # Final flush
            print(f"JSON to Vector Store conversion finished.")
            print(f"Successfully processed: {processed_count} units.")
            print(f"Errors encountered: {error_count} units.")

        except FileNotFoundError:
            print(f"Error: JSON files not found in {input_base_path} for chatbot {self.chatbot_id}.")
        except Exception as e:
            print(f"Error during JSON to Vector Store conversion: {e}")

    async def _generate_embedding_for_unit(self, unit: MemoryUnit, history_length: int) -> Optional[np.ndarray]:
        """Async helper to generate embedding for a single unit."""
        try:
            content_to_embed = await self._get_formatted_content_with_history(unit, history_length)
            embedding_np = self.embedding_handler.get_embedding(content_to_embed)
            return embedding_np
        except Exception as e:
            print(f"Failed embedding for unit {unit.id}: {e}")
            return None

    async def clear_context(self):
        for unit in self.context:
            await self._stage_memory_unit_deletion(unit.id)
        await self._flush_cache(force=True)
        self.context.clear()
        print("Context has been cleared.")

    async def close(self, auto_summarize: bool = False, role: str = "ai", system_message: Optional[str] = None):
        """Async: Flushes changes, optionally summarizes, closes connections."""
        print(f"Closing MemorySystem async...")
        # Stage final visit counts before flush
        if self._visited_unit_counts:
            print("Staging final visit counts...")
            staged_count = 0
            # Create a copy of keys to iterate over as the dict might change
            unit_ids_to_update = list(self._visited_unit_counts.keys())
            for unit_id in unit_ids_to_update:
                unit = self.memory_units_cache.get(unit_id)
                if unit:
                    unit.visit_count += self._visited_unit_counts.pop(unit_id, 0)
                    unit.last_visit = self._unit_last_visit_round.pop(unit_id, unit.last_visit)
                    await self._stage_memory_unit_update(unit, operation='core_data_update', update_session_metadata=False)
                    staged_count += 1
            self._visited_unit_counts.clear()  # Ensure cleared
            self._unit_last_visit_round.clear()
            self._internal_visit_counter = 0  # Reset counter

        # Flush context and final cache state
        # await self.flush_context()
        print("Flushing final cache before close...")
        await self._flush_cache(force=True)

        # Auto-summarize
        if auto_summarize and self.llm:
            print("Performing final auto-summarization...")
            # try:
            # Ensure external connection is available if needed for history
            # Might need to pass external config here or ensure it's enabled earlier
            await self.summarize_long_term_memory(use_external_summary=False, role=role, system_message=system_message)
            print("Flushing cache after final summarization...")
            await self._flush_cache(force=True)  # Flush again after summarization
            # except Exception as e:
            #     print(f"Error during final auto-summarization: {e}")

        # Close handlers
        if self.sqlite_handler:
            try:
                await self.sqlite_handler.close()
                print("SQLite handler closed.")
            except Exception as e:
                print(f"Error closing SQLite handler: {e}")
            self.sqlite_handler = None

        if self.vector_store:
            try:
                # If it's sqlite-vec, the handler was already closed above
                if self.vector_store.vector_db_type != 'sqlite-vec':
                    await self.vector_store.close(flush=False)  # Already flushed
                    print("Vector store handler closed.")
            except Exception as e:
                print(f"Error closing vector store handler: {e}")
            self.vector_store = None
        if self.external_sqlite_handler:
            try:
                await self.external_sqlite_handler.close()
            except Exception as e:
                print(f"Error closing external SQLite handler: {e}")
            self.external_sqlite_handler = None
        if self.external_vector_store:
            try:
                await self.external_vector_store.close(flush=False)
                print("External vector store handler closed.")
            except Exception as e:
                print(f"Error closing external vector store handler: {e}")
            self.external_vector_store = None
        await self.clear_all()
        self._is_initialized = False
        print("MemorySystem closed.")

    def get_embedding(self, text: Union[str, List[str]]):
        return self.embedding_handler.get_embedding(text)
