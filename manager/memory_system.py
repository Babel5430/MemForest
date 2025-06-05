import asyncio
import datetime
import threading
from concurrent.futures import Future, TimeoutError
from typing import Callable, Any, Optional, Dict, List, Tuple, Union, Iterable, Set

from MemForest.manager.async_memory_system import AsyncMemorySystem
from MemForest.persistence.filter_types import BaseFilter
from MemForest.memory import MemoryUnit,SessionMemory,LongTermMemory


class _AsyncRunner:
    def __init__(self):
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._loop_ready_event = threading.Event()

    def _start_loop_thread_if_needed(self):
        if self._thread is None or not self._thread.is_alive():
            self._loop_ready_event.clear()
            self._loop = asyncio.new_event_loop()
            self._thread = threading.Thread(target=self._run_loop_target, daemon=True)
            self._thread.start()
            # Wait for the loop to be set up in the new thread
            if not self._loop_ready_event.wait(timeout=5):
                print(f"ERROR: AsyncRunner event loop thread {self._thread.ident if self._thread else 'Unknown'} failed to start in time. Loop object: {self._loop}")
                if self._thread and self._thread.is_alive():
                    if self._loop and self._loop.is_running():
                         try:
                            self._loop.call_soon_threadsafe(self._loop.stop)
                         except Exception as e_stop:
                             print(f"ERROR: AsyncRunner: Exception trying to stop unresponsive loop: {e_stop}")
                    self._thread.join(timeout=2)
                self._thread = None
                # self._loop = None # Mark loop as unusable (already done in shutdown, or will be None if never assigned in new thread)
                raise RuntimeError(f"AsyncRunner event loop thread failed to start in time for loop {id(self._loop if self._loop else None)}.")
            print(f"AsyncRunner: Event loop {id(self._loop)} is ready in thread {self._thread.ident if self._thread else 'Unknown'}.")

    def _run_loop_target(self):
        loop_for_this_thread = self._loop

        if not loop_for_this_thread:
            print(
                f"ERROR: _AsyncRunner._loop was None when _run_loop_target for thread {threading.get_ident()} started.")
            # self._loop_ready_event.set()
            return

        current_thread_id = threading.get_ident()
        print(f"AsyncRunner: Thread {current_thread_id} starting to manage loop {id(loop_for_this_thread)}.")
        asyncio.set_event_loop(loop_for_this_thread)
        self._loop_ready_event.set()

        try:
            loop_for_this_thread.run_forever()
        except Exception as e_run_forever:
            print(
                f"ERROR: Exception in loop {id(loop_for_this_thread)}.run_forever() in thread {current_thread_id}: {e_run_forever}")
        finally:
            print(
                f"AsyncRunner: Loop {id(loop_for_this_thread)} run_forever finished or errored. Starting cleanup in thread {current_thread_id}.")
            try:
                tasks = asyncio.all_tasks(loop_for_this_thread)
                if tasks:
                    print(
                        f"AsyncRunner: Cancelling {len(tasks)} tasks for loop {id(loop_for_this_thread)} in thread {current_thread_id}.")
                    for task_idx, task in enumerate(tasks):
                        if not task.done():
                            task.cancel()
                    loop_for_this_thread.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))
                    print(
                        f"AsyncRunner: Tasks for loop {id(loop_for_this_thread)} cancelled and gathered in thread {current_thread_id}.")
                else:
                    print(
                        f"AsyncRunner: No tasks to cancel for loop {id(loop_for_this_thread)} in thread {current_thread_id}.")
            except RuntimeError as e_cleanup_tasks:
                print(
                    f"Error during task cleanup for loop {id(loop_for_this_thread)} in thread {current_thread_id}: {e_cleanup_tasks} (Possibly loop already closed or other issue)")
            except Exception as e_cleanup_tasks_other:
                print(
                    f"Generic error during task cleanup for loop {id(loop_for_this_thread)} in thread {current_thread_id}: {e_cleanup_tasks_other}")
            finally:
                if not loop_for_this_thread.is_closed():
                    loop_for_this_thread.close()
                    print(f"AsyncRunner: Event loop {id(loop_for_this_thread)} closed in _run_loop_target thread {current_thread_id}.")
                else:
                    print(f"AsyncRunner: Event loop {id(loop_for_this_thread)} was already closed before explicit close in _run_loop_target thread {current_thread_id}.")

    def run_coroutine(self, coro_func: Callable, *args: Any, **kwargs: Any) -> Any:
        self._start_loop_thread_if_needed()  # This will raise RuntimeError if timeout occurs

        if not self._loop or not self._loop.is_running():
            loop_status = "None" if not self._loop else ("running" if self._loop.is_running() else "not running/closed")
            print(
                f"ERROR: AsyncRunner event loop is not available or not running after start attempt. Loop: {id(self._loop) if self._loop else 'N/A'}, Status: {loop_status}")
            raise RuntimeError(
                f"AsyncRunner event loop is not available or not running after start attempt. Loop ID: {id(self._loop) if self._loop else 'N/A'}, Status: {loop_status}")

        coroutine_obj = coro_func(*args, **kwargs)
        future_obj: Future = asyncio.run_coroutine_threadsafe(coroutine_obj, self._loop)
        try:
            return future_obj.result(timeout=300)
        except TimeoutError as e:
            print(f"Timeout waiting for async operation: {coro_func.__name__} using loop {id(self._loop)}")
            # Attempt to cancel the coroutine future on timeout
            if not future_obj.done():
                future_obj.cancel()
            raise TimeoutError(f"Async operation {coro_func.__name__} timed out.") from e
        except Exception as e:
            # Includes CancelledError if future_obj was cancelled by another mechanism
            print(
                f"Exception from async operation {coro_func.__name__} using loop {id(self._loop)}: {type(e).__name__} - {e}")
            raise  # Re-raise the original exception

    def shutdown(self, wait=True):
        # Capture current thread and loop for logging and operations
        thread_to_join = self._thread
        loop_to_stop = self._loop

        current_thread_id = threading.get_ident()
        print(
            f"AsyncRunner: Shutdown requested from thread {current_thread_id}. Target thread: {thread_to_join.ident if thread_to_join else 'None'}, Target loop: {id(loop_to_stop) if loop_to_stop else 'None'}")

        if loop_to_stop and loop_to_stop.is_running():
            print(f"AsyncRunner: Requesting loop {id(loop_to_stop)} to stop (from thread {current_thread_id}).")
            try:
                loop_to_stop.call_soon_threadsafe(loop_to_stop.stop)
            except RuntimeError as e_stop_runtime:  # e.g. if loop is closing/closed
                print(f"AsyncRunner: Runtime error stopping loop {id(loop_to_stop)}: {e_stop_runtime}")
            except Exception as e_stop_generic:
                print(f"AsyncRunner: Generic error stopping loop {id(loop_to_stop)}: {e_stop_generic}")
        elif loop_to_stop:
            print(
                f"AsyncRunner: Loop {id(loop_to_stop)} exists but is not running. No stop signal sent (from thread {current_thread_id}).")
        else:
            print(f"AsyncRunner: No loop instance to stop (from thread {current_thread_id}).")

        if thread_to_join and thread_to_join.is_alive():
            print(f"AsyncRunner: Joining thread {thread_to_join.ident} (from thread {current_thread_id}).")
            if wait:
                thread_to_join.join(timeout=10)  # Increased timeout for graceful task cancellation
            if thread_to_join.is_alive():
                print(
                    f"Warning: _AsyncRunner thread {thread_to_join.ident} did not shut down cleanly after stop signal and join timeout (requested from thread {current_thread_id}).")
            else:
                print(
                    f"AsyncRunner: Thread {thread_to_join.ident} joined successfully (requested from thread {current_thread_id}).")
        elif thread_to_join:
            print(
                f"AsyncRunner: Thread {thread_to_join.ident} was not alive (requested from thread {current_thread_id}).")
        else:
            print(f"AsyncRunner: No thread instance to join (requested from thread {current_thread_id}).")

        # These should always be reset to allow for a clean restart by _start_loop_thread_if_needed
        self._thread = None
        self._loop = None
        print(
            f"AsyncRunner: Shutdown complete. _thread and _loop are None (requested from thread {current_thread_id}).")


class MemorySystem:
    def __init__(self,
                 chatbot_id: str,
                 ltm_id: str,
                 embedding_handler: Any,
                 llm: Optional[Any] = None,
                 vector_store_config: Optional[Dict[str, Any]] = None,
                 base_path: Optional[str] = None,
                 max_context_length: int = 10,
                 visit_update_interval: int = 20,
                 saving_interval: int = 20,
                 max_vector_entities: int = 20000,
                 forget_percentage: float = 0.10,
                 max_group_size: int = 60,
                 max_token_count: int = 2000
                 ):
        self._async_system = AsyncMemorySystem(
            chatbot_id=chatbot_id,
            ltm_id=ltm_id,
            embedding_handler=embedding_handler,
            llm=llm,
            vector_store_config=vector_store_config,
            base_path=base_path,
            max_context_length=max_context_length,
            visit_update_interval=visit_update_interval,
            saving_interval=saving_interval,
            max_vector_entities=max_vector_entities,
            forget_percentage=forget_percentage,
            max_group_size=max_group_size,
            max_token_count=max_token_count
        )
        self.embedding_handler = self._async_system.embedding_handler
        self._runner = _AsyncRunner()

    def _run_async_delegate(self, coro_func: Callable, *args: Any, **kwargs: Any) -> Any:
        """Internal helper to delegate to the _AsyncRunner instance."""
        return self._runner.run_coroutine(coro_func, *args, **kwargs)

    # --- Initialization and State ---
    def _async_initialize(self):
        """Sync wrapper for internal async _async_initialize."""
        self._run_async_delegate(self._async_system._async_initialize)

    def ensure_initialized(self):
        """Sync wrapper for internal async ensure_initialized."""
        self._run_async_delegate(self._async_system.ensure_initialized)

    def get_current_sesssion_id(self) -> Optional[str]:
        """Sync wrapper for get_current_sesssion_id."""
        return self._async_system.get_current_sesssion_id() # Direct call, no async needed

    def _load_initial_state(self):
        """Sync wrapper for internal async _load_initial_state."""
        self._run_async_delegate(self._async_system._load_initial_state)

    def start_session(self, session_id: Optional[str] = None):
        """Sync wrapper for async start_session."""
        self._run_async_delegate(self._async_system.start_session, session_id)

    # --- Persistence and Caching ---
    def _flush_cache(self, force: bool = False):
        """Sync wrapper for internal async _flush_cache."""
        self._run_async_delegate(self._async_system._flush_cache, force)

    def _stage_memory_unit_update(self,
                                  memory_unit: Optional[MemoryUnit],
                                  unit_id: Optional[str] = None,
                                  operation: str = 'add',
                                  update_type: Optional[str] = None,
                                  update_details: Optional[Any] = None,
                                  update_session_metadata: bool = True
                                  ):
        """Sync wrapper for internal async _stage_memory_unit_update."""
        self._run_async_delegate(self._async_system._stage_memory_unit_update, memory_unit, unit_id, operation, update_type, update_details, update_session_metadata)

    def _stage_memory_unit_deletion(self, unit_id: str):
        """Sync wrapper for internal async _stage_memory_unit_deletion."""
        self._run_async_delegate(self._async_system._stage_memory_unit_deletion, unit_id)

    def _stage_session_memory_update(self, session: SessionMemory):
        """Sync wrapper for internal async _stage_session_memory_update."""
        self._run_async_delegate(self._async_system._stage_session_memory_update, session)

    def _stage_long_term_memory_update(self, ltm: LongTermMemory):
        """Sync wrapper for internal async _stage_long_term_memory_update."""
        self._run_async_delegate(self._async_system._stage_long_term_memory_update, ltm)

    def _stage_memory_units_update(self, units: Iterable[MemoryUnit], operation: str = 'add'):
        """Sync wrapper for internal async _stage_memory_units_update."""
        self._run_async_delegate(self._async_system._stage_memory_units_update, list(units), operation)

    # --- Data Retrieval ---
    def _get_memory_unit(self, unit_id: str, use_cache: bool = True) -> Optional[MemoryUnit]:
        """Sync wrapper for internal async _get_memory_unit."""
        return self._run_async_delegate(self._async_system._get_memory_unit, unit_id, use_cache)

    def _get_session_memory(self, session_id: str, use_cache: bool = True) -> Optional[SessionMemory]:
        """Sync wrapper for internal async _get_session_memory."""
        return self._run_async_delegate(self._async_system._get_session_memory, session_id, use_cache)

    def _get_long_term_memory(self, ltm_id: str) -> Optional[LongTermMemory]:
        """Sync wrapper for internal async _get_long_term_memory."""
        return self._run_async_delegate(self._async_system._get_long_term_memory, ltm_id)

    def _load_memory_units(self, unit_ids: List[str], use_cache: bool = True) -> Dict[str, MemoryUnit]:
        """Sync wrapper for internal async _load_memory_units."""
        return self._run_async_delegate(self._async_system._load_memory_units, unit_ids, use_cache)

    def _load_sessions(self, session_ids: List[str], use_cache: bool = True) -> Dict[str, SessionMemory]:
        """Sync wrapper for internal async _load_sessions."""
        return self._run_async_delegate(self._async_system._load_sessions, session_ids, use_cache)

    def _load_units_for_session(self, session_id: str) -> Dict[str, MemoryUnit]:
        """Sync wrapper for internal async _load_units_for_session."""
        return self._run_async_delegate(self._async_system._load_units_for_session, session_id)

    def _get_current_session(self) -> Optional[SessionMemory]:
        """Sync wrapper for internal async _get_current_session."""
        return self._run_async_delegate(self._async_system._get_current_session)

    # --- Memory Management ---
    def remove_session(self, session_id: str):
        """Sync wrapper for async remove_session."""
        self._run_async_delegate(self._async_system.remove_session, session_id)

    def _get_formatted_content_with_history(self, memory_unit: MemoryUnit, history_length: int = 1) -> str:
        """Sync wrapper for internal async _get_formatted_content_with_history."""
        return self._run_async_delegate(self._async_system._get_formatted_content_with_history, memory_unit, history_length)

    def _check_and_forget_memory(self):
        """Sync wrapper for internal async _check_and_forget_memory."""
        self._run_async_delegate(self._async_system._check_and_forget_memory)

    # --- External Connection ---
    def enable_external_connection(self,
                                   external_chatbot_id: str,
                                   external_ltm_id: str,
                                   external_vector_store_config: Dict[str, Any],
                                   external_sqlite_base_path: Optional[str] = None):
        """Sync wrapper for async enable_external_connection."""
        self._run_async_delegate(self._async_system.enable_external_connection, external_chatbot_id, external_ltm_id, external_vector_store_config, external_sqlite_base_path)

    # --- Short-Term Memory (STM) ---
    def _restore_session(self, session_id: str):
        """Sync wrapper for internal async _restore_stm_from_session."""
        self._run_async_delegate(self._async_system._restore_session, session_id)

    def enable_stm(self, capacity: int = 200, restore_session_id: Optional[str] = None):
        """Sync wrapper for async enable_stm."""
        self._run_async_delegate(self._async_system.enable_stm, capacity, restore_session_id)

    def disable_stm(self):
        """Sync wrapper for async disable_stm."""
        self._run_async_delegate(self._async_system.disable_stm)

    def _remove_oldest_from_stm(self):
        """Sync wrapper for internal async _remove_oldest_from_stm."""
        self._run_async_delegate(self._async_system._remove_oldest_from_stm)

    def _remove_unit_from_stm(self, unit_id: str):
        """Sync wrapper for internal async _remove_unit_from_stm."""
        self._run_async_delegate(self._async_system._remove_unit_from_stm, unit_id)

    def _add_units_to_stm(self, memory_units_with_embeddings: List[Tuple[MemoryUnit, Any]]): # Use Any for embedding type for simplicity
        """Sync wrapper for internal async _add_units_to_stm."""
        self._run_async_delegate(self._async_system._add_units_to_stm, memory_units_with_embeddings)

    # --- Context Management ---
    def _add_to_context(self, memory_unit: MemoryUnit, embedding_history_length: int = 1):
        """Sync wrapper for internal async _add_to_context."""
        self._run_async_delegate(self._async_system._add_to_context, memory_unit, embedding_history_length)

    def add_memory(self, message: Union[str, Any], source: Optional[str] = None,
                   metadata: Optional[Dict[str, Any]] = None, creation_time: Optional[datetime.datetime] = None,
                   memory_unit_id: Optional[str] = None) -> Optional[MemoryUnit]:
        """Sync wrapper for async add_memory."""
        return self._run_async_delegate(self._async_system.add_memory, message, source, metadata, creation_time, memory_unit_id)

    def get_context(self, length: Optional[int] = None) -> List[MemoryUnit]:
        """Sync wrapper for async get_context."""
        return self._run_async_delegate(self._async_system.get_context, length)

    def clear_all(self):
        """Sync wrapper for async clear_all."""
        self._run_async_delegate(self._async_system.clear_all)

    def flush_context(self, embedding_history_length: int = 1):
        """Sync wrapper for internal async flush_context."""
        self._run_async_delegate(self._async_system.flush_context, embedding_history_length)

    def clear_context(self):
        """Sync wrapper for async clear_context."""
        self._run_async_delegate(self._async_system.clear_context)

    # --- Querying ---
    def _query_stm(self,
                   query_vector: Optional[List[float]] = None,
                   filters: Optional[BaseFilter] = None,
                   k_limit: int = 5,
                   search_range: Optional[Tuple[Optional[float], Optional[float]]] = (0.75, None)
                   ) -> List[Tuple[MemoryUnit, float]]:
        """Sync wrapper for internal async _query_stm."""
        return self._run_async_delegate(self._async_system._query_stm, query_vector, filters, k_limit, search_range)

    def _query_ltm(self,
                   target_vector_store: Any, # VectorStoreHandler
                   target_sqlite_handler: Optional[Any] = None, # AsyncSQLiteHandler
                   query_vector: Optional[List[float]] = None,
                   filters: Optional[BaseFilter] = None,
                   k_limit: int = 5,
                   search_range: Optional[Tuple[Optional[float], Optional[float]]] = (0.75, None)
                   ) -> List[Tuple[MemoryUnit, float, Any]]: # Any for embedding
        """Sync wrapper for internal async _query_ltm."""
        return self._run_async_delegate(self._async_system._query_ltm, target_vector_store, target_sqlite_handler, query_vector, filters, k_limit, search_range)

    def _fetch_neighboring_units(self,
                                 unit: MemoryUnit,
                                 source_type_for_lookup: str,
                                 find_pre: bool = True
                                 ) -> Tuple[Optional[MemoryUnit], Optional[MemoryUnit]]:
        """Sync wrapper for internal async _fetch_neighboring_units."""
        return self._run_async_delegate(self._async_system._fetch_neighboring_units, unit, source_type_for_lookup, find_pre)

    def query(self,
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
        """Sync wrapper for async query. Accepts a JSON-like dictionary for filters."""
        return self._run_async_delegate(self._async_system.query,
                          query_vector,
                          filters,
                          k_limit,
                          search_range,
                          recall_context,
                          add_ltm_to_stm,
                          query_external_first,
                          short_term_only,
                          long_term_only,
                          external_only,
                          threshold_for_stm_before_ltm)

    def _process_query_results(self,
                               core_units: List[MemoryUnit],
                               recall_context: bool,
                               source: str,
                               source_map: Optional[Dict[str, str]] = None
                               ) -> List[List[MemoryUnit]]:
        """Sync wrapper for internal async _process_query_results."""
        return self._run_async_delegate(self._async_system._process_query_results, core_units, recall_context, source, source_map)

    def _update_visit_counts(self, visited_ids: Set[str]):
        """Sync wrapper for internal async _update_visit_counts."""
        self._run_async_delegate(self._async_system._update_visit_counts, visited_ids)

    def _increment_interaction_round(self):
        """Sync wrapper for internal async _increment_interaction_round."""
        self._run_async_delegate(self._async_system._increment_interaction_round)

    # --- Filtering (Sync Helpers) ---
    def _evaluate_python_filter(self, obj: Any, attr_expr: str, op: Union[str, Any], value: Any) -> bool: # Any for FilterOperator
        """Sync wrapper for internal sync _evaluate_python_filter. No async runner needed."""
        return self._async_system._evaluate_python_filter(obj, attr_expr, op, value)

    def _evaluate_filter_object(self, obj: Any, filter_obj: BaseFilter) -> bool:
        """Sync wrapper for internal sync _evaluate_filter_object. No async runner needed."""
        return self._async_system._evaluate_filter_object(obj, filter_obj)

    # --- Summarization ---
    def summarize_long_term_memory(self, use_external_summary: bool = False, role: str = "ai", system_message: Optional[str] = None):
        """Sync wrapper for async summarize_long_term_memory."""
        self._run_async_delegate(self._async_system.summarize_long_term_memory, use_external_summary, role, system_message = system_message)

    def summarize_session(self, session_id: str, role: str = "ai", system_message: Optional[str] = None):
        """Sync wrapper for async summarize_session."""
        self._run_async_delegate(self._async_system.summarize_session, session_id, role, system_message)

    # --- Synchronization and Conversion ---
    def _get_current_ltm_unit_ids_from_sqlite(self) -> Set[str]:
        """Sync wrapper for internal async _get_current_ltm_unit_ids_from_sqlite."""
        return self._run_async_delegate(self._async_system._get_current_ltm_unit_ids_from_sqlite)

    def _get_current_ltm_units_from_sqlite(self) -> Dict[str, MemoryUnit]:
        """Sync wrapper for internal async _get_current_ltm_units_from_sqlite."""
        return self._run_async_delegate(self._async_system._get_current_ltm_units_from_sqlite)

    def synchronize_vector_store(self, batch_size=100, embedding_history_length: int = 1):
        """Sync wrapper for async synchronize_vector_store."""
        self._run_async_delegate(self._async_system.synchronize_vector_store, batch_size, embedding_history_length)

    def convert_sql_to_json(self, output_dir: Optional[str] = None):
        """Sync wrapper for async convert_sql_to_json."""
        self._run_async_delegate(self._async_system.convert_sql_to_json, output_dir)

    def convert_json_to_sqlite(self, input_dir: Optional[str] = None):
        """Sync wrapper for async convert_json_to_sqlite."""
        self._run_async_delegate(self._async_system.convert_json_to_sqlite, input_dir)

    def convert_json_to_db(self, input_dir: Optional[str] = None, embedding_history_length: int = 1,
                           batch_size: int = 100):
        """Sync wrapper for async convert_json_to_db."""
        self._run_async_delegate(self._async_system.convert_json_to_db, input_dir, embedding_history_length, batch_size)

    # --- Embeddings ---
    def _generate_embedding_for_unit(self, unit: MemoryUnit, history_length: int) -> Optional[Any]: # Use Any for embedding type
        """Sync wrapper for internal async _generate_embedding_for_unit."""
        return self._run_async_delegate(self._async_system._generate_embedding_for_unit, unit, history_length)

    def get_embedding(self, text: Union[str, List[str]]):
        """Sync wrapper for get_embedding."""
        return self._async_system.get_embedding(text) # Direct call, no async needed

    # --- Deletion and Closing ---
    def delete_ltm(self, ltm_id: Optional[str] = None):
        """Sync wrapper for async delete_ltm."""
        self._run_async_delegate(self._async_system.delete_ltm, ltm_id)

    def close(self, auto_summarize: bool = False, role: str = "ai", system_message: Optional[str] = None):
        """Sync wrapper for async close."""
        self._run_async_delegate(self._async_system.close, auto_summarize, role, system_message = system_message)
        # Shutdown the async runner after closing the system
        # self._runner.shutdown()

    # --- Helpers ---
    def if_stm_enabled(self):
        """Sync check for STM status."""
        return self._async_system._stm_enabled

    def get_stm(self):
        """Sync access to STM units (read-only recommended)."""
        return self._async_system._short_term_memory_units

    def _get_unit_group_id(self, unit_id: str) -> Optional[Union[str, List[str]]]:
        return self._run_async_delegate(self._async_system._get_unit_group_id, unit_id)

    def _handle_group_change_side_effects(self, unit_id: str, rank: int, old_group: Optional[Union[str, List[str]]],
                                          new_group: Optional[Union[str, List[str]]]):
        return self._run_async_delegate(self._async_system._handle_group_change_side_effects, unit_id, rank, old_group, new_group)

    def _stage_session_memory_deletion(self, session_id: str):
        return self._run_async_delegate(self._stage_session_memory_deletion, session_id)

    def _stage_ltm_deletion(self, ltm_id: str):
        return self._run_async_delegate(self._stage_ltm_deletion, ltm_id)
