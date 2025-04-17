# MemForest: Hierarchical Memory System for Chatbots

## 1. Introduction

MemForest is a Python library designed to equip chatbots with a sophisticated, hierarchical memory system, drawing inspiration from cognitive models of human memory. It allows chatbots to retain information across individual messages, entire conversation sessions, and over the long term, enabling more coherent, personalized, and context-aware interactions. By moving beyond simple conversational buffers, MemForest provides a framework for persistent, evolving memory.

**Key Features:**

* **Hierarchical Storage:** Organizes memories into Message, Session, and Long-Term Memory (LTM) levels, managed by distinct ranks.
* **Unified MemoryUnit:** Stores all memory types consistently for simplified management and querying.
* **Short-Term Memory (STM):** Manages recent memories using LRU and FIFO caches for quick access during conversations.
* **Long-Term Memory (LTM):** Persists all historical conversations and their summaries.
* **Flexible Persistence:** Supports SQLite (recommended) and partitioned JSON files for primary storage, synchronized with Milvus for efficient vector-based retrieval.
* **Dynamic Querying:** Allows searching STM, LTM, or both using vector similarity and metadata filtering.
* **Context Recall:** Optionally retrieves neighboring messages (context) for any queried memory, reconstructing the local dialogue flow.
* **Automatic Summarization:** Condenses message and session content into hierarchical summaries using a provided Language Model (LLM).
* **Automatic Forgetting:** Implements a configurable forgetting mechanism based on access frequency, interval, and time to manage LTM size.
* **Session Management:** Handles conversation sessions, allowing restoration (into STM) and deletion.
* **External Memory:** Supports querying read-only external Milvus collections.
* **Configurable:** Key parameters like storage paths, cache sizes, and summarization/forgetting behavior are configurable during initialization.
* **Data Conversion Utilities:** Provides tools to convert between storage formats (e.g., SQLite to Milvus).

## 2. Background & Motivation

Standard chatbots often exhibit limited memory capabilities, typically restricted to a fixed-size conversational window. This leads to "amnesia," where the chatbot forgets earlier parts of the current conversation, previous interactions, or user preferences, hindering the development of deep, long-term relationships and complex task completion.

MemForest addresses this by implementing a more persistent and structured memory system, inspired by several concepts:

* **Cognitive Memory Models:** Human memory isn't monolithic; it involves different systems like short-term working memory and long-term storage, often with hierarchical organization (Tulving E. Elements of episodic memory[J]. 1983.; Anderson J R. The architecture of cognition[M]. Psychology Press, 2013.). MemForest mimics this with its STM/LTM distinction and hierarchical MemoryUnit structure. The summarization process creates abstractions similar to how humans form generalized memories from specific events.
* **Addressing LLM Context Limits:** Large Language Models (LLMs) have finite context windows. Simply feeding the entire history is often impossible. Effective memory systems select and synthesize relevant information to fit within this window.
* **Retrieval-Augmented Generation (RAG):** Modern AI systems often retrieve relevant information from external knowledge bases to enhance generation quality (Lewis P, Perez E, Piktus A, et al. Retrieval-augmented generation for knowledge-intensive nlp tasks[J].). MemForest adapts this by treating the chatbot's own past experiences (stored in LTM) as the knowledge base, retrieving relevant memories semantically.
* **LLMs as Agents:** Recent research explores using LLMs as the core of autonomous agents that interact with environments and tools over extended periods. Persistent, structured memory is crucial for such agents to learn, adapt, and maintain context (e.g., Packer C, Fang V, Patil S G, et al. MemGPT: Towards LLMs as Operating Systems[J]. 2023.). MemForest provides a specialized memory component suitable for such architectures.

By combining hierarchical storage, summarization, vector retrieval, and explicit short/long-term management, MemForest aims to provide a robust and scalable memory foundation for more capable conversational AI.

## 3. Basic Concepts, Design, and Algorithms

### 3.1. Hierarchical Storage & MemoryUnit

MemForest organizes memory across three conceptual levels, distinguished by the `rank` attribute of a `MemoryUnit`:

* **Message Level (rank = 0):** Represents individual utterances or events in a conversation (e.g., a user message, an AI response, a system notification). These are the raw inputs to the memory system.
* **Session Level (rank = 1):** Represents a coherent conversation session. A session consists of multiple rank-0 `MemoryUnit`s. MemForest can automatically summarize all rank-0 units within a session into a single rank-1 `MemoryUnit`. This summary unit's `id` is identical to the `SessionMemory` object's ID, acting as the canonical representation of that session in the memory hierarchy.
* **Long-Term Memory Level (rank = 2):** Represents the overall memory context, typically summarizing multiple sessions. Rank-1 session summary units can be further summarized into one or more rank-2 `MemoryUnit`s. The root summary unit for a specific LTM instance often shares its `id` with the `LongTermMemory` object ID.
* Note: MemForest sets a "message limit" during summarization. If the number of tokens or messages to be summarized exceeds the designated limit, it will trigger group summarization and recursive summarization, generating a series of intermediate units.

This hierarchy allows for efficient querying at different granularities and enables progressive information compression via summarization.

The core data structure is the `MemoryUnit`:

* `id` (str): Unique identifier (UUID).
* `content` (str): The memory's text content (message, summary, etc.).
* `source` (Optional[str]): Originator (e.g., "user", "ai", "system", "ai-summary"). Defaults to "ai".
* `creation_time` (Optional[datetime]): Start timestamp of the memory event.
* `end_time` (Optional[datetime]): End timestamp of the memory event.
* `metadata` (Optional[Dict[str, Any]]): User-defined dictionary for extra information (e.g., `{"action": "speak"}`, message IDs, sentiment).
* `rank` (int): Hierarchy level (0, 1, or 2).
* `group_id` (Optional[str]): The ID of the Session or LTM this unit primarily belongs to. Crucial for grouping messages and associating summaries with their scope. For rank 0 units, it's the session ID. For rank 1 units (session summaries), it's also the session ID (which is the same as the unit's own ID). For rank 2 units (LTM summaries), it's the LTM ID.
* **Tree Links (Summarization Hierarchy):**
    * `parent_id` (Optional[str]): ID of the summary unit directly above this unit in the hierarchy (the summary this unit contributed to).
    * `children_ids` (List[str]): IDs of the units directly below this unit (the units that were summarized to create this one).
* **Sequence Links (Dialogue Flow):**
    * `pre_id` (Optional[str]): ID of the previous rank-0 `MemoryUnit` within the same session. Forms a chronological and causal chain.
    * `next_id` (Optional[str]): ID of the next rank-0 `MemoryUnit` within the same session.
* **Importance Tracking (For Forgetting):**
    * `visit_count` (int): How many times this unit has been accessed (e.g., retrieved in a query).
    * `last_visit` (int): The interaction round number when this unit was last accessed.
    * `never_delete` (bool): Flag to prevent automatic forgetting.

### 3.2. Core Components

* **SessionMemory:** Metadata object representing a session. Holds the `id` (same as the rank-1 summary `MemoryUnit` ID) and a list (`memory_unit_ids`) of all associated unit IDs within that session and The IDs of all units recursively summarized from these units (rank 0 messages + rank 1 summary).
* **LongTermMemory:** Metadata object for an LTM instance. Holds the `ltm_id` (same as the root rank-2 summary ID), associated `session_ids`, and `summary_unit_ids` (IDs of rank-2 summaries belonging to this LTM).
* **MemorySystem:** The central orchestrator class. Manages:
    * **Context Queue:** Short FIFO queue for immediate incoming messages.
    * **Short-Term Memory (STM):** Limited-capacity LRU cache for recently accessed/evicted units and their embeddings. Enables fast semantic search over recent message.
    * **LTM Management:** Handles the interaction with the storage system and is capable of storing and retrieving all non STM units. Support field storage (SQLite/JSON) and vector storage (Milvus)
    * **Persistence Handlers:** Modules (`sqlite_handler`, `json_handler`, `vector_store_handler`) abstracting storage interactions. (Future: Could be unified under an interface).
    * **Caching:** In-memory dictionaries (`memory_units_cache`, `session_memories_cache`) to reduce disk I/O. Changes are staged and flushed.
    * **Summarization & Forgetting Logic:** Coordinates calls to the summarizing and forgetting modules.

### 3.3. Persistence

* **Primary Storage (SQLite/JSON):** Stores the full `MemoryUnit`, `SessionMemory`, and `LongTermMemory` data (excluding embeddings).
    * **SQLite (Recommended):** Stores all data for a chatbot in a single `.db` file. Offers transactional integrity (atomicity) and efficient querying via SQL, especially suitable for handling many small sessions and scaling.
    * **JSON:** Stores LTMs, Sessions, and Units in separate JSON files. Can become inefficient (as the number of messages increases) and lacks transactional guarantees across files. Recommended only for smaller datasets or specific interoperability needs.
* **Vector Store (Milvus):** Stores `MemoryUnit` embeddings alongside key metadata. Enables fast semantic similarity search via `vector_store_handler`. Can be configured via `milvus_config`.
* **Synchronization:** The `MemorySystem` ensures data is written to both the primary store and Milvus during the flushing process.

### 3.4. Algorithms

**Algorithms:**

1.  **Summarization (`summarizing.py`)**:
    * **Goal:** Reduce information quantity while preserving key details, creating hierarchical representations.
    * **Process:** Operates recursively on levels (messages -> session summaries -> LTM summaries).
    * **Grouping (`Group` Concept):** Units at the current level are greedily grouped chronologically based on maximum count (`max_group_size`) and token limits (`max_token_count`) to control LLM input size and respect natural conversation breaks.
    * **LLM Call:** Each group is summarized using the configured LLM (`summarize_memory` function). Context (previous summary at the same level or initial history) can be provided.
    * **Hierarchy Creation:** A new `MemoryUnit` (with incremented `rank`) is created for each summary. Its `children_ids` point to the units in the summarized group, and the children's `parent_id` is updated to point to the new summary unit.
    * **Recursion:** The process repeats on the newly created summary units until only one root unit remains for the session or LTM.

2.  **Forgetting (`forgetting.py`)**:
    * **Goal:** Prune less important information from LTM when storage limits (e.g., `max_milvus_entities`) are reached.
    * **Trigger:** Checked periodically (e.g., after flushing changes) if LTM size exceeds a threshold.
    * **Mechanism:**
        * **Visit Tracking:** Each `MemoryUnit` stores `visit_count` and `last_visit` (round number). Accessing a unit (e.g., via query) updates its counters and resets the `last_visit` interval for itself and all ancestors up to the root to 0.
        * **Candidate Selection:** Identifies eligible leaf nodes (no `children_ids`) in the LTM tree that are *not* marked `never_delete`.
        * **Scoring:** Sorts eligible leaves by importance (least important first) based on: 1. `last_visit` (older is less important), 2. `visit_count` (lower is less important), 3. `creation_time` (older is less important).
        * **Deletion:** Deletes a calculated percentage (`forget_percentage`) of the least important leaves.
        * **Tree Maintenance:** When a unit is deleted, its `parent_id` reference in its former parent's `children_ids` list is removed. If a node loses all its children, it becomes a leaf node and eligible for deletion in future cycles.

## 4. Installation and Usage

### 4.1. Installation

Clone the repository:

```bash
git clone https://github.com/Babel5430/MemForest.git
cd MemForest
```

### 4.2. Dependencies

MemForest requires the following Python libraries:

* `numpy>=1.21.0`: For numerical operations, especially embeddings.
* `langchain-core>=0.1.0`: For LLM and message type abstractions (e.g., `BaseMessage`). You might need additional `langchain` packages depending on the specific LLM you use (e.g., `langchain-openai`).
* `pymilvus>=2.3.0,<2.4.0`: For interacting with the Milvus vector database.
* `sentence-transformers>=2.2.0`: For generating text embeddings (example uses `moka-ai/m3e-small`).
* `torch>=1.9.0`: Required by `sentence-transformers`. Installation might vary depending on your system (CPU/GPU). Check PyTorch official instructions.

You can typically install these using:

```bash
pip install numpy "langchain-core>=0.1.0" "pymilvus>=2.3.0,<2.4.0" "sentence-transformers>=2.2.0" torch
# Add other specific langchain packages if needed, e.g. pip install langchain-openai
```

Make sure you have a running Milvus instance accessible.

### 4.3. Usage Examples

```python
import os
import datetime
from MemForest.manager.memory_system import MemorySystem
from MemForest.utils.embedding_handler import EmbeddingHandler
from sentence_transformers import SentenceTransformer
# replace with your api provider
# from langchain_openai import ChatOpenAI
# llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

llm = None # Set to your actual LLM instance for summarization
# --- Initialization ---

chatbot_id = "memforest_demo_bot"

ltm_id = "primary_ltm"
# Initialize embedding handler (uses moka-ai/m3e-small by default)

embedding_model_name = 'moka-ai/m3e-small'
embedding_model = SentenceTransformer(embedding_model_name, device='cpu') # Use 'cuda' if GPU available
embedding_handler = EmbeddingHandler(model=embedding_model, device='cpu') # Match device
# Milvus configuration (Ensure Milvus is running!)

milvus_config = {
    "host": "localhost", # Or your Milvus host IP/domain
    "port": "19530", # Default Milvus port
}
# Initialize Memory System
# Storage path will default to './memory_storage/<chatbot_id>' in the current dir

memory = MemorySystem(
    chatbot_id=chatbot_id,
    ltm_id=ltm_id,
    embedding_handler=embedding_handler,
    llm=llm, # Pass LLM for summarization features
    milvus_config=milvus_config,
    persistence_mode='sqlite', # 'sqlite' or 'json', Recommended: sqlite
    max_context_length=10, # Example: Keep last 10 messages in immediate context
    max_milvus_entities=10000, # Example: Trigger forgetting above 10k entities
    forget_percentage=0.1 # Example: Forget 10% of eligible units when triggered
)

print(f"Current Session ID: {memory._current_session_id}") # Accessing private for demo
# --- Adding Memory ---

print("\n--- Adding Memory ---")

t1 = datetime.datetime.now() - datetime.timedelta(minutes=5)
unit1 = memory.add_memory("What was the project code-named?", source="user", creation_time=t1)
print(f"Added: User - '{unit1.content}'")

t2 = t1 + datetime.timedelta(seconds=30)
unit2 = memory.add_memory("The project was code-named 'MemForest'.", source="ai", creation_time=t2)
print(f"Added: AI - '{unit2.content}'")

t3 = t2 + datetime.timedelta(minutes=2)
unit3 = memory.add_memory("What is the status?", source="user", creation_time=t3)
print(f"Added: User - '{unit3.content}'")

t4 = t3 + datetime.timedelta(seconds=30)
unit4 = memory.add_memory("Project 'MemForest' is currently on hold.", source="ai", creation_time=t4)
print(f"Added: AI - '{unit4.content}'")
# --- Getting Context ---

# Gets the last 'n' messages currently held before being pushed to STM/LTM

current_context = memory.get_context(length=5)
print(f"Current Context ({len(current_context)} messages):")
for unit in current_context:
    print(f" - [{unit.creation_time.isoformat()}] {unit.source}: {unit.content}")
# --- Querying Memory ---

query_text = "Tell me about project Phoenix"
print(f"Querying for: '{query_text}'")
query_embedding = embedding_handler.get_embedding(query_text).tolist()

# Example 1: Hybrid Query (Default: STM first, then LTM) with context recall
results_hybrid = memory.query(
    query_vector=query_embedding,
    k_limit=2, # Request top 2 relevant chains/hits
    recall_context=True, # Fetch neighbors (pre_id, next_id)
    add_ltm_to_stm=True # Add results found in LTM to STM cache
)

print("\nHybrid Query Results (Recall Context=True):")
for i, chain in enumerate(results_hybrid):
    print(f" Chain {i+1}:")
    for unit in chain:
        print(f" -> {unit.source} ({unit.id[-6:]}): {unit.content}") # Show last 6 chars of ID

# Example 2: Long-Term Memory only query, no context recall, specific source
results_ltm_only = memory.query(
    query_vector=query_embedding,
    filters=[("source", "==", "ai")], # Only find AI messages
    k_limit=3,
    long_term_only=True, # Skip STM
    recall_context=False # Just return the hit unit
)

print("\nLTM-Only Query Results (AI source, Recall Context=False):")
for i, chain in enumerate(results_ltm_only): # Each "chain" has only 1 unit here
    if chain:
        unit = chain[0]
        print(f" Hit {i+1}: {unit.source} ({unit.id[-6:]}): {unit.content}")
# --- Short-Term Memory (STM) ---

# Enable STM with capacity 50, restore from the latest session if available
print("Enabling STM (Capacity=50, Restore=LATEST)...")
memory.enable_stm(capacity=50, restore_session_id="LATEST")

# Force flush context to STM and LTM(normally happens when context queue is full)
print("Flushing context to STM and LTM...")
memory._flush_context() # Private method used for demo

# Query STM only
query_text_stm = "project status"
print(f"Querying STM only for: '{query_text_stm}'")
query_embedding_stm = embedding_handler.get_embedding(query_text_stm).tolist()
results_stm_only = memory.query(
    query_vector=query_embedding_stm,
    k_limit=5,
    short_term_only=True # Query only the STM cache
)

print("\nSTM-Only Query Results:")
for i, chain in enumerate(results_stm_only):
    if chain:
        unit = chain[0]
        print(f" Hit {i+1} (STM): {unit.source} ({unit.id[-6:]}): {unit.content}")

# Example: Restore STM from a specific session (if you know its ID)
# session_to_restore = "some-known-session-id"
# print(f"Disabling and re-enabling STM, restoring from session: {session_to_restore}")
# memory.disable_stm()
# memory.enable_stm(capacity=50, restore_session_id=session_to_restore) # Use specific ID

print("Disabling STM...")
memory.disable_stm()
# --- Session Management ---

current_session_id = memory._current_session_id
print(f"Current Session ID: {current_session_id}")

# Starting new sessions typically happens implicitly or via a custom method
# Example: Force start a new session (if a public method existed)
# memory.start_new_session()
# print(f"New Session ID: {memory._current_session_id}")

# Example: Removing a session (Use with extreme caution!)
# Be very sure you want to delete a session and all its messages.
# session_id_to_remove = current_session_id # Example: remove the current one
# print(f"Attempting to remove session: {session_id_to_remove}")
# try:
#     memory.remove_session(session_id_to_remove)
#     print(f"Session {session_id_to_remove} removed.")
# except Exception as e:
#     print(f"Error removing session: {e}")

# --- Manual Summarization / Forgetting ---
# Note: These require an LLM to be configured in MemorySystem

print("\n--- Manual Triggers (Require LLM) ---")
if llm:
    # Example: Summarize a specific session
    # print(f"Attempting to summarize session: {current_session_id}")
    # memory.summarize_session(session_id=current_session_id)

    # Example: Summarize the entire LTM
    # print(f"Attempting to summarize LTM: {ltm_id}")
    # memory.summarize_long_term_memory()
    pass # Placeholder if LLM not set
else:
    print("Skipping Summarization examples (LLM not configured).")

# Auto-forgetting happens during _flush_cache if Milvus count exceeds threshold
# To manually trigger the check (primarily for testing):
# print("Manually checking forgetting condition...")
# memory._check_and_forget_memory() # Private method, normally automatic

# --- Data Conversion ---
# print("\n--- Data Conversion Utilities ---")
# Ensure you have the source data before converting
# print("Example: Convert SQLite to JSON (if using SQLite)")
# if memory.persistence_mode == 'sqlite':
#     memory.convert_sql_to_json(output_dir="./memory_storage_json_copy")
#     print("Conversion to JSON attempted (check output dir).")

# print("Example: Convert JSON to SQLite (if using JSON)")
# if memory.persistence_mode == 'json':
#     memory.convert_json_to_sqlite()
#     print("Conversion to SQLite attempted (DB file should be updated).")

# print("Example: Convert primary store to Milvus (generates embeddings)")
# Needs primary store (SQLite or JSON) populated
# if memory.persistence_mode == 'sqlite':
#     memory.convert_sql_to_milvus(embedding_history_length=1)
# elif memory.persistence_mode == 'json':
#     memory.convert_json_to_milvus(embedding_history_length=1)
# print("Conversion to Milvus attempted.")

# --- Saving & Closing ---

# Changes are flushed periodically based on SAVING_INTERVAL/VISIT_UPDATE_INTERVAL
# Force a flush manually if needed (usually not necessary):
# print("Forcing cache flush...")
# memory._flush_cache(force=True)

# Close the memory system gracefully (flushes final changes, closes connections)
memory.close(auto_summarize=False) # Auto-summarize if set auto_summarize=True and LLM configured
print("MemorySystem closed.")
```
## 5. TODO List / Future Work

This list includes planned improvements and areas for future development:

**Storage & Scalability:**

* **Database Backend:** Officially recommend and optimize for SQLite. Consider adding support for PostgreSQL + pgvector as a more scalable alternative, potentially replacing Milvus for simpler deployments.
* **SQLite Optimizations:** Continuously review query performance, refine indexing strategies (composite indices), ensure WAL mode is default.
* **Caching Strategy:** Implement strict size limits (memory/item count) and LRU eviction for in-memory caches (`memory_units_cache`, `session_memories_cache`). Ensure lazy loading from the database is the default behavior.
* **Asynchronous Processing:**
    * **Full Async IO:** Convert all database (SQLite/Postgres), Milvus, and potentially LLM/Embedding operations to use `asyncio` (e.g., `aiosqlite`, `asyncio.to_thread`) to prevent blocking. Refactor core `MemorySystem` methods (`add_memory`, `query`, `flush`, etc.) to be `async def`.
    * **Background Tasks:** Use `asyncio.create_task` for background operations like flushing, summarization, and forgetting.
* **Consistency & Atomicity:**
    * **Cross-Store Transactions:** Implement a robust strategy (like the Outbox pattern or careful sequencing with retry logic) to ensure better consistency between the primary store (SQLite) and the vector store (Milvus), especially for delete operations.
    * **Idempotency:** Review critical operations to ensure they are idempotent (safe to retry).

**Code Structure & Refinement:**

* **Persistence Interface:** Refactor persistence logic behind an abstract `PersistenceHandler` interface to decouple `MemorySystem` from specific backend implementations (SQLite, JSON).
* **Simplify Loading:** Remove code that loads large amounts of data only to filter/process it in Python; push these operations to the database layer via targeted queries.
* **Deprecate JSON:** Consider removing the JSON persistence backend entirely to simplify the codebase and focus on the more scalable database approach.

**Performance:**

* **Database Query Optimization:** Prioritize optimizing SQL queries over Python loops for data manipulation (sorting, filtering, aggregation) where possible (e.g., in forgetting mechanism).
* **Profiling:** Integrate profiling tools (`cProfile`, `line_profiler`) to systematically identify performance bottlenecks under load after major optimizations (like async IO).
* **Compiled Extensions (If Needed):** If profiling reveals persistent CPU bottlenecks in specific Python algorithms (e.g., complex graph traversal, custom scoring) that cannot be moved to the database, consider implementing those critical sections in Cython/Rust/Go.

**Features & Enhancements:**

* **Configuration File:** Allow loading settings (hyperparameters, paths, DB configs) from a `config.yaml` or similar file.
* **Enhanced Forgetting:** Explore more sophisticated strategies (relevance decay based on query similarity, graph centrality).
* **Testing:** Develop a comprehensive test suite (unit and integration tests).
* **Error Handling:** Implement more granular error handling and logging.
* **Query Enhancements:** Add features like time-decaying relevance scores.
* **Embedding Dimension:** Derive `EMBEDDING_DIMENSION` automatically from the embedding model.
* **Public Session API:** Add clear methods like `start_new_session()` or `switch_session(session_id)`.


```
