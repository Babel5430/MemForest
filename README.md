# MemForest: Advanced Hierarchical Memory for Roleplay Chatbots

[![Python Version](https://img.shields.io/pypi/pyversions/MemForest.svg)](xxx)
[![PyPI version](https://badge.fury.io/py/MemForest.svg)](yyy)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 1. Introduction

Welcome to **MemForest**, a Python library meticulously crafted to empower **roleplay chatbots** and other advanced conversational AI with a sophisticated, hierarchical memory system. Drawing inspiration from cognitive models of human memory, MemForest enables chatbots to retain and recall information with nuance across individual messages, entire conversation sessions, and over extended periods. This allows for the development of more coherent, deeply personalized, and contextually-aware AI companions that can build evolving relationships and narratives.

MemForest moves beyond simple conversational buffers, providing a robust framework for persistent, structured, and evolving memory, crucial for immersive and believable roleplaying experiences. With recent enhancements, MemForest now features asynchronous operations for improved performance and supports a wider range of vector storage solutions, offering greater flexibility in deployment.

**Key Features:**

* 🧠 **Cognitive Inspiration for Roleplay:** Designed with the needs of roleplay chatbots in mind, facilitating complex character development and memory persistence.
* 🗂️ **Hierarchical Storage:** Organizes memories into Message, Session, and Long-Term Memory (LTM) levels, managed by distinct ranks, allowing for layered recall and summarization.
* 🧱 **Unified MemoryUnit:** Stores all memory types consistently for simplified management and querying.
* ⚡ **Asynchronous Operations:** Core functionalities are built with `asyncio` for non-blocking performance, crucial for responsive chatbots.
* 💾 **Flexible Persistence:**
    * **Primary Storage:** Asynchronous SQLite for robust and efficient storage of all memory unit data.
    * **Vector Storage:** Decoupled `VectorStoreHandler` supporting multiple backends:
        * Milvus / Zilliz Cloud
        * Qdrant
        * ChromaDB
        * sqlite-vec (for local, serverless vector search)
* 🔍 **Dynamic Querying:** Allows searching Short-Term Memory (STM), LTM, or both using vector similarity and rich metadata filtering.
* 🔗 **Context Recall:** Optionally retrieves neighboring messages (context) for any queried memory, reconstructing the local dialogue flow.
* ✍️ **Automatic Summarization:** Condenses message and session content into hierarchical summaries using a provided Language Model (LLM).
* 🗑️ **Automatic Forgetting:** Implements a configurable forgetting mechanism based on access patterns to manage LTM size effectively.
* 🗣️ **Session Management:** Handles conversation sessions, allowing restoration into STM and deletion.
* 🌐 **External Memory Querying:** Supports querying read-only external vector collections.
* 🛠️ **Configurable & Extensible:** Key parameters like storage paths, cache sizes, and summarization/forgetting behavior are configurable. The `EmbeddingHandler` is an interface, allowing you to use the provided ONNX-based handler or plug in your own (e.g., using SentenceTransformers).
* 🔄 **Data Conversion Utilities:** Provides tools to convert between storage formats.

## 2. Background & Motivation

Standard chatbots often suffer from "amnesia," forgetting earlier parts of conversations, previous interactions, or user preferences. This is particularly limiting for **roleplay chatbots**, where continuity, evolving relationships, and remembering shared history are paramount for an immersive experience.

MemForest addresses this by implementing a persistent and structured memory system, inspired by:

* **Cognitive Memory Models:** Mimicking human short-term, long-term, and hierarchical memory organization.
* **Addressing LLM Context Limits:** Selectively retrieving and synthesizing relevant memories to fit within LLM context windows.
* **Retrieval-Augmented Generation (RAG):** Treating the chatbot's own past experiences as a dynamic, internal knowledge base.
* **LLMs as Agents:** Providing a dedicated, robust memory component for LLM-based agents designed for long-running, interactive scenarios like roleplaying.

## 3. Core Concepts & Design

### 3.1. Hierarchical Storage & `MemoryUnit`

MemForest organizes memory across three conceptual levels using a `rank` attribute:

* **Message Level (rank = 0):** Individual utterances or events.
* **Session Level (rank = 1):** Coherent conversation sessions, often summarized from rank-0 units.
* **Long-Term Memory Level (rank = 2):** Broader memory context, summarizing multiple sessions.

The core data structure, `MemoryUnit`, holds content, timestamps, source, metadata, hierarchical links (`parent_id`, `children_ids`), sequential links (`pre_id`, `next_id`), and importance tracking attributes.

### 3.2. Key Components

* **`MemorySystem` (Synchronous Wrapper):** Provides a simple, synchronous API for interacting with MemForest, managing an internal async event loop. Ideal for straightforward integration.
* **`AsyncMemorySystem` (Asynchronous Core):** The new, high-performance asynchronous core of the library. Use this directly if your application is already `asyncio`-based.
* **`AsyncSQLiteHandler`:** Manages all interactions with the SQLite database for primary data storage, operating asynchronously.
* **`VectorStoreHandler`:** A dedicated handler for vector database operations, supporting Milvus, Qdrant, ChromaDB, and sqlite-vec. It's initialized based on your chosen configuration.
* **`EmbeddingHandler` (Interface):**
    * MemForest expects an embedding handler instance with a `get_embedding(text_or_list_of_text)` method and a `dimension` attribute.
    * The library includes an ONNX-based `EmbeddingHandler` in `MemForest.utils.EmbeddingHandler`.
    * For ease of use and in examples, you can also create your own handler, for instance, using `sentence-transformers`.

### 3.3. Persistence Strategy

* **Primary Data Store (SQLite):** All `MemoryUnit`, `SessionMemory`, and `LongTermMemory` metadata and content are stored in a local SQLite database, managed asynchronously by `AsyncSQLiteHandler`. This ensures data integrity and allows for rich, structured queries.
* **Vector Embeddings Store:** Embeddings for memories are stored in a configured vector database, managed by `VectorStoreHandler`. This enables efficient semantic search. You can configure your preferred backend:
    * **Milvus/Zilliz Cloud:** For distributed, scalable vector search.
    * **Qdrant:** A fast and scalable vector database.
    * **ChromaDB:** An open-source embedding database.
    * **sqlite-vec:** A SQLite extension for local, serverless vector search, ideal for simpler deployments.

The `MemorySystem` (or `AsyncMemorySystem`) coordinates writes to both stores.

### 3.4. Algorithms

* **Summarization (`summarizing.py`):** Recursively groups and condenses memories using an LLM to create hierarchical summaries, from messages to session summaries, and session summaries to LTM summaries.
* **Forgetting (`forgetting.py`):** Prunes less important leaf-node memories from LTM based on visit count, last visit time, and creation time when storage limits are approached.

## 4. Installation (To be improved)

```bash
pip install MemForest
````

Or, clone the repository and install locally:

```bash
git clone [https://github.com/Babel5430/MemForest.git](https://github.com/Babel5430/MemForest.git)
cd MemForest
pip install .
```

**Dependencies:**

MemForest relies on several key libraries. Core dependencies include `numpy`, `langchain-core`, and `aiosqlite`. For its own ONNX-based embedding utility, it uses `tokenizers` and `onnxruntime`.

Depending on the vector store you choose, you'll need the respective client:

  * `pymilvus` for Milvus
  * `qdrant-client` for Qdrant
  * `chromadb` for ChromaDB
  * `sqlite-vec` for the SQLite-based vector search

The examples often use `sentence-transformers` (and its dependency `torch`) to demonstrate how to create and use an embedding handler. These are included in the default installation for convenience.

See `setup.py` and `requirements.txt` for a full list of dependencies and their versions.

## 5\. Usage Example

This example demonstrates basic usage with the synchronous `MemorySystem` wrapper.

```python
import datetime
from MemForest import MemorySystem # Main synchronous interface
from MemForest.memory import MemoryUnit # If you need to inspect units directly

# For the example, we'll use SentenceTransformers to create an embedding handler.
# MemForest is flexible; you can use its built-in ONNX handler or any other.
from sentence_transformers import SentenceTransformer

# A simple wrapper class for SentenceTransformers to match EmbeddingHandler interface
class MyEmbeddingHandler:
    def __init__(self, model_name_or_path, device='cpu'):
        self.model = SentenceTransformer(model_name_or_path, device=device)
        self.dimension = self.model.get_sentence_embedding_dimension()

    def get_embedding(self, text: str | list[str]) -> list[float] | list[list[float]]:
        embeddings = self.model.encode(text, convert_to_numpy=True)
        if isinstance(text, str):
            return embeddings.tolist()
        return embeddings.tolist()

# --- LLM Configuration (Optional, for summarization) ---
# from langchain_openai import ChatOpenAI 
# llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3, api_key="YOUR_API_KEY")
llm = None # Set to your actual LLM instance if using summarization

# --- Embedding Handler Configuration ---
# You can use a pre-trained model from sentence-transformers or your own.
# Ensure the dimension matches what you configure in vector_store_config.
embedding_handler = MyEmbeddingHandler('all-MiniLM-L6-v2', device='cpu') # 384 dimensions

# --- Vector Store Configuration ---
# Choose and configure your desired vector store.
# Ensure 'embedding_dim' matches your embedding_handler.dimension.

# Example: Qdrant (in-memory)
vector_store_config = {
    "type": "qdrant",
    "location": ":memory:", 
    "embedding_dim": embedding_handler.dimension 
}

# # Example: ChromaDB (persistent)
# vector_store_config = {
#     "type": "chroma",
#     "path": "./memforest_chroma_db",
#     "embedding_dim": embedding_handler.dimension
# }

# # Example: Milvus
# vector_store_config = {
#     "type": "milvus",
#     "host": "localhost", 
#     "port": "19530",
#     "embedding_dim": embedding_handler.dimension
# }

# # Example: sqlite-vec (uses the same SQLite DB file)
# vector_store_config = {
#     "type": "sqlite-vec",
#     # "base_path" for SQLite can be specified if different from MemorySystem's default
#     "embedding_dim": embedding_handler.dimension
# }


# --- Initialization ---
chatbot_id = "roleplay_ai_cassandra"
ltm_id = "main_story_arc" # Long-Term Memory ID for this character/story

memory = MemorySystem(
    chatbot_id=chatbot_id,
    ltm_id=ltm_id,
    embedding_handler=embedding_handler,
    llm=llm, # Pass LLM for summarization
    vector_store_config=vector_store_config,
    # base_path="./my_chatbot_memories", # Optional: custom path for SQLite DB
    max_context_length=10,
    max_vector_entities=10000, # Max entities in vector store before trying to forget
    forget_percentage=0.1 
)
print(f"MemForest initialized for {chatbot_id}. Current Session ID: {memory.get_current_sesssion_id()}")

# --- Adding Memories (Simulating a conversation) ---
print("\n--- Simulating Conversation ---")
t1 = datetime.datetime.now() - datetime.timedelta(minutes=10)
msg1 = memory.add_memory("Halt! Who goes there into the Whispering Woods?", source="cassandra_npc", creation_time=t1, metadata={"action": "speak", "emotion": "suspicious"})
print(f"Added: {msg1.source} - '{msg1.content}'")

t2 = t1 + datetime.timedelta(seconds=30)
msg2 = memory.add_memory("Just a humble traveler, hoping to pass safely.", source="player", creation_time=t2, metadata={"action": "speak", "intention": "peaceful"})
print(f"Added: {msg2.source} - '{msg2.content}'")

t3 = t2 + datetime.timedelta(minutes=1)
msg3 = memory.add_memory("The woods are dangerous. State your purpose, traveler!", source="cassandra_npc", creation_time=t3, metadata={"action": "speak", "emotion": "demanding"})
print(f"Added: {msg3.source} - '{msg3.content}'")

# --- Getting Current Context ---
current_context = memory.get_context(length=5)
print(f"\n--- Current Context (last {len(current_context)} messages) ---")
for unit in current_context:
    print(f"  [{unit.creation_time.isoformat()}] {unit.source}: {unit.content} (Action: {unit.metadata.get('action', 'N/A')})")

# --- Querying Memory (Semantic Search) ---
query_text = "traveler's purpose in the woods"
print(f"\n--- Querying for: '{query_text}' ---")
# Get embedding for the query text
query_embedding = embedding_handler.get_embedding(query_text)

# Perform a hybrid query (STM then LTM), recalling context
results = memory.query(
    query_vector=query_embedding, # Pass the list directly
    k_limit=2,          # Request top 2 relevant memory chains
    recall_context=True, # Fetch neighboring messages
    add_ltm_to_stm=True  # Cache LTM results in STM
)

print("\n--- Query Results (with context recall) ---")
for i, chain in enumerate(results):
    print(f"Found Chain {i+1}:")
    for unit in chain:
        print(f"  -> {unit.source} (ID: ...{unit.id[-6:]}): {unit.content}")

# --- STM (Short-Term Memory) Management ---
print("\n--- STM Operations ---")
if not memory.if_stm_enabled(): # Check if STM is enabled
    memory.enable_stm(capacity=50, restore_session_id="LATEST") # Enable STM, restore from latest session
    print("STM enabled and restored from latest session (if any).")

# Context messages are automatically moved to STM when context queue is full or flushed.
# Manually flush context (usually happens automatically)
memory.flush_context() 
print("Context flushed to STM/LTM.")

# Query STM only
query_stm_text = "who is in the woods"
print(f"\n--- Querying STM only for: '{query_stm_text}' ---")
query_stm_embedding = embedding_handler.get_embedding(query_stm_text)
stm_results = memory.query(
    query_vector=query_stm_embedding,
    k_limit=3,
    short_term_only=True # Query only the STM cache
)
print("\n--- STM-Only Query Results ---")
for i, chain in enumerate(stm_results):
    if chain: # STM results are single units unless context is recalled from STM
        unit = chain[0]
        print(f"  Hit {i+1} (STM): {unit.source} (...{unit.id[-6:]}): {unit.content}")

# --- Session Management ---
# current_session_id = memory.get_current_sesssion_id()
# print(f"\n--- Current Session ID: {current_session_id} ---")
# To start a new session (a new UUID will be generated):
# memory.start_session() # Clears context and starts a new session
# print(f"New Session ID: {memory.get_current_sesssion_id()}")

# --- Summarization (if LLM is configured) ---
if llm:
    print("\n--- Manual Summarization (LLM required) ---")
    current_session_id_for_summary = memory.get_current_sesssion_id()
    if current_session_id_for_summary:
        print(f"Attempting to summarize session: {current_session_id_for_summary}")
        # memory.summarize_session(session_id=current_session_id_for_summary, role="system")
        # print("Session summarization attempt complete.")
        # Summarization happens asynchronously in the background if using MemorySystem
        # For direct async usage, you would await it.
    
    # print(f"Attempting to summarize LTM: {ltm_id}")
    # memory.summarize_long_term_memory(role="system")
    # print("LTM summarization attempt complete.")
else:
    print("\nSkipping Summarization examples (LLM not configured).")

# --- Closing MemForest ---
print("\n--- Closing MemForest ---")
# Close the memory system gracefully (flushes final changes, closes connections)
# auto_summarize=True would trigger session/LTM summary if LLM is available
memory.close(auto_summarize=False) 
print("MemorySystem closed.")
```

## 6\. Future Work & Roadmap

MemForest is an actively evolving project. Here's a glimpse into our future plans:

  * 🌟 **Further Modularization and Abstraction:**
      * Refine `PersistenceHandler` interfaces for even cleaner separation of concerns.
      * Abstract core logic further to allow easier integration of custom components.
  * 🚀 **Enhanced High-Concurrency Processing:**
      * Optimize asynchronous task management for handling many concurrent users/sessions.
      * Explore strategies for distributed memory systems.
  * 💻 **Lightweight Version for Local Deployment:**
      * Develop a "MemForest-Lite" variant with minimal dependencies (e.g., SQLite-only, no separate vector DB required for basic use) for easy local chatbot deployment.
  * ☁️ **Convenient Online Deployment:**
      * Provide clear guides and tools (e.g., Dockerfiles, serverless function examples) for deploying MemForest-powered chatbots online.
  * 📤 **Streamlined Memory Import/Export:**
      * Develop robust and user-friendly mechanisms for importing/exporting entire memory archives (e.g., for character sharing, backups, or migration between backends).
  * 🧠 **Optimized Memory Usage Logic:**
      * Advanced RAG strategies: Explore more sophisticated techniques for selecting and synthesizing memories for LLM context, potentially including graph-based traversal or dynamic relevance scoring.
      * Smarter forgetting: Implement more nuanced forgetting algorithms, perhaps considering narrative importance or user-defined retention policies.
  * ⚡ **Core Component Refactoring (Performance Focus):**
      * Identify performance-critical sections through profiling and consider reimplementing them in high-performance languages (e.g., Rust, Go via Python bindings) if significant gains are achievable beyond Python's async capabilities.
  * 📊 **Improved Telemetry and Debugging:**
      * Integrate optional telemetry for monitoring memory system performance and health.
      * Provide better tools and MQL (Memory Query Language) for debugging and inspecting memory contents.
  * 🤝 **Community and Integrations:**
      * Explore deeper integrations with popular chatbot frameworks and LLM agent libraries.

We welcome contributions and feedback from the community\!

## 7\. License

MemForest is licensed under the [MIT License](https://www.google.com/search?q=LICENSE).
