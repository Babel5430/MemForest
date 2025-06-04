# MemForest: 专为角色扮演聊天机器人打造的高级分层记忆系统

[![Python Version](https://img.shields.io/pypi/pyversions/MemForest.svg)](xxx)
[![PyPI version](https://badge.fury.io/py/MemForest.svg)](yyy)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 1. 简介

欢迎来到 **MemForest**，这是一个精心设计的 Python 库，旨在为**角色扮演聊天机器人**及其他高级对话式 AI 配备一个复杂且分层的记忆系统。MemForest 从人类记忆的认知模型中汲取灵感，使聊天机器人能够在单条消息、整个对话会话以及长期范围内细致地保留和回忆信息。这使得开发更连贯、深度个性化且具备上下文感知能力的 AI 伙伴成为可能，它们能够构建不断发展的关系和叙事。

MemForest 超越了简单的对话缓冲区，为持久化、结构化和不断演进的记忆提供了一个强大的框架，这对于沉浸式和可信的角色扮演体验至关重要。通过近期的改进，MemForest 现在具备异步操作以提升性能，并支持更广泛的向量存储解决方案，提供了更大的部署灵活性。

**主要特点：**

* 🧠 **为角色扮演设计的认知启发：** 专为角色扮演聊天机器人的需求而设计，促进复杂的角色发展和记忆持久性。
* 🗂️ **分层存储：** 将记忆组织成消息、会话和长期记忆 (LTM) 三个层级，由不同的等级管理，允许分层回忆和总结。
* 🧱 **统一的 MemoryUnit：** 一致地存储所有记忆类型，简化管理和查询。
* ⚡ **异步操作：** 核心功能采用 `asyncio` 构建，实现非阻塞性能，对响应迅速的聊天机器人至关重要。
* 💾 **灵活的持久化方案：**
    * **主存储：** 异步 SQLite，用于稳健高效地存储所有记忆单元数据。
    * **向量存储：** 解耦的 `VectorStoreHandler`，支持多种后端：
        * Milvus / Zilliz Cloud
        * Qdrant
        * ChromaDB
        * sqlite-vec (用于本地、无服务器的向量搜索)
* 🔍 **动态查询：** 允许使用向量相似性和丰富的元数据过滤搜索短期记忆 (STM)、LTM 或两者。
* 🔗 **上下文回忆：** （可选）为任何查询到的记忆召回其相邻消息（上下文），重建局部对话流程。
* ✍️ **自动总结：** 使用提供的语言模型 (LLM) 将消息和会话内容浓缩成层次化总结。
* 🗑️ **自动遗忘：** 实现基于访问模式的可配置遗忘机制，有效管理 LTM 大小。
* 🗣️ **会话管理：** 处理对话会话，允许恢复到 STM 和删除。
* 🌐 **外部记忆查询：** 支持查询只读的外部向量集合。
* 🛠️ **可配置与可扩展：** 存储路径、缓存大小、总结/遗忘行为等关键参数均可配置。`EmbeddingHandler` 是一个接口，允许您使用提供的基于 ONNX 的处理程序或插入您自己的实现（例如，使用 SentenceTransformers）。
* 🔄 **数据转换工具：** 提供在存储格式之间转换的工具。

## 2. 背景与动机

标准聊天机器人常常表现出“失忆症”，忘记对话的早期部分、之前的互动或用户偏好。这对于**角色扮演聊天机器人**尤其具有限制性，因为在角色扮演中，连续性、不断发展的关系以及记住共享历史对于沉浸式体验至关重要。

MemForest 通过实现一个持久化和结构化的记忆系统来解决此问题，其灵感来源于：

* **认知记忆模型：** 模拟人类的短期、长期和分层记忆组织。
* **解决 LLM 上下文限制：** 选择性地检索和综合相关记忆以适应 LLM 的上下文窗口。
* **检索增强生成 (RAG)：** 将聊天机器人自身的过去经验视为一个动态的内部知识库。
* **LLM 作为智能体：** 为基于 LLM、专为角色扮演等长期互动场景设计的智能体提供专用的、强大的记忆组件。

## 3. 核心概念与设计

### 3.1. 分层存储与 `MemoryUnit`

MemForest 使用 `rank` 属性将记忆组织到三个概念层级：

* **消息层级 (rank = 0)：** 单个话语或事件。
* **会话层级 (rank = 1)：** 连贯的对话会话，通常由 rank-0 单元总结而来。
* **长期记忆层级 (rank = 2)：** 更广泛的记忆上下文，总结多个会话。

核心数据结构 `MemoryUnit` 保存内容、时间戳、来源、元数据、层级链接（`parent_id`, `children_ids`）、序列链接（`pre_id`, `next_id`）以及重要性跟踪属性。

### 3.2. 关键组件

* **`MemorySystem` (同步包装器)：** 提供简单易用的同步 API 与 MemForest 交互，管理内部异步事件循环。非常适合直接集成。
* **`AsyncMemorySystem` (异步核心)：** 全新、高性能的异步核心库。如果您的应用程序已经是基于 `asyncio` 的，可以直接使用它。
* **`AsyncSQLiteHandler`：** 异步管理所有与 SQLite 数据库的交互，用于主数据存储。
* **`VectorStoreHandler`：** 专用的向量数据库操作处理程序，支持 Milvus、Qdrant、ChromaDB 和 sqlite-vec。根据您的配置进行初始化。
* **`EmbeddingHandler` (接口)：**
    * MemForest 期望一个嵌入处理程序实例，该实例具有 `get_embedding(text_or_list_of_text)` 方法和 `dimension` 属性。
    * 库在 `MemForest.utils.EmbeddingHandler` 中包含一个基于 ONNX 的 `EmbeddingHandler`。
    * 为方便使用和示例演示，您也可以创建自己的处理程序，例如使用 `sentence-transformers`。

### 3.3. 持久化策略

* **主数据存储 (SQLite)：** 所有 `MemoryUnit`、`SessionMemory` 和 `LongTermMemory` 的元数据和内容都存储在本地 SQLite 数据库中，由 `AsyncSQLiteHandler` 进行异步管理。这确保了数据完整性并允许丰富的结构化查询。
* **向量嵌入存储：** 记忆的嵌入存储在配置的向量数据库中，由 `VectorStoreHandler` 管理。这使得高效的语义搜索成为可能。您可以配置您偏好的后端：
    * **Milvus/Zilliz Cloud：** 用于分布式、可扩展的向量搜索。
    * **Qdrant：** 一个快速且可扩展的向量数据库。
    * **ChromaDB：** 一个开源的嵌入数据库。
    * **sqlite-vec：** 一个 SQLite 扩展，用于本地、无服务器的向量搜索，非常适合简单的部署。

`MemorySystem` (或 `AsyncMemorySystem`) 协调对这两个存储的写入操作。

### 3.4. 核心算法

* **总结 (`summarizing.py`)：** 使用 LLM 递归地分组和精炼记忆，创建从消息到会话总结，再到 LTM 总结的层级化摘要。
* **遗忘 (`forgetting.py`)：** 当存储达到限制时，根据访问计数、最后访问时间和创建时间，修剪 LTM 中重要性较低的叶节点记忆。

## 4. 安装（待完善）

```bash
pip install MemForest
````

或者，克隆仓库并在本地安装：

```bash
git clone [https://github.com/Babel5430/MemForest.git](https://github.com/Babel5430/MemForest.git)
cd MemForest
pip install .
```

**依赖项：**

MemForest 依赖几个关键库。核心依赖包括 `numpy`、`langchain-core` 和 `aiosqlite`。对于其自带的基于 ONNX 的嵌入工具，它使用 `tokenizers` 和 `onnxruntime`。

根据您选择的向量存储，您将需要相应的客户端：

  * Milvus 需要 `pymilvus`
  * Qdrant 需要 `qdrant-client`
  * ChromaDB 需要 `chromadb`
  * 基于 SQLite 的向量搜索需要 `sqlite-vec`

示例代码通常使用 `sentence-transformers` (及其依赖 `torch`) 来演示如何创建和使用嵌入处理程序。为方便起见，这些已包含在默认安装中。

请参阅 `setup.py` 和 `requirements.txt` 获取完整的依赖项列表及其版本。

## 5\. 使用示例

此示例演示了使用同步 `MemorySystem` 包装器的基本用法。

```python
import datetime
from MemForest import MemorySystem # 主要的同步接口
from MemForest.memory import MemoryUnit # 如果您需要直接检查单元

# 在示例中，我们将使用 SentenceTransformers 创建一个嵌入处理程序。
# MemForest 非常灵活；您可以使用其内置的 ONNX 处理程序或任何其他处理程序。
from sentence_transformers import SentenceTransformer

# 一个简单的 SentenceTransformers 包装类，以匹配 EmbeddingHandler 接口
class MyEmbeddingHandler:
    def __init__(self, model_name_or_path, device='cpu'):
        self.model = SentenceTransformer(model_name_or_path, device=device)
        self.dimension = self.model.get_sentence_embedding_dimension()

    def get_embedding(self, text: str | list[str]) -> list[float] | list[list[float]]:
        embeddings = self.model.encode(text, convert_to_numpy=True)
        if isinstance(text, str):
            return embeddings.tolist()
        return embeddings.tolist()

# --- LLM 配置 (可选, 用于总结功能) ---
# from langchain_openai import ChatOpenAI 
# llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3, api_key="YOUR_API_KEY")
llm = None # 如果使用总结功能，请设置为您的实际 LLM 实例

# --- 嵌入处理程序配置 ---
# 您可以使用 sentence-transformers 的预训练模型或您自己的模型。
# 确保维度与您在 vector_store_config 中配置的维度相匹配。
embedding_handler = MyEmbeddingHandler('all-MiniLM-L6-v2', device='cpu') # 384 维

# --- 向量存储配置 ---
# 选择并配置您期望的向量存储。
# 确保 'embedding_dim' 与您的 embedding_handler.dimension 匹配。

# 示例: Qdrant (内存模式)
vector_store_config = {
    "type": "qdrant",
    "location": ":memory:", 
    "embedding_dim": embedding_handler.dimension 
}

# # 示例: ChromaDB (持久化)
# vector_store_config = {
#     "type": "chroma",
#     "path": "./memforest_chroma_db",
#     "embedding_dim": embedding_handler.dimension
# }

# # 示例: Milvus
# vector_store_config = {
#     "type": "milvus",
#     "host": "localhost", 
#     "port": "19530",
#     "embedding_dim": embedding_handler.dimension
# }

# # 示例: sqlite-vec (使用相同的 SQLite 数据库文件)
# vector_store_config = {
#     "type": "sqlite-vec",
#     # 如果 SQLite 的 'base_path' 与 MemorySystem 的默认路径不同，可以在此指定
#     "embedding_dim": embedding_handler.dimension
# }


# --- 初始化 ---
chatbot_id = "roleplay_ai_cassandra" # 角色扮演AI的ID
ltm_id = "main_story_arc" # 此角色/故事的长期记忆ID

memory = MemorySystem(
    chatbot_id=chatbot_id,
    ltm_id=ltm_id,
    embedding_handler=embedding_handler,
    llm=llm, # 传入 LLM 以启用总结功能
    vector_store_config=vector_store_config,
    # base_path="./my_chatbot_memories", # 可选：SQLite数据库的自定义路径
    max_context_length=10,
    max_vector_entities=10000, # 向量存储中实体数上限，超过则尝试遗忘
    forget_percentage=0.1 
)
print(f"MemForest 已为 {chatbot_id} 初始化。当前会话 ID: {memory.get_current_sesssion_id()}")

# --- 添加记忆 (模拟对话) ---
print("\n--- 模拟对话 ---")
t1 = datetime.datetime.now() - datetime.timedelta(minutes=10)
msg1 = memory.add_memory("站住！谁敢进入低语森林？", source="cassandra_npc", creation_time=t1, metadata={"action": "speak", "emotion": "suspicious"})
print(f"已添加: {msg1.source} - '{msg1.content}'")

t2 = t1 + datetime.timedelta(seconds=30)
msg2 = memory.add_memory("只是一位卑微的旅者，希望能安全通过。", source="player", creation_time=t2, metadata={"action": "speak", "intention": "peaceful"})
print(f"已添加: {msg2.source} - '{msg2.content}'")

t3 = t2 + datetime.timedelta(minutes=1)
msg3 = memory.add_memory("森林里很危险。说明你的目的，旅者！", source="cassandra_npc", creation_time=t3, metadata={"action": "speak", "emotion": "demanding"})
print(f"已添加: {msg3.source} - '{msg3.content}'")

# --- 获取当前上下文 ---
current_context = memory.get_context(length=5)
print(f"\n--- 当前上下文 (最近 {len(current_context)} 条消息) ---")
for unit in current_context:
    print(f"  [{unit.creation_time.isoformat()}] {unit.source}: {unit.content} (动作: {unit.metadata.get('action', 'N/A')})")

# --- 查询记忆 (语义搜索) ---
query_text = "旅者在森林里的目的"
print(f"\n--- 查询: '{query_text}' ---")
# 获取查询文本的嵌入
query_embedding = embedding_handler.get_embedding(query_text)

# 执行混合查询 (先STM后LTM), 并回忆上下文
results = memory.query(
    query_vector=query_embedding, # 直接传递列表
    k_limit=2,          # 请求最相关的2个记忆链
    recall_context=True, # 获取相邻消息
    add_ltm_to_stm=True  # 将LTM结果缓存到STM
)

print("\n--- 查询结果 (包含上下文回忆) ---")
for i, chain in enumerate(results):
    print(f"找到记忆链 {i+1}:")
    for unit in chain:
        print(f"  -> {unit.source} (ID: ...{unit.id[-6:]}): {unit.content}")

# --- STM (短期记忆) 管理 ---
print("\n--- STM 操作 ---")
if not memory.if_stm_enabled(): # 检查STM是否已启用
    memory.enable_stm(capacity=50, restore_session_id="LATEST") # 启用STM, 从最新会话恢复
    print("STM 已启用并从最新会话恢复 (如果存在)。")

# 当上下文队列满或被刷新时，上下文消息会自动移至 STM。
# 手动刷新上下文 (通常自动发生)
memory.flush_context() 
print("上下文已刷新到 STM/LTM。")

# 仅查询 STM
query_stm_text = "谁在森林里"
print(f"\n--- 仅查询 STM: '{query_stm_text}' ---")
query_stm_embedding = embedding_handler.get_embedding(query_stm_text)
stm_results = memory.query(
    query_vector=query_stm_embedding,
    k_limit=3,
    short_term_only=True # 仅查询 STM 缓存
)
print("\n--- 仅 STM 查询结果 ---")
for i, chain in enumerate(stm_results):
    if chain: # STM 结果是单个单元，除非从 STM 中回忆上下文
        unit = chain[0]
        print(f"  命中 {i+1} (STM): {unit.source} (...{unit.id[-6:]}): {unit.content}")

# --- 会话管理 ---
# current_session_id = memory.get_current_sesssion_id()
# print(f"\n--- 当前会话 ID: {current_session_id} ---")
# 开始新会话 (将生成新的 UUID):
# memory.start_session() # 清除上下文并开始新会话
# print(f"新会话 ID: {memory.get_current_sesssion_id()}")

# --- 总结功能 (如果配置了 LLM) ---
if llm:
    print("\n--- 手动总结 (需要 LLM) ---")
    current_session_id_for_summary = memory.get_current_sesssion_id()
    if current_session_id_for_summary:
        print(f"尝试总结会话: {current_session_id_for_summary}")
        # memory.summarize_session(session_id=current_session_id_for_summary, role="system")
        # print("会话总结尝试完成。")
        # 如果使用 MemorySystem，总结会在后台异步发生。
        # 若直接使用异步API，则需要 await。
    
    # print(f"尝试总结 LTM: {ltm_id}")
    # memory.summarize_long_term_memory(role="system")
    # print("LTM 总结尝试完成。")
else:
    print("\n跳过总结示例 (未配置 LLM)。")

# --- 关闭 MemForest ---
print("\n--- 关闭 MemForest ---")
# 优雅地关闭记忆系统 (刷新最终更改, 关闭连接)
# auto_summarize=True 会在 LLM 可用时触发会话/LTM 总结
memory.close(auto_summarize=False) 
print("MemorySystem 已关闭。")
```

## 6\. 未来规划与路线图

MemForest 是一个积极发展的项目。以下是我们未来计划的一瞥：

  * 🌟 **进一步模块化与抽象化：**
      * 优化 `PersistenceHandler` 接口，实现更清晰的关注点分离。
      * 进一步抽象核心逻辑，以便更容易集成自定义组件。
  * 🚀 **增强高并发处理能力：**
      * 优化异步任务管理，以处理大量并发用户/会话。
      * 探索分布式记忆系统的策略。
  * 💻 **轻量级版本，便于本地部署：**
      * 开发一个 "MemForest-Lite" 版本，依赖项最少（例如，仅 SQLite，基本使用无需独立向量数据库），方便本地聊天机器人部署。
  * ☁️ **便捷的在线部署方案：**
      * 提供清晰的指南和工具（例如 Dockerfiles、无服务器函数示例），用于在线部署由 MemForest 驱动的聊天机器人。
  * 📤 **流线型记忆导入/导出：**
      * 开发强大且用户友好的机制，用于导入/导出整个记忆存档（例如，用于角色共享、备份或在不同后端之间迁移）。
  * 🧠 **优化记忆使用逻辑：**
      * 高级 RAG 策略：探索更复杂的记忆选择和综合技术以适应 LLM 上下文，可能包括基于图的遍历或动态相关性评分。
      * 更智能的遗忘机制：实现更细致的遗忘算法，或许会考虑叙事重要性或用户定义的保留策略。
  * ⚡ **核心组件重构 (性能焦点)：**
      * 通过性能分析识别性能关键部分，如果能获得超越 Python 异步能力的显著提升，考虑使用高性能语言（例如，通过 Python 绑定的 Rust、Go）重新实现这些部分。
  * 📊 **改进遥测与调试：**
      * 集成可选的遥测功能，用于监控记忆系统的性能和健康状况。
      * 提供更好的工具和 MQL (记忆查询语言) 来调试和检查记忆内容。
  * 🤝 **社区与集成：**
      * 探索与流行的聊天机器人框架和 LLM 智能体库的更深度集成。

我们欢迎社区的贡献和反馈！

## 7\. 许可证

MemForest 采用 [MIT 许可证](https://www.google.com/search?q=LICENSE)授权。
