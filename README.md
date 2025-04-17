# MemForest: chatbot的分层记忆系统

## 1. 简介

MemForest 是一个 Python 库，旨在为chatbot配备一个复杂的分层记忆系统，其灵感来源于人类记忆的认知模型。它允许chatbot跨越单条消息、整个会话以及长期保留信息，从而实现更连贯、个性化和上下文感知的交互。MemForest不是简单的对话buffer，它为持久的、不断发展的记忆提供了一个框架。

**主要特点：**

* **分层存储：** 将记忆组织成消息、会话和长期记忆 (LTM) 三个层级，由不同的等级管理。
* **统一的 MemoryUnit：** 一致地存储所有记忆类型，简化管理和查询。
* **短期记忆 (STM)：** 使用 LRU 和 FIFO 缓存管理近期记忆（包括从上下文中淘汰的记忆和从长期记忆库中召回的记忆），以便在对话期间快速访问。
* **长期记忆 (LTM)：** 持久化存储所有历史对话及其总结。
* **多种的储存选择：** 支持 SQLite（推荐）和分区 JSON 文件作为主要存储，并与 Milvus 同步以实现高效的基于向量的检索。
* **动态查询：** 允许使用向量相似性和元数据过滤搜索 STM、LTM 或两者。
* **上下文回忆：** 对任何查询到的记忆，（可选地）召回它的相邻消息（上下文），重建局部对话流程。
* **自动总结：** 使用提供的语言模型 (LLM) 将消息和会话内容浓缩成层次化总结。
* **自动遗忘：** 实现基于访问间隔、频率和创建时间的，可配置的遗忘机制，以管理 LTM 的容量。
* **会话管理：** 处理对话会话，允许恢复某轮会话（到 STM）和删除。
* **外部记忆：** 支持查询只读的外部 Milvus 集合。
* **可配置：** 初始化期间可以配置存储路径、缓存大小以及总结/遗忘行为等关键参数。
* **数据转换实用程序：** 提供在存储格式之间转换的工具（例如，SQLite 到 Milvus）。

## 2. 背景与动机

标准的chatbot通常表现出有限的记忆能力，通常仅限于固定大小的对话窗口。这会导致“失忆”，即chatbot忘记当前对话的早期部分、之前的互动或用户偏好，从而阻碍了深入的长期关系和复杂任务的完成。

MemForest 通过实现一个更持久和结构化的记忆系统来解决这个问题，该系统受到了以下几个概念的启发：

* **认知记忆模型：** 人类记忆并非是单一的；它涉及不同的系统，如短期工作记忆和长期存储，通常具有分层组织结构 (Tulving E. Elements of episodic memory[J]. 1983.; Anderson J R. The architecture of cognition[M]. Psychology Press, 2013.)。MemForest 通过其 STM/LTM 的区分和层次化的 MemoryUnit 结构来模仿这一点。总结过程创建的抽象类似于人类如何从具体事件中形成概括性记忆。
* **解决 LLM 上下文限制：** 大型语言模型 (LLM) 具有有限的上下文窗口。简单地输入整个历史记录通常是不可能的。有效的记忆系统会选择和综合相关信息以适应此窗口。
* **检索增强生成 (RAG)：** 现代 AI 系统通常从外部知识库检索相关信息以提高生成质量 (Lewis P, Perez E, Piktus A, et al. Retrieval-augmented generation for knowledge-intensive nlp tasks[J].)。MemForest 通过将chatbot自己的过去经验（存储在 LTM中）视为知识库，并基于语义检索相关记忆。
* **LLM 作为agent：** 最近的研究探索了如何使用 LLM 作为自主agent的核心，这些agent可以在较长时间内与环境和工具进行交互。持久的、结构化的记忆对于此类agent的学习、适应和维持上下文至关重要 (e.g.，Packer C, Fang V, Patil S G, et al. MemGPT: Towards LLMs as Operating Systems[J]. 2023.)。MemForest 提供了一个适用于此类架构的专用记忆组件。

通过结合层次化存储、总结、向量检索和显式的短期/长期管理，MemForest 旨在为新型的对话 AI 提供一个健壮且可扩展的记忆组件。

## 3. 基本概念、设计与算法

### 3.1. 分层存储与 MemoryUnit

MemForest 将记忆组织成三个概念层级，通过 `MemoryUnit` 的 `rank` 属性区分：

* **消息层级 (rank = 0)：** 代表对话中的单个话语或事件（例如，user消息、ai 回复、system提醒）。这些是记忆系统的最初输入。
* **会话层级 (rank = 1)：** 代表一个连贯的会话。一个会话由多个 rank-0 的 `MemoryUnit` 组成。MemForest 可以自动将一个会话中的所有 rank-0 单元汇总成一个 rank-1 的 `MemoryUnit`。这个root单元的 `id` 与 `SessionMemory` 对象的 ID 相同，表示该会话在记忆层次结构中的MemoryUnit。
* **长期记忆层级 (rank = 2)：** 代表整体记忆上下文，通常汇总多个会话。Rank-1 的会话总结单元可以进一步汇总成一个或多个 rank-2 的 `MemoryUnit`。特定 LTM 的root单元与其 `LongTermMemory` 对象的 ID 相同。
* 注 ： MemForest在总结时设置“消息条数上限”，如果待总结的消息token数或条数超出了指定上限，将触发分组总结和递归总结，产生一系列中间层的单元。

这种层级结构允许在不同的粒度级别进行高效查询，并通过总结实现渐进式信息压缩。

核心数据结构 `MemoryUnit`：

* `id` (str): 唯一标识符 (UUID)。
* `content` (str): 记忆的文本内容（消息、总结等）。
* `source` (Optional[str]): 发起者（例如，“user”、“ai”、“system”、“ai-summary”）。默认为 “ai”。
* `creation_time` (Optional[datetime]): 记忆事件的开始时间戳。
* `end_time` (Optional[datetime]): 记忆事件的结束时间戳。
* `metadata` (Optional[Dict[str, Any]]): 用户定义的字典，用于存储额外信息（例如，`{"action": "speak"}`、消息 ID、情感）。
* `rank` (int): 层级级别（0、1 或 2）。
* `group_id` (Optional[str]): 此单元主要所属的会话或 LTM 的 ID。对于消息分组以及将总结与其范围关联至关重要。对于 rank 0 的单元，它是会话 ID。对于 rank 1 的单元（会话总结），它也是会话 ID（与单元自身的 ID 相同）。对于 rank 2 的单元（LTM 总结），它是 LTM ID。
* **树状链接（总结层级）：**
    * `parent_id` (Optional[str]): 在层次化结构中，直接位于此单元之上的总结单元的 ID（某单元总结得来的新单元的ID）。
    * `children_ids` (List[str]): 直接位于此单元之下的单元的 ID（用于创建此总结单元的子单元）。
* **序列链接（对话流程）：**
    * `pre_id` (Optional[str]): 同一会话中前一个 rank-0 的 `MemoryUnit` 的 ID。形成一个按时间顺序及因果顺序排列的链。
    * `next_id` (Optional[str]): 同一会话中下一个 rank-0 的 `MemoryUnit` 的 ID。
* **重要性记录（用于遗忘）：**
    * `visit_count` (int): 此单元被访问（例如，在查询中检索到）的次数。
    * `last_visit` (int): 最后一次访问此单元，是在第几轮交互轮数。
    * `never_delete` (bool): 防止自动遗忘的标志。

### 3.2. 核心组件

* **SessionMemory：** 代表会话的元数据对象。保存 `id`（与 rank-1 总结 `MemoryUnit` 的 ID 相同），以及一个记忆单元的ID 列表 (`memory_unit_ids`)，该列表记录了该会话中所有单元、由这些单元递归总结而来的全部单元的ID （rank 0 消息 + rank 1 总结）。
* **LongTermMemory：** LTM 实例的元数据对象。保存 `ltm_id`（与根 rank-2 总结 ID 相同）、关联的 `session_ids` 和 `summary_unit_ids`（属于此 LTM 的 rank-2 总结的 ID）。
* **MemorySystem：** 中心协调类。管理：
    * **上下文队列：** 用于立即传入消息的短 FIFO 队列。
    * **短期记忆 (STM)：** 有限容量的 LRU 缓存，用于记录最近访问/从上下文中淘汰的单元及其embedding。支持对最近的消息进行快速语义搜索。
    * **LTM 管理：** 处理与储存系统的交互，能够存储和检索所有非 STM 单元。支持字段储存（SQLite/JSON）和向量存储（Milvus）
    * **持久化处理程序：** 抽象存储交互的模块 (`sqlite_handler`, `json_handler`, `vector_store_handler`)。（计划：可以统一在一个接口下）。
    * **缓存：** 内存字典 (`memory_units_cache`, `session_memories_cache`)，以减少磁盘 I/O。更改会被暂存直到刷新。
    * **总结与遗忘逻辑：** 协调对总结和遗忘模块的调用。

### 3.3. 持久化储存

* **主存储（SQLite/JSON）：** 存储完整的 `MemoryUnit`、`SessionMemory` 和 `LongTermMemory` 数据（不包括embedding）。
    * **SQLite（推荐）：** 将一个chatbot的所有数据存储在一个 `.db` 文件中。提供事务完整性（原子性）和通过 SQL 进行的高效查询，尤其适用于处理许多小型会话和扩展。
    * **JSON：** 将 LTM、会话和单元存储在单独的 JSON 文件中。可能会变得低效（随着消息条数增多）并且缺乏跨文件的事务保证。仅推荐用于较小的数据集或特定的互操作性需求。
* **向量存储（Milvus）：** 存储 `MemoryUnit` embedding以及关键元数据。通过 `vector_store_handler` 实现快速的语义相似性搜索。可以通过 `milvus_config` 进行配置。
* **同步：** `MemorySystem` 确保在刷新过程中将数据写入主存储和 Milvus。

### 3.4. 算法

**算法：**

1.  **总结 (`summarizing.py`)**:
    * **目标：** 在保留关键细节的同时减少信息量，创建分层表示。
    * **过程：** 在不同层级上递归操作（消息 -> 会话总结 -> LTM 总结）。
    * **分组（`Group` 概念）：** 当前级别的单元根据最大计数 (`max_group_size`) 和令牌限制 (`max_token_count`) 贪婪地按时间顺序分组，以控制 LLM 输入大小并避免人为切分自然对话。
    * **LLM 调用：** 使用配置的 LLM (`summarize_memory` 函数) 总结每个group。可以提供历史记录（同一级别的先前的总结，或初始历史记录）。
    * **层级创建：** 为每个总结创建一个新的 `MemoryUnit`（`rank` 递增）。其 `children_ids` 指向被总结组中的单元，并且更新子单元的 `parent_id` 以指向新的总结单元。
    * **递归：** 在新创建的总结单元上重复该过程，直到会话或 LTM 中只剩下一个根单元。

2.  **遗忘 (`forgetting.py`)**:
    * **目标：** 当达到存储限制（例如，`max_milvus_entities`）时，从 LTM 中删除不太重要的信息。
    * **触发：** 定期检查（例如，在刷新更改后），如果 LTM 大小超过阈值则启动。
    * **机制：**
        * **访问跟踪：** 每个 `MemoryUnit` 存储 `visit_count` 和 `last_visit`（最后一次访问的轮数）。访问一个单元（例如，通过查询）会更新其计数器并将自身及其所有祖先（直到根节点）的 `last_visit` 为本轮对话。
        * **候选选择：** 识别 LTM 树中 *未* 标记为 `never_delete` 的符合条件的叶节点（没有 `children_ids`）。
        * **评分：** 根据重要性（最不重要的在前）对符合条件的叶节点进行排序，依据：1. `last_visit`（越早不重要），2. `visit_count`（越低越不重要），3. `creation_time`（越旧越不重要）。
        * **删除：** 删除占所有叶子节点数特定百分比 (`forget_percentage`) 的最不重要的叶节点。
        * **树维护：** 当一个单元被删除时，其在父节点的 `children_ids` 列表中的 `parent_id` 引用将被移除。如果一个节点失去所有子节点，它将成为一个叶节点，并在未来的周期中有资格被删除。

## 4. 安装与使用

### 4.1. 安装

克隆仓库：

```bash
git clone https://github.com/Babel5430/MemForest.git
cd MemForest
```

### 4.2. 依赖

MemForest 需要以下 Python 库：

* `numpy>=1.21.0`: 用于数值运算，尤其是embedding。
* `langchain-core>=0.1.0`: 用于 LLM 和消息类型抽象（例如，`BaseMessage`）。您可能需要其他 `langchain` 包，具体取决于您使用的 LLM（例如，`langchain-openai`）。
* `pymilvus>=2.3.0,<2.4.0`: 用于与 Milvus 向量数据库交互。
* `sentence-transformers>=2.2.0`: 用于生成文本embedding（示例使用 `moka-ai/m3e-small`）。
* `torch>=1.9.0`: `sentence-transformers` 需要。安装可能因您的系统（CPU/GPU）而异。请查看 PyTorch 官方说明。

您通常可以使用以下命令安装这些依赖：

```bash
pip install numpy "langchain-core>=0.1.0" "pymilvus>=2.3.0,<2.4.0" "sentence-transformers>=2.2.0" torch
# 根据需要添加其他特定的 langchain 包，例如：pip install langchain-openai
```

确保您已运行并可以访问 Milvus 实例。

### 4.3. 使用示例

```python
import os
import datetime
from MemForest.manager.memory_system import MemorySystem
from MemForest.utils.embedding_handler import EmbeddingHandler
from sentence_transformers import SentenceTransformer
# 替换为您的api提供商
# from langchain_openai import ChatOpenAI
# llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

llm = None # 设置为您的实际 LLM 实例以进行总结
# --- 初始化 ---

chatbot_id = "memforest_demo_bot"

ltm_id = "primary_ltm"
# 初始化embedding处理程序（默认使用 moka-ai/m3e-small）

embedding_model_name = 'moka-ai/m3e-small'
embedding_model = SentenceTransformer(embedding_model_name, device='cpu') # 如果有 GPU 可用，则使用 'cuda'
embedding_handler = EmbeddingHandler(model=embedding_model, device='cpu') # 匹配设备
# Milvus 配置（确保 Milvus 正在运行！）

milvus_config = {
    "host": "localhost", # 或您的 Milvus 主机 IP/域名
    "port": "19530", # 默认 Milvus 端口
}
# 初始化记忆系统
# 存储路径将默认为当前目录下的 './memory_storage/<chatbot_id>'

memory = MemorySystem(
    chatbot_id=chatbot_id,
    ltm_id=ltm_id,
    embedding_handler=embedding_handler,
    llm=llm, # 传递 LLM 以实现总结功能
    milvus_config=milvus_config,
    persistence_mode='sqlite', # 'sqlite' 或 'json', 推荐sqlite
    max_context_length=10, # 示例：在上下文中保留最近的 10 条消息
    max_milvus_entities=10000, # 示例：超过 1 万个实体时触发遗忘
    forget_percentage=0.1 # 示例：触发时遗忘 10% 的符合条件的单元
)

print(f"当前会话 ID：{memory._current_session_id}") # 为了演示而访问私有属性
# --- 添加记忆 ---

t1 = datetime.datetime.now() - datetime.timedelta(minutes=5)
unit1 = memory.add_memory("项目的代号是什么？", source="user", creation_time=t1)
print(f"已添加：用户 - '{unit1.content}'")

t2 = t1 + datetime.timedelta(seconds=30)
unit2 = memory.add_memory("该项目的代号是“MemForest”。", source="ai", creation_time=t2)
print(f"已添加：AI - '{unit2.content}'")

t3 = t2 + datetime.timedelta(minutes=2)
unit3 = memory.add_memory("状态如何？", source="user", creation_time=t3)
print(f"已添加：用户 - '{unit3.content}'")

t4 = t3 + datetime.timedelta(seconds=30)
unit4 = memory.add_memory("“MemForest”项目目前处于暂停状态。", source="ai", creation_time=t4)
print(f"已添加：AI - '{unit4.content}'")
# --- 获取上下文 ---

# 获取在推送到 STM/LTM 之前当前持有的最后 'n' 条消息

current_context = memory.get_context(length=5)
print(f"当前上下文（{len(current_context)} 条消息）：")
for unit in current_context:
    print(f" - [{unit.creation_time.isoformat()}] {unit.source}: {unit.content}")
# --- 查询记忆 ---

query_text = "告诉我关于凤凰项目的信息"
print(f"正在查询：'{query_text}'")
query_embedding = embedding_handler.get_embedding(query_text).tolist()

# 示例 1：混合查询（默认：先 STM，然后 LTM），带上下文回忆
results_hybrid = memory.query(
    query_vector=query_embedding,
    k_limit=2, # 请求前 2 个相关的链/命中
    recall_context=True, # 获取邻居（pre_id, next_id）
    add_ltm_to_stm=True # 将在 LTM 中找到的结果添加到 STM 缓存
)

print("\n混合查询结果（Recall Context=True）：")
for i, chain in enumerate(results_hybrid):
    print(f" chain {i+1}:")
    for unit in chain:
        print(f" -> {unit.source} ({unit.id[-6:]}): {unit.content}") # 显示 ID 的最后 6 个字符

# 示例 2：仅长期记忆查询，不带上下文回忆，特定来源
results_ltm_only = memory.query(
    query_vector=query_embedding,
    filters=[("source", "==", "ai")], # 仅查找 AI 消息
    k_limit=3,
    long_term_only=True, # 跳过 STM
    recall_context=False # 只返回命中的单元
)

print("\n仅 LTM 查询结果（AI 来源，Recall Context=False）：")
for i, chain in enumerate(results_ltm_only): # 此处每个“链”只有一个单元
    if chain:
        unit = chain[0]
        print(f" 命中 {i+1}: {unit.source} ({unit.id[-6:]}): {unit.content}")
# --- 短期记忆 (STM) ---

# 启用容量为 50 的 STM，如果可用，则从最新会话恢复
print("启用 STM（容量=50，恢复=最新）...")
memory.enable_stm(capacity=50, restore_session_id="LATEST")

# 强制将上下文刷新到 STM和 LTM（通常在上下文队列满时发生）
print("将上下文刷新到 STM和 LTM...")
memory._flush_context() # 为演示使用私有方法

# 仅查询 STM
query_text_stm = "项目状态"
print(f"仅查询 STM：'{query_text_stm}'")
query_embedding_stm = embedding_handler.get_embedding(query_text_stm).tolist()
results_stm_only = memory.query(
    query_vector=query_embedding_stm,
    k_limit=5,
    short_term_only=True # 仅查询 STM 缓存
)

print("\n仅 STM 查询结果：")
for i, chain in enumerate(results_stm_only):
    if chain:
        unit = chain[0]
        print(f" 命中 {i+1} (STM): {unit.source} ({unit.id[-6:]}): {unit.content}")

# 示例：从特定会话恢复 STM（如果您知道其 ID）
# session_to_restore = "some-known-session-id"
# print(f"禁用并重新启用 STM，从会话恢复：{session_to_restore}")
# memory.disable_stm()
# memory.enable_stm(capacity=50, restore_session_id=session_to_restore) # 使用特定 ID

print("禁用 STM...")
memory.disable_stm()
# --- 会话管理 ---

current_session_id = memory._current_session_id
print(f"当前会话 ID：{current_session_id}")

# 通常隐式或通过自定义方法启动新会话
# 示例：强制启动新会话（如果存在公共方法）
# memory.start_new_session()
# print(f"新会话 ID：{memory._current_session_id}")

# 示例：删除会话（谨慎使用！）
# 请务必确定要删除会话及其所有消息。
# session_id_to_remove = current_session_id # 示例：删除当前会话
# print(f"尝试删除会话：{session_id_to_remove}")
# try:
#     memory.remove_session(session_id_to_remove)
#     print(f"会话 {session_id_to_remove} 已删除。")
# except Exception as e:
#     print(f"删除会话时出错：{e}")

# --- 手动总结/遗忘 ---
# 注意：这些需要 MemorySystem 中配置 LLM

print("\n--- 手动触发（需要 LLM） ---")
if llm:
    # 示例：总结特定会话
    # print(f"尝试总结会话：{current_session_id}")
    # memory.summarize_session(session_id=current_session_id)

    # 示例：总结整个 LTM
    # print(f"尝试总结 LTM：{ltm_id}")
    # memory.summarize_long_term_memory()
    pass # 如果未设置 LLM，则作为占位符
else:
    print("跳过总结示例（未配置 LLM）。")

# 如果 Milvus 计数超过阈值，则在 _flush_cache 期间发生自动遗忘
# 要手动触发检查（主要用于测试）：
# print("手动检查遗忘条件...")
# memory._check_and_forget_memory() # 私有方法，通常是自动的

# --- 数据转换 ---
# print("\n--- 数据转换实用程序 ---")
# 确保在转换之前拥有源数据
# print("示例：将 SQLite 转换为 JSON（如果使用 SQLite）")
# if memory.persistence_mode == 'sqlite':
#     memory.convert_sql_to_json(output_dir="./memory_storage_json_copy")
#     print("已尝试转换为 JSON（检查输出目录）。")

# print("示例：将 JSON 转换为 SQLite（如果使用 JSON）")
# if memory.persistence_mode == 'json':
#     memory.convert_json_to_sqlite()
#     print("已尝试转换为 SQLite（应更新数据库文件）。")

# print("示例：将主存储转换为 Milvus（生成embedding）")
# 需要填充主存储（SQLite 或 JSON）
# if memory.persistence_mode == 'sqlite':
#     memory.convert_sql_to_milvus(embedding_history_length=1)
# elif memory.persistence_mode == 'json':
#     memory.convert_json_to_milvus(embedding_history_length=1)
# print("已尝试转换为 Milvus。")

# --- 保存与关闭 ---

# 根据 SAVING_INTERVAL/VISIT_UPDATE_INTERVAL 定期刷新更改
# 如果需要，手动强制刷新（通常不需要）：
# print("强制刷新缓存...")
# memory._flush_cache(force=True)

# 轻松地关闭内存系统（刷新最终更改，关闭连接）
memory.close(auto_summarize=False) # 如果auto_summarize=True 且 LLM 已配置，将自动总结
print("MemorySystem closed.")
```

## 5. TODO List / Future Work

本清单包含计划中的改进和未来开发方向：

**存储与可扩展性：**

* **数据库后端：** 正式推荐并优化SQLite。考虑增加对PostgreSQL + pgvector的支持作为更易扩展的替代方案，可能取代Milvus以简化部署。
* **SQLite优化：** 持续审查查询性能，优化索引策略（复合索引），确保WAL模式为默认设置。
* **缓存策略：** 对内存缓存（`memory_units_cache`, `session_memories_cache`）实施严格的容量限制（内存/条目数）和LRU淘汰机制。确保默认采用数据库惰性加载。
* **异步处理：**
    * **全异步IO：** 将所有数据库（SQLite/Postgres）、Milvus及潜在LLM/embedding操作转换为使用`asyncio`（例如`aiosqlite`, `asyncio.to_thread`）以避免阻塞。重构核心`MemorySystem`方法（`add_memory`, `query`, `flush`等）为`async def`。
    * **后台任务：** 使用`asyncio.create_task`处理刷新、总结生成和遗忘等后台操作。
* **一致性与原子性：**
    * **跨存储事务：** 实施健壮的策略（如发件箱模式或结合重试逻辑的谨慎序列化操作），以增强主存储（SQLite）与向量存储（Milvus）间的一致性，特别是删除操作。
    * **幂等性：** 审查关键操作以确保其幂等性（可安全重试）。

**代码结构与优化：**

* **持久化接口：** 将持久化逻辑重构到抽象`PersistenceHandler`接口后端，解耦`MemorySystem`与具体实现（SQLite、JSON）。
* **简化加载：** 移除仅用于过滤/处理大量数据的Python代码，将操作下推到数据库层通过定向查询实现。
* **弃用JSON：** 考虑完全移除JSON持久化后端以简化代码库，专注于更可扩展的数据库方案。

**性能：**

* **数据库查询优化：** 优先优化SQL查询而非Python循环进行数据操作（排序、过滤、聚合），例如在遗忘机制中。
* **性能分析：** 集成分析工具（`cProfile`, `line_profiler`）以系统识别主要优化（如异步IO）后的负载瓶颈。
* **编译型扩展（必要时）：** 若分析显示Python算法（如复杂图遍历、自定义评分）存在持续CPU瓶颈且无法下推到数据库，考虑用Cython/Rust/Go实现关键部分。

**功能与增强：**

* **配置文件：** 支持从`config.yaml`等文件加载设置（超参数、路径、数据库配置）。
* **增强遗忘机制：** 探索更复杂策略（基于查询相似性的相关性衰减、图中心性）。
* **测试：** 开发完整的测试套件（单元测试和集成测试）。
* **错误处理：** 实现更细粒度的错误处理和日志记录。
* **查询增强：** 添加时间衰减相关性评分等功能。
* **embedding维度：** 从embedding模型自动推导`EMBEDDING_DIMENSION`。
* **公开会话API：** 添加明确方法如`start_new_session()`或`switch_session(session_id)`。
