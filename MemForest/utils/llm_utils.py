"""
Utilities for LLM-related operations including summarization and token counting.
"""

from typing import List, Dict, Optional, Any
from MemForest.memory.memory_unit import MemoryUnit
import tiktoken
from langchain_core.language_models import BaseChatModel
from transformers import AutoTokenizer

# Initialize tokenizer


DEFAULT_SYSTEM_MESSAGE = [
    {
        "role": "system",
        "content": """以"{role}"的第一人称视角总结以下互动,
要求:
    1.使用口语化中文表达
    2.包含"{role}"视角的主观体验
    3.保留关键信息和决策节点
    4.总字数控制在200字以内"""
    }
]


def get_num_tokens(memory_unit: MemoryUnit, model: str = "gpt-3.5-turbo-0613") -> int:
    """
    Calculate the number of tokens in a MemoryUnit's content.

    Args:
        memory_unit: The MemoryUnit to analyze
        model: The model name for tokenization

    Returns:
        Number of tokens
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(memory_unit.content)) + 1


def summarize_memory(
        memory_units: List[MemoryUnit],
        llm: BaseChatModel,
        system_message: Optional[List[Dict[str, Any]]] = DEFAULT_SYSTEM_MESSAGE,
        logit_bias: Optional[Dict[Any, Any]] = None
) -> Optional[str]:
    """
    Summarize a list of MemoryUnits using an LLM.

    Args:
        memory_units: List of MemoryUnits to summarize
        llm: Language model to use
        system_message: Instructions for the LLM
        logit_bias: Optional bias for certain tokens

    Returns:
        The summarized content or None if failed
    """
    if not system_message:
        system_message = DEFAULT_SYSTEM_MESSAGE

    history = [{
        "role": "user",
        "content": '\n'.join(
            f"{unit.source}-{unit.metadata['action']}: {unit.content}"
            for unit in memory_units
        )
    }]

    messages = system_message + history
    response = llm.invoke(
        messages,
        max_tokens=2000,
        temperature=0.3,
        presence_penalty=0.2,
        logit_bias=logit_bias,
        stream=False
    )

    return response.content if response else None


def get_blocked_words_bias(blocked_words: List[str]) -> Dict[str, int]:
    """
    Create logit bias against certain words.

    Args:
        blocked_words: List of words to block

    Returns:
        Dictionary of token IDs to bias values
    """
    tokenizer = AutoTokenizer.from_pretrained('deepseek_v3', trust_remote_code=True)
    blocked_tokens = []
    for word in blocked_words:
        blocked_tokens.extend(tokenizer.tokenize(word))

    blocked_token_ids = tokenizer.convert_tokens_to_ids(blocked_tokens)
    return {str(token_id): -200 for token_id in blocked_token_ids}