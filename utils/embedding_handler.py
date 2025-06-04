from typing import List, Union
import numpy as np
import onnxruntime
from tokenizers import Tokenizer

EMBEDDING_DIMENSION = 512
DEFAULT_MAX_SEQ_LENGTH = 512

class EmbeddingHandler:
    def __init__(self, model_path: str, tokenizer_path: str, dimension: int = EMBEDDING_DIMENSION,
                 max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH):
        self.session = onnxruntime.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.tokenizer = Tokenizer.from_file(tokenizer_path)

        pad_token_id = self.tokenizer.token_to_id("[PAD]")
        if pad_token_id is None: pad_token_id = 0

        self.tokenizer.enable_padding(pad_id=pad_token_id, pad_token="[PAD]", length=max_seq_length)
        self.tokenizer.enable_truncation(max_length=max_seq_length)

        self.dimension = dimension
        self.input_names = [input_node.name for input_node in self.session.get_inputs()]
        self.output_names = [output_node.name for output_node in self.session.get_outputs()]

    def _mean_pooling(self, model_output: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
        """ Performs mean pooling on token embeddings using the attention mask. """
        token_embeddings = model_output  # (batch_size, seq_length, hidden_size)
        input_mask_expanded = np.expand_dims(attention_mask, axis=-1).astype(float)
        sum_embeddings = np.sum(token_embeddings * input_mask_expanded, axis=1)
        sum_mask = np.clip(input_mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)
        return sum_embeddings / sum_mask

    def get_embedding(self, text: Union[str, List[str]]) -> np.ndarray:
        if isinstance(text, str):
            texts = [text]
        else:
            texts = text
        if not texts: return np.array([]).reshape(0, self.dimension)

        encoded_inputs = self.tokenizer.encode_batch(texts)
        input_feed = {}
        if 'input_ids' in self.input_names:
            input_feed['input_ids'] = np.array([e.ids for e in encoded_inputs], dtype=np.int64)
        if 'attention_mask' in self.input_names:
            input_feed['attention_mask'] = np.array([e.attention_mask for e in encoded_inputs], dtype=np.int64)
        if 'token_type_ids' in self.input_names:
            input_feed['token_type_ids'] = np.array([e.type_ids for e in encoded_inputs], dtype=np.int64)

        model_outputs = self.session.run(self.output_names, input_feed)
        last_hidden_state = model_outputs[0]

        if 'attention_mask' not in input_feed:
            raise ValueError("Attention mask is required for mean pooling but not found in input_feed.")
        embeddings = self._mean_pooling(last_hidden_state, input_feed['attention_mask'])
        if len(embeddings.shape) == 1: embeddings = embeddings.reshape(1, -1)
        if embeddings.shape[0] == 0: return np.array([]).reshape(0, self.dimension)
        norm = np.linalg.norm(embeddings, ord=2, axis=1, keepdims=True)
        norm = np.where(norm == 0, 1e-9, norm)
        normalized_embeddings = embeddings / norm

        return normalized_embeddings