import sentence_transformers
import torch
from typing import List, Union
import numpy as np
# --- Configuration ---
DEFAULT_STORAGE_PATH = "memory_storage"
EMBEDDING_DIMENSION = 512  # Dimension of moka-ai/m3e-small embeddings

class EmbeddingHandler:
    device = None
    model:sentence_transformers.SentenceTransformer = None
    def __init__(self, model:sentence_transformers.SentenceTransformer, device: str = "cpu"):
        self.model = model
        self.device = device

    def get_embedding(self, text: Union[str,List[str]]) -> np.ndarray:
        with torch.no_grad():
            embeddings = self.model.encode(text)
            norm = np.linalg.norm(embeddings, ord=2,axis=-1)
            if len(embeddings.shape)==2:
                norm = norm.reshape((norm.shape[0],1))
            embeddings = embeddings / norm
        res=embeddings
        return res