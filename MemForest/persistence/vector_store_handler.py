from pymilvus import Collection, connections, FieldSchema, DataType, CollectionSchema, utility
from typing import Dict, Any, List, Optional

EMBEDDING_DIMENSION = 512  # Dimension of embeddings


class VectorStoreHandler:
    def __init__(self, chatbot_id: str, long_term_memory_id: str, host: str = "localhost", port: str = "19530",
                 index_params: Optional[Dict[str, Any]] = None):
        self.chatbot_id = chatbot_id.replace('-', '_')
        self.long_term_memory_id = long_term_memory_id.replace('-', '_')
        self.host = host
        self.port = port
        self.collection_name = f"chatbot_{self.chatbot_id}_ltm_{self.long_term_memory_id}"
        connections.connect(host=host, port=port)
        self.collection = self._get_or_create_collection(index_params)
        self.collection.load()
        self.output_fields = ["id", "parent_id", "content", "creation_time", "end_time", "source", "metadata",
                              "last_visit", "visit_count", "never_delete", "children_ids", "embedding", "rank",
                              "pre_id", "next_id", "group_id"]

    def _get_or_create_collection(self, index_params: Optional[Dict[str, Any]] = None):
        if not utility.has_collection(self.collection_name):
            fields = [
                FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=64),
                FieldSchema(name="parent_id", dtype=DataType.VARCHAR, max_length=64, nullable=True),
                FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=8000),
                FieldSchema(name="creation_time", dtype=DataType.FLOAT, nullable=True),
                FieldSchema(name="end_time", dtype=DataType.FLOAT, nullable=True),
                FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=64, nullable=True),
                FieldSchema(name="metadata", dtype=DataType.JSON, nullable=True),
                FieldSchema(name="last_visit", dtype=DataType.INT32),
                FieldSchema(name="visit_count", dtype=DataType.INT32),
                FieldSchema(name="never_delete", dtype=DataType.BOOL, default_value=False),
                FieldSchema(name="children_ids", dtype=DataType.JSON, nullable=True),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIMENSION),
                FieldSchema(name="rank", dtype=DataType.INT8),
                FieldSchema(name="pre_id", dtype=DataType.VARCHAR, max_length=64, nullable=True),
                FieldSchema(name="next_id", dtype=DataType.VARCHAR, max_length=64, nullable=True),
                FieldSchema(name="group_id", dtype=DataType.VARCHAR, max_length=64, nullable=True),
            ]
            schema = CollectionSchema(auto_id=False, fields=fields,
                                      description=f"Long-term memory for chatbot {self.chatbot_id} with ID {self.long_term_memory_id}")
            collection = Collection(name=self.collection_name, schema=schema)
            default_index_params = {
                "metric_type": "IP",
                "params": {"nlist": 64},
                "index_type": "FLAT"
            }
            final_index_params = index_params if index_params is not None else default_index_params
            collection.create_index(field_name="embedding", index_params=final_index_params)
            return collection
        else:
            return Collection(self.collection_name)

    def has_collection(self):
        return utility.has_collection(self.collection_name)

    def insert(self, data: List[Dict[str, Any]], flush=True):
        self.collection.insert(data)
        if flush:
            self.collection.flush()

    def upsert(self, data: List[Dict[str, Any]], flush=True):
        self.collection.upsert(data)
        if flush:
            self.collection.flush()

    def count_entities(self, consistently: bool = False):
        if consistently:
            results = self.collection.query(expr="", output_fields=["count(*)"])
            return results[0]["count(*)"]
        else:
            return self.collection.num_entities

    def query(self, expr: str = None, top_k: int = 10, output_fields: Optional[List[str]] = None,
              consistency_level="Strong"):
        return self.collection.query(expr=expr, limit=top_k, output_fields=output_fields,
                                     consistency_level=consistency_level)

    def get(self, id: str, output_fields: Optional[List[str]] = None):
        result = self.collection.query(f'id == \"{id}\"',
                                       output_fields=output_fields if output_fields else self.output_fields)
        return result[0] if len(result) > 0 else None

    def search(self, vectors: List[List[float]], expr: str = None, top_k: int = 5,
               output_fields: Optional[List[str]] = None, search_params: Optional[Dict[str, Any]] = None,
               consistency_level: str = "Strong"):
        deault_search_params = {
            "metric_type": "IP",
            "params": {"nprobe": 10,
                       "radius": 0.8,
                       "range_filter": 1.0},
        }
        return self.collection.search(
            data=vectors,
            anns_field="embedding",
            param=search_params if search_params else deault_search_params,
            limit=top_k,
            expr=expr,
            output_fields=output_fields if output_fields else self.output_fields,
            consistency_level=consistency_level
        )

    def clone_from(self, chatbot_id: str, long_term_memory_id: str, expr: str = "id != \'\'"):
        _chatbot_id = chatbot_id.replace('-', '_')
        _long_term_memory_id = long_term_memory_id.replace('-', '_')
        original_name = f"chatbot_{_chatbot_id}_ltm_{_long_term_memory_id}"
        original = Collection(original_name)
        original.load()
        entities = original.query(expr=expr, output_fields=["*"])
        self.collection.upsert(entities)
        self.collection.flush()
        original.release()
        self.collection.delete(f"id == \"{long_term_memory_id}\"")
        print(f"Successfully cloned collection {original_name} to {self.collection.name}")

    def delete(self, expr: str, flush=True):
        self.collection.delete(expr=expr)
        if flush:
            self.collection.flush()

    def flush(self):
        self.collection.flush()

    def close(self, flush=True):
        if flush:
            self.flush()
        self.collection.release()
