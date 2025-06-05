import asyncio
from typing import Dict, Any, List, Optional, Set, Tuple
from datetime import datetime
import numpy as np

try:
    from pymilvus import Collection, connections, FieldSchema, DataType, CollectionSchema, utility
    MILVUS_AVAILABLE = True
except ImportError:
    print("Milvus client (pymilvus) not installed. Milvus/Milvus-lite support will be disabled.")
    MILVUS_AVAILABLE = False

try:
    import chromadb
    from chromadb.api.models.Collection import Collection as ChromaCollection
    from chromadb.api.types import Include

    CHROMA_AVAILABLE = True
except ImportError:
    print("ChromaDB client not installed. Chroma support will be disabled.")
    CHROMA_AVAILABLE = False

try:
    from qdrant_client import QdrantClient, models
    from qdrant_client.http.models import Distance, VectorParams, PointStruct, Filter, FilterSelector, FieldCondition, \
        Range, MatchValue, MatchText, MatchAny, MatchExcept, GeoBoundingBox, GeoRadius, HasIdCondition, NestedCondition, \
        ScrollResult, ScoredPoint

    QDRANT_AVAILABLE = True
except ImportError:
    print("Qdrant client (qdrant-client) not installed. Qdrant support will be disabled.")
    QDRANT_AVAILABLE = False

try:
    from MemForest.persistence.sqlite_handler import AsyncSQLiteHandler, SQLITE_VEC_AVAILABLE
except ImportError:
    print("AsyncSQLiteHandler or sqlite_handler module not found. Cannot delegate to sqlite-vec.")
    class AsyncSQLiteHandler:
        def __init__(self, *args, **kwargs):
            raise NotImplementedError("AsyncSQLiteHandler not available.")
        async def search_vectors_sqlite_vec(self, *args, **kwargs):
            raise NotImplementedError("AsyncSQLiteHandler not available.")


    SQLITE_VEC_AVAILABLE = False

NONE_PLACE_HOLDER = '__NONE__'


class AsyncVectorStoreBackend:
    async def initialize(self, config: Dict[str, Any]):
        """Asynchronously initializes the backend."""
        pass

    async def has_collection(self, collection_name: str) -> bool:
        """Checks if a collection exists asynchronously."""
        pass

    async def get_or_create_collection(self, collection_name: str, embedding_dim: int, index_params: Optional[
        Dict[str, Any]] = None) -> Any:  # Return type depends on backend
        """Gets or creates a collection asynchronously."""
        pass

    async def insert(self, collection: Any, data: List[Dict[str, Any]]):
        """Inserts data asynchronously."""
        pass

    async def upsert(self, collection: Any, data: List[Dict[str, Any]]):
        """Upserts data asynchronously."""
        pass

    async def delete(self, collection: Any, ids: Optional[List[str]] = None, expr: Optional[str] = None):
        """Deletes data by ids or expression asynchronously."""
        pass

    async def count_entities(self, collection: Any, consistently: bool = False) -> int:
        """Counts entities in the collection asynchronously."""
        pass

    async def query(self, collection: Any, expr: str, top_k: int, output_fields: Optional[List[str]] = None) -> List[
        Dict[str, Any]]:
        """Queries data by expression asynchronously."""
        pass

    async def get(self, collection: Any, id: str, output_fields: Optional[List[str]] = None) -> Optional[
        Dict[str, Any]]:
        """Gets data by ID asynchronously."""
        pass

    async def get_all_unit_ids(self, collection: Any) -> List[str]:
        """Gets all unit IDs in the collection asynchronously."""
        raise NotImplementedError("Backend must implement get_all_unit_ids")

    async def search(self, collection: Any, vectors: List[List[float]], expr: Optional[str], top_k: int,
                     output_fields: Optional[List[str]] = None, search_params: Optional[Dict[str, Any]] = None) -> List[
        Dict[str, Any]]:
        """Performs vector search asynchronously."""
        pass

    async def flush(self, collection: Any):
        """Flushes data to persistent storage asynchronously."""
        pass

    async def delete_collection(self, collection_name: str):
        """Deletes a collection asynchronously."""
        raise NotImplementedError("Backend must implement delete_collection")

    async def close(self, collection: Any):
        """Closes the connection/releases resources asynchronously."""
        pass

    async def _run_sync(self, func, *args, **kwargs):
        """Runs a synchronous function in a separate thread to avoid blocking the event loop."""
        return await asyncio.to_thread(func, *args, **kwargs)


# --- Milvus/Milvus Lite Backend Handler (Async Wrapper) ---
class MilvusHandler(AsyncVectorStoreBackend):
    def __init__(self):
        self._conn_alias = f"milvus_conn_{id(self)}"
        self._connected = False

    async def initialize(self, config: Dict[str, Any]):
        """Asynchronously initializes Milvus/Milvus-lite connection."""
        if not MILVUS_AVAILABLE:
            raise RuntimeError("Milvus client not available.")
        print(config)
        uri = config.get("uri")
        host = config.get("host")
        port = config.get("port")
        secure = config.get("secure", False)

        connection_params = {"alias": self._conn_alias, "secure": secure}

        if uri:
            connection_params["uri"] = uri
        elif host and port:
            connection_params["host"] = host
            connection_params["port"] = port
        else:
            # Default to localhost:19530 if no specific config provided
            print("MilvusHandler: No specific connection config provided, defaulting to localhost:19530.")
            connection_params["host"] = "localhost"
            connection_params["port"] = "19530"

        try:
            await self._run_sync(connections.connect, **connection_params)
            self._connected = True
            print(f"MilvusHandler: Connection established with alias {self._conn_alias}")
        except Exception as e:
            print(f"MilvusHandler: Failed to connect to Milvus/Milvus-lite: {e}")
            self._connected = False
            raise ConnectionError(f"Could not connect to Milvus/Milvus-lite: {e}") from e

    async def has_collection(self, collection_name: str) -> bool:
        """Checks if a Milvus collection exists asynchronously."""
        if not self._connected: raise ConnectionError("Milvus not connected.")
        if not MILVUS_AVAILABLE: raise RuntimeError("Milvus client not available.")
        return await self._run_sync(utility.has_collection, collection_name, using=self._conn_alias)

    async def get_or_create_collection(self, collection_name: str, embedding_dim: int,
                                       index_params: Optional[Dict[str, Any]] = None) -> Any:
        """Gets or creates a Milvus collection asynchronously."""
        if not self._connected: raise ConnectionError("Milvus not connected.")
        if not MILVUS_AVAILABLE: raise RuntimeError("Milvus client not available.")
        if await self.has_collection(collection_name):
            collection = await self._run_sync(Collection, collection_name, using=self._conn_alias)
            print(f"MilvusHandler: Got existing collection: {collection_name}")
            try:
                await self._run_sync(collection.peek, limit=1)
            except Exception as e:
                try:
                    await self._run_sync(collection.load)
                    print(f"MilvusHandler: Loaded collection: {collection_name}")
                except Exception as e:
                    print(f"MilvusHandler: Failed to load collection {collection_name}: {e}")
                    raise RuntimeError(f"Failed to load Milvus collection {collection_name}: {e}") from e
            return collection
        else:
            print(f"MilvusHandler: Creating new collection: {collection_name}")
            fields = [
                FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=256),
                FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=8000),
                FieldSchema(name="creation_time", dtype=DataType.DOUBLE, nullable=True),
                FieldSchema(name="end_time", dtype=DataType.DOUBLE, nullable=True),
                FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=64, nullable=True),
                FieldSchema(name="metadata", dtype=DataType.JSON, nullable=True),
                FieldSchema(name="last_visit", dtype=DataType.INT32),
                FieldSchema(name="visit_count", dtype=DataType.INT32),
                FieldSchema(name="never_delete", dtype=DataType.BOOL, default_value=False),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=embedding_dim),
                FieldSchema(name="rank", dtype=DataType.INT8)
            ]
            schema = CollectionSchema(auto_id=False, fields=fields,
                                      description=f"Memory units for {collection_name}")
            collection = await self._run_sync(Collection, name=collection_name, schema=schema, using=self._conn_alias)
            default_index_params = {
                "metric_type": "IP",
                "index_type": "FLAT",
                "params": {"nlist": 64}
            }
            final_index_params = index_params if index_params is not None else default_index_params
            print(f"MilvusHandler: Creating index with params: {final_index_params}")
            await self._run_sync(collection.create_index, field_name="embedding", index_params=final_index_params)

            await self._run_sync(collection.load)
            print(f"MilvusHandler: Loaded collection: {collection_name}")
            return collection

    async def insert(self, collection : Any, data: List[Dict[str, Any]], flush=False):
        """Inserts data into Milvus collection asynchronously."""
        if not self._connected: raise ConnectionError("Milvus not connected.")
        if not MILVUS_AVAILABLE: raise RuntimeError("Milvus client not available.")
        if not data: return
        try:
            data_copy = []
            for unit in data:
                unit_copy = unit.copy()
                data_copy.append(unit_copy)
            insert_result = await self._run_sync(collection.insert, data_copy)
            if flush:
                await self._run_sync(collection.flush)
            # print(f"MilvusHandler: Inserted {insert_result.insert_count} entities.")
        except Exception as e:
            print(f"MilvusHandler: Error during insert: {e}")
            raise

    async def upsert(self, collection : Any, data: List[Dict[str, Any]], flush=False):
        """Upserts data into Milvus collection asynchronously."""
        if not self._connected: raise ConnectionError("Milvus not connected.")
        if not MILVUS_AVAILABLE: raise RuntimeError("Milvus client not available.")
        if not data: return
        try:
            data_copy = []
            for unit in data:
                unit_copy = unit.copy()
                data_copy.append(unit_copy)
            upsert_result = await self._run_sync(collection.upsert, data_copy)
            if flush:
                await self._run_sync(collection.flush)
            # print(f"MilvusHandler: Upserted {upsert_result.upsert_count} entities.")
        except Exception as e:
            print(f"MilvusHandler: Error during upsert: {e}")
            raise

    async def delete(self, collection : Any, ids: Optional[List[str]] = None, expr: Optional[str] = None,
                     flush=False):
        """Deletes data from Milvus collection by ids or expression asynchronously."""
        if not self._connected: raise ConnectionError("Milvus not connected.")
        if not MILVUS_AVAILABLE: raise RuntimeError("Milvus client not available.")
        if not ids and not expr: return
        try:
            if ids:
                ids_quoted = [f"'{id}'" for id in ids]
                delete_expr = f"id in [{', '.join(ids_quoted)}]"
                # print(f"MilvusHandler: Deleting with expr: {delete_expr}")
                delete_result = await self._run_sync(collection.delete, delete_expr)
                if flush:
                    await self._run_sync(collection.flush)
                # print(f"MilvusHandler: Deleted {delete_result.delete_count} entities by IDs.")
            elif expr:
                # print(f"MilvusHandler: Deleting with expr: {expr}")
                delete_result = await self._run_sync(collection.delete, expr)
            if flush:
                await self._run_sync(collection.flush)
            #  print(f"MilvusHandler: Deleted {delete_result.delete_count} entities by expression.")

        except Exception as e:
            print(f"MilvusHandler: Error during delete: {e}")
            raise

    async def count_entities(self, collection : Any, consistently: bool = False) -> int:
        """Counts entities in Milvus collection asynchronously."""
        if not self._connected: raise ConnectionError("Milvus not connected.")
        if not MILVUS_AVAILABLE: raise RuntimeError("Milvus client not available.")
        try:
            if consistently:
                results = await self._run_sync(collection.query, expr="", output_fields=["count(*)"])
                return results[0]["count(*)"] if results else 0
            else:
                return await self._run_sync(lambda c: c.num_entities, collection)
        except Exception as e:
            print(f"MilvusHandler: Error getting entity count: {e}")
            return 0

    async def query(self, collection : Any, expr: str, top_k: int, output_fields: Optional[List[str]] = None) -> \
            List[Dict[str, Any]]:
        """Queries Milvus data by expression asynchronously."""
        if not self._connected: raise ConnectionError("Milvus not connected.")
        if not MILVUS_AVAILABLE: raise RuntimeError("Milvus client not available.")
        if not expr: return []

        default_output_fields = ["id", "content", "creation_time", "end_time", "source", "metadata",
                                 "last_visit", "visit_count", "never_delete", "rank"]
        fields_to_get = output_fields if output_fields is not None else default_output_fields

        try:
            results = await self._run_sync(collection.query, expr=expr, limit=top_k, output_fields=fields_to_get,
                                           consistency_level="Strong")
            return results
        except Exception as e:
            print(f"MilvusHandler: Error during query: {e}")
            return []

    async def get(self, collection : Any, id: str, output_fields: Optional[List[str]] = None) -> Optional[
        Dict[str, Any]]:
        """Gets a single entity from Milvus by ID asynchronously."""
        if not self._connected: raise ConnectionError("Milvus not connected.")
        if not MILVUS_AVAILABLE: raise RuntimeError("Milvus client not available.")
        expr = f'id == "{id}"'
        try:
            results = await self.query(collection, expr=expr, top_k=1, output_fields=output_fields)
            return results[0] if results else None
        except Exception as e:
            return None

    async def get_all_unit_ids(self, collection: Any) -> List[str]:
        """Gets all unit IDs from Milvus collection asynchronously."""
        if not self._connected: raise ConnectionError("Milvus not connected.")
        if not MILVUS_AVAILABLE: raise RuntimeError("Milvus client not available.")
        try:
            # Use a large limit, or paginate if necessary for huge collections
            # Query with an expression that matches all entities, asking only for 'id' field
            results = await self._run_sync(collection.query, expr="id != ''", output_fields=["id"], limit=50000,
                                           consistency_level="Strong")
            return [str(hit['id']) for hit in results if 'id' in hit]
        except Exception as e:
            print(f"MilvusHandler: Error getting all unit IDs: {e}")
            return []

    async def search(self, collection : Any, vectors: List[List[float]], expr: Optional[str], top_k: int,
                     output_fields: Optional[List[str]] = None, search_params: Optional[Dict[str, Any]] = None) -> List[
        Dict[str, Any]]:
        """Performs vector search on Milvus asynchronously."""
        if not self._connected: raise ConnectionError("Milvus not connected.")
        if not MILVUS_AVAILABLE: raise RuntimeError("Milvus client not available.")
        if not vectors: return []
        default_search_params = {
            "metric_type": "IP",
            "params": {"nprobe": 10,
                       "radius": 0.8,
                       "range_filter": 1.0},
        }
        final_search_params = search_params if search_params is not None else default_search_params

        default_output_fields = ["id",  "content", "creation_time", "end_time", "source", "metadata",
                                 "last_visit", "visit_count", "never_delete",  "rank"]
        fields_to_get = output_fields if output_fields is not None else default_output_fields

        try:
            results = await self._run_sync(collection.search,
                                           data=vectors,
                                           anns_field="embedding",
                                           param=final_search_params,
                                           limit=top_k,
                                           expr=expr,
                                           output_fields=fields_to_get,
                                           consistency_level="Strong")

            all_hits = []
            if results:
                for hit_list in results:
                    for hit in hit_list:
                        hit_dict = {
                            "id": str(hit.id),
                            "distance": hit.distance,
                        }
                        if hit.entity:
                            for k in output_fields:
                                hit_dict[k] = hit.entity.get(k)
                        all_hits.append(hit_dict)

            return all_hits
        except Exception as e:
            print(f"MilvusHandler: Error during search: {e}")
            return []

    async def flush(self, collection: Any):
        """Flushes Milvus collection asynchronously."""
        if not self._connected: raise ConnectionError("Milvus not connected.")
        if not MILVUS_AVAILABLE: raise RuntimeError("Milvus client not available.")
        try:
            await self._run_sync(collection.flush)
            # print("MilvusHandler: Any flushed.")
        except Exception as e:
            print(f"MilvusHandler: Error during flush: {e}")
            # raise

    async def delete_collection(self, collection_name: str):
        """Deletes a Milvus collection asynchronously."""
        if not self._connected: raise ConnectionError("Milvus not connected.")
        if not MILVUS_AVAILABLE: raise RuntimeError("Milvus client not available.")
        try:
            await self._run_sync(utility.drop_collection, collection_name, using=self._conn_alias)
            print(f"MilvusHandler: Successfully deleted collection '{collection_name}'.")
        except Exception as e:
            print(f"MilvusHandler: Error deleting collection '{collection_name}': {e}")
            raise  # Propagate error

    async def close(self, collection: Any, flush=True):
        """Closes Milvus connection asynchronously."""
        if self._connected and MILVUS_AVAILABLE:
            try:
                if flush:
                    await self.flush(collection)
                await self._run_sync(collection.release)
                # print(f"MilvusHandler: Released collection {collection.name}.")
            except Exception as e:
                print(f"MilvusHandler: Error releasing collection {collection.name}: {e}")
            try:
                await self._run_sync(connections.disconnect, self._conn_alias)
                self._connected = False
                print(f"MilvusHandler: Disconnected with alias {self._conn_alias}.")
            except Exception as e:
                print(f"MilvusHandler: Error disconnecting with alias {self._conn_alias}: {e}")
        elif self._connected:
            print("Milvus client not available, cannot close connection.")
        else:
            # print("MilvusHandler: Not connected, nothing to close.")
            pass


def _flatten_metadata(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Flattens the 'metadata' dictionary within each item in a list of data dictionaries.
    Keys from the nested 'metadata' dictionary are brought up to the top level.
    Assumes the input data dicts conform to the structure expected from MemoryUnit.to_dict.
    """
    # NOTE: the key in metadata should not conflict with key in memory_unit!!
    flattened_item = data.copy()
    nested_metadata = flattened_item.pop('metadata', {})
    for key, value in nested_metadata.items():
        if key in flattened_item:
            print(f"Warning: key '{key}' conflicted with the key in memory_unit. Skip.")
            continue
        else:
            flattened_item[key] = value
    return flattened_item


def _reassemble_metadata(result: Dict[str, Any], standard_memory_unit_keys: Set[str]) -> Dict[str, Any]:
    """
    Reassembles flattened metadata keys back into a nested 'metadata' dictionary
    within each result item.
    `standard_memory_unit_keys` are the expected top-level keys of the MemoryUnit
    (excluding the nested 'metadata' key itself).
    """
    core_keys = standard_memory_unit_keys.copy()
    core_keys.update(['id', 'content', 'embedding', 'distance'])
    reassembled_item = {}
    nested_metadata = {}
    for key, value in result.items():
        if key in core_keys:
            reassembled_item[key] = value
        else:
            nested_metadata[key] = value
    reassembled_item['metadata'] = nested_metadata

    return reassembled_item


def _none_to_placeholder(metadata: Dict[str, Any]):
    for key, value in metadata.items():
        metadata[key] = value if value is not None else NONE_PLACE_HOLDER
    return metadata


def _placeholder_to_none(metadata: Dict[str, Any]):
    for key, value in metadata.items():
        metadata[key] = value if value == NONE_PLACE_HOLDER else None
    return metadata


class ChromaHandler(AsyncVectorStoreBackend):
    def __init__(self):
        self._client: Optional[chromadb.Client] = None
        self._connected = False
        self._standard_memory_unit_attrs = {
            'id', 'content', 'creation_time', 'end_time', 'source',
            'last_visit', 'visit_count', 'never_delete', 'rank', 'embedding'
        }

    async def initialize(self, config: Dict[str, Any]):
        """Asynchronously initializes Chroma client."""
        if not CHROMA_AVAILABLE:
            raise RuntimeError("ChromaDB client not available.")

        path = config.get("path")
        host = config.get("host")
        port = config.get("port")
        headers = config.get("headers")
        ssl = config.get("ssl", False)
        grpc_port = config.get("grpc_port")
        in_memory = config.get("in_memory", True)
        try:
            if path:
                self._client = await self._run_sync(chromadb.PersistentClient, path=path)
                print(f"ChromaHandler: Initialized PersistentClient at {path}")
            elif host and port:
                self._client = await self._run_sync(chromadb.HttpClient, host=host, port=port, headers=headers, ssl=ssl,
                                                    grpc_port=grpc_port)
                print(f"ChromaHandler: Initialized HttpClient for {host}:{port}")
            elif in_memory:
                self._client = await self._run_sync(chromadb.Client)
                print("ChromaHandler: Initialized in-memory Client.")
            else:
                raise ValueError("Chroma config requires 'path', 'host' and 'port', or 'in_memory=True'.")

            self._connected = True
        except Exception as e:
            print(f"ChromaHandler: Failed to initialize client: {e}")
            self._client = None
            self._connected = False
            raise ConnectionError(f"Could not initialize Chroma client: {e}") from e

    async def has_collection(self, collection_name: str) -> bool:
        """Checks if a Chroma collection exists asynchronously."""
        if not self._connected or not self._client: raise ConnectionError("Chroma not initialized.")
        if not CHROMA_AVAILABLE: raise RuntimeError("ChromaDB client not available.")
        try:
            await self._run_sync(self._client.get_collection, name=collection_name)
            return True
        except:
            return False

    async def get_or_create_collection(self, collection_name: str, embedding_dim: int,
                                       index_params: Optional[Dict[str, Any]] = None) -> Any:
        """Gets or creates a Chroma collection asynchronously."""
        if not self._connected or not self._client: raise ConnectionError("Chroma not initialized.")
        if not CHROMA_AVAILABLE: raise RuntimeError("ChromaDB client not available.")

        try:
            collection_metadata = index_params if index_params else {}
            collection_metadata["hnsw:space"] = "cosine"
            collection = await self._run_sync(self._client.get_or_create_collection, name=collection_name,
                                              metadata=collection_metadata)
            print(f"ChromaHandler: Got or created collection: {collection_name}")
            return collection
        except Exception as e:
            print(f"ChromaHandler: Failed to get or create collection {collection_name}: {e}")
            raise RuntimeError(f"Failed to get or create Chroma collection {collection_name}: {e}") from e

    async def insert(self, collection: Any, data: List[Dict[str, Any]], flush=False):
        """Inserts data into Chroma collection asynchronously."""
        if not self._connected or not self._client: raise ConnectionError("Chroma not initialized.")
        if not CHROMA_AVAILABLE: raise RuntimeError("ChromaDB client not available.")
        if not data: return
        ids = []
        embeddings = []
        metadatas = []
        documents = []

        for d in data:
            ids.append(d.get('id'))
            embeddings.append(d.get('embedding'))
            metadata = {k: v for k, v in d.items() if k not in ['id', 'embedding', 'content']}
            if 'metadata' in metadata:
                metadata = _flatten_metadata(metadata)
            metadata = _none_to_placeholder(metadata)
            metadatas.append(metadata)
            documents.append(d.get('content', ''))
        valid_entries = [(i, e, m, doc) for i, e, m, doc in zip(ids, embeddings, metadatas, documents) if
                         i is not None and e is not None]
        if not valid_entries: return
        valid_ids, valid_embeddings, valid_metadatas, valid_documents = zip(*valid_entries)
        try:
            await self._run_sync(collection.add,
                                 embeddings=list(valid_embeddings),
                                 metadatas=list(valid_metadatas),
                                 documents=list(valid_documents),
                                 ids=list(valid_ids))
            if flush:
                await self._run_sync(self._client.persist)
            # print(f"ChromaHandler: Inserted {len(valid_ids)} entities.")
        except Exception as e:
            print(f"ChromaHandler: Error during insert: {e}")
            raise

    async def upsert(self, collection: Any, data: List[Dict[str, Any]], flush=False):
        """Upserts data into Chroma collection asynchronously."""
        if not self._connected or not self._client: raise ConnectionError("Chroma not initialized.")
        if not CHROMA_AVAILABLE: raise RuntimeError("ChromaDB client not available.")
        if not data: return

        ids = []
        embeddings = []
        metadatas = []
        documents = []

        for d in data:
            ids.append(d.get('id'))
            embeddings.append(d.get('embedding'))
            metadata = {k: v for k, v in d.items() if k not in ['id', 'embedding', 'content']}
            if 'metadata' in metadata:
                metadata = _flatten_metadata(metadata)
            metadata = _none_to_placeholder(metadata)
            metadatas.append(metadata)
            documents.append(d.get('content', ''))
        valid_entries = [(i, e, m, doc) for i, e, m, doc in zip(ids, embeddings, metadatas, documents) if
                         i is not None and e is not None]
        if not valid_entries: return
        valid_ids, valid_embeddings, valid_metadatas, valid_documents = zip(*valid_entries)
        print(len(valid_embeddings))
        try:
            await self._run_sync(collection.upsert,
                                 embeddings=list(valid_embeddings),
                                 metadatas=list(valid_metadatas),
                                 documents=list(valid_documents),
                                 ids=list(valid_ids))
            if flush:
                await self._run_sync(self._client.persist)
            print(f"ChromaHandler: Upserted {len(valid_ids)} entities.")
        except Exception as e:
            print(f"ChromaHandler: Error during upsert: {e}")
            raise

    async def delete(self, collection: Any, ids: Optional[List[str]] = None, expr=None, flush=False):
        """Deletes data from Chroma collection by ids or expression (metadata filter) asynchronously."""
        if not self._connected or not self._client: raise ConnectionError("Chroma not initialized.")
        if not CHROMA_AVAILABLE: raise RuntimeError("ChromaDB client not available.")
        if not ids and not expr: return  # Nothing to delete

        # Assuming `expr` is already a dictionary compatible with Chroma's `where` filter.
        # Example: expr={'source': 'user', '$and': [{'rank': {'$gte': 0}}, {'rank': {'$lte': 1}}]}
        if expr and not isinstance(expr, dict):
            print(
                f"Error: Chroma delete expression (`expr`) must be a dictionary metadata filter, got {type(expr)}. Deleting by IDs only if provided.")

        delete_ids = ids if ids else None
        delete_where = expr if expr and isinstance(expr, dict) else None

        if not delete_ids and not delete_where:
            print("Warning: Called delete without IDs or a valid dictionary expression.")
            return
        try:
            await self._run_sync(collection.delete, ids=delete_ids, where=delete_where)
            if flush:
                await self._run_sync(self._client.persist)
        except Exception as e:
            print(f"ChromaHandler: Error during delete: {e}")
            raise

    async def count_entities(self, collection: Any, consistently: bool = False) -> int:
        """Counts entities in Chroma collection asynchronously."""
        if not self._connected or not self._client: raise ConnectionError("Chroma not initialized.")
        if not CHROMA_AVAILABLE: raise RuntimeError("ChromaDB client not available.")
        try:
            return await self._run_sync(collection.count)
        except Exception as e:
            print(f"ChromaHandler: Error getting entity count: {e}")
            return 0

    async def query(self, collection: Any, expr, top_k: int,
                    output_fields: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Queries Chroma data by expression (metadata filter) asynchronously."""
        if not self._connected or not self._client: raise ConnectionError("Chroma not initialized.")
        if not CHROMA_AVAILABLE: raise RuntimeError("ChromaDB client not available.")
        if not expr: return []

        # Assuming `expr` is already a dictionary compatible with Chroma's `where` filter.
        # Example: expr={'source': 'user', '$and': [{'rank': {'$gte': 0}}, {'rank': {'$lte': 1}}]}
        if not isinstance(expr, dict):
            print(f"Error: Chroma query requires 'expr' to be a dictionary metadata filter, got {type(expr)}")
            return []
        include_params = []
        fields_to_include_in_metadata = []
        default_output_fields = ["id", "content", "creation_time", "end_time", "source", "metadata",
                                 "last_visit", "visit_count", "never_delete", "rank", "embedding"]
        if output_fields:
            if "*" in output_fields:
                output_fields = default_output_fields
            if "embedding" in output_fields: include_params.append("embeddings")
            if "content" in output_fields: include_params.append("documents")
            fields_to_include_in_metadata = [f for f in output_fields if f not in ["embedding", "content", "id"]]

        if not output_fields:
            if "documents" not in include_params: include_params.append("documents")
            if "metadatas" not in include_params: include_params.append("metadatas")
        elif fields_to_include_in_metadata:
            if "metadatas" not in include_params: include_params.append("metadatas")

        try:
            all_matching_results = await self._run_sync(collection.get,
                                                        where=expr,
                                                        include=include_params if include_params else None)

            processed_results = []
            if all_matching_results and all_matching_results.get('ids'):
                ids = all_matching_results['ids']
                embeddings = all_matching_results.get('embeddings', [])
                metadatas = all_matching_results.get('metadatas', [])
                documents = all_matching_results.get('documents', [])

                for i in range(len(ids)):
                    hit_id = ids[i]
                    hit_dict = {"id": hit_id}

                    if embeddings and i < len(embeddings):
                        hit_dict['embedding'] = embeddings[i]
                    if documents and i < len(documents):
                        hit_dict['content'] = documents[i]
                    if metadatas and i < len(metadatas) and metadatas[i] is not None:
                        item_metadata = metadatas[i]
                        item_metadata = _placeholder_to_none(item_metadata)
                        item_metadata = _reassemble_metadata(item_metadata, self._standard_memory_unit_attrs)
                        if not output_fields:
                            hit_dict.update(item_metadata)
                        else:
                            for field in fields_to_include_in_metadata:
                                if field in item_metadata:
                                    hit_dict[field] = item_metadata[field]
                    processed_results.append(hit_dict)
            return processed_results[:top_k]

        except Exception as e:
            print(f"ChromaHandler: Error during get/query with filter: {e}")
            return []

    async def get(self, collection: Any, id: str, output_fields: Optional[List[str]] = None) -> Optional[
        Dict[str, Any]]:
        """Gets a single entity from Chroma by ID asynchronously."""
        if not self._connected or not self._client: raise ConnectionError("Chroma not initialized.")
        if not CHROMA_AVAILABLE: raise RuntimeError("ChromaDB client not available.")

        include_params = []
        fields_to_include_in_metadata = []

        default_output_fields = ["id", "content", "creation_time", "end_time", "source", "metadata",
                                 "last_visit", "visit_count", "never_delete", "rank", "embedding"]

        if output_fields:
            if "*" in output_fields:
                output_fields = default_output_fields
            if "embedding" in output_fields: include_params.append("embeddings")
            if "content" in output_fields: include_params.append("documents")
            fields_to_include_in_metadata = [f for f in output_fields if f not in ["embedding", "content", "id"]]

        if not output_fields:
            if "documents" not in include_params: include_params.append("documents")
            if "metadatas" not in include_params: include_params.append("metadatas")
        elif fields_to_include_in_metadata:
            if "metadatas" not in include_params: include_params.append("metadatas")

        try:
            result = await self._run_sync(collection.get,
                                          ids=[id],
                                          include=include_params if include_params else None
                                          )

            if result and result.get('ids'):
                hit_id = result['ids'][0]
                hit_dict = {"id": hit_id}

                if result.get('embeddings') and result['embeddings'][0] is not None:
                    hit_dict['embedding'] = result['embeddings'][0]
                if result.get('documents') and result['documents'][0] is not None:
                    hit_dict['content'] = result['documents'][0]
                if result.get('metadatas') and result['metadatas'][0] is not None:
                    item_metadata = result['metadatas'][0]
                    item_metadata = _placeholder_to_none(item_metadata)
                    item_metadata = _reassemble_metadata(item_metadata, self._standard_memory_unit_attrs)
                    if not output_fields:
                        hit_dict.update(item_metadata)
                    else:
                        for field in fields_to_include_in_metadata:
                            if field in item_metadata:
                                hit_dict[field] = item_metadata[field]
                return hit_dict

            return None
        except Exception as e:
            print(f"ChromaHandler: Error during get by ID {id}: {e}")
            return None

    async def get_all_unit_ids(self, collection: Any) -> List[str]:
        """Gets all unit IDs from Chroma collection asynchronously."""
        if not self._connected or not self._client: raise ConnectionError("Chroma not initialized.")
        if not CHROMA_AVAILABLE: raise RuntimeError("ChromaDB client not available.")
        try:
            # Chroma get() with no IDs and include=[] gets all IDs by default
            results = await self._run_sync(collection.get, include=[])  # include=[] gets only IDs
            return results.get('ids', []) if results and isinstance(results, dict) else []  # Ensure result is a dict
        except Exception as e:
            print(f"ChromaHandler: Error getting all unit IDs: {e}")
            return []

    async def search(self, collection: Any, vectors: List[List[float]], expr, top_k: int,
                     output_fields: Optional[List[str]] = None, search_params: Optional[Dict[str, Any]] = None) -> List[
        Dict[str, Any]]:
        """Performs vector search on Chroma asynchronously."""
        if not self._connected or not self._client: raise ConnectionError("Chroma not initialized.")
        if not CHROMA_AVAILABLE: raise RuntimeError("ChromaDB client not available.")
        if not vectors: return []
        if len(vectors) > 1:
            print("Warning: Chroma search currently only fully supports searching with a single query vector.")
        query_vector = vectors[0]
        if expr and not isinstance(expr, dict):
            print(
                f"Error: Chroma search expression (`expr`) must be a dictionary metadata filter, got {type(expr)}. Proceeding without filter.")
            filter_where = None
        else:
            filter_where = expr

        additional_where = search_params.get("where") if search_params else None
        final_where = filter_where
        if additional_where:
            if final_where:
                final_where = {"$and": [final_where, additional_where]}
            else:
                final_where = additional_where

        where_document = search_params.get("where_document", None) if search_params else None

        include_params = []
        fields_to_include_in_metadata = []

        default_output_fields = ["id", "content", "creation_time", "end_time", "source", "metadata",
                                 "last_visit", "visit_count", "never_delete", "rank", "embedding"]
        if output_fields:
            if "*" in output_fields:
                output_fields = default_output_fields
            if "embedding" in output_fields: include_params.append("embeddings")
            if "content" in output_fields: include_params.append("documents")
            fields_to_include_in_metadata = [f for f in output_fields if f not in ["embedding", "content", "id"]]

        # If no specific output_fields are requested, include documents and metadatas by default
        if not output_fields:
            if "documents" not in include_params: include_params.append("documents")
            if "metadatas" not in include_params: include_params.append("metadatas")
            if "distances" not in include_params: include_params.append(
                "distances")
        elif fields_to_include_in_metadata:
            if "metadatas" not in include_params: include_params.append("metadatas")
            if "distances" not in include_params: include_params.append("distances")
        else:
            if "distances" not in include_params: include_params.append("distances")

        additional_include = search_params.get("include", []) if search_params else []
        for inc in additional_include:
            if inc not in include_params:
                include_params.append(inc)

        search_range = search_params.get("search_range")
        min_sim, max_sim = None, None
        if isinstance(search_range, tuple) and len(search_range) == 2:
            min_sim, max_sim = search_range

        try:
            fetch_limit = top_k
            if search_range is not None:
                fetch_limit = max(top_k, 20)
            results = await self._run_sync(collection.query,
                                           query_embeddings=[query_vector] if query_vector is not None else None,
                                           query_texts=None,
                                           n_results=fetch_limit,
                                           where=final_where,
                                           where_document=where_document,
                                           include=include_params if include_params else None
                                           )
            all_hits_from_query = []
            # print(type(results))
            # print(results)
            if results and results.get('ids'):
                ids = results['ids'][0]
                embeddings = results.get('embeddings', [[]])[0]
                metadatas = results.get('metadatas', [[]])[0]
                documents = results.get('documents', [[]])[0]
                distances = results.get('distances', [[]])[0]

                for i in range(len(ids)):
                    hit_id = ids[i]
                    hit_dict = {"id": hit_id}
                    distance = distances[i] if distances is not None and i < len(distances) and distances[
                        i] is not None else None
                    score = 1.0 - distance if distance is not None is not None else 0.0
                    hit_dict['distance'] = 1 - distance
                    if embeddings is not None and i < len(embeddings) and embeddings[i] is not None:
                        hit_dict['embedding'] = embeddings[i]
                    if documents and i < len(documents) and documents[i] is not None:
                        hit_dict['content'] = documents[i]
                    if metadatas and i < len(metadatas) and metadatas[i] is not None:
                        item_metadata = metadatas[i]
                        item_metadata_reassembled = _reassemble_metadata(item_metadata, self._standard_memory_unit_attrs)
                        if not output_fields:
                            hit_dict.update(item_metadata_reassembled)
                        else:
                            if 'metadata' in output_fields:
                                hit_dict['metadata'] = item_metadata_reassembled.get('metadata', {})
                            all_possible_metadata_keys = set(
                                item_metadata_reassembled.keys()) - self._standard_memory_unit_attrs.union(
                                {'id', 'content', 'embedding', 'distance', 'score'})
                            standard_metadata_fields = self._standard_memory_unit_attrs - {'id', 'content', 'embedding',
                                                                                           'metadata'}
                            for field in output_fields:
                                if field in ['id', 'embedding', 'content']: continue
                                if field in standard_metadata_fields and field in item_metadata_reassembled:
                                    hit_dict[field] = item_metadata_reassembled[field]
                                elif field in all_possible_metadata_keys and field in item_metadata_reassembled:
                                    hit_dict[field] = item_metadata_reassembled[field]
                                elif field == 'score':
                                    hit_dict['score'] = score
                                else:
                                    print(f"Warning: field {field} is not in the database. Skip.")

                    all_hits_from_query.append(hit_dict)
            filtered_hits = []
            for hit in all_hits_from_query:
                score = hit.get('distance', 0.0)
                passes_range_filter = True
                if (min_sim is not None and score < min_sim) or \
                        (max_sim is not None and score > max_sim):
                    passes_range_filter = False
                if passes_range_filter:
                    filtered_hits.append(hit)
            filtered_hits.sort(key=lambda x: x.get('distance', -float('inf')) if 'distance' in x else -float('inf'),
                               reverse=True)
            return filtered_hits[:top_k]
        except Exception as e:
            print(f"ChromaHandler: Error during search: {e}")
            return []

    async def flush(self, collection: Any):
        """Flushes Chroma collection asynchronously."""
        if not self._connected or not self._client: raise ConnectionError("Chroma not initialized.")
        if not CHROMA_AVAILABLE: raise RuntimeError("ChromaDB client not available.")
        if hasattr(self._client, 'persist'):
            try:
                await self._run_sync(self._client.persist)
                # print("ChromaHandler: Client persisted.")
            except Exception as e:
                print(f"ChromaHandler: Error during persist: {e}")
                # raise

    async def delete_collection(self, collection_name: str):
        """Deletes a Chroma collection asynchronously."""
        if not self._connected or not self._client: raise ConnectionError("Chroma not initialized.")
        if not CHROMA_AVAILABLE: raise RuntimeError("ChromaDB client not available.")
        try:
            await self._run_sync(self._client.delete_collection, name=collection_name)
            print(f"ChromaHandler: Successfully deleted collection '{collection_name}'.")
        except Exception as e:  # Catch specific Chroma exceptions if known, e.g., if collection doesn't exist.
            print(f"ChromaHandler: Error deleting collection '{collection_name}': {e}")
            raise  # Propagate error

    async def close(self, collection: Any, flush=True):
        """Closes Chroma connection/releases resources asynchronously."""
        if self._connected and self._client and CHROMA_AVAILABLE:
            try:
                await self._run_sync(self._client.PersistentClient)
                # print("ChromaHandler: Client persisted on close.")
            except Exception as e:
                print(f"ChromaHandler: Error during persist on close: {e}")
            self._client = None
            self._connected = False
            print("ChromaHandler: Client connection notionally closed.")
        elif self._connected:
            print("ChromaDB client not available, cannot close.")
        else:
            pass


class QdrantHandler(AsyncVectorStoreBackend):
    def __init__(self):
        self._client: Optional[QdrantClient] = None
        self._connected = False
        self._standard_memory_unit_attrs = {
            'id', 'content', 'creation_time', 'end_time', 'source',
            'last_visit', 'visit_count', 'never_delete', 'rank', 'embedding'
        }

    async def initialize(self, config: Dict[str, Any]):
        """Asynchronously initializes Qdrant client."""
        if not QDRANT_AVAILABLE:
            raise RuntimeError("Qdrant client not available.")

        self._query_vector_name = config.get("vector_name")
        if self._query_vector_name == "":
            self._query_vector_name = None

        path = config.get("path")
        location = config.get("location", ':memory:')  # Can be 'localhost:6333' or ':memory:'
        url = config.get("url")
        api_key = config.get("api_key")
        prefer_grpc = config.get("prefer_grpc", False)
        timeout = config.get("timeout")
        host = config.get("host")
        port = config.get("port")
        try:
            if path:
                self._client = await self._run_sync(QdrantClient, path=path, prefer_grpc=prefer_grpc, timeout=timeout)
                print(f"QdrantHandler: Initialized PersistentClient at {path}")
            elif location:
                self._client = await self._run_sync(QdrantClient, location=location, api_key=api_key,
                                                    prefer_grpc=prefer_grpc, timeout=timeout)
                print(f"QdrantHandler: Initialized Client with location: {location}")
            elif url:
                self._client = await self._run_sync(QdrantClient, url=url, api_key=api_key, prefer_grpc=prefer_grpc,
                                                    timeout=timeout)
                print(f"QdrantHandler: Initialized Client with url: {url}")
            elif host and port:
                self._client = await self._run_sync(QdrantClient, host=host, port=port, api_key=api_key,
                                                    prefer_grpc=prefer_grpc, timeout=timeout)
                print(f"QdrantHandler: Initialized Client with host:port {host}:{port}")
            else:
                raise ValueError("Qdrant config requires 'path', 'location', 'url', or 'host' and 'port'.")

            await self._run_sync(self._client.get_collections)
            self._connected = True
            if self._query_vector_name:
                print(f"QdrantHandler: Configured to use vector name '{self._query_vector_name}' for queries.")
            else:
                print("QdrantHandler: Configured to use default unnamed vector for queries.")

        except Exception as e:
            print(f"QdrantHandler: Failed to initialize client: {e}")
            self._client = None
            self._connected = False
            raise ConnectionError(f"Could not initialize Qdrant client: {e}") from e

    async def has_collection(self, collection_name: str) -> bool:
        """Checks if a Qdrant collection exists asynchronously."""
        if not self._connected or not self._client: raise ConnectionError("Qdrant not initialized.")
        if not QDRANT_AVAILABLE: raise RuntimeError("Qdrant client not available.")
        try:
            return await self._run_sync(self._client.collection_exists, collection_name=collection_name)
        except Exception as e:
            print(f"QdrantHandler: Error checking collection existence: {e}")
            return False

    async def get_or_create_collection(self, collection_name: str, embedding_dim: int,
                                       index_params: Optional[Dict[str, Any]] = None) -> str:
        """Gets or creates a Qdrant collection asynchronously."""
        if not self._connected or not self._client: raise ConnectionError("Qdrant not initialized.")
        if not QDRANT_AVAILABLE: raise RuntimeError("Qdrant client not available.")

        vector_params = VectorParams(size=embedding_dim, distance=Distance.COSINE)
        if index_params and index_params.get("metric_type"):
            metric_type = str(index_params["metric_type"]).upper()
            if metric_type == "L2":
                vector_params.distance = Distance.EUCLID

        try:
            if not await self.has_collection(collection_name):
                print(f"QdrantHandler: Creating new collection: {collection_name}")
                await self._run_sync(self._client.create_collection,
                                     collection_name=collection_name,
                                     vectors_config=vector_params)
                await self._run_sync(self._client.create_payload_index,
                                     collection_name=collection_name,
                                     field_name="source",
                                     field_schema=models.PayloadSchemaType.KEYWORD)
                await self._run_sync(self._client.create_payload_index,
                                     collection_name=collection_name,
                                     field_name="rank",
                                     field_schema=models.PayloadSchemaType.INTEGER)
                await self._run_sync(self._client.create_payload_index,
                                     collection_name=collection_name,
                                     field_name="creation_time",
                                     field_schema=models.PayloadSchemaType.FLOAT)
                await self._run_sync(self._client.create_payload_index,
                                     collection_name=collection_name,
                                     field_name="end_time",
                                     field_schema=models.PayloadSchemaType.FLOAT)
                await self._run_sync(self._client.create_payload_index,
                                     collection_name=collection_name,
                                     field_name="last_visit",
                                     field_schema=models.PayloadSchemaType.INTEGER)
                await self._run_sync(self._client.create_payload_index,
                                     collection_name=collection_name,
                                     field_name="visit_count",
                                     field_schema=models.PayloadSchemaType.INTEGER)
                await self._run_sync(self._client.create_payload_index,
                                     collection_name=collection_name,
                                     field_name="never_delete",
                                     field_schema=models.PayloadSchemaType.BOOL)
            print(f"QdrantHandler: Got or created collection: {collection_name}")
            return collection_name
        except Exception as e:
            print(f"QdrantHandler: Failed to get or create collection {collection_name}: {e}")
            raise RuntimeError(f"Failed to get or create Qdrant collection {collection_name}: {e}") from e

    async def insert(self, collection_name: str, data: List[Dict[str, Any]], flush=False):
        """Inserts data into Qdrant collection asynchronously."""
        await self.upsert(collection_name, data, flush=flush)

    async def upsert(self, collection_name: str, data: List[Dict[str, Any]], flush=False):
        """Upserts data into Qdrant collection asynchronously."""
        if not self._connected or not self._client: raise ConnectionError("Qdrant not initialized.")
        if not QDRANT_AVAILABLE: raise RuntimeError("Qdrant client not available.")
        if not data: return

        points_to_upsert: List[PointStruct] = []
        for d in data:
            try:
                point_id = str(d.get('id'))
                vector = d.get('embedding')
                if not point_id or vector is None:
                    print(f"Warning: Skipping upsert for item with missing ID or embedding: {d.get('id')}")
                    continue

                payload = {k: v for k, v in d.items() if k not in ['id', 'embedding']}

                if 'creation_time' in payload:
                    if isinstance(payload['creation_time'], datetime):
                        payload['creation_time'] = payload['creation_time'].timestamp()
                    elif isinstance(payload['creation_time'], str):
                        try:
                            payload['creation_time'] = datetime.fromisoformat(payload['creation_time']).timestamp()
                        except:
                            payload['creation_time'] = None
                    elif not isinstance(payload['creation_time'], (int, float, type(None))):
                        print(
                            f"Warning: creation_time for {point_id} is not a datetime, string, number or None. Storing as is.")

                if 'end_time' in payload:
                    if isinstance(payload['end_time'], datetime):
                        payload['end_time'] = payload['end_time'].timestamp()
                    elif isinstance(payload['end_time'], str):
                        try:
                            payload['end_time'] = datetime.fromisoformat(payload['end_time']).timestamp()
                        except:
                            payload['end_time'] = None  # Fail gracefully
                    elif not isinstance(payload['end_time'], (int, float, type(None))):
                        print(
                            f"Warning: end_time for {point_id} is not a datetime, string, number or None. Storing as is.")
                if isinstance(vector, np.ndarray):
                    vector = vector.flatten().tolist()
                points_to_upsert.append(
                    PointStruct(
                        id=point_id,
                        vector=vector,
                        payload=payload
                    )
                )
            except Exception as e:
                print(f"QdrantHandler: Error preparing point for upsert (ID: {d.get('id')}): {e}")

        if not points_to_upsert: return

        try:

            response = await self._run_sync(self._client.upsert,
                                            collection_name=collection_name,
                                            wait=flush,
                                            points=points_to_upsert)
            print(f"QdrantHandler: Upserted {len(points_to_upsert)} points. Status: {response.status}")
        except Exception as e:
            print(f"QdrantHandler: Error during upsert: {e}")
            raise

    async def delete(self, collection_name: str, ids: Optional[List[str]] = None, expr: Optional[Filter] = None,
                     flush=False):
        """Deletes data from Qdrant collection by ids or expression (filter) asynchronously."""
        if not self._connected or not self._client: raise ConnectionError("Qdrant not initialized.")
        if not QDRANT_AVAILABLE: raise RuntimeError("Qdrant client not available.")
        if not ids and not expr: return

        try:
            if ids:
                delete_selector = FilterSelector(filter=Filter(must=[HasIdCondition(has_id=[str(uid) for uid in ids])]))
            elif expr:
                if not isinstance(expr, Filter):
                    print(
                        f"Error: Qdrant delete expression (`expr`) must be a Qdrant Filter object, got {type(expr)}. Deleting by IDs only if provided.")
                    return
                delete_selector = FilterSelector(filter=expr)
            else:
                print("Warning: Called delete without IDs or a valid Qdrant Filter expression.")
                return

            response = await self._run_sync(self._client.delete,
                                            collection_name=collection_name,
                                            points_selector=delete_selector,
                                            wait=flush
                                            )
            # print(f"QdrantHandler: Deleted points. Status: {response.status}")

        except Exception as e:
            print(f"QdrantHandler: Error during delete: {e}")
            raise

    async def count_entities(self, collection_name: str, consistently: bool = False) -> int:
        """Counts entities in Qdrant collection asynchronously."""
        if not self._connected or not self._client: raise ConnectionError("Qdrant not initialized.")
        if not QDRANT_AVAILABLE: raise RuntimeError("Qdrant client not available.")
        try:
            count_result = await self._run_sync(self._client.count,
                                                collection_name=collection_name,
                                                exact=consistently)
            return count_result.count
        except Exception as e:
            print(f"QdrantHandler: Error getting entity count: {e}")
            return 0

    async def query(self, collection_name: str, expr: Optional[Filter], top_k: int,
                    output_fields: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Queries Qdrant data by expression (filter), without vector search."""
        if not self._connected or not self._client: raise ConnectionError("Qdrant not initialized.")
        if not QDRANT_AVAILABLE: raise RuntimeError("Qdrant client not available.")
        if expr is None:
            pass
        if expr is not None and not isinstance(expr, Filter):
            print(f"Error: Qdrant query requires 'expr' to be a Qdrant Filter object, got {type(expr)}")
            return []

        all_results: List[Dict[str, Any]] = []
        scroll_page_size = min(top_k, 1000)
        current_offset = None

        fetch_all_payload = output_fields is None or "*" in output_fields
        with_payload = True if fetch_all_payload else models.PayloadSelectorInclude(include=output_fields)
        with_vectors = not fetch_all_payload and output_fields and "embedding" in output_fields

        if not fetch_all_payload and output_fields:
            payload_fields_to_include = [f for f in output_fields if f not in ["id", "embedding", "distance", "score"]]
            with_payload = models.PayloadSelectorInclude(
                include=payload_fields_to_include) if payload_fields_to_include else False

        lookup_params = models.ReadConsistency.MAJORITY
        while len(all_results) < top_k:
            try:
                points, next_page_offset = await self._run_sync(self._client.scroll,
                                                                collection_name=collection_name,
                                                                filter=expr,  # Pass the Qdrant Filter object
                                                                limit=min(scroll_page_size, top_k - len(all_results)),
                                                                # Fetch up to remaining needed
                                                                offset=current_offset,
                                                                with_payload=with_payload,
                                                                with_vectors=with_vectors,
                                                                read_consistency=lookup_params)
                if not points:
                    break
                for point in points:
                    if len(all_results) >= top_k: break
                    item_dict = {"id": str(point.id), "score":  0.0}
                    if point.payload:
                        item_dict.update(point.payload)
                    if point.vector is not None:
                        item_dict['embedding'] = point.vector
                    all_results.append(item_dict)

                current_offset = next_page_offset
                if current_offset is None:
                    break
            except Exception as e:
                print(f"QdrantHandler: Error during query (scroll): {e}")
                break
        return all_results[:top_k]

    async def get(self, collection_name: str, id: str, output_fields: Optional[List[str]] = None) -> Optional[
        Dict[str, Any]]:
        """Gets a single entity from Qdrant by ID asynchronously."""
        if not self._connected or not self._client: raise ConnectionError("Qdrant not initialized.")
        if not QDRANT_AVAILABLE: raise RuntimeError("Qdrant client not available.")

        with_payload = True if output_fields is None or "*" in output_fields else models.PayloadSelectorInclude(
            include=[f for f in output_fields if f != 'id'])
        with_vectors = True if output_fields is None or "*" in output_fields or "embedding" in output_fields else False

        if with_vectors and isinstance(with_payload,
                                       models.PayloadSelectorInclude) and "embedding" in with_payload.include:
            with_payload.include.remove("embedding")

        try:
            points = await self._run_sync(self._client.retrieve,
                                          collection_name=collection_name,
                                          ids=[str(id)],
                                          with_payload=with_payload,
                                          with_vectors=with_vectors)

            if points and len(points) > 0:
                hit_data = points[0]
                item_dict = {"id": str(hit_data.id),
                             "score": 0.0}
                if hit_data.payload:
                    item_dict.update(hit_data.payload)
                if hit_data.vector:
                    item_dict['embedding'] = hit_data.vector
                return item_dict

            return None
        except Exception as e:
            print(f"QdrantHandler: Error during get by ID {id}: {e}")
            return None

    async def get_all_unit_ids(self, collection_name: str) -> List[str]:
        """Gets all unit IDs from Qdrant collection asynchronously using scroll pagination."""
        if not self._connected or not self._client: raise ConnectionError("Qdrant not initialized.")
        if not QDRANT_AVAILABLE: raise RuntimeError("Qdrant client not available.")
        try:
            all_ids = []
            scroll_page_size = 1000
            current_offset = None

            while True:
                scroll_result: ScrollResult = await self._run_sync(self._client.scroll,
                                                                   collection_name=collection_name,
                                                                   limit=scroll_page_size,
                                                                   offset=current_offset,
                                                                   with_payload=False,
                                                                   with_vectors=False)

                points = scroll_result[0]
                next_page_offset = scroll_result[1]
                if points:
                    all_ids.extend([str(p.id) for p in points])
                if next_page_offset is None:
                    break
                current_offset = next_page_offset

            return all_ids
        except Exception as e:
            print(f"QdrantHandler: Error getting all unit IDs: {e}")
            return []

    async def search(self, collection_name: str, vectors: List[List[float]], expr: Optional[Filter], top_k: int,
                     output_fields: Optional[List[str]] = None, search_params: Optional[Dict[str, Any]] = None) -> List[
        Dict[str, Any]]:
        """Performs vector search on Qdrant asynchronously."""
        if not self._connected or not self._client: raise ConnectionError("Qdrant not initialized.")
        if not QDRANT_AVAILABLE: raise RuntimeError("Qdrant client not available.")
        if not vectors: return []
        if len(vectors) > 1:
            print("Warning: Qdrant search currently only fully supports searching with a single query vector.")

        query_vector = vectors[0]
        if isinstance(query_vector, np.ndarray):
            # print(query_vector.shape)
            query_vector = query_vector.flatten().tolist()
            # print(len(query_vector))

        # if self._query_vector_name:
        #     query_vector = (self._query_vector_name, query_vector)
        # else:
        #     pass

        if expr is not None and not isinstance(expr, Filter):
            print(
                f"Error: Qdrant search expression (`expr`) must be a Qdrant Filter object, got {type(expr)}. Proceeding without filter.")
            filter_arg = None
        else:
            filter_arg = expr

        search_range: Optional[Tuple[Optional[float], Optional[float]]] = search_params.get(
            'search_range') if search_params else None
        min_sim, max_sim = None, None
        if isinstance(search_range, tuple) and len(search_range) == 2:
            min_sim, max_sim = search_range

        effective_search_params = models.SearchParams(hnsw_ef=64)
        if search_params:
            params_for_qdrant = {k: v for k, v in search_params.items() if k != 'search_range'}
            try:
                effective_search_params = models.SearchParams(**params_for_qdrant)
            except Exception as e:
                # print(
                #     f"Warning: Failed to create Qdrant SearchParams from {params_for_qdrant}: {e}. Using default params.")
                effective_search_params = models.SearchParams(hnsw_ef=128)

        score_threshold_arg = min_sim if min_sim is not None and isinstance(min_sim, (int, float)) else None
        fetch_limit = top_k
        perform_post_filtering = False
        if max_sim is not None and isinstance(max_sim, (int, float)):
            perform_post_filtering = True
            fetch_limit = int(top_k * 2.5)

        with_payload = True if output_fields is None or "*" in output_fields else models.PayloadSelectorInclude(
            include=[f for f in output_fields if f != 'id'])
        with_vectors = True if output_fields is None or "*" in output_fields or "embedding" in output_fields else False

        if with_vectors and isinstance(with_payload,
                                       models.PayloadSelectorInclude) and "embedding" in with_payload.include:
            with_payload.include.remove("embedding")
        # try:
        search_result: List[ScoredPoint] = await self._run_sync(self._client.search,
                                                                collection_name=collection_name,
                                                                query_vector=query_vector,
                                                                query_filter=filter_arg,
                                                                limit=fetch_limit,
                                                                with_payload=with_payload,
                                                                with_vectors=with_vectors,
                                                                search_params=effective_search_params,
                                                                score_threshold=score_threshold_arg
                                                                )

        all_hits = []
        for hit in search_result:
            item_dict = {"id": str(hit.id), "distance": float(hit.score)}
            if hit.payload:
                item_dict.update(hit.payload)
            if hit.vector:
                item_dict['embedding'] = hit.vector
            all_hits.append(item_dict)

        if perform_post_filtering and max_sim is not None:
            filtered_hits = [hit for hit in all_hits if hit.get('distance', -float('inf')) <= max_sim]
        else:
            filtered_hits = all_hits
        filtered_hits.sort(key=lambda x: x.get('distance', -float('inf')), reverse=True)
        return filtered_hits[:top_k]
        # except Exception as e:
        #     print(f"QdrantHandler: Error during search: {e}")
        #     return []

    async def flush(self, collection_name: str):
        """Explicitly flushes data to persistent storage (not typically needed with async client ops)."""
        print(
            f"QdrantHandler: Explicit flush called for {collection_name}. Qdrant handles persistence automatically or via `wait=True`.")
        pass

    async def delete_collection(self, collection_name: str):
        """Deletes a Qdrant collection asynchronously."""
        if not self._connected or not self._client: raise ConnectionError("Qdrant not initialized.")
        if not QDRANT_AVAILABLE: raise RuntimeError("Qdrant client not available.")
        try:
            if await self.has_collection(collection_name):
                await self._run_sync(self._client.delete_collection, collection_name=collection_name)
                print(f"QdrantHandler: Successfully deleted collection '{collection_name}'.")
            else:
                print(f"QdrantHandler: Any '{collection_name}' does not exist, skipping deletion.")
        except Exception as e:
            print(f"QdrantHandler: Error deleting collection '{collection_name}': {e}")
            raise  # Propagate error

    async def close(self, collection_name: str, flush=True):
        """Closes Qdrant connection/releases resources asynchronously."""
        if self._connected and self._client and QDRANT_AVAILABLE:
            # if flush:
            # print("QdrantHandler: Flush requested on close. Note: Qdrant persists automatically.")\
            # print("QdrantHandler: Async client typically manages connection internally. No explicit close call made.")
            self._client = None
            self._connected = False
        elif self._connected:
            print("Qdrant client not available, cannot close.")
        else:
            pass  # Not connected


# --- Main Vector Store Handler (Factory/Dispatcher/SqliteVec Wrapper) ---
class VectorStoreHandler:
    """
    Initializes the VectorStoreHandler with a specific vector database backend.
    """
    def __init__(self, chatbot_id: str, long_term_memory_id: str, config: Dict[str, Any]):
        self.chatbot_id = chatbot_id.replace('-', '_')
        self.long_term_memory_id = long_term_memory_id.replace('-', '_')
        self.config = config
        self.vector_db_type = config.get("type", "qdrant").lower()
        self.embedding_dim = config.get("embedding_dim")

        self._sqlite_handler: Optional[AsyncSQLiteHandler] = None

        if not self.embedding_dim or not isinstance(self.embedding_dim, int) or self.embedding_dim <= 0:
            raise ValueError("Vector store configuration must include a valid positive integer 'embedding_dim'.")

        self._backend: Optional[AsyncVectorStoreBackend] = None
        self._collection: Optional[Any] = None

        self.collection_name = f"chatbot_{self.chatbot_id}_ltm_{self.long_term_memory_id}"
        if self.vector_db_type == "milvus" or self.vector_db_type == "milvus-lite":
            if MILVUS_AVAILABLE:
                self._backend = MilvusHandler()
            else:
                raise RuntimeError(
                    f"Milvus client is not installed. Cannot use vector_db_type '{self.vector_db_type}'.")
        elif self.vector_db_type == "chroma":
            if CHROMA_AVAILABLE:
                self._backend = ChromaHandler()
            else:
                raise RuntimeError(
                    f"ChromaDB client is not installed. Cannot use vector_db_type '{self.vector_db_type}'.")
        elif self.vector_db_type == "qdrant":
            if QDRANT_AVAILABLE:
                self._backend = QdrantHandler()
            else:
                raise RuntimeError(
                    f"Qdrant client is not installed. Cannot use vector_db_type '{self.vector_db_type}'.")
        elif self.vector_db_type == "sqlite-vec":
            if not SQLITE_VEC_AVAILABLE:
                print("Warning: sqlite-vec requested but Python package not available.")
            self._sqlite_handler = AsyncSQLiteHandler()
            self._backend = None
            self._collection = None
            print("VectorStoreHandler configured for potential sqlite-vec delegation.") # noqa

        else:
            raise ValueError(
                f"Unsupported vector_db_type: {self.vector_db_type}. Supported types: 'milvus', 'milvus-lite', 'chroma', 'qdrant', 'sqlite-vec'.")
    async def initialize(self):
        """Initializes the chosen backend or the sqlite_handler for sqlite-vec."""
        if self.vector_db_type in ["milvus", "milvus-lite", "chroma", "qdrant"]:
            if self._backend is None:
                raise RuntimeError("Vector store backend not initialized (this should not happen).")
            backend_config = self.config.get(self.vector_db_type, {})
            await self._backend.initialize(backend_config)
            index_params = self.config.get("index_params")
            self._collection = await self._backend.get_or_create_collection(
                self.collection_name, self.embedding_dim, index_params
            )
            print(f"VectorStoreHandler initialized external backend: {self.vector_db_type}.") # noqa

        elif self.vector_db_type == "sqlite-vec":
            if self._sqlite_handler is None:
                 raise RuntimeError("sqlite-vec chosen, but AsyncSQLiteHandler was not instantiated.")

            sqlite_config = self.config.get('sqlite', {})
            base_path = sqlite_config.get('base_path', self.config.get('base_path', "memory_storage"))
            await self._sqlite_handler.initialize(
                chatbot_id=self.chatbot_id,
                base_path=base_path,
                use_sqlite_vec=True,
                embedding_dim=self.embedding_dim
            )
            await self._sqlite_handler.initialize_db()

            if not self._sqlite_handler.use_sqlite_vec and SQLITE_VEC_AVAILABLE:
                 print("Warning: sqlite-vec was requested, package is available, but AsyncSQLiteHandler did not enable it.") # noqa
            elif not SQLITE_VEC_AVAILABLE:
                 print("Warning: sqlite-vec was requested, but package is not available. Vector operations will fail.") # noqa
            else:
                 print("VectorStoreHandler initialization completed for sqlite-vec.")

    # --- Delegate methods to the backend or sqlite_handler ---
    async def has_collection(self) -> bool:
        """Checks if the managed collection exists (external) or if sqlite-vec is usable."""
        if self.vector_db_type in ["milvus", "milvus-lite", "chroma", "qdrant"]:
            if self._backend is None:
                print("Warning: has_collection called before backend initialization.")  # noqa
                return False
            # Check existence using the specific collection name
            return await self._backend.has_collection(self.collection_name)
        elif self.vector_db_type == "sqlite-vec":
            # For sqlite-vec, "collection" exists if the handler is configured and library is available
            if not self._sqlite_handler:
                return False
            # Check if the VSS table exists (optional, depends on sqlite_handler readiness)
            # A simple check might just be if sqlite-vec is intended to be used
            return self._sqlite_handler.use_sqlite_vec and SQLITE_VEC_AVAILABLE
        else:
            return False  # Should not happen

    async def insert(self, data: List[Dict[str, Any]], flush: bool = False):
        """
        Inserts data asynchronously.
        For sqlite-vec, this delegates to upsert as SQLite's INSERT OR REPLACE is atomic.
        Expects data items to be dictionaries compatible with MemoryUnit.to_dict() plus an 'embedding' key.
        """
        if self.vector_db_type in ["milvus", "milvus-lite", "chroma", "qdrant"]:
            if self._backend is None or self._collection is None:
                raise RuntimeError("External vector store backend or collection not initialized.")  # noqa
            # External stores might need specific data formatting (handled in their handlers)
            await self._backend.insert(self._collection, data, flush=flush)
        elif self.vector_db_type == "sqlite-vec":
            if not self._sqlite_handler or not self._sqlite_handler.use_sqlite_vec:
                print(
                    "Warning: insert called for sqlite-vec, but handler/sqlite-vec not configured. Operation skipped.")  # noqa
                return
            # Delegate to upsert in sqlite_handler, which handles main table and VSS table
            # Ensure data includes 'embedding' if sqlite_vec is used
            await self._sqlite_handler.upsert_memory_units(data)
            # Note: flush is not directly applicable to sqlite; handled by commits

    async def upsert(self, data: List[Dict[str, Any]], flush: bool = False):
        """
        Upserts data asynchronously.
        Expects data items to be dictionaries compatible with MemoryUnit.to_dict() plus an 'embedding' key.
        """
        if self.vector_db_type in ["milvus", "milvus-lite", "chroma", "qdrant"]:
            if self._backend is None or self._collection is None:
                raise RuntimeError("External vector store backend or collection not initialized.")  # noqa
            await self._backend.upsert(self._collection, data, flush=flush)
        elif self.vector_db_type == "sqlite-vec":
            if not self._sqlite_handler:
                print("Warning: upsert called for sqlite-vec, but handler not configured. Operation skipped.")  # noqa
                return
            # Delegate to the sqlite_handler's upsert method
            await self._sqlite_handler.upsert_memory_units(data)
            # Note: flush is not directly applicable to sqlite; handled by commits

    async def delete(self, ids: Optional[List[str]] = None, expr: Optional[Any] = None):
        """
        Deletes data by IDs or expression asynchronously.
        For sqlite-vec, only deletion by ID is directly supported via _sqlite_handler.delete_memory_units.
        The 'expr' parameter is ignored for sqlite-vec in this wrapper.
        """
        if self.vector_db_type in ["milvus", "milvus-lite", "chroma", "qdrant"]:
            if self._backend is None or self._collection is None:
                raise RuntimeError("External vector store backend or collection not initialized.")  # noqa
            # Pass expr directly, backend handler must interpret (str for Milvus, dict for Chroma)
            await self._backend.delete(self._collection, ids=ids, expr=expr)
        elif self.vector_db_type == "sqlite-vec":
            if not self._sqlite_handler:
                print("Warning: delete called for sqlite-vec, but handler not configured. Operation skipped.")  # noqa
                return
            if expr is not None:
                print(
                    "Warning: delete called with 'expr' for sqlite-vec. Expression-based deletion is not supported here; deleting by IDs only.")  # noqa
            if ids:
                await self._sqlite_handler.delete_memory_units(ids)
            # else: No operation if only expr is provided for sqlite-vec

    async def count_entities(self, consistently: bool = False) -> int:
        """
        Counts entities asynchronously.
        For sqlite-vec, counts entries in the VSS table to represent vectored items.
        """
        if self.vector_db_type in ["milvus", "milvus-lite", "chroma", "qdrant"]:
            if self._backend is None or self._collection is None:
                print("Warning: count_entities called before external backend initialization.")  # noqa
                return 0
            return await self._backend.count_entities(self._collection, consistently=consistently)
        elif self.vector_db_type == "sqlite-vec":
            if not self._sqlite_handler or not self._sqlite_handler.use_sqlite_vec or not SQLITE_VEC_AVAILABLE:
                print(
                    "Warning: count_entities called for sqlite-vec, but handler/sqlite-vec not configured/available. Returning 0.")  # noqa
                return 0
            # Count entries specifically in the VSS table as it represents items with vectors
            conn = await self._sqlite_handler._get_connection()
            try:
                # Check if VSS table exists before querying
                async with conn.execute(
                        "SELECT name FROM sqlite_master WHERE type='table' AND name='memory_units_vss'") as cursor:  # noqa Check type='table' for VSS
                    vss_exists = await cursor.fetchone()
                    if not vss_exists:
                        print("Warning: memory_units_vss table does not exist. Cannot count entities.")  # noqa
                        return 0

                async with conn.execute("SELECT COUNT(*) FROM memory_units_vss") as cursor:
                    row = await cursor.fetchone()
                    return row[0] if row else 0
            except Exception as e:
                print(f"Error counting sqlite-vec entities in VSS table: {e}")  # noqa
                return 0
        else:
            return 0  # Should not happen

    async def query(self, expr: Any, top_k: int, output_fields: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Queries data by expression (metadata filter) asynchronously, without vector search.
        For sqlite-vec, this queries the main `memory_units` table.
        'expr' should be a SQL WHERE clause string for sqlite-vec.
        """
        if self.vector_db_type in ["milvus", "milvus-lite", "chroma", "qdrant"]:
            if self._backend is None or self._collection is None:
                print("Warning: query called before external backend initialization. Returning empty.")  # noqa
                return []
            # Backend handlers need to interpret expr (str for Milvus, dict for Chroma)
            return await self._backend.query(self._collection, expr=expr, top_k=top_k, output_fields=output_fields)
        elif self.vector_db_type == "sqlite-vec":
            if not self._sqlite_handler:
                print("Warning: query called for sqlite-vec, but handler not configured. Returning empty.")  # noqa
                return []
            if not isinstance(expr, str):
                print(
                    f"Warning: query called for sqlite-vec, but 'expr' is not a string WHERE clause. Got: {type(expr)}. Returning empty.")  # noqa
                return []

            conn = await self._sqlite_handler._get_connection()
            # Base query selects from the main data table
            sql = f"SELECT * FROM memory_units"
            params: List[Any] = []
            if expr:
                # Append the WHERE clause provided by the user
                # IMPORTANT: User-provided 'expr' must be carefully handled to prevent SQL injection
                # In a real application, parameter binding or strict validation is crucial.
                # Assuming 'expr' is trusted or pre-validated for this context.
                sql += f" WHERE {expr}"  # Use the expression directly

            # Add LIMIT
            sql += f" LIMIT ?"
            params.append(top_k)

            results = []
            try:
                async with conn.execute(sql, tuple(params)) as cursor:
                    rows = await cursor.fetchall()
                    for row in rows:
                        # Use the handler's conversion method
                        unit = self._sqlite_handler._row_to_memory_unit(row,await self._sqlite_handler._get_connection())
                        if unit:
                            unit_dict = unit.to_dict()  # Convert to dict
                            # Apply output_fields filtering if needed
                            if output_fields:
                                filtered_dict = {"id": unit_dict.get("id")}  # Always include id
                                for field in output_fields:
                                    if field == "embedding":
                                        if self._sqlite_handler.use_sqlite_vec and SQLITE_VEC_AVAILABLE:
                                            embedding = await self._sqlite_handler.get_embedding_for_unit(unit.id)
                                            if embedding:
                                                filtered_dict["embedding"] = embedding
                                            else:
                                                print(f"Warning: Embedding not found for unit {unit.id} during query.")
                                        else:
                                            print(
                                                f"Warning: 'embedding' requested but sqlite-vec is not available for unit {unit.id}.")
                                    elif field in unit_dict:
                                        filtered_dict[field] = unit_dict[field]
                            else:
                                results.append(unit_dict)  # Return full dict if no filter
            except Exception as e:
                print(f"Error querying sqlite-vec metadata from memory_units table: {e}")  # noqa
                return []
            return results
        else:
            return []  # Should not happen

    async def get(self, id: str, output_fields: Optional[List[str]] = None) -> Optional[Dict[str, Any]]:
        """
        Gets a single data item by ID asynchronously.
        For sqlite-vec, loads from the main `memory_units` table. Embedding is not included.
        """
        if self.vector_db_type in ["milvus", "milvus-lite", "chroma", "qdrant"]:
            if self._backend is None or self._collection is None:
                print("Warning: get called before external backend initialization. Returning None.")  # noqa
                return None
            return await self._backend.get(self._collection, id=id, output_fields=output_fields)
        elif self.vector_db_type == "sqlite-vec":
            if not self._sqlite_handler:
                print("Warning: get called for sqlite-vec, but handler not configured. Returning None.")  # noqa
                return None
            # Load the memory unit using the handler's method
            unit = await self._sqlite_handler.load_memory_unit(id)
            if unit:
                unit_dict = unit.to_dict()
                if output_fields and "embedding" in output_fields:
                    if self._sqlite_handler.use_sqlite_vec and SQLITE_VEC_AVAILABLE:
                        embedding = await self._sqlite_handler.get_embedding_for_unit(id)
                        if embedding:
                            unit_dict["embedding"] = embedding
                        else:
                            print(f"Warning: Embedding not found for unit {id} despite 'embedding' being requested.")
                    else:
                        print(
                            f"Warning: 'embedding' requested for unit {id}, but sqlite-vec is not enabled or available.")
                if output_fields:
                    filtered_dict = {f: unit_dict.get(f) for f in output_fields if f in unit_dict}  # noqa
                    if "id" not in output_fields and "id" in unit_dict:
                        filtered_dict["id"] = unit_dict["id"]
                    return filtered_dict
                else:
                    return unit_dict  # Return full dict
            return None
        else:
            return None

    async def get_all_unit_ids(self) -> List[str]:
        """Gets all unit IDs in the collection asynchronously."""
        if self.vector_db_type == 'sqlite-vec':
            return await self._sqlite_handler.load_all_memory_unit_ids()
        else:
            return await self._backend.get_all_unit_ids(self._collection)

    async def search(self, vectors: List[List[float]], expr: Optional[Any], top_k: int,
                     output_fields: Optional[List[str]] = None, search_params: Optional[Dict[str, Any]] = None) -> List[
        Dict[str, Any]]:  # noqa
        """
        Performs vector search asynchronously.
        For sqlite-vec, delegates to `_sqlite_handler.search_vectors_sqlite_vec`.
        'expr' should be a SQL WHERE clause string applicable to the 'memory_units' table (aliased as 'mu').
        Returns results including a 'distance' field (similarity score).
        """
        if self.vector_db_type in ["milvus", "milvus-lite", "chroma", "qdrant"]:
            if self._backend is None or self._collection is None:
                print("Warning: search called before external backend initialization. Returning empty.")  # noqa
                return []
            return await self._backend.search(self._collection, vectors=vectors, expr=expr, top_k=top_k,
                                              output_fields=output_fields, search_params=search_params)

        elif self.vector_db_type == "sqlite-vec":
            if not self._sqlite_handler or not self._sqlite_handler.use_sqlite_vec or not SQLITE_VEC_AVAILABLE:
                print(
                    "Warning: search called for sqlite-vec, but handler/sqlite-vec not configured/available. Returning empty.")  # noqa
                return []
            if not vectors:
                print("Warning: search called with no query vectors.")  # noqa
                return []
            if len(vectors) > 1:
                print(
                    "Warning: sqlite-vec search in this handler currently supports only one query vector at a time. Using the first one.")  # noqa

            query_vector = vectors[0]

            sql_expr_fragment: Optional[str] = None
            if expr is not None:
                if isinstance(expr, str):
                    sql_expr_fragment = expr
                else:
                    print(
                        f"Warning: search called for sqlite-vec with non-string 'expr'. Type: {type(expr)}. Ignoring expression.")  # noqa
            current_search_params = search_params if search_params is not None else {}
            search_range = current_search_params.get('search_range')

            search_results_tuples = await self._sqlite_handler.search_vectors_sqlite_vec(
                [query_vector], top_k, expr=sql_expr_fragment, search_range=search_range
            )

            # --- Format results ---
            unit_ids = [uid for uid, score in search_results_tuples]
            if not unit_ids:
                return []

            # Load the full MemoryUnit data for the found IDs
            # This returns a Dict[str, MemoryUnit]
            units_dict = await self._sqlite_handler.load_memory_units(unit_ids)
            embeddings_map: Dict[str, List[float]] = {}
            if output_fields and "embedding" in output_fields:
                if self._sqlite_handler.use_sqlite_vec and SQLITE_VEC_AVAILABLE:
                    embeddings_map = await self._sqlite_handler.get_embeddings_for_units(unit_ids)
                else:
                    print(
                        "Warning: 'embedding' requested in output_fields, but sqlite-vec is not enabled or available.")

            # Combine loaded data with scores and format output
            formatted_results = []
            # Keep the order from the similarity search
            for unit_id, score in search_results_tuples:
                unit = units_dict.get(unit_id)
                if unit:
                    unit_data = unit.to_dict()
                    unit_data['distance'] = score
                    if unit_id in embeddings_map:
                        unit_data["embedding"] = embeddings_map[unit_id]

                    if output_fields:
                        filtered_data = {"id": unit_data.get("id")}  # Always include id
                        if 'distance' in output_fields or 'score' in output_fields:  # Allow 'score' or 'distance' # noqa
                            filtered_data['distance'] = score
                        # Include other requested fields
                        for field in output_fields:
                            if field not in ["id", "distance", "score"] and field in unit_data:  # noqa
                                filtered_data[field] = unit_data[field]
                        formatted_results.append(filtered_data)
                    else:
                        # Return full data + score if no specific fields requested
                        formatted_results.append(unit_data)
                    # else: Warn if VSS found an ID not in the main table?
                    #     print(f"Warning: Unit {unit_id} found in VSS search but not in main table.") # noqa

                return formatted_results

        else:
            return []  # Should not happen

    async def flush(self):
        """
        Flushes data to persistent storage asynchronously.
        Not applicable to sqlite-vec where commits handle persistence.
        """
        if self.vector_db_type in ["milvus", "milvus-lite", "chroma", "qdrant"]:
            if self._backend is None or self._collection is None:
                # print("Warning: Vector store backend or collection not initialized, flush ignored.") # noqa
                return
            await self._backend.flush(self._collection)
        elif self.vector_db_type == "sqlite-vec":
            # Flushing for sqlite-vec is implicitly handled by commits within sqlite_handler methods.
            # print("Info: flush called for sqlite-vec; operation is handled by commits.") # noqa
            pass  # No explicit action needed here

    async def delete_collection(self, collection_name: str):
        """Deletes a specified collection from the backend (not applicable to sqlite-vec)."""
        if self.vector_db_type in ["milvus", "milvus-lite", "chroma", "qdrant"]:
            if self._backend is None:
                raise RuntimeError(
                    f"External vector store backend ({self.vector_db_type}) not initialized for delete_collection.")
            if not hasattr(self._backend, 'delete_collection'):
                raise NotImplementedError(f"Backend {self.vector_db_type} does not implement delete_collection.")

            await self._backend.delete_collection(collection_name)

            # If the deleted collection was the one this handler instance was managing
            if self._collection and hasattr(self._collection, 'name') and self._collection.name == collection_name:
                self._collection = None
                print(
                    f"VectorStoreHandler: Active collection '{collection_name}' was deleted. Internal reference cleared.")
            elif self.collection_name == collection_name:  # If it was the default collection for this handler
                self._collection = None
                print(
                    f"VectorStoreHandler: Default collection '{collection_name}' for this handler was deleted. Internal reference cleared.")

        elif self.vector_db_type == "sqlite-vec":
            print(
                f"Info: delete_collection called for sqlite-vec with target '{collection_name}'. This operation is not applicable as sqlite-vec uses SQLite tables directly, not named collections in the same way. Table cleanup is a DB admin task.")

    async def close(self, flush=True):
        """
        Closes connections/releases resources asynchronously.
        For sqlite-vec, this delegates to the sqlite_handler's close method.
        """
        if self.vector_db_type in ["milvus", "milvus-lite", "chroma", "qdrant"]:
            if self._backend and self._collection:
                await self._backend.close(self._collection, flush=flush)
            self._backend = None
            self._collection = None
            print(f"Closed external vector store backend: {self.vector_db_type}")  # noqa
        elif self.vector_db_type == "sqlite-vec":
            if self._sqlite_handler:
                await self._sqlite_handler.close()
                self._sqlite_handler = None  # Clear the reference
                print("Closed sqlite_handler connection via VectorStoreHandler.")  # noqa
            else:
                # print("Info: close called for sqlite-vec, but handler was already closed or not initialized.") # noqa
                pass
