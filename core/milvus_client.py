"""
core/milvus_client.py
Handles Milvus vector database operations:
- Connection
- Schema creation
- Insertions
- Similarity search
"""

from pymilvus import (
    connections,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    utility
)
from config.settings import (
    MILVUS_HOST,
    MILVUS_PORT,
    COLLECTION_NAME,
    EMBED_DIM
)
import uuid
import numpy as np

class MilvusClient:
    def __init__(self):
        self.alias = "default"
        self._connect()
        self.collection = self._get_or_create_collection()

    # ---------------- CONNECTION ----------------
    def _connect(self):
        if not connections.has_connection(self.alias):
            connections.connect(alias=self.alias, host=MILVUS_HOST, port=MILVUS_PORT)
        print(f"[Milvus] Connected to {MILVUS_HOST}:{MILVUS_PORT}")

    # ---------------- SCHEMA SETUP ----------------
    def _get_or_create_collection(self):
        if utility.has_collection(COLLECTION_NAME):
            print(f"[Milvus] Using existing collection: {COLLECTION_NAME}")
            return Collection(name=COLLECTION_NAME)

        print(f"[Milvus] Creating new collection: {COLLECTION_NAME}")

        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, auto_id=False, max_length=64),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=EMBED_DIM),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=4096),
            FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="chunk_index", dtype=DataType.INT64),
            FieldSchema(name="metadata", dtype=DataType.JSON),
        ]

        schema = CollectionSchema(fields, description="Research Assistant Documents")
        collection = Collection(name=COLLECTION_NAME, schema=schema)
        collection.create_index("embedding", {"metric_type": "IP", "index_type": "IVF_FLAT", "params": {"nlist": 128}})
        print("[Milvus] Created collection and index successfully.")
        return collection

    # ---------------- INSERTION ----------------
    def insert_documents(self, docs):
        """
        docs: list of dicts like:
        {
          'content': str,
          'embedding': list[float],
          'source': str,
          'chunk_index': int,
          'metadata': dict
        }
        """
        ids = [str(uuid.uuid4()) for _ in docs]
        embeddings = [self.normalize_vector(d["embedding"]) for d in docs]
        contents = [d["content"] for d in docs]
        sources = [d.get("source", "unknown") for d in docs]
        chunks = [d.get("chunk_index", 0) for d in docs]
        metas = [d.get("metadata", {}) for d in docs]


        self.collection.insert([ids, embeddings, contents, sources, chunks, metas])
        self.collection.flush()
        print(f"[Milvus] Inserted {len(docs)} documents.")


    # ---------------- UTILITY ----------------
    def count(self):
        return self.collection.num_entities

    def clear(self):
        self.collection.drop()
        print(f"[Milvus] Dropped collection {COLLECTION_NAME}")

  # ---------------- RETRIEVE ----------------
    # def search(self, query_vector, top_k=20):
        # """
        # Perform similarity search on embeddings.
        # Returns a list of hits with content and metadata.
        # """
        # # Ensure query_vector is a list
        # self.collection.load()
        # print(f"--- DEBUG: Total entities in collection: {self.collection.num_entities} ---")
        # print(f"param->{self.collection.index().params}")
        # query_vector = query_vector.tolist() if hasattr(query_vector, "tolist")else query_vector
        # if not isinstance(query_vector[0], (float, int)):
        #     raise ValueError("query_vector must be a list of floats")

        # # Search params (depends on index type)
        # search_params = {"metric_type": "IP"}  # use only metric_type for FLAT
        # if self.collection.index().params.get("index_type") == "HNSW":
        #     search_params["params"] = {"ef": 64}

        # results = self.collection.search(
        #     data=[query_vector],
        #     anns_field="embedding",
        #     param=search_params,
        #     limit=top_k,
        #     output_fields=["content", "source", "chunk_index", "metadata"]
        # )

        # if not results or len(results[0]) == 0:
        #     return []

        # hits = results[0]
        # formatted = [
        #     {
        #         "content": h.entity.get("content"),
        #         "source": h.entity.get("source"),
        #         "chunk_index": h.entity.get("chunk_index"),
        #         "metadata": h.entity.get("metadata"),
        #         "score": float(h.score),
        #     }
        #     for h in hits
        # ]
        # return formatted
        
    def search(self, query_vector, top_k=20):
        """
        Perform similarity search on embeddings.
        Returns a list of hits with content and metadata.
        """
        # Ensure query_vector is a list
        self.collection.load()
        print(f"--- DEBUG: Total entities in collection: {self.collection.num_entities} ---")
        print(f"param->{self.collection.index().params}")

        try:
            if hasattr(query_vector, "tolist"):
                query_vector = query_vector.tolist()
            elif not isinstance(query_vector, list):
                query_vector = list(query_vector)
        
        # Handle nested lists (e.g., [[1,2,3]] -> [1,2,3])
            if isinstance(query_vector, list) and len(query_vector) > 0 and isinstance(query_vector[0], list):
                query_vector = query_vector[0]
        
        # Ensure all elements are float
            query_vector = [float(x) for x in query_vector]
        
        except Exception as e:
            print(f"Error converting query_vector: {e}")
            print(f"Query vector content: {query_vector}")
            raise ValueError("query_vector must be a list of floats")
        print(f"Query vector dimension: {len(query_vector)}")

        # Search params - MUST have both metric_type AND params
        index_type = self.collection.index().params.get("index_type", "FLAT")

        if index_type == "HNSW":
            search_params = {
                "metric_type": "IP",
                "params": {"ef": 64}
            }
        elif index_type == "FLAT":
            search_params = {
                "metric_type": "IP",
                "params": {}  # FLAT needs empty params dict, not missing
            }
        else:
        # For other index types (IVF_FLAT, etc.)
            search_params = {
                "metric_type": "IP",
                "params": {"nprobe": 10}
            }

        print(f"Search params: {search_params}")

        results = self.collection.search(
        data=[query_vector],
        anns_field="embedding",
        param=search_params,
        limit=top_k,
        output_fields=["content", "source", "chunk_index", "metadata"]
        )

        if not results or len(results[0]) == 0:
            print("No results found!")
            return []

        hits = results[0]
        print(f"Found {len(hits)} results")

        formatted = [
        {
            "content": h.entity.get("content"),
            "source": h.entity.get("source"),
            "chunk_index": h.entity.get("chunk_index"),
            "metadata": h.entity.get("metadata"),
            "score": float(h.score),
        }
        for h in hits
        ]
        return formatted



    @staticmethod
    def normalize_vector(vec):
        """Normalize vector to unit length for IP metric"""
        vec = np.array(vec)
        norm = np.linalg.norm(vec)
        return (vec / norm).tolist() if norm > 0 else vec.tolist()
    
    def _create_collections_memory(self):
        """Create collections for different memory types"""
        if utility.has_collection("conversation_history"):
            print(f"[Milvus] Using existing collection: conversation_history")
            self.conv_collection = Collection(name="conversation_history")
        conv_fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=100),
            FieldSchema(name="user_id", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="session_id", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="query", dtype=DataType.VARCHAR, max_length=5000),
            FieldSchema(name="answer", dtype=DataType.VARCHAR, max_length=10000),
            FieldSchema(name="query_embedding", dtype=DataType.FLOAT_VECTOR, dim=768),
            FieldSchema(name="timestamp", dtype=DataType.INT64),
            FieldSchema(name="metadata", dtype=DataType.JSON)
        ]
        
        conv_schema = CollectionSchema(fields=conv_fields, description="Conversation history")
        
        try:
            self.conv_collection = Collection(name="conversation_history", schema=conv_schema)
        except:
            self.conv_collection = Collection(name="conversation_history")
        
        index_params = {
            "metric_type": "L2",
            "index_type": "HNSW",
            "params": {"M": 16, "efConstruction": 256}
        }
        self.conv_collection.create_index(field_name="query_embedding", index_params=index_params)
        
        if utility.has_collection("user_preferences"):
            print(f"[Milvus] Using existing collection: user_preferences")
            self.pref_collection = Collection(name="user_preferences")
        pref_fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=200),
            FieldSchema(name="user_id", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="preferences", dtype=DataType.JSON)
        ]
        
        pref_schema = CollectionSchema(fields=pref_fields, description="User preferences")
        
        try:
            self.pref_collection = Collection(name="user_preferences", schema=pref_schema)
        except:
            self.pref_collection = Collection(name="user_preferences")


