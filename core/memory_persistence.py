from pymilvus import Collection, CollectionSchema, FieldSchema, DataType
import json
from datetime import datetime
from typing import List, Dict
import uuid
from core.milvus_client import MilvusClient
class MilvusMemoryStore:
    def __init__(self):
        self.milvus_client= MilvusClient()
    
   
    
    def save_conversation(self, user_id: str, query: str, answer: str,
                         query_embedding: List[float], session_id: str = None,
                         metadata: Dict = None):
        """Save conversation with semantic search capability"""
        conv_id = str(uuid.uuid4())
        session_id = session_id or str(uuid.uuid4())
        timestamp = int(datetime.now().timestamp() * 1000)
        metadata = metadata or {}
        
        self.milvus_client.conv_collection.insert([
            [conv_id],
            [user_id],
            [session_id],
            [query],
            [answer],
            [query_embedding],
            [timestamp],
            [metadata]
        ])
        self.milvus_client.conv_collection.flush()
    
    def search_similar_conversations(self, user_id: str, query_embedding: List[float],
                                    top_k: int = 5) -> List[Dict]:
        """Find similar past conversations using vector similarity"""
        self.milvus_client.conv_collection.load()
        
        search_params = {"metric_type": "L2", "params": {"ef": 64}}
        
        results = self.milvus_client.conv_collection.search(
            data=[query_embedding],
            anns_field="query_embedding",
            param=search_params,
            limit=top_k,
            expr=f'user_id == "{user_id}"', 
            output_fields=["query", "answer", "timestamp", "metadata"]
        )
        
        if not results or len(results[0]) == 0:
            return []
        
        return [
            {
                "query": hit.entity.get("query"),
                "answer": hit.entity.get("answer"),
                "timestamp": hit.entity.get("timestamp"),
                "metadata": hit.entity.get("metadata"),
                "similarity": float(hit.score)
            }
            for hit in results[0]
        ]
    
    def get_recent_conversations(self, user_id: str, session_id: str = None,
                                last_n: int = 5) -> List[Dict]:
        """Get recent conversations chronologically"""
        self.milvus_client.conv_collection.load()
        
        # Query with filter
        expr = f'user_id == "{user_id}"'
        if session_id:
            expr += f' && session_id == "{session_id}"'
        
        results = self.milvus_client.conv_collection.query(
            expr=expr,
            output_fields=["query", "answer", "timestamp", "session_id"],
            limit=last_n
        )
        
        # Sort by timestamp
        results.sort(key=lambda x: x["timestamp"], reverse=True)
        return results[:last_n]
    
    def save_preferences(self, user_id: str, preferences: Dict):
        """Save all user preferences"""
        pref_id = f"pref_{user_id}"
        
        # Delete existing preferences
        self.milvus_client.pref_collection.delete(expr=f'user_id == "{user_id}"')
        
        # Insert new preferences
        self.milvus_client.pref_collection.insert([
            [pref_id],
            [user_id],
            [preferences]
        ])
        self.milvus_client.pref_collection.flush()
    
    def get_preferences(self, user_id: str) -> Dict:
        """Get user preferences"""
        self.milvus_client.pref_collection.load()
        
        results = self.milvus_client.pref_collection.query(
            expr=f'user_id == "{user_id}"',
            output_fields=["preferences"]
        )
        
        return results[0]["preferences"] if results else {}