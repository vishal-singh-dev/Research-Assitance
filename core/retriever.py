"""
core/retriever.py
Coordinates embedding, retrieval, and response composition.
"""

from core.embeddings import Embedder
from core.milvus_client import MilvusClient
from core.gemini_client import GeminiClient
from core.hf_client import HuggingFaceClient
from core.memory_persistence import MilvusMemoryStore
#from config.settings import RETRIEVAL_TOP_K


class Retriever:
    def __init__(self):
        self.embeddings = Embedder()
        self.milvus = MilvusClient()
        self.llm = HuggingFaceClient()
        self.memorystore= MilvusMemoryStore()

    async def ingest(self, text: str, source: str):
        """
        Chunk, embed, and insert documents into Milvus.
        """
        docs = await self.embeddings.prepare_documents(text, source)
        self.milvus.insert_documents(docs)

    async def query(self, user_query: str) -> str:
        """
        Full retrieval + answer generation flow.
        """
        # 1. Embed query
        chunked_query = self.embeddings.chunk_text(user_query)
        embedding_result = (await self.embeddings.embed_texts(chunked_query))[0]
        query_vector=embedding_result["embedding"]
        print("embed")
        existing_answers=self.memorystore.search_similar_conversations("abc", query_vector)
        answers = [ans["answer"] for ans in existing_answers if "answer" in ans]
        if answers:
            return "\n\n".join(answers)

        # 2. Retrieve relevant context
        results = self.milvus.search(query_vector, top_k=5)
        print("result")
        if len(results) > 0:
            context_snippets = [r["content"] for r in results]
            context = "\n\n".join(context_snippets)
            answer = await self.llm.ask_model(user_query, context)
        else:
            answer = await self.llm.ask_model(user_query, "")
        print("gemini") 
        self.memorystore.save_conversation("abc", user_query, answer, query_vector)
        return answer
    
    async def query_with_context(self, user_query: str, context: str) -> str:
        """
        Directly ask Gemini with provided context.
        """
        answer = await self.llm.ask_model(user_query, context)
        self.memorystore.save_conversation("abc", user_query, answer,[])
        return answer
