"""
core/embeddings.py
Handles embedding generation for document chunks using the BAAI model.
Returns list[dict] instead of Document objects.
"""

import asyncio
from typing import List, Dict
from sentence_transformers import SentenceTransformer
#from utils.logger import log


class Embedder:
    def __init__(self, model_name: str = "BAAI/bge-base-en-v1.5"):
        print(f"🔧 Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)

    async def embed_texts(
        self, chunks: List[str], metadata: List[Dict] = None
    ) -> List[Dict]:
        """
        Asynchronously embed a list of text chunks.
        Returns list of dicts: {text, embedding, metadata}
        """
        metadata = metadata or [{} for _ in chunks]

        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(None, lambda: self.model.encode(chunks, convert_to_numpy=True))


        results = []
        for i, emb in enumerate(embeddings):
            entry = {
                "chunk_index": i,
                "embedding": emb.tolist() if hasattr(emb, "tolist") else emb,
                "metadata": metadata[i],
                "content": chunks[i],
            }
            results.append(entry)

        print(f"✅ Generated {len(results)} embeddings.")
        return results


    def chunk_text(self, text: str, size: int = 50, overlap: int = 10) -> List[str]:
        return [text[i:i+size] for i in range(0, len(text), size - overlap)]
