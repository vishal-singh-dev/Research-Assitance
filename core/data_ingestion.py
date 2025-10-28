from core.milvus_client import MilvusClient
from core.embeddings import Embedder
import os
from core.pydantic_models import Document
import shutil
class DocumentIngestionService:
    def __init__(self, input_dir: str, processed_dir: str, milvus_client: MilvusClient, embedder: Embedder):
        self.input_dir = input_dir
        self.processed_dir = processed_dir
        self.milvus_client = milvus_client
        self.embedder = embedder

    async def process_documents(self):
        files = [f for f in os.listdir(self.input_dir) if os.path.isfile(os.path.join(self.input_dir, f))]
        if not files:
            print("📂 No new documents found.")
            return

        for file in files:
            file_path = os.path.join(self.input_dir, file)
            print(f"📖 Reading {file_path}...")

            # Only handle text files for now
            if not file.lower().endswith(".txt"):
                print(f"⚠️ Skipping unsupported file: {file}")
                continue

            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()

            # Embed
            chunks=self.embedder.chunk_text(text)
            print(f"🔍 Chunked into {len(chunks)} pieces.")
            docs = await self.embedder.embed_texts(chunks)
            #docs = [Document(content=text, embedding=embeddings[0], source=file)]

            # Insert into Milvus
            validated_docs = [Document(**d) for d in docs]
            self.milvus_client.insert_documents(docs)

            # Move file to processed folder
            dest_path = os.path.join(self.processed_dir, file)
            shutil.move(file_path, dest_path)
            print(f"✅ Moved {file} → {self.processed_dir}")

        print("🎯 All documents processed successfully.")