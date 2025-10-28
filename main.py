from utils.chat_session import SessionManager
from pymilvus import connections, Collection
from core.data_ingestion import DocumentIngestionService
from core.embeddings import Embedder
from core.milvus_client import MilvusClient
import asyncio
from utils.cli_interface import ResearchAssistantCLI
async def main():

    embedder = Embedder()
    print("embedder loaded")
    milvus_client = MilvusClient()
    milvus_client._connect()
    print("connected to milvus")
    milvus_client._get_or_create_collection()

   
    service = DocumentIngestionService(
        input_dir="./data/documents",
        processed_dir="./data/docs_embedded",
        milvus_client=milvus_client,
        embedder=embedder
    )
    print("Document Ingestion Service initialized.")
    # 3️⃣ Process docs
    await service.process_documents()
    print("Document processed.")
if __name__ == "__main__":
    asyncio.run(main())
    chat = SessionManager()
    print("=== Research Assistant (Gemini) ===")
    asyncio.run(ResearchAssistantCLI().run())

    # while True:
    #     user_input = input("\nYou: ")
    #     if user_input.lower() in {"exit", "quit"}:
    #         break
    #     response = chat.chat(user_input)
    #     print("\nAssistant:", response)
