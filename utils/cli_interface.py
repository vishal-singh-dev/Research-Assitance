"""
chat/interface_cli.py
CLI chat interface with session memory.
"""

import asyncio
from core.retriever import Retriever
from .chat_session import SessionManager
from core.gemini_client import GeminiClient
from core.data_ingestion import DocumentIngestionService
from core.milvus_client import MilvusClient
from core.langGraph_retriever import RetrieverAgent,QState
from core.langGraph_retriever import QState

class ResearchAssistantCLI:
    def __init__(self):
        #self.retriever = Retriever()
        self.retriever = RetrieverAgent()
        self.session = SessionManager()
        self.llm=GeminiClient()
        self.collection=MilvusClient().collection
    async def run(self):
        print("\n🧠 Research Assistant Agent (CLI Mode)")
        print("Type ':help' for commands, ':quit' to exit.\n")
        state: QState = {
    "messages": [],
    "category": ""
}
        graph=self.retriever.build_graph()
        app=graph.compile()


        while True:
            user_input = input("You: ").strip()

            if user_input.lower() in ["quit", "exit"]:
                print("🔚 Exiting Research Assistant. Goodbye!")
                break

            elif user_input.lower() in ["help"]:
                print("""
Commands:
  :ingest <path>   - Ingest a text file
  :clear           - Clear chat session memory
  :quit / :exit    - Exit the chat
  :help            - Show this help message
""")
                continue

            elif user_input.startswith(":ingest "):
                path = user_input.replace(":ingest ", "").strip()
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        text = f.read()
                    await self.retriever.ingest(text, source=path)
                    print(f"📄 Document '{path}' successfully ingested.")
                except Exception as e:
                    print(f"[Error] Failed to ingest document: {e}")
                continue

            elif user_input.lower() == ":clear":
                self.session.clear()
                print("🧹 Session memory cleared.\n")
                continue

            elif not user_input:
                continue
            try:
               state["messages"].append(user_input)
               await app.ainvoke(state)
            except Exception as e:
                print(f"{e.__context__}")
               

   
