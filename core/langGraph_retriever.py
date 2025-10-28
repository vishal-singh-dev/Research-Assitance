
from langgraph.graph import StateGraph, START, END
from typing import List,TypedDict
from core.retriever import Retriever
from utils.chat_session import SessionManager 
class QState(TypedDict):
    messages: List[str]
    category: str

class RetrieverAgent:
    def __init__(self):
            self.retriever = Retriever()
            self.session = SessionManager()

    def classify_question(self,state: QState):
        q = state["messages"][-1].lower()
        if "summary" in q:
            return {"category": "summarize"}
        elif "multistep" in q:
            return {"category": "multistep"}
        else:
            return {"category": "direct"}

    async def direct_answer(self,state: QState):
        try:
            q = state["messages"][-1].lower()
            self.session.add_turn("user", q)
            response = await self.retriever.query(q)

            print(f"\n🤖 Agent:\n{response}\n")
            self.session.add_turn("assistant", response)
            state["messages"].append(response)
            return response

        except Exception as e:
            print(f"{e.__context__}")

    async def summarization(self, state: QState):
        try:
            q = state["messages"][-1].lower()
            self.session.add_turn("user", q)
            context = await self.retriever.query(q)
            prompt = (
            f"Context:\n{context}\n\n"
            f"Task: Write a concise summary that answers the question below, using only the context.\n"
            f"Question: {q}"
            )
            summary = await self.retriever.query(prompt)
            return summary

            # return {
            # "messages": [AIMessage(content=summary)]
            # }

        except Exception as e:
            print(f"{e.__context__}")

        
    async def multi_step(self,state: QState):
        q = state["messages"][-1].lower()
        decomposition_prompt = ( f"Break this question into 3–5 smaller questions that can be answered individually:\n\n{q}" )
        subq_response = await self.retriever.query(decomposition_prompt)
        subqs = [q.strip("-• ") for q in subq_response.strip().split("\n") if q.strip()]
        answers = [] 
        for sq in subqs: answer = await self.retriever.query(sq)
        answers.append(f"**Q:** {sq}\n**A:** {answer}")
        synthesis_context = "\n\n".join(answers) 
        synthesis_prompt = ( f"The following are answers to sub-questions derived from the main question:\n\n" f"{synthesis_context}\n\n" f"Now synthesize a final, complete answer to the original question:\n\n{q}" ) 
        final_answer = await self.retriever.query_with_context(query=q, context=synthesis_prompt)

        return final_answer
    
    def build_graph(self)->StateGraph:

        graph = StateGraph(QState)
        graph.add_node("classify", self.classify_question)
        graph.add_node("direct_flow", self.direct_answer)
        graph.add_node("summarize_flow", self.summarization)
        graph.add_node("multistep_flow", self.multi_step)

        # Entry point
        graph.add_edge(START, "classify")
        # Conditional routing based on the category set by classify_question
        def route_after_classify(state: QState) -> str:
            return state["category"]
        graph.add_conditional_edges(
            "classify",
            route_after_classify,
            {"direct": "direct_flow", "summarize": "summarize_flow", "multistep": "multistep_flow"}
        )
        # End edges
        graph.add_edge("direct_flow", END)
        graph.add_edge("summarize_flow", END)
        graph.add_edge("multistep_flow", END)
        return graph
