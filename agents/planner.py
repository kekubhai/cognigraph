# Planner Agent Node

from init import get_llm
from langchain.prompts import PromptTemplate

class PlannerAgent:
    def __init__(self, llm=None):
        self.llm = llm or get_llm()
        self.prompt = PromptTemplate(
            input_variables=["query", "history"],
            template=(
                "You are a planning agent for a RAG system. Based on the user query, decide the best action:\n"
                "- 'retrieve': If the user is asking for information that might be in documents\n"
                "- 'summarize': If the user wants a summary of previously retrieved information\n"
                "- 'search': If the user wants to search for new information\n\n"
                "User Query: {query}\n"
                "Previous History: {history}\n\n"
                "Choose ONE action (retrieve/summarize/search):"
            )
        )

    def decide(self, query: str, history: str = "") -> str:
        prompt = self.prompt.format(query=query, history=history)
        response = self.llm.invoke(prompt)  # Updated to use invoke method
        action = response.strip().lower()
        if action not in ["retrieve", "summarize", "search"]:
            action = "retrieve"
        return action

def planner_node(state):
    agent = PlannerAgent()
    query = state.get("query", "")
    history = state.get("history", "")
    action = agent.decide(query, history)
    state["action"] = action
    return state
