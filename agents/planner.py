# Planner Agent Node

from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFacePipeline

class PlannerAgent:
    def __init__(self, llm: HuggingFacePipeline):
        self.llm = llm
        self.prompt = PromptTemplate(
            input_variables=["query", "history"],
            template=(
                "You are a planning agent. Given the user query and conversation history, "
                "decide the next action: 'retrieve', 'summarize', or 'search'.\n"
                "User Query: {query}\n"
                "History: {history}\n"
                "Respond with only one word: retrieve, summarize, or search."
            )
        )

    def decide(self, query: str, history: str = "") -> str:
        prompt = self.prompt.format(query=query, history=history)
        response = self.llm(prompt)
        action = response.strip().lower()
        if action not in ["retrieve", "summarize", "search"]:
            action = "retrieve"
        return action
