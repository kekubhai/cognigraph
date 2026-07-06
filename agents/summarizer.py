from init import get_llm
from langgraph.graph import StateGraph, END
from typing import TypedDict, List














class SummarizerAgent:
    def __init__(self, llm=None):
        self.llm = llm or get_llm()

    def summarize(self, chunks, user_query):
        if not chunks:
            return "No relevant information found."

        max_input_tokens = 400
        tokenizer = self.llm.pipeline.tokenizer
        context = ""

        for chunk in chunks:
            candidate = f"{context}\n{chunk}" if context else chunk
            token_count = len(tokenizer.encode(candidate, truncation=True))
            if token_count <= max_input_tokens:
                context = candidate
            else:
                remaining = max_input_tokens - len(tokenizer.encode(context, truncation=True)) if context else max_input_tokens
                if remaining > 20:
                    truncated = tokenizer.decode(
                        tokenizer.encode(chunk, truncation=True)[:remaining],
                        skip_special_tokens=True
                    )
                    context = f"{context}\n{truncated}" if context else truncated
                break

        prompt = f"Summarize the following information to answer the user query: '{user_query}'\n\nInformation:\n{context}\n\nSummary:"
        result = self.llm.invoke(prompt)
        return result

def summarizer_node(state):
    agent = SummarizerAgent()
    docs = state.get("documents", [])
    query = state.get("query", "")
    summary = agent.summarize(docs, query)
    state["summary"] = summary
    return state
