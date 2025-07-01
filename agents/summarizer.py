from init import get_llm

class SummarizerAgent:
    def __init__(self, llm=None):
        self.llm = llm or get_llm()

    def summarize(self, chunks, user_query):
        context = "\n".join(chunks)
        prompt = f"Summarize the following in the context of the user query: {user_query}\n{context}"
        result = self.llm(prompt)
        return result

def summarizer_node(state):
    agent = SummarizerAgent()
    docs = state.get("documents", [])
    query = state.get("query", "")
    summary = agent.summarize(docs, query)
    state["summary"] = summary
    return state
