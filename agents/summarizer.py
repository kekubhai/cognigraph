from init import get_llm

class SummarizerAgent:
    def __init__(self, llm=None):
        self.llm = llm or get_llm()

    def summarize(self, chunks, user_query):
        if not chunks:
            return "No relevant information found."
        
        context = "\n".join(chunks)
        prompt = f"Summarize the following information to answer the user query: '{user_query}'\n\nInformation:\n{context}\n\nSummary:"
        result = self.llm.invoke(prompt)  # Updated to use invoke method
        return result

def summarizer_node(state):
    agent = SummarizerAgent()
    docs = state.get("documents", [])
    query = state.get("query", "")
    summary = agent.summarize(docs, query)
    state["summary"] = summary
    return state
