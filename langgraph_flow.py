# LangGraph workflow setup for CogniGraph
import os
import yaml
from agents.planner import PlannerAgent
from agents.retriever import RetrieverAgent
from agents.summarizer import SummarizerAgent
from memory.memory import Memory
from init import get_llm, get_embeddings, get_chromadb_path
from dotenv import load_dotenv

# Load environment variables and config
def load_config():
    load_dotenv()
    with open("config/model_settings.yaml", "r") as f:
        config = yaml.safe_load(f)
    return config

# Simple function-based workflow for CogniGraph

# Simple workflow function
def run_flow(user_query=None):
    if not user_query:
        return {"error": "No query provided"}
    
    # Initialize components
    config = load_config()
    memory = Memory()
    planner = PlannerAgent()
    retriever = RetrieverAgent()
    summarizer = SummarizerAgent()
    
    # Create initial state
    state = {
        "query": user_query,
        "history": str(memory.get_history())
    }
    
    # Step 1: Planner decides action
    action = planner.decide(user_query, state["history"])
    state["action"] = action
    
    # Step 2: Take action based on planner's decision
    if action == "retrieve":
        docs = retriever.retrieve(user_query)
        state["documents"] = docs
    elif action == "summarize":
        docs = retriever.retrieve(user_query)
        state["documents"] = docs
        summary = summarizer.summarize(docs, user_query)
        state["summary"] = summary
    elif action == "search":
        # Placeholder for search logic
        state["documents"] = ["[Search not implemented]"]
    
    # Step 3: Summarize if needed
    if "documents" in state and action != "summarize" and state["documents"]:
        summary = summarizer.summarize(state["documents"], user_query)
        state["summary"] = summary
    
    # Log to memory
    memory.log(user_query, None, state.get("summary", ""))
    
    return state

print("CogniGraph workflow ready. Use the CLI to interact.")
