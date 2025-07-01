# LangGraph workflow setup for CogniGraph
import os
import yaml
from agents.planner import PlannerAgent, planner_node
from agents.retriever import RetrieverAgent, retriever_node
from agents.summarizer import SummarizerAgent, summarizer_node
from memory.memory import Memory
from langchain.llms import HuggingFacePipeline
from langgraph.graph import StateGraph, END
from langgraph.graph.state import State
from dotenv import load_dotenv

# Load environment variables and config
def load_config():
    load_dotenv()
    with open("config/model_settings.yaml", "r") as f:
        config = yaml.safe_load(f)
    return config

# Agent node functions
def planner_node(state):
    query = state["query"]
    history = state.get("history", "")
    action = state["planner"].decide(query, history)
    return {"action": action}

def retriever_node(state):
    query = state["query"]
    docs = state["retriever"].retrieve(query)
    return {"chunks": docs}

def summarizer_node(state):
    chunks = state["chunks"]
    query = state["query"]
    summary = state["summarizer"].summarize(chunks, query)
    return {"summary": summary}

# LangGraph state definition
class CogniGraphState(State):
    query: str
    history: str = ""
    action: str = ""
    chunks: list = []
    summary: str = ""
    planner: object = None
    retriever: object = None
    summarizer: object = None
    memory: object = None

def run_flow(user_query=None):
    config = load_config()
    chroma_path = os.getenv("CHROMADB_PATH", "./chromadb")
    model_name = os.getenv("MODEL_NAME", config["model"]["name"])
    hf_token = os.getenv("HUGGINGFACE_TOKEN", None)
    top_k = config["retriever"].get("top_k", 5)

    # Initialize agents
    llm = HuggingFacePipeline.from_model_id(model_id=model_name, task="text-generation", device=-1, model_kwargs={"temperature": config["model"].get("temperature", 0.2), "max_length": config["model"].get("max_length", 2048)}, huggingfacehub_api_token=hf_token)
    planner = PlannerAgent(llm)
    retriever = RetrieverAgent(chroma_path, model_name)
    summarizer = SummarizerAgent(model_name, hf_token)
    memory = Memory()

    # Build LangGraph workflow
    workflow = StateGraph(CogniGraphState)
    workflow.add_node("planner", planner_node)
    workflow.add_node("retriever", retriever_node)
    workflow.add_node("summarizer", summarizer_node)

    # Edges: planner -> retriever/summarizer, retriever -> summarizer, summarizer -> END
    def router(state):
        if state["action"] == "retrieve":
            return "retriever"
        elif state["action"] == "summarize":
            return "summarizer"
        else:
            return "retriever"  # fallback

    workflow.add_edge("planner", router)
    workflow.add_edge("retriever", "summarizer")
    workflow.add_edge("summarizer", END)
    workflow.set_entry_point("planner")

    app = workflow.compile()

    print("CogniGraph workflow started. Use the CLI to interact.")
    if user_query is not None:
        # For CLI use
        state = {
            "query": user_query,
            "planner": planner,
            "retriever": retriever,
            "summarizer": summarizer,
            "memory": memory
        }
        result = app.invoke(state)
        memory.log(user_query, None, result.get("summary", ""))
        return result.get("summary", "")
    # For server/interactive mode, could add loop here
