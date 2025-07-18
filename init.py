import os
import yaml
try:
    from langchain_community.llms import HuggingFacePipeline
    from langchain_community.embeddings import HuggingFaceEmbeddings
except ImportError:
    # Fallback to older imports if langchain-community is not available
    from langchain.llms import HuggingFacePipeline
    from langchain.embeddings import HuggingFaceEmbeddings
from transformers import pipeline
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Load model settings from YAML
with open('config/model_settings.yaml', 'r') as f:
    config = yaml.safe_load(f)

MODEL_NAME = os.getenv('MODEL_NAME', config['model']['name'])
HUGGINGFACE_TOKEN = os.getenv('HUGGINGFACE_TOKEN', None)
CHROMADB_PATH = os.getenv('CHROMADB_PATH', './chromadb')
EMBEDDING_MODEL = config['model'].get('embedding_model', 'BAAI/bge-base-en-v1.5')

# Initialize Hugging Face LLM pipeline for LangChain
llm_pipeline = pipeline(
    "text2text-generation",
    model=MODEL_NAME,
    tokenizer=MODEL_NAME,
    token=HUGGINGFACE_TOKEN,
    max_length=config['model'].get('max_length', 512),
    temperature=config['model'].get('temperature', 0.2)
)
llm = HuggingFacePipeline(pipeline=llm_pipeline)

# Initialize embedding model for ChromaDB
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def get_llm():
    return llm

def get_embeddings():
    return embeddings

def get_chromadb_path():
    return CHROMADB_PATH
