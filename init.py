import os
import yaml
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

_base_dir = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(_base_dir, 'config', 'model_settings.yaml'), 'r') as f:
    config = yaml.safe_load(f)

MODEL_NAME = os.getenv('MODEL_NAME', config['model']['name'])
HUGGINGFACE_TOKEN = os.getenv('HUGGINGFACE_TOKEN', None) or None
CHROMADB_PATH = os.getenv('CHROMADB_PATH', './chromadb')
EMBEDDING_MODEL = config['model'].get('embedding_model', 'BAAI/bge-base-en-v1.5')

model_kwargs = {}
if HUGGINGFACE_TOKEN:
    model_kwargs["token"] = HUGGINGFACE_TOKEN

llm = HuggingFacePipeline.from_model_id(
    model_id=MODEL_NAME,
    task="text2text-generation",
    model_kwargs=model_kwargs,
    pipeline_kwargs={
        "max_length": config['model'].get('max_length', 512),
    },
    device=-1,
)

embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def get_llm():
    return llm

def get_embeddings():
    return embeddings

def get_chromadb_path():
    return CHROMADB_PATH
