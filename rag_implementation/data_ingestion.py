# Imports:
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from chromadb import PersistentClient
from tqdm import tqdm
from litellm import completion
from tenacity import retry, wait_exponential

load_dotenv(override= True)

# Constants:
MODEL = 'llama3.1:8b'
DB_NAME = str(Path(__file__).parent.parent / 'preprocessed_db')
collection_name = 'docs'
embedding_model = 'text-embedding-3-large'
KNOWLEDGE_BASE_PATH = Path(__file__).parent.parent / 'knowledge_base'
wait = wait_exponential(multiplier= 1, min= 10, max= 240)