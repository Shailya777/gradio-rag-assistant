# Imports
from openai import OpenAI
from dotenv import load_dotenv
from chromadb import PersistentClient
from litellm import completion
from pydantic import BaseModel, Field
from pathlib import Path
from tenacity import retry, wait_exponential

# Constants:
load_dotenv()
MODEL= 'openai/gpt-4.1-nano'
DB_NAME= str(Path(__file__).parent.parent / 'preprocessed_db')
collection_name = 'docs'
embedding_model = 'text-embedding-3-large'
KNOWLEDGE_BASE_PATH = Path(__file__).parent.parent / 'Knowledge-Base'
wait = wait_exponential(multiplier= 1, min= 10, max= 240)
openai = OpenAI()
chroma = PersistentClient(path= DB_NAME)
collection = chroma.get_collection(collection_name)

#Chunks to Retrieve:
RETRIEVAL_K= 20

# Final Chunks to Use:
FINAL_K= 10