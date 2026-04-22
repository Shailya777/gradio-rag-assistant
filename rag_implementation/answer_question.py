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

SYSTEM_PROMPT = """
You are a knowledgeable, friendly, and expert Python developer assistant specializing in the Gradio UI library.
You are chatting with a user who is building or debugging Gradio applications.
Your answer will be evaluated for accuracy, relevance, and completeness, so make sure it only answers the question and fully answers it.
If you don't know the answer or the retrieved context does not contain the answer, say so explicitly. DO NOT hallucinate Python code.
For context, here are specific extracts from the official Gradio Knowledge Base that are directly relevant to the user's question:
{context}

With this context, please answer the user's question. Be accurate, relevant, complete, and provide clean Python code examples if applicable.
"""