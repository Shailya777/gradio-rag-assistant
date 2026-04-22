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
You are a knowledgeable, friendly, and expert Python developer assistant specializing in the gradio UI library.
You are chatting with a user who is building or debugging gradio applications.
Your answer will be evaluated for accuracy, relevance, and completeness, so make sure it only answers the question and fully answers it.
If you don't know the answer or the retrieved context does not contain the answer, say so explicitly. DO NOT hallucinate Python code.
For context, here are specific extracts from the official gradio Knowledge Base that are directly relevant to the user's question:
{context}

With this context, please answer the user's question. Be accurate, relevant, complete, and provide clean Python code examples if applicable.
"""

class Result(BaseModel):
    """
    Represents a single chunk of retrieved context from the vector database.

    This class standardizes the format of documents fetched from ChromaDB,
    making it easy to pass the raw text and its associated tags into the
    reranking and final generation phases.

    Attributes:
        page_content (str): The actual text content of the chunk (including the
                            LLM-generated headline, summary, and raw gradio Markdown).
        metadata (dict): Contextual tags associated with the chunk, such as the source
                         file path and the Markdown header hierarchy.
    """
    page_count: str
    metadata: dict


class RankOrder(BaseModel):
    """
    Schema for enforcing structured JSON output during the LLM reranking phase.

    When we ask the frontier model (OpenAI) to evaluate and sort the retrieved
    chunks by relevance, we use this Pydantic model to guarantee the LLM returns
    a clean list of integers. This prevents the pipeline from crashing due to
    unexpected conversational text in the response.

    Attributes:
        order (list[int]): A list of 1-based indices representing the optimal
                           relevance order (most relevant chunk ID first, the least relevant last).
    """
    order: list[int] = Field(
        description='The order of relevance of chunks, from most relevant to least relevant, by chunk id number'
    )

def rewrite_query(question, history= []):
    """
    Rewrite the user's question to be a more specific question that is more
    likely to surface relevant content in the Knowledge Base.

    :param question: User's Current question to be rewritten.
    :param history: History of conversation.
    :return: LLM's Rewritten question.
    """

    query_rewrite_sys_prompt = f"""
    You are in a conversation with a user, answering questions about the company Insurellm.
    You are about to look up information in a Knowledge Base to answer the user's question.

    This is the history of your conversation so far with the user:
    {history}

    And this is the user's current question:
    {question}

    Respond only with a short, refined question that you will use to search the Knowledge Base.
    It should be a VERY short specific question most likely to surface content. Focus on the question details.
    IMPORTANT: Respond ONLY with the precise knowledgebase query, nothing else.
    """
    response = completion(model= MODEL,
                          message= [{'role': 'system', 'content': query_rewrite_sys_prompt}])
    return response.choices[0].message.content