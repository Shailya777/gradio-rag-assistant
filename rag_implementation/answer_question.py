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
    page_content: str
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
    Rewrite the user's question to be a more specific, standalone search query
    that is optimized to surface relevant technical gradio documentation.

    :param question: User's Current question to be rewritten.
    :param history: History of conversation.
    :return: LLM's Rewritten question.
    """

    query_rewrite_sys_prompt = f"""
    You are in a conversation with a developer, answering questions about the gradio Python library.
    You are about to look up information in a Vector Database Knowledge Base to answer the user's question.

    This is the history of your conversation so far with the user:
    {history}

    And this is the user's current question:
    {question}

    Respond ONLY with a short, highly refined search query that you will use to search the database.
    It should be a specific, technical question or phrase most likely to surface relevant code and explanations.
    IMPORTANT: Respond ONLY with the precise query, nothing else. No conversational text.
    """
    response = completion(model= MODEL,
                          messages= [{'role': 'system', 'content': query_rewrite_sys_prompt}])
    return response.choices[0].message.content

def fetch_context_unranked(question):
    """
    Retrieves the most semantically relevant chunks from the vector database for a given question.

    This function converts the input question into an embedding vector using OpenAI,
    queries the local ChromaDB collection for the closest mathematical matches, and
    packages the raw database output into a list of structured Result objects.

    :param:
        question (str): The search query (either the raw user input or the rewritten query).

    :return:
        list[Result]: A list of Result objects containing the page_content and metadata
                      for the top K most relevant chunks.
    """
    # Turning question into Embedding Vector:
    query = openai.embeddings.create(model= embedding_model, input= [question]).data[0].embedding

    # Using Embedding Vector of Question to Search into Vector Store:
    results = collection.query(query_embeddings= [query], n_results= RETRIEVAL_K)

    # Appending results from vector store to a list of Result Objects
    chunks= []
    for result in zip(results['documents'][0], results['metadatas'][0]):
        chunks.append(Result(page_content= result[0], metadata= result[1]))

    return chunks


def rerank_chunks(question, chunks):
    """
    Evaluates and sorts retrieved database chunks by true semantic relevance to the user's question.

    Vector similarity search can sometimes surface tangentially related chunks.
    This function acts as a precision filter, passing the retrieved chunks to a
    frontier LLM to evaluate their actual utility in answering the specific question.
    The LLM returns an ordered list of IDs, which is used to re-sort the chunks
    from most to least relevant.

    :param: question (str): The user's original question.
    :param: chunks (list[Result]): The deduplicated list of chunks retrieved from ChromaDB.

    :return: list[Result]: The newly sorted list of Result objects, optimized for final context injection.
    """

    rerank_chunks_sys_prompt= """
    You are a document re-ranker.
    You are provided with a question and a list of relevant chunks of text from a query of a knowledge base.
    The chunks are provided in the order they were retrieved; this should be approximately ordered by relevance, but you may be able to improve on that.
    You must rank order the provided chunks by relevance to the question, with the most relevant chunk first.
    Reply only with the list of ranked chunk ids, nothing else. Include all the chunk ids you are provided with, reranked.
    """

    rerank_chunks_user_prompt= f'''
    The user has asked the following question:\n\n{question}\n\nOrder all the chunks of text by relevance to the question,
     from most relevant to least relevant. Include all the chunk ids you are provided with, reranked.\n\n
     '''
    rerank_chunks_user_prompt += "Here are the Chunks:\n\n"

    for index, chunk in enumerate(chunks):
        rerank_chunks_user_prompt += f"# CHUNK ID: {index + 1}:\n\n{chunk.page_content}\n\n"

    rerank_chunks_user_prompt += "Reply only with the list of ranked chunk ids, nothing else."

    messages= [{'role': 'system', 'content': rerank_chunks_sys_prompt},
               {'role': 'user', 'content': rerank_chunks_user_prompt}
               ]

    response = completion(model= MODEL, messages= messages, response_format= RankOrder)
    reply = response.choices[0].message.content
    order = RankOrder.model_validate_json(reply).order
    return [chunks[i-1] for i in order]

def merge_chunks(chunks1, chunks2):
    """
    Merges two lists of retrieved database chunks while removing any exact duplicates.

    When querying the vector database with both the original user prompt and
    the optimized rewritten query, overlapping results are highly likely. This
    function acts as a deduplication filter, combining the lists while ensuring
    no redundant text is sent to the LLM context window.

    :param: chunks1 (list[Result]): The first list of retrieved Result objects.
    :param: chunks2 (list[Result]): The second list of retrieved Result objects to be merged.

    :return: list[Result]: A unified, deduplicated list of unique chunks.
    """
    merged= chunks1[:]

    existing= [chunk.page_content for chunk in chunks1]

    for chunk in chunks2:
        if chunk.page_content not in existing:
            merged.append(chunk)

    return merged


def fetch_context(original_question):
    """
    Orchestrates the Advanced RAG retrieval pipeline to fetch, deduplicate, and rerank context.

    This function executes a multistep retrieval strategy:
    1. Rewrites the user's query for optimal vector search.
    2. Queries ChromaDB twice (using both the original and rewritten queries) to maximize recall.
    3. Merges the results to eliminate duplicate chunks.
    4. Uses a frontier LLM to rerank the merged chunks based on true semantic relevance.
    5. Truncates the final list to the top K chunks to optimize the final LLM context window.

    :param original_question: original_question (str): The raw question provided by the user.
    :return: list[Result]: A highly curated, ranked list of the top FINAL_K chunks ready for answer generation.
    """

    # Rewritten question generated by LLM:
    rewritten_question = rewrite_query(original_question)

    # Fetching Chunks from Vector Store using Original Question:
    chunks1 = fetch_context_unranked(question=original_question)

    # Fetching Chunks from Vector Store using Rewritten Question:
    chunks2 = fetch_context_unranked(question=rewritten_question)

    # Merging Chunks from both Questions to avoid Duplication:
    chunks = merge_chunks(chunks1, chunks2)

    # Re-Ranking Chunks:
    reranked_chunks = rerank_chunks(original_question, chunks)

    return reranked_chunks[:FINAL_K]

def make_messages(question, history, chunks):
    """
    Compiles the retrieved chunks, conversation history, and system instructions into the final LLM payload.

    This function acts as the prompt formatter. It extracts the raw text and source
    paths from the top-ranked chunks, injects them into the master system prompt,
    and concatenates the conversation history with the user's current question to
    create a stateful, context-aware message array.

    :param: question (str): The user's original question.
    :param: history (list[dict]): The previous conversation turns, formatted as OpenAI message dictionaries.
    :param: chunks (list[Result]): The final, curated list of top-ranked Result objects.
    :return: list[dict]: A complete message array ready to be sent to the completion API.
    """
    # Joining content from all the chunks into single context string:
    context = '\n\n'.join(
        f"Extract from {chunk.metadata['source']}:\n{chunk.page_content}" for chunk in chunks
    )

    # System Prompt:
    system_prompt= SYSTEM_PROMPT.format(context= context)

    return (
        [{'role': 'system', 'content': system_prompt}] +
        history +
        [{'role': 'user', 'content': question}]
    )

def answer_question(question, history):
    """
    The main execution runtime for generating a RAG-assisted answer.

    This function acts as the primary entry point for the backend system. It orchestrates
    the retrieval of highly relevant documentation, compiles the prompt payload with
    historical context, and executes the final generation call to the frontier model.

    :param: question (str): The raw question provided by the user.
    :param: history (list[dict]): The conversational history of the current chat session.
    :return: tuple: A tuple containing:
            - str: The final, generated text answer from the LLM.
            - list[Result]: The top K chunks used as context, which can be passed
                            to the frontend UI for source citation.
    """
    # Getting Context from Vector Store:
    chunks= fetch_context(question)

    # Making Messages payload for LLM:
    messages = make_messages(question, history, chunks)

    # Response from LLM:
    response = completion(model= MODEL, messages= messages)

    return response.choices[0].message.content, chunks

if __name__ == '__main__':
    reply, chunks = answer_question('What are the main parameters for ChatInterface?', [])
    print(reply)
    print(chunks)