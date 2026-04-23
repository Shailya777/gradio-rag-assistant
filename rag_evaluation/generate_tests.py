# Imports:
import os
import json
import random
from pathlib import Path
from litellm import completion
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Constants:
load_dotenv(override= True)
MODEL= 'ollama/llama3.1:8b'
KNOWLEDGE_BASE_PATH = Path(__file__).parent.parent / 'knowledge-base'
TEST_FILE_PATH= Path(__file__).parent / 'tests.jsonl'
NUM_TESTS_TO_GENERATE = 100

class TestItem(BaseModel):
    """
    Schema for forcing the LLM to output perfect test data.
    """
    question: str = Field(
        description= 'A highly specific question that can be answered by the text.'
    )

    keywords: list[str] = Field(
        description= '3-5 critical words that MUST appear in the retrieved context.'
    )

    reference_answer: str = Field(
        description= 'The exact, correct answer based ONLY on the text.'
    )

    category: str = Field(
        description= 'Must be one of: direct_fact, conceptual, code_syntax, troubleshooting'
    )

def get_random_markdown_chunks(num_chunks):
    """
    Grabs a random sample of Markdown files from the Knowledge Base.
    :param num_chunks: Number of chunks to return.
    :return: Chunks of Markdown files.
    """

    all_files= list(KNOWLEDGE_BASE_PATH.rglob('*.md'))
    random_sampled_files = random.sample(all_files, min(num_chunks, len(all_files)))

    chunks = []

    for file in random_sampled_files:
        with open(file, 'r', encoding="utf-8") as f:
            text = f.read()
            chunks.append(text[:2000]) # If the file is huge, grab the first 2000 characters to keep context clean

    return chunks

if __name__ == '__main__':
    x = get_random_markdown_chunks(2)
    print(x)