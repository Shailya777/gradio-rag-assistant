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
