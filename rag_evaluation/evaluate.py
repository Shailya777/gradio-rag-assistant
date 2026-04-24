# Imports:
import os
import json
from pathlib import Path
from tqdm import tqdm
from pydantic import BaseModel, Field
from litellm import completion
from dotenv import load_dotenv
from rag_implementation.answer_question import answer_question

# Constants:
load_dotenv(override= True)
JUDGE_MODEL = 'openai/gpt-4.1-nano'
TEST_FILE = Path(__file__).parent / 'tests.jsonl'
RESULTS_FILE = Path(__file__).parent / 'evaluation_results.csv'

class AnswerEval(BaseModel):
    """
    Schema for LLM-as-a-judge evaluation of answer.
    """
    accuracy: int = Field(
        description= 'Score 1-5: Is the generated answer factually correct compared to the reference answer? 1 is completely wrong, 5 is perfect.'
    )

    completeness: int = Field(
        description= 'Score 1-5: Did it include all necessary information from the reference answer?'
    )

    relevance: int = Field(
        description= 'Score 1-5: Did it directly answer the prompt without rambling?'
    )

    feedback: int = Field(
        description= '1 sentence explaining the scores.'
    )
