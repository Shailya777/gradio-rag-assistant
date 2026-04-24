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

