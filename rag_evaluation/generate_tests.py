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

