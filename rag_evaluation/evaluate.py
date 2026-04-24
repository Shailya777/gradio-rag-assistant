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

def evaluate_pipeline():
    """

    :return:
    """

    # Loading the Tests Dataset:
    tests= []
    with open(TEST_FILE, 'r', encoding= 'utf-8') as f:
        for line in f:
            tests.append(json.loads(line))
    print(f'Loaded {len(tests)} test cases.')

# Main Evaluation Loop:
    results_log = []

    for test in tqdm(tests, desc= 'Evaluating RAG Pipeline'):
        question = test['question']
        expected_keywords = test['keywords']
        reference_answer = test['reference_answer']
        category = test['category']

        # Running the Pipeline:
        try:
            # generating Answer from LLM for Test Question:
            generated_answer, retrieved_chunks= answer_question(question, history= [])
        except Exception as e:
            tqdm.write(f'Pipeline Crashed on Question- {question}: Error- {e}')
            continue

        # Keyword Coverage:
        ## Combining all Retrieved Chunks in one String for Easy Search:
        all_retrieved_text = ' '.join([chunk.page_content.lower() for chunk in retrieved_chunks])

        keywords_found = 0
        for keyword in expected_keywords:
            if keyword.lower() in all_retrieved_text:
                keywords_found += 1

        keyword_coverage_pct = (keywords_found / len(expected_keywords)) * 100 if expected_keywords else 0

if __name__ == '__main__':
    evaluate_pipeline()