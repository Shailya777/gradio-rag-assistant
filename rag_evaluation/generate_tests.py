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

def generate_tests():
    """
    Feeds chunks to the LLM to generate synthetic test questions.
    """
    print(f'Sampling {NUM_TESTS_TO_GENERATE} documents to generate tests...')
    chunks= get_random_markdown_chunks(NUM_TESTS_TO_GENERATE)

    generated_tests = []

    for i, chunk in enumerate(chunks):
        print(f'Generating question {i + 1}/{NUM_TESTS_TO_GENERATE}...')

        system_prompt = """
        You are an expert QA engineer creating a test dataset for a Python gradio RAG system.
        I will give you a chunk of gradio documentation. 
        Your job is to read it, and generate ONE difficult, highly specific test question that can be answered by this text.
        Respond ONLY in valid JSON matching the provided schema.
        """

        user_prompt = f"DOCUMENTATION CHUNK: \n\n{chunk}"

        try:
            response = completion(
                model= MODEL,
                messages= [
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': user_prompt}
                    ],
                response_format= TestItem
            )

            # Parsing the json Response:
            reply = response.choices[0].message.content
            test_data = TestItem.model_validate_json(reply)
            generated_tests.append(test_data.model_dump())

        except Exception as e:
            print(f'Skipping Chunk due to Error: {e}')

    # Saving Generated Tests to JSONL (JSON Lines) file:
    with open(TEST_FILE_PATH, 'w', encoding="utf-8") as f:
        for test in generated_tests:
            f.write(json.dumps(test) + '\n')

    print(f'\nSuccessfully Generated {len(generated_tests)} Tests and saved to {TEST_FILE_PATH.name}\n')

if __name__ == '__main__':
    generate_tests()