# Imports:
import os
import math
import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from pydantic import BaseModel, Field
from litellm import completion
from dotenv import load_dotenv
from rag_implementation.answer_question import answer_question

# Constants:
load_dotenv(override= True)
JUDGE_MODEL = 'openai/gpt-4o'
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

    feedback: str = Field(
        description= '1 sentence explaining the scores.'
    )

def calculate_mrr(keywords, retrieved_chunks):
    """
    Calculates Mean Reciprocal Rank (how high up was the first good result?)
    :param keywords: Keywords to search for
    :param retrieved_chunks: Retrieved Chunks
    :return: MRR Score
    """

    mrr_scores= []

    for keyword in keywords:

        for rank, chunk in enumerate(retrieved_chunks, start= 1):
            if keyword.lower() in chunk.page_content.lower():
                mrr_scores.append(1.0 / rank)
                break # Keyword Found! Move to the next keyword.
        else:
            mrr_scores.append(0.0) # Keyword wasn't in any chunk

    return sum(mrr_scores) / len(mrr_scores) if mrr_scores else 0.0

def calculate_ndcg(keywords, retrieved_chunks, k= 10):
    """
    Calculates nDCG (Normalized Discounted Cumulative Gain)(did we put the best results at the very top?)
    :param keywords: Keywords to search for
    :param retrieved_chunks: Retrieved Chunks
    :param k: Number of Chunks to Search in
    :return: nDCG Score
    """

    ndcg_scores= []

    for keyword in keywords:

        # Creating a list of 1s and 0s (1 if chunk has the keyword, 0 if not)
        relevance = [1 if keyword.lower() in chunk.page_content.lower() else 0 for chunk in retrieved_chunks[:k]]

        # Calculating DCG (Discounted Cumulative Gain):
        dcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(relevance))

        # Calculate Ideal DCG (What if all the 1s were perfectly at the top?)
        ideal_relevance = sorted(relevance, reverse=True)
        icdg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal_relevance))

        ndcg_scores.append(dcg / icdg if icdg > 0 else 0.0)

    return sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0

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

        # Calculating MRR:
        mrr_scores = calculate_mrr(keywords= expected_keywords, retrieved_chunks= retrieved_chunks)

        # Calculating nDCG:
        ndcg_scores = calculate_ndcg(keywords= expected_keywords, retrieved_chunks= retrieved_chunks)

        # Answer Grading:
        judge_llm_system_prompt= '''
        You are an impartial, expert AI grader evaluating a RAG pipeline.
        Compare the Generated Answer to the Reference Answer. 
        Grade strictly on a scale of 1 to 5. 
        Respond ONLY in the requested JSON format.
        '''

        judge_llm_user_prompt= f'''
        User Question: {question}
        
        Perfect Reference Answer: {reference_answer}
        
        Generated Answer: {generated_answer}
        '''

        try:
            judge_llm_response = completion(
                model= JUDGE_MODEL,
                messages= [
                    {'role': 'system', 'content': judge_llm_system_prompt},
                    {'role': 'user', 'content': judge_llm_user_prompt},
                ],
                response_format= AnswerEval,
            )

            # Parsing the Response to check if it is in requested format:
            eval_data= AnswerEval.model_validate_json(judge_llm_response.choices[0].message.content)

            # Logging the Results:
            results_log.append({
                'Category': category,
                'Question': question,
                'Keyword_Coverage_%': keyword_coverage_pct,
                'Accuracy_Score': eval_data.accuracy,
                'Completeness_Score': eval_data.completeness,
                'Relevance_Score': eval_data.relevance,
                'Mean_Reciprocal_Rank_(MRR)': mrr_scores,
                'Normalized_Discounted_Cumulative_Gain_(nDCG)': ndcg_scores,
                'Judge_Feedback': eval_data.feedback,
            })

        except Exception as e:
            tqdm.write(f'Judge Failed on Question- {question}: Error- {e}')
            continue

    # Saving Evaluation Results to Dataframe:
    df = pd.DataFrame(results_log)

    # Writing Evaluation data to CSV File, deleting the file if already exists:
    if RESULTS_FILE.exists():
        RESULTS_FILE.unlink()
    df.to_csv(RESULTS_FILE, index= False)

    # Evaluation Data:
    print("\n" + "=" * 50)
    print("📊 EVALUATION COMPLETE 📊")
    print("=" * 50)
    print(f"Average Keyword Coverage: {df['Keyword_Coverage_%'].mean():.1f}%")
    print(f"Average Accuracy:         {df['Accuracy_Score'].mean():.2f} / 5.0")
    print(f"Average Completeness:     {df['Completeness_Score'].mean():.2f} / 5.0")
    print(f"Average Relevance:        {df['Relevance_Score'].mean():.2f} / 5.0")
    print(f"Average MRR:              {df['Mean_Reciprocal_Rank_(MRR)'].mean():.4f}")
    print(f"Average nDCG:             {df['Normalized_Discounted_Cumulative_Gain_(nDCG)'].mean():.4f}")
    print(f"\nDetailed results saved to {RESULTS_FILE.name}")

if __name__ == '__main__':
    evaluate_pipeline()