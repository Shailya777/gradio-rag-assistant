# Imports:
import gradio as gr
import pandas as pd
from pathlib import Path

# Path to Evaluation Results file:
RESULTS_FILE = Path(__file__).parent / 'rag_evaluation/evaluation_results.csv'

def load_metrics():
    """
    Loads the CSV and calculates the aggregate metrics.
    """

    if not RESULTS_FILE.exists():
        return pd.DataFrame(), 0, 0, 0, 0, 0, 0

    df = pd.read_csv(filepath_or_buffer= RESULTS_FILE)

    # Calculating Averages:
    avg_coverage = df['Keyword_Coverage_%'].mean()
    avg_mrr = df['Mean_Reciprocal_Rank_(MRR)'].mean()
    avg_ndcg = df['Normalized_Discounted_Cumulative_Gain_(nDCG)'].mean()
    avg_accuracy = df['Accuracy_Score'].mean()
    avg_completeness = df['Completeness_Score'].mean()
    avg_relevance = df['Relevance_Score'].mean()

    return avg_coverage, avg_mrr, avg_ndcg, avg_accuracy, avg_completeness, avg_relevance

if __name__ == '__main__':
    a,s,d,f,g,h= load_metrics()
    print(a,s,d,f,g,h)