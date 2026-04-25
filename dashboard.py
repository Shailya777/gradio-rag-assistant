# Imports:
import gradio as gr
import pandas as pd
from pathlib import Path

# Path to Evaluation Results file:
RESULTS_FILE = Path(__file__).parent / 'rag_evaluation/evaluation_results.csv'

def create_stat_card(title, value, suffix= '', color= '#2563eb'):
    """
    Generates a beautiful HTML/CSS card for the metrics.
    :param title:
    :param value:
    :param suffix:
    :param color:
    :return:
    """

    return f"""
    <div style="background-color: white; padding: 20px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05); border: 1px solid #e5e7eb; border-left: 5px solid {color}; text-align: center;">
        <p style="margin: 0; font-size: 14px; color: #6b7280; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;">{title}</p>
        <h2 style="margin: 10px 0 0 0; font-size: 32px; font-weight: 800; color: #111827;">{value:.2f}{suffix}</h2>
    </div>
    """

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

    return df, avg_coverage, avg_mrr, avg_ndcg, avg_accuracy, avg_completeness, avg_relevance

# Gradio UI:

with gr.Blocks() as demo:
    gr.Markdown('# 📊 RAG Architecture Evaluation Dashboard')
    gr.Markdown('Interactive Visualization of the Advanced Retrieval and Generative Metrics')

    # Loading Metrics:
    df, cov, mrr, ndcg, acc, comp, rel = load_metrics()

    gr.Markdown('### 🔍 Retrieval Performance (Vector Database)')
    with gr.Row():
        gr.Number(value= cov, label= "Keyword Coverage (%)", precision= 1, interactive= False)
        gr.Number(value= mrr, label= "Mean Reciprocal Rank (MRR)", precision= 4, interactive= False)
        gr.Number(value= ndcg, label= "Normalized DCG (nDCG)", precision= 4, interactive= False)

    gr.Markdown('### 🧠 Generation Quality (LLM-as-a-Judge)')
    with gr.Row():
        gr.Number(value= acc, label= "Accuracy (out of 5)", precision= 2, interactive= False)
        gr.Number(value= comp, label= "Completeness (out of 5)", precision= 2, interactive= False)
        gr.Number(value= rel, label= "Relevance (out of 5)", precision= 2, interactive= False)

    gr.Markdown('### 📝 Detailed Evaluation Logs')
    gr.DataFrame(value= df, interactive= False)


if __name__ == '__main__':
    demo.launch(inbrowser= True)