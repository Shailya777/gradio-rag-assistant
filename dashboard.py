# Imports:
import gradio as gr
import pandas as pd
from pathlib import Path

# Path to Evaluation Results file:
RESULTS_FILE = Path(__file__).parent / 'rag_evaluation/evaluation_results.csv'

def create_stat_card(title, value, suffix= '', color= '#2563eb'):
    """
    Generates a beautiful HTML/CSS card for the metrics.
    :param title: Title of the Metric to display.
    :param value: Value of the Metric to display.
    :param suffix: Suffix to display with Metric Value.
    :param color: Color to display Metric Value in.
    :return: None
    """

    return f"""
    <div style="background-color: white; padding: 20px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05); border: 1px solid #e5e7eb; border-left: 5px solid {color}; text-align: center;">
        <p style="margin: 0; font-size: 14px; color: #6b7280; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;">{title}</p>
        <h2 style="margin: 10px 0 0 0; font-size: 32px; font-weight: 800; color: #111827;">{value:.2f}{suffix}</h2>
    </div>
    """

def load_dashboard_data():
    """
    Loads the CSV, calculates aggregates, and preps the chart data.
    """

    if not RESULTS_FILE.exists():
        empty_card = create_stat_card('Dashboard Results', 0)
        return empty_card, empty_card, empty_card, empty_card, empty_card, empty_card, pd.DataFrame(), pd.DataFrame()

    df = pd.read_csv(filepath_or_buffer= RESULTS_FILE)

    # Overall Metrics for HTML Cards:
    cov_html = create_stat_card(title='Keyword Coverage', value=df['Keyword_Coverage_%'].mean(), suffix='%',
                                color= '#10b981')
    mrr_html = create_stat_card(title= 'Avg MRR', value= df['Mean_Reciprocal_Rank_(MRR)'].mean(), suffix= '', color= '#3b82f6')
    ndcg_html = create_stat_card(title= 'Avg nDCG', value= df['Normalized_Discounted_Cumulative_Gain_(nDCG)'].mean(), suffix= '', color= '#8b5cf6')

    acc_html = create_stat_card(title= 'Accuracy', value= df['Accuracy_Score'].mean(), suffix= ' / 5', color= '#f59e0b')
    comp_html = create_stat_card(title= 'Completeness', value= df['Completeness_Score'].mean(), suffix= ' / 5', color= '#f97316')
    rel_html = create_stat_card(title= 'Relevance', value= df['Relevance_Score'].mean(), suffix= ' / 5', color= '#ef4444')




# Gradio UI:

with gr.Blocks() as demo:
    gr.Markdown('# 📊 RAG Architecture Evaluation Dashboard')
    gr.Markdown('Interactive Visualization of the Advanced Retrieval and Generative Metrics')

    # Loading Metrics:
    df, cov, mrr, ndcg, acc, comp, rel = load_metrics()

    gr.Markdown('### 🔍 Retrieval Performance (Vector Database)')
    with gr.Row():
        gr.Number(value=cov, label="Keyword Coverage (%)", precision=1, interactive=False)
        gr.Number(value=mrr, label="Mean Reciprocal Rank (MRR)", precision=4, interactive=False)
        gr.Number(value=ndcg, label="Normalized DCG (nDCG)", precision=4, interactive=False)

    gr.Markdown('### 🧠 Generation Quality (LLM-as-a-Judge)')
    with gr.Row():
        gr.Number(value=acc, label="Accuracy (out of 5)", precision=2, interactive=False)
        gr.Number(value=comp, label="Completeness (out of 5)", precision=2, interactive=False)
        gr.Number(value=rel, label="Relevance (out of 5)", precision=2, interactive=False)

    gr.Markdown('### 📝 Detailed Evaluation Logs')
    gr.DataFrame(value=df, interactive=False)

if __name__ == '__main__':
    load_dashboard_data()