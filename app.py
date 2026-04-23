import gradio as gr
from rag_implementation.answer_question import  answer_question

def chat(message, history):
    """
    Acts as the bridge between gradio's UI and RAG backend.
    """

    # Getting Answer and Context (Chunks) from LLM:
    reply_text, chunks = answer_question(message, history)

    # Building a Markdown Accordion to Display Context:
    if chunks:
        sources_md= "\n\n<details>\n<summary> <b>View Retrieved Sources</b></summary>\n\n"

        # Looping Through found Chunks:
        for i, chunk in enumerate(chunks):
            file_name = chunk.metadata.get('source', 'Unknown File')
            #print(file_name)
            clean_file_name = file_name.split('/')[-1]

            sources_md += f"**{i+1}. {clean_file_name}**\n"

            # Preview Text from Chunk:
            preview_text = chunk.page_content[:150].replace('\n', ' ')

            # Adding Preview Text:
            sources_md += f"> {preview_text}...\n\n"

        sources_md += "</details>"

        # Adding Source Information to LLM's Reply:
        reply_text += sources_md

    return reply_text

# gradio UI:
demo = gr.ChatInterface(
    fn= chat,
    title= 'Gradio Assistant',
    description='''
    I am an expert Python developer assistant. 
    Ask me how to build, layout, or debug your Gradio applications!
    ''',
    examples= [
        "How do I add a button to a Blocks layout?",
        "What are the main parameters for ChatInterface?",
        "How do I process an uploaded image?"
    ]

)

if __name__ == '__main__':
    demo.launch(inbrowser= True)