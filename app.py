import gradio as gr
from rag_implementation.answer_question import  answer_question

def chat(message, history):
    """
    Acts as the bridge between gradio's UI and RAG backend.
    """

    reply_text, chunks = answer_question(message, history)
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