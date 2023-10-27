import gradio as gr
from loguru import logger

from src.chatbot import respond

def feedback(data: gr.LikeData):
    if data.liked:
        logger.info("User liked the response.")
    else:
        logger.info("User downvoted the response.")

with gr.Blocks() as chat_app:
    chatbot = gr.Chatbot()
    textbox = gr.Textbox(show_label=False, placeholder="Ask questions to Wikipedia.")
    clear = gr.ClearButton([textbox, chatbot])
    chatbot.like(feedback, None, None)

    textbox.submit(respond, [textbox, chatbot], [textbox, chatbot])
    textbox.submit(lambda x: gr.update(value=''), [],[textbox])
