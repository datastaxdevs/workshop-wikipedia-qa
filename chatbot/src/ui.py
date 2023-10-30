import gradio as gr
from loguru import logger

from src.chatbot import Chatbot

def feedback(data: gr.LikeData):
    if data.liked:
        logger.info("User liked the response.")
    else:
        logger.info("User downvoted the response.")
    logger.info(data.value)

def user(user_message, history):
    return "", history + [[user_message, None]]

cb = Chatbot()

with gr.Blocks() as chat_app:
    chatbox = gr.Chatbot()
    textbox = gr.Textbox(show_label=False, placeholder="Ask questions to Wikipedia.")
    clear = gr.ClearButton([textbox, chatbox])
    chatbox.like(feedback, None, None)

    textbox.submit(user, [textbox, chatbox], [textbox, chatbox], queue=False).then(
        cb.respond, chatbox, chatbox)

    clear.click(lambda: None, None, chatbox, queue=False)