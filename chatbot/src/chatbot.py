from loguru import logger
from queue import SimpleQueue, Empty
from typing import Any, Union, Dict, List

from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.schema import LLMResult
from langchain.schema.runnable import RunnablePassthrough

from src.astra import Astra


q = SimpleQueue()
job_done = object() # signals the processing is done

# not using the streaming callback handler currently, only
# works with some types of chains and not all
class StreamingGradioCallbackHandler(BaseCallbackHandler):
    def __init__(self, q: SimpleQueue):
        self.q = q

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        """Run when LLM starts running. Clean the queue."""
        while not self.q.empty():
            try:
                self.q.get(block=False)
            except Empty:
                continue

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Run on new LLM token. Only available when streaming is enabled."""
        self.q.put(token)

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Run when LLM ends running."""
        self.q.put(job_done)

    def on_llm_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        """Run when LLM errors."""
        self.q.put(job_done)

class Chatbot:

    def __init__(self):
        self.model = model = "/models/mistral-7b-instruct-v0.1.Q4_K_M.gguf"
        
        template = """You are a helpful question answering bot.
        Use the following pieces of retrieved context to answer the question.
        If the retrieved context does not answer the question say "I don't know". 
        You provide correct answers to questions. 
        
        Context: {context}
        Question: {question}
        """

        prompt = PromptTemplate(
            template=template, 
            input_variables=["context", "question"]
        )

        callback_manager = CallbackManager([StreamingGradioCallbackHandler(q)])

        # if you run out of memory, adjust n_batch and n_gpu_layers
        self.llm = LlamaCpp(
            model_path=model,
            temperature=0,
            max_tokens=2000,
            top_p=1,
            streaming=True,
            callback_manager=callback_manager, 
            verbose=True, 
            #n_batch=1024, # adjust as needed
            #n_gpu_layers=8, # adjust as needed
            n_ctx=1024, # context size
        )

        db = Astra()
        self.retriever = db.get_vectorstore().as_retriever(k=3)

        # option 1: simple RAG
        self.rag_chain = {
            "context": db.get_vectorstore().as_retriever(k=3),
            "question": RunnablePassthrough()
        } | prompt | self.llm
        
    def respond(self, history):
        question = history[-1][0]
        history[-1][1] = self.rag_chain.invoke(question)

        docs = self.retriever.get_relevant_documents(question)
        output_str = ""
        for doc in docs:
            output_str += doc.metadata["title"] + ": " + doc.metadata["url"] + "\n"
        logger.info(output_str)

        return history
