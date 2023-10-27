from queue import SimpleQueue, Empty
from threading import Thread
from typing import Any, Union, Dict, List

from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.schema import LLMResult
from langchain.chains import LLMChain

q = SimpleQueue()
job_done = object() # signals the processing is done

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
        template = """{question}"""
        self.prompt = PromptTemplate(template=template, input_variables=["question"])

        callback_manager = CallbackManager([StreamingGradioCallbackHandler(q)])

        llm = LlamaCpp(
            model_path=model,
            temperature=0.75,
            max_tokens=2000,
            top_p=1,
            streaming=True,
            callback_manager=callback_manager, 
            verbose=True, 
        )
  
        self.chain = LLMChain(
            llm=llm,
            prompt=self.prompt,
        )

    def respond(self, history):
        question = history[-1][0]
        thread = Thread(target=self.chain.run, kwargs={"question": question})
        thread.start()
        history[-1][1] = ""
        while True:
            next_token = q.get(block=True)
            if next_token is job_done:
                break
            history[-1][1] += next_token
            yield history
        thread.join()
