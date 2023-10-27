from langchain.llms import LlamaCpp
from langchain.prompts import PromptTemplate

# initialize once
model = "/models/mistral-7b-instruct-v0.1.Q4_K_M.gguf"
template = """Question: {question}
Answer: Let's work this out in a step by step way to be sure we have the right answer."""
prompt = PromptTemplate(template=template, input_variables=["question"])

llm = LlamaCpp(
    model_path=model,
    temperature=0.75,
    max_tokens=2000,
    top_p=1,
    verbose=False, 
)
  

def respond(message, chat_history):
    output = llm(message)
    chat_history.append((message, output))

    return "", chat_history
