# wikipedia_demo
Real-time document Q&amp;A using Pulsar, Cassandra, LangChain, and open-source language models.

Don't want to complete the exercises?  The complete working code is available on the `complete` branch.

## Project overview

This workshop code runs a Retrieval Augmented Generation (RAG) application stack that takes data from Wikipedia, stores it in a vector database (Astra DB), and provides a chat interface for asking questions about the Wikipedia documents.

The project uses Astra Streaming (serverless Apache Pulsar) and Astra DB (serverless Apache Cassandra) and 4 microservices built using:

- Python
- LangChain for the LLM framework
- Open source Instructor Embedding model
- Open source Mistral 7B LLM
- Gradio for a simple chat web UI
- Fast API to provide the document embedding service
  
## Running the project

The project consists of 4 microservices

- `docstream` Gets random Wikipedia articles in English and adds them to a Pulsar topic for processing
- `embeddings` A RESTful API service that turns text into embeddings.
- `procstream` Consumes articles from the Pulsar topic, scrapes the webpage to get the full text, generates embeddings, and stores in Astra DB
- `chatbot` Provides both the UI for the chatbot and the agent code for running the chatbot

### With docker

`docker compose up --build`

Individual services can also be started directly.  Note that `procstream` and `chatbot` require that the `embeddings` microservice is running.  

- `docker compose up --build docstream`
- `docker compose up --build embeddings`
- `docker compose up --build procstream`
- `docker compose up --build chatbot`
  
### Without docker

If you do not wish to run with docker, you can run each of the 4 microservices separately. Use pip to install the requirements for each microservice and then run it directly with python.

```
cd docstream
pip install -r requirements.txt
python app.py
```

```
cd embeddings
pip install -r requirements.txt
gunicorn --workers 1 -k uvicorn.workers.UvicornWorker app:app --bind 0.0.0.0:8000
```

```
cd procstream
pip install -r requirements.txt
python app.py
```

```
cd chatbot
pip install -r requirements.txt
python app.py
```

## Using the services

You can access the embeddings API in your Chrome browser at http://127.0.0.1:8000/docs.

The chatbot can be opened in your Chrome browser at http://127.0.0.1:7860. 
