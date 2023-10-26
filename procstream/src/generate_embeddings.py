from loguru import logger
import os

import cassio
from langchain.embeddings import LocalAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Cassandra

from newspaper import Article

class GenerateEmbeddings:

    def __init__(self):
        self.session = self._connect_to_astra()
        self.embedding_model = self._init_embedding_model()        
        self.ASTRA_TABLE = os.environ.get("ASTRA_TABLE")
        self.ASTRA_KEYSPACE = os.environ.get("ASTRA_KEYSPACE")

        self.vectorstore = Cassandra(
            embedding=self.embedding_model,
            session=self.session,
            keyspace=self.ASTRA_KEYSPACE,
            table_name=self.ASTRA_TABLE,
        )

        # can replace with a more sophisticated text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=80,
        )

        logger.info(f"Initialized langchain embedding service with Astra DB")
        logger.info(f"ASTRA_TABLE={self.ASTRA_TABLE}")
        logger.info(f"ASTRA_KEYSPACE={self.ASTRA_KEYSPACE}")

    def _connect_to_astra(self):
        ASTRA_TOKEN = os.environ.get("ASTRA_TOKEN")
        ASTRA_DATABASE_ID = os.environ.get("ASTRA_DATABASE_ID")

        cassio.init(token=ASTRA_TOKEN, database_id=ASTRA_DATABASE_ID)
        logger.info(f"Connected to Astra DB: {ASTRA_DATABASE_ID}")
        return cassio.config.resolve_session()

    def _init_embedding_model(self):
        EMBEDDING_MODEL_API_KEY = os.environ.get("EMBEDDING_MODEL_API_KEY")
        EMBEDDING_MODEL_API_BASE = os.environ.get("EMBEDDING_MODEL_API_BASE")
        EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL")

        return LocalAIEmbeddings(
            openai_api_key=EMBEDDING_MODEL_API_KEY,
            openai_api_base=EMBEDDING_MODEL_API_BASE,
            model=EMBEDDING_MODEL,
            max_retries=3,
        )

    @staticmethod
    def _get_article_text(url: str) -> str:
        article = Article(url)
        article.download()
        article.parse()
        return article.text

    def embed(self, title: str, url: str) -> None:
        """Chunk, embed, and store a webpage in Astra DB (serveless Cassandra)

        :param title: title of the document
        :param url: url of the document
        """
        full_text = self._get_article_text(url)
        chunks = self.text_splitter.split_text(text=full_text)
        
        # add the title and url for the article to each chunk as metadata
        metadata = {"title": title, "url": url}
        metadatas = [metadata for _ in range(len(chunks))]

        self.vectorstore.add_texts(texts=chunks, metadatas=metadatas)
        logger.info(f"Stored {len(chunks)} vectors for article: {title}.")