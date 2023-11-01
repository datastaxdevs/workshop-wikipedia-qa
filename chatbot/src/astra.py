import cassio
from langchain.embeddings import LocalAIEmbeddings
from langchain.vectorstores import Cassandra

from loguru import logger
import os


class Astra:
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
    
    def _connect_to_astra(self):
        ASTRA_TOKEN = os.environ.get("ASTRA_TOKEN")
        ASTRA_DATABASE_ID = os.environ.get("ASTRA_DATABASE_ID")

        cassio.init(token=ASTRA_TOKEN, database_id=ASTRA_DATABASE_ID)
        logger.info(f"Connected to Astra DB: {ASTRA_DATABASE_ID}")
        return cassio.config.resolve_session()
    
    def _init_embedding_model(self):
        # embedding model used to encode queries
        EMBEDDING_MODEL_API_KEY = os.environ.get("EMBEDDING_MODEL_API_KEY")
        EMBEDDING_MODEL_API_BASE = os.environ.get("EMBEDDING_MODEL_API_BASE")
        EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL")

        return LocalAIEmbeddings(
            openai_api_key=EMBEDDING_MODEL_API_KEY,
            openai_api_base=EMBEDDING_MODEL_API_BASE,
            model=EMBEDDING_MODEL,
            max_retries=3,
        )
    
    def get_vectorstore(self):
        return self.vectorstore
