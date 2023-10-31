import unittest

from langchain.embeddings import LocalAIEmbeddings
import requests

class TestLangchainIntegration(unittest.TestCase):
    """Note file runs an integration between langchain and the
    locally running embedding service.
    """

    def setUp(self):
        self.api_url = "http://127.0.0.1:8000"
        self.embedding_model = LocalAIEmbeddings(
            openai_api_key="no_key",
            openai_api_base=self.api_url,
            model="instructor-base",
            max_retries=1,
        )

    def test_embedding_api(self):
        """Test calling the API directly and ensure it returns data that matches
        OpenAI spec.
        """
        response = requests.post(
            self.api_url + "/embeddings", 
            json={"input": ["this is a document embedding test"], "model": "model_name"})
        r = response.json()

        self.assertCountEqual(r.keys(), ["object", "data", "model"])
        self.assertCountEqual(r["data"][0].keys(), ["object", "embedding", "index"])
        self.assertEqual(type(r["data"][0]["embedding"]), list)
        self.assertEqual(len(r["data"]), 1) 

    def test_langchain_embed_documents(self):
        embedding = self.embedding_model.embed_documents(["this is a document embedding test", 
                                                        "a 2nd test document."])
        self.assertEqual(len(embedding), 2)
        self.assertEqual(type(embedding[0]), list)
        self.assertEqual(type(embedding[0][0]), float)

    def test_langchain_embed_query(self):
        embedding = self.embedding_model.embed_query("this is a query embedding test")
        self.assertEqual(type(embedding), list)
        self.assertEqual(type(embedding[0]), float)
