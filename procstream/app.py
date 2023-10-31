import fire
from loguru import logger
import orjson
import os
import pulsar
import time

from src.generate_embeddings import GenerateEmbeddings

def process_doc_stream() -> None:
    service_url = os.environ.get("PULSAR_URL")
    token = os.environ.get("PULSAR_TOKEN")
    pulsar_topic = os.environ.get("PULSAR_FULL_TOPIC")

    client = pulsar.Client(service_url,
                           authentication=pulsar.AuthenticationToken(token))

    consumer = client.subscribe(pulsar_topic, 'consumer')

    embedding_service = GenerateEmbeddings()

    waiting_for_msg = True
    while waiting_for_msg:
        try:
            msg = consumer.receive()
            data = orjson.loads(msg.data())
            embedding_service.embed(data["title"], data["url"])
            consumer.acknowledge(msg)
        except Exception as e:
            logger.warning(f"{e}")
            waiting_for_msg = False

        # sleep so wikipedia doesn't block us
        time.sleep(1)

    client.close()

if __name__ == "__main__":
    fire.Fire(process_doc_stream)