import fire
from loguru import logger
import orjson
import os
import pulsar
import requests

def list_wikipedia_articles(num_articles: int = 100) -> list[dict]:
    """Use the wikipedia API to get a list of random articles
    
    :returns: list of dicts with title and url fields for each article
    """

    wikipedia_session = requests.Session()
    wp_api_url = "https://en.wikipedia.org/w/api.php"
    api_params = {
        "action": "query",
        "format": "json",
        "generator": "random",
        "grnnamespace": "0",
        "prop": "categories",
        "grnlimit": num_articles,
    }

    response = wikipedia_session.get(url=wp_api_url, params=api_params)
    data = response.json()
    results = data["query"]["pages"]
    return [ 
        {
            "title": v["title"], 
            "url": f"https://en.wikipedia.org/?curid={k}"
        } 
        for k, v in results.items()]


def stream_wikipedia_docs() -> None:
    # connect to pulsar stream
    service_url = os.environ.get("PULSAR_URL")
    token = os.environ.get("PULSAR_TOKEN")
    pulsar_topic = os.environ.get("PULSAR_FULL_TOPIC")

    # load token
    client = pulsar.Client(service_url,
                           authentication=pulsar.AuthenticationToken(token))

    producer = client.create_producer(pulsar_topic)

    try:
        while True:
            articles = list_wikipedia_articles()
            for article in articles:
                producer.send(orjson.dumps(article))    
            logger.info(f"Added {len(articles)} articles to stream.")
    except Exception as e:
        logger.warning(f"{e}")
        client.close()

if __name__ == "__main__":
    fire.Fire(stream_wikipedia_docs)