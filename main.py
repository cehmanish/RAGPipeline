from web_scraper import WebScraper
from vector_store import VectorStore
from flask import Flask, request


app = Flask(__name__)

@app.route("/")
def hello():
    return "Hello World"

@app.route("/process", methods=["GET"])
def process():
    # Scraping the Content from the Web 
    source_url = request.args.get("url")
    scraper = WebScraper()
    result = scraper.scrape(source_url)

    # Storing Text into Vector Database 
    store = VectorStore()
    # Add a test document
    content = result['content']
    metadata = {
        'url': source_url,
        'title': result['metadata']['title'],
    }
    chunks_added = store.add_document(content, metadata)
    print(f"Added {chunks_added} chunks")
    return "process completed"


@app.route("/search", methods=["GET"])
def search():
    query = request.args.get("query")
    store = VectorStore()
    results = store.search(query)
    return results

@app.route("/status")
def status():
    store = VectorStore()
    stats = store.get_stats()
    return stats

if __name__ == "__main__":
    app.run(debug=True)