from app.rag.ingest import embedAndStore

class Ingestion:     
    def ingest(self, texts: list[str]):
        print(len(texts), " texts sended to ingestion")
        embedAndStore(texts)