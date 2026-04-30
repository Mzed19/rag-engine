from app.core.ingestion import Ingestion
from app.core.vector_store import VectorStore
from app.core.embeddings import embed

ingestion = Ingestion()
store = VectorStore(dim=384)

texts = [
    "Gato",
    "Cachorro",
    "Edifício"
]
    
ingestion.ingest(texts)

searched = store.search(embed(["Gato"]))
print(searched)