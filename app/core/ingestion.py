from app.core.vector_store import vector_store
from app.core.embeddings import embed

def embedAndStore(texts):
    embeddings = embed(texts)
    vector_store.add(texts, embeddings)