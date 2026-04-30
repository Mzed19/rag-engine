from app.core.embeddings import embed
from app.core.vector_store import vector_store

# Inicializa o vetor de armazenamento FAISS

def embedAndStore(texts: list[str]):
    embeddings = embed(texts)
    vector_store.add(texts, embeddings)
