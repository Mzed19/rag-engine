from app.core.embeddings import embed
from app.core.vector_store import VectorStore

# Inicializa o vetor de armazenamento FAISS
vector_store = VectorStore(dim=384)

def embedAndStore(texts: list[str]):
    embeddings = embed(texts)
    vector_store.add(texts, embeddings)
