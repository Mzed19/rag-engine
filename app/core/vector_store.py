import faiss
import numpy as np

class VectorStore:
    def __init__(self, dim: int):
        self.index = faiss.IndexFlatIP(dim)  # 🔥 AQUI
        self.documents = []

    def add(self, texts, embeddings):
        embeddings = embeddings.astype("float32")
        faiss.normalize_L2(embeddings)

        self.index.add(embeddings)
        self.documents.extend(texts)

    def search(self, query_embedding, k=5):
        query_embedding = np.array(query_embedding).astype("float32")
        faiss.normalize_L2(query_embedding)

        scores, indices = self.index.search(query_embedding, k)

        return [
            (self.documents[int(i)], float(scores[0][idx]))
            for idx, i in enumerate(indices[0])
            if i != -1
        ]

vector_store = VectorStore(dim=384)