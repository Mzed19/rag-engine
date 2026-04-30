import faiss
import numpy as np

class VectorStore:
    def __init__(self, dim: int):
        self.index = faiss.IndexFlatL2(dim)
        self.documents = []

    def add(self, texts: list[str], embeddings: np.ndarray):
        self.index.add(embeddings)
        self.documents.extend(texts)
        print(len(self.documents), " documents stored in vector store")

    def search(self, query_embedding: np.ndarray, k=5):
        query_embedding = np.array(query_embedding).astype("float32")

        distances, indices = self.index.search(query_embedding, k)

        return [
            self.documents[int(i)]
            for i in indices[0]
            if i != -1
        ]