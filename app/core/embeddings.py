from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')

def embed(texts: list[str]) -> np.ndarray:
    embeddings = model.encode(texts)
    print(len(embeddings)," generated embeddings")
    return np.array(embeddings).astype('float32')