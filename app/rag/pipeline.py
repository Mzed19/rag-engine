from transformers import pipeline
from app.core.embeddings import embed
from app.core.vector_store import vector_store

generator = pipeline(
    "text-generation",
    model="tiiuae/falcon-rw-1b",
    max_new_tokens=80,
    temperature=0.5,  # 🔥 menos aleatório
    top_p=0.9,
    repetition_penalty=1.2,
)

def ask(question: str):
    query_embedding = embed([question])

    results = vector_store.search(query_embedding, k=3)
    print("DOCS:", vector_store.documents)
    print("RESULTS:", results)
    context = "\n".join([text for text, _ in results])

    prompt = f"""
        Responda usando APENAS o contexto abaixo.
        Se não souber, diga "não sei".

        Contexto:
        {context}

        Pergunta: {question}

        Resposta em português, curta:
        """

    response = generator(prompt)[0]["generated_text"]

    return response