from app.core.vector_store import vector_store
from app.core.embeddings import embed
from app.core.ingestion import embedAndStore

texts = [
    "Os equinos são animais de grande porte",
    "Felinos são animais domésticos populares, conhecidos por sua independência e habilidades de caça",
    "Gatos são conhecidos por sua agilidade e independência",
    "Existem diversos tipos de cães, como o pastor alemão, o labrador e o buldogue",
    "Felinos são animais da mesma família dos gatos",
    "Um gato foi atropelado ontem na rua principal",
]

embedAndStore(texts)

searched = vector_store.search(embed(["Me explique sobre gatos"]))
print(searched)