from app.rag.ingest import embedAndStore
from app.rag.pipeline import ask

texts = [
    "Gatos são animais domésticos conhecidos por sua independência",
    "Felinos são animais da mesma família dos gatos",
    "Cães são animais domésticos conhecidos por sua lealdade",
    "Equinos são animais de grande porte usados para transporte"
]

embedAndStore(texts)

response = ask("O que são gatos?")

print("\n🧠 RESPOSTA:\n")
print(response)