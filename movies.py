# Versão atualizada para Pinecone SDK v4+
# Autor: Alex Fagundes
# Descrição: Criação e carga de índice "movies" no Pinecone com dados de exemplo

from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
import numpy as np
import time  # 👈 importa o módulo de tempo

# ========================
# CONFIGURAÇÕES
# ========================
PINECONE_API_KEY = "pcsk_6vD99L_QtRtaygMMp7ySFmcAKi5UAKJ6p6AfqszK15wu7RcSAmwos3AA1NiDjCjZFrBoC"
INDEX_NAME = "movies-1000d"
DIMENSION = 384

# ========================
# 1. CONECTAR AO PINECONE
# ========================
pc = Pinecone(api_key=PINECONE_API_KEY)

# Exclui índice antigo (se existir)
if INDEX_NAME in [idx["name"] for idx in pc.list_indexes()]:
    print(f"🗑️ Excluindo índice existente '{INDEX_NAME}'...")
    pc.delete_index(INDEX_NAME)

# Cria índice novo usando ServerlessSpec
print(f"🚀 Criando índice '{INDEX_NAME}' com {DIMENSION} dimensões...")
pc.create_index(
    name=INDEX_NAME,
    dimension=DIMENSION,
    metric="cosine",
    spec=ServerlessSpec(
        cloud="aws",        # pode ser "aws" ou "gcp"
        region="us-east-1"  # ajuste conforme sua região
    )
)

# Conecta ao índice
index = pc.Index(INDEX_NAME)
print(f"✅ Conectado ao índice: {INDEX_NAME}")

# ========================
# 2. MODELO DE EMBEDDING
# ========================
model = SentenceTransformer("intfloat/e5-base-v2") #all-MiniLM-L6-v2

def ajustar_dimensao(vetor, tamanho=DIMENSION):
    vetor = np.array(vetor)
    if len(vetor) > tamanho:
        return vetor[:tamanho].tolist()
    elif len(vetor) < tamanho:
        return np.pad(vetor, (0, tamanho - len(vetor))).tolist()
    return vetor.tolist()

# ========================
# 3. BASE DE DADOS DE FILMES
# ========================
filmes = [
    {"id": "1", "title": "Avatar", "genre": "Ficção Científica", "summary": "Um ex-fuzileiro naval se vê dividido entre seguir ordens humanas e proteger o mundo alienígena de Pandora."},
    {"id": "2", "title": "Titanic", "genre": "Romance / Drama", "summary": "Um artista pobre e uma jovem rica se apaixonam a bordo do infame navio Titanic."},
    {"id": "3", "title": "Matrix", "genre": "Ação / Ficção Científica", "summary": "Um hacker descobre que o mundo real é uma simulação criada por máquinas para controlar a humanidade."},
    {"id": "4", "title": "O Senhor dos Anéis", "genre": "Fantasia / Aventura", "summary": "Um grupo improvável embarca em uma jornada épica para destruir um anel de poder maligno."},
    {"id": "5", "title": "Interestelar", "genre": "Ficção Científica / Drama", "summary": "Um grupo de astronautas viaja por um buraco de minhoca em busca de um novo lar para a humanidade."}
]

# ========================
# 4. INSERIR EMBEDDINGS
# ========================
print("\n🚀 Gerando embeddings e enviando para o Pinecone...")

batch = []
for filme in filmes:
    texto = f"{filme['title']} - {filme['genre']} - {filme['summary']}"
    print("✅ Carregando informações do filme: " + texto)
    embedding = model.encode(texto)
    embedding = ajustar_dimensao(embedding)

    batch.append({
        "id": filme["id"],
        "values": embedding,
        "metadata": filme
    })

index.upsert(vectors=batch)
print("✅ Dados enviados com sucesso!")

WAIT_SECONDS = 10
time.sleep(WAIT_SECONDS)

# ========================
# 5. PESQUISA
# ========================
def pesquisar_filmes(texto_de_busca, top_k=3):
    print(f"\n🔎 Buscando por: '{texto_de_busca}'...")
    query_vector = ajustar_dimensao(model.encode(texto_de_busca))
    results = index.query(vector=query_vector, top_k=top_k, include_metadata=True)

    print("\n🎬 Resultados encontrados:")
    for i, match in enumerate(results["matches"], start=1):
        meta = match["metadata"]
        print(f"\n{i}. 🎞️ {meta['title']}")
        print(f"   🧩 Similaridade: {match['score']:.4f}")
        print(f"   🎭 Gênero: {meta['genre']}")
        print(f"   📝 {meta['summary'][:150]}...")

# ========================
# 6. TESTE
# ========================
pesquisar_filmes("hacker")
pesquisar_filmes("Interestelar")
pesquisar_filmes("Filme sobre navio")
pesquisar_filmes("Filme de terror")
