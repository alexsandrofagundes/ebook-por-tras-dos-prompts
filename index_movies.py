# index_movies.py
# Versão atualizada para Pinecone SDK v4+
# Autor: Alex Fagundes
# Descrição: Criação e carga de índice "movies" no Pinecone com dados de exemplo

from pinecone import Pinecone, ServerlessSpec
import os
import time  # 👈 importa o módulo de tempo

# =============== CONFIGURAÇÃO ===============
API_KEY = "pcsk_6vD99L_QtRtaygMMp7ySFmcAKi5UAKJ6p6AfqszK15wu7RcSAmwos3AA1NiDjCjZFrBoC"
INDEX_NAME = "movies-index"
DIMENSION = 1000  # tamanho do vetor (número de dimensões)
REGION = "us-east-1"  # pode ajustar conforme a sua conta

# =============== INICIALIZAÇÃO ===============
pc = Pinecone(api_key=API_KEY)

# =============== CRIAÇÃO DO ÍNDICE ===============
# Lista índices existentes
existing_indexes = [index.name for index in pc.list_indexes()]

# Cria se ainda não existir
if INDEX_NAME not in existing_indexes:
    print(f"🆕 Criando índice '{INDEX_NAME}'...")
    pc.create_index(
        name=INDEX_NAME,
        dimension=DIMENSION,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=REGION)
    )
    print(f"✅ Índice '{INDEX_NAME}' criado com sucesso!")
else:
    print(f"ℹ️ Índice '{INDEX_NAME}' já existe.")

# =============== CONEXÃO AO ÍNDICE ===============
index = pc.Index(INDEX_NAME)

# =============== DADOS DE EXEMPLO ===============
# Vetores de exemplo simulando embeddings de filmes
sample_movies = [
    {
        "id": "movie_1",
        "values": [0.1] * DIMENSION,
        "metadata": {"title": "Inception", "genre": "Sci-Fi", "year": 2010}
    },
    {
        "id": "movie_2",
        "values": [0.2] * DIMENSION,
        "metadata": {"title": "The Dark Knight", "genre": "Action", "year": 2008}
    },
    {
        "id": "movie_3",
        "values": [0.3] * DIMENSION,
        "metadata": {"title": "Interstellar", "genre": "Adventure", "year": 2014}
    }
]

# =============== INSERÇÃO DOS VETORES ===============
print("🎬 Inserindo vetores no índice...")
index.upsert(vectors=sample_movies)
print("✅ Vetores inseridos com sucesso!")

# =============== TESTE DE CONSULTA ===============
# Consulta de similaridade (simulando embedding de entrada)
query_vector = [0.15] * DIMENSION

WAIT_SECONDS = 10
time.sleep(WAIT_SECONDS)

print("🔍 Buscando filmes similares...")
results = index.query(
    vector=query_vector,
    top_k=2,
    include_metadata=True
)

# Exibe os resultados
print("\n🎯 Resultados da busca:")
for match in results["matches"]:
    print(f"🎞️ {match['metadata']['title']} ({match['metadata']['year']}) - Score: {match['score']:.3f}")

print("\n🏁 Finalizado com sucesso!")
