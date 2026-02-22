# 🎬 Movie Semantic Search com Pinecone

Projeto de busca semântica utilizando embeddings e indexação vetorial com Pinecone.

Este projeto demonstra como transformar dados estruturados (filmes) em embeddings vetoriais e realizar buscas inteligentes baseadas em similaridade semântica.

---

## 🚀 Tecnologias Utilizadas

- Python 3.x
- Pinecone (Vector Database)
- Sentence Transformers
- python-dotenv

---

## 📌 Como Funciona

O fluxo do projeto é dividido em duas etapas principais:

### 1️⃣ Indexação (`index_movies.py`)

- Criação ou conexão com um índice no Pinecone
- Carregamento da base de filmes
- Geração de embeddings (título + gênero + resumo)
- Envio dos vetores para o Pinecone

### 2️⃣ Busca Semântica (`movies.py`)

- Recebe um termo de busca (simulando um prompt)
- Converte o termo em embedding
- Realiza busca vetorial no índice
- Retorna os filmes mais similares com base na similaridade semântica

---

## 🧠 Exemplo de Busca

Buscando por: "hacker"

Resultados encontrados:

Matrix
Similaridade: 0.8565
Gênero: Ação / Ficção Científica
Um hacker descobre que o mundo real é uma simulação criada por máquinas...
