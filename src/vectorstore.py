"""Embeddings e armazenamento vetorial - Requisito 1 (pág. 6).

EMBEDDINGS: `paraphrase-multilingual-MiniLM-L12-v2` (Sentence-BERT).
A Documento de Aula 4 recomenda "BERTimbau ou Sentence-BERT em português" para
qualidade superior, e este modelo é o usado no código da Aula 6. Gera vetores
de 384 dimensões; a 1IAST Aula 2 lembra que mais dimensões custam mais
infraestrutura sem garantia de ganho, e que a escolha é de equilíbrio.

ARMAZENAMENTO: ChromaDB, o banco do hands-on da POSTECH Aula 3. Persiste em
disco (reaproveitável entre execuções) e indexa com HNSW.

Duas escolhas de configuração com justificativa nas aulas:

* Métrica de cosseno. "É a mais utilizada em NLP" (POSTECH Aula 3) e ignora
  a magnitude - que em texto varia com o tamanho do comentário, não com o
  significado.
* Sem chunking. A 1IAST Aula 2 explica que o chunking existe porque
  "documentos extensos raramente cabem inteiros na janela de contexto" e que
  "chunks curtos demais podem fragmentar contexto essencial". Aqui o
  comentário mediano tem 53 caracteres e o maior, 208: cada avaliação já é a
  menor unidade de sentido. Dividir só destruiria contexto.
"""
from __future__ import annotations

import logging
import shutil

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from src.config import COLECAO_CHROMA, DIR_CHROMA, MODELO_EMBEDDING

logger = logging.getLogger(__name__)
LOTE = 2000


def carregar_embeddings(dispositivo: str | None = None) -> HuggingFaceEmbeddings:
    """Instancia o modelo de embeddings, usando GPU quando disponível."""
    if dispositivo is None:
        import torch
        dispositivo = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu")
    logger.info("Embeddings: %s em %s", MODELO_EMBEDDING, dispositivo)
    return HuggingFaceEmbeddings(
        model_name=MODELO_EMBEDDING,
        model_kwargs={"device": dispositivo},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 64},
    )


def construir(documentos, embeddings=None, recriar: bool = True) -> Chroma:
    """Vetoriza os documentos e grava a coleção no ChromaDB."""
    embeddings = embeddings or carregar_embeddings()
    if recriar and DIR_CHROMA.exists():
        shutil.rmtree(DIR_CHROMA)

    banco = Chroma(
        collection_name=COLECAO_CHROMA,
        embedding_function=embeddings,
        persist_directory=str(DIR_CHROMA),
        collection_metadata={"hnsw:space": "cosine"},
    )
    for inicio in range(0, len(documentos), LOTE):
        lote = documentos[inicio:inicio + LOTE]
        banco.add_documents(lote, ids=[d.id for d in lote])
        logger.info("  indexados %d/%d", min(inicio + LOTE, len(documentos)), len(documentos))
    return banco


def carregar(embeddings=None) -> Chroma:
    """Abre a coleção já persistida, sem revetorizar nada."""
    return Chroma(
        collection_name=COLECAO_CHROMA,
        embedding_function=embeddings or carregar_embeddings(),
        persist_directory=str(DIR_CHROMA),
    )


if __name__ == "__main__":
    import time
    from src.knowledge_base import construir as construir_base

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    t0 = time.time()
    banco = construir(construir_base())
    logger.info("Coleção pronta: %d vetores em %.0f s", banco._collection.count(), time.time() - t0)
