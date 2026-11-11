"""Recuperação híbrida com re-ranking - Requisito 2 (pág. 6-7).

A arquitetura segue a 1IAST Aula 2 quase literalmente:

    "a estratégia híbrida tornou-se dominante em aplicações empresariais,
    combinando busca vetorial (semântica) com a busca por palavra-chave
    (lexical). Após a recuperação inicial, os resultados podem ser reordenados
    por um modelo secundário para garantir que os fragmentos mais úteis estejam
    no topo. Tipicamente o método Reciprocal Rank Fusion (RRF) (...)"

Três estágios, todos componentes nativos do LangChain:

    1. BM25Retriever  ┐
                      ├─> EnsembleRetriever (RRF, c=60) ─> 2. fusão
       Chroma         ┘
    3. ContextualCompressionRetriever com Cross-Encoder ─> re-ranking

Por que híbrida: a busca densa entende paráfrase ("demorou muito" encontra
"entrega lenta") mas dilui termo literal; o BM25 acerta o termo exato
("boleto", nome de loja, nome de produto) mas não entende sinônimo. A Documento
de Aula 3 cita o mesmo desenho no buscador do Mercado Livre.

Por que re-ranking: o bi-encoder compara vetores calculados SEPARADAMENTE para
pergunta e documento. O Cross-Encoder lê os dois JUNTOS e produz uma nota de
relevância mais precisa (POSTECH Aula 3). É caro demais para rodar nos 33 mil
documentos, então roda só sobre os candidatos já filtrados pela fusão.

A nota do Cross-Encoder é usada SÓ para ordenar. Medimos que ela não serve
como confiança (ver config.LIMIAR_ESCOPO); para isso usamos `escore_escopo`.
"""
from __future__ import annotations

import logging
from typing import Any, Optional, Sequence

import numpy as np
from langchain.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.retrievers import BM25Retriever
from langchain_core.callbacks import Callbacks
from langchain_core.documents import BaseDocumentCompressor, Document

from src.config import (K_CANDIDATOS, K_ESCOPO, K_RRF, MODELO_RERANKER,
                        PESO_DENSO, PESO_LEXICO, TOP_N_RERANK)
from src.preprocessing import tokenizar_bm25

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Filtros por metadado
# --------------------------------------------------------------------------
_OPERADORES = {
    "$eq": lambda v, a: v == a, "$ne": lambda v, a: v != a,
    "$gt": lambda v, a: v > a, "$gte": lambda v, a: v >= a,
    "$lt": lambda v, a: v < a, "$lte": lambda v, a: v <= a,
    "$in": lambda v, a: v in a,
}


def _atende(meta: dict, filtro: dict) -> bool:
    """Avalia um filtro no formato `where` do ChromaDB sobre um dicionário.

    Assim os dois recuperadores aceitam o MESMO filtro: o Chroma aplica antes
    da busca, nativamente; o BM25 aplica aqui. Suporta $eq, $ne, $gt, $gte,
    $lt, $lte, $in e a combinação $and.
    """
    if "$and" in filtro:
        return all(_atende(meta, f) for f in filtro["$and"])
    for campo, cond in filtro.items():
        valor = meta.get(campo)
        if valor is None:
            return False                  # sem o metadado, não há como afirmar
        if not isinstance(cond, dict):
            cond = {"$eq": cond}
        for op, alvo in cond.items():
            # Avalia SÓ o operador pedido. Montar um dicionário com todos de uma
            # vez avaliaria também `valor in alvo`, que quebra quando o alvo é
            # um número (ex.: {"nota_media": {"$lte": 2}}).
            if op not in _OPERADORES:
                raise ValueError(f"Operador de filtro não suportado: {op}")
            if not _OPERADORES[op](valor, alvo):
                return False
    return True


class BM25ComFiltro(BM25Retriever):
    """BM25Retriever que respeita o mesmo filtro de metadados do Chroma.

    O índice BM25 é montado uma vez sobre a base inteira; o filtro só
    restringe quais documentos podem aparecer no resultado.
    """
    filtro: Optional[dict] = None

    def _get_relevant_documents(self, query, *, run_manager=None):
        scores = self.vectorizer.get_scores(self.preprocess_func(query))
        ordem = np.argsort(scores)[::-1]
        saida = []
        for i in ordem:
            if scores[i] <= 0:
                break
            if self.filtro and not _atende(self.docs[i].metadata, self.filtro):
                continue
            saida.append(self.docs[i])
            if len(saida) >= self.k:
                break
        return saida


# --------------------------------------------------------------------------
# Re-ranking com nota exposta
# --------------------------------------------------------------------------
class RerankerCrossEncoder(BaseDocumentCompressor):
    """Cross-Encoder que reordena E registra a nota em `metadata['relevancia']`.

    O `CrossEncoderReranker` nativo do LangChain reordena, mas descarta a nota.
    Ela é exposta para que o usuário veja a ordem de relevância (Requisito 3).
    O modelo devolve logits; aplicamos a sigmoide só para leitura em [0, 1].
    """
    modelo: Any
    top_n: int = TOP_N_RERANK

    def compress_documents(self, documents: Sequence[Document], query: str,
                           callbacks: Callbacks = None) -> Sequence[Document]:
        if not documents:
            return []
        notas = self.modelo.score([(query, d.page_content) for d in documents])
        ordenados = sorted(zip(documents, notas), key=lambda x: x[1], reverse=True)
        saida = []
        for doc, nota in ordenados[: self.top_n]:
            doc.metadata["relevancia"] = round(float(1 / (1 + np.exp(-nota))), 4)
            saida.append(doc)
        return saida


# --------------------------------------------------------------------------
# Montagem
# --------------------------------------------------------------------------
class RecuperadorRAG:
    """Monta os três estágios e permite inspecionar cada um."""

    def __init__(self, banco, documentos: list[Document], reranker_modelo=None):
        self.banco = banco
        logger.info("Indexando BM25 sobre %d documentos...", len(documentos))
        self.bm25 = BM25ComFiltro.from_documents(
            documentos, preprocess_func=tokenizar_bm25, k=K_CANDIDATOS)
        logger.info("Carregando Cross-Encoder: %s", MODELO_RERANKER)
        self.reranker = RerankerCrossEncoder(
            modelo=reranker_modelo or HuggingFaceCrossEncoder(model_name=MODELO_RERANKER))

    def _denso(self, filtro):
        kwargs = {"k": K_CANDIDATOS}
        if filtro:
            kwargs["filter"] = filtro
        return self.banco.as_retriever(search_kwargs=kwargs)

    def _lexico(self, filtro):
        return self.bm25.model_copy(update={"filtro": filtro})

    def ensemble(self, filtro: dict | None = None, modo: str = "hibrido",
                 peso_denso: float = PESO_DENSO, peso_lexico: float = PESO_LEXICO):
        """Estágios 1 e 2. `modo` existe para que a comparação seja mensurável."""
        if modo == "denso":
            return self._denso(filtro)
        if modo == "lexico":
            return self._lexico(filtro)
        return EnsembleRetriever(
            retrievers=[self._lexico(filtro), self._denso(filtro)],
            weights=[peso_lexico, peso_denso], c=K_RRF, id_key="doc_id")

    def recuperador(self, filtro: dict | None = None, modo: str = "hibrido",
                    com_rerank: bool = True, **pesos):
        """Os três estágios encadeados, como um único retriever do LangChain."""
        base = self.ensemble(filtro, modo, **pesos)
        if not com_rerank:
            return base
        return ContextualCompressionRetriever(base_compressor=self.reranker,
                                              base_retriever=base)

    def escore_escopo(self, pergunta: str) -> float:
        """Camada 1 da abstenção: quão perto a pergunta está da base.

        Média do cosseno das K_ESCOPO avaliações mais próximas. O Chroma
        devolve DISTÂNCIA de cosseno (1 - similaridade), daí a conversão.
        """
        pares = self.banco.similarity_search_with_score(pergunta, k=K_ESCOPO)
        return float(np.mean([1 - dist for _, dist in pares])) if pares else 0.0

    def recuperar(self, pergunta: str, filtro: dict | None = None, **kw) -> list[Document]:
        return self.recuperador(filtro, **kw).invoke(pergunta)

    def inspecionar(self, pergunta: str, filtro: dict | None = None) -> dict:
        """Executa estágio por estágio e devolve o que cada um produziu.

        É o que o enunciado pede para "demonstrar como ocorre" a recuperação.
        """
        lex = self._lexico(filtro).invoke(pergunta)
        den = self._denso(filtro).invoke(pergunta)
        fundidos = self.ensemble(filtro).invoke(pergunta)
        rerank = self.reranker.compress_documents(fundidos, pergunta)
        return {
            "tokens_bm25": tokenizar_bm25(pergunta),
            "lexico": lex, "denso": den,
            "fundidos": fundidos, "selecionados": rerank,
        }
