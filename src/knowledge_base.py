"""Construção da base de conhecimento - Requisito 1 do enunciado (pág. 6).

O enunciado pede que a equipe "defina e justifique" seis estratégias. Cada uma
está numa função, com a justificativa no próprio docstring:

    Seleção dos dados ............ selecionar_avaliacoes()
    Limpeza e preparação ......... selecionar_avaliacoes() + preprocessing.py
    Valores ausentes ............. selecionar_avaliacoes() + enriquecer_com_metadados()
    Organização dos documentos ... deduplicar() + para_documentos()
    Embeddings ................... vectorstore.py
    Armazenamento e indexação .... vectorstore.py
"""
from __future__ import annotations

import logging
import re
import unicodedata

import pandas as pd
from langchain_core.documents import Document

from src.config import DIR_ARTEFATOS, DIR_DADOS, MIN_CARACTERES
from src.preprocessing import limpar_leve

logger = logging.getLogger(__name__)


def _chave_deduplicacao(texto: str) -> str:
    """Forma canônica do texto: sem acento, sem pontuação, minúsculo."""
    t = unicodedata.normalize("NFKD", texto.lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9 ]", "", t).strip()


def selecionar_avaliacoes(dir_dados=DIR_DADOS) -> pd.DataFrame:
    """SELEÇÃO, LIMPEZA e AUSENTES no texto.

    Seleção: só entram avaliações com comentário escrito. A nota sozinha não
    explica nada - o enunciado pede insights "presentes nos conteúdos textuais".
    Das 99.224 avaliações, cerca de 57% não têm texto algum.

    O documento é a MENSAGEM do cliente; o título fica como metadado. A mensagem
    concentra a justificativa ("veio quebrado e o vendedor não respondeu"); o
    título costuma ser só um rótulo ("Ruim", "Recomendo").

    Ausentes: avaliação sem mensagem mas com título usa o título - é o único
    texto que aquele cliente deixou. Sem nenhum dos dois, sai da base.

    Tamanho mínimo: abaixo de MIN_CARACTERES ("Ok", "Bom", ":)") não há
    informação recuperável que sustente uma resposta.
    """
    df = pd.read_csv(dir_dados / "olist_order_reviews_dataset.csv")
    total = len(df)

    df["titulo"] = df["review_comment_title"].fillna("").map(limpar_leve)
    df["mensagem"] = df["review_comment_message"].fillna("").map(limpar_leve)
    df["texto"] = df["mensagem"].where(df["mensagem"] != "", df["titulo"])

    com_texto = df[df["texto"] != ""]
    selecionadas = com_texto[com_texto["texto"].str.len() >= MIN_CARACTERES].copy()

    logger.info("Avaliações no arquivo     : %6d", total)
    logger.info("Com algum texto           : %6d (%.1f%%)", len(com_texto), 100 * len(com_texto) / total)
    logger.info("Com >= %d caracteres       : %6d (%.1f%%)", MIN_CARACTERES, len(selecionadas), 100 * len(selecionadas) / total)
    return selecionadas


def deduplicar(df: pd.DataFrame) -> pd.DataFrame:
    """ORGANIZAÇÃO: colapsa comentários repetidos, guardando a contagem.

    Milhares de clientes escrevem a mesma frase curta. Sem deduplicar, essas
    frases ocupariam todo o top-k da recuperação e o modelo veria a mesma
    evidência dez vezes. A contagem não se perde: "relatado por 47 clientes" é
    uma evidência mais forte que a frase isolada, e entra no contexto do LLM.
    """
    df = df.copy()
    df["_chave"] = df["texto"].map(_chave_deduplicacao)
    antes = len(df)
    agregado = (
        df.sort_values("review_creation_date")
        .groupby("_chave", as_index=False)
        .agg(
            texto=("texto", "first"),
            titulo=("titulo", "first"),
            n_ocorrencias=("review_id", "size"),
            review_ids=("review_id", lambda s: list(s)[:20]),
            order_id=("order_id", "first"),
            nota_media=("review_score", "mean"),
            primeira_data=("review_creation_date", "min"),
        )
    )
    logger.info("Após deduplicação         : %6d documentos (-%d)", len(agregado), antes - len(agregado))
    return agregado.drop(columns=["_chave"])


def enriquecer_com_metadados(docs: pd.DataFrame, dir_dados=DIR_DADOS) -> pd.DataFrame:
    """Metadados do pedido - "Desafio Adicional" do enunciado (pág. 8-9, opcional).

    Os metadados viram filtros de recuperação (ChromaDB filtra antes de buscar)
    e contexto para o modelo. Também permitem validar a recuperação contra
    rótulos que não estão no texto (ver evaluation.py).

    AUSENTES: pedido nunca entregue não tem data de entrega, logo não tem
    atraso calculável. O campo fica vazio - não vira "entregue no prazo".
    """
    pedidos = pd.read_csv(
        dir_dados / "olist_orders_dataset.csv",
        parse_dates=["order_purchase_timestamp", "order_delivered_customer_date",
                     "order_estimated_delivery_date"],
    )
    pedidos["atraso_dias"] = (
        pedidos["order_delivered_customer_date"] - pedidos["order_estimated_delivery_date"]
    ).dt.days
    pedidos["entrega_atrasada"] = (pedidos["atraso_dias"] > 0).astype("boolean")
    pedidos.loc[pedidos["atraso_dias"].isna(), "entrega_atrasada"] = pd.NA

    docs = docs.merge(pedidos[["order_id", "order_status", "atraso_dias", "entrega_atrasada"]],
                      on="order_id", how="left")

    itens = pd.read_csv(dir_dados / "olist_order_items_dataset.csv", usecols=["order_id", "product_id"])
    produtos = pd.read_csv(dir_dados / "olist_products_dataset.csv",
                           usecols=["product_id", "product_category_name"])
    itens = itens.merge(produtos, on="product_id", how="left")
    categoria = (itens.groupby("order_id")["product_category_name"]
                 .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else None)
                 .rename("categoria").reset_index())
    docs = docs.merge(categoria, on="order_id", how="left")

    docs["ano_mes"] = pd.to_datetime(docs["primeira_data"]).dt.strftime("%Y-%m")
    for campo in ("categoria", "atraso_dias", "entrega_atrasada"):
        logger.info("  cobertura de %-17s: %.1f%%", campo, 100 * docs[campo].notna().mean())
    return docs


def para_documentos(df: pd.DataFrame) -> list[Document]:
    """Converte cada linha num `Document` do LangChain.

    O ChromaDB só aceita metadados escalares (texto, número, booleano) e rejeita
    nulos. Campos ausentes são omitidos do dicionário - nunca preenchidos com
    um valor inventado. A lista de review_ids vira texto separado por vírgula.
    """
    documentos = []
    for _, r in df.iterrows():
        meta = {
            "doc_id": r["doc_id"],
            "n_ocorrencias": int(r["n_ocorrencias"]),
            "nota_media": round(float(r["nota_media"]), 2),
            "review_ids": ",".join(r["review_ids"][:5]),
            "ano_mes": r["ano_mes"],
        }
        if r["titulo"]:
            meta["titulo"] = r["titulo"]
        if pd.notna(r["categoria"]):
            meta["categoria"] = r["categoria"]
        if pd.notna(r["atraso_dias"]):
            meta["atraso_dias"] = int(r["atraso_dias"])
        if pd.notna(r["entrega_atrasada"]):
            meta["entrega_atrasada"] = bool(r["entrega_atrasada"])
        if pd.notna(r["order_status"]):
            meta["status_pedido"] = r["order_status"]
        documentos.append(Document(page_content=r["texto"], metadata=meta, id=r["doc_id"]))
    return documentos


def construir(dir_dados=DIR_DADOS, salvar: bool = True) -> list[Document]:
    """Executa o Requisito 1 completo e devolve os documentos prontos."""
    df = selecionar_avaliacoes(dir_dados)
    df = deduplicar(df)
    df = enriquecer_com_metadados(df, dir_dados)
    df = df.reset_index(drop=True)
    df.insert(0, "doc_id", [f"doc_{i:06d}" for i in range(len(df))])

    if salvar:
        DIR_ARTEFATOS.mkdir(parents=True, exist_ok=True)
        df.to_parquet(DIR_ARTEFATOS / "base_conhecimento.parquet", index=False)
    documentos = para_documentos(df)
    logger.info("Base final                : %6d documentos", len(documentos))
    return documentos


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    docs = construir()
    print("\nexemplo:", docs[100].page_content[:80], "|", docs[100].metadata)
