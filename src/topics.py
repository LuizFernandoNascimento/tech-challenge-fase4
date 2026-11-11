"""Camada de tópicos para perguntas de frequência (Documento de Aula 6).

Por que esta camada existe: o enunciado pergunta "quais problemas aparecem com
MAIOR FREQUÊNCIA nas avaliações?" (pág. 4) e "quais reclamações relacionadas à
entrega aparecem com maior frequência?" (pág. 5). O RAG recupera dez avaliações
semelhantes à pergunta; não tem como saber o que é frequente entre 33 mil.
Medimos o efeito: o modelo escreveu "reclamações recorrentes" com base em DUAS
avaliações.

A Documento de Aula 6 aponta exatamente esta integração:

    "Tendência forte: a integração de modelagem de tópicos com LLMs em pipelines
    de RAG. BERTopic é usado para indexar e organizar a base de conhecimento."

O BERTopic agrupa as 33 mil avaliações em temas (embeddings -> UMAP ->
HDBSCAN -> c-TF-IDF, como na aula) e cada avaliação recebe um tópico. As
contagens são REAIS e somam as duplicatas: uma frase "relatada por 47
clientes" conta 47.

Reaproveita os embeddings já gravados no ChromaDB - não vetoriza nada de novo.
"""
from __future__ import annotations

import logging
import pickle

import numpy as np
import pandas as pd
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from src.config import (DIR_ARTEFATOS, LIMIAR_OUTLIER, SEMENTE,
                        TOPICO_METODO, TOPICO_MIN_CLUSTER)

logger = logging.getLogger(__name__)
SEM_TOPICO = -1


def _capitalizar(rotulo: str) -> str:
    """Primeira letra maiúscula, sem alterar o resto (preserva siglas)."""
    return rotulo[:1].upper() + rotulo[1:]


class RotuloTopico(BaseModel):
    id: int
    rotulo: str = Field(description="Rótulo curto em português, 2 a 6 palavras, no tom de um analista.")


class Rotulos(BaseModel):
    topicos: list[RotuloTopico]


PROMPT_ROTULOS = ChatPromptTemplate.from_template(
    "Você vai nomear temas descobertos automaticamente em avaliações de clientes de "
    "um e-commerce brasileiro. Para cada tema há as palavras mais características, "
    "exemplos reais e a PROPORÇÃO de avaliações negativas (nota 1 ou 2).\n\n"
    "REGRA DE POLARIDADE - siga à risca:\n"
    "- Tema MISTO (entre 20% e 80% negativas): o rótulo nomeia o ASSUNTO, sem "
    "julgamento de valor, porque o mesmo tema reúne elogios e reclamações. "
    "Ex.: 'Prazo de entrega', 'Relógios', 'Cor do produto', 'Presentes de Natal'.\n"
    "- Tema NEGATIVO (80% ou mais): o rótulo pode nomear o problema. "
    "Ex.: 'Produto não recebido', 'Cobrança indevida'.\n"
    "- Tema POSITIVO (20% ou menos): o rótulo pode nomear o elogio. "
    "Ex.: 'Entrega antes do prazo', 'Produto de boa qualidade'.\n\n"
    "Use 2 a 6 palavras, em português, específicas e fiéis aos exemplos.\n\n{temas}")


def _polaridade_topicos(metas, topicos) -> dict[int, float]:
    """% de avaliações negativas (nota <= 2) em cada tópico, somando duplicatas."""
    df = pd.DataFrame({"topico": topicos,
                       "neg": [m["n_ocorrencias"] * (m["nota_media"] <= 2) for m in metas],
                       "n": [m["n_ocorrencias"] for m in metas]})
    g = df.groupby("topico")[["neg", "n"]].sum()
    return (100 * g["neg"] / g["n"]).to_dict()


def _classe(pct_neg: float) -> str:
    return "NEGATIVO" if pct_neg >= 80 else ("POSITIVO" if pct_neg <= 20 else "MISTO")


def _exemplos_centrais(textos, emb, topicos, t: int, n: int = 3) -> list[str]:
    """As n avaliações mais próximas do centro do tópico - as mais típicas dele."""
    idx = np.where(np.asarray(topicos) == t)[0]
    centro = emb[idx].mean(axis=0)
    centro /= np.linalg.norm(centro)
    sims = (emb[idx] / np.linalg.norm(emb[idx], axis=1, keepdims=True)) @ centro
    return [textos[i] for i in idx[np.argsort(sims)[::-1][:n]]]


def _bloco(modelo, t, textos, emb, topicos, pol, extra: str = "") -> str:
    palavras = ", ".join(w for w, _ in modelo.get_topic(t)[:8])
    exemplos = " | ".join(d[:110] for d in _exemplos_centrais(textos, emb, topicos, t, n=4))
    return (f"[{t}] {_classe(pol[t])} ({pol[t]:.0f}% negativas){extra}\n"
            f"     palavras: {palavras}\n     exemplos: {exemplos}")


def _rotular(modelo, ids, textos, emb, topicos, pol, lote: int = 12) -> dict[int, str]:
    """Nomeia os tópicos com o LLM, em lotes para economizar cota.

    O rótulo considera a polaridade do tópico (ver PROMPT_ROTULOS). Medimos que
    29 dos 85 tópicos misturam elogios e reclamações: um rótulo com julgamento
    ("Relógio entregue conforme esperado") ficava absurdo quando o tópico
    aparecia numa lista de problemas - e lá estavam 153 reclamações sobre relógios.
    """
    from src.llm import juiz
    cadeia = PROMPT_ROTULOS | juiz(Rotulos)
    rotulos = {}
    for i in range(0, len(ids), lote):
        grupo = ids[i:i + lote]
        saida = cadeia.invoke({"temas": "\n".join(_bloco(modelo, t, textos, emb, topicos, pol) for t in grupo)})
        rotulos.update({r.id: r.rotulo for r in saida.topicos if r.id in grupo})
    # Tópicos que o modelo pulou ganham uma segunda chance, um a um.
    for t in [t for t in ids if t not in rotulos]:
        saida = cadeia.invoke({"temas": _bloco(modelo, t, textos, emb, topicos, pol)})
        rotulos.update({r.id: r.rotulo for r in saida.topicos if r.id == t})
    faltando = [t for t in ids if t not in rotulos]
    if faltando:
        logger.warning("Tópicos sem rótulo do LLM (ficam com palavras-chave): %s", faltando)
    return rotulos


PROMPT_DESEMPATE = ChatPromptTemplate.from_template(
    "Os temas abaixo receberam o MESMO rótulo, mas são grupos distintos de avaliações. "
    "Mantenha a regra de polaridade: tema MISTO nomeia o assunto sem julgamento. "
    "Dê a cada um um rótulo curto (2 a 7 palavras) que o DIFERENCIE dos demais, com "
    "base no que os exemplos mostram de específico (ex.: 'consta como entregue', "
    "'sem resposta do vendedor', 'parte do pedido'). Os rótulos devem ser todos "
    "diferentes entre si.\n\n{temas}")


def _desempatar(rotulos: dict[int, str], modelo, textos, emb, topicos, pol) -> dict[int, str]:
    """Segunda passada: tópicos que ficaram com rótulos iguais ganham nomes distintos.

    Na primeira passada os tópicos são nomeados em lotes, e o modelo não vê os
    outros lotes - então repete nomes. Rótulos repetidos confundem o leitor e
    escondem que vários tópicos tratam do mesmo assunto.
    """
    from src.llm import juiz
    cadeia = PROMPT_DESEMPATE | juiz(Rotulos)
    novos = dict(rotulos)
    for rodada in range(3):
        grupos = {}
        for t, r in novos.items():
            grupos.setdefault(r.strip().lower(), []).append(t)
        repetidos = [ids for ids in grupos.values() if len(ids) > 1]
        if not repetidos:
            break
        for ids in repetidos:
            blocos = [_bloco(modelo, t, textos, emb, topicos, pol, f" | rótulo atual: {novos[t]}") for t in ids]
            saida = cadeia.invoke({"temas": "\n".join(blocos)})
            novos.update({r.id: r.rotulo for r in saida.topicos if r.id in ids})
    # Último recurso, determinístico: o que ainda repetir ganha a palavra mais
    # característica do próprio tópico entre parênteses.
    vistos = {}
    for t in sorted(novos):
        chave = novos[t].strip().lower()
        if chave in vistos:
            novos[t] = f"{novos[t]} ({modelo.get_topic(t)[0][0]})"
        vistos[novos[t].strip().lower()] = t
    logger.info("Desempate concluído")
    return novos


def construir(banco, rotular: bool = True) -> pd.DataFrame:
    """Treina o BERTopic, reduz outliers, nomeia os tópicos e grava no Chroma."""
    from bertopic import BERTopic
    from hdbscan import HDBSCAN
    from sklearn.feature_extraction.text import CountVectorizer
    from umap import UMAP

    from src.preprocessing import NEGACOES, _recursos_nltk

    dados = banco._collection.get(include=["documents", "embeddings", "metadatas"])
    # REPRODUTIBILIDADE: o Chroma não garante a ordem dos documentos - ela mudou
    # depois que gravamos metadados. O UMAP é sensível à ordem da entrada: mesma
    # semente com outra ordem dá outros tópicos (medimos: 84 virou 85). Ordenar
    # por doc_id torna o resultado independente do armazenamento.
    ordem = np.argsort([m["doc_id"] for m in dados["metadatas"]])
    dados = {k: [dados[k][i] for i in ordem] for k in ("ids", "documents", "embeddings", "metadatas")}
    textos = dados["documents"]
    emb = np.array(dados["embeddings"], dtype="float32")
    doc_ids = [m["doc_id"] for m in dados["metadatas"]]
    logger.info("Tópicos: %d documentos, embeddings reaproveitados do Chroma", len(textos))

    stop, _ = _recursos_nltk()
    # Negações ficam no vocabulário: "não chegou" e "chegou" são temas opostos.
    # min_df=1 porque, no c-TF-IDF, cada tópico vira um único documento.
    vetorizador = CountVectorizer(stop_words=sorted(stop - NEGACOES), ngram_range=(1, 2), min_df=1)
    modelo = BERTopic(
        umap_model=UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine",
                        random_state=SEMENTE),
        hdbscan_model=HDBSCAN(min_cluster_size=TOPICO_MIN_CLUSTER, metric="euclidean",
                              cluster_selection_method=TOPICO_METODO, prediction_data=True,
                              core_dist_n_jobs=1),
        vectorizer_model=vetorizador, language="multilingual")
    topicos, _ = modelo.fit_transform(textos, emb)
    antes = float(np.mean(np.array(topicos) == SEM_TOPICO))

    novos = modelo.reduce_outliers(textos, topicos, strategy="embeddings",
                                   embeddings=emb, threshold=LIMIAR_OUTLIER)
    modelo.update_topics(textos, topics=novos, vectorizer_model=vetorizador)
    depois = float(np.mean(np.array(novos) == SEM_TOPICO))
    logger.info("Tópicos: %d | sem tópico %.1f%% -> %.1f%% (limiar %.2f)",
                len(set(novos)) - 1, 100 * antes, 100 * depois, LIMIAR_OUTLIER)

    ids = sorted(t for t in set(novos) if t != SEM_TOPICO)
    pol = _polaridade_topicos(dados["metadatas"], novos)
    rotulos = _rotular(modelo, ids, textos, emb, novos, pol) if rotular else {}
    if rotulos:
        rotulos = _desempatar(rotulos, modelo, textos, emb, novos, pol)
        restantes = len(rotulos) - len({r.strip().lower() for r in rotulos.values()})
        logger.info("Rótulos únicos: %d de %d (repetidos restantes: %d)",
                    len(rotulos) - restantes, len(rotulos), restantes)

    info = pd.DataFrame({
        "topico": ids,
        # Padroniza a inicial maiúscula: o LLM ora capitaliza, ora não.
        "rotulo": [_capitalizar(rotulos.get(t, "_".join(w for w, _ in modelo.get_topic(t)[:3]))) for t in ids],
        "palavras": [", ".join(w for w, _ in modelo.get_topic(t)[:8]) for t in ids],
        "pct_negativas": [round(pol[t], 1) for t in ids],
        "polaridade": [_classe(pol[t]) for t in ids],
    })
    mapa = pd.DataFrame({"doc_id": doc_ids, "topico": [int(t) for t in novos]})

    DIR_ARTEFATOS.mkdir(parents=True, exist_ok=True)
    info.to_parquet(DIR_ARTEFATOS / "topicos_info.parquet", index=False)
    base = pd.read_parquet(DIR_ARTEFATOS / "base_conhecimento.parquet")
    (DIR_ARTEFATOS / "impressao_base.txt").write_text(impressao_base(base) + "\n")
    mapa.to_parquet(DIR_ARTEFATOS / "topicos_docs.parquet", index=False)
    with open(DIR_ARTEFATOS / "bertopic.pkl", "wb") as f:
        pickle.dump(modelo, f)

    # O tópico vira metadado no Chroma: permite buscar exemplos DENTRO de um tópico.
    metas = [{**m, "topico": int(t)} for m, t in zip(dados["metadatas"], novos)]
    for i in range(0, len(metas), 2000):
        banco._collection.update(ids=dados["ids"][i:i + 2000], metadatas=metas[i:i + 2000])
    return info


def carregar_tabelas(dir_artefatos=DIR_ARTEFATOS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Base de conhecimento com o tópico de cada documento + informações dos tópicos."""
    base = pd.read_parquet(dir_artefatos / "base_conhecimento.parquet")
    mapa = pd.read_parquet(dir_artefatos / "topicos_docs.parquet")
    info = pd.read_parquet(dir_artefatos / "topicos_info.parquet")
    return base.merge(mapa, on="doc_id", how="left"), info


def tabela_frequencia(base: pd.DataFrame, info: pd.DataFrame, mascara=None) -> pd.DataFrame:
    """Frequência real de cada tópico num recorte da base.

    `avaliacoes` soma `n_ocorrencias`: cada duplicata colapsada na deduplicação
    volta a contar como a avaliação que ela é. O percentual é sobre o recorte
    INTEIRO, incluindo as avaliações sem tópico - elas não somem da conta.
    """
    recorte = base if mascara is None else base[mascara]
    total = int(recorte["n_ocorrencias"].sum())
    freq = (recorte.groupby("topico")["n_ocorrencias"].sum().rename("avaliacoes").reset_index()
            .merge(info, on="topico", how="left"))
    freq.loc[freq["topico"] == SEM_TOPICO, "rotulo"] = "(sem tópico atribuído)"
    freq["percentual"] = (100 * freq["avaliacoes"] / total).round(1)
    freq.attrs["total"] = total
    return freq.sort_values("avaliacoes", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    import time
    from src.chaves import carregar_chave
    from src.llm import ativar_cache
    from src.vectorstore import carregar

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    carregar_chave(); ativar_cache()
    t0 = time.time()
    info = construir(carregar())
    logger.info("Concluído em %.0f s. Exemplos de rótulos:", time.time() - t0)
    for _, r in info.head(12).iterrows():
        logger.info("  [%3d] %-38s | %s", r.topico, r.rotulo, r.palavras[:55])


def impressao_base(base: pd.DataFrame) -> str:
    """Impressão digital (SHA-256) dos pares doc_id/texto da base de conhecimento.

    Os tópicos versionados no repositório referem-se a doc_ids; a impressão
    garante que a base reconstruída (no Colab, por exemplo) é a MESMA base em
    que os tópicos foram treinados - senão o doc_000123 poderia ser outra avaliação.
    """
    import hashlib
    h = hashlib.sha256()
    for doc_id, texto in base.sort_values("doc_id")[["doc_id", "texto"]].itertuples(index=False):
        h.update(f"{doc_id}\t{texto}\n".encode("utf-8"))
    return h.hexdigest()


def aplicar(banco, dir_artefatos=DIR_ARTEFATOS) -> int:
    """Grava no Chroma o tópico de cada documento a partir das tabelas salvas.

    Alternativa a `construir()`: reaproveita os tópicos já treinados e
    rotulados, sem retreinar o BERTopic (o UMAP pode variar entre máquinas) nem
    gastar chamadas de LLM nos rótulos. Confere a impressão digital da base.
    """
    base = pd.read_parquet(dir_artefatos / "base_conhecimento.parquet")
    esperada = (dir_artefatos / "impressao_base.txt").read_text().strip()
    if impressao_base(base) != esperada:
        raise ValueError("A base reconstruída difere da base em que os tópicos foram treinados. "
                         "Rode topics.construir() para retreinar.")
    mapa = dict(pd.read_parquet(dir_artefatos / "topicos_docs.parquet").itertuples(index=False))
    dados = banco._collection.get(include=["metadatas"])
    metas = [{**m, "topico": int(mapa.get(m["doc_id"], SEM_TOPICO))} for m in dados["metadatas"]]
    for i in range(0, len(metas), 2000):
        banco._collection.update(ids=dados["ids"][i:i + 2000], metadatas=metas[i:i + 2000])
    return len(metas)
