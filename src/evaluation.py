"""Avaliação do sistema RAG.

A 1IAST Aula 2 divide a avaliação em duas dimensões, e este módulo segue a
mesma divisão:

RECUPERAÇÃO - "um gerador excelente não consegue produzir uma resposta
confiável quando a evidência correta simplesmente não foi recuperada"
    * avaliar_por_metadado(): precisão@k contra rótulos que NÃO estão no texto
    * avaliar_temas(): precisão@k com relevância por léxico declarado
    * varrer_pesos(): justifica os pesos da fusão com número, não com palpite

GERAÇÃO - "mede se a resposta gerada é baseada exclusivamente no contexto"
    * avaliar_ragas(): Faithfulness, Answer Relevancy, Context Precision
      (RAGAS - ensinado na 1IAST Aula 2 e na POSTECH Aula 3)

Sobre o gabarito por metadado: é uma validação complementar ao RAGAS, não um
substituto. `entrega_atrasada` vem das DATAS do pedido; `categoria`, do
catálogo de produtos. Nenhum dos dois está no texto indexado - então, quando o
sistema pergunta "atraso na entrega" e devolve pedidos que de fato atrasaram
segundo as datas, não há como ser coincidência léxica.
"""
from __future__ import annotations

import logging
import unicodedata

import pandas as pd

logger = logging.getLogger(__name__)

CASOS_METADADO = [
    ("Quais avaliações relatam atraso na entrega?",
     lambda m: m.get("entrega_atrasada") is True, "pedido realmente atrasou (datas)"),
    ("Quais clientes ficaram muito insatisfeitos com a compra?",
     lambda m: m.get("nota_media", 3) <= 2, "nota 1 ou 2"),
    ("Quais clientes ficaram muito satisfeitos com a compra?",
     lambda m: m.get("nota_media", 3) >= 4, "nota 4 ou 5"),
    ("Avaliações sobre roupa de cama, mesa e banho",
     lambda m: m.get("categoria") == "cama_mesa_banho", "categoria cama_mesa_banho"),
    ("Avaliações sobre produtos de beleza e saúde",
     lambda m: m.get("categoria") == "beleza_saude", "categoria beleza_saude"),
]

CONFIGURACOES = [
    ("lexico (BM25)", {"modo": "lexico", "com_rerank": False}),
    ("denso (Chroma)", {"modo": "denso", "com_rerank": False}),
    ("hibrido (RRF)", {"modo": "hibrido", "com_rerank": False}),
    ("hibrido + rerank", {"modo": "hibrido", "com_rerank": True}),
]


def avaliar_por_metadado(rec, documentos, k: int = 10) -> pd.DataFrame:
    """Precisão@k e lift sobre a taxa-base, por configuração de recuperação."""
    metas = [d.metadata for d in documentos]
    linhas = []
    for pergunta, relevante, criterio in CASOS_METADADO:
        base = sum(relevante(m) for m in metas) / len(metas)
        for nome, kw in CONFIGURACOES:
            docs = rec.recuperar(pergunta, **kw)[:k]
            prec = sum(relevante(d.metadata) for d in docs) / max(len(docs), 1)
            linhas.append({"criterio": criterio, "config": nome, f"precisao@{k}": round(prec, 3),
                           "taxa_base": round(base, 3), "lift": round(prec / base, 1) if base else None})
            logger.info("  %-34s %-18s P@%d=%.2f (base %.2f)", criterio, nome, k, prec, base)
    return pd.DataFrame(linhas)


TEMAS = {
    "problemas de entrega": ["atras", "demor", "prazo", "nao cheg", "nao receb",
                             "entreg", "correio", "extravi"],
    "qualidade do produto": ["qualidade", "quebrad", "defeito", "danificad",
                             "estrag", "fragil", "rasg", "errad"],
    "elogios ao atendimento": ["atendimento", "atencios", "suporte", "respond",
                               "educad", "resolveu"],
}


def _norm(t: str) -> str:
    t = unicodedata.normalize("NFKD", t.lower())
    return "".join(c for c in t if not unicodedata.combining(c))


def avaliar_temas(rec, k: int = 10) -> pd.DataFrame:
    """Precisão@k temática. Viés declarado: o critério é léxico e favorece o BM25."""
    linhas = []
    for tema, termos in TEMAS.items():
        for nome, kw in CONFIGURACOES:
            docs = rec.recuperar(f"avaliações sobre {tema}", **kw)[:k]
            ok = sum(any(t in _norm(d.page_content) for t in termos) for d in docs)
            linhas.append({"tema": tema, "config": nome, f"precisao@{k}": round(ok / max(len(docs), 1), 3)})
    return pd.DataFrame(linhas)


def varrer_pesos(rec, k: int = 10, pesos=(1.0, 0.9, 0.8, 0.7, 0.5, 0.3, 0.0)) -> pd.DataFrame:
    """Precisão média nos 3 casos não-categoria, variando o peso da busca densa."""
    casos = CASOS_METADADO[:3]
    linhas = []
    for pd_ in pesos:
        precs = []
        for pergunta, relevante, _ in casos:
            docs = rec.recuperar(pergunta, com_rerank=False, peso_denso=pd_,
                                 peso_lexico=round(1 - pd_, 2))[:k]
            precs.append(sum(relevante(d.metadata) for d in docs) / max(len(docs), 1))
        linhas.append({"peso_denso": pd_, "peso_lexico": round(1 - pd_, 2),
                       **{f"caso_{i+1}": round(p, 2) for i, p in enumerate(precs)},
                       "media": round(sum(precs) / len(precs), 3)})
    return pd.DataFrame(linhas)


def _texto_do_llm(r: dict) -> str:
    """Parte da resposta que o LLM escreveu.

    Na rota de frequência, a tabela de temas e contagens é calculada pelo
    código (e conferida por validação própria); o LLM só escreve a síntese que
    vem depois de "Juntos: ...". Medir fidelidade da tabela contra as
    avaliações seria pedir que exemplos individuais "provassem" uma contagem.
    """
    if r.get("rota") == "frequencia" and "\n\n" in r["resposta"]:
        return r["resposta"].split("\n\n", 1)[1]
    return r["resposta"]


def avaliar_ragas(resultados: list[dict], llm, embeddings, max_contextos: int = 10,
                  run_config=None) -> pd.DataFrame:
    """Métricas RAGAS sobre respostas já geradas pelo pipeline (LLM-as-a-Judge).

    O juiz é de OUTRO provedor (Groq), não o Gemini que gerou as respostas:
    um modelo avaliando a si mesmo tende a ser complacente (1IAST Aula 2/3).
    `resultados`: dicts com pergunta, resposta, contextos e rota.
    """
    from datasets import Dataset
    from ragas import evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (Faithfulness, ResponseRelevancy,
                               LLMContextPrecisionWithoutReference)

    validos = [r for r in resultados if r.get("contextos")]

    def _dataset(limite):
        return Dataset.from_list([
            {"user_input": r["pergunta"], "response": _texto_do_llm(r),
             "retrieved_contexts": r["contextos"][:limite]}
            for r in validos
        ])

    kw = dict(llm=LangchainLLMWrapper(llm), embeddings=LangchainEmbeddingsWrapper(embeddings),
              run_config=run_config, show_progress=False)
    # Fidelidade com TODOS os contextos: a síntese de frequência cita até 60
    # exemplos; cortar a lista faria afirmações corretas parecerem sem suporte.
    rel = evaluate(_dataset(None), metrics=[Faithfulness(), ResponseRelevancy()], **kw)
    # A precisão de contexto custa uma chamada ao juiz POR contexto; avaliamos
    # os `max_contextos` primeiros (os de melhor posição). Não há resposta-
    # referência humana: por isso a variante SEM referência, e não uma
    # referência inventada (que tornaria a métrica circular).
    prec = evaluate(_dataset(max_contextos), metrics=[LLMContextPrecisionWithoutReference()], **kw)
    df = rel.to_pandas()
    df["llm_context_precision_without_reference"] = prec.to_pandas()["llm_context_precision_without_reference"].values
    df.insert(0, "rota", [r.get("rota") for r in validos])
    df.insert(0, "rotulo", [r.get("rotulo") for r in validos])
    return df
