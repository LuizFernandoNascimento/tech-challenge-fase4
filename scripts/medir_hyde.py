"""Mede o efeito do HyDE na recuperação (Precisão@10), com e sem transformação.

Mesmo caminho do grafo: busca híbrida com a consulta (pergunta, ou pergunta +
avaliações hipotéticas) e re-ranking contra a pergunta ORIGINAL. Gabaritos:
metadados (CASOS_METADADO) e termos temáticos (TEMAS) de src/evaluation.py.
"""
import json
import pandas as pd
from langchain_core.output_parsers import StrOutputParser

from src.chaves import carregar_chave
from src.llm import ativar_cache, auxiliar
from src.knowledge_base import para_documentos
from src.vectorstore import carregar
from src.retrieval import RecuperadorRAG
from src.generation import PROMPT_HYDE
from src.evaluation import CASOS_METADADO, TEMAS, _norm
from src.config import DIR_ARTEFATOS


def main():
    print("chaves:", carregar_chave())
    ativar_cache()
    docs = para_documentos(pd.read_parquet(f"{DIR_ARTEFATOS}/base_conhecimento.parquet"))
    rec = RecuperadorRAG(carregar(), docs)
    hyde = PROMPT_HYDE | auxiliar() | StrOutputParser()

    casos = [(p, rel, crit) for p, rel, crit in CASOS_METADADO]
    casos += [(f"avaliações sobre {tema}",
               (lambda termos: lambda d: any(t in _norm(d.page_content) for t in termos))(termos),
               f"tema: {tema}") for tema, termos in TEMAS.items()]

    linhas = []
    for pergunta, relevante, criterio in casos:
        hipoteses = hyde.invoke({"pergunta": pergunta})
        for modo, consulta in [("sem HyDE", pergunta), ("com HyDE", f"{pergunta}\n{hipoteses}")]:
            cand = rec.ensemble(None).invoke(consulta)
            top = rec.reranker.compress_documents(cand, pergunta)[:10]
            # Casos de metadado recebem metadata; temáticos recebem o Document.
            ok = sum(relevante(d) if criterio.startswith("tema") else relevante(d.metadata) for d in top)
            linhas.append({"criterio": criterio, "modo": modo, "P@10": ok / max(len(top), 1)})
        print(f"{criterio:38} sem={linhas[-2]['P@10']:.2f} com={linhas[-1]['P@10']:.2f}", flush=True)

    df = pd.DataFrame(linhas).pivot(index="criterio", columns="modo", values="P@10")
    print("\n", df.round(2).to_string(), "\n\nmédia:\n", df.mean().round(3).to_string())
    df.to_csv(f"{DIR_ARTEFATOS}/hyde_comparacao.csv")


if __name__ == "__main__":
    main()
