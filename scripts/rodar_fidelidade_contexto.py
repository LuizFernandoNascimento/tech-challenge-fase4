"""Fidelidade (RAGAS) medida contra o contexto que o gerador de fato recebeu.

Complementa scripts/rodar_ragas.py: lá o juiz recebeu só o texto das avaliações;
aqui recebe texto + metadados, exatamente como em generation.formatar_contexto.
Acrescenta a coluna `faithfulness_contexto_gerador` em artifacts/ragas_resultados.csv.
"""
import json
import pandas as pd
from langchain_groq import ChatGroq
from ragas.run_config import RunConfig

from src.chaves import carregar_chave
from src.llm import ativar_cache
from src.knowledge_base import para_documentos
from src.evaluation import contextos_do_gerador, avaliar_fidelidade
from src.config import DIR_ARTEFATOS


def main():
    print("chaves:", carregar_chave())
    ativar_cache()
    resultados = [r for r in json.load(open(f"{DIR_ARTEFATOS}/bateria_resultados.json", encoding="utf-8"))
                  if r["status"] == "respondida" and r["contextos"]]
    docs = para_documentos(pd.read_parquet(f"{DIR_ARTEFATOS}/base_conhecimento.parquet"))
    resultados = contextos_do_gerador(resultados, docs)
    juiz = ChatGroq(model="openai/gpt-oss-120b", temperature=0, max_retries=6,
                    timeout=120, reasoning_effort="low")
    notas = avaliar_fidelidade(resultados, juiz, RunConfig(max_workers=1, max_retries=10, max_wait=60, timeout=300))

    arq = f"{DIR_ARTEFATOS}/ragas_resultados.csv"
    df = pd.read_csv(arq)
    assert list(df["user_input"]) == [r["pergunta"] for r in resultados], "ordem diferente"
    df["faithfulness_contexto_gerador"] = notas
    df.to_csv(arq, index=False)
    cols = ["faithfulness", "faithfulness_contexto_gerador"]
    print(df[["rotulo", "rota"] + cols].round(2).to_string())
    print("\nmédia por rota:\n", df.groupby("rota")[cols].mean().round(3).to_string())
    print("\nmédia geral:\n", df[cols].mean().round(3).to_string())


if __name__ == "__main__":
    main()
