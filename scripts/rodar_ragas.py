"""Avaliação RAGAS das respostas da bateria (artifacts/bateria_resultados.json).

Juiz: gpt-oss-120b no Groq (outro provedor que não o gerador). Executado em
série (max_workers=1) para respeitar o limite de 8.000 tokens/min do Groq.
Uso: python -m scripts.rodar_ragas [n]   (n = só as n primeiras, para teste)
"""
import json, sys, time
import pandas as pd
from langchain_groq import ChatGroq
from ragas.run_config import RunConfig

from src.chaves import carregar_chave
from src.llm import ativar_cache
from src.vectorstore import carregar_embeddings
from src.evaluation import avaliar_ragas
from src.config import DIR_ARTEFATOS


def main():
    print("chaves:", carregar_chave())
    ativar_cache()  # reexecuções reaproveitam os julgamentos já feitos
    resultados = [r for r in json.load(open(f"{DIR_ARTEFATOS}/bateria_resultados.json", encoding="utf-8"))
                  if r["status"] == "respondida"]
    if len(sys.argv) > 1:
        resultados = resultados[: int(sys.argv[1])]
    # Instanciado aqui, e não pela cadeia de fallback: o RAGAS precisa de um
    # único objeto LLM, e na avaliação queremos esperar o limite por minuto
    # (mais tentativas) em vez de trocar de juiz no meio da medição.
    juiz = ChatGroq(model="openai/gpt-oss-120b", temperature=0, max_retries=6,
                    timeout=120, reasoning_effort="low")
    cfg = RunConfig(max_workers=1, max_retries=10, max_wait=60, timeout=300)
    t = time.time()
    df = avaliar_ragas(resultados, juiz, carregar_embeddings(), run_config=cfg)
    print(f"{len(df)} respostas avaliadas em {time.time()-t:.0f}s")
    cols = ["faithfulness", "answer_relevancy", "llm_context_precision_without_reference"]
    print(df[["rotulo", "rota"] + cols].round(2).to_string())
    print("\nmédia por rota:\n", df.groupby("rota")[cols].mean().round(3).to_string())
    print("\nmédia geral:\n", df[cols].mean().round(3).to_string())
    print("\nfalhas (NaN):", df[cols].isna().sum().to_dict())
    if len(sys.argv) == 1:
        df.to_csv(f"{DIR_ARTEFATOS}/ragas_resultados.csv", index=False)


if __name__ == "__main__":
    main()
