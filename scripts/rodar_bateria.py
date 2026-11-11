import json, logging, time, pandas as pd
logging.basicConfig(level=logging.WARNING)
from src.chaves import carregar_chave
from scripts.bateria_testes import BATERIA
print("chaves:", carregar_chave())
from src.llm import ativar_cache
ativar_cache()
from src.knowledge_base import para_documentos
from src.vectorstore import carregar
from src.retrieval import RecuperadorRAG
from src.graph import construir_grafo
from src.interface import criar_perguntar, formatar

docs = para_documentos(pd.read_parquet("artifacts/base_conhecimento.parquet"))
rec = RecuperadorRAG(carregar(), docs)
perguntar = criar_perguntar(construir_grafo(rec))

saida_txt, saida_json = [], []
for grupo, rotulo, pergunta, filtro, esperado in BATERIA:
    t0 = time.time()
    try:
        r = perguntar(pergunta, filtro=filtro, mostrar=False)
        erro = None
    except Exception as e:
        r, erro = {"pergunta": pergunta, "status": "ERRO", "resposta": str(e)[:300], "trilha": []}, e
    dt = time.time() - t0
    bloco = f"\n\n{'#'*92}\n# [{grupo}] {rotulo}  ({dt:.1f}s)\n{'#'*92}" + formatar(r)
    saida_txt.append(bloco)
    st = r.get("status", "?")
    acertou = {"responder": st == "respondida",
               "abster": st in ("sem_evidencia", "fora_do_escopo"),
               "bloquear": st == "bloqueada"}[esperado]
    print(f"{'OK ' if acertou else 'ERRO'} [{grupo:9}] {st:15} rota={str(r.get('rota')):10} {dt:5.1f}s  {pergunta[:52]}", flush=True)
    saida_json.append({
        "grupo": grupo, "rotulo": rotulo, "pergunta": pergunta, "filtro": filtro,
        "esperado": esperado, "acertou": acertou,
        "status": r.get("status"), "resposta": r.get("resposta"), "limitacoes": r.get("limitacoes"),
        "ids_evidencias": r.get("ids_evidencias", []), "ids_invalidos": r.get("ids_invalidos", []),
        "escore_escopo": r.get("escore_escopo"), "trilha": r.get("trilha", []), "segundos": round(dt, 1),
        "contextos": [d.page_content for d in r.get("relevantes", [])],
        "rota": r.get("rota"),
        "validacao_frequencia": ({k: r["frequencia"][k] for k in ("temas_invalidos", "ids_invalidos", "numeros_no_texto")}
                                 if r.get("frequencia") else None),
    })
    time.sleep(1)

open("artifacts/bateria_saida.txt", "w").write("".join(saida_txt))
json.dump(saida_json, open("artifacts/bateria_resultados.json", "w"), ensure_ascii=False, indent=1)
print("\nsalvo em artifacts/bateria_saida.txt e artifacts/bateria_resultados.json")
