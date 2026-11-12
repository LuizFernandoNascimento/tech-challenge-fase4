"""Orquestração do pipeline com LangGraph - Requisitos 2, 3 e 4.

A POSTECH Aula 5 dá, como exemplo de aresta condicional, o fluxo de que este
projeto precisa:

    "Se o agente de busca não encontrar o resultado, retorne para o nó de
    query-rewrite; se encontrar, siga para o nó de sumarização."

O grafo implementa isso, com três camadas de abstenção (Requisito 4) e uma
rota de frequência (Router Chain, POSTECH Aula 2):

  verificar_entrada ──(injeção)──────────────────────────────────> recusar
        │
  verificar_escopo ───(fora do escopo, CAMADA 1, sem LLM)────────> abster
        │
  rotear ──(pergunta de frequência)──> analisar_frequencia ──────> fim
        │                              (tópicos: o LLM escolhe, o código conta)
        │
  transformar (HyDE) ─> recuperar ─> julgar (CAMADA 2) ─┬─> gerar (CAMADA 3)
                            ^                           ├─> reescrever ─┐
                            └───────────────────────────┼───────────────┘
                                                        └─> abster

O ciclo recuperar -> julgar -> reescrever -> recuperar é o padrão de
Self-Reflection da aula, e só é possível porque o LangGraph permite ciclos.

Cada nó registra sua passagem em `trilha`. Isso é rastreabilidade do CAMINHO,
não só da resposta: dá para ver por que o sistema respondeu ou recusou.
"""
from __future__ import annotations

import logging
from typing import Optional, TypedDict

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import END, StateGraph

from src.config import LIMIAR_ESCOPO, MAX_REESCRITAS, MIN_EVIDENCIAS
from src.generation import (PROMPT_HYDE, PROMPT_STEPBACK, Julgamento,
                            RespostaFundamentada, formatar_contexto,
                            montar_cadeia, montar_julgador)
from src.llm import RastreadorModelos
from src.frequency import (PROMPT_FREQUENCIA, SelecaoTopicos, analisar as analisar_freq,
                           e_pergunta_de_frequencia)
from src.guardrails import detectar_injecao, remover_citacoes, validar_citacoes

logger = logging.getLogger(__name__)


def _br(valor: float, casas: int = 0) -> str:
    """Número no padrão brasileiro: 12.345 e 17,3."""
    texto = f"{valor:,.{casas}f}"
    return texto.replace(",", "§").replace(".", ",").replace("§", ".")


class Estado(TypedDict, total=False):
    """Estado compartilhado entre os nós (POSTECH Aula 5: 'fonte da verdade')."""
    pergunta: str
    filtro: Optional[dict]
    usar_hyde: bool
    escore_escopo: float
    consulta_busca: str
    documentos: list[Document]
    relevantes: list[Document]
    justificativa_julgamento: str
    tentativas: int
    status: str            # respondida | sem_evidencia | fora_do_escopo | bloqueada
    rota: str              # rag | frequencia
    frequencia: dict
    resposta: str
    ids_evidencias: list[str]
    ids_invalidos: list[str]
    limitacoes: str
    trilha: list[str]


def construir_grafo(recuperador, gerador=None, julgador_llm=None, auxiliar=None,
                    tabelas_topicos=None):
    """Monta o grafo. Os modelos vêm das cadeias de fallback de llm.py, salvo
    se forem passados explicitamente (útil para testes)."""
    from src import llm as fabrica
    from src.topics import carregar_tabelas
    base_topicos, info_topicos = tabelas_topicos or carregar_tabelas()
    cadeia_frequencia = PROMPT_FREQUENCIA | fabrica.principal(SelecaoTopicos)
    cadeia_resposta = montar_cadeia(gerador or fabrica.principal(RespostaFundamentada))
    julgador = montar_julgador(julgador_llm or fabrica.principal(Julgamento))
    aux = auxiliar or fabrica.auxiliar()
    cadeia_hyde = PROMPT_HYDE | aux | StrOutputParser()
    cadeia_stepback = PROMPT_STEPBACK | aux | StrOutputParser()

    def _chamar(cadeia, entrada):
        """Invoca a cadeia registrando quais modelos tentaram e qual respondeu."""
        r = RastreadorModelos()
        saida = cadeia.invoke(entrada, config={"callbacks": [r]})
        return saida, r.resumo()

    def verificar_entrada(e: Estado) -> Estado:
        trecho = detectar_injecao(e["pergunta"])
        if trecho:
            return {"status": "bloqueada", "trilha": [f"verificar_entrada:bloqueada('{trecho}')"]}
        return {"tentativas": 0, "trilha": ["verificar_entrada:ok"]}

    def verificar_escopo(e: Estado) -> Estado:
        escore = recuperador.escore_escopo(e["pergunta"])
        marca = "ok" if escore >= LIMIAR_ESCOPO else "fora"
        return {"escore_escopo": escore,
                "trilha": e["trilha"] + [f"verificar_escopo:{escore:.3f}({marca})"]}

    def rotear(e: Estado) -> Estado:
        rota = "frequencia" if e_pergunta_de_frequencia(e["pergunta"]) else "rag"
        return {"rota": rota, "trilha": e["trilha"] + [f"rotear:{rota}"]}

    def analisar_frequencia(e: Estado) -> Estado:
        r = RastreadorModelos()
        cadeia = cadeia_frequencia.with_config(callbacks=[r])
        f = analisar_freq(e["pergunta"], base_topicos, info_topicos, recuperador.banco,
                          recuperador.banco.embeddings, cadeia, filtro_usuario=e.get("filtro"))
        if not f["evidencia_suficiente"]:
            return {"frequencia": f, "status": "sem_evidencia", "ids_evidencias": [],
                    "resposta": "Nenhum dos temas encontrados nas avaliações responde a essa pergunta.",
                    "limitacoes": f"Foram avaliados os {f['n_temas_avaliados']} temas mais frequentes do recorte.",
                    "trilha": e["trilha"] + [f"analisar_frequencia:sem_tema[{r.resumo()}]"]}
        linhas = [f"Recorte analisado: {_br(f['total_recorte'])} {f['recorte']}.",
                  "Temas que respondem à pergunta (contagens exatas, somando avaliações repetidas):"]
        for i, t in enumerate(f["temas_selecionados"], start=1):
            linhas.append(f"  {i}. {t['rotulo']} — {_br(t['avaliacoes'])} avaliações "
                          f"({_br(t['percentual'], 1)}% do recorte)")
        pct = 100 * f["soma_selecionados"] / f["total_recorte"]
        linhas.append(f"  Juntos: {_br(f['soma_selecionados'])} avaliações ({_br(pct, 1)}% do recorte).")
        resposta = "\n".join(linhas) + "\n\n" + f["sintese"]
        limit = (f"{_br(f['pct_sem_topico'], 1)}% do recorte não foi atribuído a nenhum tema e "
                 f"não entra nesta conta. As contagens refletem quem escreveu comentário, não "
                 f"todos os clientes.")
        if f["numeros_no_texto"]:
            limit += f" Atenção: a síntese contém números não verificados: {f['numeros_no_texto']}."
        return {"frequencia": f, "status": "respondida", "resposta": resposta,
                "ids_evidencias": f["ids_evidencias"], "ids_invalidos": f["ids_invalidos"],
                "relevantes": [f["exemplos"][i] for i in f["ids_evidencias"]],
                "limitacoes": limit,
                "trilha": e["trilha"] + [f"analisar_frequencia:{len(f['temas_selecionados'])}_temas[{r.resumo()}]"]}

    def transformar(e: Estado) -> Estado:
        if not e.get("usar_hyde", True):
            return {"consulta_busca": e["pergunta"], "trilha": e["trilha"] + ["transformar:off"]}
        hipoteses, modelos = _chamar(cadeia_hyde, {"pergunta": e["pergunta"]})
        # A pergunta original fica na consulta: o BM25 precisa dos termos
        # literais; as avaliações hipotéticas acrescentam a linguagem do cliente.
        return {"consulta_busca": f"{e['pergunta']}\n{hipoteses}",
                "trilha": e["trilha"] + [f"transformar:hyde[{modelos}]"]}

    def recuperar(e: Estado) -> Estado:
        # Busca com a consulta transformada; re-ranking contra a pergunta
        # ORIGINAL, que é o que de fato precisa ser respondido.
        candidatos = recuperador.ensemble(e.get("filtro")).invoke(e["consulta_busca"])
        docs = recuperador.reranker.compress_documents(candidatos, e["pergunta"])
        return {"documentos": docs, "trilha": e["trilha"] + [f"recuperar:{len(docs)}"]}

    def julgar(e: Estado) -> Estado:
        if not e["documentos"]:
            return {"relevantes": [], "trilha": e["trilha"] + ["julgar:0"]}
        veredito, modelos = _chamar(julgador, {"contexto": formatar_contexto(e["documentos"]),
                                               "pergunta": e["pergunta"]})
        ids = set(veredito.ids_relevantes)
        relevantes = [d for d in e["documentos"] if d.metadata["doc_id"] in ids]
        return {"relevantes": relevantes, "justificativa_julgamento": veredito.justificativa,
                "trilha": e["trilha"] + [f"julgar:{len(relevantes)}/{len(e['documentos'])}[{modelos}]"]}

    def rota_apos_julgar(e: Estado) -> str:
        if len(e["relevantes"]) >= MIN_EVIDENCIAS:
            return "gerar"
        if e.get("tentativas", 0) < MAX_REESCRITAS:
            return "reescrever"
        return "abster"

    def reescrever(e: Estado) -> Estado:
        nova, modelos = _chamar(cadeia_stepback, {"pergunta": e["pergunta"]})
        nova = nova.strip()
        return {"consulta_busca": nova, "tentativas": e.get("tentativas", 0) + 1,
                "trilha": e["trilha"] + [f"reescrever[{modelos}]:'{nova[:60]}'"]}

    def gerar(e: Estado) -> Estado:
        saida, modelos = _chamar(cadeia_resposta, {"contexto": formatar_contexto(e["relevantes"]),
                                                   "pergunta": e["pergunta"]})
        recuperados = {d.metadata["doc_id"] for d in e["relevantes"]}
        validos, invalidos = validar_citacoes(saida.ids_evidencias, recuperados, saida.resposta)
        status = "respondida" if saida.evidencia_suficiente and validos else "sem_evidencia"
        return {"resposta": remover_citacoes(saida.resposta, invalidos), "ids_evidencias": validos,
                "ids_invalidos": invalidos, "limitacoes": saida.limitacoes,
                "status": status, "trilha": e["trilha"] + [f"gerar:{status}[{modelos}]"]}

    def abster(e: Estado) -> Estado:
        if e.get("escore_escopo", 1) < LIMIAR_ESCOPO:
            status = "fora_do_escopo"
            motivo = (f"A pergunta não se parece com nenhum tema presente nas avaliações "
                      f"(proximidade {e['escore_escopo']:.2f}, mínimo {LIMIAR_ESCOPO}).")
        else:
            status = "sem_evidencia"
            motivo = (f"Foram recuperadas avaliações, mas apenas {len(e.get('relevantes', []))} "
                      f"foram julgadas relevantes (mínimo {MIN_EVIDENCIAS}), mesmo após "
                      f"reformular a busca.")
        return {"status": status, "ids_evidencias": [], "limitacoes": motivo,
                "resposta": ("Não há avaliações suficientes na base para responder a essa "
                             "pergunta com segurança. A base contém apenas avaliações escritas "
                             "por clientes sobre suas compras."),
                "trilha": e["trilha"] + [f"abster:{status}"]}

    def recusar(e: Estado) -> Estado:
        return {"resposta": "A pergunta foi bloqueada pelo guardrail de entrada por conter "
                            "um padrão de instrução ao sistema (possível prompt injection).",
                "ids_evidencias": [], "limitacoes": "", "trilha": e["trilha"] + ["recusar"]}

    g = StateGraph(Estado)
    for nome, fn in [("verificar_entrada", verificar_entrada), ("verificar_escopo", verificar_escopo),
                     ("rotear", rotear), ("analisar_frequencia", analisar_frequencia),
                     ("transformar", transformar), ("recuperar", recuperar), ("julgar", julgar),
                     ("reescrever", reescrever), ("gerar", gerar), ("abster", abster),
                     ("recusar", recusar)]:
        g.add_node(nome, fn)

    g.set_entry_point("verificar_entrada")
    g.add_conditional_edges("verificar_entrada",
                            lambda e: "recusar" if e.get("status") == "bloqueada" else "verificar_escopo",
                            {"recusar": "recusar", "verificar_escopo": "verificar_escopo"})
    g.add_conditional_edges("verificar_escopo",
                            lambda e: "rotear" if e["escore_escopo"] >= LIMIAR_ESCOPO else "abster",
                            {"rotear": "rotear", "abster": "abster"})
    g.add_conditional_edges("rotear", lambda e: e["rota"],
                            {"frequencia": "analisar_frequencia", "rag": "transformar"})
    g.add_edge("transformar", "recuperar")
    g.add_edge("recuperar", "julgar")
    g.add_conditional_edges("julgar", rota_apos_julgar,
                            {"gerar": "gerar", "reescrever": "reescrever", "abster": "abster"})
    g.add_edge("reescrever", "recuperar")
    for fim in ("gerar", "abster", "recusar", "analisar_frequencia"):
        g.add_edge(fim, END)
    return g.compile()
