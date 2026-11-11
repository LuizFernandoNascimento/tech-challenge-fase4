"""Interface de consulta: a função `perguntar()`.

Mostra as cinco etapas que o Requisito 2 (pág. 7) pede para "demonstrar como
ocorre": transformação da pergunta, recuperação semântica, seleção dos mais
relevantes, construção do contexto e geração da resposta. E, em seguida, as
evidências que sustentam a resposta (Requisito 3) com o caminho percorrido
pelo grafo - inclusive quando ele decide não responder (Requisito 4).
"""
from __future__ import annotations

import textwrap

from src.config import LIMIAR_ESCOPO
from src.generation import formatar_contexto

LARGURA = 92


def _secao(titulo: str) -> str:
    return f"\n{'═' * 3} {titulo} {'═' * max(3, LARGURA - len(titulo) - 5)}"


def _quebrar(texto: str, recuo: int = 2) -> str:
    return textwrap.fill(texto, LARGURA, initial_indent=" " * recuo, subsequent_indent=" " * recuo)


def _linha_doc(d, posicao: int | None = None) -> str:
    # A nota do Cross-Encoder não é exibida: medimos que ela é baixa para
    # perguntas agregadas mesmo quando a avaliação é relevante, e mostrá-la ao
    # lado de um "✓" do julgamento confundiria o leitor. Ela só ordena - então
    # o que se mostra é a posição.
    m = d.metadata
    partes = [f"[{m['doc_id']}]"]
    if posicao is not None:
        partes.insert(0, f"#{posicao:<2}")
    partes.append(f"nota {m['nota_media']:.0f}")
    if m.get("n_ocorrencias", 1) > 1:
        partes.append(f"{m['n_ocorrencias']} clientes")
    if m.get("entrega_atrasada") is True:
        partes.append(f"atraso {m.get('atraso_dias')}d")
    if m.get("categoria"):
        partes.append(m["categoria"][:18])
    return f"  {' | '.join(partes)}\n      \"{d.page_content[:150]}\""


def formatar(resultado: dict, mostrar_contexto: bool = False) -> str:
    r = resultado
    out = []
    out.append(_secao("PERGUNTA"))
    out.append(_quebrar(r["pergunta"]))
    if r.get("filtro"):
        out.append(f"  filtro de metadados: {r['filtro']}")

    if "escore_escopo" in r:
        marca = "dentro do escopo" if r["escore_escopo"] >= LIMIAR_ESCOPO else "FORA do escopo"
        out.append(f"  proximidade com a base: {r['escore_escopo']:.3f} (limiar {LIMIAR_ESCOPO}) -> {marca}")

    if r.get("consulta_busca"):
        out.append(_secao("1. TRANSFORMAÇÃO DA PERGUNTA (HyDE)"))
        for linha in r["consulta_busca"].splitlines()[:5]:
            out.append(_quebrar(linha.strip(), 2)) if linha.strip() else None

    if r.get("documentos"):
        out.append(_secao("2. RECUPERAÇÃO HÍBRIDA (BM25 + Chroma, fusão RRF c=60, re-ranking)"))
        out.append(f"  {len(r['documentos'])} avaliações após o re-ranking com Cross-Encoder")

        out.append(_secao("3. SELEÇÃO (julgamento de relevância pelo LLM)"))
        rel = {d.metadata["doc_id"] for d in r.get("relevantes", [])}
        out.append(f"  {len(rel)} de {len(r['documentos'])} julgadas relevantes")
        if r.get("justificativa_julgamento"):
            out.append(_quebrar(f"critério: {r['justificativa_julgamento']}"))
        out.append("  (✓ = relevante segundo o julgamento; posição = ordem do re-ranking)")
        for pos, d in enumerate(r["documentos"], start=1):
            marca = "✓" if d.metadata["doc_id"] in rel else "·"
            out.append(f"{marca}" + _linha_doc(d, pos)[1:])

        if mostrar_contexto and r.get("relevantes"):
            out.append(_secao("4. CONTEXTO ENVIADO AO MODELO"))
            out.append(textwrap.indent(formatar_contexto(r["relevantes"]), "  "))

    out.append(_secao("5. RESPOSTA" if r.get("status") == "respondida" else "RESPOSTA"))
    out.append(_quebrar(r.get("resposta", "")))
    out.append(f"\n  status: {r.get('status', '?').upper()}")
    if r.get("limitacoes"):
        out.append(_quebrar(f"limitações: {r['limitacoes']}"))

    if r.get("ids_evidencias"):
        out.append(_secao(f"EVIDÊNCIAS CITADAS ({len(r['ids_evidencias'])})"))
        por_id = {d.metadata["doc_id"]: d for d in r.get("relevantes", [])}
        for i in r["ids_evidencias"]:
            if i in por_id:
                out.append(_linha_doc(por_id[i]))
    if r.get("ids_invalidos"):
        out.append(f"\n  ⚠ citações descartadas pelo guardrail (não recuperadas): {r['ids_invalidos']}")

    out.append(_secao("CAMINHO NO GRAFO"))
    out.append("  " + "  ->  ".join(r.get("trilha", [])))
    return "\n".join(out)


def criar_perguntar(grafo):
    """Devolve a função `perguntar` já ligada ao grafo construído."""
    def perguntar(pergunta: str, filtro: dict | None = None, usar_hyde: bool = True,
                  mostrar: bool = True, mostrar_contexto: bool = False) -> dict:
        try:
            resultado = grafo.invoke({"pergunta": pergunta, "filtro": filtro, "usar_hyde": usar_hyde})
        except Exception as erro:
            # Com as cadeias de fallback, só chega aqui se TODOS os modelos
            # falharam - na prática, cota gratuita esgotada em todos (D3-D5).
            # Melhor uma mensagem clara do que interromper o notebook.
            texto = str(erro)
            cota = any(t in texto for t in ("429", "quota", "Rate limit", "RESOURCE_EXHAUSTED"))
            if not cota:
                raise
            resultado = {"status": "cota_esgotada", "rota": None, "trilha": ["erro:cota_esgotada"],
                         "resposta": "Cota gratuita esgotada em todos os modelos da cadeia de "
                                     "fallback. As cotas renovam diariamente; tente mais tarde.",
                         "ids_evidencias": [], "documentos": [], "relevantes": []}
        resultado["pergunta"] = pergunta
        resultado["filtro"] = filtro
        if mostrar:
            print(formatar(resultado, mostrar_contexto))
        return resultado
    return perguntar


def evidencias(resultado: dict, base) -> "pd.DataFrame":
    """Rastreabilidade (Requisito 3): cada avaliação citada na resposta,
    ligada de volta ao(s) `review_id` original(is) do dataset da Olist.

    `base`: DataFrame de artifacts/base_conhecimento.parquet.
    """
    import pandas as pd
    ids = resultado.get("ids_evidencias") or []
    if not ids:
        return pd.DataFrame()
    colunas = ["doc_id", "texto", "nota_media", "n_ocorrencias", "categoria",
               "primeira_data", "entrega_atrasada", "review_ids"]
    tabela = base[base["doc_id"].isin(ids)][[c for c in colunas if c in base.columns]].copy()
    tabela["doc_id"] = pd.Categorical(tabela["doc_id"], categories=ids, ordered=True)
    return tabela.sort_values("doc_id").reset_index(drop=True)
