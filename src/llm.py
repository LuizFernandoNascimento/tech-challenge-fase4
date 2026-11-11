"""Fábrica de modelos de linguagem com fallback e cache.

Duas técnicas ensinadas, aplicadas ao limite medido da camada gratuita
(docs/decisoes_tecnicas.md, D3 a D5):

* `.with_fallbacks()` - POSTECH Aula 2: "Se a API retornar um erro 500 ou Rate
  Limit, a chain alterna automaticamente para outro modelo." Cada papel do
  pipeline recebe uma cadeia; quando a cota de um modelo acaba, a próxima
  chamada vai para o seguinte, sem intervenção.
* Cache de respostas - POSTECH Aula 4. A mesma pergunta, com o mesmo contexto,
  não volta a consumir cota. Isso protege a cota durante o desenvolvimento e
  quando o avaliador re-executa o notebook.

O fallback é aplicado DEPOIS de `.with_structured_output()` em cada modelo: os
provedores implementam saída estruturada de jeitos diferentes, e cada um
precisa da sua própria versão da cadeia.
"""
from __future__ import annotations

import logging
import os

from src.config import (CADEIA_AUXILIAR, CADEIA_JUIZ, CADEIA_PRINCIPAL,
                        DIR_ARTEFATOS, TEMPERATURA, TENTATIVAS_POR_MODELO,
                        TIMEOUT_SEGUNDOS)

logger = logging.getLogger(__name__)
_CACHE_ATIVO = False


def ativar_cache() -> None:
    """Liga o cache em disco (SQLite) para todas as chamadas ao LLM."""
    global _CACHE_ATIVO
    if _CACHE_ATIVO:
        return
    from langchain_community.cache import SQLiteCache
    from langchain_core.globals import set_llm_cache
    DIR_ARTEFATOS.mkdir(parents=True, exist_ok=True)
    set_llm_cache(SQLiteCache(database_path=str(DIR_ARTEFATOS / "cache_llm.sqlite")))
    _CACHE_ATIVO = True


def _instanciar(provedor: str, modelo: str):
    if provedor == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=modelo, temperature=TEMPERATURA,
                                      max_retries=TENTATIVAS_POR_MODELO,
                                      timeout=TIMEOUT_SEGUNDOS)
    if provedor == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(model=modelo, temperature=TEMPERATURA,
                        max_retries=TENTATIVAS_POR_MODELO, timeout=TIMEOUT_SEGUNDOS)
    raise ValueError(f"Provedor desconhecido: {provedor}")


def _disponiveis(cadeia):
    """Descarta provedores sem chave configurada, em vez de falhar na hora da chamada."""
    chaves = {"google": "GOOGLE_API_KEY", "groq": "GROQ_API_KEY"}
    usaveis = [(p, m) for p, m in cadeia if os.environ.get(chaves[p])]
    if not usaveis:
        raise RuntimeError("Nenhuma chave de API configurada para esta cadeia de modelos.")
    return usaveis


def montar(cadeia, esquema=None):
    """Monta a cadeia de fallback. Com `esquema`, cada modelo devolve esse Pydantic."""
    modelos = []
    for provedor, nome in _disponiveis(cadeia):
        m = _instanciar(provedor, nome)
        if esquema is not None:
            # O gpt-oss não usa function calling no Groq; json_schema é o modo nativo.
            m = (m.with_structured_output(esquema, method="json_schema")
                 if provedor == "groq" else m.with_structured_output(esquema))
        modelos.append(m)
    principal, *reservas = modelos
    return principal.with_fallbacks(reservas) if reservas else principal


def principal(esquema=None):
    return montar(CADEIA_PRINCIPAL, esquema)


def auxiliar(esquema=None):
    return montar(CADEIA_AUXILIAR, esquema)


def juiz(esquema=None):
    return montar(CADEIA_JUIZ, esquema)


def modelo_juiz_bruto():
    """Primeiro modelo da cadeia do juiz, sem fallback - o RAGAS precisa do objeto LLM."""
    provedor, nome = _disponiveis(CADEIA_JUIZ)[0]
    return _instanciar(provedor, nome)


# --------------------------------------------------------------------------
# Rastreamento: qual modelo de fato respondeu
# --------------------------------------------------------------------------
# Com fallback, a resposta pode vir de qualquer modelo da cadeia. Registrar
# quem tentou e quem respondeu é rastreabilidade - e torna visível, na trilha
# do grafo, quando o limite da camada gratuita foi atingido.
from langchain_core.callbacks import BaseCallbackHandler


class RastreadorModelos(BaseCallbackHandler):
    def __init__(self):
        self.chamadas: dict = {}
        self.ordem: list = []

    def on_chat_model_start(self, serialized, messages, *, run_id, invocation_params=None,
                            metadata=None, **kw):
        # `ls_model_name` é o campo padrão que todo provedor preenche (é o que o
        # LangSmith usa); os parâmetros de invocação variam de provedor para provedor.
        p = invocation_params or {}
        nome = (metadata or {}).get("ls_model_name") or p.get("model") or p.get("model_name") or "?"
        self.chamadas[run_id] = [str(nome).replace("models/", ""), "…"]
        self.ordem.append(run_id)

    def on_llm_end(self, response, *, run_id, **kw):
        if run_id in self.chamadas:
            self.chamadas[run_id][1] = "✓"

    def on_llm_error(self, error, *, run_id, **kw):
        if run_id in self.chamadas:
            cota = "429" in str(error) or "quota" in str(error).lower() or "rate" in str(error).lower()
            self.chamadas[run_id][1] = "✗cota" if cota else "✗"

    def resumo(self) -> str:
        return " → ".join(f"{n}{s}" for n, s in (self.chamadas[r] for r in self.ordem))
