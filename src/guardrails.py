"""Guardrails - 1IAST Aula 6 (Ética, Viés, Segurança).

A aula alerta para um risco que vale exatamente para este projeto:

    "A forma indireta [de prompt injection] ocorre quando o modelo consome
    conteúdo externo (...) que contém instruções maliciosas. A segunda forma é
    especialmente perigosa em aplicações RAG, porque a aplicação tende a tratar
    conteúdo recuperado como contexto confiável."

A base de conhecimento é feita de texto escrito por clientes anônimos. Qualquer
um que escreveu "ignore as instruções anteriores" numa avaliação colocou esse
texto dentro do prompt do sistema. A aula recomenda defesa em camadas; aqui são
três, nas posições que ela descreve:

* ENTRADA  - a pergunta do usuário é verificada antes de qualquer busca.
* CONTEXTO - cada avaliação recuperada é marcada como dado não-confiável e
  delimitada por tags (separação entre instruções e dados).
* SAÍDA    - a resposta só é aceita se citar evidências que de fato foram
  recuperadas ("exigir citação ou evidência quando a aplicação depende de
  grounding").
"""
from __future__ import annotations

import re
import unicodedata

PADROES_INJECAO = [
    r"ignor[ea]\w* (as |todas as )?(instru|regras|ordens)",
    r"desconsider[ea]\w* (as |todas as )?(instru|regras)",
    r"esque[cç]a (as |tudo|todas)",
    r"(voce|tu) (agora )?(e|sera|vai ser) (um|uma) ",
    r"system ?prompt", r"prompt (do|de) sistema",
    r"revel[ea]\w* (o |suas? )?(prompt|instru)",
    r"ignore (all |the |previous |prior )?(instructions|rules)",
    r"you are now", r"act as", r"jailbreak",
]
_REGEX = re.compile("|".join(PADROES_INJECAO), re.IGNORECASE)


def _norm(texto: str) -> str:
    t = unicodedata.normalize("NFKD", texto.lower())
    return "".join(c for c in t if not unicodedata.combining(c))


def detectar_injecao(texto: str) -> str | None:
    """Devolve o trecho suspeito, ou None se o texto parecer limpo."""
    achado = _REGEX.search(_norm(texto))
    return achado.group(0) if achado else None


def neutralizar(texto: str) -> str:
    """Prepara uma avaliação para entrar no contexto como dado, nunca como ordem.

    Remove sinais de tag que poderiam fechar o delimitador XML e, se houver
    padrão de injeção, marca o trecho explicitamente para o modelo.
    """
    t = texto.replace("<", "‹").replace(">", "›")
    if detectar_injecao(t):
        t = f"[conteúdo marcado como suspeito pelo guardrail] {t}"
    return t


def validar_citacoes(ids_citados: list[str], ids_recuperados: set[str],
                     texto: str = "") -> tuple[list[str], list[str]]:
    """Separa citações válidas das inventadas pelo modelo.

    Confere o campo estruturado E os ids escritos no próprio texto da resposta.
    Medimos que o modelo às vezes cita no texto ids que não repete no campo
    estruturado (37 casos na bateria, D22); conferir só o campo deixaria
    passar um id inventado escrito no texto.
    """
    todos = list(dict.fromkeys(list(ids_citados) + re.findall(r"doc_\d{6}", texto or "")))
    validos = [i for i in todos if i in ids_recuperados]
    invalidos = [i for i in todos if i not in ids_recuperados]
    return validos, invalidos


def remover_citacoes(texto: str, invalidos: list[str]) -> str:
    """Tira do texto as citações inválidas, marcando onde estavam."""
    for i in invalidos:
        texto = texto.replace(i, "citação removida")
    return texto
