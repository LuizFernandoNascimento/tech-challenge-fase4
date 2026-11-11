"""Pré-processamento de texto em português (Documento de Aula 2).

A aula recomenda manter DUAS versões do pipeline:

    "Em projeto sério, você documenta essa decisão e mantém duas versões do
    pipeline: uma agressiva para BoW/TF-IDF, outra leve para modelos
    contextuais."

É exatamente o que acontece aqui:

* `limpar_leve` - o texto que vai para o EMBEDDING e para o LLM. Preserva
  acentos, maiúsculas e pontuação, porque modelos contextuais usam isso.
* `tokenizar_bm25` - os tokens que vão para o BM25. Agressivo: minúsculas,
  sem acento, contrações expandidas, stop words removidas e stemming RSLP.

Uma regra vale para os dois: NEGAÇÕES NUNCA SÃO REMOVIDAS. A aula alerta que
"para sentiment analysis em PT, perder 'não' pode arruinar o modelo" - e
"chegou" versus "não chegou" é a diferença inteira numa reclamação de entrega.

As duas funções são idempotentes: aplicar duas vezes dá o mesmo resultado.
"""
from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

# Mapa de contrações do português informal (Documento de Aula 2), ampliado com
# abreviações que aparecem nas próprias avaliações da Olist.
CONTRACOES = {
    "vc": "você", "vcs": "vocês", "tb": "também", "tbm": "também",
    "pq": "porque", "obg": "obrigado", "obgd": "obrigado", "blz": "beleza",
    "td": "tudo", "vlw": "valeu", "q": "que", "msm": "mesmo", "mt": "muito",
    "mto": "muito", "mts": "muitos", "n": "não", "nao": "não", "naum": "não",
    "hj": "hoje", "qdo": "quando", "qnd": "quando", "ate": "até",
    "pra": "para", "pro": "para o", "dps": "depois", "cmg": "comigo",
}

# Palavras que NUNCA saem, mesmo constando da lista de stop words do NLTK.
NEGACOES = {"não", "nao", "nem", "nunca", "jamais", "nada", "nenhum",
            "nenhuma", "sem", "ninguém", "ninguem"}


def limpar_leve(texto: str) -> str:
    """Limpeza mínima para modelos contextuais (embedding e LLM)."""
    if not isinstance(texto, str):
        return ""
    t = re.sub(r"<[^>]+>", " ", texto)                 # HTML residual
    t = re.sub(r"http\S+|www\.\S+", " ", t)              # URLs
    t = re.sub(r"\S+@\S+", " ", t)                       # e-mails
    t = re.sub(r"([!?.])\1{2,}", r"\1\1", t)             # "!!!!!!" -> "!!"
    return re.sub(r"\s+", " ", t).strip()


def _sem_acento(texto: str) -> str:
    nfkd = unicodedata.normalize("NFKD", texto)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


@lru_cache(maxsize=1)
def _recursos_nltk():
    """Carrega stop words e stemmer RSLP uma única vez."""
    import nltk
    for recurso in ("stopwords", "rslp", "punkt", "punkt_tab"):
        nltk.download(recurso, quiet=True)
    from nltk.corpus import stopwords
    from nltk.stem import RSLPStemmer

    stop = set(stopwords.words("portuguese")) - NEGACOES
    stop |= {_sem_acento(s) for s in stop} - {_sem_acento(n) for n in NEGACOES}
    return stop, RSLPStemmer()


def tokenizar_bm25(texto: str) -> list[str]:
    """Pipeline agressivo para o BM25 (Documento de Aula 2, Passo 3).

    limpeza -> minúsculas -> contrações -> tokenização NLTK -> stop words
    (preservando negações) -> stemming RSLP.

    O stemming agrupa "atrasou", "atraso" e "atrasada" no mesmo radical, que é
    o que uma busca por palavra-chave precisa para não perder variações.
    """
    from nltk.tokenize import word_tokenize

    stop, stemmer = _recursos_nltk()
    t = limpar_leve(texto).lower()
    t = " ".join(CONTRACOES.get(p, p) for p in t.split())
    tokens = word_tokenize(t, language="portuguese")
    saida = []
    for tk in tokens:
        tk_norm = _sem_acento(tk)
        if not re.fullmatch(r"[a-z0-9]+", tk_norm):
            continue
        if tk_norm in NEGACOES or _sem_acento(tk) in {_sem_acento(n) for n in NEGACOES}:
            saida.append("nao" if tk_norm in {"nao", "não"} else tk_norm)
            continue
        if tk_norm in stop or len(tk_norm) < 2:
            continue
        saida.append(stemmer.stem(tk_norm))
    return saida
