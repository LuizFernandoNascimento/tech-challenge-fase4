"""Carrega as chaves de API no ambiente do processo, sem nunca imprimi-las.

Local: lê de um arquivo .env. No Colab: lê do painel Secrets (ícone de chave 🔑).
"""
import os
from pathlib import Path

CHAVES = ("GOOGLE_API_KEY", "GROQ_API_KEY")


def carregar_chave(arquivo: str | None = None) -> dict:
    """Devolve {nome: carregada?} - nunca o valor."""
    try:
        from google.colab import userdata  # type: ignore
        for nome in CHAVES:
            try:
                os.environ.setdefault(nome, userdata.get(nome))
            except Exception:
                pass
    except ImportError:
        arquivo = arquivo or os.environ.get("ARQUIVO_CHAVE", str(Path.home() / "Downloads" / "gemini.env"))
        if Path(arquivo).exists():
            for linha in Path(arquivo).read_text().splitlines():
                nome, _, valor = linha.partition("=")
                if nome.strip() in CHAVES and valor.strip():
                    os.environ.setdefault(nome.strip(), valor.strip().strip('"').strip("'"))
    return {nome: bool(os.environ.get(nome)) for nome in CHAVES}
