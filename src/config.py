"""Configuração central do projeto.

Cada constante aponta para a aula que justifica a escolha. O enunciado delega
a escolha de tecnologias à equipe; aqui ficam registradas as escolhas e onde
cada uma foi ensinada.
"""
import os
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
DIR_DADOS = Path(os.environ.get("DIR_DADOS", RAIZ / "data"))
DIR_ARTEFATOS = Path(os.environ.get("DIR_ARTEFATOS", RAIZ / "artifacts"))
DIR_CHROMA = DIR_ARTEFATOS / "chroma"

# --- Base de conhecimento -------------------------------------------------
# Comentários menores que isso ("Ok", "Bom", ":)") não sustentam nenhum insight.
MIN_CARACTERES = 12

# --- Embeddings (Documento de Aula 4 e 6) ---------------------------------
# A Aula 4 recomenda "Sentence-BERT em português" para qualidade superior.
# Este modelo específico é o usado literalmente no código da Aula 6 (BERTopic).
MODELO_EMBEDDING = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# --- Banco vetorial (POSTECH Aula 3) ---------------------------------------
# ChromaDB foi o banco usado no hands-on. Usa HNSW (ANN, também ensinado) e
# aceita filtro por metadado ANTES da busca - o que a 1IAST Aula 2 descreve
# como "filtragem pré-recuperação que reduz o espaço de busca".
COLECAO_CHROMA = "avaliacoes_olist"

# --- Recuperação híbrida (1IAST Aula 2) -------------------------------------
# "Reciprocal Rank Fusion (...) onde k é uma constante (comumente 60)".
K_RRF = 60
# Pesos medidos, não chutados: ver a varredura em notebooks/ (seção de avaliação).
PESO_DENSO = 0.7
PESO_LEXICO = 0.3
K_CANDIDATOS = 50          # quantos cada recuperador devolve antes da fusão

# --- Re-ranking (POSTECH Aula 3 e 1IAST Aula 2) -----------------------------
# Cross-Encoder avalia pergunta e documento JUNTOS, ao contrário do bi-encoder
# da busca inicial. Modelo treinado no mMARCO, versão multilíngue do MS MARCO
# que inclui português.
MODELO_RERANKER = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
TOP_N_RERANK = 10          # quantas evidências seguem para o LLM

# --- Modelagem de tópicos (Documento de Aula 6) -------------------------------
# Responde às perguntas de FREQUÊNCIA do enunciado (pág. 4 e 5), que o RAG
# sozinho não responde: ele recupera exemplos, não conta.
# Escolhido entre 6 configurações (ver docs/decisoes_tecnicas.md, D16):
# eom/50 dá 84 tópicos com coerência c_v de 0,621 e a menor taxa de avaliações
# sem tópico entre as configurações que não colapsaram.
TOPICO_MIN_CLUSTER = 50
TOPICO_METODO = "eom"
# Outliers só voltam a um tópico se a similaridade com o centróide for >= 0,70.
# Abaixo disso a inspeção mostrou encaixes errados ("falhas na costura" ->
# "peça faltando"). Reduz as avaliações sem tópico de 42,8% para 16,9%.
LIMIAR_OUTLIER = 0.70
SEMENTE = 42

# --- Modelos de linguagem (ver docs/decisoes_tecnicas.md, D1 a D5) -----------
# A camada gratuita do Gemini permite 20 requisições/dia POR MODELO (D3). Por
# isso cada papel tem uma cadeia de fallback (POSTECH Aula 2, .with_fallbacks):
# quando a cota de um modelo acaba, a cadeia passa para o próximo sozinha.
#
# Cada item é (provedor, modelo). A ordem é a ordem de tentativa.
CADEIA_PRINCIPAL = [            # julgamento de relevância e geração da resposta
    ("google", os.environ.get("MODELO_LLM", "gemini-3.5-flash")),
    ("google", "gemini-3.1-flash-lite"),
    ("groq", "openai/gpt-oss-120b"),
    # Último recurso: o 120b tem teto de 200 mil tokens/dia, que a avaliação
    # RAGAS chegou a esgotar (D5). O 20b tem cota própria e foi testado com os
    # três esquemas de saída estruturada.
    ("groq", "openai/gpt-oss-20b"),
]
CADEIA_AUXILIAR = [             # HyDE e reescrita: etapas de menor risco
    ("google", "gemini-3.5-flash-lite"),
    ("groq", "openai/gpt-oss-20b"),
]
CADEIA_JUIZ = [                 # métricas do RAGAS: outra família de modelo (D5)
    ("groq", "openai/gpt-oss-120b"),
    ("google", "gemini-3.1-flash-lite"),
]
# 1IAST Aula 3: "valores baixos (0.1 - 0.2) tornam o modelo conservador e
# determinístico" - é o que se quer de um sistema que deve se ater às evidências.
TEMPERATURA = 0.1
# Uma tentativa só por modelo: em erro de cota, o cliente do Gemini dorme ~20 s
# antes de cada nova tentativa. Com fallback disponível, esperar é desperdício.
TENTATIVAS_POR_MODELO = 1
TIMEOUT_SEGUNDOS = 45

# --- Abstenção em três camadas (Requisito 4) --------------------------------
# CAMADA 1 - escopo, ANTES de chamar o LLM.
# Média da similaridade de cosseno das 10 avaliações mais próximas da pergunta.
# Calibrado em 50 perguntas (scripts/calibracao_perguntas.py), incluindo TODAS as
# 18 perguntas que o enunciado cita (páginas 4, 5, 8 e 9).
#
# Regra que orienta o valor: esta camada NUNCA pode recusar uma pergunta do
# enunciado. A mais baixa delas pontua 0,509 ("Quais evidências presentes nos
# comentários sustentam determinado insight?"), seguida de 0,510 ("Quais
# problemas aparecem com maior frequência nas avaliações?"). São as mais
# genéricas - falam "das avaliações" sem tema concreto.
#
# Com 0,48: 18/18 do enunciado aceitas, com folga de 0,029. Com 0,50 a folga
# seria de 0,009 - uma reformulação derrubaria a pergunta. O custo é barrar só
# 8/16 perguntas fora do escopo nesta camada; as demais seguem para a camada 2.
#
# Histórico: o valor anterior, 0,53, foi calibrado só com as perguntas da
# página 8 e recusou duas perguntas literais das páginas 4 e 5. Ver D13.
#
# Por que NÃO a nota do Cross-Encoder: medimos e ela não separa os grupos
# ("sintomas da gripe" pontuou acima de quatro perguntas legítimas). O modelo foi
# treinado para "este trecho responde à pergunta?", e perguntas agregadas
# ("quais os principais problemas?") não são respondidas por nenhuma avaliação
# isolada. O Cross-Encoder serve para ORDENAR, não para medir confiança.
LIMIAR_ESCOPO = float(os.environ.get("LIMIAR_ESCOPO", "0.48"))
K_ESCOPO = 10

# CAMADA 2 - suficiência, julgada pelo LLM (LLM-as-a-Judge, 1IAST Aulas 2 e 3).
MIN_EVIDENCIAS = 3          # avaliações julgadas relevantes para poder responder
MAX_REESCRITAS = 1          # POSTECH Aula 5: ciclo buscar -> reescrever -> buscar

# CAMADA 3 - geração: campo `evidencia_suficiente` + validação das citações.
