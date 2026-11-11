# Inventário de escopo — Fase 4 (17 PDFs lidos)
**Régua: nada entra na solução que não esteja aqui.**

## As três trilhas

| Trilha | Aulas | Natureza |
|---|---|---|
| **1IAST – IA Generativa** | 6 | teoria de GenAI |
| **Documento de Aula** | 6 | fundamentos de NLP |
| **POSTECH** | 5 | prática LangChain |

---

## TRILHA 1IAST — IA GENERATIVA

**Aula 1 — Arquitetura Transformer**
Encoder/decoder, self-attention (Q,K,V), multi-head, positional encoding, BPE,
fine-tuning, LoRA, famílias encoder-only/decoder-only/encoder-decoder, HuggingFace Transformers.

**Aula 2 — RAG**  ← AULA CENTRAL DO TECH CHALLENGE
- Memória paramétrica vs. não-paramétrica (Lewis et al., 2020)
- Pipeline: ingestão → chunking → embeddings → indexação → recuperação → geração
- Chunking: fixed-size com overlap 10–20%, recursive, **semantic chunking**
- Metadados no chunk para filtragem pré-recuperação; indexação hierárquica
- Vector stores citados: Pinecone, Milvus, Qdrant, Weaviate, pgvector, **FAISS**, Bedrock KB
- Índices: Flat, **kNN exato** ("custa caro quando o índice cresce"), **ANN/HNSW**
- **Recuperação lexical (BM25) + vetorial + HÍBRIDA**
- **Reciprocal Rank Fusion, constante k = 60**, "sem exigir normalização de escores"
- **Re-ranking** por modelo secundário
- Geração: "responder **apenas com base no contexto**; **exigir recusa explícita quando o
  contexto for insuficiente**; devolver a **lista de fontes** usadas"
- Avaliação: **recall@k**, **faithfulness/groundedness**, **answer relevancy**
- Operacional: latência, custo/requisição, drift, **pontuação de confiança** → revisão humana
- Ferramentas: **RAGAS**, ARES, LangSmith, Galileo, Vectara HHEM, LLM-as-a-Judge
- Nuvem: Bedrock KB (AWS), Azure AI Search, **Vertex AI Embeddings / Vector Search / RAG Engine (GCP)**

**Aula 3 — Engenharia de Prompts**
Zero-shot, few-shot, **chain-of-thought**, self-consistency, prompt chaining;
instrução/contexto/exemplos/política de saída; **tags XML** como fronteiras;
personas; "lost in the middle" (instrução no fim); temperatura, top-k, top-p;
prompt injection direta e indireta, prompt leaking; **DeepEval, RAGAS, G-Eval**.

**Aula 4 — Agents**
Agente vs. workflow; **ReAct**; tool calling com **JSON Schema**; 5 elementos da ferramenta
(nome, descrição, schema entrada, schema saída, **classificação de risco**);
MCP, A2A, AP2; planners, Plan-and-Solve, Tree of Thoughts; hierárquicos e multiagente;
memória curto/longo prazo; observabilidade: logs, métricas, **traces**, OpenTelemetry.

**Aula 5 — LLMs Multimodais**
CLIP, contrastive learning, encoder visual + LLM, cross-attention, adapters, BLIP-2, LLaVA;
VQA; **Multimodal RAG**; chunking com OCR/layout; visual grounding; MMMU, HallusionBench.

**Aula 6 — Ética, Viés, Segurança e Impacto Social**
Sistema sociotécnico; **NIST AI RMF**, **NIST AI 600-1**; **OWASP LLM Top 10 (2025)**;
seis famílias de risco; **prompt injection indireta é especialmente perigosa em RAG**
("a aplicação tende a tratar conteúdo recuperado como contexto confiável");
**guardrails em camadas** (entrada, contexto, saída, ação) — na saída:
"**exigir citação ou evidência quando a aplicação depende de grounding**";
excessive agency; LGPD, ANPD, EU AI Act; governança de dados em RAG
(permissão por documento/trecho/usuário); resposta a incidentes.

---

## TRILHA DOCUMENTO DE AULA — NLP

**Aula 1 — Fundamentos de NLP**
NLP/NLU/NLG; três ondas; ambiguidade, polissemia, ironia; morfologia/sintaxe/semântica/pragmática;
**hipótese distribucional (Harris, 1954)**; NLTK, **spaCy `pt_core_news_lg`**; CountVectorizer;
português tem morfologia rica → BERTimbau, NILC, FastText.

**Aula 2 — Pré-processamento de Texto**
Regex: HTML, URLs, e-mails; lowercase; acentos (decisão depende do downstream);
**expansão de contrações PT-BR** ("vc"→"você", "pq"→"por que");
tokenização: split / NLTK `word_tokenize(language="portuguese")` / spaCy / subword BPE;
**stemming RSLP** vs. **lemmatization spaCy**; stop words — cuidado: remover "não" arruína sentimento;
**pipeline modular e idempotente**.

**Aula 3 — BoW e TF-IDF**
CountVectorizer; **n-grams** (bigrams capturam "não gostei"); TF-IDF, `sublinear_tf`,
`min_df`/`max_df`; esparsidade; **BM25 como "primo do TF-IDF"**;
caso Mercado Livre: "**primeiro estágio do retrieval ainda é BM25; embeddings entram no
reranking dos top-K. Essa arquitetura híbrida mantém latência baixa e qualidade alta**";
SPLADE, ColBERT.

**Aula 4 — Embeddings**
Word2Vec (CBOW/Skip-gram, negative sampling), GloVe, **FastText (subwords, ideal p/ PT-BR)**;
NILC/USP; **BERTimbau**; similaridade de cosseno; analogias; t-SNE, UMAP;
embeddings contextuais; "para qualidade superior, **BERTimbau ou Sentence-BERT em português**";
bancos vetoriais: Pinecone, Weaviate, Chroma, pgvector.

**Aula 5 — Modelos Clássicos (ML tradicional)**
Naive Bayes Multinomial (baseline obrigatório), **Regressão Logística**, LinearSVC;
**Pipeline do scikit-learn** (evita data leakage); GridSearchCV, StratifiedKFold;
métricas: precisão, recall, **F1 macro/micro**, matriz de confusão, armadilha da acurácia;
**NER com spaCy** (PER/ORG/LOC/MISC/DATE/MONEY), seqeval, BIO tagging.

**Aula 6 — Modelagem de Tópicos**
LSA (SVD sobre TF-IDF), **LDA** (Gensim, alpha/beta, Gibbs/Variational), **NMF**;
**coherence c_v** (>0,5 = interpretável), perplexidade (correlaciona mal), elbow plot;
**pyLDAvis**; **BERTopic** = Sentence-BERT + **UMAP** + **HDBSCAN** + c-TF-IDF;
visualizações nativas; "inspeção humana é insubstituível".

---

## TRILHA POSTECH — LANGCHAIN

**Aula 1 — Introduction to LangChain** — LangChain (167 menções), OpenAI (28).

**Aula 2 — Chains and Agents**
**LCEL**; protocolo `Runnable` (`.invoke/.batch/.stream` + async); operador `|`;
`RunnableParallel`, **`RunnablePassthrough`** ("ferramenta principal para injetar contexto
em sistemas RAG"), `RunnableLambda`, `RunnableConfig`;
**Router Chains** / `RunnableBranch`; ReAct, Plan-and-Execute;
**Custom Tools**: `@tool`, `BaseTool`, `StructuredTool.from_function`;
**`.with_structured_output()` + Pydantic**; **`.with_fallbacks()`**, retries, self-healing;
**LangSmith** e Callbacks.

**Aula 3 — Vector Databases and Memory Management**
RAG; embeddings (`text-embedding-3-small`, 1536 dim); cosseno/euclidiana/**dot product**
("se normalizados, idêntico ao cosseno e mais rápido");
**ChromaDB (hands-on)**, Pinecone, FAISS; **HNSW**, **IVF**;
**Re-ranking com Cross-Encoder** (vs. Bi-Encoder); **Query Transformation: HyDE, Step-back**;
**RAGAS** (Faithfulness, Answer Relevance, Context Recall/Precision);
memória: Buffer, Window, Summary, **`RunnableWithMessageHistory`**, Knowledge Graph Memory.
Fecho: "construiu um pipeline usando **LangChain e ChromaDB**".

**Aula 4 — Integrations, Tools, Deployment**
Tool calling; **Pydantic mandatório**; ReAct, Plan-and-Solve;
**LangServe + FastAPI** (REST); **LangSmith**; Docker/K8s; **semantic caching**; MAS.

**Aula 5 — LangGraph**
**`StateGraph`**, nodes, **conditional edges**, **State com `TypedDict`**;
**ciclos permitidos → Self-Reflection**; exemplo literal da aula:
"*se o agente de busca não encontrar o resultado, retorne para o nó de query-rewrite;
se encontrar, siga para o nó de sumarização*";
**Checkpointers** (SQLite/Postgres/Redis), time-travel debugging;
**Human-in-the-loop**: Interrupt, Approvals, Overrides; Supervisor pattern; GraphRAG.
