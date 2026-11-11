"""Escolha do tamanho mínimo de tópico do BERTopic (Documento de Aula 6).

UMAP roda uma única vez; as variações de HDBSCAN reaproveitam a projeção.
"""
import logging, time
import numpy as np, pandas as pd
logging.basicConfig(level=logging.WARNING)
from umap import UMAP
from hdbscan import HDBSCAN
from bertopic import BERTopic
from bertopic.dimensionality import BaseDimensionalityReduction
from sklearn.feature_extraction.text import CountVectorizer
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel
from src.vectorstore import carregar
from src.preprocessing import _recursos_nltk, NEGACOES



def main():
    col = carregar()._collection
    dados = col.get(include=["documents", "embeddings", "metadatas"])
    ordem = np.argsort([m["doc_id"] for m in dados["metadatas"]])
    textos = [dados["documents"][i] for i in ordem]
    emb = np.array([dados["embeddings"][i] for i in ordem], dtype="float32")
    print(f"{len(textos)} documentos, embeddings {emb.shape} (reaproveitados do Chroma)")

    t0 = time.time()
    reduzidos = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine",
                     random_state=42).fit_transform(emb)
    print(f"UMAP: {time.time()-t0:.0f}s")

    stop, _ = _recursos_nltk()
    vetorizador = CountVectorizer(stop_words=sorted(stop - NEGACOES), ngram_range=(1, 2), min_df=1)
    analisar = CountVectorizer(stop_words=sorted(stop - NEGACOES)).build_analyzer()
    tokens = [analisar(t) for t in textos]
    dic = Dictionary(tokens)

    print(f"\n{'metodo':>6} {'min_cluster':>11} {'topicos':>8} {'outliers':>9} {'c_v':>6} {'maior topico':>13}")
    for metodo, mcs in [("eom", 50), ("eom", 100), ("eom", 200), ("leaf", 50), ("leaf", 100), ("leaf", 200)]:
        t0 = time.time()
        m = BERTopic(umap_model=BaseDimensionalityReduction(),
                     hdbscan_model=HDBSCAN(min_cluster_size=mcs, metric="euclidean", core_dist_n_jobs=1,
                                           cluster_selection_method=metodo, prediction_data=True),
                     vectorizer_model=vetorizador, language="multilingual", calculate_probabilities=False)
        topicos, _ = m.fit_transform(textos, reduzidos)
        ids = [t for t in set(topicos) if t != -1]
        palavras = [[w for w, _ in m.get_topic(t)[:10] if " " not in w] for t in ids]
        palavras = [p for p in palavras if len(p) >= 3]
        cv = CoherenceModel(topics=palavras, texts=tokens, dictionary=dic, coherence="c_v", processes=1).get_coherence()
        outl = np.mean(np.array(topicos) == -1)
        cont = pd.Series([t for t in topicos if t != -1]).value_counts()
        maior = cont.iloc[0] / len(topicos) if len(cont) else 0
        print(f"{metodo:>6} {mcs:>11} {len(ids):>8} {outl:>8.1%} {cv:>6.3f} {maior:>12.1%}", flush=True)


if __name__ == "__main__":
    main()
