"""Quantos outliers voltam a um tópico em cada limiar de similaridade."""
import numpy as np, pandas as pd, logging
logging.basicConfig(level=logging.WARNING)


def main():
    from umap import UMAP
    from hdbscan import HDBSCAN
    from bertopic import BERTopic
    from sklearn.feature_extraction.text import CountVectorizer
    from src.vectorstore import carregar
    from src.preprocessing import _recursos_nltk, NEGACOES
    d = carregar()._collection.get(include=["documents", "embeddings", "metadatas"])
    textos, emb = d["documents"], np.array(d["embeddings"], dtype="float32")
    stop, _ = _recursos_nltk()
    m = BERTopic(umap_model=UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine", random_state=42),
                 hdbscan_model=HDBSCAN(min_cluster_size=50, metric="euclidean", cluster_selection_method="eom",
                                       prediction_data=True, core_dist_n_jobs=1),
                 vectorizer_model=CountVectorizer(stop_words=sorted(stop - NEGACOES), ngram_range=(1, 2), min_df=1),
                 language="multilingual")
    topicos, _ = m.fit_transform(textos, emb)
    topicos = np.array(topicos)
    print(f"tópicos: {len(set(topicos)) - 1} | sem tópico: {np.mean(topicos == -1):.1%}")
    # similaridade de cada outlier com o centróide do tópico mais próximo
    te = m.topic_embeddings_[1:]                      # linha 0 é o tópico -1
    te = te / np.linalg.norm(te, axis=1, keepdims=True)
    out = np.where(topicos == -1)[0]
    sims = (emb[out] / np.linalg.norm(emb[out], axis=1, keepdims=True)) @ te.T
    melhor = sims.max(axis=1)
    print(f"\nsimilaridade do outlier com o tópico mais próximo: "
          f"p25={np.percentile(melhor,25):.2f} mediana={np.median(melhor):.2f} p75={np.percentile(melhor,75):.2f}")
    print(f"\n{'limiar':>7} {'reatribuídos':>13} {'sem tópico final':>17}")
    for lim in (0.5, 0.6, 0.65, 0.7, 0.75, 0.8):
        n = int((melhor >= lim).sum())
        print(f"{lim:>7.2f} {n:>13,} {(len(out)-n)/len(topicos):>16.1%}")
    # inspeção: exemplos reatribuídos perto do limiar 0.7
    print("\nexemplos com similaridade entre 0.68 e 0.72 (fronteira do limiar 0,7):")
    rotulos = {t: "_".join(w for w, _ in m.get_topic(t)[:4]) for t in set(topicos) if t != -1}
    idx = np.where((melhor >= 0.68) & (melhor <= 0.72))[0][:6]
    for i in idx:
        t = int(np.argmax(sims[i]))
        print(f"  {melhor[i]:.2f} -> [{rotulos[t]}]  \"{textos[out[i]][:75]}\"")


if __name__ == "__main__":
    main()
