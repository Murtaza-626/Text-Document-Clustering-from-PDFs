from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfVectorizer


def vectorize_documents(texts):
    if not texts:
        return np.array([])
    
    # model = SentenceTransformer('all-mpnet-base-v2')
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    embeddings = []
    for text in texts:
        words = text.split()
        
        # Use larger chunks with overlap for better context retention
        chunk_size = 400
        overlap = 50
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            if chunk:
                chunks.append(chunk)
        
        if not chunks:
            chunks = [text]
        
        chunk_embeddings = model.encode(chunks, show_progress_bar=False)
        
        # Weighted average: give more weight to first and last chunks
        # (intros and conclusions are most topic-representative)
        weights = np.ones(len(chunk_embeddings))
        if len(chunk_embeddings) > 2:
            weights[0] = 2.0   # intro
            weights[-1] = 1.5  # conclusion
        weights = weights / weights.sum()
        
        doc_embedding = np.average(chunk_embeddings, axis=0, weights=weights)
        embeddings.append(doc_embedding)
    
    embeddings_np = np.array(embeddings)
    embeddings_np = normalize(embeddings_np, norm='l2')
    return embeddings_np


def find_best_k(embeddings, k_min=2, k_max=None):
    """
    Tests multiple k values and returns silhouette scores for each.
    Helps user pick the best k instead of guessing.
    """
    n = len(embeddings)
    if k_max is None:
        k_max = min(10, n - 1)
    
    results = {}
    for k in range(k_min, k_max + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=15)
        labels = kmeans.fit_predict(embeddings)
        score = silhouette_score(embeddings, labels)
        results[k] = round(score, 4)
    
    best_k = max(results, key=results.get)
    return results, best_k


def perform_k_means(embeddings, k):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=15)
    labels = kmeans.fit_predict(embeddings)
    score = silhouette_score(embeddings, labels) if k > 1 else 0
    return labels, kmeans.cluster_centers_, score


def reduce_dimensions(embeddings):
    reducer = PCA(n_components=2, random_state=42)
    coords_2d = reducer.fit_transform(embeddings)
    return coords_2d, reducer


def find_outliers(embeddings, labels, centers, threshold=2.0):
    distances = []
    for i, emb in enumerate(embeddings):
        center = centers[labels[i]]
        dist = np.linalg.norm(emb - center)
        distances.append(dist)
    avg = np.mean(distances)
    std = np.std(distances)
    outliers = [i for i, d in enumerate(distances) if d > avg + (threshold * std)]
    return outliers


def describe_centroids(embeddings, labels, centers, doc_names, cleaned_texts, top_n=8):
    """
    Elaborates each cluster centroid with its most representative keywords.

    Strategy:
      1. Group each document's cleaned text by cluster.
      2. Run TF-IDF on the grouped texts to surface the most distinctive
         terms for each cluster.
      3. Re-rank those TF-IDF candidates by their cosine similarity to the
         centroid vector in the embedding space, so the final keywords are
         grounded in the mathematical centre of the cluster.

    Returns
    -------
    dict  {cluster_id: {"documents": [...], "keywords": [...]}}
    """
    model = SentenceTransformer('all-MiniLM-L6-v2')
    unique_clusters = sorted(set(labels))
    results = {}

    for cluster_id in unique_clusters:
        # Collect indices of documents in this cluster
        member_indices = [i for i, lbl in enumerate(labels) if lbl == cluster_id]
        member_names   = [doc_names[i] for i in member_indices]
        member_texts   = [cleaned_texts[i] for i in member_indices]

        # --- TF-IDF: find top candidate terms for the cluster ---
        try:
            tfidf = TfidfVectorizer(
                max_features=100,
                stop_words='english',
                ngram_range=(1, 2),     # include bigrams for richer phrases
                min_df=1
            )
            tfidf_matrix = tfidf.fit_transform(member_texts)
            feature_names = tfidf.get_feature_names_out()

            # Sum TF-IDF weights across all documents in the cluster
            tfidf_scores = np.asarray(tfidf_matrix.sum(axis=0)).flatten()
            # Keep top 30 candidates for re-ranking
            top_candidate_idx = tfidf_scores.argsort()[::-1][:30]
            candidate_terms = [feature_names[i] for i in top_candidate_idx]
        except ValueError:
            # Happens when a cluster has a single very short document
            candidate_terms = []

        # --- Re-rank by cosine similarity to the centroid ---
        centroid = centers[cluster_id]                         # shape (embedding_dim,)
        centroid_norm = centroid / (np.linalg.norm(centroid) + 1e-9)

        if candidate_terms:
            term_embeddings = model.encode(candidate_terms, show_progress_bar=False)
            term_embeddings = normalize(term_embeddings, norm='l2')
            # Cosine similarity = dot product with normalised centroid
            similarities = term_embeddings @ centroid_norm
            ranked_idx = similarities.argsort()[::-1]
            top_keywords = [candidate_terms[i] for i in ranked_idx[:top_n]]
        else:
            top_keywords = []

        results[cluster_id] = {
            "documents": member_names,
            "keywords":  top_keywords,
        }

    return results
