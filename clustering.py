from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize


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
    return reducer.fit_transform(embeddings)


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
