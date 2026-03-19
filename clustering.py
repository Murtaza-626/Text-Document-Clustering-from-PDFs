from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


# def vectorize_documents(texts):
#     if not texts:
#         return np.array([])
#     # 'all-MiniLM-L6-v2' captures actual meaning, not just words 
#     model = SentenceTransformer('all-MiniLM-L6-v2')
#     return model.encode(texts, show_progress_bar=False)

def vectorize_documents(texts):
    if not texts:
        return np.array([])
    # model = SentenceTransformer('all-MiniLM-L6-v2')
    model = SentenceTransformer('all-mpnet-base-v2')
    
    embeddings = []
    for text in texts:
        # Chunk the text into ~500-word segments
        words = text.split()
        chunks = [' '.join(words[i:i+300]) for i in range(0, len(words), 300)]
        
        if not chunks:
            chunks = [text]
        
        # Encode all chunks and average the embeddings
        chunk_embeddings = model.encode(chunks, show_progress_bar=False)
        doc_embedding = np.mean(chunk_embeddings, axis=0)
        embeddings.append(doc_embedding)
    
    # Normalize the embeddings (important for cosine similarity)
    embeddings_np = np.array(embeddings)
    embeddings_np = normalize(embeddings_np, norm='l2')
    
    return embeddings_np


def perform_k_means(embeddings, k):
    """
    Groups documents into k topics.
    Returns labels, cluster centers, and quality score.
    """
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)
    
    # Measuring Clustering Quality 
    score = silhouette_score(embeddings, labels) if k > 1 else 0
    
    return labels, kmeans.cluster_centers_, score

def reduce_dimensions(embeddings):
    """
    Converts high-dimensional vectors to 2D for the scatter chart.
    """
    reducer = PCA(n_components=2, random_state=42)
    return reducer.fit_transform(embeddings)



def get_cluster_keywords(processed_texts, labels, n_words=5):
    """Generates dominant keywords for each cluster to describe the topic."""
    df = pd.DataFrame({'text': processed_texts, 'label': labels})
    summaries = {}
    for i in sorted(df['label'].unique()):
        category_text = df[df['label'] == i]['text']
        vectorizer = TfidfVectorizer(stop_words='english', max_features=n_words)
        vectorizer.fit(category_text)
        summaries[i] = list(vectorizer.get_feature_names_out())
    return summaries

def find_outliers(embeddings, labels, centers, threshold=2.0):
    """Flags documents that are unusually far from their cluster center."""
    distances = []
    for i, emb in enumerate(embeddings):
        center = centers[labels[i]]
        dist = np.linalg.norm(emb - center)
        distances.append(dist)
    
    # Simple statistical outlier detection (Z-score logic)
    avg = np.mean(distances)
    std = np.std(distances)
    outliers = [i for i, d in enumerate(distances) if d > avg + (threshold * std)]
    return outliers