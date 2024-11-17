from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN

def apply_kmeans(data, n_clusters=10, random_state=42):
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    return kmeans.fit_predict(data)

def apply_dbscan(data, eps=0.5, min_samples=5):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    return dbscan.fit_predict(data)
