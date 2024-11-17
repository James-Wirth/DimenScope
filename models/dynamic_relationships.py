from sklearn.neighbors import NearestNeighbors

def compute_connections(data, n_neighbors=5):
    nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(data)
    connections = nbrs.kneighbors_graph(data).toarray()
    return connections
