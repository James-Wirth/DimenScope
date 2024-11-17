import umap
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings("ignore", message=".*n_jobs value 1 overridden*")

def apply_umap(data, n_neighbors=15, min_dist=0.1, n_components=2, random_state=42):
    reducer = umap.UMAP(
        n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components, random_state=random_state
    )
    return reducer.fit_transform(data)

def apply_tsne(data, n_components=2, perplexity=30, random_state=42):
    tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=random_state)
    return tsne.fit_transform(data)
