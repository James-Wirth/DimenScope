from visualizations.plotly_visuals import create_scatter_plot
from models.dimensionality import apply_umap
from models.clustering import apply_kmeans
from data.dataset_loader import load_mnist_data
from app.dashboard import create_dashboard

def main():
    x_train, y_train = load_mnist_data()
    reduced_data = apply_umap(x_train[:1000])  # Use a subset for speed
    cluster_labels = apply_kmeans(reduced_data, n_clusters=10)

    umap_fig = create_scatter_plot(reduced_data, y_train[:1000], title="UMAP Projection of MNIST", label_name="Digit")
    cluster_fig = create_scatter_plot(reduced_data, cluster_labels, title="Clustering After UMAP", label_name="Cluster")

    app = create_dashboard(umap_fig, cluster_fig)
    app.run_server(debug=True)

if __name__ == '__main__':
    main()
