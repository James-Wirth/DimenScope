from dash import Dash, dcc, html
import dash_bootstrap_components as dbc

def create_dashboard(umap_fig, cluster_fig):
    app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

    app.layout = dbc.Container([
        dbc.Tabs([
            dbc.Tab(dcc.Graph(figure=umap_fig), label="UMAP"),
            dbc.Tab(dcc.Graph(figure=cluster_fig), label="Clustering")
        ])
    ])

    return app
