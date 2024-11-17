import plotly.express as px
import pandas as pd

def create_scatter_plot(data, labels, title, label_name="Label"):
    df = pd.DataFrame({
        'x': data[:, 0],
        'y': data[:, 1],
        'label': labels
    })
    fig = px.scatter(
        df, x='x', y='y', color=df['label'].astype(str),
        title=title, labels={'color': label_name},
        hover_data=['label']
    )
    return fig
