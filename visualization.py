import plotly.express as px
import plotly.graph_objects as go
import pandas as pd


def create_silhouette_chart(scores_dict, best_k):
    """
    Builds a line chart of Silhouette Score vs Number of Clusters.
    Adds a dashed vertical line at the recommended best_k.
    Returns the Plotly figure.
    """
    scores_df = pd.DataFrame({
        'k (Number of Clusters)': list(scores_dict.keys()),
        'Silhouette Score': list(scores_dict.values())
    })
    fig = px.line(
        scores_df,
        x='k (Number of Clusters)',
        y='Silhouette Score',
        markers=True,
        title='Silhouette Score vs Number of Clusters'
    )
    fig.add_vline(
        x=best_k,
        line_dash="dash",
        line_color="green",
        annotation_text=f"Best k={best_k}"
    )
    return fig

def create_cluster_chart(doc_names, labels, coords_2d, silhouette_score, outlier_indices=None, centers_2d=None):
    """
    Generates an interactive Plotly scatter plot.
    - Each dot is a document
    - Colors represent clusters
    - Outliers are highlighted with a red X marker
    """
    if outlier_indices is None:
        outlier_indices = []

    outlier_set = set(outlier_indices)

    # Prepare the data for Plotly
    df_viz = pd.DataFrame({
        "Document": doc_names,
        "Cluster": [f"Topic {label}" for label in labels],
        "X": coords_2d[:, 0],
        "Y": coords_2d[:, 1],
        "Is_Outlier": [i in outlier_set for i in range(len(doc_names))]
    })

    df_normal = df_viz[~df_viz["Is_Outlier"]]
    df_outliers = df_viz[df_viz["Is_Outlier"]]

    # Plot normal points coloured by cluster
    fig = px.scatter(
        df_normal,
        x="X", y="Y",
        color="Cluster",
        hover_data=["Document"],
        title=f"2D Document Projection (Silhouette Score: {silhouette_score:.3f})",
        color_discrete_sequence=px.colors.qualitative.Bold,
        template="plotly_white"
    )
    fig.update_traces(marker=dict(size=12, line=dict(width=1, color='DarkSlateGrey')))

    # Overlay outliers as red X markers
    if not df_outliers.empty:
        fig.add_trace(go.Scatter(
            x=df_outliers["X"],
            y=df_outliers["Y"],
            mode="markers",
            name="⚠ Outlier",
            hovertext=df_outliers["Document"],
            hoverinfo="text+name",
            marker=dict(
                symbol="x",
                size=16,
                color="red",
                line=dict(width=2, color="darkred")
            )
        ))

    # Overlay centroids as gold stars
    if centers_2d is not None and len(centers_2d) > 0:
        for i, (cx, cy) in enumerate(centers_2d):
            fig.add_trace(go.Scatter(
                x=[cx],
                y=[cy],
                mode="markers+text",
                name=f"★ Centroid {i}",
                text=[f"C{i}"],
                textposition="top center",
                hovertemplate=f"<b>Centroid {i}</b><br>x: {cx:.3f}<br>y: {cy:.3f}<extra></extra>",
                marker=dict(
                    symbol="star",
                    size=20,
                    color="gold",
                    line=dict(width=1.5, color="darkorange")
                ),
                showlegend=True
            ))

    return fig, df_results_table(df_viz)

def df_results_table(df):
    """Returns a simplified version of the dataframe for the UI table."""
    result = df[['Document', 'Cluster']].copy()
    if 'Is_Outlier' in df.columns:
        result = df[['Document', 'Cluster', 'Is_Outlier']].copy()
        result['Is_Outlier'] = result['Is_Outlier'].map({True: '⚠ Yes', False: 'No'})
        result = result.rename(columns={'Is_Outlier': 'Outlier'})
    return result.sort_values(by='Cluster')
