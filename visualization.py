import plotly.express as px
import pandas as pd

def create_cluster_chart(doc_names, labels, coords_2d, silhouette_score):
    """
    Generates an interactive Plotly scatter plot.
    - Each dot is a document 
    - Colors represent clusters [cite: 28]
    """
    # Prepare the data for Plotly
    df_viz = pd.DataFrame({
        "Document": doc_names,
        "Cluster": [f"Topic {label}" for label in labels],
        "X": coords_2d[:, 0],
        "Y": coords_2d[:, 1]
    })
    
    # Create the scatter plot
    fig = px.scatter(
        df_viz, 
        x="X", y="Y", 
        color="Cluster", 
        hover_data=["Document"],
        title=f"2D Document Projection (Silhouette Score: {silhouette_score:.3f})",
        color_discrete_sequence=px.colors.qualitative.Bold,
        template="plotly_white"
    )
    
    # Customize the look: larger dots with borders [cite: 29]
    fig.update_traces(marker=dict(size=12, line=dict(width=1, color='DarkSlateGrey')))
    
    return fig, df_results_table(df_viz)

def df_results_table(df):
    """Returns a simplified version of the dataframe for the UI table."""
    return df[['Document', 'Cluster']].sort_values(by='Cluster')