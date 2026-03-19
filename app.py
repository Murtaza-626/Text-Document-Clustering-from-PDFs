import streamlit as st
import numpy as np
from processor import extract_text_from_pdf, clean_and_lemmatize
from clustering import perform_k_means, reduce_dimensions, vectorize_documents
import plotly.express as px
import pandas as pd
from visualization import create_cluster_chart, df_results_table


# Page configuration
st.set_page_config(page_title="DocCluster AI", page_icon="📄", layout="wide")

# Initialize session state
if 'documents' not in st.session_state:
    st.session_state['documents'] = {}
if 'embeddings' not in st.session_state:
    st.session_state['embeddings'] = None
if 'doc_names' not in st.session_state:
    st.session_state['doc_names'] = []

st.title("Text Document Clustering from PDFs")
st.markdown("Automated organization of documents using Data Mining techniques.")


# STEP 1: PDF UPLOAD & EXTRACTION

st.header("Data Ingestion & Extraction")
uploaded_files = st.file_uploader(
    "Upload PDF documents for analysis", 
    type="pdf", 
    accept_multiple_files=True
)

if uploaded_files:
    for file in uploaded_files:
        if file.name not in st.session_state['documents']:
            with st.spinner(f"Extracting text from {file.name}..."):
                text, pages = extract_text_from_pdf(file)
                if text:
                    st.session_state['documents'][file.name] = {
                        "raw_text": text,
                        "cleaned_text": None,
                        "pages": pages
                    }
                    if file.name not in st.session_state['doc_names']:
                        st.session_state['doc_names'].append(file.name)
    
    st.success(f"Successfully extracted {len(st.session_state['documents'])} files.")
    
    # Remove files that are no longer in the uploaded list
    currently_uploaded_names = [file.name for file in uploaded_files]
    removed_files = [name for name in st.session_state['doc_names'] if name not in currently_uploaded_names]
    
    if removed_files:
        for removed_file in removed_files:
            # Remove from documents
            if removed_file in st.session_state['documents']:
                del st.session_state['documents'][removed_file]
            # Remove from doc_names
            if removed_file in st.session_state['doc_names']:
                st.session_state['doc_names'].remove(removed_file)
        
        # Regenerate embeddings to match remaining documents
        remaining_cleaned_texts = [
            st.session_state['documents'][name]["cleaned_text"]
            for name in st.session_state['doc_names']
            if st.session_state['documents'][name].get("cleaned_text") is not None
        ]
        
        if remaining_cleaned_texts and len(remaining_cleaned_texts) == len(st.session_state['doc_names']):
            # All remaining documents are cleaned, regenerate embeddings
            st.session_state['embeddings'] = vectorize_documents(remaining_cleaned_texts)
        else:
            # Not all remaining documents are cleaned, clear embeddings
            st.session_state['embeddings'] = None
        
        st.info(f"Removed {len(removed_files)} file(s) and updated data accordingly.")
else:
    # No files uploaded, clear everything
    if st.session_state['documents']:
        st.session_state['documents'] = {}
        st.session_state['doc_names'] = []
        st.session_state['embeddings'] = None
        st.info("All files removed.")


# Text Preprocessing:
st.header("Text Preprocessing")
if st.session_state['documents']:
    if st.button("Clean and Preprocess Text"):
        with st.spinner("Cleaning text and applying lemmatization..."):
            for filename, data in st.session_state['documents'].items():
                if data["cleaned_text"] is None:
                    cleaned = clean_and_lemmatize(data["raw_text"])
                    st.session_state['documents'][filename]["cleaned_text"] = cleaned
        st.success("Preprocessing Complete!")


    first_doc = list(st.session_state['documents'].values())[0]
    
    if first_doc.get("cleaned_text"):
        st.markdown("### 🔍 Preprocessing Comparison")
        
        with st.expander("View Cleaning Results"):
            with st.container(height=500, border=True):
                for sample_name in st.session_state['doc_names']:
                    doc_data = st.session_state['documents'][sample_name]
                    
                    if doc_data['cleaned_text'] is not None:
                        st.markdown(f"**Document:** {sample_name}")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.caption("Raw Text (Snippet)")
                            # Using height in text_area makes the individual snippets compact
                            st.info(st.session_state['documents'][sample_name]['raw_text'][:1000] + "...")
                        
                        with col2:
                            st.caption("Cleaned & Lemmatized")
                            st.success(st.session_state['documents'][sample_name]['cleaned_text'][:1000] + "...")
                        
                        st.divider()
                    else:
                        st.info("Click 'Clean and Preprocess Text' to see results.")


# STEP 3: TEXT VECTORIZATION

st.header("Text Vectorization")
st.markdown("""
Here we convert the cleaned text into **Dense Semantic Embeddings** using the `all-MiniLM-L6-v2` neural network. 
This maps each document into a 384-dimensional mathematical space based on its meaning.
""")

# Check if documents exist and have been cleaned
is_cleaned = bool(st.session_state['documents'] and list(st.session_state['documents'].values())[0].get("cleaned_text"))

if is_cleaned:
    if st.button("Generate Document Embeddings"):
        with st.spinner("Loading model and generating semantic vectors. This may take a moment..."):
            # Gather all cleaned texts in the exact order of doc_names
            texts_to_vectorize = [
                st.session_state['documents'][name]["cleaned_text"] 
                for name in st.session_state['doc_names']
            ]
            
            # Generate the embeddings
            embeddings = vectorize_documents(texts_to_vectorize)
            
            # Store in session state for the clustering step
            st.session_state['embeddings'] = embeddings
            
        st.success("Vectorization Complete!")
        
    # Display embedding stats if they exist
    if st.session_state['embeddings'] is not None:
        emb_shape = st.session_state['embeddings'].shape
        st.write(f"**Total Vectors Generated:** {emb_shape[0]} (One per document)")
        st.write(f"**Vector Dimensions:** {emb_shape[1]} (Features per document)")
        
        with st.expander("Following are the raw numerical vectors"):
            st.dataframe(st.session_state['embeddings'])
else:
    st.info("Please complete Step 2 (Preprocessing) before generating embeddings.")


# --- STEP 4 & 5: CLUSTERING & VISUALIZATION ---
st.header("Step 4 & 5: Clustering & Visualizing Results")

if st.session_state['embeddings'] is not None:
    num_docs = len(st.session_state['doc_names'])
    
    # Clustering is only meaningful with 3 or more documents
    if num_docs < 3:
        st.warning("Please upload at least 3 documents to perform clustering.")
    else:
        # Maximum clusters is equal to number of documents or 10, whichever is smaller
        max_k = min(10, num_docs)
        
        k_value = st.slider("Select Number of Topics (k)", 2, max_k, 2)
        
        if st.button("Run Analysis"):
            with st.spinner("Processing..."):
                
                # 1. Logic (clustering.py)
                labels, centers, score = perform_k_means(st.session_state['embeddings'], k_value)
                coords_2d = reduce_dimensions(st.session_state['embeddings'])
                
                # 2. Visualisation (visualisation.py)
                fig, table_df = create_cluster_chart(
                    st.session_state['doc_names'], 
                    labels, 
                    coords_2d, 
                    score
                )
                
                # 3. Store in session state
                st.session_state['viz_fig'] = fig
                st.session_state['viz_table'] = table_df
                st.session_state['score'] = score

        if 'viz_fig' in st.session_state:
            st.metric("Silhouette Score", f"{st.session_state['score']:.3f}")
            st.plotly_chart(st.session_state['viz_fig'], use_container_width=True)
            st.dataframe(st.session_state['viz_table'], use_container_width=True, hide_index=True)
            
            if st.button("Generate Cluster Summaries"):
            # Statistical Summaries
                st.header("Cluster Statistical Summaries")
                
                # Get labels from the table dataframe
                cluster_labels = st.session_state['viz_table']['Cluster'].values
                doc_names = st.session_state['viz_table']['Document'].values
                
                # Calculate cluster statistics
                unique_clusters = sorted(set(cluster_labels))
                cluster_stats = []
                
                for cluster_id in unique_clusters:
                    docs_in_cluster = [doc_names[i] for i, label in enumerate(cluster_labels) if label == cluster_id]
                    cluster_size = len(docs_in_cluster)
                    percentage = (cluster_size / len(doc_names)) * 100
                    
                    cluster_stats.append({
                        'Cluster': f"Cluster {cluster_id}",
                        'Document Count': cluster_size,
                        'Percentage': f"{percentage:.1f}%",
                        'Documents': ', '.join(docs_in_cluster)
                    })
                
                # Displaying cluster statistics
                st.subheader("📊 Cluster Size Distribution")
                cluster_summary_df = pd.DataFrame(cluster_stats)
                st.dataframe(cluster_summary_df, use_container_width=True, hide_index=True)
                
                # Displaying cluster composition
                st.subheader("📋 Detailed Cluster Composition")
                for cluster_id in unique_clusters:
                    docs_in_cluster = [doc_names[i] for i, label in enumerate(cluster_labels) if label == cluster_id]
                    cluster_size = len(docs_in_cluster)
                    percentage = (cluster_size / len(doc_names)) * 100
                    
                    with st.expander(f"Cluster {cluster_id} ({cluster_size} documents, {percentage:.1f}%)"):
                        for doc in docs_in_cluster:
                            st.write(f"• {doc}")
                
                # Display overall statistics
                st.subheader("📈 Overall Clustering Statistics")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Documents", len(doc_names))
                
                with col2:
                    st.metric("Number of Clusters", len(unique_clusters))
                
                with col3:
                    avg_cluster_size = len(doc_names) / len(unique_clusters)
                    st.metric("Avg Documents per Cluster", f"{avg_cluster_size:.1f}")
                
                # Display distribution bar chart
                st.subheader("📊 Cluster Distribution Chart")
                cluster_counts = [len([label for label in cluster_labels if label == cid]) for cid in unique_clusters]
                distribution_df = pd.DataFrame({
                    'Cluster': [f'Cluster {i}' for i in unique_clusters],
                    'Document Count': cluster_counts
                })
                
                dist_chart = px.bar(
                    distribution_df,
                    x='Cluster',
                    y='Document Count',
                    title='Documents per Cluster',
                    color='Cluster',
                    text='Document Count'
                )
                dist_chart.update_layout(height=400, showlegend=False)
                st.plotly_chart(dist_chart, use_container_width=True)