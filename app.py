import streamlit as st
import numpy as np
from processor import extract_text_from_pdf, clean_and_lemmatize
from clustering import perform_k_means, reduce_dimensions, find_best_k, find_outliers, vectorize_documents
import plotly.express as px
import pandas as pd
from visualization import create_cluster_chart, df_results_table, create_silhouette_chart

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

st.header("Step 2: Text Preprocessing")
if st.session_state['documents']:
    if st.button("Clean and Preprocess Text"):
        with st.spinner("Cleaning text and applying lemmatization..."):
            for filename, data in st.session_state['documents'].items():
                if data["cleaned_text"] is None:
                    cleaned = clean_and_lemmatize(data["raw_text"])
                    st.session_state['documents'][filename]["cleaned_text"] = cleaned
        st.success("Preprocessing Complete!")

    # Show previews only if cleaning has been done
    first_doc = list(st.session_state['documents'].values())[0]
    if first_doc.get("cleaned_text"):
        with st.expander("🔍 View Cleaning Results"):
            with st.container(height=500, border=True):           
                for sample_name in st.session_state['doc_names']:
                    if st.session_state['documents'][sample_name]['cleaned_text'] is not None:
                        st.markdown(f"**Sample from:** {sample_name}")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("**Raw Text**")
                            st.info(st.session_state['documents'][sample_name]['raw_text'][:400] + "...")
                        with col2:
                            st.markdown("**Cleaned Text**")
                            st.success(st.session_state['documents'][sample_name]['cleaned_text'][:400] + "...")
                        st.divider()
else:
    st.info("Upload documents in Step 1 to proceed.")


# STEP 3: TEXT VECTORIZATION

st.header("Step 3: Text Vectorization")
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


st.header("Step 4 & 5: Clustering & Visualizing Results")

if st.session_state['embeddings'] is not None:
    num_docs = len(st.session_state['doc_names'])
    
    if num_docs < 3:
        st.warning("Please upload at least 3 documents to perform clustering.")
    else:
        max_k = min(10, num_docs - 1)

        # Auto K Suggester 
        if st.button("🔍 Find Best k (Recommended)"):
            with st.spinner("Testing k values from 2 to 10..."):
                scores_dict, best_k = find_best_k(st.session_state['embeddings'], k_max=max_k)
                st.session_state['k_scores'] = scores_dict
                st.session_state['best_k'] = best_k

        if 'k_scores' in st.session_state:
            best_k = st.session_state['best_k']
            st.success(f"✅ Recommended k = **{best_k}** (highest silhouette score: {st.session_state['k_scores'][best_k]})")
            fig_k = create_silhouette_chart(st.session_state['k_scores'], best_k)
            st.plotly_chart(fig_k, use_container_width=True)

        k_value = st.slider(
            "Select Number of Topics (k)",
            2, max_k,
            st.session_state.get('best_k', 2)  # defaults to suggested k if available
        )
        
        if st.button("Run Analysis"):
            with st.spinner("Processing..."):
                labels, centers, score = perform_k_means(st.session_state['embeddings'], k_value)
                coords_2d = reduce_dimensions(st.session_state['embeddings'])
                outlier_indices = find_outliers(st.session_state['embeddings'], labels, centers)
                fig, table_df = create_cluster_chart(
                    st.session_state['doc_names'], labels, coords_2d, score,
                    outlier_indices=outlier_indices
                )
                st.session_state['viz_fig'] = fig
                st.session_state['viz_table'] = table_df
                st.session_state['score'] = score
                st.session_state['outlier_indices'] = outlier_indices


        # Display results if they exist
        if 'viz_fig' in st.session_state:
            st.metric("Silhouette Score", f"{st.session_state['score']:.3f}")
            st.plotly_chart(st.session_state['viz_fig'], use_container_width=True)
            st.dataframe(st.session_state['viz_table'], use_container_width=True, hide_index=True)

            # --- OUTLIER SUMMARY ---
            outlier_indices = st.session_state.get('outlier_indices', [])
            if outlier_indices:
                outlier_names = [st.session_state['doc_names'][i] for i in outlier_indices]
                st.warning(f"⚠ **{len(outlier_names)} outlier(s) detected** (shown as red ✕ on the chart):")
                for name in outlier_names:
                    st.write(f"  • {name}")
            else:
                st.success("No outliers detected — all documents fit well within their clusters.")
            
            # --- STATISTICAL SUMMARIES ---
            if st.button("Generate Statistical Summaries"):
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
                
                # Display cluster statistics
                st.subheader("📊 Cluster Size Distribution")
                cluster_summary_df = pd.DataFrame(cluster_stats)
                st.dataframe(cluster_summary_df, use_container_width=True, hide_index=True)
                
                # Display cluster composition
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
