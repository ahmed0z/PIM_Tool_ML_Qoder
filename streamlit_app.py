#!/usr/bin/env python3
"""
AutoPatternChecker Streamlit App
A user-friendly web interface for pattern learning and validation.
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import zipfile
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import AutoPatternChecker modules
from autopatternchecker import (
    DataIngester, PatternProfiler, NormalizationEngine,
    ClusterAnalyzer, EmbeddingGenerator, FAISSIndexer,
    ActiveLearningManager, create_app, ModelTrainer
)
from autopatternchecker.utils import setup_logging

# Page configuration
st.set_page_config(
    page_title="AutoPatternChecker",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configure for large file uploads (set via command line: --server.maxUploadSize=200)
# st.set_option('server.maxUploadSize', 200)  # 200MB - Cannot be set at runtime

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
        padding: 1rem;
        border-radius: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables."""
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = None
    if 'key_profiles' not in st.session_state:
        st.session_state.key_profiles = None
    if 'normalization_rules' not in st.session_state:
        st.session_state.normalization_rules = None
    if 'clustering_results' not in st.session_state:
        st.session_state.clustering_results = None
    if 'embeddings_data' not in st.session_state:
        st.session_state.embeddings_data = None
    if 'trained_models' not in st.session_state:
        st.session_state.trained_models = None
    if 'config' not in st.session_state:
        st.session_state.config = get_default_config()

def get_default_config():
    """Get default configuration."""
    return {
        'key_columns': ['key_part1', 'key_part2', 'key_part3'],
        'value_column': 'value',
        'chunk_size': 10000,
        'encoding': 'utf-8',
        'min_signature_frequency': 5,
        'rare_signature_pct_threshold': 1.0,
        'tfidf_ngram_range': [2, 4],
        'tfidf_max_features': 2000,
        'hdbscan_min_cluster_size': 5,
        'hdbscan_min_samples': 2,
        'kmeans_n_clusters': 8,
        'embedding_model': 'all-MiniLM-L6-v2',
        'embedding_batch_size': 64,
        'use_embeddings': True,
        'index_type': 'flat',
        'metric': 'cosine',
        'review_batch_size': 100,
        'auto_accept_confidence_threshold': 0.95,
        'manual_review_threshold_lower': 0.6,
        'manual_review_threshold_upper': 0.95,
        'retrain_threshold': 10
    }

def main():
    """Main Streamlit app."""
    
    # Initialize session state
    initialize_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">🔍 AutoPatternChecker</h1>', unsafe_allow_html=True)
    st.markdown("**Automated pattern learning and validation system**")
    
    # Sidebar
    with st.sidebar:
        st.header("📋 Navigation")
        page = st.selectbox(
            "Choose a page:",
            ["🏠 Home", "📊 Data Upload", "🔍 Pattern Analysis", "🤖 Model Training", 
             "✅ Validation", "📈 Analytics", "⚙️ Settings"]
        )
        
        st.header("ℹ️ About")
        st.info("""
        AutoPatternChecker learns patterns from your CSV data and validates new values automatically.
        
        **Features:**
        - Pattern learning
        - Clustering analysis
        - Semantic similarity
        - Model training
        - Real-time validation
        """)
    
    # Main content based on selected page
    if page == "🏠 Home":
        show_home_page()
    elif page == "📊 Data Upload":
        show_data_upload_page()
    elif page == "🔍 Pattern Analysis":
        show_pattern_analysis_page()
    elif page == "🤖 Model Training":
        show_model_training_page()
    elif page == "✅ Validation":
        show_validation_page()
    elif page == "📈 Analytics":
        show_analytics_page()
    elif page == "⚙️ Settings":
        show_settings_page()

def show_home_page():
    """Show home page."""
    st.header("Welcome to AutoPatternChecker! 🎉")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🚀 Quick Start")
        st.markdown("""
        1. **Upload your CSV data** in the Data Upload page
        2. **Analyze patterns** to understand your data
        3. **Train models** for better accuracy
        4. **Validate new values** in real-time
        """)
        
        if st.button("📊 Go to Data Upload", type="primary"):
            st.session_state.page = "📊 Data Upload"
            st.rerun()
    
    with col2:
        st.subheader("📊 Current Status")
        if st.session_state.processed_data is not None:
            st.success("✅ Data loaded and processed")
            st.metric("Total Rows", len(st.session_state.processed_data))
            st.metric("Unique Keys", len(st.session_state.key_profiles) if st.session_state.key_profiles else 0)
        else:
            st.warning("⚠️ No data loaded yet")
            st.info("Upload your CSV data to get started")
    
    # Show recent activity
    st.subheader("📈 Recent Activity")
    if st.session_state.processed_data is not None:
        st.info("✅ Data processing completed")
        if st.session_state.key_profiles:
            st.info("✅ Pattern analysis completed")
        if st.session_state.trained_models:
            st.info("✅ Model training completed")
    else:
        st.info("No recent activity. Upload data to get started!")

def show_data_upload_page():
    """Show data upload page."""
    st.header("📊 Data Upload")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type="csv",
        help="Upload your CSV file with key columns and value column"
    )
    
    if uploaded_file is not None:
        try:
            # Check file size
            file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
            st.info(f"📁 File size: {file_size_mb:.1f} MB")
            
            # For large files, show progress
            if file_size_mb > 50:
                st.warning("⚠️ Large file detected. Processing may take a few minutes...")
                progress_bar = st.progress(0)
                status_text = st.empty()
            
            # Read the CSV file with chunking for large files
            if file_size_mb > 50:
                status_text.text("Reading CSV file...")
                progress_bar.progress(20)
                
                # Read in chunks for large files
                chunk_size = 10000
                chunks = []
                uploaded_file.seek(0)  # Reset file pointer
                
                for i, chunk in enumerate(pd.read_csv(uploaded_file, chunksize=chunk_size)):
                    chunks.append(chunk)
                    progress = min(20 + (i * 5), 80)
                    progress_bar.progress(progress)
                    status_text.text(f"Reading chunk {i+1}...")
                
                df = pd.concat(chunks, ignore_index=True)
                progress_bar.progress(100)
                status_text.text("File reading completed!")
            else:
                df = pd.read_csv(uploaded_file)
            
            st.success(f"✅ File uploaded successfully! Shape: {df.shape}")
            
            # Show data preview
            st.subheader("📋 Data Preview")
            st.dataframe(df.head(10))
            
            # Show data info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Rows", len(df))
            with col2:
                st.metric("Total Columns", len(df.columns))
            with col3:
                st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            
            # Column configuration
            st.subheader("⚙️ Column Configuration")
            
            col1, col2 = st.columns(2)
            with col1:
                key_columns = st.multiselect(
                    "Select key columns (for composite key):",
                    df.columns.tolist(),
                    default=df.columns[:3].tolist() if len(df.columns) >= 3 else df.columns.tolist(),
                    help="These columns will be combined to create composite keys"
                )
            
            with col2:
                value_column = st.selectbox(
                    "Select value column:",
                    df.columns.tolist(),
                    index=len(df.columns)-1 if len(df.columns) > 3 else 0,
                    help="This column contains the values to be validated"
                )
            
            # Update config
            if key_columns and value_column:
                st.session_state.config['key_columns'] = key_columns
                st.session_state.config['value_column'] = value_column
                
                # Process data button
                if st.button("🔄 Process Data", type="primary"):
                    with st.spinner("Processing data..."):
                        try:
                            # For large files, show progress
                            if file_size_mb > 50:
                                progress_bar = st.progress(0)
                                status_text = st.empty()
                                status_text.text("Starting data processing...")
                            
                            # Process the data
                            ingester = DataIngester(st.session_state.config)
                            
                            if file_size_mb > 50:
                                status_text.text("Processing data in chunks...")
                                progress_bar.progress(30)
                            
                            processed_df, key_stats_df = ingester.process_file(uploaded_file)
                            
                            if file_size_mb > 50:
                                progress_bar.progress(100)
                                status_text.text("Data processing completed!")
                            
                            # Store in session state
                            st.session_state.processed_data = processed_df
                            st.session_state.key_stats_df = key_stats_df
                            
                            st.success("✅ Data processed successfully!")
                            
                            # Show processing results
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Processed Rows", len(processed_df))
                            with col2:
                                st.metric("Unique Keys", len(key_stats_df))
                            with col3:
                                st.metric("Avg Values per Key", f"{len(processed_df) / len(key_stats_df):.1f}")
                            
                        except Exception as e:
                            st.error(f"❌ Error processing data: {str(e)}")
            
        except Exception as e:
            st.error(f"❌ Error reading file: {str(e)}")

def show_pattern_analysis_page():
    """Show pattern analysis page."""
    st.header("🔍 Pattern Analysis")
    
    if st.session_state.processed_data is None:
        st.warning("⚠️ Please upload and process data first.")
        return
    
    # Generate profiles
    if st.button("🔍 Analyze Patterns", type="primary"):
        with st.spinner("Analyzing patterns..."):
            try:
                profiler = PatternProfiler(st.session_state.config)
                key_profiles = profiler.analyze_key_patterns(
                    st.session_state.key_stats_df, 
                    st.session_state.processed_data
                )
                
                st.session_state.key_profiles = key_profiles
                st.success(f"✅ Pattern analysis completed for {len(key_profiles)} keys!")
                
            except Exception as e:
                st.error(f"❌ Error analyzing patterns: {str(e)}")
    
    # Show analysis results
    if st.session_state.key_profiles:
        st.subheader("📊 Analysis Results")
        
        # Key statistics
        total_keys = len(st.session_state.key_profiles)
        free_text_keys = sum(1 for p in st.session_state.key_profiles.values() if p.get('is_free_text', False))
        numeric_keys = sum(1 for p in st.session_state.key_profiles.values() if p.get('is_numeric_unit', False))
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Keys", total_keys)
        with col2:
            st.metric("Free Text Keys", free_text_keys)
        with col3:
            st.metric("Numeric Keys", numeric_keys)
        with col4:
            st.metric("Structured Keys", total_keys - free_text_keys - numeric_keys)
        
        # Key selection
        selected_key = st.selectbox(
            "Select a key to analyze:",
            list(st.session_state.key_profiles.keys())
        )
        
        if selected_key:
            profile = st.session_state.key_profiles[selected_key]
            
            # Show key details
            st.subheader(f"🔍 Analysis for: {selected_key}")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Values", profile['count'])
                st.metric("Unique Values", profile['unique_values'])
                st.metric("Unique Ratio", f"{profile['unique_ratio']:.3f}")
            
            with col2:
                st.metric("Is Free Text", "Yes" if profile.get('is_free_text', False) else "No")
                st.metric("Is Numeric Unit", "Yes" if profile.get('is_numeric_unit', False) else "No")
                st.metric("Top Signature", profile['top_signatures'][0]['sig'] if profile['top_signatures'] else "N/A")
            
            # Show top signatures
            if profile['top_signatures']:
                st.subheader("📈 Top Signatures")
                sig_data = pd.DataFrame(profile['top_signatures'])
                st.dataframe(sig_data)
                
                # Signature chart
                fig = px.bar(
                    sig_data, 
                    x='sig', 
                    y='count',
                    title="Signature Distribution",
                    labels={'sig': 'Signature', 'count': 'Count'}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Show sample values
            if 'sample_values' in profile:
                st.subheader("📝 Sample Values")
                sample_df = pd.DataFrame({'Values': profile['sample_values']})
                st.dataframe(sample_df)

def show_model_training_page():
    """Show model training page."""
    st.header("🤖 Model Training")
    
    if st.session_state.key_profiles is None:
        st.warning("⚠️ Please analyze patterns first.")
        return
    
    # Training options
    st.subheader("⚙️ Training Configuration")
    
    col1, col2 = st.columns(2)
    with col1:
        train_clustering = st.checkbox("Train Clustering Models", value=True)
        train_embeddings = st.checkbox("Train Embedding Models", value=True)
    
    with col2:
        train_normalization = st.checkbox("Train Normalization Models", value=True)
        use_gpu = st.checkbox("Use GPU (if available)", value=False)
    
    # Advanced settings
    with st.expander("🔧 Advanced Settings"):
        col1, col2 = st.columns(2)
        with col1:
            hdbscan_min_cluster_size = st.slider("HDBSCAN Min Cluster Size", 3, 20, 5)
            embedding_batch_size = st.slider("Embedding Batch Size", 16, 128, 64)
        
        with col2:
            kmeans_n_clusters = st.slider("KMeans N Clusters", 2, 20, 8)
            embedding_model = st.selectbox(
                "Embedding Model",
                ["all-MiniLM-L6-v2", "all-mpnet-base-v2"],
                help="all-MiniLM-L6-v2 is faster, all-mpnet-base-v2 is more accurate"
            )
    
    # Start training
    if st.button("🚀 Start Training", type="primary"):
        with st.spinner("Training models..."):
            try:
                # Update config
                st.session_state.config['hdbscan_min_cluster_size'] = hdbscan_min_cluster_size
                st.session_state.config['kmeans_n_clusters'] = kmeans_n_clusters
                st.session_state.config['embedding_batch_size'] = embedding_batch_size
                st.session_state.config['embedding_model'] = embedding_model
                
                # Initialize trainer
                trainer = ModelTrainer(st.session_state.config)
                
                # Prepare training data
                training_data = trainer.prepare_training_data(
                    st.session_state.key_profiles, 
                    st.session_state.processed_data
                )
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Train clustering models
                if train_clustering:
                    status_text.text("Training clustering models...")
                    clustering_models = trainer.train_clustering_models(training_data)
                    progress_bar.progress(25)
                    st.success(f"✅ Clustering models trained for {len(clustering_models)} keys")
                
                # Train embedding models
                if train_embeddings:
                    status_text.text("Training embedding models...")
                    embedding_models = trainer.train_embedding_models(training_data)
                    progress_bar.progress(50)
                    if embedding_models:
                        st.success(f"✅ Embedding model trained with {embedding_models['training_examples']} examples")
                    else:
                        st.warning("⚠️ No embedding training data available")
                
                # Train normalization models
                if train_normalization:
                    status_text.text("Training normalization models...")
                    normalization_models = trainer.train_normalization_models(training_data)
                    progress_bar.progress(75)
                    if normalization_models:
                        st.success(f"✅ Normalization model trained with {len(normalization_models['rules'])} rules")
                    else:
                        st.warning("⚠️ No normalization training data available")
                
                # Save models
                status_text.text("Saving models...")
                trainer.save_trained_models("./trained_models")
                progress_bar.progress(100)
                
                # Store in session state
                st.session_state.trained_models = trainer.trained_models
                
                st.success("🎉 Model training completed successfully!")
                
                # Show training summary
                summary = trainer.get_training_summary()
                st.subheader("📊 Training Summary")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Models Trained", len(summary['models_trained']))
                with col2:
                    st.metric("Clustering Keys", summary['training_data_size']['clustering_keys'])
                with col3:
                    st.metric("Embedding Pairs", summary['training_data_size']['embedding_pairs'])
                
            except Exception as e:
                st.error(f"❌ Error training models: {str(e)}")

def show_validation_page():
    """Show validation page."""
    st.header("✅ Value Validation")
    
    if st.session_state.key_profiles is None:
        st.warning("⚠️ Please analyze patterns first.")
        return
    
    # Validation form
    st.subheader("🔍 Validate New Values")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Key selection
        selected_key = st.selectbox(
            "Select composite key:",
            list(st.session_state.key_profiles.keys())
        )
        
        if selected_key:
            # Parse key parts
            key_parts = selected_key.split('||')
            st.info(f"Key parts: {key_parts}")
    
    with col2:
        # Value input
        value_to_validate = st.text_input(
            "Enter value to validate:",
            placeholder="e.g., SMB Mini Jack Right Angle"
        )
    
    # Validate button
    if st.button("🔍 Validate Value", type="primary") and value_to_validate:
        with st.spinner("Validating value..."):
            try:
                # Simple validation logic (you can enhance this)
                profile = st.session_state.key_profiles[selected_key]
                
                # Check if value matches patterns
                matches_pattern = False
                suggested_fix = None
                confidence = 0.0
                issues = []
                
                # Basic pattern matching
                if profile.get('candidate_regex'):
                    import re
                    if re.match(profile['candidate_regex'], value_to_validate):
                        matches_pattern = True
                        confidence = 0.9
                    else:
                        issues.append("format_mismatch")
                        # Suggest fix based on sample values
                        if profile.get('sample_values'):
                            suggested_fix = profile['sample_values'][0]
                
                # Determine verdict
                if matches_pattern:
                    verdict = "accepted"
                elif suggested_fix:
                    verdict = "reformatted"
                else:
                    verdict = "needs_review"
                    confidence = 0.3
                
                # Show results
                st.subheader("📊 Validation Results")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Verdict", verdict.title())
                with col2:
                    st.metric("Confidence", f"{confidence:.2f}")
                with col3:
                    st.metric("Issues", len(issues))
                
                if issues:
                    st.warning(f"Issues found: {', '.join(issues)}")
                
                if suggested_fix:
                    st.info(f"Suggested fix: {suggested_fix}")
                
                # Show nearest examples
                if profile.get('sample_values'):
                    st.subheader("🔍 Nearest Examples")
                    examples_df = pd.DataFrame({'Examples': profile['sample_values'][:5]})
                    st.dataframe(examples_df)
                
            except Exception as e:
                st.error(f"❌ Error validating value: {str(e)}")

def show_analytics_page():
    """Show analytics page."""
    st.header("📈 Analytics Dashboard")
    
    if st.session_state.key_profiles is None:
        st.warning("⚠️ Please analyze patterns first.")
        return
    
    # Key statistics
    st.subheader("📊 Key Statistics")
    
    profiles_data = []
    for key, profile in st.session_state.key_profiles.items():
        profiles_data.append({
            'Key': key,
            'Count': profile['count'],
            'Unique Values': profile['unique_values'],
            'Unique Ratio': profile['unique_ratio'],
            'Is Free Text': profile.get('is_free_text', False),
            'Is Numeric Unit': profile.get('is_numeric_unit', False)
        })
    
    profiles_df = pd.DataFrame(profiles_data)
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Keys", len(profiles_df))
    with col2:
        st.metric("Avg Values per Key", f"{profiles_df['Count'].mean():.1f}")
    with col3:
        st.metric("Free Text Keys", profiles_df['Is Free Text'].sum())
    with col4:
        st.metric("Numeric Keys", profiles_df['Is Numeric Unit'].sum())
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Count distribution
        fig1 = px.histogram(
            profiles_df, 
            x='Count', 
            title="Distribution of Value Counts per Key",
            nbins=20
        )
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        # Unique ratio distribution
        fig2 = px.histogram(
            profiles_df, 
            x='Unique Ratio', 
            title="Distribution of Unique Ratios",
            nbins=20
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    # Key types pie chart
    key_types = {
        'Free Text': profiles_df['Is Free Text'].sum(),
        'Numeric Unit': profiles_df['Is Numeric Unit'].sum(),
        'Structured': len(profiles_df) - profiles_df['Is Free Text'].sum() - profiles_df['Is Numeric Unit'].sum()
    }
    
    fig3 = px.pie(
        values=list(key_types.values()),
        names=list(key_types.keys()),
        title="Key Types Distribution"
    )
    st.plotly_chart(fig3, use_container_width=True)
    
    # Top keys by count
    st.subheader("🏆 Top Keys by Value Count")
    top_keys = profiles_df.nlargest(10, 'Count')[['Key', 'Count', 'Unique Values', 'Unique Ratio']]
    st.dataframe(top_keys, use_container_width=True)

def show_settings_page():
    """Show settings page."""
    st.header("⚙️ Settings")
    
    st.subheader("🔧 Configuration")
    
    # Basic settings
    col1, col2 = st.columns(2)
    
    with col1:
        st.session_state.config['chunk_size'] = st.number_input(
            "Chunk Size",
            min_value=1000,
            max_value=100000,
            value=st.session_state.config['chunk_size'],
            step=1000
        )
        
        st.session_state.config['min_signature_frequency'] = st.number_input(
            "Min Signature Frequency",
            min_value=1,
            max_value=50,
            value=st.session_state.config['min_signature_frequency']
        )
    
    with col2:
        st.session_state.config['hdbscan_min_cluster_size'] = st.number_input(
            "HDBSCAN Min Cluster Size",
            min_value=2,
            max_value=50,
            value=st.session_state.config['hdbscan_min_cluster_size']
        )
        
        st.session_state.config['kmeans_n_clusters'] = st.number_input(
            "KMeans N Clusters",
            min_value=2,
            max_value=50,
            value=st.session_state.config['kmeans_n_clusters']
        )
    
    # Advanced settings
    with st.expander("🔬 Advanced Settings"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.session_state.config['embedding_model'] = st.selectbox(
                "Embedding Model",
                ["all-MiniLM-L6-v2", "all-mpnet-base-v2"],
                index=0 if st.session_state.config['embedding_model'] == "all-MiniLM-L6-v2" else 1
            )
            
            st.session_state.config['tfidf_max_features'] = st.number_input(
                "TF-IDF Max Features",
                min_value=100,
                max_value=10000,
                value=st.session_state.config['tfidf_max_features']
            )
        
        with col2:
            st.session_state.config['embedding_batch_size'] = st.number_input(
                "Embedding Batch Size",
                min_value=16,
                max_value=256,
                value=st.session_state.config['embedding_batch_size']
            )
            
            st.session_state.config['index_type'] = st.selectbox(
                "FAISS Index Type",
                ["flat", "ivf", "ivfpq"],
                index=0 if st.session_state.config['index_type'] == "flat" else 1
            )
    
    # Save configuration
    if st.button("💾 Save Configuration", type="primary"):
        st.success("✅ Configuration saved!")
    
    # Export/Import
    st.subheader("📁 Export/Import")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📤 Export Configuration"):
            config_json = json.dumps(st.session_state.config, indent=2)
            st.download_button(
                "Download Config",
                config_json,
                "autopatternchecker_config.json",
                "application/json"
            )
    
    with col2:
        uploaded_config = st.file_uploader(
            "Import Configuration",
            type="json"
        )
        
        if uploaded_config:
            try:
                config_data = json.load(uploaded_config)
                st.session_state.config.update(config_data)
                st.success("✅ Configuration imported successfully!")
            except Exception as e:
                st.error(f"❌ Error importing configuration: {str(e)}")

if __name__ == "__main__":
    main()