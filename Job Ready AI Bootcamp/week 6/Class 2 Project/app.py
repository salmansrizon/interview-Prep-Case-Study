"""
Customer Segmentation Engine
─────────────────────────────
A production-grade Streamlit application for Unsupervised Learning.
Covers K-Means Clustering, PCA, and Market Basket Analysis.

Runs 100% offline. No API keys required.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import config
from src.data.loader import DataLoader
from src.data.preprocessor import DataPreprocessor
from src.features.engineering import FeatureEngineer
from src.models.kmeans_engine import KMeansEngine
from src.models.pca_engine import PCAEngine
from src.models.market_basket import MarketBasketAnalyzer
from src.visualization.charts import ClusterVisualizer, MBAVisualizer
from src.utils.logger import get_logger

logger = get_logger("app")

# ── Page Config ─────────────────────────────────────
st.set_page_config(
    page_title="Customer Segmentation Engine",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──────────────────────────────────────
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: 700; color: #2E86AB; }
    .sub-header { font-size: 1.2rem; color: #555; margin-bottom: 1rem; }
    .metric-card { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                   color: white; border-radius: 10px; padding: 1rem; text-align: center; }
    .stAlert { border-radius: 8px; }
    .cluster-badge { display: inline-block; padding: 4px 12px; border-radius: 12px; 
                   font-weight: 600; font-size: 0.85rem; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────
st.sidebar.markdown("## ⚙️ Configuration")

module = st.sidebar.radio(
    "Select Module",
    ["🎯 K-Means Clustering", "📉 PCA & Dimensionality Reduction", "🛒 Market Basket Analysis"],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Dataset Options")

data_source = st.sidebar.radio(
    "Data Source",
    ["Generate Synthetic (Offline)", "Upload CSV"],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.info(
    """
    **Class — Unsupervised Learning**
    - K-Means Clustering
    - Principal Component Analysis (PCA)
    - Market Basket Analysis
    """
)

# ── Main Content ──────────────────────────────────────
st.markdown('<div class="main-header">🎯 Customer Segmentation Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Unsupervised Learning — 100% Offline</div>', unsafe_allow_html=True)

# ── Shared State ──────────────────────────────────────
if "dataset" not in st.session_state:
    st.session_state.dataset = None
if "preprocessed" not in st.session_state:
    st.session_state.preprocessed = None
if "kmeans_model" not in st.session_state:
    st.session_state.kmeans_model = None
if "pca_model" not in st.session_state:
    st.session_state.pca_model = None
if "mba_results" not in st.session_state:
    st.session_state.mba_results = None

# ── Data Loading ──────────────────────────────────────
def load_data():
    if st.session_state.dataset is not None:
        return st.session_state.dataset

    loader = DataLoader()
    if data_source == "Generate Synthetic (Offline)":
        df = loader.generate_customer_dataset(n_customers=1000)
        st.session_state.dataset = df
        path = loader.save_processed(df, "synthetic_customers")
        st.sidebar.success(f"Generated {len(df)} records")
        return df
    else:
        uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])
        if uploaded:
            df = pd.read_csv(uploaded)
            st.session_state.dataset = df
            return df
    return None

# ── MODULE 1: K-Means Clustering ──────────────────────
if module == "🎯 K-Means Clustering":
    st.subheader("1. K-Means Clustering — Customer Segmentation")

    col1, col2 = st.columns([2, 1])

    with col2:
        st.markdown("#### 🔧 Parameters")
        n_clusters = st.slider("Number of Clusters (K)", 2, 15, 5, 1)
        init_method = st.selectbox("Initialization", ["k-means++", "random"], index=0)
        max_iter = st.slider("Max Iterations", 100, 1000, 300, 50)

        st.markdown("---")
        st.markdown("#### 📚 About K-Means")
        st.markdown("""
        **K-Means** partitions data into K clusters by minimizing 
        within-cluster sum of squares (WCSS).

        **Algorithm:**
        1. Initialize K centroids
        2. Assign each point to nearest centroid
        3. Recalculate centroids as mean of assigned points
        4. Repeat until convergence
        """)

    with col1:
        df = load_data()
        if df is None:
            st.info("👈 Use the sidebar to generate or upload data.")
        else:
            st.markdown("**Dataset Preview**")
            st.dataframe(df.head(8), use_container_width=True)

            # Preprocess
            preprocessor = DataPreprocessor()
            features_df = preprocessor.prepare_for_clustering(df)
            st.session_state.preprocessed = features_df

            st.markdown("**Features for Clustering**")
            st.dataframe(features_df.head(8), use_container_width=True)

            if st.button("🚀 Run K-Means", type="primary"):
                with st.spinner("Clustering customers..."):
                    kmeans = KMeansEngine(
                        n_clusters=n_clusters,
                        init=init_method,
                        max_iter=max_iter,
                    )
                    labels, inertia, centers = kmeans.fit(features_df)
                    st.session_state.kmeans_model = kmeans

                    df_result = df.copy()
                    df_result["Cluster"] = labels

                    # Metrics
                    st.markdown("---")
                    st.markdown("#### 📊 Clustering Results")

                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Clusters", n_clusters)
                    m2.metric("Inertia (WCSS)", f"{inertia:,.0f}")
                    m3.metric("Silhouette Score", f"{kmeans.silhouette:.3f}")
                    m4.metric("Calinski-Harabasz", f"{kmeans.calinski:.1f}")

                    # Cluster distribution
                    st.markdown("---")
                    st.markdown("**Cluster Distribution**")
                    cluster_counts = df_result["Cluster"].value_counts().sort_index()

                    fig_pie = px.pie(
                        values=cluster_counts.values,
                        names=[f"Cluster {i}" for i in cluster_counts.index],
                        title="Customer Distribution by Cluster",
                        color_discrete_sequence=px.colors.qualitative.Set3,
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)

                    # Cluster profiles
                    st.markdown("---")
                    st.markdown("**Cluster Profiles (Mean Values)**")
                    profile = df_result.groupby("Cluster")[features_df.columns].mean().round(2)
                    st.dataframe(profile, use_container_width=True)

                    # 2D Visualization using PCA
                    st.markdown("---")
                    st.markdown("**Cluster Visualization (PCA-2D)**")

                    pca_viz = PCAEngine(n_components=2)
                    pca_viz.fit(features_df)
                    coords = pca_viz.transform(features_df)

                    viz_df = pd.DataFrame(coords, columns=["PC1", "PC2"])
                    viz_df["Cluster"] = labels.astype(str)

                    fig_scatter = px.scatter(
                        viz_df, x="PC1", y="PC2", color="Cluster",
                        title="Customer Clusters in 2D PCA Space",
                        color_discrete_sequence=px.colors.qualitative.Bold,
                        opacity=0.7,
                    )
                    st.plotly_chart(fig_scatter, use_container_width=True)

                    # Elbow method
                    st.markdown("---")
                    st.markdown("**Elbow Method — Optimal K Finder**")

                    k_range = range(2, 16)
                    inertias = []
                    silhouettes = []

                    for k in k_range:
                        km_temp = KMeansEngine(n_clusters=k, init="k-means++", max_iter=300)
                        km_temp.fit(features_df)
                        inertias.append(km_temp.inertia)
                        silhouettes.append(km_temp.silhouette)

                    fig_elbow = make_subplots(specs=[[{"secondary_y": True}]])
                    fig_elbow.add_trace(
                        go.Scatter(x=list(k_range), y=inertias, name="Inertia (WCSS)",
                                   mode="lines+markers", line=dict(color="#1f77b4")),
                        secondary_y=False,
                    )
                    fig_elbow.add_trace(
                        go.Scatter(x=list(k_range), y=silhouettes, name="Silhouette Score",
                                   mode="lines+markers", line=dict(color="#ff7f0e")),
                        secondary_y=True,
                    )
                    fig_elbow.update_xaxes(title_text="Number of Clusters (K)")
                    fig_elbow.update_yaxes(title_text="Inertia (WCSS)", secondary_y=False)
                    fig_elbow.update_yaxes(title_text="Silhouette Score", secondary_y=True)
                    fig_elbow.update_layout(title="Elbow Method & Silhouette Analysis")
                    st.plotly_chart(fig_elbow, use_container_width=True)

# ── MODULE 2: PCA ─────────────────────────────────────
elif module == "📉 PCA & Dimensionality Reduction":
    st.subheader("2. Principal Component Analysis (PCA)")

    col1, col2 = st.columns([2, 1])

    with col2:
        st.markdown("#### 🔧 Parameters")
        n_components = st.slider("Number of Components", 2, 10, 2, 1)
        auto_select = st.checkbox("Auto-select by variance threshold (95%)", value=False)

        st.markdown("---")
        st.markdown("#### 📚 About PCA")
        st.markdown("""
        **PCA** transforms data into a new coordinate system 
        where the greatest variance lies on the first axis.

        **Key Concepts:**
        - **Principal Components**: Directions of maximum variance
        - **Explained Variance**: How much info each PC captures
        - **Dimensionality Reduction**: Compress data while preserving structure
        """)

    with col1:
        df = load_data()
        if df is None:
            st.info("👈 Use the sidebar to generate or upload data.")
        else:
            st.markdown("**Dataset Preview**")
            st.dataframe(df.head(8), use_container_width=True)

            preprocessor = DataPreprocessor()
            features_df = preprocessor.prepare_for_clustering(df)

            if st.button("🚀 Run PCA", type="primary"):
                with st.spinner("Reducing dimensions..."):
                    pca = PCAEngine(n_components=n_components)
                    pca.fit(features_df)
                    st.session_state.pca_model = pca

                    transformed = pca.transform(features_df)

                    # Metrics
                    st.markdown("---")
                    st.markdown("#### 📊 PCA Results")

                    m1, m2, m3 = st.columns(3)
                    m1.metric("Components", pca.n_components)
                    m2.metric("Total Variance Explained", f"{pca.total_variance:.1%}")
                    m3.metric("Original Dimensions", len(features_df.columns))

                    # Explained variance
                    st.markdown("---")
                    st.markdown("**Explained Variance by Component**")

                    var_df = pd.DataFrame({
                        "Component": [f"PC{i+1}" for i in range(pca.n_components)],
                        "Variance": pca.explained_variance,
                        "Cumulative": pca.cumulative_variance,
                    })

                    fig_var = go.Figure()
                    fig_var.add_trace(go.Bar(
                        x=var_df["Component"], y=var_df["Variance"],
                        name="Individual", marker_color="#2ca02c"
                    ))
                    fig_var.add_trace(go.Scatter(
                        x=var_df["Component"], y=var_df["Cumulative"],
                        name="Cumulative", mode="lines+markers",
                        line=dict(color="#d62728", width=3)
                    ))
                    fig_var.update_layout(
                        title="Explained Variance Ratio",
                        xaxis_title="Principal Component",
                        yaxis_title="Variance Ratio",
                        yaxis=dict(tickformat=".0%"),
                    )
                    st.plotly_chart(fig_var, use_container_width=True)

                    # Feature loadings
                    st.markdown("---")
                    st.markdown("**Feature Loadings (Top Contributors)**")

                    loadings = pd.DataFrame(
                        pca.components,
                        columns=features_df.columns,
                        index=[f"PC{i+1}" for i in range(pca.n_components)]
                    )

                    # Heatmap
                    fig_heat = px.imshow(
                        loadings,
                        title="PCA Component Loadings",
                        color_continuous_scale="RdBu_r",
                        aspect="auto",
                    )
                    st.plotly_chart(fig_heat, use_container_width=True)

                    # 2D scatter
                    if n_components >= 2:
                        st.markdown("---")
                        st.markdown("**Data Projection (PC1 vs PC2)**")

                        proj_df = pd.DataFrame(transformed[:, :2], columns=["PC1", "PC2"])

                        fig_proj = px.scatter(
                            proj_df, x="PC1", y="PC2",
                            title="Data in First Two Principal Components",
                            opacity=0.6,
                            color_discrete_sequence=["#9467bd"],
                        )
                        st.plotly_chart(fig_proj, use_container_width=True)

                    # Scree plot for all possible components
                    st.markdown("---")
                    st.markdown("**Scree Plot — All Components**")

                    pca_full = PCAEngine(n_components=min(len(features_df.columns), 20))
                    pca_full.fit(features_df)

                    scree_df = pd.DataFrame({
                        "Component": range(1, len(pca_full.explained_variance) + 1),
                        "Variance": pca_full.explained_variance,
                    })

                    fig_scree = px.line(
                        scree_df, x="Component", y="Variance",
                        markers=True, title="Scree Plot — Eigenvalue Decay",
                    )
                    fig_scree.add_hline(y=1/len(features_df.columns), line_dash="dash",
                                        annotation_text="Kaiser Criterion (1/dim)")
                    st.plotly_chart(fig_scree, use_container_width=True)

# ── MODULE 3: Market Basket Analysis ──────────────────
else:
    st.subheader("3. Market Basket Analysis — Association Rules")

    col1, col2 = st.columns([2, 1])

    with col2:
        st.markdown("#### 🔧 Parameters")
        min_support = st.slider("Min Support", 0.01, 0.3, 0.05, 0.01)
        min_confidence = st.slider("Min Confidence", 0.1, 1.0, 0.3, 0.05)
        min_lift = st.slider("Min Lift", 0.5, 5.0, 1.5, 0.1)
        max_len = st.slider("Max Itemset Length", 2, 6, 3, 1)

        st.markdown("---")
        st.markdown("#### 📚 About MBA")
        st.markdown("""
        **Market Basket Analysis** discovers associations 
        between products using the Apriori algorithm.

        **Key Metrics:**
        - **Support**: Frequency of itemset in transactions
        - **Confidence**: P(RHS | LHS) — likelihood of RHS given LHS
        - **Lift**: Confidence / Expected confidence. >1 means positive association
        """)

    with col1:
        st.markdown("**Generate Transaction Data**")

        n_transactions = st.number_input("Number of Transactions", 100, 10000, 1000, 100)
        n_products = st.number_input("Number of Products", 10, 100, 50, 5)

        if st.button("🛒 Generate & Analyze", type="primary"):
            with st.spinner("Mining association rules..."):
                loader = DataLoader()
                transactions = loader.generate_transaction_data(
                    n_transactions=n_transactions,
                    n_products=n_products,
                )

                st.markdown("**Sample Transactions**")
                for i, t in enumerate(transactions[:5]):
                    st.write(f"Transaction {i+1}: {', '.join(t)}")

                mba = MarketBasketAnalyzer(
                    min_support=min_support,
                    min_confidence=min_confidence,
                    min_lift=min_lift,
                    max_len=max_len,
                )

                rules = mba.fit(transactions)
                st.session_state.mba_results = rules

                if rules.empty:
                    st.warning("No rules found. Try lowering thresholds.")
                else:
                    st.markdown("---")
                    st.markdown(f"#### 📊 Found {len(rules)} Association Rules")

                    # Top rules
                    st.markdown("**Top Rules by Lift**")
                    top_rules = rules.nlargest(10, "lift")[[
                        "antecedents", "consequents", "support",
                        "confidence", "lift", "conviction"
                    ]]
                    top_rules["antecedents"] = top_rules["antecedents"].apply(lambda x: ", ".join(list(x)))
                    top_rules["consequents"] = top_rules["consequents"].apply(lambda x: ", ".join(list(x)))
                    st.dataframe(top_rules.round(3), use_container_width=True)

                    # Scatter: Support vs Confidence, colored by Lift
                    st.markdown("---")
                    st.markdown("**Rule Landscape — Support vs Confidence**")

                    fig_mba = px.scatter(
                        rules, x="support", y="confidence",
                        size="lift", color="lift",
                        hover_data=["antecedents", "consequents", "conviction"],
                        title="Association Rules: Support vs Confidence (size/color = Lift)",
                        color_continuous_scale="Viridis",
                    )
                    st.plotly_chart(fig_mba, use_container_width=True)

                    # Network-style visualization
                    st.markdown("---")
                    st.markdown("**Product Association Network**")

                    # Get top 15 rules for cleaner visualization
                    viz_rules = rules.nlargest(15, "lift")

                    nodes = set()
                    edges = []
                    for _, row in viz_rules.iterrows():
                        ant = list(row["antecedents"])
                        con = list(row["consequents"])
                        for a in ant:
                            for c in con:
                                nodes.add(a)
                                nodes.add(c)
                                edges.append((a, c, row["lift"]))

                    nodes = list(nodes)
                    node_indices = {n: i for i, n in enumerate(nodes)}

                    edge_x, edge_y = [], []
                    node_x, node_y = [], []

                    # Simple circular layout
                    import math
                    for i, node in enumerate(nodes):
                        angle = 2 * math.pi * i / len(nodes)
                        node_x.append(math.cos(angle))
                        node_y.append(math.sin(angle))

                    fig_net = go.Figure()

                    # Edges
                    for a, c, lift in edges:
                        i, j = node_indices[a], node_indices[c]
                        fig_net.add_trace(go.Scatter(
                            x=[node_x[i], node_x[j], None],
                            y=[node_y[i], node_y[j], None],
                            mode="lines",
                            line=dict(width=lift*0.5, color="rgba(100,100,100,0.5)"),
                            hoverinfo="none",
                            showlegend=False,
                        ))

                    # Nodes
                    fig_net.add_trace(go.Scatter(
                        x=node_x, y=node_y,
                        mode="markers+text",
                        marker=dict(size=20, color="#2E86AB"),
                        text=nodes,
                        textposition="top center",
                        hoverinfo="text",
                    ))

                    fig_net.update_layout(
                        title="Product Association Network (Top 15 Rules by Lift)",
                        showlegend=False,
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        plot_bgcolor="white",
                    )
                    st.plotly_chart(fig_net, use_container_width=True)

                    # Download results
                    st.markdown("---")
                    csv = rules.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Rules as CSV",
                        data=csv,
                        file_name="association_rules.csv",
                        mime="text/csv",
                    )

# ── Footer ────────────────────────────────────────────
st.markdown("---")
st.caption("Built for Unsupervised Learning Module | Runs 100% Offline | scikit-learn + Streamlit + mlxtend")
