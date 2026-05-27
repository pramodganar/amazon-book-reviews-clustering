# ============================================================
# AMAZON BOOK REVIEWS CLUSTERING PROJECT
# STREAMLIT DASHBOARD
# ============================================================

# ============================================================
# IMPORT LIBRARIES
# ============================================================

import streamlit as st

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from wordcloud import WordCloud

import warnings
warnings.filterwarnings("ignore")


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(

    page_title="Amazon Book Reviews Clustering",

    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("📚 Amazon Book Reviews Clustering Dashboard")

st.markdown(
    """
    NLP-powered customer review segmentation using
    clustering algorithms.
    """
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    df = pd.read_csv(
        "data/processed/clustered_reviews.csv"
    )

    return df


@st.cache_data
def load_tsne():

    tsne_df = pd.read_csv(
        "data/processed/tsne_features.csv"
    )

    return tsne_df


@st.cache_data
def load_model_comparison():

    comparison_df = pd.read_csv(
        "reports/model_comparison.csv"
    )

    return comparison_df


df = load_data()

tsne_df = load_tsne()

comparison_df = load_model_comparison()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("Navigation")

page = st.sidebar.radio(

    "Go To",

    [

        "Project Overview",

        "Dataset Overview",

        "Cluster Distribution",

        "t-SNE Visualization",

        "Review Explorer",

        "Word Clouds",

        "Model Comparison",

        "Business Insights"
    ]
)


# ============================================================
# PROJECT OVERVIEW
# ============================================================

if page == "Project Overview":

    st.header("📌 Project Overview")

    st.markdown(
        """
        ## Objective

        Build an NLP-powered clustering system
        for Amazon Book Reviews.

        ## Algorithms Used

        - KMeans
        - Agglomerative Clustering
        - DBSCAN

        ## NLP Techniques

        - TF-IDF
        - PCA
        - t-SNE
        - Text Preprocessing

        ## Business Applications

        - Customer Segmentation
        - Review Pattern Analysis
        - Recommendation Systems
        - Fake Review Detection
        """
    )

    st.subheader("Dataset Shape")

    st.write(df.shape)

    st.subheader("Dataset Preview")

    st.dataframe(df.head())


# ============================================================
# DATASET OVERVIEW
# ============================================================

elif page == "Dataset Overview":

    st.header("📊 Dataset Overview")

    st.subheader("Columns")

    st.write(df.columns.tolist())

    st.subheader("Missing Values")

    st.dataframe(
        df.isnull().sum()
    )

    st.subheader("Review Length Statistics")

    st.dataframe(
        df["review_length"].describe()
    )

    st.subheader("Word Count Statistics")

    st.dataframe(
        df["word_count"].describe()
    )

    st.subheader("Dataset Sample")

    st.dataframe(df.head(20))


# ============================================================
# CLUSTER DISTRIBUTION
# ============================================================

elif page == "Cluster Distribution":

    st.header("📈 Cluster Distribution")

    cluster_column = st.selectbox(

        "Select Clustering Model",

        [

            "kmeans_cluster",

            "agg_cluster",

            "dbscan_cluster"
        ]
    )

    fig, ax = plt.subplots(
        figsize=(10,6)
    )

    sns.countplot(

        x=cluster_column,

        data=df,

        ax=ax
    )

    ax.set_title(
        "Cluster Distribution"
    )

    st.pyplot(fig)

    st.subheader("Cluster Counts")

    st.dataframe(
        df[cluster_column]
        .value_counts()
    )


# ============================================================
# t-SNE VISUALIZATION
# ============================================================

elif page == "t-SNE Visualization":

    st.header("🧠 t-SNE Cluster Visualization")

    tsne_df["cluster"] = (
        df["kmeans_cluster"]
        .iloc[:len(tsne_df)]
        .values
    )

    fig, ax = plt.subplots(
        figsize=(12,8)
    )

    sns.scatterplot(

        x="tsne_1",

        y="tsne_2",

        hue="cluster",

        palette="tab10",

        data=tsne_df,

        ax=ax
    )

    ax.set_title(
        "t-SNE Visualization"
    )

    st.pyplot(fig)


# ============================================================
# REVIEW EXPLORER
# ============================================================

elif page == "Review Explorer":

    st.header("🔍 Review Explorer")

    cluster = st.selectbox(

        "Select Cluster",

        sorted(
            df["kmeans_cluster"]
            .unique()
        )
    )

    filtered_df = df[
        df["kmeans_cluster"] == cluster
    ]

    st.subheader(
        f"Cluster {cluster} Reviews"
    )

    st.write(
        f"Total Reviews: {len(filtered_df)}"
    )

    sample_size = min(
        20,
        len(filtered_df)
    )

    sample_reviews = filtered_df.sample(
        sample_size
    )

    for idx, row in sample_reviews.iterrows():

        st.markdown("---")

        st.write(
            f"### 📖 {row['title']}"
        )

        st.write(
            row["review_text"][:500]
        )


# ============================================================
# WORD CLOUDS
# ============================================================

elif page == "Word Clouds":

    st.header("☁️ Cluster Word Clouds")

    cluster = st.selectbox(

        "Select Cluster for Word Cloud",

        sorted(
            df["kmeans_cluster"]
            .unique()
        )
    )

    cluster_text = " ".join(

        df[
            df["kmeans_cluster"] == cluster
        ]["cleaned_text"]
    )

    wordcloud = WordCloud(

        width=1000,

        height=500,

        background_color="white"
    ).generate(cluster_text)

    fig, ax = plt.subplots(
        figsize=(14,7)
    )

    ax.imshow(wordcloud)

    ax.axis("off")

    st.pyplot(fig)


# ============================================================
# MODEL COMPARISON
# ============================================================

elif page == "Model Comparison":

    st.header("⚙️ Clustering Model Comparison")

    st.subheader("Evaluation Metrics")

    st.dataframe(comparison_df)

    metric = st.selectbox(

        "Select Metric",

        [

            "Silhouette_Score"
        ]
    )

    fig, ax = plt.subplots(
        figsize=(10,6)
    )

    sns.barplot(

        x="Model",

        y=metric,

        data=comparison_df,

        ax=ax
    )

    ax.set_title(
        f"{metric} Comparison"
    )

    st.pyplot(fig)


# ============================================================
# BUSINESS INSIGHTS
# ============================================================

elif page == "Business Insights":

    st.header("💡 Business Insights")

    st.markdown(
        """
        ## Key Findings

        ### 📌 Customer Segments Identified

        - Analytical Reviewers
        - Emotional Readers
        - Short Opinion Reviewers
        - Highly Expressive Customers
        - Book Enthusiasts

        ### 📌 Business Applications

        - Personalized Recommendations
        - Customer Behavior Analysis
        - Fake Review Detection
        - Marketing Optimization

        ### 📌 Project Achievements

        - NLP-based clustering
        - Multiple clustering algorithms
        - Dimensionality reduction
        - Interactive dashboard
        """
    )

    st.subheader("Top Reviewed Books")

    top_books = (
        df["title"]
        .value_counts()
        .head(10)
    )

    fig, ax = plt.subplots(
        figsize=(12,6)
    )

    sns.barplot(

        x=top_books.values,

        y=top_books.index,

        ax=ax
    )

    ax.set_title(
        "Top Reviewed Books"
    )

    st.pyplot(fig)


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.markdown(
    "Built using Streamlit | NLP | Clustering | Machine Learning"
)