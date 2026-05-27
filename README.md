# Amazon Book Reviews Clustering Project

## Overview

This project performs NLP-powered customer review segmentation using unsupervised machine learning techniques on Amazon Book Reviews.

The system clusters similar customer reviews and identifies hidden review patterns using TF-IDF vectorization, PCA, and multiple clustering algorithms.

---

# Project Objectives

- Discover hidden customer segments
- Analyze review behavior patterns
- Perform NLP-based clustering
- Compare multiple clustering algorithms
- Generate business insights
- Build an interactive Streamlit dashboard

---

# Technologies Used

## Programming Language
- Python

## Libraries
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- nltk
- wordcloud
- streamlit

---

# Machine Learning Techniques

## NLP
- Text preprocessing
- Tokenization
- Stopword removal
- TF-IDF Vectorization

## Dimensionality Reduction
- PCA
- t-SNE

## Clustering Algorithms
- KMeans
- Agglomerative Clustering
- DBSCAN

## Evaluation Metrics
- Silhouette Score
- Davies-Bouldin Index
- Calinski-Harabasz Score

---

# Project Architecture

Raw Reviews
↓
Text Preprocessing
↓
TF-IDF Vectorization
↓
Feature Engineering
↓
PCA Dimensionality Reduction
↓
Clustering Models
↓
Model Evaluation
↓
Business Insights
↓
Streamlit Dashboard

---

# Business Applications

- Customer Segmentation
- Recommendation Systems
- Review Pattern Analysis
- Fake Review Detection
- Marketing Optimization

---

# Dashboard Features

- Dataset Overview
- Cluster Visualization
- Word Clouds
- Review Explorer
- Model Comparison
- Business Insights

---

# How to Run

## Install Requirements

```bash
pip install -r requirements.txt

# Live Demo

https://amazon-book-reviews-clustering-ayciweghgchjtvcqmdymsd.streamlit.app/