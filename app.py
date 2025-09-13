import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os

st.set_page_config(page_title="Customer Segmentation", layout="wide")

st.title("Customer Segmentation using K-Means")

# Load the dataset
@st.cache_data
def load_data():
    df = pd.read_csv('./data/Mall_Customers.csv')
    return df

df = load_data()

st.sidebar.header("Data Exploration")

if st.sidebar.checkbox("Show Raw Data"):
    st.subheader("Raw Data")
    st.dataframe(df)

if st.sidebar.checkbox("Show Data Description"):
    st.subheader("Data Description")
    st.write(df.describe())

if st.sidebar.checkbox("Check for Missing Values"):
    st.subheader("Missing Values")
    st.write(df.isnull().sum())

st.sidebar.header("Clustering Parameters")
num_clusters = st.sidebar.slider("Select number of clusters (K)", 2, 10, 5)

# Select features for clustering
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# K-Means Clustering
kmeans = KMeans(n_clusters=num_clusters, init='k-means++', random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_scaled)

st.subheader(f"Customer Segments (K={num_clusters})")

# Visualize the clusters
fig1, ax1 = plt.subplots(figsize=(10, 7))
sns.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)', hue='Cluster', data=df, palette='viridis', s=100, alpha=0.8, ax=ax1)
ax1.set_title('Customer Segments (K-Means Clustering)')
ax1.set_xlabel('Annual Income (k$)')
ax1.set_ylabel('Spending Score (1-100)')
ax1.legend(title='Cluster')
ax1.grid(True)
st.pyplot(fig1)

# Display the characteristics of each cluster
st.subheader("Characteristics of Each Cluster")
cluster_summary = df.groupby('Cluster')[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].mean()
st.dataframe(cluster_summary)

# Save the clustered data to a new CSV file
output_path = './data/Mall_Customers_Clustered.csv'
df.to_csv(output_path, index=False)
st.success(f"Clustered data saved to {output_path}")

# Visualizations for distributions
st.subheader("Data Distributions")
fig2, axes = plt.subplots(1, 3, figsize=(18, 5))
sns.histplot(df['Age'], bins=20, kde=True, ax=axes[0])
axes[0].set_title('Distribution of Age')

sns.histplot(df['Annual Income (k$)'], bins=20, kde=True, ax=axes[1])
axes[1].set_title('Distribution of Annual Income (k$)')

sns.histplot(df['Spending Score (1-100)'], bins=20, kde=True, ax=axes[2])
axes[2].set_title('Distribution of Spending Score (1-100)')
plt.tight_layout()
st.pyplot(fig2)

# Visualizations for relationships
st.subheader("Relationships between Features")
fig3, axes = plt.subplots(1, 3, figsize=(18, 5))
sns.scatterplot(x='Age', y='Annual Income (k$)', hue='Gender', data=df, ax=axes[0])
axes[0].set_title('Age vs Annual Income (k$)')

sns.scatterplot(x='Age', y='Spending Score (1-100)', hue='Gender', data=df, ax=axes[1])
axes[1].set_title('Age vs Spending Score (1-100)')

sns.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)', hue='Gender', data=df, ax=axes[2])
axes[2].set_title('Annual Income (k$) vs Spending Score (1-100)')
plt.tight_layout()
st.pyplot(fig3)

# Elbow Method Plot
st.subheader("Elbow Method for Optimal K")
wcss = []
for i in range(1, 11):
    kmeans_elbow = KMeans(n_clusters=i, init='k-means++', random_state=42, n_init=10)
    kmeans_elbow.fit(X_scaled)
    wcss.append(kmeans_elbow.inertia_)

fig4, ax4 = plt.subplots(figsize=(8, 5))
ax4.plot(range(1, 11), wcss, marker='o', linestyle='--')
ax4.set_title('Elbow Method for Optimal K')
ax4.set_xlabel('Number of Clusters (K)')
ax4.set_ylabel('WCSS')
ax4.grid(True)
st.pyplot(fig4)
