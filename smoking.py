import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage

# Load the dataset
data = pd.read_csv('smoking.csv')

"""
The purpose of this python file is to develop an unsupervised learning model to glean 
relevant information regarding smokers. 
"""

# Exploratory data analysis
print(data.dtypes)
print(data.isna().sum())

# Validate NaN values
print(data['smoke'].unique())
nonsmoker_rows = data[data['smoke'] == 'No']
nonsmoker_nan_summary = nonsmoker_rows[['amt_weekends', 'amt_weekdays', 'type']].isna().sum()
print(nonsmoker_nan_summary)

print(data['type'].unique())

# Treat missing values by filling them with 0
data.fillna(0, inplace=True)
print(data.isna().sum())

# --- Start of Unsupervised Learning Model Development ---

# Select relevant features for clustering (numerical features only)
numeric_columns = data.select_dtypes(include=[np.number]).columns
df_numeric = data[numeric_columns]

# Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_numeric)

# Step 1: K-Means Clustering
# Determine the optimal number of clusters using the Elbow Method
inertia = []
cluster_range = range(1, 11)

for k in cluster_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_data)
    inertia.append(kmeans.inertia_)

# Plot the Elbow Method
plt.figure(figsize=(8, 5))
plt.plot(cluster_range, inertia, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal Clusters')
plt.show()

# Apply K-Means with the optimal number of clusters (e.g., choose from elbow plot)
optimal_clusters = 3
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
kmeans_labels = kmeans.fit_predict(scaled_data)

# Add cluster labels to the original dataset
data['kmeans_cluster'] = kmeans_labels

# Step 2: Hierarchical Clustering
# Perform hierarchical clustering using Ward's method
linked = linkage(scaled_data, method='ward')

# Plot the dendrogram
plt.figure(figsize=(10, 7))
dendrogram(linked, truncate_mode='lastp', p=12, leaf_rotation=45, leaf_font_size=10, show_contracted=True)
plt.title('Dendrogram for Hierarchical Clustering')
plt.xlabel('Data Points')
plt.ylabel('Euclidean Distance')
plt.show()

# Apply Agglomerative Clustering
hierarchical = AgglomerativeClustering(n_clusters=optimal_clusters, linkage='ward')
hierarchical_labels = hierarchical.fit_predict(scaled_data)

# Add hierarchical cluster labels to the dataset
data['hierarchical_cluster'] = hierarchical_labels

# Step 3: Visualize Clusters using PCA
pca = PCA(n_components=2)
pca_data = pca.fit_transform(scaled_data)

plt.figure(figsize=(8, 5))
sns.scatterplot(x=pca_data[:, 0], y=pca_data[:, 1], hue=kmeans_labels, palette='viridis', s=100, alpha=0.7)
plt.title('K-Means Clustering Visualization (PCA)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Cluster')
plt.show()

# Analyze and interpret clusters
print(data[['kmeans_cluster', 'hierarchical_cluster', 'smoke', 'amt_weekends', 'amt_weekdays', 'type']].head())

# Save the data with cluster labels for further analysis
data.to_csv('smoking_with_clusters.csv', index=False)

from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy.stats import f_oneway
import numpy as np

# Perform PCA to understand feature importance
pca = PCA(n_components=scaled_data.shape[1])  # Keep all components
pca_data_full = pca.fit_transform(scaled_data)

# Explained variance ratio
explained_variance = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

# Display explained variance for each component
print("Explained Variance Ratio (PCA):", explained_variance)
print("Cumulative Variance:", cumulative_variance)

# Select the top 2 components for visualization
pca_top = PCA(n_components=2)
pca_data_top = pca_top.fit_transform(scaled_data)

# Check feature importance by analyzing PCA loadings
pca_loadings = pd.DataFrame(pca_top.components_, columns=df_numeric.columns, index=['PC1', 'PC2'])
print("\nPCA Loadings (Feature Importance):")
print(pca_loadings)

# Verify cluster separation using silhouette score for K-means
silhouette_avg = silhouette_score(scaled_data, kmeans_labels)
print("\nSilhouette Score (K-Means):", silhouette_avg)

# Perform ANOVA to check if clusters are significantly different across key features
anova_results = {}
for feature in df_numeric.columns:
    grouped_data = [df_numeric.loc[data['kmeans_cluster'] == cluster, feature] for cluster in range(optimal_clusters)]
    anova_results[feature] = f_oneway(*grouped_data)

# Display ANOVA results
anova_summary = {k: (v.statistic, v.pvalue) for k, v in anova_results.items()}
anova_df = pd.DataFrame.from_dict(anova_summary, orient='index', columns=['F-statistic', 'p-value'])
anova_df = anova_df.sort_values(by='p-value')

print("\nANOVA Results (Feature Differentiation by Cluster):")
print(anova_df)

# Visualize the cumulative explained variance to confirm how many components are needed
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Cumulative Explained Variance by PCA Components')
plt.axhline(y=0.90, color='r', linestyle='--', label='90% Variance')
plt.legend()
plt.grid(True)
plt.show()

# Visualize PCA feature loadings
plt.figure(figsize=(10, 5))
sns.barplot(x=pca_loadings.columns, y=pca_loadings.loc['PC1'], color='skyblue')
plt.title('Feature Importance (PCA Loadings - PC1)')
plt.xlabel('Features')
plt.ylabel('Contribution to PC1')
plt.grid(True)
plt.show()

# Final output of important test results
anova_df.head(), explained_variance[:3], silhouette_avg

