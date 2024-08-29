import pandas as pd
import numpy as np
from sqlalchemy import create_engine, Column, Integer, String, ARRAY, Numeric
from sqlalchemy.orm import sessionmaker, declarative_base
from openai import OpenAI
from dotenv import load_dotenv
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
import os

# Load from environment
load_dotenv('.env', override=True)
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY_TEAM')
OPENAI_ENDPOINT_EM = os.getenv('OPENAI_ENDPOINT_EM')
embedding_model = 'text-embedding-3-small'

client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])

# Database setup
engine = create_engine('your-database-connection-string')
Session = sessionmaker(bind=engine)
session = Session()
Base = declarative_base()

# Function to get embeddings
def get_embeddings(text):
    response = client.embeddings.create(input=text, model=embedding_model)
    text_embedding = response.data[0].embedding
    return text_embedding

# Define the ORM class for the table
class RelationshipNSDUH(Base):
    __tablename__ = 'relationship_nsduh'
    id = Column(Integer, primary_key=True, autoincrement=True)
    relation_name = Column(String(256))
    relation_description = Column(String(2048))
    relation_name_embedding = Column(ARRAY(Numeric))
    relation_description_embedding = Column(ARRAY(Numeric))

# Read and preprocess data
relationships = session.query(RelationshipNSDUH).all()
data = [{
    'id': r.id,
    'relation_name': r.relation_name,
    'relation_description': r.relation_description,
    'relation_name_embedding': np.array(r.relation_name_embedding, dtype=float),
    'description_embedding': np.array(r.relation_description_embedding, dtype=float)
} for r in relationships]

df = pd.DataFrame(data)

# Combine name and description embeddings
df['combined_embedding'] = df.apply(lambda row: np.concatenate([row['relation_name_embedding'], row['description_embedding']]), axis=1)

# Normalize the combined embeddings
scaler = StandardScaler()
normalized_embeddings = scaler.fit_transform(np.vstack(df['combined_embedding']))

# Apply DBSCAN for initial clustering
dbscan = DBSCAN(eps=0.5, min_samples=3, metric='cosine')
df['dbscan_labels'] = dbscan.fit_predict(normalized_embeddings)

# Function to determine optimal number of clusters using silhouette score
def optimal_num_clusters(data, max_k, output_file='rel_silhouette_scores.csv'):
    silhouette_scores = []
    k_values = range(2, max_k+1)
    
    for k in k_values:
        clusterer = AgglomerativeClustering(n_clusters=k, metric='cosine', linkage='average')
        cluster_labels = clusterer.fit_predict(data)
        silhouette_avg = silhouette_score(data, cluster_labels, metric='cosine')
        silhouette_scores.append(silhouette_avg)
    
    # Save silhouette scores to a CSV file
    silhouette_data = pd.DataFrame({
        'Number of Clusters': k_values,
        'Silhouette Score': silhouette_scores
    })
    silhouette_data.to_csv(output_file, index=False)
    print(f"Silhouette scores saved to {output_file}")
    
    # Plot silhouette scores
    plt.plot(k_values, silhouette_scores, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score for Optimal k')
    plt.show()
    
    # Find the optimal number of clusters
    optimal_k = silhouette_scores.index(max(silhouette_scores)) + 2
    return optimal_k

# Determine optimal number of clusters
max_k = min(50, len(df))  # Adjust max_k as needed
optimal_k = optimal_num_clusters(normalized_embeddings, max_k)

# Apply Agglomerative Clustering with the optimal number of clusters
agg_clustering = AgglomerativeClustering(n_clusters=optimal_k, metric='cosine', linkage='average')
df['agg_labels'] = agg_clustering.fit_predict(normalized_embeddings)

# Combine DBSCAN and Agglomerative Clustering results
df['final_cluster'] = df.apply(lambda row: f"DBSCAN_{row['dbscan_labels']}_AGG_{row['agg_labels']}", axis=1)

# Print cluster summary
print(df.groupby('final_cluster').agg({
    'relation_name': 'count',
    'relation_name': lambda x: ', '.join(x.head(5))  # Show top 5 relationships in each cluster
}))

# Save results
df.to_csv('clustered_relationships.csv', index=False)

# Visualize hierarchical clustering
linkage_matrix = linkage(normalized_embeddings, method='average', metric='cosine')
plt.figure(figsize=(10, 7))
dendrogram(linkage_matrix)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Sample Index')
plt.ylabel('Distance')
plt.show()
