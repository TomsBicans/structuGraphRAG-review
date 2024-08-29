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
engine = create_engine('your_database_connection_string')
Session = sessionmaker(bind=engine)
session = Session()
Base = declarative_base()

# Function to get embeddings
def get_embeddings(text):
    response = client.embeddings.create(input=text, model=embedding_model)
    text_embedding = response.data[0].embedding
    return text_embedding

# Define the ORM class for the table
class EntityNSDUH(Base):
    __tablename__ = 'entity_nsduh'
    id = Column(Integer, primary_key=True, autoincrement=True)
    entity_name = Column(String(256))
    variable_code = Column(String(32))
    entity_description = Column(String(2048))
    entity_name_embedding = Column(ARRAY(Numeric))
    entity_description_embedding = Column(ARRAY(Numeric))

# Read and preprocess data
entities = session.query(EntityNSDUH).all()
data = [{
    'id': e.id,
    'entity_name': e.entity_name,
    'entity_description': e.entity_description,
    'entity_name_embedding': np.array(e.entity_name_embedding, dtype=float),
    'description_embedding': np.array(e.entity_description_embedding, dtype=float)
} for e in entities]

df = pd.DataFrame(data)

# Combine name and description embeddings
df['combined_embedding'] = df.apply(lambda row: np.concatenate([row['entity_name_embedding'], row['description_embedding']]), axis=1)

# Normalize the combined embeddings
scaler = StandardScaler()
normalized_embeddings = scaler.fit_transform(np.vstack(df['combined_embedding']))

# Apply DBSCAN for initial clustering
dbscan = DBSCAN(eps=0.5, min_samples=3, metric='cosine')
df['dbscan_labels'] = dbscan.fit_predict(normalized_embeddings)

# Function to determine optimal number of clusters using silhouette score
def optimal_num_clusters(data, max_k):
    silhouette_scores = []
    for k in range(2, max_k+1):
        clusterer = AgglomerativeClustering(n_clusters=k, metric='cosine', linkage='average')
        cluster_labels = clusterer.fit_predict(data)
        silhouette_avg = silhouette_score(data, cluster_labels, metric='cosine')
        silhouette_scores.append(silhouette_avg)
    
    # Plot silhouette scores
    plt.plot(range(2, max_k+1), silhouette_scores, marker='o')
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
    'entity_name': 'count',
    'entity_name': lambda x: ', '.join(x.head(5))  # Show top 5 entities in each cluster
}))

# Visualize hierarchical clustering
linkage_matrix = linkage(normalized_embeddings, method='average', metric='cosine')
plt.figure(figsize=(10, 7))
dendrogram(linkage_matrix)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Sample Index')
plt.ylabel('Distance')
plt.show()

# Save results
df.to_csv('clustered_entities.csv', index=False)
