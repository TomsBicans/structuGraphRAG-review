import pandas as pd

# Load the CSV file
df = pd.read_csv('2_dbscan_clustered_relationships.csv')

# Filter out noise (DBSCAN label -1)
non_noise_df = df[df['dbscan_labels'] != -1]
noise_df = df[df['dbscan_labels'] == -1]

# Calculate the number of clusters
num_clusters = non_noise_df['final_cluster'].nunique()

# Calculate the number of relationships in each cluster
cluster_sizes = non_noise_df['final_cluster'].value_counts().reset_index()
cluster_sizes.columns = ['cluster_id', 'num_relationships']

# Sort clusters by size
sorted_clusters = cluster_sizes.sort_values(by='num_relationships', ascending=False)

# Create a new DataFrame to store the cluster and relationship information
clustered_relationships = non_noise_df[['id', 'final_cluster', 'relation_name', 'relation_description']]

# Rename columns for better clarity
clustered_relationships.rename(columns={'final_cluster': 'cluster_id'}, inplace=True)

# Sort the DataFrame by cluster_id to group relationships by cluster
# Ensure clusters are sorted by size
clustered_relationships['cluster_size'] = clustered_relationships['cluster_id'].map(sorted_clusters.set_index('cluster_id')['num_relationships'])
clustered_relationships_sorted = clustered_relationships.sort_values(by=['cluster_size', 'cluster_id'], ascending=[False, True])

# Drop the cluster_size column as it is not needed in the output CSV
clustered_relationships_sorted = clustered_relationships_sorted.drop(columns=['cluster_size'])

# Save the clustered relationships to a new CSV file
clustered_relationships_sorted.to_csv('final_clustered_relationships_4.csv', index=False)

# Save the noise DataFrame to a new CSV file
noise_df.to_csv('noise_relationships_df.csv', index=False)

# Print the number of clusters and the number of relationships in each cluster
print(f"Number of clusters: {num_clusters}")
print(f"Relationships per cluster:\n{sorted_clusters}")

# Save the cluster sizes to a new CSV file
sorted_clusters.to_csv('relationship_cluster_sizes_4.csv', index=False)

# Print confirmation
print("Clustered relationships have been saved to 'final_clustered_relationships_4.csv'")
print("Cluster sizes have been saved to 'relationship_cluster_sizes_4.csv'")
print("Noise relationships have been saved to 'noise_relationships_df.csv'")
