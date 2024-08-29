from pymilvus import connections, utility

# Function to remove the Milvus collection
def remove_milvus_type(collection_name, type_name):
    # Connect to Milvus
    connections.connect("default", host="localhost", port="19530")

    # Check if the collection exists
    if utility.has_collection(collection_name):
        # Drop the collection
        utility.drop_collection(collection_name)
        print(f"Collection '{collection_name}' has been successfully removed.")
    else:
        print(f"Collection '{collection_name}' does not exist.")

    # Disconnect from Milvus
    connections.disconnect("default")

if __name__ == "__main__":
    collection_name = "nsduh"
    type_name = "relationship"
    remove_milvus_type(collection_name)