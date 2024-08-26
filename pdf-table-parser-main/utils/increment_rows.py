import pandas as pd

# Define the starting values
start_value = 1748
start_weight = 163
end_value = 1951

# Generate the data
data = [f"V{start_value + i} - REPWT{start_weight + i}: Replicate Weight {start_weight + i}" for i in range(end_value - start_value + 1)]

# Create a DataFrame
df = pd.DataFrame(data, columns=["Replicate Weights"])

# Save to CSV
df.to_csv('replicate_weights.csv', index=False)
