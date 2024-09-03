# Entity and Relationship Clustering Pipeline

This repository contains scripts and data for clustering extracted entities and relationships from a given text. Due to the fragmented and noisy nature of these extractions, clustering algorithms are used to group them into meaningful clusters. Additionally, categories or classes can be abstracted from these clusters.

## Overview

The pipeline operates on the core concepts distributed across the entire codebook, with clustering based on the semantic relationships between entities and relationships. The process is iterative, involving three main steps, which require some manual intervention:

1. **Embedding**: Convert entities and relationships into vector representations.
2. **Clustering**: Group the vectors into meaningful clusters based on their semantic similarity.
3. **Category Abstraction**: Abstract categories or classes from the clusters and determine whether to proceed to the next iteration.

## Usage

All scripts in this repository can be executed directly. The CSV files included are the outputs from the first iteration of the pipeline.


