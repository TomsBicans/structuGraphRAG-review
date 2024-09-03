# NSDUH RAG Construction

This repository contains scripts for constructing the NSDUH Retrieval-Augmented Generation (RAG) approach.

## Overview

The NSDUH data is categorized into three main components:

1. **Knowledge Description**: Extracted from the preface part of the codebook, which is unformatted literal text.
2. **Survey Question**: Extracted from the variable section of the codebook, organized in a table-like format.
3. **Answer**: Provided in the dataset, which is stored as a CSV file.

### Ontology Construction

The ontology of NSDUH is constructed based on the knowledge description and survey questions. The survey questions are used to generate entity and relationship triples, which are then utilized to build the ontology. Guided by the ontology, data from the dataset is extracted and used to populate the knowledge graph.

### RAG Query

RAG queries are categorized into two types:

- **Knowledge Query**: Queries the knowledge graph, which is built upon the ontology.
- **Data Query**: Queries the dataset, which is stored in a CSV file.

## Data Persistence

The data persistence layer is built using PostgreSQL and Milvus:

- **PostgreSQL**: Tables are created based on the ontology design. Relationships, such as incidents, have independent tables and contain references to entities defined in the entity tables.
- **Milvus**: Serves two functions—storing embeddings of entities and relationships as a vector database and performing efficient semantic searches.

## Folder and File Descriptions

- **db_model**: Contains class definitions for the tables in PostgreSQL, following the SQLAlchemy syntax.
- **nsduh_query**: The RAG query module, which includes classes for the knowledge query and data query.
- **milvus_load_kg**: Loads the knowledge graph into the Milvus vector database from the PostgreSQL database.

