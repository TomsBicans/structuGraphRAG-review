import pandas as pd
from sqlalchemy import create_engine, text, Column, Integer, String, ForeignKey, BigInteger, ARRAY, Numeric, Boolean, Sequence
from sqlalchemy.orm import sessionmaker, relationship, declarative_base
from sqlalchemy.dialects.postgresql import UUID
import uuid
import os
from openai import OpenAI
from dotenv import load_dotenv
import json
from db_models import *
import numpy as np
from pymilvus import FieldSchema, CollectionSchema, DataType, Collection
from pymilvus import connections, utility
import random
from tqdm import tqdm

load_dotenv('.env', override=True)

os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY_TEAM')
OPENAI_ENDPOINT_EM = os.getenv('OPENAI_ENDPOINT_EM')
llm_model = "gpt-4o"
embedding_model = 'text-embedding-3-small'

client = OpenAI(
    api_key= os.environ['OPENAI_API_KEY']
)

engine = create_engine('your_db_connection_string')
Session = sessionmaker(bind=engine)
session = Session()

# Function to get embeddings
def get_embeddings(text):
    response = client.embeddings.create(input=text, model=embedding_model)
    text_embedding = response.data[0].embedding
    return text_embedding


def query_data(session):
    # substance, incident type, topic, question, preface, preface content
    # Query all data from PrefaceNSDUH
    preface_nsduh_data = session.query(PrefaceNSDUH).all()
    # Query all data from PrefaceContentNSDUH
    preface_content_nsduh_data = session.query(PrefaceContentNSDUH).all()

    substance_data = session.query(Substance).all()
    substance_incident_type_data = session.query(SubstanceIncidentType).all()

    topic_data = session.query(TopicNSDUH).all()
    question_data = session.query(QuestionNSDUH).all()


    return preface_nsduh_data, preface_content_nsduh_data, substance_data, \
        substance_incident_type_data, topic_data, question_data



if __name__ == '__main__':
    # Connect to Milvus
    connections.connect("default", host="localhost", port="19530")

    # Define the updated schema for Milvus collection
    fields = [
        FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=36, is_primary=True),
        FieldSchema(name="type", dtype=DataType.VARCHAR, max_length=50),
        FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=50),  # New field
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1536)
    ]
    schema = CollectionSchema(fields, "Indexing for NSDUH preface, questions, substance entity, and incident")

    # Create a new Milvus collection with the updated schema
    collection = Collection("nsduh", schema)

    # Lists to hold data for insertion into Milvus
    ids = []
    types = []
    sources = []  # New list for sources
    embeddings = []

    preface_nsduh_data, preface_content_nsduh_data, substance_data, \
        substance_incident_type_data, topic_data, question_data = query_data(session)

    print("Begin preface_nsduh")
    # Traverse PrefaceNSDUH data and generate embeddings for titles
    for record in preface_nsduh_data:
        embedding = get_embeddings(record.title)
        ids.append(str(record.id))
        types.append("title")
        sources.append("preface_nsduh")  # Add source for title
        embeddings.append(embedding)

    print("Begin preface_content_nsduh")
    # Traverse PrefaceContentNSDUH data and generate embeddings for content
    for record in preface_content_nsduh_data:
        embedding = get_embeddings(record.content)
        ids.append(str(record.id))
        types.append("content")
        sources.append("preface_content_nsduh")  # Add source for content
        embeddings.append(embedding)

    print("Begin entity")

    for record in substance_data:
        embedding = get_embeddings(record.substance_name)
        ids.append(str(record.id))
        types.append("entity")
        sources.append("substance")  # Add source for content
        embeddings.append(embedding)

    print("Begin relationship for substance_incident_type")
    for record in substance_incident_type_data:
        embedding = get_embeddings(record.sit_name)
        ids.append(str(record.id))
        types.append("relationship")
        sources.append("substance_incident_type")  # Add source for content
        embeddings.append(embedding)

    print("Begin topic")
    for record in topic_data:
        embedding = get_embeddings(record.topic_description)
        ids.append(str(record.id))
        types.append("topic")
        sources.append("topic_nsduh")  # Add source for content
        embeddings.append(embedding)

    print("Begin question")
    for record in question_data:
        embedding = get_embeddings(record.question_content)
        ids.append(str(record.id))
        types.append("question")
        sources.append("question_nsduh")  # Add source for content
        embeddings.append(embedding)

    # Convert embeddings to numpy array
    embeddings = np.array(embeddings)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1)[:, np.newaxis]

    # Insert data into Milvus
    entities = [
        ids,       # UUIDs as strings
        types,     # Type of content ('title' or 'content')
        sources,   # New field for sources
        embeddings # Embeddings
    ]

    collection.insert(entities)

    # Create an index for the embeddings field
    index_params = {"index_type": "IVF_FLAT", "params": {"nlist": 128}, "metric_type": "IP"}
    collection.create_index(field_name="embedding", index_params=index_params)

    # Load the collection into memory for searching
    collection.load()

    print("Data has been successfully inserted into Milvus and indexed.")


