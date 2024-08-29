import os
import uuid
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from pymilvus import connections, Collection
from sqlalchemy import create_engine, Column, Integer, String, func
from sqlalchemy.orm import sessionmaker, declarative_base, aliased
from sqlalchemy.dialects.postgresql import UUID
from db_models import *

import json
import re

# Load environment variables
load_dotenv('.env', override=True)

# OpenAI setup
# os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY_TEAM')
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY_PERSONAL')
OPENAI_ENDPOINT_EM = os.getenv('OPENAI_ENDPOINT_EM')
llm_model = "gpt-4o"
embedding_model = 'text-embedding-3-small'

client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])

# Database setup
engine = create_engine('your_db_connection_string')
Session = sessionmaker(bind=engine)
session = Session()

Base = declarative_base()
    
def parse_json(response):
    # Remove code block formatting if present
    response = response.strip('```json').strip('```')
    # Remove any leading or trailing whitespaces
    response = response.strip()
    # Replace any newline characters
    response = response.replace('\n', '')
    return response

# Function to get embeddings
def get_embeddings(text):
    response = client.embeddings.create(input=text, model=embedding_model)
    return response.data[0].embedding

from sqlalchemy import text

def preprocess_question(question):
    prompt = f"""Analyze the following question related to the National Survey on Drug Use and 
    Health (NSDUH) dataset:

    Question: {question}

    Categorize this question and determine if it requires querying the actual dataset or 
    if it can be answered using only the codebook knowledge.

    Please provide your analysis as a JSON object with the following structure:
    {{
        "reasoning": "Your explanation for the categorization",
        "category": "data_query" or "codebook_knowledge",    
        "entities": ["List of relevant entities mentioned in the question"],
        "relationships": ["List of relevant relationships mentioned in the question"],
        "requires_sql": true or false,
        "question_type": "Your assessment of the type of question (e.g., 'prevalence', 'trend', 'comparison', 'definition', etc.)"
    }}

    Note: 
    - "data_query" means the question requires querying the actual dataset.
    - "codebook_knowledge" means the question can be answered using only the codebook or metadata.
    - Entities in NSDUH typically include substances, demographics, or other measured variables.
    - Relationships typically involve connections between entities, such as usage patterns or correlations.
    - Set "requires_sql" to true if actual data needs to be retrieved from the database.

    Ensure your entire response is a valid JSON object."""

    response = client.chat.completions.create(
        model=llm_model,  # or whichever model you prefer
        messages=[
            {"role": "system", "content": "You are an AI assistant specializing in analyzing questions about the NSDUH dataset."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,  # Lower temperature for more consistent categorization
        max_tokens=500  # Adjust as needed
    )

    # Extract the generated answer
    answer_text = response.choices[0].message.content
    answer_text = parse_json(answer_text)

    # Parse the JSON response
    try:
        analysis = json.loads(answer_text)
    except json.JSONDecodeError:
        # If parsing fails, return an error message
        analysis = {
            "error": "Failed to generate a valid JSON response",
            "raw_response": answer_text
        }

    return analysis

# Function to retrieve content based on the list of ids and sources
def retrieve_content(session, id_source_list):
    contents = []
    for id_str, source in id_source_list:
        uuid_val = uuid.UUID(id_str) if isinstance(id_str, str) else id_str
        
        # Dynamically construct the query based on the source
        query = text(f"SELECT * FROM {source} WHERE id = :id")
        result = session.execute(query, {"id": uuid_val}).first()
        
        if result:
            # Convert result to a dictionary
            content = {}
            for column, value in result._mapping.items():
                content[column] = value
            content['source'] = source  # Add source to the content dictionary
            contents.append(content)
    
    return contents

# Function to search Milvus and retrieve related content
def search_and_retrieve_content(question, top_k=7):
    question_embedding = get_embeddings(question)
    question_em_np = np.array([question_embedding])
    question_em_np = question_em_np / np.linalg.norm(question_em_np, axis=1)[:, np.newaxis]

    search_params = {"metric_type": "IP", "params": {"nprobe": 10}}
    results = collection.search(
        # expr="type=='entity'",
        data=question_em_np,
        anns_field="embedding",
        param=search_params,
        limit=top_k,
        output_fields=["id", "type", "source"]
    )

    id_source_list = [(result.entity.get('id'), result.entity.get('source')) for result in results[0]]
    contents = retrieve_content(session, id_source_list)

    return results[0], contents

# Function to search Milvus and retrieve related content
def locate_relationship(question, top_k=7):
    question_embedding = get_embeddings(question)
    question_em_np = np.array([question_embedding])
    question_em_np = question_em_np / np.linalg.norm(question_em_np, axis=1)[:, np.newaxis]

    search_params = {"metric_type": "IP", "params": {"nprobe": 20}}
    # search_params = {"metric_type": "COSINE", "params": {"nprobe": 20}}
    results = collection.search(
        expr="type=='relationship'",
        data=question_em_np,
        anns_field="embedding",
        param=search_params,
        limit=top_k,
        output_fields=["id", "type", "source"]
    )

    id_source_list = [(result.entity.get('id'), result.entity.get('source')) for result in results[0]]
    contents = retrieve_content(session, id_source_list)

    return results[0], contents

# Main function to ask a question and get the answer
def ask_question(question):
    print(f"Question: {question}")
    search_results, contents = search_and_retrieve_content(question)

    print("\nRelevant content:")
    for result, content in zip(search_results, contents):
        print(f"ID: {content.get('id')}")
        print(f"Type: {result.entity.get('type')}")
        print(f"Source: {content.get('source')}")
        print(f"Distance: {result.distance}")
        
        # Determine which field to print based on the source
        if content['source'] == 'preface_nsduh':
            print(f"Title: {content.get('title', 'N/A')}")
        elif content['source'] == 'preface_content_nsduh':
            print(f"Content: {content.get('content', 'N/A')}")
        else:
            print(f"Content: {content}")  # Fallback for unknown sources
        
        print("---")
            # Generate answer using GPT
    answer = generate_answer(question, contents)

    print("\nGenerated Answer:")
    print(json.dumps(answer, indent=2))

    return answer

def query_and_rank_incidents(session, sit_id):
    try:
        # Alias for Substance table
        substance_alias = aliased(Substance)

        # Query to aggregate, rank, and get substance name
        results = session.query(
            SubstanceIncidentCase.substance_id,
            substance_alias.substance_name,
            func.count(SubstanceIncidentCase.substance_id).label('count')
        ).join(
            substance_alias, SubstanceIncidentCase.substance_id == substance_alias.id
        ).filter(
            SubstanceIncidentCase.sit_id == sit_id
        ).group_by(
            SubstanceIncidentCase.substance_id, substance_alias.substance_name
        ).order_by(
            func.count(SubstanceIncidentCase.substance_id).desc()
        ).all()
        
        # Convert results to a list of dictionaries
        results_dict = [
            {
                'uuid': result.substance_id,
                'name': result.substance_name,
                'count': result.count
            }
            for result in results
        ]
        
        # Display the results
        print("Ranking of substances for the given incident type:")
        for result in results_dict:
            print(f"Substance Name: {result['name']}, Substance ID: {result['uuid']}, Count: {result['count']}")
        
        return results_dict
    
    except Exception as e:
        print(f"Error querying database: {e}")



# Data function to ask a question and get the answer
def ask_data_question(pre, question):
    print(f"Question: {question}")
    search_results, contents = locate_relationship(question)

    print("\nRelevant content:")
    for result, content in zip(search_results, contents):
        print(f"ID: {content.get('id')}")
        print(f"Type: {result.entity.get('type')}")
        print(f"Source: {content.get('source')}")
        print(f"Distance: {result.distance}")
        
        # Determine which field to print based on the source
        if content['source'] == 'preface_nsduh':
            print(f"Title: {content.get('title', 'N/A')}")
        elif content['source'] == 'preface_content_nsduh':
            print(f"Content: {content.get('content', 'N/A')}")
        else:
            print(f"Content: {content}")  # Fallback for unknown sources
        
        print("---")
    # Generate answer using GPT

    id_list = locate_relationship_gpt(question, contents)#['e3b7abda-27be-4d52-aab0-1e0eccbc69c2']
    statistic_results = query_and_rank_incidents(session, id_list[0])
    answer = generate_answer_statistics(question, statistic_results)
    # answer = parse_json(answer)
    print("\nGenerated Answer:")
    print(json.dumps(answer, indent=2))

    return answer


def locate_relationship_gpt(question, candidate_relationships):
    # Prepare the list of candidate relationships for the prompt
    relationship_list = "\n".join([f"- {rel['id']}: {rel['sit_name']}" for rel in candidate_relationships])
    
    prompt = f"""Analyze the following question related to the National Survey on Drug Use and 
    Health (NSDUH) dataset and the list of candidate relationships:

    Question: {question}

    Candidate Relationships:
    {relationship_list}

    Identify the relationship(s) that are most directly relevant to answering this question. 
    Focus on the specific context of the question, considering any particular scenarios or 
    keywords mentioned, and select only the relationship(s) that directly address this context. 
    Avoid including relationships that are only broadly related or do not specifically align 
    with the question's focus.

    Please provide your analysis as a JSON object with the following structure:
    {{
        "reasoning": "Your explanation for selecting the most relevant relationship(s)",
        "relevant_relationship_ids": ["List of IDs for the most relevant relationship(s)"]
    }}

    Ensure your entire response is a valid JSON object, and prioritize selecting the relationship(s) that best match the specific context of the question."""


    response = client.chat.completions.create(
        model=llm_model,  # or whichever model you prefer
        messages=[
            {"role": "system", "content": "You are an AI assistant specializing in analyzing questions about the NSDUH dataset and identifying relevant relationships."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,  # Lower temperature for more consistent analysis
        max_tokens=500  # Adjust as needed
    )

    # Extract the generated answer
    answer_text = response.choices[0].message.content

    answer_text = parse_json(answer_text)

    # Parse the JSON response
    try:
        analysis = json.loads(answer_text)
        return analysis["relevant_relationship_ids"]
    except json.JSONDecodeError:
        # If parsing fails, return an error message
        return ["Error: Failed to generate a valid JSON response"]
    except KeyError:
        # If the expected key is missing, return an error message
        return ["Error: Unexpected response format"]

import json
from uuid import UUID

def convert_uuid(obj):
    if isinstance(obj, UUID):
        return str(obj)
    elif isinstance(obj, dict):
        return {convert_uuid(key): convert_uuid(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_uuid(element) for element in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_uuid(element) for element in obj)
    else:
        return obj

def generate_answer(question, contents):
    # Convert all UUID objects to strings
    converted_contents = convert_uuid(contents)
    
    # Prepare the context from the converted contents
    try:
        context = json.dumps(converted_contents, indent=2)
    except TypeError as e:
        print(f"Error in JSON serialization: {e}")
        print("Contents causing the error:")
        print(converted_contents)
        raise

    # Rest of the function remains the same
    prompt = f"""You are an expert on federal surveys and social science, with deep familiarity with 
    the National Survey on Drug Use and Health (NSDUH) dataset. Use the following information to 
    answer the question. Your response should be thorough and demonstrate your expertise.

    Question: {question}

    Relevant Information:
    {context}

    Please provide a comprehensive answer based on the given information. Structure your response as a JSON object with the following keys:
    - "answer": Your detailed response to the question
    - "confidence": Your confidence in the answer on a scale of 0 to 1
    - "reasoning": Your thought process in arriving at the answer
    - "additional_info": Any additional relevant information or caveats

    Ensure your entire response is a valid JSON object."""

    # Call the OpenAI API
    response = client.chat.completions.create(
        model=llm_model,  # or whichever model you prefer
        messages=[
            {"role": "system", "content": "You are an AI assistant specializing in federal surveys and social science, particularly knowledgeable about the NSDUH dataset."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.0,  # Lower temperature for more focused answers
        max_tokens=1000  # Adjust as needed
    )

    # Extract the generated answer
    answer_text = response.choices[0].message.content
    answer_text = parse_json(answer_text)

    # Parse the JSON response
    try:
        answer_json = json.loads(answer_text)
    except json.JSONDecodeError:
        # If parsing fails, return an error message in JSON format
        answer_json = {
            "error": "Failed to generate a valid JSON response",
            "raw_response": answer_text
        }

    return answer_json

def generate_answer_statistics_2(question, statistics):
    # Convert all UUIDs in statistics to strings if they are not already
    for stat in statistics:
        if isinstance(stat['uuid'], uuid.UUID):
            stat['uuid'] = str(stat['uuid'])

    # Prepare the context for the AI model by formatting the statistics
    stats_context = json.dumps(statistics, indent=2)
    
    # Prepare the prompt for the AI model
    prompt = f"""You are an expert on federal surveys and social science, with deep familiarity with 
    the National Survey on Drug Use and Health (NSDUH) dataset. Use the following information to 
    answer the question. Your response should be thorough and demonstrate your expertise.

    Question: {question}

    Relevant Statistics:
    {stats_context}

    Please provide a comprehensive answer based on the given information. Structure your response as a JSON object with the following keys:
    - "reasoning": Your thought process in arriving at the answer
    - "answer": Your detailed response to the question
    - "statistics_summary": A summary of the statistics provided
    - "confidence": Your confidence in the answer on a scale of 0 to 1
    - "additional_info": Any additional relevant information or caveats

    Ensure your entire response is a valid JSON object."""

    # Call the OpenAI API
    response = client.chat.completions.create(
        model=llm_model,  # or whichever model you prefer
        messages=[
            {"role": "system", "content": "You are an AI assistant specializing in federal surveys and social science, particularly knowledgeable about the NSDUH dataset."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.0,  # Lower temperature for more focused answers
        max_tokens=1000  # Adjust as needed
    )

    # Extract the generated answer
    answer_text = response.choices[0].message.content
    answer_text = parse_json(answer_text)

    # Parse the JSON response
    try:
        answer_json = json.loads(answer_text)
    except json.JSONDecodeError:
        # If parsing fails, return an error message in JSON format
        answer_json = {
            "error": "Failed to generate a valid JSON response",
            "raw_response": answer_text
        }

    return answer_json

def generate_answer_statistics(question, statistics):
    # Convert all UUIDs in statistics to strings if they are not already
    for stat in statistics:
        for key, value in stat.items():
            if isinstance(value, uuid.UUID):
                stat[key] = str(value)

    # Prepare the context for the AI model by formatting the statistics
    stats_context = json.dumps(statistics, indent=2)
    
    # Prepare the prompt for the AI model
    prompt = f"""You are an expert on federal surveys and social science, with deep familiarity with 
    the National Survey on Drug Use and Health (NSDUH) dataset. Use the following information to 
    answer the question. Your response should be thorough and demonstrate your expertise.

    Question: {question}

    Relevant Statistics:
    {stats_context}

    Please follow these steps to provide a comprehensive answer:
    1. Analyze the structure of the provided statistics. Identify the key fields (e.g., Substance Name, Count) and how they relate to the question.
    2. If the question asks about "most" or "least", sort the data based on the relevant numeric field (likely "Count") and consider the top or bottom 5 entries.
    3. For other types of questions, consider all provided statistics.
    4. Formulate your answer based on this analysis, making sure to discuss trends, patterns, or notable differences among the substances.
    5. If specific substances are mentioned in the question, ensure you address them directly in your answer.

    Structure your response as a JSON object with the following keys:
    - "answer": Your detailed response to the question, including discussion of top/bottom results if applicable
    - "statistics_summary": A summary of the key statistics, highlighting trends or patterns
    - "confidence": Your confidence in the answer on a scale of 0 to 1
    - "reasoning": Your thought process in arriving at the answer, including how you interpreted and sorted the data
    - "additional_info": Any additional relevant information or caveats

    Ensure your entire response is a valid JSON object."""

    # Call the OpenAI API
    response = client.chat.completions.create(
        model=llm_model,  # or whichever model you prefer
        messages=[
            {"role": "system", "content": "You are an AI assistant specializing in federal surveys and social science, particularly knowledgeable about the NSDUH dataset."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.0,  # Lower temperature for more focused answers
        max_tokens=1500  # Adjust as needed for comprehensive answers
    )

    # Extract the generated answer
    answer_text = response.choices[0].message.content
    answer_text = parse_json(answer_text)

    # Parse the JSON response
    try:
        answer_json = json.loads(answer_text)
    except json.JSONDecodeError:
        # If parsing fails, return an error message in JSON format
        answer_json = {
            "error": "Failed to generate a valid JSON response",
            "raw_response": answer_text
        }

    return answer_json


if __name__ == "__main__":
    # Connect to Milvus
    connections.connect("default", host="localhost", port="19530")
    collection = Collection("nsduh")
    collection.load()

    # Query preface data
    # preface_nsduh_data, preface_content_nsduh_data = query_preface_data(session)

    # Example usage
    # What is the most commonly encountered drug in early exposure cases?
    # question = "Which mathematic equation is used for the statistics of the NSDUH dataset?"
    # question = "Is there any gender-based analysis and conclusion in NSDUH dataset 2022?"
    # question = "What is the least used drug in the Prescription Drug Misuse incidents?"
    question = "In the 2022 National Survey on Drug Use and Health (NSDUH) dataset, what is the least used drug in the prescription drug misuse incidents?"
    # question = 'In the 2022 National Survey on Drug Use and Health (NSDUH) dataset, which drug is most commonly associated with vehicle-related illegal activities?'
    # question ="In the 2022 National Survey on Drug Use and Health (NSDUH) dataset, what are the top 2 drugs most commonly associated with vehicle-related illegal activities?"
    
    pre_category = preprocess_question(question)
    print(pre_category)

    category = pre_category.get('category')
    if category=='data_query':
        answer = ask_data_question(pre_category, question)
    elif category=='codebook_knowledge':
        answer = ask_question(question)
    # print(answer)

    # Close connections
    session.close()
    connections.disconnect("default")