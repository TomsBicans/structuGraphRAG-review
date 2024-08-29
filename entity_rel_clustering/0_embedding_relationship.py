import pandas as pd
from sqlalchemy import create_engine, text, Column, Integer, String, ARRAY, Numeric
from sqlalchemy.orm import sessionmaker, declarative_base
import os
from openai import OpenAI
from dotenv import load_dotenv

from tqdm import tqdm


# Load from environment
load_dotenv('.env', override=True)
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY_TEAM')
OPENAI_ENDPOINT_EM = os.getenv('OPENAI_ENDPOINT_EM')
llm_model = "gpt-4o"
embedding_model = 'text-embedding-3-small'

client = OpenAI(
    api_key= os.environ['OPENAI_API_KEY']
)

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
class RelationshipNSDUH(Base):
    __tablename__ = 'relationship_nsduh'
    id = Column(Integer, primary_key=True, autoincrement=True)
    relation_name = Column(String(256))
    source_entity_name = Column(String(256))
    target_entity_name = Column(String(256))
    source_entity_id = Column(Integer)
    target_entity_id = Column(Integer)
    variable_code = Column(String(256))
    relation_description = Column(String(2048))
    relation_description_embedding = Column(ARRAY(Numeric))
    relation_name_embedding = Column(ARRAY(Numeric))

df_relationship = pd.read_csv("1_relationships.csv", dtype=str)

# Iterate over the DataFrame and insert data into the database with tqdm progress bar
error_log = []
try:
    for index, row in tqdm(df_relationship.iterrows(), total=df_relationship.shape[0]):
        try:
            relation_name = row['relationship'].strip()
            source_entity_name = row['source_entity'].strip()
            target_entity_name = row['target_entity'].strip()
            variable_code = row['question_code']
            if isinstance(variable_code, str):
                variable_code = variable_code.strip()
            else:
                variable_code = ''
            relation_description = row['description'].strip()

            relation_name_embedding = get_embeddings(relation_name)
            relation_description_embedding = get_embeddings(relation_description)

            new_relationship = RelationshipNSDUH(
                relation_name=relation_name,
                source_entity_name=source_entity_name,
                target_entity_name=target_entity_name,
                variable_code=variable_code,
                relation_description=relation_description,
                relation_name_embedding=relation_name_embedding,
                relation_description_embedding=relation_description_embedding
            )
            
            session.add(new_relationship)
            session.commit()  # Commit each row individually

        except Exception as e:
            session.rollback()
            error_log.append((index, row, str(e)))
            raise  # Re-raise the exception to stop the process

except Exception as e:
    print(f"Stopped processing due to an error: {e}")

# Print out the log of errors
if error_log:
    print("\nError Log:")
    for error in error_log:
        print(f"Row {error[0]}: {error[1]} - Error: {error[2]}")

print("CSV file has been loaded into the PostgreSQL database table with errors logged.")
