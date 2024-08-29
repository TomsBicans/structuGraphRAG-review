import pandas as pd
from sqlalchemy import create_engine, text, Column, Integer, String, ForeignKey, BigInteger, ARRAY, Numeric, Boolean
from sqlalchemy.orm import sessionmaker, relationship,  declarative_base
import os
from openai import OpenAI
from dotenv import load_dotenv

from tqdm import tqdm


# Load from environment
load_dotenv('.env', override=True)
# NEO4J_URI = os.getenv('NEO4J_URI')
# NEO4J_USERNAME = os.getenv('NEO4J_USER')
# NEO4J_PASSWORD = os.getenv('NEO4J_PWD')
# NEO4J_DATABASE = os.getenv('NEO4J_DB')

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
class EntityNSDUH(Base):
    __tablename__ = 'entity_nsduh'
    id = Column(Integer, primary_key=True, autoincrement=True)
    entity_name = Column(String(256))
    variable_code = Column(String(32))
    entity_description = Column(String(2048))
    entity_name_embedding = Column(ARRAY(Numeric))
    entity_description_embedding = Column(ARRAY(Numeric))

df_entity = pd.read_csv("1_entities.csv", dtype=str)


# Iterate over the DataFrame and insert data into the database with tqdm progress bar
error_log = []
try:
    for index, row in tqdm(df_entity.iterrows(), total=df_entity.shape[0]):
        try:
            entity_name = row['entity'].strip()
            entity_description = row['description'].strip()
            variable_code = row['question_code']
            if isinstance(variable_code, str):
                variable_code = variable_code.strip()
            else:
                variable_code = ''

            entity_name_embedding = get_embeddings(entity_name)
            entity_description_embedding = get_embeddings(entity_description)

            new_entity = EntityNSDUH(
                entity_name=entity_name,
                variable_code=variable_code,
                entity_description=entity_description,
                entity_name_embedding=entity_name_embedding,
                entity_description_embedding=entity_description_embedding
            )
            
            session.add(new_entity)
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