# models.py

from sqlalchemy import Column, Integer, String, ForeignKey, BigInteger, ARRAY, Numeric, Boolean
from sqlalchemy.orm import declarative_base
from sqlalchemy.dialects.postgresql import UUID
import uuid

Base = declarative_base()

class PersonNSDUH(Base):
    __tablename__ = 'person_nsduh'
    person_id = Column(BigInteger, primary_key=True, nullable=False)
    gender = Column(Integer)
    age_range_id = Column(Integer)
    rural_urban_id = Column(Integer)
    source_dataset = Column(Integer)
    year = Column(Integer)

class Substance(Base):
    __tablename__ = 'substance'
    order_number = Column(Integer, nullable=False)
    substance_code = Column(String(10), nullable=True)
    substance_name = Column(String(80), nullable=True)
    substance_schedule = Column(Integer, default=0, nullable=True)
    source_dataset = Column(Integer, nullable=True)
    year = Column(Integer, nullable=True)
    id = Column(UUID(as_uuid=True), primary_key=True, nullable=False)


class Gender(Base):
    __tablename__ = 'gender'
    gender_id = Column(Integer, primary_key=True, nullable=False)
    gender_name = Column(String(20))
    source_dataset = Column(Integer)
    year = Column(Integer)

class AgeRange(Base):
    __tablename__ = 'age_range'
    age_range_id = Column(Integer, primary_key=True, nullable=False)
    age_range_lvalue = Column(Integer)
    age_range_rvalue = Column(Integer)
    source_dataset = Column(Integer)
    year = Column(Integer)

class RuralUrban(Base):
    __tablename__ = 'rural_urban'
    rural_urban_id = Column(Integer, primary_key=True, nullable=False)
    rural_urban_code = Column(String(10))
    rural_urban_description = Column(String(120))
    year = Column(Integer)
    population = Column(Integer)
    state_code = Column(String(2))
    county_name = Column(String(40))
    rucc = Column(Integer)

class SubstanceIncidentCase(Base):
    __tablename__ = 'substance_incident_case'
    order_number = Column(BigInteger, nullable=False)
    person_id = Column(BigInteger, nullable=True)
    source_dataset = Column(Integer, nullable=True)
    year = Column(Integer, nullable=True)
    id = Column(UUID(as_uuid=True), primary_key=True, nullable=False)
    sit_id = Column(UUID(as_uuid=True), nullable=True)
    substance_id = Column(UUID(as_uuid=True), nullable=True)



class SubstanceIncidentType(Base):
    __tablename__ = 'substance_incident_type'
    order_number = Column(Integer, nullable=False)
    sit_name = Column(String(128), nullable=True)
    source_dataset = Column(Integer, nullable=True)
    year = Column(Integer, nullable=True)
    id = Column(UUID(as_uuid=True), primary_key=True, nullable=False)

class Dataset(Base):
    __tablename__ = 'dataset'
    dataset_id = Column(Integer, primary_key=True, nullable=False)
    dataset_name = Column(String(50))
    organization = Column(String(50))
    ontology_prefix = Column(String(100))
    dataset_description = Column(String(512))
    year = Column(Integer)

class State(Base):
    __tablename__ = 'state'
    state_id = Column(Integer, primary_key=True, nullable=False)
    state_name = Column(String(20))
    state_abbr = Column(String(2))
    fips = Column(String(2))

class County(Base):
    __tablename__ = 'county'
    county_id = Column(Integer, primary_key=True, nullable=False)
    county_name = Column(String(40))
    fips = Column(String(5))
    state_id = Column(Integer)
    state_abbr = Column(String(2))
    latitude = Column(Numeric(9, 6))
    longitude = Column(Numeric(9, 6))

class MHTreatmentProvider(Base):
    __tablename__ = 'mh_treatment_provider'
    tp_id = Column(Integer, primary_key=True, nullable=False)
    tp_name = Column(String(64))
    tp_name_sub = Column(String(64))
    address_line_1 = Column(String(128))
    address_line_2 = Column(String(128))
    city_id = Column(Integer)
    state_code = Column(String(2))
    zip_code = Column(String(5))
    phone = Column(String(32))
    intake1 = Column(String(32))
    intake2 = Column(String(32))
    intake1a = Column(String(32))
    intake2a = Column(String(32))
    latitude = Column(Numeric(9, 6))
    longitude = Column(Numeric(9, 6))

class MentalHealthIssue(Base):
    __tablename__ = 'mental_health_issue'
    mhi_id = Column(Integer, primary_key=True, nullable=False)
    mhi_name = Column(String(128))
    mhi_code = Column(String(128))
    mhi_description = Column(String(256))
    source_dataset = Column(Integer)
    year = Column(Integer)

class MentalHealthCase(Base):
    __tablename__ = 'mental_health_case'
    mhi_case_id = Column(BigInteger, primary_key=True)
    person_id = Column(BigInteger)
    mhi_id = Column(Integer)
    time_id = Column(Integer)
    description = Column(String(128))

class RUCC(Base):
    __tablename__ = 'rucc'
    rucc_id = Column(Integer, primary_key=True, nullable=False)
    rucc_code = Column(Integer)
    year = Column(Integer)
    description = Column(String(512))

class MHServiceCategory(Base):
    __tablename__ = 'mh_service_category'
    mhsc_id = Column(Integer, primary_key=True, nullable=False)
    mhsc_name = Column(String(64))
    mhsc_code = Column(String(16))
    mhsc_year = Column(Integer)

class MHService(Base):
    __tablename__ = 'mh_service'
    mhs_id = Column(Integer, primary_key=True, nullable=False)
    mhs_code = Column(String(16))
    mhs_name = Column(String(256))
    mhs_description = Column(String(2048))
    mhs_year = Column(Integer)
    mhsc_id = Column(Integer)

class City(Base):
    __tablename__ = 'city'
    city_id = Column(Integer, primary_key=True, nullable=False)
    city_name = Column(String(64))
    county_id = Column(Integer)
    state_id = Column(Integer)
    latitude = Column(Numeric(9, 6))
    longitude = Column(Numeric(9, 6))
    year = Column(Integer, default=2024)
    population = Column(Integer)
    ranking = Column(Integer)
    density = Column(Numeric(7, 1))
    incorporated = Column(Boolean)
    zips = Column(String(2048))

class TPServices(Base):
    __tablename__ = 'tp_services'
    id = Column(Integer, primary_key=True)
    tp_id = Column(Integer)
    mhs_id = Column(Integer)
    year = Column(Integer)

class MHServiceEmbedding(Base):
    __tablename__ = 'mh_service_embedding'
    id = Column(BigInteger, primary_key=True, nullable=False, autoincrement=True)
    mhs_id = Column(Integer)
    mhs_name_embedding = Column(ARRAY(Numeric))
    mhs_description_embedding = Column(ARRAY(Numeric))
    em_year = Column(Integer)

class CountyEmbedding(Base):
    __tablename__ = 'county_embedding'
    id = Column(BigInteger, primary_key=True, nullable=False, autoincrement=True)
    county_id = Column(Integer)
    county_state_embedding = Column(ARRAY(Numeric))
    em_year = Column(Integer)

class TopicNSDUH(Base):
    __tablename__ = 'topic_nsduh'
    order_number = Column(Integer, nullable=False)
    topic_description = Column(String(8000), nullable=True)
    dataset_id = Column(Integer, nullable=True)
    year = Column(Integer, nullable=True)
    id = Column(UUID(as_uuid=True), primary_key=True, nullable=False)

# Define the Topic Embedding model
class TopicNSDUHEmbedding(Base):
    __tablename__ = 'topic_nsduh_embedding'
    id = Column(Integer, primary_key=True, autoincrement=True)
    topic_id = Column(Integer)
    topic_description_embedding = Column(ARRAY(Numeric))

class QuestionNSDUH(Base):
    __tablename__ = 'question_nsduh'
    order_number = Column(Integer, nullable=False)
    question_content = Column(String(2048), nullable=True)
    question_code = Column(String(128), nullable=True)
    year = Column(Integer, nullable=True)
    dataset_id = Column(Integer, nullable=True)
    id = Column(UUID(as_uuid=True), primary_key=True, nullable=False)
    topic_id = Column(UUID(as_uuid=True), nullable=True)


# Define the chunk table model
class PrefaceChunkNSDUH(Base):
    __tablename__ = 'preface_chunk_nsduh'
    id = Column(Integer, primary_key=True, autoincrement=True)
    chunk_content = Column(String(2048))
    dataset = Column(Integer)
    year = Column(Integer)
    chunk_embedding = Column(ARRAY(Numeric))

class PrefaceNSDUH(Base):
    __tablename__ = 'preface_nsduh'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title = Column(String(1024))
    page_idx = Column(Integer)
    source_dataset = Column(Integer)
    year = Column(Integer)
    order_number = Column(Integer, nullable=False, unique=True)

class PrefaceContentNSDUH(Base):
    __tablename__ = 'preface_content_nsduh'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    content = Column(String(1000))
    page_idx = Column(Integer)
    title_id = Column(UUID(as_uuid=True))
    type = Column(String)
    order_number = Column(Integer, nullable=False, unique=True)