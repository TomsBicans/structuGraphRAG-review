# Ontology construction and data instance filtering

The extracted entity and relation from NSDUH is used here to construct ontology by leveraging the hierachy class structure in the codebook. The data instance will be mapped to and fetched into database laster. 

## Files and Scripts

### [Extracted_entity_and_relation_to_ontology.ipynb](Extracted_entity_and_relation_to_ontology.ipynb)
This script used generated entity and relation from GPT model to form the ontology by leveraging the hierachy class structure in the codebook.

### [Filter_none_reponse_ang_get_data_instance.ipynb](Filter_none_reponse_ang_get_data_instance.ipynb)
This script used to filter out data instances with none response, and prepare for knowledge graph construction.
