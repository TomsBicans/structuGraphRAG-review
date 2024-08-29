import networkx as nx
import rdflib
from rdflib.namespace import RDF, RDFS, FOAF, Namespace
from rdflib import Graph, URIRef, Literal
from urllib.parse import quote

# Load the GraphML file
graph = nx.read_graphml('/Users/saillab/Shengting/graphrag-local-ollama/ragtest_alldata/output/20240730-145009/artifacts/embedded_graph.3.graphml')

# Initialize RDF graph
g = Graph()

# Define namespaces and prefixes
EX = Namespace("http://example.org/")
g.bind("foaf", FOAF)
g.bind("ex", EX)

def encode_uri(node):
    """ Encode the node to ensure it's a valid URI. """
    return URIRef(EX[quote(str(node))])

# Add nodes to RDF graph
for node in graph.nodes():
    node_uri = encode_uri(node)
    g.add((node_uri, RDF.type, FOAF.Person))
    # Add any additional node properties here, if available

# Add edges to RDF graph
for edge in graph.edges():
    subject = encode_uri(edge[0])
    predicate = FOAF.knows
    obj = encode_uri(edge[1])
    g.add((subject, predicate, obj))

# Serialize RDF graph to Turtle format
ttl_file_path = 'embedded_graph3.ttl'
g.serialize(destination=ttl_file_path, format='turtle')

print(f"RDF graph has been saved to {ttl_file_path}")
