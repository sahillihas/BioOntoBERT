# Onto2Sen Generating sentences for using Biomedical Ontologies

from owlready2 import *
from rdflib import Graph, URIRef
import csv
from tqdm import tqdm
import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Load the ontology
path_onto = "Ontology-path"
onto = get_ontology(path_onto).load()

# Initialize a list to store concept labels and synonyms
concept_synonyms = []

# Create an RDFLib graph and load the ontology into it
graph = Graph()
graph.parse(path_onto)

# Open the CSV file in write mode for writing annotation properties
with open('CSV2/Anatomy.csv', "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Concept label", "Property label", "Property"])

    # Iterate through all classes in the ontology to extract annotation properties
    for concept in tqdm(onto.classes(), total=len(list(onto.classes())), desc="Concepts"):
        concept_label = concept.label.first()

        for annotation_property in onto.annotation_properties():
            property_label = annotation_property.label.first()
            property_uri = URIRef(annotation_property.iri)

            query = graph.query(
                """
                SELECT ?value
                WHERE {
                    <""" + concept.iri + """> <""" + str(property_uri) + """> ?value .
                }
                """
            )

            for row in query:
                writer.writerow([concept_label, property_label, row[0].toPython()])

# Write subclass relationships to the CSV file
with open('CSV2/Anatomy.csv', "a", newline="") as file:
    writer = csv.writer(file)

    for subclass in tqdm(onto.classes(), total=len(list(onto.classes())), desc="Classes"):
        for sub in subclass.subclasses():
            writer.writerow([sub.label.first(), "is a", subclass.label.first()])

# Download necessary NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# Load CSV data into pandas DataFrame
path_csv = "CleanedData/merged.csv"
data = pd.read_csv(path_csv)
data = data[data['Concept label'].notna()]

# Clean and preprocess text in 'Property label' column
data['Property label'] = data['Property label'].apply(lambda x: x.lower())

# Save the preprocessed DataFrame to a new CSV file
merged_file_path = 'CleanedData/mergedv2.csv'
data.to_csv(merged_file_path, index=False)

# Perform text cleaning functions on data
def clean_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = re.sub('\s+', ' ', text)
    return text.strip()

def lowercase_text(text):
    return text.lower()

def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    tokens = nltk.word_tokenize(text)
    filtered_tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(filtered_tokens)

def lemmatize_text(text):
    lemmatizer = WordNetLemmatizer()
    tokens = nltk.word_tokenize(text)
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(lemmatized_tokens)

# Clean text from the text file
text_file = 'CleanedData/OntoData.txt'
with open(text_file, 'rt') as file:
    data = file.read()
    words = data.split()
    print('Number of words in text file:', len(words))
