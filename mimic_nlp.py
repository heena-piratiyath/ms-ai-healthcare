import pandas as pd

#diagnoses_icd_df = pd.read_csv('/content/DIAGNOSES_ICD.csv.gz')
#diagnoses_icd_df.info()
#diagnoses_icd_df.iloc[0]
#print(len(diagnoses_icd_df))

#arr_subject_id=[]
#arr_hadm_id=[]
#for row in range(0, len(diagnoses_icd_df)):
#  if(diagnoses_icd_df.loc[row, 'ICD9_CODE']=='430'):
#    arr_subject_id.append(diagnoses_icd_df.loc[row, 'SUBJECT_ID'])

    # print(diagnoses_icd_df.loc[row, 'SUBJECT_ID'])
#print('length of array is:',len(arr_subject_id))

! pip install -U pip setuptools wheel

! pip install -U spacy

! python -m spacy download en_core_web_sm

! pip install scispacy

! pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.3/en_core_sci_sm-0.5.3.tar.gz

# upload file from local drive and open it for reading (NOTEEVENTS file)
from google.colab import files
uploaded = files.upload()

import pandas as pd

noteevents_df = pd.read_csv('99591_Sepsis_Notes.csv')

noteevents_df.info()
noteevents_df.iloc[0]
print(len(noteevents_df))

from google.colab import drive
drive.mount('/content/drive')

"""# **Data Loading and Preprocessing (Sepsis Notes)**

---


"""

import pandas as pd
import numpy as np
import spacy
import scispacy
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import torch
from transformers import BertTokenizer, BertModel

notes_df = pd.read_csv("99591_Sepsis_Notes.csv")

# Basic cleaning (lowercase)
notes_df['TEXT'] = notes_df['TEXT'].astype(str).str.lower()

print(len(notes_df.columns))

#print(notes_df.head())

print(notes_df['TEXT'])

! python -m spacy download en_core_web_lg # Download the en_core_web_lg model

"""# **SpaCy and SciSpaCy Entity Extraction**
---


"""

# Load spaCy  models
nlp_spacy = spacy.load('en_core_web_lg')

def extract_entities(text, nlp_model):
    doc = nlp_model(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

notes_df['SPACY_ENTITIES'] = notes_df['TEXT'].apply(
    lambda x: extract_entities(x, nlp_spacy))

print("Entity Extraction Examples:")
print(notes_df[['TEXT', 'SPACY_ENTITIES']].head())

!python -m spacy download en_core_web_md

"""# **Visualization of Medical Notes**




"""

import spacy
nlp_spacy = spacy.load('en_core_web_md')

# Extract the text notes from the DataFrame and convert them to a list
notes = noteevents_df['TEXT'].tolist()

doc = []
for i in range(len(notes)):
  doc.append(nlp_spacy(notes[i]))

from spacy import displacy
for i in range(len(doc)):
  displacy.render(doc[i], style="ent", jupyter=True)
  #\separator line
  print('--------------------------------------------------------------------------')

# Load spaCy and scispaCy models
nlp_scispacy = spacy.load('en_core_sci_sm')

def extract_entities(text, nlp_model):
    doc = nlp_model(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

notes_df['SCISPACY_ENTITIES'] = notes_df['TEXT'].apply(
    lambda x: extract_entities(x, nlp_scispacy))

print("Entity Extraction Examples:")
print(notes_df[['TEXT', 'SCISPACY_ENTITIES']].head())

# Load spaCy and scispaCy models
nlp_spacy = spacy.load('en_core_web_lg')
nlp_scispacy = spacy.load('en_core_sci_sm')


def extract_entities(text, nlp_model):
    doc = nlp_model(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities


# Apply entity extraction to the 'TEXT' column of the DataFrame
notes_df['SPACY_ENTITIES'] = notes_df['TEXT'].apply(
    lambda x: extract_entities(x, nlp_spacy))
notes_df['SCISPACY_ENTITIES'] = notes_df['TEXT'].apply(
    lambda x: extract_entities(x, nlp_scispacy))

# Print the complete entity extraction for each row
for index, row in notes_df.iterrows():
    print(f"Row {index}:")
    print("Text:", row['TEXT'])
    print("SpaCy Entities:", row['SPACY_ENTITIES'])
    print("SciSpaCy Entities:", row['SCISPACY_ENTITIES'])
    print("-" * 20)

!pip install gensim # Install the gensim library

"""# **Word2Vec Training and t-SNE (Word Embeddings)**

---


"""

# Tokenize for Word2Vec
from gensim.models import Word2Vec # Import Word2Vec from gensim.models

def tokenize(text):
    doc = nlp_spacy(text)
    return [token.text for token in doc if not token.is_punct and not token.is_space]

notes_df['TOKENS'] = notes_df['TEXT'].apply(tokenize)

# Train Word2Vec
sentences = notes_df['TOKENS'].tolist()
word2vec_model = Word2Vec(sentences, vector_size=100, window=5,
                          min_count=5, workers=4)
word2vec_model.train(sentences, total_examples=len(sentences), epochs=10)

# t-SNE Visualization (Word2Vec)
words = list(word2vec_model.wv.key_to_index)
vectors = [word2vec_model.wv[word] for word in words]
tsne = TSNE(n_components=2, random_state=42)
#vectors_tsne = tsne.fit_transform(vectors)

vectors_array = np.array(vectors) #This line converts the vectors list to numpy array

vectors_tsne = tsne.fit_transform(vectors_array) #Pass the numpy array to fit_transform


plt.figure(figsize=(12, 8))
plt.scatter(vectors_tsne[:, 0], vectors_tsne[:, 1])
for i, word in enumerate(words):
    if i < 100:  # Limit the number of words annotated for clarity
        plt.annotate(word, xy=(vectors_tsne[i, 0], vectors_tsne[i, 1]))
plt.title('t-SNE Visualization of Word2Vec Embeddings')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.show()

"""# **ClinicalBERT Embeddings and t-SNE (Note Embeddings)**

---


"""

# Load ClinicalBERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Function to get ClinicalBERT embeddings
def get_clinicalbert_embeddings(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True,
                      padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    # Use mean pooling to get a sentence-level embedding
    embeddings = outputs.last_hidden_state.mean(dim=1).numpy()
    return embeddings

# Generate embeddings
notes_df['CLINICALBERT_EMBEDDINGS'] = notes_df['TEXT'].apply(
    get_clinicalbert_embeddings)

# Prepare embeddings for t-SNE (ClinicalBERT)
embeddings_list = notes_df['CLINICALBERT_EMBEDDINGS'].tolist()
embeddings_matrix = [emb.flatten() for emb in embeddings_list]

embeddings_matrix = np.array(embeddings_matrix)

# Apply t-SNE (ClinicalBERT)
tsne = TSNE(n_components=2, random_state=42)
embeddings_tsne = tsne.fit_transform(embeddings_matrix)

# Plot t-SNE (ClinicalBERT)
plt.figure(figsize=(12, 8))
plt.scatter(embeddings_tsne[:, 0], embeddings_tsne[:, 1])
plt.title('t-SNE Visualization of ClinicalBERT Embeddings')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.show()

!pip install medspacy

!python -m spacy download en_core_web_lg

!pip install medspacy

!pip install --upgrade medspacy

import pandas as pd
import medspacy
from medspacy.custom_tokenizer import create_medspacy_tokenizer # Import create_medspacy_tokenizer
import medspacy
from medspacy.ner import TargetMatcher


# Load the English language model using spacy.load()
nlp = medspacy.load("en_core_web_sm")

# Get the custom tokenizer function
create_tokenizer = create_medspacy_tokenizer(nlp)

# Load MedspaCy
nlp_medspacy = medspacy.load()  #This line loads MedspaCy

notes_df = pd.read_csv("99591_Sepsis_Notes.csv")
def extract_medspacy_entities(text):
    doc = nlp_medspacy(text)
    return [(ent.text, ent.label_) for ent in doc.ents]

notes_df['MEDSPACY_ENTITIES'] = notes_df['TEXT'].apply(extract_medspacy_entities)

print("MedSpaCy Entity Extraction Examples:")
print(notes_df[['TEXT', 'MEDSPACY_ENTITIES']].head())
for index, row in notes_df.iterrows():
    print(f"Row {index}:")
    print("Text:", row['TEXT'])
    print("MedSpaCy Entities:", row['MEDSPACY_ENTITIES'])
    print("-" * 20)

def count_mentions(entities, target_entity):
    count = 0
    for ent, label in entities:
        if ent.lower() == target_entity.lower():
            count += 1
    return count

notes_df['hypotension_mentions'] = notes_df['SCISPACY_ENTITIES'].apply(
    lambda x: count_mentions(x, 'hypotension'))

print("Example Feature:")
print(notes_df[['TEXT', 'hypotension_mentions']].head())
