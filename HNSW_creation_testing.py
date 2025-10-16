# implementing HNSW with FAISS. Evals based on book descriptions for 1000 random books

#imports and setup
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss #using cpu for support on all computer
from tqdm import tqdm
import os
import pickle


# read in books dataset (only title, authors, and description)
# note that we only care about ones with descriptions
cols_needed = ['Title','Authors','Description']
books_dataset = pd.read_csv('BooksDatasetClean.csv', usecols=cols_needed)

books_dataset = books_dataset[books_dataset['Description'].notna()] # filter by not null descriptions

print(books_dataset.head(1)) # print first item

# create hnsw faiss dataset (default params)
text_column = books_dataset['Description']
id_column = None
model_name = 'all-MiniLM-L6-v2'
batch_size = 64

# encode text in batches w progress 
model = SentenceTransformer(model_name)

texts = text_column.tolist() # convert to list
num_embeddings = len(texts)

embeddings = []

for i in tqdm(range(0, num_embeddings, batch_size), desc="Encoding descriptions into vectors..."):
    text_batch = texts[i:i+batch_size] # grab batch_size num texts
    embedding = model.encode(text_batch, convert_to_numpy=True, show_progress_bar=False)
    embeddings.append(embedding)
embeddings = np.vstack(embeddings).astype("float32") #encode into shape num_embeddings x embedding dimensions

# normalize for cosine similarity
faiss.normalize_L2(embeddings)

dims = embeddings.shape[1]

# create HNSW index + wrap wtih ID map to keep ids
M = 128 # NOTE param for HNSW -> number of neighbors in graph (typical range of 16-64)
ef_construction = 1000 # param of size of candidates for neighbors (larger = more accurate but slower and more memory)
# note that ef construction is effor put in when constructing

# create HNSW flat index (metric inner product for nomalized vctores = cosine similarity)
index = faiss.IndexHNSWFlat(dims, M, faiss.METRIC_INNER_PRODUCT)
index.hnsw.efConstruction = ef_construction
index.hnsw.efSearch = 50 # set efsearch for queries (tradeoff of speed vs accuracy) will be changed  later anyway

# map to int ids
index = faiss.IndexIDMap(index)

# prep ids 
if id_column and id_column in books_dataset:
    ids = books_dataset[id_column].to_numpy().astype("int64")
else:
    ids = np.arange(num_embeddings, dtype="int64")

# add vectors in single call (not large enough to justify chunks probably)
index.add_with_ids(embeddings, ids)



# save ind and metadata mapping
faiss.write_index(index, "books_dataset_cosine.index")

meta = books_dataset.copy()
meta["_faiss_id"] = ids
with open("hnsw_metadata.pkl", "wb") as f:
    pickle.dump(meta, f)

print(f"Index build with {num_embeddings} embeddings and {dims} dimensions")

