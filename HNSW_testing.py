# implementing HNSW with FAISS. Evals based on book descriptions for 1000 random books

#imports and setup
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss #using cpu for support on all computer
from tqdm import tqdm
import os



# read in books dataset (only title, authors, and description)
# note that we only care about ones with descriptions
cols_needed = ['Title','Authors','Description']
books_dataset = pd.read_csv('BooksDatasetClean.csv', usecols=cols_needed)

books_dataset = books_dataset[books_dataset['Description'].notna()] # filter by not null descriptions

print(books_dataset.head(1)) # print first item

# create hnsw faiss dataset (default params)



# read in test data (id, queries) where id is title + author

# evaluate on test data (top 5 accuracy, top 1 accuracy) 
