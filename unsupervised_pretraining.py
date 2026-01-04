# training using triplet loss and Matryoshka embeddings
# starting model is "all-MiniLM-L6-v2"

#TODO: imports
import json
import os
from tqdm import tqdm
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

#TODO: collect book_ids that are non null in goodreads books + at least 14 reviews
books = os.path.join("./Datasets", "goodreads_books.json")
book_count = 2360655
reviews_path = os.path.join("./Datasets", "goodreads_reviews_dedup.json")
review_count = 9324443

# find all good book ids
good_book_ids = set()
with open(books, 'r') as books_data:
    for line in tqdm(books_data, total=book_count):
        try:
            line = json.loads(line)
            if not line['description']:
                continue
            else:
                good_book_ids.add(line['book_id'])
        except json.JSONDecodeError as e:
            print(f"Error: {e}")

# find books with a desc and at least 14 reviews
num_per_book_id = defaultdict(int)

with open(reviews_path, 'r') as reviews_data:
    for line in tqdm(reviews_data, total=review_count):
        try:
            line = json.loads(line)
            if not line['review_text']:
                continue
            else:
                b_id = line['book_id']
                if b_id in good_book_ids:
                    num_per_book_id[b_id] += 1

        except json.JSONDecodeError as e:
            print(f"Error: {e}")

# for each item in good_book_ids if num_per_book < 14 then remove
book_ids_large_group = set()
for b_id in tqdm(good_book_ids, total=len(good_book_ids)):
    if num_per_book_id[b_id] >= 14:
        book_ids_large_group.add(b_id)



#TODO: create dataloader (grabs up to 64 positives and 64+ negatives)
    # dict of b_id -> list -> dataloader wrapper
b_id_to_groups = defaultdict(list)

#TODO: updateme
# go thru every one of books and reviews and if in bidlarge group add to that
with open(books, 'r') as books_data:
    for line in tqdm(books_data, total=book_count):
        try:
            line = json.loads(line)
            if not line['description']:
                continue
            else:
                b_id = line['book_id']
                if b_id in b_id_to_groups

        except json.JSONDecodeError as e:
            print(f"Error: {e}")


with open(reviews_path, 'r') as reviews_data:
    for line in tqdm(reviews_data, total=review_count):
        try:
            line = json.loads(line)
            if not line['review_text']:
                continue
            else:
                b_id = line['book_id']
                if b_id in good_book_ids:
                    num_per_book_id[b_id] += 1

        except json.JSONDecodeError as e:
            print(f"Error: {e}")



#TODO: train with ohem (worst negatives and positives, batch size 128 -> useful samples (harder than a threshold)


    # after each round, lets validate with a val set 
        # make sure that same groups have higher similarity than random


#TODO: final testing set is the same
