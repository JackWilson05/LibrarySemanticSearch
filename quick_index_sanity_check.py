import os, pickle, faiss, numpy as np, pandas as pd

INDEX_PATH = "books_dataset_cosine.index"
META_PATH = "hnsw_metadata.pkl"

assert os.path.exists(INDEX_PATH), "index missing"
assert os.path.exists(META_PATH), "meta missing"

index = faiss.read_index(INDEX_PATH)
with open(META_PATH, "rb") as f:
    meta = pickle.load(f)

print("index type:", type(index))
print("index.ntotal:", index.ntotal)
print("meta rows:", len(meta))
print("_faiss_id present:", "_faiss_id" in meta.columns)
if "_faiss_id" in meta.columns:
    print("_faiss_id dtype:", meta["_faiss_id"].dtype)
    print("unique ids:", meta["_faiss_id"].nunique())
    print("min id, max id:", meta["_faiss_id"].min(), meta["_faiss_id"].max())
    print("duplicates in _faiss_id:", meta[meta.duplicated(subset=['_faiss_id'], keep=False)].shape[0])


from sentence_transformers import SentenceTransformer
import numpy as np
from itertools import islice

MODEL_NAME = "all-MiniLM-L6-v2"
model = SentenceTransformer(MODEL_NAME)

def search_text(text, k=5):
    q = model.encode([text], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(q)
    D, I = index.search(q, k)
    return D[0], I[0]

# choose sample row indices to test
sample_idxs = [0, len(meta)//2, len(meta)-1]
for i in sample_idxs:
    row = meta.iloc[i]
    fid = int(row["_faiss_id"])
    title = row.get("Title", "")
    authors = row.get("Authors", "")
    print(f"\nMETA ROW idx={i} _faiss_id={fid} -> {authors} - {title}")
    desc = row.get("Description", "")
    if not isinstance(desc, str) or desc.strip()=="":
        print("  description empty — skipping")
        continue
    D, I = search_text(desc, k=5)
    print("  returned ids:", I.tolist())
    # check if expected id present
    if fid in I:
        print("  OK: expected id found in returned ids (position)", list(I).index(fid))
    else:
        print("  NOT FOUND: expected id NOT in returned ids")
        # show top returned mapped to meta (if exist)
        for rid in I:
            if rid == -1: 
                print("    rid -1 (empty)")
                continue
            try:
                rrow = meta[meta["_faiss_id"]==int(rid)].iloc[0]
                print("    ->", int(rid), "=>", rrow.get("Authors",""), "-", rrow.get("Title",""))
            except IndexError:
                print("    ->", int(rid), "=> NOT FOUND IN METADATA")


import random
N = min(100, len(meta))
idxs = random.sample(range(len(meta)), N)
hits = 0
for i in idxs:
    row = meta.iloc[i]
    fid = int(row["_faiss_id"])
    desc = row["Description"]
    D, I = search_text(desc, k=1)
    if I[0] == fid:
        hits += 1
print(f"{hits}/{N} self-hits (top-1) for sampled rows")




ids_in_index = None
try:
    # common: IndexIDMap has .id_map attribute (list)
    ids_in_index = np.array(index.id_map).astype(int)
except Exception:
    try:
        # some wrappers: index.index.id_map
        ids_in_index = np.array(index.index.id_map).astype(int)
    except Exception:
        ids_in_index = None

print("ids_in_index available:", ids_in_index is not None)
if ids_in_index is not None:
    print("sample ids_in_index (first 20):", ids_in_index[:20])
    # compare sets
    meta_ids = np.array(meta["_faiss_id"].astype("int64"))
    print("meta ids: min/max/len:", meta_ids.min(), meta_ids.max(), len(meta_ids))
    print("index ids: min/max/len:", ids_in_index.min(), ids_in_index.max(), len(ids_in_index))
    print("meta - index = ", set(meta_ids) - set(ids_in_index))
    print("index - meta = ", set(ids_in_index) - set(meta_ids))





