# Loading FAISS index and metadata, running evaluation for top-1 and top-5 accuracy
# This code will:
# 1. check that the FAISS index file and metadata file exist
# 2. load them if present
# 3. encode the provided queries (desc strings) with SentenceTransformer
# 4. run searches (k=5), map returned ids to "Authors - Title" from metadata
# 5. compute top-1 and top-5 accuracy and print per-query results
# If the files are not present in this environment, the script will print instructions and the ready-to-run evaluation function.

#TODO: understand everything in here and clean it up to my style

import os
import pickle
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from itertools import combinations
from sklearn.metrics.pairwise import cosine_similarity
import random
import math



INDEX_PATH = "books_dataset_cosine.index"
META_PATH = "hnsw_metadata.pkl"
MODEL_NAME = "all-MiniLM-L6-v2"
K_RERANK = 1500 # how many neighbors searched before rerank
EF_SEARCH = 10000 # how many nodes explored at search time


#TODO: implement reranking for top 250 queries using traditional vector similarity

# The queries and expected author_title values (from the assistant's previous response)
queries_map = {
  0: {
    "desc": "poems written by a young, terminally ill child reflecting on life, hope, and family through simple but touching verses",
    "author_title": "'By Stepanek, Mattie J. T. - Journey Through Heartsongs'"
  },
  1: {
    "desc": "a Soviet-era novel about characters drawn to American jazz, memory, and cultural displacement",
    "author_title": "'By Aksyonov, Vassily; Heim, Michael Henry - In Search of Melancholy Baby'"
  },
  2: {
    "desc": "a playful book mixing humor and diet advice, suggesting how many calories one burns during certain romantic encounters",
    "author_title": "'By Smith, Richard - The Dieter\\'s Guide to Weight Loss During Sex'"
  },
  3: {
    "desc": "reporting and investigation into how dangerous germs can be weaponized, the U.S. efforts to study them, and the risks of bioterrorism",
    "author_title": "'By Miller, Judith; Engelberg, Stephen; Broad, William J. - Germs: Biological Weapons and America\\'s Secret War'"
  },
  4: {
    "desc": "a reader-friendly guide to the Bible that discusses how the historical context and ethical questions should shape how we read sacred texts",
    "author_title": "'By Gomes, Peter J. - The Good Book: Reading the Bible with Mind and Heart'"
  },
  5: {
    "desc": "a Southern memoir about growing up in poverty, family struggles, and finding meaning in hardship through storytelling",
    "author_title": "'By Bragg, Rick - All Over but the Shoutin\\''"
  },
  6: {
    "desc": "behind-the-scenes stories from someone who worked on Capitol Hill, exposing political maneuvering, compromises, and insider culture",
    "author_title": "'By Jackley, John L. - Hill Rat: Blowing the Lid Off Congress'"
  },
  7: {
    "desc": "an affectionate collection of essays about cats: their personalities, habits, and quirks, told by different cat lovers",
    "author_title": "'By Aymar, Brandt (EDT) - Personality of the Cat'"
  },
  8: {
    "desc": "a political critique arguing that a recent administration weakened national security by poor decisions and hidden agendas",
    "author_title": "'By Gertz, Bill - Betrayal: How the Clinton Administration Undermined American Security'"
  },
  9: {
    "desc": "a novel about memory, loss, and a pivotal summer that changes people’s lives, quietly introspective with emotional depth",
    "author_title": "'By Kay, Terry - Shadow Song'"
  },
  10: {
    "desc": "a history that explains how codebreaking and cryptography helped the Allies in World War II, showing how secret communications shaped battles",
    "author_title": "'By Haufler, Hervie - Codebreakers\\' Victory: How the Allied Cryptographers Won World War II'"
  },
  11: {
    "desc": "an anthology collecting new voices in literature — short fiction and poems from emerging authors across the country",
    "author_title": "'By Kulka, John (EDT); Danford, Natalie (EDT) - Best New American Voices 2003'"
  },
  12: {
    "desc": "short spiritual reflections inspired by monastic life, adapted to everyday living and mindful insight",
    "author_title": "'By Moore, Thomas - Meditations: On the Monk Who Dwells in Daily Life'"
  },
  13: {
    "desc": "a novel about a woman researching Jackie Kennedy, juggling personal relationships and academic pressures",
    "author_title": "'By Preston, Caroline - Jackie by Josie: A Novel'"
  },
  14: {
    "desc": "a spiritual story set in a troubled city where a gentle, healing figure arrives and gradually transforms lives through compassion",
    "author_title": "'By Girzone, Joseph F. - Joshua and the City'"
  },
  15: {
    "desc": "a book that explores medieval concepts of love and courtship, with translations and commentary on romance in the Middle Ages",
    "author_title": "'By Hopkins, Andrea - The Book of Courtly Love: The Passionate Code of the Troubadours'"
  },
  16: {
    "desc": "a philosophical/theological reflection on how much morality people can reasonably be expected to live up to, balancing ideals with human imperfections",
    "author_title": "'By Kushner, Harold S. - How Good Do We Have to Be? A New Understanding of Morality'"
  },
  17: {
    "desc": "a collection of first-person stories by women across America, ranging in time and place, highlighting personal, cultural, and historical experiences",
    "author_title": "'By Conway, Jill Ker (EDT) - Written by Herself: Autobiographies of American Women'"
  },
  18: {
    "desc": "a biography tracing the life and influence of a major American labor leader, focusing on union politics and social change",
    "author_title": "'By Robinson, Archie - George Meany And His Times: A Biography'"
  },
  19: {
    "desc": "a series of oral histories from everyday Americans sharing dreams, struggles, and reflections on work, identity, and resilience",
    "author_title": "'By Terkel, Studs - American Dreams: Lost & Found'"
  },
  20: {
    "desc": "a memoir blending humor, culture, and social commentary, written by a performer who reflects on identity, fame, and everyday absurdities",
    "author_title": "'By Bernhard, Sandra - Love, Love, and Love'"
  },
  21: {
    "desc": "an investigative book reconstructing a major disaster or crash, weaving technical detail, personal stories, and policy fallout",
    "author_title": "'By Goldman, Kevin - Conflicting Accounts: The Creation and Crash of...'"
  },
  22: {
    "desc": "a clinical yet readable guide to panic disorder, covering symptoms, treatments, and therapeutic strategies for both professionals and patients",
    "author_title": "'By Pollack, Mark H.; Rosenbaum, J. F. - Panic Disorder and Its Treatment (Medical Psychology)'"
  },
  23: {
    "desc": "a biography of King Henry VIII focusing on his political, religious, and marital decisions and how they shaped England",
    "author_title": "'By Scarisbrick, J. J. - Henry VIII (English Monarchs Series)'"
  },
  24: {
    "desc": "a book exploring how children develop moral sense, empathy, and responsibility, using real stories and psychological insight",
    "author_title": "'By Coles, Robert - The Moral Intelligence of Children'"
  }
}

# testing with journey through heartsongs and if wed wanted quit we would have raised goldfish
# Encode both sentences into embeddings
model = SentenceTransformer(MODEL_NAME)
s1 = "Mattie J. T. Stepanek takes us on a Journey Through Heartsongs with more of his moving poems. These poems share the rare wisdom that Mattie has acquired through his struggle with a rare form of muscular dystrophy and the death of his three siblings from the same disease. His life view was one of love and generosity and as a poet and a peacemaker, his desire was to bring his message of peace to as many people as possible."
s1_v2 = "Collects poems written by the eleven-year-old muscular dystrophy patient, sharing his feelings and thoughts about his life, the deaths of his siblings, nature, faith, and hope."
s2 = "Poems deal with family relationships, child development, aging, and the roles of parents and children"
emb1 = model.encode(s1, convert_to_numpy=True)
emb2 = model.encode(s2, convert_to_numpy=True)
emb1_v2 = model.encode(s1_v2, convert_to_numpy=True)

# Normalize (for cosine)
emb1 = emb1 / np.linalg.norm(emb1)
emb2 = emb2 / np.linalg.norm(emb2)
emb1_v2 = emb1_v2 / np.linalg.norm(emb1_v2)

# Cosine similarity = dot product of normalized vectors
cosine_sim_1_2 = np.dot(emb1, emb2)
cosine_sim_1v2_2 = np.dot(emb1_v2, emb2)
cosine_sim_1_1v2 = np.dot(emb1, emb1_v2)

print("Cosine similarity 1 and 2:", cosine_sim_1_2)
print("Cosine similarity 1_v2 and 2:", cosine_sim_1v2_2)
print("Cosine similarity 1 and 1v2:", cosine_sim_1_1v2)


def initial_eval_embedding_quality(model_name, texts):
  model = SentenceTransformer(model_name)

  # 1) Encode all query descriptions
  embs = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
  embs = embs / np.linalg.norm(embs, axis=1, keepdims=True)  # normalize for cosine

  # 2) Compute cosine similarity matrix (NxN)
  sim_matrix = cosine_similarity(embs)
  if len(texts) < 250:
    # --- Plot 1: Similarity heatmap (bottom triangle only) ---
    # create blue -> yellow colormap exactly
    cmap = mcolors.LinearSegmentedColormap.from_list("blue_yellow", ["#0b3d91", "#ffe066"])

    mask = np.triu(np.ones_like(sim_matrix, dtype=bool), k=1)  # mask upper triangle
    fig, ax = plt.subplots(figsize=(8, 8))
    # mask the upper triangle so only lower triangle + diag is visible
    masked = np.ma.masked_where(mask, sim_matrix)
    cax = ax.imshow(masked, cmap=cmap, vmin=0.0, vmax=1.0)
    cbar = plt.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Cosine similarity")
    ax.set_title("Embedding Similarity Heatmap (lower triangle)")
    ax.set_xticks(range(len(texts)))
    ax.set_yticks(range(len(texts)))
    ax.set_xticklabels(range(len(texts)), rotation=90)
    ax.set_yticklabels(range(len(texts)))
    plt.tight_layout()
    plt.show()

    # --- Plot 2: Distribution of correct vs incorrect similarities ---
  n = len(sim_matrix)
  incorrect_sims = [sim_matrix[i, j] for i, j in combinations(range(n), 2)]
  correct_sims = np.diag(sim_matrix)

  # Build dense common bins across [0,1]
  common_bins = np.linspace(0.0, 1.0, 51)

  plt.figure(figsize=(9, 4))
  counts_incorrect, bin_edges, _ = plt.hist(
      incorrect_sims, bins=common_bins, alpha=0.6,
      color="red", label="Incorrect pairs", edgecolor='none'
  )

  # plot correct_sims histogram or vline
  if np.ptp(correct_sims) < 1e-6:
      val = float(np.mean(correct_sims))
      max_h = counts_incorrect.max() if len(counts_incorrect) > 0 else 1
      plt.vlines(val, 0, max_h * 0.95, colors="green", linewidth=6,
                 label=f"Correct pairs (n={len(correct_sims)})")
      plt.xlim(0.0, 1.0)
      plt.ylim(0, max_h * 1.05)
  else:
      plt.hist(correct_sims, bins=common_bins, alpha=0.6,
               color="green", label="Correct pairs (self-sim)",
               edgecolor='none')

  # --- Add dashed red lines for right-tail significance thresholds ---
  p_vals = [0.1, 0.05, 0.01, 0.001, 0.0001, 0.00001, 0.000001]
  quantiles = [1 - p for p in p_vals]  # right-tail
  thresholds = np.quantile(incorrect_sims, quantiles)
  ymax = plt.gca().get_ylim()[1]
  for p, t in zip(p_vals, thresholds):
      plt.axvline(t, color='red', linestyle='--', linewidth=1)
      plt.text(t, ymax * 0.95, f"p={p}", rotation=90,
               va='top', ha='right', color='red', fontsize=8)

  plt.title("Distribution of Similarities (Correct = diagonal; Incorrect = off-diagonal)")
  plt.xlabel("Cosine similarity")
  plt.ylabel("Count")
  plt.legend()
  plt.tight_layout()
  plt.show()


texts = [v["desc"] for v in queries_map.values()]
initial_eval_embedding_quality(MODEL_NAME, texts)

# now trying with varying numbers of true descs 
# 100, 1000, 10000
cols_needed = ['Title','Authors','Description']
books_dataset = pd.read_csv('BooksDatasetClean.csv', usecols=cols_needed)

books_dataset = books_dataset[books_dataset['Description'].notna()] # filter by not null descriptions

texts = random.sample(books_dataset["Description"].to_list(), 100) # list, n
initial_eval_embedding_quality(MODEL_NAME, texts)

#skipping these bc alr ran and take a while
"""
texts = random.sample(books_dataset["Description"].to_list(), 1000) # list, n
initial_eval_embedding_quality(MODEL_NAME, texts)

texts = random.sample(books_dataset["Description"].to_list(), 2500) # list, n
initial_eval_embedding_quality(MODEL_NAME, texts)"""

# 1 in 10000 -> 63 % sim random, 1 in 100000 -> 75% random sim, 1 in 1000000 -> 95% 
# for dataset of 10,000,000 -> going to have 10 95% matches, and 100 75% matches







def normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = str(s).lower().strip()
    # remove punctuation that commonly differs (commas, periods, ellipses)
    import re
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s


def _parse_author_title(s: str):
    s = (s or "").strip().strip("'\"")
    # common format: "By X - Title" or "X - Title"
    if s.lower().startswith("by "):
        s2 = s[3:].strip()
    else:
        s2 = s
    parts = s2.split(" - ", 1)
    if len(parts) == 2:
        authors = parts[0].strip()
        title = parts[1].strip()
    else:
        # fallback: try comma separation or put whole in title
        authors = ""
        title = s2
    return authors, title

def run_evaluation(index_path=INDEX_PATH, meta_path=META_PATH, queries=queries_map, model_name=MODEL_NAME, k=K_RERANK):
    # check files exist
    if not os.path.exists(index_path) or not os.path.exists(meta_path):
        print("Index or metadata file not found in this environment.")
        print(f"Expected index at: {index_path} (exists={os.path.exists(index_path)})")
        print(f"Expected metadata at: {meta_path} (exists={os.path.exists(meta_path)})")
        print("\nBelow is the evaluation function you can run locally where the files exist. It will load the index and metadata and print top-1 and top-5 accuracy.\n")
        import inspect
        print(inspect.getsource(run_evaluation))
        return None

    # load index and metadata
    index = faiss.read_index(index_path)
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)

    # build mapping from faiss id -> "Authors - Title"
    if "_faiss_id" in meta.columns:
        id_to_row = meta.set_index("_faiss_id")
    else:
        # try to infer: if metadata used different col name
        id_to_row = meta.copy()
        if "_faiss_id" not in id_to_row.columns:
            raise ValueError("metadata does not contain _faiss_id column; cannot map ids to rows. Check the metadata file. Columns: " + ", ".join(id_to_row.columns))

        id_to_row = id_to_row.set_index("_faiss_id")

    def id_to_author_title(fid):
        try:
            row = id_to_row.loc[fid]
            authors = row.get("Authors", "")
            title = row.get("Title", "")
            return f"{authors} - {title}".strip()
        except Exception:
            return ""

    # load model and encode queries
    model = SentenceTransformer(model_name)
    q_texts = [queries[i]["desc"] for i in sorted(queries.keys())]
    q_expected = [queries[i]["author_title"] for i in sorted(queries.keys())]

    q_embs = model.encode(q_texts, convert_to_numpy=True, show_progress_bar=True).astype("float32")
    # normalize as index was built with normalized vectors for cosine-inner-product
    faiss.normalize_L2(q_embs)

    # ensure efSearch is set (try both ways)
    try:
        # if index is IDMap wrapper, inner index at .index
        if hasattr(index, "index") and hasattr(index.index, "hnsw"):
            index.index.hnsw.efSearch = EF_SEARCH
        elif hasattr(index, "hnsw"):
            index.hnsw.efSearch = EF_SEARCH
    except Exception:
        pass

    # ------------------- MINIMAL DIFF START: request larger candidate set and rerank exactly -------------------
    # request K_RERANK candidates from FAISS (approximate)
    D, I = index.search(q_embs, k)  # D shape: (nq, k), I shape: (nq, k)

    # helper to reconstruct a vector for a given external id (fid).
    # This tries a few approaches so it works with Index and IndexIDMap.
    def _reconstruct_vec_for_fid(fid):
        # try direct reconstruct (may work when index stored by external ids)
        try:
            vec = index.reconstruct(fid)
            return np.asarray(vec, dtype="float32")
        except Exception:
            pass
        # try mapping external id -> internal id for IndexIDMap
        try:
            inner = index.index if hasattr(index, "index") else index
            # index.id_map maps internal -> external id (faiss uses 'id_map' attr)
            if hasattr(index, "id_map"):
                # convert to numpy and find internal idx
                id_map = np.array(index.id_map)  # internal->external
                internal_indices = np.where(id_map == fid)[0]
                if internal_indices.size > 0:
                    internal = int(internal_indices[0])
                    vec = inner.reconstruct(internal)
                    return np.asarray(vec, dtype="float32")
        except Exception:
            pass
        # final fallback: try inner.reconstruct with fid (some builds keep external==internal)
        try:
            inner = index.index if hasattr(index, "index") else index
            vec = inner.reconstruct(fid)
            return np.asarray(vec, dtype="float32")
        except Exception:
            pass
        raise RuntimeError(f"Could not reconstruct vector for id {fid}")

    # cache reconstructed vectors to avoid repeated reconstruct calls
    vec_cache = {}

    # rerank top K_RERANK candidates exactly using dot product (vectors normalized => cosine)
    topk_final = 5  # final k we want to evaluate (top-5 and top-1)
    n_queries = I.shape[0]
    reranked_ids = np.full((n_queries, topk_final), -1, dtype=np.int64)
    reranked_sims = np.full((n_queries, topk_final), np.nan, dtype=np.float32)

    for qi in range(n_queries):
        cand_ids = I[qi].astype("int64")
        cand_ids = cand_ids[cand_ids != -1]
        if cand_ids.size == 0:
            continue

        # collect candidate embeddings (reconstruct)
        cand_vecs = []
        valid_cand_ids = []
        for cid in cand_ids:
            cid_int = int(cid)
            if cid_int in vec_cache:
                vec = vec_cache[cid_int]
            else:
                try:
                    vec = _reconstruct_vec_for_fid(cid_int)
                except RuntimeError:
                    # skip if cannot reconstruct
                    continue
                # ensure normalized (should be normalized already)
                vec = vec.astype("float32")
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec = vec / norm
                vec_cache[cid_int] = vec
            cand_vecs.append(vec)
            valid_cand_ids.append(cid_int)

        if len(valid_cand_ids) == 0:
            continue

        cand_embs = np.vstack(cand_vecs)  # (n_cand, dim)
        q_vec = q_embs[qi].astype("float32")
        q_vec = q_vec / np.linalg.norm(q_vec)

        # exact dot products (cosine)
        sims = cand_embs @ q_vec  # shape (n_cand,)
        order = np.argsort(-sims)[:topk_final]
        sel_ids = np.array(valid_cand_ids)[order]
        sel_sims = sims[order]

        n_sel = sel_ids.shape[0]
        reranked_ids[qi, :n_sel] = sel_ids
        reranked_sims[qi, :n_sel] = sel_sims
    # ------------------- MINIMAL DIFF END ----------------------------------------------------------------------

    # Use reranked_ids / reranked_sims for evaluation (instead of raw I/D)
    top1_correct = 0
    top5_correct = 0
    results = []

    for qi, expected in enumerate(q_expected):
        returned_ids = reranked_ids[qi].tolist()
        returned_titles = [id_to_author_title(int(x)) if x != -1 else "" for x in returned_ids]
        # normalize for comparison
        expected_norm = normalize_text(expected)
        returned_norms = [normalize_text(t) for t in returned_titles]

        is_top1 = (len(returned_norms) > 0 and expected_norm == returned_norms[0])
        is_top5 = expected_norm in returned_norms

        top1_correct += int(is_top1)
        top5_correct += int(is_top5)
        results.append({
            "query_index": qi,
            "expected": expected,
            "returned_top5": returned_titles,
            "top1_match": is_top1,
            "top5_match": is_top5
        })

    n = len(q_expected)
    top1_acc = top1_correct / n
    top5_acc = top5_correct / n

    print(f"Evaluated {n} queries. Top-1 accuracy: {top1_acc:.3f} ({top1_correct}/{n}). Top-5 accuracy: {top5_acc:.3f} ({top5_correct}/{n}).\n")
    # print per-query summary
    for r in results:
        print(f"Q{r['query_index']:02d} expected: {r['expected']}\n  Top5 returned: {r['returned_top5']}\n  Top1 match: {r['top1_match']}, Top5 match: {r['top5_match']}\n")

    # -------------------------------------------------------------------------
    # ADDITIONAL CSV-based top-1 similarity check (unchanged logic, but aligned to reranked top-1)
    # -------------------------------------------------------------------------
    books_df = books_dataset
    books_df['_norm_title'] = books_df['Title'].apply(normalize_text)
    books_df['_norm_authors'] = books_df['Authors'].apply(normalize_text)

    top1_sims_csv = []
    for qi in range(reranked_ids.shape[0]):
        top1_id = int(reranked_ids[qi, 0])
        if top1_id == -1:
            top1_sims_csv.append(math.nan)
            continue

        # get the returned title string from your metadata mapping function (same as used for printing)
        returned_title_str = id_to_author_title(top1_id)  # uses id_to_row built earlier
        authors_ret, title_ret = _parse_author_title(returned_title_str)
        norm_title_ret = normalize_text(title_ret)
        norm_authors_ret = normalize_text(authors_ret)

        found_row = None
        if books_df is not None:
            # exact title match using normalized title
            cand = books_df[books_df['_norm_title'] == norm_title_ret]
            if len(cand) == 1:
                found_row = cand.iloc[0]
            elif len(cand) > 1:
                # tiebreak by authors similarity (exact normalized match preferred)
                cand2 = cand[cand['_norm_authors'] == norm_authors_ret]
                if len(cand2) >= 1:
                    found_row = cand2.iloc[0]
                else:
                    # fallback: pick first candidate (could be refined)
                    found_row = cand.iloc[0]
            else:
                # fallback: try substring contains on title (looser match)
                cand_sub = books_df[books_df['_norm_title'].str.contains(norm_title_ret, na=False)]
                if len(cand_sub) >= 1:
                    # if multiple, prefer those matching authors
                    cand2 = cand_sub[cand_sub['_norm_authors'] == norm_authors_ret]
                    found_row = (cand2.iloc[0] if len(cand2) >= 1 else cand_sub.iloc[0])

        if found_row is None:
            # couldn't find in CSV — compute sim against metadata-desc (if available) or mark NaN
            try:
                # fallback to metadata description lookup (meta has the Description)
                meta_row = id_to_row.loc[top1_id]
                desc_text = meta_row.get("Description", "")
                if not isinstance(desc_text, str) or desc_text.strip() == "":
                    top1_sims_csv.append(math.nan)
                    continue
            except Exception:
                top1_sims_csv.append(math.nan)
                continue
        else:
            desc_text = found_row['Description']

        # encode the description and compute cosine similarity with query embedding
        desc_emb = model.encode([desc_text], convert_to_numpy=True, show_progress_bar=False).astype("float32")
        # normalize both desc_emb and q vector (q vectors were normalized earlier with faiss.normalize_L2)
        desc_emb = desc_emb / np.linalg.norm(desc_emb, axis=1, keepdims=True)
        q_vec = q_embs[qi].astype("float32")
        q_vec = q_vec / np.linalg.norm(q_vec)
        sim = float(np.dot(q_vec, desc_emb[0]))
        top1_sims_csv.append(sim)

    # quick stats and histogram (will plot after rest of evaluation if you prefer)
    top1_sims_arr = np.array(top1_sims_csv, dtype="float32")
    valid_mask = ~np.isnan(top1_sims_arr)
    print(f"Top-1 (query vs returned-description-from-CSV) computed for {valid_mask.sum()}/{len(top1_sims_arr)} queries")
    if valid_mask.sum() > 0:
        print("stats: mean={:.4f}, std={:.4f}, min={:.4f}, max={:.4f}".format(
            float(np.nanmean(top1_sims_arr)), float(np.nanstd(top1_sims_arr)),
            float(np.nanmin(top1_sims_arr)), float(np.nanmax(top1_sims_arr))
        ))

    # Overlay plot: green filled histogram (CSV sims) + red dots (FAISS reported top-1 sims)
    csv_vals = top1_sims_arr[valid_mask]
    faiss_vals_all = np.asarray(D[:, 0], dtype="float32")  # FAISS reported similarity for its top candidate
    faiss_vals = faiss_vals_all[valid_mask]  # align to CSV-computed indices

    # Compute histogram bins and counts for green bars
    n_bins = 20
    counts, bins = np.histogram(csv_vals, bins=n_bins, range=(0.0, 1.0))
    bin_width = bins[1] - bins[0]
    bin_centers = bins[:-1] + bin_width / 2.0

    plt.figure(figsize=(9, 4))
    plt.bar(bin_centers, counts, width=bin_width * 0.98, color="green", alpha=0.6, edgecolor="k",
            label="CSV-desc sims (query vs CSV description)")

    # place FAISS red dots close to the top of their bins for visibility
    bin_idx = np.clip(np.digitize(faiss_vals, bins) - 1, 0, len(counts) - 1)
    rng = np.random.default_rng(0)
    jitter = (rng.random(len(faiss_vals)) - 0.5) * (counts.max() * 0.05) if len(faiss_vals) > 0 else np.array([])
    y_positions = counts[bin_idx] * 0.9 + jitter
    plt.scatter(faiss_vals, y_positions, color="red", s=30, zorder=5, label="FAISS top1 sims (reported)")

    plt.title("Query vs TOP-1 description similarity: CSV (green bars) + FAISS (red dots)")
    plt.xlabel("Cosine similarity")
    plt.ylabel("Count")
    plt.xlim(0.0, 1.0)
    plt.legend(loc="upper left")
    plt.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    plt.show()

    # return structured results
    return {"n": n, "top1_correct": top1_correct, "top5_correct": top5_correct, "top1_acc": top1_acc, "top5_acc": top5_acc, "per_query": results}


# Run evaluation (will either execute or print the runnable function if files are missing)
eval_results = run_evaluation()
print(f"Top 1: {eval_results["top1_acc"]}, Top 5: {eval_results["top5_acc"]}")

