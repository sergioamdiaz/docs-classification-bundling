#*******************************************************************************
# IMPORTS:
#*******************************************************************************

import os, re, json, yaml, argparse
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict
from pprint import pprint
from typing import Literal

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

SEED = 42
np.random.seed(SEED)

#-------------------------------------------------------------------------------
# Constants for functions
#-------------------------------------------------------------------------------

# Which file extensions will be considered.
ALLOWED_EXTS = {".pdf", ".txt"}

# NOTE: These constants can be added in the future to the config file. For now they are hardcoded here for simplicity.

#*******************************************************************************
# DATA LOADING AND PREPROCESSING:
#*******************************************************************************

def load_config(path: str) -> dict:
    """ Loads the configuration from a YAML file. The file must contain the following keys:
    - data_dir: str, path to the directory containing the documents to classify.
    - descriptions_file: str, path to the JSON file with document-type descriptions.
    - output_dir: str, path to the directory where the output will be saved. """
    
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"The file was not found at {path}. Create a YAML config file with the required keys."
        )
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    # validate config structure
    required_keys = {"data_dir", "descriptions_file", "output_dir"}
    if not isinstance(config, dict) or not required_keys.issubset(config["paths"].keys()):
        raise ValueError(f"YAML file must contain the following keys in 'paths': {required_keys}")
    return config

#-------------------------------------------------------------------------------

def load_doc_type_descriptions(path: Path) -> dict[str, list[str]]:
    """ Loads the document-type descriptions from a JSON file. The file must 
    be a dict where the keys are document types and the values are lists of descriptions. """
    
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"The file was not found at {path}. Create the JSON with descriptions by type."
        )
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # validate documents-type description structure
    if not isinstance(data, dict) or not all(isinstance(v, list) for v in data.values()):
        raise ValueError("JSON file structure must be dict[str, list[str]]")
    return data

def list_files(root_dir: Path, allowed_exts: set) -> list[Path]:
    """ Returns a list of paths to all the allowed files in the given directory. """
    files = []
    for root, _, filenames in os.walk(root_dir):
        for fn in filenames:
            abs_path = Path(root) / fn
            if abs_path.suffix.lower() in allowed_exts:
                files.append(abs_path)
    files.sort()
    return files

#-------------------------------------------------------------------------------
# Extracting Pages from a Document:

# Read .txt files:
def _read_txt(path: Path) -> str:
    """ Reads a text file and returns its content as a string. """
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

# PDF: Uses PyMuPDF if it's installed:
# - PyMuPDF understand the PDF as a graphic object (positions, bounding boxes, images, layouts, etc.).
def _read_pdf(path: Path) -> list[str]:
    """ Reads a PDF file and returns its pages as a list of strings. """
    try:
        import pymupdf
    except ImportError:
        raise ImportError("Install PyMuPDF: pip install pymupdf")

    doc = pymupdf.open(path)
    pages = []
    for i in range(doc.page_count):
        txt = doc.load_page(i).get_text("text")
        pages.append(txt or "")
    doc.close()
    return pages # Each page will be just a list of strings.

# If the file is .txt, split into “pages” (chunks); if .pdf, extract real pages with the function above:
def extract_pages(path: Path) -> list[str]:
    """ Reads a file (pdf, txt) and returns its pages as a list of strings """
    ext = path.suffix.lower()
    if ext == ".txt":
        # Simulated "pages”: splits them by amount of chars.
        txt = _read_txt(path)
        # split naive: every ~2000 chars = "a page”
        chunk_size = 2000
        return [txt[i:i+chunk_size] for i in range(0, len(txt), chunk_size)] or [""]
    
    elif ext == ".pdf":
        return _read_pdf(path)
    else:
        return []

#-------------------------------------------------------------------------------
def _clean_text(s: str) -> str:
    ''' Cleans text by removing null characters and extra whitespace. Replace them with a single spaces.'''
    s = s.replace("\x00", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _clean_pages(pages: list[str]) -> list[str]:
    ''' Clean the pages of a text document by applying _clean_text.'''
    return [_clean_text(p) for p in pages]

#-------------------------------------------------------------------------------
# Building the dataset of pages:
@dataclass
class PageRecord:
    doc_path: Path
    doc_name: str
    page_idx: int
    text: str
    
def build_page_records(paths: list[Path]) -> list[PageRecord]:
    records = []
    for path in paths:
        pages = _clean_pages(extract_pages(path))
        doc_name = Path(path).stem # Gets filename without extension
        for i, p in enumerate(pages):
            records.append(PageRecord(doc_path=path, doc_name=doc_name, page_idx=i+1, text=p))
    return records

#*******************************************************************************
# DEFINING EMBEDDING MODEL
#*******************************************************************************
class Embedder:
    """ Base class. Does not execute any logic, just defines the arquitecture for the 
    embedder (a contract). Defines a public API. The actual logic will be implemented 
    in the child class."""
    def embed(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError

class SentenceTransformerEmbedder(Embedder):
    """ Inherits the Embedder class following the contract. Implements the logic of the embedder.""" 
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", batch_size: int = 32):
        
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError("Install sentence-transformers: pip install sentence-transformers")
        
        self.model = SentenceTransformer(model_name)
        self.batch_size = batch_size

    def embed(self, texts: list[str]) -> np.ndarray:
        # The tokenization is handled internally with the encode() method of the model, we just pass the list of strings
        emb = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            normalize_embeddings=True # we ask for normalized vectors. Helps with cosine similarity
        )
        return np.asarray(emb, dtype=np.float32)

#-------------------------------------------------------------------------------
# Training function
def train_embedder(page_records: list[PageRecord], embedder: Embedder) -> np.ndarray:
        texts = [r.text for r in page_records]
        page_embs = embedder.embed(texts)
        return page_embs

#*******************************************************************************
# SEGMENTATION FUNCTION
#*******************************************************************************

# Auxiliary function for the no backtracking segmentation function.
def _next_usable_labels(df: pd.DataFrame, start_idx: int, k: int) -> list[str]:
    """
    Returns up to k labels of usable pages. Looking forward from start_idx (excluding start_idx).
    """
    labels = []
    for j in range(start_idx + 1, len(df)): # +1 because we want the upcoming pages. Don't care about the current one.
        if df.loc[j, "usable"]:
            labels.append(df.loc[j, "kmeans_label"])
            if len(labels) >= k:
                break
    return labels

# Segmentation function with no backtracking and lookahead agreement:
def _segment_labels_no_backtracking(df_doc: pd.DataFrame, 
                                   lookahead_usable: int = 2, 
                                   min_agreement: int = 2,
                                   keep_original_for_unusable: bool = False,
                                   forbid_return_to_used_labels: bool = True) -> pd.DataFrame:
    """
    Creates a column 'segment_label' where:
      - There is no coming back to previous labels (contiguous segments).
      - Unusable pages do not trigger changes.
      - A change is only confirmed if there is support in upcoming usable pages.
      
    Input: DataFrame where each row is a page of the same document.
    Output: Returns the same DataFrame with a new column 'segment_label'.

    The 'keep_original_for_unusable' argument:
      - When False: unusable pages inherit current_label (useful for continuous segmentation).
      - When True: unusable pages conserve their original label, but still do not trigger changes.
    """
    
    df = df_doc.sort_values("page_idx").reset_index(drop=True).copy() # Just a sanity check, pages should already be in order.

    # Find initial label: first usable.
    current_segm_label: None | str = None # This variable holds the current segment-label.
    
    for i in range(len(df)):
        if df.usable[i]:
            current_segm_label = df["kmeans_label"][i]
            break

    # If there are no usable pages, do nothing. Most likely, the code will not reach this point.
    if current_segm_label is None:
        df["segment_label"] = df["kmeans_label"]
        df["segment_change_at"] = False
        return df

    used_labels: set[str] = {current_segm_label}
    segmented: list[str] = []
    changed_flags: list[bool] = []

    """ This is where the segmentation logic happens: """
    for i in range(len(df)):
        usable = df.usable[i]
        orig_label = df["kmeans_label"][i] # Original label of the currently inspected page.

        # Not usable page:
        if not usable:
            # Doesn't trigger any changes.
            if keep_original_for_unusable: # User preference. False by default.
                segmented.append(orig_label)
            else:
                segmented.append(current_segm_label)
            changed_flags.append(False)
            continue

        # Usable page from now on:
        if orig_label == current_segm_label: # The page's label matches the current segment label.
            segmented.append(current_segm_label)
            changed_flags.append(False)
            continue
        
        # If coming back to used labels is forbidden, label with the current segment label.
        if forbid_return_to_used_labels and orig_label in used_labels:
            segmented.append(current_segm_label)
            changed_flags.append(False)
            continue

        # Here we find how many upcoming usable pages are in the lookahead window
        future = _next_usable_labels(df, i, lookahead_usable) # list of the upcoming usable labels.

        # - If there is not enough usable pages ahead, we can:
        #   (a) Do not change (conservative) or (b) change anyway. Here I do it conservative.

        # Enter this 'if' if at least one page in the window is not usable.
        if len(future) < lookahead_usable: # Is telling that all of the labels need to be usable to consider a change.
            segmented.append(current_segm_label) # No changes, keep current label.
            changed_flags.append(False)
            continue
        
        # NOTE: So far it is been requested for all the pages in the window to be usable to consider a change.
        # This can be adjusted for example by using a bigger window and a accepting some unusable pages.
        
        """How many pages in the future match the label of the currently inspected page:"""
        agreement = sum(1 for lab in future if lab == orig_label) # Generates a list of 1s ([1,1,...]) and sum them all.

        if agreement >= min_agreement:
            # Change confirmed
            current_segm_label = orig_label
            used_labels.add(current_segm_label) # Mark this label as used.
            segmented.append(current_segm_label)
            changed_flags.append(True)
        else:
            # We treat it as noise, keep previous label
            segmented.append(current_segm_label)
            changed_flags.append(False)

    df["segment_label"] = segmented
    df["segment_change_at"] = changed_flags
    return df

#*******************************************************************************
# K-MEANS CLUSTERING
#*******************************************************************************
def build_type_centroids(desc_by_type: dict[str, list[str]], 
                         embedder: Embedder) -> tuple[list[str], np.ndarray]:
    """ Builds a centroid vector for each document type, by embedding their descriptions 
    and averaging them. Returns the list of type names and the array of centroids."""
    type_names = []
    centroid_vecs = []
    
    for keys, desc_list in desc_by_type.items():
        desc_list = [d for d in desc_list if d and d.strip()] # filter empty descriptions and checks if after stripping is still not empty.      
        if not desc_list:
            continue
        
        vecs = embedder.embed(desc_list) # embed all descriptions for this type. Returns array of shape (num_descriptions, 384)
        centroid = vecs.mean(axis=0) # mean along all descriptions to get only one vector per type.
        centroid = centroid / (np.linalg.norm(centroid) + 1e-12) # normalize again just in case. 1e-12 is to avoid division by zero.
        type_names.append(keys) # add the name of the type described.
        centroid_vecs.append(centroid)
        
    return type_names, np.vstack(centroid_vecs)

def cluster_pages_kmeans_seeded( page_embs: np.ndarray,
                                type_centroids: np.ndarray,
                                random_state: int = 42,
                                max_iter: int = 300) -> tuple[np.ndarray, object]:
    """
    Clusteriza páginas usando KMeans, inicializando centros con los centroides de doc-types.
    Retorna cluster_id por página (0..k-1).
    """
    try:
        from sklearn.cluster import KMeans
    except ImportError:
        raise ImportError("Install scikit-learn: pip install scikit-learn")
    
    
    k = type_centroids.shape[0]
    km = KMeans( n_clusters=k,
                init=type_centroids,   # seeded
                n_init=1,              # important: no re-initialize, we want to keep the seeded centroids as they are.
                max_iter=max_iter,
                random_state=random_state,
                algorithm="lloyd" )
    
    cluster_ids = km.fit_predict(page_embs)
    return cluster_ids, km

def _normalize_centroids(cluster_centroids: np.ndarray) -> np.ndarray:
    """ Normalize cluster centroids. This is necessary for cosine similarity. """
    return cluster_centroids / (np.linalg.norm(cluster_centroids, axis=1, keepdims=True) + 1e-12)

#-------------------------------------------------------------------------------
# Mapping clusters to types:
def map_clusters_to_types( cluster_centroids: np.ndarray,
                          type_centroids: np.ndarray,
                          type_names: list[str] ) -> tuple[dict[int, str], pd.DataFrame]:
    """
    returns:
      - mapping: cluster_id -> doc_type
      - df_scores: Similarity between cluster x type. How close is each cluster to each type?
    """
    sims = cluster_centroids @ type_centroids.T  # Shape should be k*k for k = # of types
    df_scores = pd.DataFrame(sims, columns=type_names)
    df_scores.insert(0, "cluster_id", np.arange(len(cluster_centroids)))

    mapping = {}
    for i in range(len(cluster_centroids)):
        row = df_scores.loc[i, type_names]
        if row.isna().all():
            mapping[i] = "unknown"
        else:
            mapping[i] = row.idxmax()
    return mapping, df_scores

#-------------------------------------------------------------------------------
# Fixxing overlapping clusters

def _find_overlapping_clusters( mapping: dict[int, str] ) -> dict[str, list[int]]:
    """ returns a dict where the keys are the doc-types and the values are the list of overlapping clusters. """
    groups = defaultdict(list) # dict of lists

    for cluster_id, types, in mapping.items():
        # The keys now are the doc-types. Every time the same doc-type is found, it is added to the list.
        groups[types].append(cluster_id)
        
    return {v: ks for v, ks in groups.items() if len(ks) > 1} # filter out singletons.

def _hungarian_remapping(scores: pd.DataFrame, mapping: dict[int, str]) -> dict[int, str]:
    """ Uses the Hungarian algorithm to maximes the similarity between clusters and doc-types JUST WHERE CLUSTERS OVERLAPPED. """
    new_mapping = mapping.copy()
    # Each v is one list of overlapping clusters
    for v in _find_overlapping_clusters(mapping).values():
        # sub matrix with only overlapping clusters
        # - 'cluster_id' column is dropped to have a square matrix
        # - inplace = False -> returns a copy
        # - reset_index(drop=True) -> because 'cluster_id' was dropped
        sub_scores = scores.drop(columns='cluster_id', inplace=False).reset_index(drop=True).iloc[v,v]
        
        print('\nSub matrix of overlapping scores:\n')
        print(sub_scores)
        
        rows, cols = linear_sum_assignment(sub_scores, maximize=True)

        for i, j in zip(sub_scores.index, cols):
            new_mapping[int(i)] = sub_scores.columns[j]

    return new_mapping

#*******************************************************************************
# RESULTS DATAFRAME
#*******************************************************************************

def _build_pages_df(page_records: list[PageRecord]) -> pd.DataFrame:
    """Returns a DataFrame where each row is a page, with its document name, path, page index and text)."""
    df_pages = pd.DataFrame({
        "doc_name": [r.doc_name for r in page_records],
        "doc_path": [r.doc_path for r in page_records],
        "page_idx": [r.page_idx for r in page_records],
        "text_len": [len(r.text) for r in page_records]
    })
    return df_pages

def _extend_df_labels(df_pages: pd.DataFrame, cluster_ids: np.ndarray, mapping: dict[int, str]) -> pd.DataFrame:

    df_pages["cluster_id"]   = cluster_ids
    df_pages["kmeans_label"] = df_pages["cluster_id"].map(mapping)
    df_pages["usable"]       = (df_pages["text_len"] > 50)

    df_pages_segm = (df_pages
                     .groupby("doc_path", group_keys=False)
                     .apply(lambda g: _segment_labels_no_backtracking(g))
                     )
               
    return df_pages_segm

def summary_df(page_records: list[PageRecord], cluster_ids: np.ndarray, mapping: dict[int, str]) -> pd.DataFrame:
    
    df_pages = _build_pages_df(page_records) # name, path, page index and text.
    df_pages_segm = _extend_df_labels(df_pages, cluster_ids, mapping) # labels, no backtracking.
    
    # "first and last page" DataFrame
    df_pages_segm_summary = (df_pages_segm
                                .groupby(["doc_path", "segment_label"], as_index=False, sort=False) # Keep keys as columns
                                .agg(first_page=("page_idx", "min"),
                                    last_page=("page_idx", "max"),
                                    n_pages=("page_idx", "count"))
                                )
    # Add a column with the total number of pages in the document:
    df_pages_segm_summary["total_pages_doc"] = (df_pages_segm_summary
                                                    .groupby("doc_path")["last_page"]
                                                    .transform("max")
                                                    )
    df_pages_segm_summary["label_ratio"] = (df_pages_segm_summary.n_pages / 
                                                    df_pages_segm_summary.total_pages_doc
                                                    ).round(2)
    
    return df_pages_segm_summary

#*******************************************************************************
# OPTION 1: JSON FILE WITH THE DOCUMENTS SEGMENTATION
#*******************************************************************************

def _save_segmentation_json(df: pd.DataFrame, output_path: Path) -> dict[str, list[dict]]:
    """ Saves the segmentation results in a JSON file and returns the corresponding dict. 
    The JSON will be a dict where the KEYS are the document paths and the VALUES are a list
    of dicts, one per segment, containing the following information:
    
    - doc_type: Label of the segment.
    - first_page: First page of the segment (1-based, inclusive).
    - last_page: Last page of the segment (1-based, inclusive).
    - n_pages: Number of pages in the segment.
    - label_ratio: Percentage of the doc that corresponds to the segment. """
    
    output = {} # Keys will be the doc paths
    for doc_path, group in df.groupby("doc_path"):
        segments = [] # list of dicts
        for _, row in group.iterrows():
            segments.append({
                "doc_type": row["segment_label"],
                "first_page": int(row["first_page"]),
                "last_page": int(row["last_page"]),
                "n_pages": int(row["n_pages"]),
                "percent_of_doc": float(row["label_ratio"])
            })
        output[str(doc_path)] = segments
        
    json_path = output_path / "labels_kmeans_preds.json"
    
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4)
    
    print(f"\nSegmentation .json file saved to {json_path}\n")
    return output

#*******************************************************************************
# OPTION 2: BUILD FINAL DOCUMENTS
#*******************************************************************************

def _build_master_pdf_type(df: pd.DataFrame, 
                          doc_type: str,
                          output_dir: Path) -> Path:
    """ Builds a master PDF concatenating the pages (first_page, last_page) 
    of all documents that belong to that type. 
    Returns the path of the master PDF file.
    - Note: first_page and last_page are inclusive
    """
    try:
        from pypdf import PdfReader, PdfWriter
    except ImportError:
        raise ImportError("Install pypdf: pip install pypdf")

    writer = PdfWriter()
    total_pages = 0
    
    # Select only the rows corresponding to the given doc_type, and the necessary columns.
    sub_df = df.loc[ df["segment_label"] == doc_type , ["doc_path", "first_page", "last_page"]]
    print(f"\nNumber of documents to process: {len(sub_df)}\n")
    
    # Iterates over the rows of the sub DataFrame.
    for doc_path, first_page, last_page in sub_df.itertuples(index=False, name=None):
    # name=None breaks the tuple, unpacking the columns directly into variables.
        print(f"Processing document: {doc_path.name!r}, pages {first_page} to {last_page}")
        
        reader = PdfReader(doc_path)
        n_pages = len(reader.pages)
        total_pages += (last_page - first_page + 1) # +1 because both are inclusive.
        
        if last_page > n_pages:
            raise ValueError(f"last_page value ({last_page}) is greater than the number of pages {n_pages} in document {doc_path}")
        
        for i in range(first_page-1, last_page): # -1 because page_idx is 1-based and pypdf is 0-based.
            writer.add_page(reader.pages[i])
    
    print(f"\nTotal number of pages found belonging to type {doc_type!r}:", total_pages)

    master_path = output_dir / f"{doc_type}_master.pdf"

    with open(master_path, "wb") as f: # Creates an empty file as f.
    # "xb" mode -> write binary, but only if the file does not exist. If it already exists, 
    # it raises an error. This is to avoid overwriting existing files.
    # "wb" mode -> write binary, overwrite existing files without warning.
        writer.write(f)

    print(f"Number of pages written in master PDF: {len(writer.pages)}")
    print(f"\nMaster PDF file created at: {master_path}\n")
    return master_path

def _build_master_pdfs(df: pd.DataFrame, 
                       doc_types: list[str], 
                       output_dir: Path) -> dict[str, Path]:
    """ Builds master PDFs for a list of given document types. 
    Returns a dict with doc_type as key and master PDF path as value."""
    master_paths = {}
    for doc_type in doc_types:
        print(f"\nBuilding master PDF for document type: {doc_type!r}")
        master_path = _build_master_pdf_type(df, doc_type, output_dir)
        master_paths[doc_type] = master_path
    
    print("Your final documents were successfully created:--------------------")
    pprint(master_paths)
    print("-----------------------------------------------------------------\n")    
    
    return master_paths


#*******************************************************************************
# MAIN FUNCTION
#*******************************************************************************
def main_function(data_dir: Path, 
                  descriptions_file: Path, 
                  output_dir: Path, 
                  types: list[str],
                  option: Literal["opt1", "opt2"] = "opt1") -> dict[str] | None:
    """ Main function that wraps the whole process of creating the segmentation.
    Takes as input the main paths.
    
    The 'option' argument determines the final output:
        - option="opt1": Creates a segmentation .json file.
        - option="opt2": Creates master PDFs for each document type."""

    # Body:
    doc_type_desc: dict[str, list[str]] = load_doc_type_descriptions(descriptions_file)
    print("\nNumber of document types:", len(doc_type_desc))
    
    files: list[Path] = list_files(data_dir, ALLOWED_EXTS)
    print("Total files/docs:", len(files))
    
    page_records = build_page_records(files)
    print("Total pages:", len(page_records), "\n")
    
    embedder = SentenceTransformerEmbedder()
    
    # Vectorizing the pages:----------------------------------------------------
    page_embs: np.ndarray = train_embedder(page_records, embedder)
    print("\nPage embeddings shape (vectors):", page_embs.shape, "\n")
    #---------------------------------------------------------------------------
    
    type_names, type_centroids = build_type_centroids(doc_type_desc, embedder)
    
    cluster_ids, km_model = cluster_pages_kmeans_seeded(page_embs, type_centroids, random_state=SEED)
    
    cluster_centroids: np.ndarray = km_model.cluster_centers_
    # Normalize again since KMeans does not guarantee that the centroids land in the unit sphere, even if the initial centroids were normalized.:
    cluster_centroids = _normalize_centroids(cluster_centroids)

    cluster_to_type, df_cluster_type_scores = map_clusters_to_types(cluster_centroids, type_centroids, type_names)
    
    cluster_to_type: dict[int, str] = _hungarian_remapping(df_cluster_type_scores, cluster_to_type)
    print("\nFinal cluster to type mapping:\n")
    pprint(cluster_to_type)
    print()
    
    # Summary DataFrame:--------------------------------------------------------
    df_pages_segm_summary = summary_df(page_records, cluster_ids, cluster_to_type)
    print("\nSummary DataFrame:",df_pages_segm_summary.head(20))
    #---------------------------------------------------------------------------
    
    # Option 1: JSON file with the segmentation of each document:----------------
    if option == "opt1": # This is the default option.
        return _save_segmentation_json(df_pages_segm_summary, output_dir)
    
    # Option 2: Build final documents:------------------------------------------
    elif option == "opt2":  
        _build_master_pdfs(df_pages_segm_summary, types, output_dir)



#*******************************************************************************
# MAIN:
#*******************************************************************************

if __name__ == "__main__":
    """ To run the code, execute the following command in the terminal, providing the path to your config.yaml file:
    >>> python kmeans_module.py --config ../config/config.yaml  
    (Assuming the config.yaml folder is parallel the module folder) """
    # Load configuration:
    parser = argparse.ArgumentParser(description="K-means clustering for document classification.")
    parser.add_argument("--config", type=Path, required=True, help="Path to the YAML config file.") # argparse automatically converts the type to Path.
    args = parser.parse_args() # Here is when the scripts reads from the command line.
    config: dict = load_config(args.config) # The name of the attribute "config" is taken from the flag name in add_argument().
    
    config_abs_path = args.config.resolve().parent # Paths will be relative to the config.yaml parent folder
    
    # Main Paths:
    data_dir = config_abs_path / config["paths"]["data_dir"]
    descriptions_file = config_abs_path / config["paths"]["descriptions_file"]
    output_dir = config_abs_path / config["paths"]["output_dir"]
    types = config["types"] # List of doc-types to build each master PDF.
    
    main_function(data_dir, descriptions_file, output_dir, types)