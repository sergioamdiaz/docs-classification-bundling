#*******************************************************************************
# IMPORTS:
#*******************************************************************************

import json
from pathlib import Path
from pprint import pprint
from typing import Literal

import numpy as np
import pandas as pd

try:
    from src.data import list_files, load_doc_type_descriptions, build_page_records
    print("Data module imported correctly \n")
    
    from src.embeddings import SentenceTransformerEmbedder, train_embedder, build_type_centroids
    print("Embeddings module imported correctly \n")
    
    from src.kmeans_clustering import cluster_pages_kmeans_seeded, map_clusters_to_types, normalize_centroids, hungarian_remapping
    print("KMeans clustering module imported correctly \n")
    
    from src.df_records import summary_df
    print("DataFrame records module imported correctly \n")
    
except ImportError as e:
    print(f"Import Error: {e} \n")

SEED = 42
np.random.seed(SEED)

#-------------------------------------------------------------------------------
# Constants for functions
#-------------------------------------------------------------------------------

# Which file extensions will be considered.
ALLOWED_EXTS = {".pdf", ".txt"}

# NOTE: These constants can be added in the future to the config file. For now they are hardcoded here for simplicity.

#*******************************************************************************
# FULL PIPELINE MAIN FUNCTION
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
    cluster_centroids = normalize_centroids(cluster_centroids)

    cluster_to_type, df_cluster_type_scores = map_clusters_to_types(cluster_centroids, type_centroids, type_names)
    
    cluster_to_type: dict[int, str] = hungarian_remapping(df_cluster_type_scores, cluster_to_type)
    print("\nFinal cluster to type mapping:\n")
    pprint(cluster_to_type)
    print()
    
    # Summary DataFrame:--------------------------------------------------------
    df_pages_segm_summary = summary_df(page_records, cluster_ids, cluster_to_type)
    print("\nSummary DataFrame: (first 20 rows)", df_pages_segm_summary.head(20))
    #---------------------------------------------------------------------------
    
    # Option 1: JSON file with the segmentation of each document:----------------
    if option == "opt1": # This is the default option.
        return save_segmentation_json(df_pages_segm_summary, output_dir)
    
    # Option 2: Build final documents:------------------------------------------
    elif option == "opt2":  
        build_master_pdfs(df_pages_segm_summary, types, output_dir)
        
#*******************************************************************************
# OPTION 1: JSON FILE WITH THE DOCUMENTS SEGMENTATION
#*******************************************************************************

def save_segmentation_json(df: pd.DataFrame, output_path: Path) -> dict[str, list[dict]]:
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

def build_master_pdfs(df: pd.DataFrame, 
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

