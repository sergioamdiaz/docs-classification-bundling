#*******************************************************************************
# IMPORTS:
#*******************************************************************************

from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

#*******************************************************************************
# KMEANS CLUSTERING:
#*******************************************************************************

def cluster_pages_kmeans_seeded(page_embs: np.ndarray,
                                type_centroids: np.ndarray,
                                random_state: int = 42,
                                max_iter: int = 300) -> tuple[np.ndarray, object]:
    """
    cluster pages using k-means, initializing the centroids with the type centroids obtained from the doc-types.
    Returns the cluster id for each page and the k-means model.
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

def normalize_centroids(cluster_centroids: np.ndarray) -> np.ndarray:
    """ Normalize cluster centroids. This is necessary for cosine similarity. """
    return cluster_centroids / (np.linalg.norm(cluster_centroids, axis=1, keepdims=True) + 1e-12)

#-------------------------------------------------------------------------------
# Mapping clusters to types:
def map_clusters_to_types(cluster_centroids: np.ndarray,
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

def hungarian_remapping(scores: pd.DataFrame, mapping: dict[int, str]) -> dict[int, str]:
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