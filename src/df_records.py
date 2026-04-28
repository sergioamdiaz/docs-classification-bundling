#*******************************************************************************
# IMPORTS:
#*******************************************************************************

import pandas as pd
import numpy as np
try:
    from src.data import PageRecord
    print("Data module imported correctly \n")
except ImportError as e:
    print(f"Import Error: {e} \n")

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
            labels.append(df.loc[j, "label"])
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
            current_segm_label = df["label"][i]
            break

    # If there are no usable pages, do nothing. Most likely, the code will not reach this point.
    if current_segm_label is None:
        df["segment_label"] = df["label"]
        df["segment_change_at"] = False
        return df

    used_labels: set[str] = {current_segm_label}
    segmented: list[str] = []
    changed_flags: list[bool] = []

    """ This is where the segmentation logic happens: """
    for i in range(len(df)):
        usable = df.usable[i]
        orig_label = df["label"][i] # Original label of the currently inspected page.

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
    df_pages["label"] = df_pages["cluster_id"].map(mapping)
    df_pages["usable"]       = (df_pages["text_len"] > 50)

    df_pages_segm = (df_pages
                     .groupby("doc_path", group_keys=False)
                     .apply(lambda g: _segment_labels_no_backtracking(g)
                            .assign(doc_path=g.name) # to restore the doc_path after groupby (it became the index after the apply)
                     ))
    df_pages_segm.insert(0, "doc_path", df_pages_segm.pop("doc_path")) # Move doc_path back to the first column.
               
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