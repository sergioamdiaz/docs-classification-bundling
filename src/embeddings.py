#*******************************************************************************
# IMPORTS:
#*******************************************************************************

import numpy as np
try:
    from src.data import PageRecord
    print("Data module imported correctly \n")
except ImportError as e:
    print(f"Import Error: {e} \n")

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
# EMBED DOCUMENT TYPES:
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