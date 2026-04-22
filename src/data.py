#*******************************************************************************
# IMPORTS:
#*******************************************************************************

import os, re, json, yaml
from pathlib import Path
from dataclasses import dataclass

#-------------------------------------------------------------------------------
# Constants for functions
#-------------------------------------------------------------------------------

# Which file extensions will be considered.
ALLOWED_EXTS = {".pdf", ".txt"}

# NOTE: These constants can be added in the future to the config file. For now they are hardcoded here for simplicity.

#*******************************************************************************
# DATA LOADING AND PREPROCESSING:
#*******************************************************************************

# Loading Data: ----------------------------------------------------------------

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
        raise ImportError(f"\nInstall PyMuPDF: pip install pymupdf\n")

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