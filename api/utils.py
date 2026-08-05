import numpy as np


def embedding_to_pgvector(embedding: np.ndarray) -> str:
    """Convert a numpy embedding to pgvector's text format: '[0.1,0.2,...]'"""
    return "[" + ",".join(str(float(x)) for x in embedding) + "]"


def blob_path_to_url(blob_path: str) -> str:
    """'pics/9.jpg' (as stored in the DB) -> '/pics/9.jpg' (browser-facing URL)"""
    return "/" + blob_path.lstrip("/")
