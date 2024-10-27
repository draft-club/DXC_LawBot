import faiss
import logging
import numpy as np
from constants import FAISS_INDEX_PATH
from model_setup import voyage_embedding_dimension


# Initialize FAISS index
def initialize_faiss_index():
    try:
        faiss_index = faiss.read_index(FAISS_INDEX_PATH)
        logging.info("Loaded existing FAISS index.")

        # Verify dimension match with embedding model
        if faiss_index.d != voyage_embedding_dimension:
            raise ValueError(
                f"Loaded FAISS index has dimension {faiss_index.d}, but expected {voyage_embedding_dimension}.")
    except (Exception, ValueError) as e:
        logging.info(f"Creating new FAISS index due to: {e}")
        faiss_index = faiss.IndexFlatL2(voyage_embedding_dimension)

    return faiss_index


# Function to save FAISS index
def save_faiss_index(faiss_index):
    faiss.write_index(faiss_index, FAISS_INDEX_PATH)
    logging.info("FAISS index saved successfully.")


# Load the FAISS index at module level so it can be reused
faiss_index = initialize_faiss_index()
