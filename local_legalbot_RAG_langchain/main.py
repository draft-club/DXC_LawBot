import argparse
import logging
from tqdm import tqdm
from time import time
from load_and_prepare import clear_database, load_documents, split_documents, add_to_chroma
from retrieve import query_rag

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def main():
    # Start timer
    start_time = time()

    # Parse arguments
    parser = argparse.ArgumentParser(description="Process and query a document database.")
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    parser.add_argument("query_text", nargs="?", type=str, help="The query text (optional).")
    args = parser.parse_args()

    # Reset the database if requested
    if args.reset:
        logger.info("✨ Clearing Database")
        clear_database()

    # Create or update the data store
    logger.info("📄 Loading documents")
    documents = load_documents()

    logger.info("🔍 Splitting documents into chunks")
    chunks = []
    for doc in tqdm(documents, desc="Splitting documents", unit="doc"):
        chunks.extend(split_documents([doc]))

    logger.info("🗄️ Adding chunks to the database")
    for chunk in tqdm(chunks, desc="Adding to Chroma", unit="chunk"):
        add_to_chroma([chunk])

    # Query phase if query_text is provided
    if args.query_text:
        logger.info(f"🔎 Querying the database with text: '{args.query_text}'")
        query_rag(args.query_text)

    # End timer and print elapsed time
    end_time = time()
    elapsed_time = end_time - start_time
    logger.info(f"⏱️ Total execution time: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main()
