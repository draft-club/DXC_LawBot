import argparse
from load_and_prepare import clear_database, load_documents, split_documents, add_to_chroma
from retrieve import query_rag

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Process and query a document database.")
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    parser.add_argument("query_text", nargs="?", type=str, help="The query text (optional).")
    args = parser.parse_args()

    # Reset the database if requested
    if args.reset:
        print("✨ Clearing Database")
        clear_database()

    # Create or update the data store
    documents = load_documents()
    chunks = split_documents(documents)
    add_to_chroma(chunks)

    # Query phase if query_text is provided
    if args.query_text:
        query_rag(args.query_text)

if __name__ == "__main__":
    main()
