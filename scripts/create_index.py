# scripts/create_index.py

import json
import os
import sys
import chromadb
from sentence_transformers import SentenceTransformer

# --- Configuration ---
# Path to the processed corpus file
CORPUS_PATH = os.path.join('data', 'processed', 'ipc_corpus.json')
# Path to the persistent ChromaDB storage
DB_PATH = os.path.join('db_stores', 'chroma_db')
# Name of the ChromaDB collection
COLLECTION_NAME = "ipc_sections"
# The specified Hugging Face model for embeddings
MODEL_NAME = 'law-ai/InLegalBERT'


def load_corpus(path):
    """Loads the JSON corpus from the given path."""
    print(f"Loading corpus from {path}...")
    try:
        with open(path, 'r', encoding='utf-8') as f:
            corpus = json.load(f)
        print(f"Successfully loaded {len(corpus)} documents.")
        return corpus
    except FileNotFoundError:
        print(f"Error: Corpus file not found at {path}")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {path}")
        sys.exit(1)

def main():
    """
    Main function to create and populate the vector database.
    """
    # 1. Load the processed data
    corpus = load_corpus(CORPUS_PATH)
    documents = [doc['text'] for doc in corpus]
    metadata = [{'id': doc['id']} for doc in corpus]
    ids = [doc['id'] for doc in corpus]

    # 2. Initialize the embedding model
    print(f"Initializing sentence transformer model: '{MODEL_NAME}'...")
    print("Note: The first time this runs, it will download the model, which may take some time.")
    try:
        model = SentenceTransformer(MODEL_NAME)
    except Exception as e:
        print(f"Error initializing SentenceTransformer model: {e}")
        sys.exit(1)
    
    # 3. Generate embeddings for the documents
    print("Generating embeddings for all documents... This may take a few minutes.")
    embeddings = model.encode(documents, show_progress_bar=True)
    print("Embeddings generated successfully.")

    # 4. Initialize and set up the ChromaDB client
    print(f"Setting up ChromaDB persistent client at: {DB_PATH}")
    # Using a persistent client to save the database to disk
    client = chromadb.PersistentClient(path=DB_PATH)

    # 5. Create or get the collection
    print(f"Creating or getting ChromaDB collection: '{COLLECTION_NAME}'")
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    # 6. Add the documents, embeddings, and metadata to the collection
    print("Adding documents to the collection...")
    # ChromaDB's add method can take lists of items to add in bulk.
    collection.add(
        embeddings=embeddings,
        documents=documents,
        metadatas=metadata,
        ids=ids
    )

    print("\n--- Success! ---")
    print(f"Vector database has been created and populated.")
    print(f"Total documents indexed: {collection.count()}")
    print(f"Database is stored at: '{DB_PATH}'")

if __name__ == "__main__":
    main()