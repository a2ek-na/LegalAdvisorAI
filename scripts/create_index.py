import json
import os
import sys
import chromadb
from sentence_transformers import SentenceTransformer

CORPUS_PATH = os.path.join('data', 'processed', 'ipc_corpus.json')
DB_PATH = os.path.join('db_stores', 'chroma_db')
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
    corpus = load_corpus(CORPUS_PATH)
    documents = [doc['text'] for doc in corpus]
    metadata = [{'id': doc['id']} for doc in corpus]
    ids = [doc['id'] for doc in corpus]

    print(f"Initializing sentence transformer model: '{MODEL_NAME}'...")
    #The first  it will download the model, thoda time lagega..
    try:
        model = SentenceTransformer(MODEL_NAME)
    except Exception as e:
        print(f"Error initializing SentenceTransformer model: {e}")
        sys.exit(1)
    
    #Generating embeddings.....
    embeddings = model.encode(documents, show_progress_bar=True)
    print("Embeddings generated successfully.")

    #set up the ChromaDB at ==> DB_PATH ....
    client = chromadb.PersistentClient(path=DB_PATH)

    # getting db collection from ==>COLLECTION_NAME ...
    collection = client.get_or_create_collection(name=COLLECTION_NAME)
    
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