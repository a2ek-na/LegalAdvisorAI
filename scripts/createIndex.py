import json
import os
import sys
import chromadb
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

corpusPath = os.path.join('data', 'processed', 'ipc_corpus.json')
dbPath = os.path.join('dbStores', 'chromaDB')
collectionName = "ipc_sections"
modelName = 'law-ai/InLegalBERT' # Yeh specific Hugging Face model hai

def loadCorpus(path):
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

# <--- NAYA: Embeddings manually banane ka function --->
def generateEmbeddingsInBatches(documents, model, tokenizer, batchSize=32):
    """
    Generates embeddings in batches for higher efficiency.
    """
    allEmbeddings = []
    print(f"Generating embeddings in batches of {batchSize}...")
    
    # Agar GPU hai toh use karo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # documents list ko batchSize ke chunks me todo
    for i in tqdm(range(0, len(documents), batchSize), desc="Encoding Batches"):
        batchDocs = documents[i:i + batchSize]
        
        # 1. Poore batch ko ek saath tokenize karo
        encodedInput = tokenizer(
            batchDocs, padding=True, truncation=True, return_tensors='pt'
        ).to(device)
        
        # 2. Model ka output lo
        with torch.no_grad():
            modelOutput = model(**encodedInput)

        # 3. Mean Pooling karo (same logic, ab batch par kaam kar raha hai)
        lastHiddenState = modelOutput.last_hidden_state
        attentionMask = encodedInput['attention_mask']
        maskExpanded = attentionMask.unsqueeze(-1).expand(lastHiddenState.size()).float()
        sumEmbeddings = torch.sum(lastHiddenState * maskExpanded, 1)
        sumMask = torch.clamp(maskExpanded.sum(1), min=1e-9)
        meanPooledEmbeddings = sumEmbeddings / sumMask
        
        # 4. Batch ke saare embeddings ko CPU me daalo aur final list me add karo
        allEmbeddings.extend(meanPooledEmbeddings.cpu().numpy().tolist())
        
    return allEmbeddings

# In your main() function, simply change the call:
# embeddings = generateEmbeddings(documents, model, tokenizer)
# -- TO --
# embeddings = generateEmbeddingsInBatches(documents, model, tokenizer, batchSize=32)
def main():
    """
    Main function to create and populate the vector database.
    """
    corpus = loadCorpus(corpusPath)
    documents = [doc['text'] for doc in corpus]
    metadata = [{'id': doc['id']} for doc in corpus]
    ids = [doc['id'] for doc in corpus]

    # <--- BADLA GAYA: SentenceTransformer ke bajaye AutoTokenizer aur AutoModel load karo --->
    print(f"Initializing Hugging Face model: '{modelName}'...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(modelName)
        model = AutoModel.from_pretrained(modelName)
    except Exception as e:
        print(f"Error initializing AutoModel or AutoTokenizer: {e}")
        sys.exit(1)
    
    # <--- BADLA GAYA: Naya function call karke embeddings banao --->
    embeddings = generateEmbeddings(documents, model, tokenizer)
    print("Embeddings generated successfully.")

    # ChromaDB client setup karo
    client = chromadb.PersistentClient(path=dbPath)

    # Collection banao ya get karo
    collection = client.get_or_create_collection(name=collectionName)
    
    # Collection me data add karo
    collection.add(
        embeddings=embeddings,
        documents=documents,
        metadatas=metadata,
        ids=ids
    )

    print("\n--- Success! ---")
    print(f"Vector database has been created and populated.")
    print(f"Total documents indexed: {collection.count()}")
    print(f"Database is stored at: '{dbPath}'")

if __name__ == "__main__":
    main()