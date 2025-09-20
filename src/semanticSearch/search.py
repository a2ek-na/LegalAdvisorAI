import os
import torch
import chromadb
from transformers import AutoTokenizer, AutoModel

class SemanticSearch:
    """
    A class to handle semantic search using AutoModel for embeddings.
    """
    def __init__(self):
        """
        Initializes the component, loading the model and connecting to the database.
        """
        self.dbPath = os.path.join('dbStores', 'chromaDB')
        self.collectionName = "ipc_sections"
        self.modelName = 'law-ai/InLegalBERT'

        print("Initializing Semantic Search...")

        # --- Model aur Tokenizer Load Karo ---
        try:
            print(f"Loading transformer model: '{self.modelName}'")
            self.tokenizer = AutoTokenizer.from_pretrained(self.modelName)
            self.model = AutoModel.from_pretrained(self.modelName)
            
            # Agar GPU hai toh use karo
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            print(f"Model loaded successfully on {self.device}.")

        except Exception as e:
            print(f"Error loading model: {e}")
            raise

        # --- ChromaDB se Connect Karo ---
        try:
            print(f"Connecting to vector database at: '{self.dbPath}'")
            self.client = chromadb.PersistentClient(path=self.dbPath)
            # Collection (database table) ko get karo
            self.collection = self.client.get_collection(name=self.collectionName)
            
            print("Successfully connected to the database.")
            print(f"Total documents in collection: {self.collection.count()}")
        except Exception as e:
            print(f"Error connecting to ChromaDB: {e}")
            raise

    def _generateEmbedding(self, text: str):
        """
        Generates a single embedding for a given text using mean pooling.
        (Private method)
        """
        # 1. Input text ko tokenize karo
        encodedInput = self.tokenizer(
            text, padding=True, truncation=True, return_tensors='pt'
        ).to(self.device)

        # 2. Model ka output lo
        with torch.no_grad():
            modelOutput = self.model(**encodedInput)

        # 3. Mean Pooling karo
        lastHiddenState = modelOutput.last_hidden_state
        attentionMask = encodedInput['attention_mask']
        maskExpanded = attentionMask.unsqueeze(-1).expand(lastHiddenState.size()).float()
        sumEmbeddings = torch.sum(lastHiddenState * maskExpanded, 1)
        sumMask = torch.clamp(maskExpanded.sum(1), min=1e-9)
        meanPooledEmbedding = sumEmbeddings / sumMask
        
        # 4. Embedding ko CPU me daalo aur list banao
        return meanPooledEmbedding.cpu().numpy().tolist()[0]

    def search(self, query: str, topN: int = 5):
        """
        Performs a semantic search.
        """
        if not query:
            return []

        print(f"\nPerforming search for query: '{query}'")
        
        # Query ke liye embedding banao
        queryEmbedding = self._generateEmbedding(query)

        # ChromaDB me query karo
        results = self.collection.query(
            query_embeddings=[queryEmbedding],
            n_results=topN
        )
        
        ids = results.get('ids', [[]])[0]
        distances = results.get('distances', [[]])[0]
        documents = results.get('documents', [[]])[0]
        
        formattedResults = []
        for i in range(len(ids)):
            formattedResults.append({
                'id': ids[i],
                'document': documents[i],
                'distance': distances[i] # Kam distance matlab behtar match
            })
            
        print(f"Found {len(formattedResults)} results.")
        return formattedResults

# --- Example Usage ---
if __name__ == '__main__':
    try:
        searchEngine = SemanticSearch()
        
        # Apni query yahan likho
        userQuery = "theft of property at night"
        
        searchResults = searchEngine.search(userQuery, topN=3)
        
        print("\n--- Search Results ---")
        for result in searchResults:
            print(f"ID: {result['id']}")
            print(f"Document: {result['document'][:250]}...") # Pehle 250 characters dikhao
            print("-" * 20)
            
    except Exception as e:
        print(f"An error occurred during the process: {e}")