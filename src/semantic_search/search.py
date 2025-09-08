import chromadb
from sentence_transformers import SentenceTransformer
import os

class SemanticSearch:
    """
    A class to handleing the  emantic search andd Sentence Transformers.
    """
    def __init__(self):
        """
        Initializes the compponent, loading the model and connecting to the database.
        """

        self.db_path = os.path.join('db_stores', 'chroma_db')
        self.collection_name = "ipc_sections"
        self.model_name = 'law-ai/InLegalBERT'

        print("Initializing Semantic Search...")
        
        #load the Sentence Transformer model
        try:
            print(f"Loading sentence transformer model: '{self.model_name}'")
            self.model = SentenceTransformer(self.model_name)
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise 

        try:
            print(f"Connecting to vector database at: '{self.db_path}'")
            self.client = chromadb.PersistentClient(path=self.db_path)
            #client k ander ja k yeh wla database lao...
            self.collection = self.client.get_collection(name=self.collection_name)
            
            print("Successfully connected to the database.")
            print(f"Total documents in collection: {self.collection.count()}")
        except Exception as e:
            print(f"Error connecting to ChromaDB: {e}")
            raise 

    def search(self, query: str, top_n: int = 5):
        """
        Performs a semantic search.

        it take -> 
            query (str): The user's search query.
            top_n (int): The number of top results to return as woh equal h 5 k..

        it rweturns:
            list: A list of search results, where each result is a dictionary.
        """
        if not query:
            return []

        print(f"\nPerforming search for query: '{query}'")
        
        query_embedding = self.model.encode(query, convert_to_tensor=False)

        results = self.collection.query(
            #query_embegging transformer ka naam h...
            query_embeddings=[query_embedding.tolist()],
            n_results=top_n
        )
        
        ids = results.get('ids', [[]])[0]
        #distances = results.get('distances', [[]])[0]
        documents = results.get('documents', [[]])[0]
        
        formatted_results = []
        for i in range(len(ids)):
            formatted_results.append({
                'id': ids[i],
                'document': documents[i],
                #'distance': distances[i] # A smaller distance means a better match
            })
            
        print(f"Found {len(formatted_results)} results.")
        return formatted_results