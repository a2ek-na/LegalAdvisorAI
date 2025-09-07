from src.semantic_search.search import SemanticSearch
import pprint

def run_search():
    """
    Initializes the search engine and performs a sample query.
    """
    # This will load the model and connect to the DB.
    try:
        search_engine = SemanticSearch()
    except Exception as e:
        print(f"Failed to initialize the search engine: {e}")
        return

    # --- Define your search query here ---
    query = "someone entered my house"
    
    results = search_engine.search(query=query, top_n=5)

    print("\n--- Search Results ---")
    if results:
        for i, result in enumerate(results):
            print(f"\n{i+1}. Section ID: {result['id']}")
            print(f"   Similarity Score (Distance): {result['distance']:.4f}")
            print("-" * 20)
    else:
        print("No results found.")

if __name__ == "__main__":
    run_search()