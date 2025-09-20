from src.semanticSearch.search import SemanticSearch

def runSearch():
    """
    Initializes the search engine and performs a sample query.
    """
    # Yeh model load karega aur DB se connect karega.
    try:
        searchEngine = SemanticSearch()
        # Main function ka connection...
    except Exception as e:
        print(f"Search engine initialize nahi ho paya: {e}")
        return

    # Apna search query yahan daalo...->
    query = "My former colleague is deliberately spreading false rumors about me online to damage my professional reputation"
    
    results = searchEngine.search(query=query, topN=5)

    print("\n--- Search Results ---")
    if results:
        for i, result in enumerate(results):
            # Document ka text print karo
            print(f"\n{i+1}. Section ID: {result['id']}")
            # print(f"   Document: {result['document']}")

            # Neeche wali line uncomment karke similarity score dekh sakte ho
            # print(f"   Similarity Score (Distance): {result['distance']:.4f}")
            print("-" * 20)
    else:
        print("Koi results nahi mile.")

if __name__ == "__main__":
    runSearch()

# --- Model run karne ke liye ---
# -> Environment activate karo ->>> .\.venv\Scripts\activate
# -> Run karne ke liye ->>> python main.py