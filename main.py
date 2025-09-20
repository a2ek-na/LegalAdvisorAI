from src.semanticSearch.search import SemanticSearch

def runSearch():
    """
    Initializes the search engine and runs an interactive search CLI.
    """
    # Yeh model load karega aur DB se connect karega (sirf ek baar).
    try:
        searchEngine = SemanticSearch()
        print("\n--- Interactive IPC Search CLI ---")
        print("Type your legal query below. Type 'quit' or 'exit' to stop.")
    except Exception as e:
        print(f"Search engine initialize nahi ho paya: {e}")
        return

    # --- Interactive Search Loop ---
    while True:
        # User se query lo
        query = input("\nEnter your query > ")

        # Exit karne ka condition check karo
        if query.lower() in ['quit', 'exit']:
            print("Exiting the search. Goodbye!")
            break
        
        # Agar query khali hai toh continue karo
        if not query.strip():
            continue

        # Search perform karo
        results = searchEngine.search(query=query, topN=5)

        # Results print karo
        if results:
            print("\nFound matching IPC sections:")
            for result in results:
                print(f"- Section {result['id']}")
        else:
            print("Koi results nahi mile.")

if __name__ == "__main__":
    runSearch()

# --- Model run karne ke liye ---
# -> Environment activate karo ->>> .\.venv\Scripts\activate
# -> Run karne ke liye ->>> python main.py