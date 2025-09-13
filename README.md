# LegalAdvisorAI: Semantic Search for the Indian Penal Code
An AI-powered semantic search engine for the Indian Penal Code (IPC), allowing users to find relevant legal sections using natural language queries.

---
## About The Project
Traditional keyword-based search methods for legal documents can be inefficient and often require precise legal terminology, making them challenging for the general public. LegalAdvisorAI solves this problem by leveraging state-of-the-art Natural Language Processing (NLP) to understand the *meaning* and *intent* behind a query.

This project uses a specialized language model, **InLegalBERT**, which is fine-tuned on Indian legal text, and a **ChromaDB** vector database to perform fast and contextually-aware searches. Instead of matching keywords, it matches the semantic meaning of your query to the text of the IPC sections.

---
## Key Features
* **Natural Language Queries:** Search for legal sections by describing a situation in plain English.
* **Semantic Understanding:** Goes beyond keywords to find results based on contextual meaning.
* **Specialized Legal Model:** Utilizes `law-ai/InLegalBERT` for a nuanced understanding of legal jargon.
* **Efficient Vector Search:** Powered by ChromaDB for fast and scalable similarity searches.

---
## Tech Stack
This project is built with the following technologies:
* [Python](https://www.python.org/)
* [Pandas](https://pandas.pydata.org/) for data preprocessing
* [Sentence-Transformers](https://www.sbert.net/) for model interaction
* [Hugging Face Transformers](https://huggingface.co/docs/transformers/index) for the base model
* [ChromaDB](https://www.trychroma.com/) as the vector database

---
## Project Structure
The repository is organized as follows:
```

.
├── data
│   ├── processed/ipc\_corpus.json   \# Cleaned, final JSON data
│   └── raw/ipc\_sections.csv        \# The initial raw dataset
├── db\_stores
│   └── chroma\_db/                  \# The persistent vector database
├── scripts
│   ├── preprocessData.py           \# Script to clean the raw CSV
│   └── createIndex.py              \# Script to build the vector database
├── src
│   └── semantic\_search
│       └── search.py               \# The SemanticSearch engine class
├── .gitignore
├── main.py                         \# The main script to run a search
└── requirements.txt                \# Project dependencies

````

---
## Getting Started
Follow these steps to set up and run the project locally.

### Prerequisites
* Python 3.11.4

### Installation
1.  **Clone the repository:**
    ```sh
    git clone [https://github.com/a2ek-na/LegalAdvisorAI.git](https://github.com/a2ek-na/LegalAdvisorAI.git)
    ```
2.  **Navigate into the project directory:**
    ```sh
    cd Implementation
    ```
3.  **Create and activate a virtual environment:**
    * **Windows:**
        ```sh
        python -m venv .venv
        .\.venv\Scripts\activate
        ```
    * **macOS / Linux:**
        ```sh
        python3 -m venv .venv
        source .venv/bin/activate
        ```
4.  **Install the required dependencies:**
    ```sh
    pip install -r requirements.txt
    ```

---
## Usage
To use the search engine, you must first process the data and build the database.

1.  **Preprocess the Raw Data (One-time step):**
    Run the preprocessing script to clean the CSV and create the JSON corpus.
    ```sh
    python scripts/preprocessData.py
    ```
2.  **Create the Vector Database (One-time step):**
    Run the indexing script to generate embeddings and populate ChromaDB. This will take a few minutes as it downloads and runs the AI model.
    ```sh
    python scripts/createIndex.py
    ```
3.  **Run a Search:**
    Modify the `query` variable inside `main.py` with your question and run the script.
    ```python
    # Inside main.py
    query = "someone stole my wallet from my pocket"
    ```
    Execute the script from your terminal:
    ```sh
    python main.py
    ```
    **Example Output:**
    ```
    Initializing Semantic Search...
    Model loaded successfully.
    Successfully connected to the database.
    Total documents in collection: 442

    Performing search for query: 'someone stole my wallet from my pocket'
    Found 5 results.

    --- Search Results ---

    1. Section ID: IPC_378
    - - - - - - - - - - - - - - - -

    2. Section ID: IPC_379
    - - - - - - - - - - - - - - - -
    ```

---
## License
Distributed under the MIT License. See `LICENSE.txt` for more information.

````
