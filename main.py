import os
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle 


KNOWLEDGE_BASE_FILE = "knowledge_base.txt"

MODEL_NAME = "intfloat/multilingual-e5-large" 

FAISS_INDEX_STORAGE = "faiss_index.bin"
ORIGINAL_SENTENCES_STORAGE = "original_sentences.pkl"

def load_contexts(file_path):
    """Loads contextual information from a specified text file, treating each non-empty line as a distinct context."""
    if not os.path.exists(file_path):
        print(f"ERROR: Knowledge base file '{file_path}' not found. Please create it.")
        return []
    with open(file_path, 'r', encoding='utf-8') as f:
        contexts = [line.strip() for line in f if line.strip()] 
    print(f"SUCCESS: Loaded {len(contexts)} contexts from '{file_path}'.")
    return contexts

def generate_and_persist_embeddings(contexts, model_id, index_file, contexts_file):
    """
    Generates vector embeddings for the provided contexts, constructs a FAISS index,
    and saves both the index and the original contexts for future use.
    """
    print(f"ACTION: Attempting to load SentenceTransformer model: {model_id}...")
    try:
        language_model = SentenceTransformer(model_id)
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to load the model. Check your internet connection or verify the model ID. Details: {e}")
        return None, None

    print("ACTION: Generating numerical embeddings for your contexts. This may take a moment...")
    embeddings = language_model.encode(contexts, show_progress_bar=True)
    print(f"SUCCESS: Generated {len(embeddings)} embeddings, each with a dimension of {embeddings.shape[1]}.")

    embeddings = np.array(embeddings).astype('float32')

    embedding_dimension = embeddings.shape[1]
    faiss_index = faiss.IndexFlatL2(embedding_dimension)

    faiss_index.add(embeddings)
    print(f"SUCCESS: FAISS index created and populated with {faiss_index.ntotal} vectors.")

    faiss.write_index(faiss_index, index_file)
    print(f"PERSISTENCE: FAISS index saved to '{index_file}'.")

    with open(contexts_file, 'wb') as f:
        pickle.dump(contexts, f)
    print(f"PERSISTENCE: Original contexts saved to '{contexts_file}'.")

    return faiss_index, language_model

def load_persisted_embeddings(index_file, contexts_file):
    """Attempts to load a previously saved FAISS index and its corresponding original contexts."""
    if os.path.exists(index_file) and os.path.exists(contexts_file):
        print(f"ACTION: Loading FAISS index from '{index_file}'...")
        faiss_index = faiss.read_index(index_file)
        print(f"ACTION: Loading contexts from '{contexts_file}'...")
        with open(contexts_file, 'rb') as f:
            contexts = pickle.load(f)
        return faiss_index, contexts
    return None, None 

def main_execution_flow():
    """Controls the main workflow of the semantic search system."""

    faiss_index_ready, contexts_ready = load_persisted_embeddings(FAISS_INDEX_STORAGE, ORIGINAL_SENTENCES_STORAGE)

    if faiss_index_ready is None or contexts_ready is None:
        contexts_to_process = load_contexts(KNOWLEDGE_BASE_FILE)
        if not contexts_to_process:
            print("CRITICAL: No contexts available to build the search system. Exiting.")
            return

        faiss_index_ready, language_model_ready = generate_and_persist_embeddings(
            contexts_to_process, MODEL_NAME, FAISS_INDEX_STORAGE, ORIGINAL_SENTENCES_STORAGE
        )
        if faiss_index_ready is None:
            print("CRITICAL: Failed to build embeddings and FAISS index. Exiting.")
            return
    else:
        contexts_to_process = contexts_ready
        print("SUCCESS: Embeddings and contexts loaded from previous session.")

    try:
        language_model_ready = SentenceTransformer(MODEL_NAME)
        print(f"SUCCESS: Model '{MODEL_NAME}' loaded for processing user queries.")
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to load the query model. {e}")
        return

    print("\n--- Semantic Search System Activated ---")
    print("Enter your query to find relevant information. Type 'خروج' (Arabic for exit) or 'exit' to terminate.")

    while True:
        user_query = input("Your Query (استعلامك): ").strip()

        if user_query.lower() == 'خروج' or user_query.lower() == 'exit':
            print("ACTION: Exiting Semantic Search System. Goodbye!")
            break

        if not user_query:
            print("WARNING: Query cannot be empty. Please enter your question.")
            continue

        query_embedding = language_model_ready.encode([user_query]).astype('float32')

        num_results_to_retrieve = 3 

        distances, indices = faiss_index_ready.search(query_embedding, num_results_to_retrieve)

        print("\n--- Top Relevant Contexts ---")
        if indices.size == 0 or indices[0][0] == -1: 
            print("No relevant contexts found for your query.")
        else:
            for i, idx in enumerate(indices[0]):
                if idx >= 0 and idx < len(contexts_to_process): 
                    print(f"Rank {i+1} (Similarity Score: {1 - distances[0][i]:.4f}): {contexts_to_process[idx]}")
                else:
                    print(f"WARNING: Retrieved an invalid index ({idx}) from FAISS. Skipping.")
        print("----------------------------")

if __name__ == "__main__":
    main_execution_flow()
