import json
import logging
import os
import numpy as np
from sentence_transformers import SentenceTransformer
import time
import torch # Just for GPU checks

# --- Configuration ---
TEXT_DATA_FILE = "DataExtract/Data/node_text_data.json" 
OUTPUT_FOLDER = "LLM_embed/embed_data"
EMBEDDING_OUTPUT_FILE = os.path.join(OUTPUT_FOLDER, "node_sbert_embeddings.npz") 
MODEL_NAME = 'all-MiniLM-L6-v2' # SBERT Model
BATCH_SIZE = 64 # Process texts in batches for efficiency

# --- Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Helper Functions ---
def load_json_file(filename):
    """Loads data from a JSON file."""
    if not os.path.exists(filename):
        logging.error(f"Error: File '{filename}' not found.")
        return None
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON from '{filename}': {e}")
        return None
    except Exception as e:
        logging.error(f"An error occurred while reading '{filename}': {e}")
        return None

# --- Main Embedding Generation Script ---
def main():
    logging.info(f"Loading node text data from {TEXT_DATA_FILE}...")
    node_texts = load_json_file(TEXT_DATA_FILE)

    if node_texts is None:
        logging.error("Failed to load text data. Exiting.")
        return

    # Ensure keys are integers if they were loaded as strings from JSON
    try:
        node_texts_int_keys = {int(k): v for k, v in node_texts.items()}
        node_ids = list(node_texts_int_keys.keys())
        texts_to_encode = [node_texts_int_keys[nid] for nid in node_ids]
        logging.info(f"Loaded text for {len(node_ids)} nodes.")
    except (ValueError, TypeError) as e:
        logging.error(f"Error converting node IDs to integers: {e}. Make sure keys in {TEXT_DATA_FILE} are numeric strings.")
        return

    if not texts_to_encode:
        logging.warning("No text data found to encode.")
        return

    # --- Determine Device ---
    device = 'cpu' # Default to CPU
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        if num_gpus > 0:
            # Use the first available GPU (index 0)
            device = 'cuda:0'
            logging.info(f"CUDA is available. Found {num_gpus} GPU(s). Using GPU 0.")
        else:
            logging.warning("CUDA reports available but no devices found. Using CPU.")
    else:
        logging.warning("CUDA not available. Using CPU.")
    logging.info(f"Using device: {device}")
    
    # --- End Determine Device ---


    logging.info(f"Loading Sentence Transformer model: {MODEL_NAME}...")
    # This will download the model the first time it's run
    try:
        # Pass the determined device to the model constructor
        model = SentenceTransformer(MODEL_NAME, device=device)
        logging.info("Model loaded successfully.")
    except Exception as e:
        logging.error(f"Failed to load Sentence Transformer model '{MODEL_NAME}': {e}")
        logging.error("Ensure you have internet connectivity for the first download.")
        logging.error("Ensure PyTorch is correctly installed if trying to use GPU.")
        return

    logging.info(f"Generating embeddings in batches of {BATCH_SIZE}...")
    start_time = time.time()

    # Encode the texts into embeddings
    try:
        # Replace None or empty strings with a placeholder to avoid errors
        processed_texts = [text if text else " " for text in texts_to_encode]

        # Generate embeddings (will use the device specified during model init)
        embeddings = model.encode(processed_texts, batch_size=BATCH_SIZE, show_progress_bar=True)

        # Ensure embeddings are numpy arrays
        embeddings = np.array(embeddings)

    except Exception as e:
        logging.error(f"Error during embedding generation: {e}", exc_info=True)
        return

    end_time = time.time()
    logging.info(f"Embedding generation complete. Time taken: {end_time - start_time:.2f} seconds.")
    logging.info(f"Shape of embeddings array: {embeddings.shape}") # Should be (num_nodes, embedding_dimension)

    # --- Save Embeddings ---
    logging.info(f"Saving embeddings to {EMBEDDING_OUTPUT_FILE}...")
    try:
        np.savez_compressed(
            EMBEDDING_OUTPUT_FILE,
            node_ids=np.array(node_ids, dtype=int), # Save the IDs
            embeddings=embeddings                 # Save the embedding vectors
        )
        logging.info("Embeddings saved successfully.")
    except Exception as e:
        logging.error(f"Failed to save embeddings: {e}")

if __name__ == "__main__":
    main()