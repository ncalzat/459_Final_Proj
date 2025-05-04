import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
import os
import logging
from typing import Tuple, Dict, List
import networkx as nx
from DataExtract.graph_builder import build_graph_from_jsonl

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_embeddings(embedding_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load embeddings from a .npz file.
    
    Args:
        embedding_path: Path to the .npz file containing embeddings
        
    Returns:
        Tuple of (node_ids, embeddings)
    """
    try:
        data = np.load(embedding_path)
        return data['node_ids'], data['embeddings']
    except Exception as e:
        logging.error(f"Error loading embeddings from {embedding_path}: {e}")
        raise

def reduce_dimensions(embeddings: np.ndarray, n_components: int = 128) -> np.ndarray:
    """
    Apply PCA to reduce embedding dimensions.
    
    Args:
        embeddings: 2D array of embeddings
        n_components: Number of dimensions to reduce to
        
    Returns:
        Reduced dimension embeddings
    """
    pca = PCA(n_components=n_components)
    return pca.fit_transform(embeddings)

def create_negative_edges(graph: nx.Graph, num_neg_samples: int) -> List[Tuple[int, int]]:
    """
    Create negative edges (non-existent edges) for evaluation.
    
    Args:
        graph: NetworkX graph
        num_neg_samples: Number of negative samples to generate
        
    Returns:
        List of negative edge tuples
    """
    nodes = list(graph.nodes())
    negative_edges = set()
    while len(negative_edges) < num_neg_samples:
        u, v = np.random.choice(nodes, 2, replace=False)
        if not graph.has_edge(u, v) and (u, v) not in negative_edges and (v, u) not in negative_edges:
            negative_edges.add((u, v))
    return list(negative_edges)

def make_features(embeddings: np.ndarray, node_ids: np.ndarray, edge_list: List[Tuple[int, int]]) -> np.ndarray:
    node_to_idx = {nid: idx for idx, nid in enumerate(node_ids)}
    features = []
    for u, v in edge_list:
        if u in node_to_idx and v in node_to_idx:
            u_emb = embeddings[node_to_idx[u]]
            v_emb = embeddings[node_to_idx[v]]
            # Concatenate embeddings
            features.append(np.concatenate([u_emb, v_emb]))
    return np.array(features)

def supervised_link_prediction(embeddings: np.ndarray, node_ids: np.ndarray, 
                              train_edges: List[Tuple[int, int]], train_neg_edges: List[Tuple[int, int]],
                              test_edges: List[Tuple[int, int]], test_neg_edges: List[Tuple[int, int]]) -> Tuple[float, float]:
    # Prepare training data
    X_train_pos = make_features(embeddings, node_ids, train_edges)
    X_train_neg = make_features(embeddings, node_ids, train_neg_edges)
    y_train = np.concatenate([np.ones(len(X_train_pos)), np.zeros(len(X_train_neg))])
    X_train = np.vstack([X_train_pos, X_train_neg])

    # Prepare test data
    X_test_pos = make_features(embeddings, node_ids, test_edges)
    X_test_neg = make_features(embeddings, node_ids, test_neg_edges)
    y_test = np.concatenate([np.ones(len(X_test_pos)), np.zeros(len(X_test_neg))])
    X_test = np.vstack([X_test_pos, X_test_neg])

    # Train classifier
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    y_scores = clf.predict_proba(X_test)[:, 1]

    auc_roc = roc_auc_score(y_test, y_scores)
    avg_precision = average_precision_score(y_test, y_scores)
    return auc_roc, avg_precision

def evaluate_embeddings_supervised(embedding_path: str, graph: nx.Graph, 
                                  test_size: float = 0.2, n_components: int = 128) -> Dict[str, float]:
    """
    Evaluate link prediction performance of embeddings.
    
    Args:
        embedding_path: Path to embeddings .npz file
        graph: NetworkX graph
        test_size: Proportion of edges to use for testing
        n_components: Number of PCA components to use
        
    Returns:
        Dictionary containing evaluation metrics
    """
    node_ids, embeddings = load_embeddings(embedding_path)
    if embeddings.shape[1] > n_components:
        logging.info(f"Reducing dimensions from {embeddings.shape[1]} to {n_components}")
        embeddings = reduce_dimensions(embeddings, n_components)
    edges = list(graph.edges())
    train_edges, test_edges = train_test_split(edges, test_size=test_size, random_state=42)
    num_train_neg = len(train_edges)
    num_test_neg = len(test_edges)
    train_neg_edges = create_negative_edges(graph, num_train_neg)
    test_neg_edges = create_negative_edges(graph, num_test_neg)
    auc_roc, avg_precision = supervised_link_prediction(
        embeddings, node_ids, train_edges, train_neg_edges, test_edges, test_neg_edges
    )
    return {
        'auc_roc': auc_roc,
        'average_precision': avg_precision
    }

def main():
    import os
    from DataExtract.graph_builder import build_graph_from_jsonl

    # Path to the JSONL file used for graph construction
    jsonl_path = "DataExtract/Data/stackexchange_cdxtoolkit_data_all_fixed.jsonl"
    if not os.path.exists(jsonl_path):
        raise FileNotFoundError(f"JSONL file for graph construction not found: {jsonl_path}")

    logging.info(f"Loading graph and metadata from {jsonl_path}...")
    graph, url_to_id, id_to_url, nodes_data = build_graph_from_jsonl(jsonl_path)
    logging.info(f"Graph loaded: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges.")

    # Paths to embedding files
    sbert_path = "LLM_embed/embed_data/node_sbert_embeddings.npz"
    llm_paths = {
        #INSTERT PATHS FOR LLM EMBEDINGS HERE
    }

    # Check that embedding files exist
    for path in [sbert_path] + list(llm_paths.values()):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Embedding file not found: {path}")

    # Evaluate SBERT embeddings
    logging.info("Evaluating SBERT embeddings (supervised)...")
    sbert_results = evaluate_embeddings_supervised(sbert_path, graph)
    logging.info(f"SBERT Results: {sbert_results}")

    # Evaluate LLM embeddings for each layer
    llm_results = {}
    for layer_name, path in llm_paths.items():
        logging.info(f"Evaluating LLM embeddings for {layer_name} (supervised)...")
        results = evaluate_embeddings_supervised(path, graph)
        llm_results[layer_name] = results
        logging.info(f"{layer_name} Results: {results}")

    # Print comparison table
    print("\nLink Prediction Performance Comparison (Supervised):")
    print("Model\t\t\tAUC-ROC\t\tAverage Precision")
    print("-" * 50)
    print(f"SBERT\t\t\t{sbert_results['auc_roc']:.4f}\t\t{sbert_results['average_precision']:.4f}")
    for layer_name, results in llm_results.items():
        print(f"LLM {layer_name}\t\t{results['auc_roc']:.4f}\t\t{results['average_precision']:.4f}")

if __name__ == "__main__":
    main()
