import os
from sentence_transformers import SentenceTransformer
from pymilvus import MilvusClient, Collection, connections
import warnings

# Configuration
DB_FILE = "./milvus_lite.db"
COLLECTION_NAME = "doc_sections"
# Customize this path to your local SFR-Embedding-Code model directory
MODEL_PATH = os.getenv("MODEL_PATH", "./models")

# Suppress warnings
warnings.filterwarnings('ignore')

client = MilvusClient(DB_FILE)

print("=" * 60)
print("MoonBit Documentation Search Tool")
print("=" * 60)


def load_model():
    """Load SFR-Embedding-Code model (same config as creation)"""
    if not os.path.exists(MODEL_PATH):
        raise Exception(f"Model path not found: {MODEL_PATH}")
    
    print("\n1. Loading SFR-Embedding-Code model...")
    
    # Enable offline mode
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    
    # Ensure pooling configuration exists (same as create-vecdb.py)
    pooling_config_path = os.path.join(MODEL_PATH, "1_Pooling", "config.json")
    if not os.path.exists(pooling_config_path):
        print("   Creating pooling configuration...")
        os.makedirs(os.path.dirname(pooling_config_path), exist_ok=True)
        
        pooling_config = {
            "word_embedding_dimension": 1024,
            "pooling_mode_cls_token": True,
            "pooling_mode_mean_tokens": False,
            "pooling_mode_max_tokens": False,
            "pooling_mode_mean_sqrt_len_tokens": False,
            "pooling_mode_weightedmean_tokens": False,
            "pooling_mode_lasttoken": False,
            "include_prompt": True
        }
        
        with open(pooling_config_path, 'w', encoding='utf-8') as f:
            import json
            json.dump(pooling_config, f, indent=4)
        print(f"   Pooling config created: {pooling_config_path}")
    
    # Load model
    model = SentenceTransformer(
        MODEL_PATH, 
        trust_remote_code=True,
        cache_folder=MODEL_PATH,
        local_files_only=True
    )
    
    print("   Model loaded successfully")
    return model


# Load model
model = load_model()

# Check collection
print("\n2. Checking collection...")
if not client.has_collection(COLLECTION_NAME):
    print(f"   ERROR: Collection '{COLLECTION_NAME}' not found!")
    print("   Please run: python scripts/create-vecdb.py")
    exit(1)

# Check index type
try:
    connections.connect("default", uri=DB_FILE)
    collection = Collection(COLLECTION_NAME)
    
    index_info = collection.index()
    index_type = index_info.params.get('index_type', 'Unknown')
    metric_type = index_info.params.get('metric_type', 'Unknown')
    
    print(f"   Collection found")
    print(f"   Index type: {index_type}")
    print(f"   Metric: {metric_type}")
    
    if index_type == 'HNSW':
        params = index_info.params.get('params', {})
        print(f"   M: {params.get('M', 'N/A')}")
        print(f"   Search will use ef=128")
    elif index_type == 'IVF_FLAT':
        params = index_info.params.get('params', {})
        print(f"   nlist: {params.get('nlist', 'N/A')}")
        print(f"   Search will use nprobe=16")
    
    collection.load()
    
except Exception as e:
    print(f"   Collection info: {e}")

print("\n3. Ready to search!")


def search_similar_docs(query_text, top_k=5):
    """Search for similar documents using ANN"""
    # Generate query vector with normalization (must match create-vecdb.py)
    query_vector = model.encode(
        query_text, 
        convert_to_numpy=True,
        normalize_embeddings=True  # CRITICAL: Must normalize for cosine metric
    )
    query_vector_list = query_vector.tolist()
    
    # Get actual metric type from collection
    actual_metric = "COSINE"  # Default
    index_type = "HNSW"  # Default
    
    try:
        # Only connect if not already connected
        if not connections.has_connection("default"):
            connections.connect("default", uri=DB_FILE)
        
        collection = Collection(COLLECTION_NAME)
        index_info = collection.index()
        
        # Get actual metric and index type from collection
        actual_metric = index_info.params.get('metric_type', 'COSINE')
        index_type = index_info.params.get('index_type', 'HNSW')
        
        print(f"Using index: {index_type}, metric: {actual_metric}")
        
    except Exception as e:
        print(f"Warning: Could not read index info, using defaults: {e}")
    
    # Set search parameters based on actual metric and index type
    search_params = {"metric_type": actual_metric}
    
    if index_type == 'HNSW':
        search_params["params"] = {"ef": 128}
    elif index_type in ['IVF_FLAT', 'IVF_PQ']:
        search_params["params"] = {"nprobe": 16}
    # For FLAT index, no additional params needed
    
    # Perform ANN search
    results = client.search(
        collection_name=COLLECTION_NAME,
        data=[query_vector_list],
        limit=top_k,
        output_fields=["file_name", "content"],
        search_params=search_params
    )
    
    # Display results with proper score formatting
    print(f"\nQuery: {query_text}")
    print(f"\nFound {len(results[0])} similar documents:")
    
    for i, hit in enumerate(results[0]):
        # Format score based on actual metric
        if actual_metric == "COSINE":
            # For cosine: similarity = 1 - distance, but clamp to [0,1]
            similarity_score = max(0.0, min(1.0, 1.0 - hit['distance']))
            score_label = "Cosine Similarity"
        elif actual_metric == "L2":
            # For L2: smaller distance = more similar
            similarity_score = hit['distance']
            score_label = "L2 Distance"
        elif actual_metric == "IP":
            # For Inner Product: higher = more similar
            similarity_score = hit['distance']
            score_label = "Inner Product"
        else:
            # Unknown metric: show raw distance
            similarity_score = hit['distance']
            score_label = "Distance"
        
        print(f"\nResult {i+1}:")
        print(f"  File: {hit['entity']['file_name']}")
        print(f"  {score_label}: {similarity_score:.4f}")
        print(f"  Content: {hit['entity']['content'][:200]}...")
    
    return results[0]


def interactive_search():
    """Interactive search mode"""
    print("\n" + "=" * 60)
    print("Interactive Search Mode")
    print("=" * 60)
    print("Enter your query (type 'quit' to exit)")
    
    while True:
        try:
            query = input("\nQuery: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not query:
                continue
            
            print("-" * 60)
            search_similar_docs(query, top_k=3)
            print("-" * 60)
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Search error: {e}")


def evaluate_retrieval_quality():
    """Evaluate retrieval quality with standard IR metrics"""
    print("\n" + "=" * 60)
    print("Retrieval Quality Evaluation")
    print("=" * 60)
    
    # Test queries with expected relevant keywords (ground truth indicators)
    test_cases = [
        {
            "query": "function definition",
            "relevant_keywords": ["function", "def", "fn", "method", "procedure"],
            "description": "Function-related documentation"
        },
        {
            "query": "variable declaration", 
            "relevant_keywords": ["variable", "var", "let", "const", "declaration"],
            "description": "Variable declaration syntax"
        },
        {
            "query": "error handling",
            "relevant_keywords": ["error", "exception", "try", "catch", "handle"],
            "description": "Error handling mechanisms"
        },
        {
            "query": "type system",
            "relevant_keywords": ["type", "interface", "struct", "enum", "generic"],
            "description": "Type system features"
        }
    ]
    
    total_queries = len(test_cases)
    recall_at_k = {1: 0, 3: 0, 5: 0}
    mrr_scores = []
    
    for i, test_case in enumerate(test_cases, 1):
        query = test_case["query"]
        relevant_keywords = test_case["relevant_keywords"]
        
        print(f"\n[{i}/{total_queries}] Testing: '{query}'")
        print(f"Expected: {test_case['description']}")
        
        # Get search results (suppress output temporarily)
        import io
        import sys
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        
        try:
            results = search_similar_docs(query, top_k=5)
        finally:
            sys.stdout = old_stdout
        
        # Evaluate relevance based on content matching
        relevant_positions = []
        for pos, result in enumerate(results):
            content = result['entity']['content'].lower()
            file_name = result['entity']['file_name'].lower()
            
            # Check if any relevant keywords appear in content or filename
            is_relevant = any(keyword in content or keyword in file_name 
                            for keyword in relevant_keywords)
            
            if is_relevant:
                relevant_positions.append(pos + 1)  # 1-indexed
        
        # Calculate metrics
        if relevant_positions:
            # Recall@K
            for k in [1, 3, 5]:
                if any(pos <= k for pos in relevant_positions):
                    recall_at_k[k] += 1
            
            # MRR (Mean Reciprocal Rank)
            first_relevant = min(relevant_positions)
            mrr_scores.append(1.0 / first_relevant)
        else:
            mrr_scores.append(0.0)
        
        # Show top results with relevance assessment
        print(f"Results (top 3):")
        for j, result in enumerate(results[:3]):
            content = result['entity']['content'].lower()
            file_name = result['entity']['file_name'].lower()
            is_relevant = any(keyword in content or keyword in file_name 
                            for keyword in relevant_keywords)
            
            relevance_mark = "✓" if is_relevant else "✗"
            
            # Get score based on metric type
            if hasattr(result, 'distance'):
                score = result['distance']
            else:
                score = result.get('score', 'N/A')
            
            print(f"  {j+1}. {relevance_mark} {result['entity']['file_name'][:50]}...")
            print(f"     Score: {score}")
    
    # Calculate final metrics
    print(f"\n" + "=" * 60)
    print("RETRIEVAL QUALITY METRICS")
    print("=" * 60)
    
    for k in [1, 3, 5]:
        recall_k = recall_at_k[k] / total_queries
        print(f"Recall@{k}: {recall_k:.3f} ({recall_at_k[k]}/{total_queries})")
    
    mrr = sum(mrr_scores) / len(mrr_scores)
    print(f"MRR (Mean Reciprocal Rank): {mrr:.3f}")
    
    # Quality assessment
    print(f"\nQuality Assessment:")
    if recall_at_k[5] / total_queries >= 0.8:
        print(" Good: >80% queries have relevant results in top-5")
    elif recall_at_k[5] / total_queries >= 0.6:
        print(" Fair: 60-80% queries have relevant results in top-5")
    else:
        print(" Poor: <60% queries have relevant results in top-5")
    
    if mrr >= 0.7:
        print(" Good: High precision (MRR ≥ 0.7)")
    elif mrr >= 0.5:
        print("Fair: Moderate precision (MRR ≥ 0.5)")
    else:
        print(" Poor: Low precision (MRR < 0.5)")


def demo_search():
    """Demo search with sample queries"""
    print("\n" + "=" * 60)
    print("Demo Search")
    print("=" * 60)
    
    # Test queries (English and Chinese)
    test_queries = [
        ("function definition", "函数定义"),
        ("variable declaration", "变量声明"),
        ("error handling", "错误处理"),
        ("type system", "类型系统")
    ]
    
    for en_query, cn_query in test_queries:
        print(f"\n{'='*80}")
        print(f"Query comparison: '{en_query}' vs '{cn_query}'")
        print(f"{'='*80}")
        
        # English query
        print(f"\n--- English: {en_query} ---")
        search_similar_docs(en_query, top_k=2)
        
        # Chinese query
        print(f"\n--- Chinese: {cn_query} ---")
        search_similar_docs(cn_query, top_k=2)


if __name__ == "__main__":
    try:
        # Choose mode
        print("\n" + "=" * 60)
        print("Select Mode")
        print("=" * 60)
        print("1. Demo Search (Recommended)")
        print("2. Interactive Search")
        print("3. Retrieval Quality Evaluation")
        
        choice = input("\nMode (1/2/3, default 1): ").strip()
        
        if choice == "2":
            interactive_search()
        elif choice == "3":
            evaluate_retrieval_quality()
        else:
            demo_search()
        
    except Exception as e:
        print(f"\nError: {e}")
        print("Please check logs and try again")
