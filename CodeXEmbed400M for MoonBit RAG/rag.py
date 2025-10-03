#!/usr/bin/env python3
"""
MoonBit Documentation RAG System
Using SFR-Embedding-Code model with Milvus Lite (no Docker required)
"""

import os
from sentence_transformers import SentenceTransformer
from pymilvus import MilvusClient
import warnings

warnings.filterwarnings('ignore')


class MoonBitRAG:
    def __init__(self, db_file="./milvus_lite.db", collection_name="doc_sections"):
        """Initialize MoonBit RAG retrieval system"""
        if not os.path.exists(db_file):
            raise FileNotFoundError(
                f"Database file {db_file} not found! "
                f"Please run: python scripts/create-vecdb.py"
            )
        
        self.client = MilvusClient(db_file)
        self.collection_name = collection_name
        
        if not self.client.has_collection(collection_name):
            raise Exception(
                f"Collection {collection_name} not found! "
                f"Please run: python scripts/create-vecdb.py"
            )
        
        self.model = self._load_model()
        print("RAG system initialized successfully!")
    
    def _load_model(self):
        """Load local SFR-Embedding-Code model with proper pooling configuration"""
        # Customize this path to your local SFR-Embedding-Code model directory
        model_path = os.getenv("MODEL_PATH", "./models")
        
        if not os.path.exists(model_path):
            raise Exception(f"Model path not found: {model_path}")
        
        print("Loading SFR-Embedding-Code model...")
        
        # Enable offline mode
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
        
        # Ensure pooling configuration exists (same as create-vecdb.py)
        import json
        pooling_dir = os.path.join(model_path, '1_Pooling')
        pooling_config_path = os.path.join(pooling_dir, 'config.json')
        
        # Create pooling directory if not exists
        if not os.path.exists(pooling_dir):
            os.makedirs(pooling_dir)
        
        # Get correct embedding dimension
        config_path = os.path.join(model_path, 'config.json')
        with open(config_path, 'r') as f:
            main_config = json.load(f)
        
        embedding_dim = main_config.get('hidden_size', 1024)
        
        # Standard SFR pooling configuration
        correct_pooling_config = {
            "word_embedding_dimension": embedding_dim,
            "pooling_mode_cls_token": True,
            "pooling_mode_mean_tokens": False,
            "pooling_mode_max_tokens": False,
            "pooling_mode_mean_sqrt_len_tokens": False,
            "pooling_mode_weightedmean_tokens": False,
            "pooling_mode_lasttoken": False,
            "include_prompt": True
        }
        
        # Check and update pooling config if needed
        config_needs_update = True
        if os.path.exists(pooling_config_path):
            try:
                with open(pooling_config_path, 'r') as f:
                    existing_config = json.load(f)
                
                if (existing_config.get('word_embedding_dimension') == embedding_dim and
                    existing_config.get('pooling_mode_cls_token') == True and
                    existing_config.get('include_prompt') == True):
                    config_needs_update = False
            except:
                pass
        
        # Update config if needed
        if config_needs_update:
            with open(pooling_config_path, 'w') as f:
                json.dump(correct_pooling_config, f, indent=2)
        
        # Load model with trust_remote_code
        model = SentenceTransformer(
            model_path, 
            trust_remote_code=True,
            cache_folder=model_path,
            local_files_only=True
        )
        
        print(f"Model loaded successfully (vector dim: {embedding_dim})")
        
        return model
    
    def search(self, query, top_k=5):
        """
        Semantic search for MoonBit documentation
        
        Args:
            query (str): Search query (supports English, Chinese, code snippets)
            top_k (int): Number of results to return
            
        Returns:
            list: Search results with file names, scores, and content
        """
        # Generate query embedding with normalization (must match create-vecdb.py)
        query_vector = self.model.encode(
            query, 
            convert_to_numpy=True,
            normalize_embeddings=True  # CRITICAL: Must normalize for cosine metric
        )
        
        try:
            # Get actual metric type from collection (same as search-vecdb.py)
            actual_metric = "COSINE"  # Default
            index_type = "HNSW"  # Default
            
            try:
                from pymilvus import Collection, connections
                
                # Only connect if not already connected
                if not connections.has_connection("default"):
                    connections.connect("default", uri=self.client._uri)
                
                collection = Collection(self.collection_name)
                index_info = collection.index()
                
                # Get actual metric and index type
                actual_metric = index_info.params.get('metric_type', 'COSINE')
                index_type = index_info.params.get('index_type', 'HNSW')
                
            except Exception as e:
                pass  # Use defaults
            
            # Set search parameters based on actual metric and index type
            search_params = {"metric_type": actual_metric}
            
            if index_type == 'HNSW':
                search_params["params"] = {"ef": 128}
            elif index_type in ['IVF_FLAT', 'IVF_PQ']:
                search_params["params"] = {"nprobe": 16}
            
            # Search in Milvus Lite
            results = self.client.search(
                collection_name=self.collection_name,
                data=[query_vector.tolist()],
                limit=top_k,
                output_fields=["file_name", "content"],
                search_params=search_params
            )
            
            # Format results with proper score calculation
            search_results = []
            for hit in results[0]:
                # Format score based on actual metric
                if actual_metric == "COSINE":
                    # For cosine: similarity = 1 - distance, clamped to [0,1]
                    similarity = max(0.0, min(1.0, 1.0 - hit['distance']))
                elif actual_metric == "L2":
                    # For L2: use raw distance (smaller = more similar)
                    similarity = hit['distance']
                elif actual_metric == "IP":
                    # For Inner Product: use raw score (higher = more similar)
                    similarity = hit['distance']
                else:
                    # Unknown metric: use raw distance
                    similarity = hit['distance']
                
                result = {
                    "file_name": hit['entity']['file_name'],
                    "similarity": similarity,
                    "metric": actual_metric,
                    "content": hit['entity']['content'],
                    "preview": hit['entity']['content'][:300] + "..." 
                              if len(hit['entity']['content']) > 300 
                              else hit['entity']['content']
                }
                search_results.append(result)
            
            return search_results
            
        except Exception as e:
            print(f"Search error: {e}")
            return []
    
    def interactive_search(self):
        """Interactive search mode"""
        print("\n" + "=" * 60)
        print("MoonBit Documentation RAG System")
        print("=" * 60)
        print("Powered by SFR-Embedding-Code-400M_R")
        print("Enter your query (type 'quit' to exit)\n")
        
        while True:
            query = input("Query: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not query:
                continue
            
            print(f"\nSearching: {query}")
            print("-" * 60)
            
            results = self.search(query, top_k=3)
            
            if not results:
                print("No results found")
                continue
            
            for i, result in enumerate(results, 1):
                print(f"\nResult {i}:")
                print(f"  File: {result['file_name']}")
                print(f"  Similarity: {result['similarity']:.4f}")
                print(f"  Preview: {result['preview']}")
                print("-" * 40)
            
            print()


def demo_search():
    """Demo search with sample queries"""
    rag = MoonBitRAG()
    
    demo_queries = [
        "How to define a function?",
        "function definition syntax",
        "Variable declaration",
        "let x = 42",
        "Error handling in MoonBit",
        "Async programming",
        "Type system"
    ]
    
    print("\n" + "=" * 60)
    print("Demo Search Results")
    print("=" * 60)
    
    for query in demo_queries:
        print(f"\nQuery: {query}")
        print("-" * 30)
        
        results = rag.search(query, top_k=2)
        
        if results:
            for i, result in enumerate(results, 1):
                print(f"{i}. {result['file_name']}")
                print(f"   Similarity: {result['similarity']:.4f}")
                print(f"   Preview: {result['preview'][:150]}...")
        else:
            print("No results found")
        
        print()


def check_database():
    """Check database status"""
    try:
        db_file = "./milvus_lite.db"
        client = MilvusClient(db_file)
        
        if client.has_collection("doc_sections"):
            print("Database Status:")
            print(f"  Collection: doc_sections (OK)")
            print(f"  Database: {os.path.abspath(db_file)}")
        else:
            print("ERROR: Collection not found")
            print("Please run: python scripts/create-vecdb.py")
            
    except Exception as e:
        print(f"Database check failed: {e}")


if __name__ == "__main__":
    try:
        # Check database
        check_database()
        print()
        
        # Create RAG instance
        rag = MoonBitRAG()
        
        # Select mode
        print("\nSelect Mode:")
        print("  [1] Demo Search (Recommended)")
        print("  [2] Interactive Search")
        
        mode = input("\nMode (1/2, default 1): ").strip()
        
        if mode == "2":
            rag.interactive_search()
        else:
            demo_search()
        
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nSetup Steps:")
        print("  1. python scripts/unity.py")
        print("  2. node scripts/extract.js processed")
        print("  3. python scripts/complement.py")
        print("  4. python scripts/create-vecdb.py")
        print("  5. python rag_example.py")
        
    except Exception as e:
        print(f"\nError: {e}")
        print("\nPlease ensure:")
        print("  1. Document processing is complete")
        print("  2. Database is created (create-vecdb.py)")
        print("  3. Dependencies installed (pip install -r requirements.txt)")
