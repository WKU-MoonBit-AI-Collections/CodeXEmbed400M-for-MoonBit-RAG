import os
from sentence_transformers import SentenceTransformer
from pymilvus import MilvusClient, FieldSchema, CollectionSchema, DataType, Collection, connections
import glob
import json
import warnings

# Configuration
COLLECTION_NAME = "doc_sections"
DB_FILE = "./milvus_lite.db"
# Customize this path to your local SFR-Embedding-Code model directory
MODEL_PATH = os.getenv("MODEL_PATH", "./models")

# Suppress model loading warnings
warnings.filterwarnings('ignore')

client = MilvusClient(DB_FILE)

print("=" * 60)
print("MoonBit Documentation Vector Database Creator")
print("=" * 60)


def load_model():
    """Load SFR-Embedding-Code model from local path"""
    if not os.path.exists(MODEL_PATH):
        raise Exception(f"Model path not found: {MODEL_PATH}")
    
    print("\n1. Loading SFR-Embedding-Code model...")
    
    # Enable offline mode
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    
    # Ensure pooling configuration exists
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
            json.dump(pooling_config, f, indent=4)
        print(f"   Pooling config created: {pooling_config_path}")
    
    # Load model with trust_remote_code
    model = SentenceTransformer(
        MODEL_PATH, 
        trust_remote_code=True,
        cache_folder=MODEL_PATH,
        local_files_only=True
    )
    
    # Test model and get vector dimension
    test_embedding = model.encode("test")
    vector_dim = len(test_embedding)
    
    print(f"   Model loaded successfully")
    print(f"   Vector dimension: {vector_dim}")
    
    return model, vector_dim


def create_collection(vector_dim):
    """Create Milvus collection with schema"""
    print("\n2. Creating collection...")
    
    if client.has_collection(COLLECTION_NAME):
        client.drop_collection(COLLECTION_NAME)
        print(f"   Dropped existing collection: {COLLECTION_NAME}")
    
    # Define collection schema
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=vector_dim),
        FieldSchema(name="file_name", dtype=DataType.VARCHAR, max_length=500),
        FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535)
    ]
    
    schema = CollectionSchema(fields, "MoonBit documentation sections")
    
    # Create collection
    client.create_collection(
        collection_name=COLLECTION_NAME,
        schema=schema
    )
    
    print(f"   Collection created: {COLLECTION_NAME}")
    print(f"   Vector dimension: {vector_dim}")


def create_index():
    """Create IVF_FLAT index for efficient ANN search (Milvus Lite compatible)"""
    print("\n3. Creating IVF_FLAT index...")
    
    try:
        connections.connect("default", uri=DB_FILE)
        collection = Collection(COLLECTION_NAME)
        
        # IVF_FLAT index parameters (Milvus Lite supported)
        index_params = {
            "metric_type": "COSINE",
            "index_type": "IVF_FLAT",
            "params": {
                "nlist": 128  # Number of cluster units
            }
        }
        
        # Create index
        collection.create_index(
            field_name="vector",
            index_params=index_params,
            index_name="vector_ivf_index"
        )
        
        # Load collection to activate index
        collection.load()
        
        # Verify index
        index_info = collection.index()
        print(f"   Index created successfully!")
        print(f"   Type: {index_info.params.get('index_type', 'N/A')}")
        print(f"   Metric: {index_info.params.get('metric_type', 'N/A')}")
        print(f"   nlist: {index_info.params.get('params', {}).get('nlist', 'N/A')}")
        print(f"   Note: Using IVF_FLAT (Milvus Lite compatible)")
        
        return True
        
    except Exception as e:
        print(f"   IVF_FLAT index creation failed: {e}")
        print("   Continuing without explicit index...")
        return False


def process_files(model):
    """Process markdown files and insert into database"""
    print("\n4. Processing documents...")
    
    markdown_files = glob.glob("final_sections/*.md")
    
    if not markdown_files:
        print("   ERROR: No files found in final_sections/")
        print("   Please run preprocessing steps first")
        return False
    
    print(f"   Found {len(markdown_files)} files")
    
    entities = []
    processed_count = 0
    
    for file_path in markdown_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                
                # Truncate if content exceeds VARCHAR limit
                if len(content) > 65000:
                    content = content[:65000] + "...[truncated]"
                
                file_name = os.path.basename(file_path)
                if len(file_name) > 500:
                    file_name = file_name[:500]
                
                # Generate embedding with normalization for cosine similarity
                # CRITICAL: Normalize vectors for proper cosine distance calculation
                embedding = model.encode(
                    content, 
                    convert_to_numpy=True,
                    normalize_embeddings=True  # L2 normalization for cosine metric
                )
                
                # Verify normalization (vector norm should be ~1.0)
                import numpy as np
                vec_norm = np.linalg.norm(embedding)
                
                # Debug: Print first file's info
                if processed_count == 0:
                    print(f"\n   First file normalization check:")
                    print(f"   - File: {file_name}")
                    print(f"   - Vector norm: {vec_norm:.6f}")
                    print(f"   - Vector sample: [{embedding[0]:.4f}, {embedding[1]:.4f}, ..., {embedding[-1]:.4f}]")
                    print(f"   - Expected norm: 1.0")
                
                if abs(vec_norm - 1.0) > 0.01:
                    print(f"   WARNING: Vector norm = {vec_norm:.4f} for {file_name}")
                
                # Prepare entity (no ID field, using auto_id=True)
                entity = {
                    "vector": embedding.tolist(),
                    "file_name": file_name,
                    "content": content
                }
                entities.append(entity)
                processed_count += 1
                
                if processed_count % 100 == 0:
                    print(f"   Processed {processed_count}/{len(markdown_files)} files...")
        
        except Exception as e:
            print(f"   Error processing {file_path}: {e}")
            continue
    
    print(f"\n   Statistics:")
    print(f"   - Files processed: {len(entities)}")
    print(f"   - Vector dimension: {len(entities[0]['vector']) if entities else 0}")
    
    # Insert data
    try:
        print(f"\n   Inserting {len(entities)} documents...")
        client.insert(collection_name=COLLECTION_NAME, data=entities)
        print(f"   Successfully inserted {len(entities)} documents")
        
        # Load collection into memory
        client.load_collection(collection_name=COLLECTION_NAME)
        print(f"   Collection loaded into memory")
        print(f"   Database file: {os.path.abspath(DB_FILE)}")
        
        return True
        
    except Exception as e:
        print(f"   ERROR: Data insertion failed: {e}")
        return False


def verify_search(model):
    """Verify search functionality"""
    print("\n" + "=" * 60)
    print("Verification Test")
    print("=" * 60)
    
    try:
        test_query = "function definition"
        print(f"Testing query: '{test_query}'")
        
        # CRITICAL: Must normalize query vector to match database vectors
        query_vector = model.encode(
            test_query, 
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        search_results = client.search(
            collection_name=COLLECTION_NAME,
            data=[query_vector.tolist()],
            limit=2,
            output_fields=["file_name"],
            search_params={"metric_type": "COSINE", "params": {"ef": 128}}
        )
        
        if search_results and len(search_results[0]) > 0:
            first_result = search_results[0][0]
            similarity = 1 - first_result['distance']
            print(f"Search successful!")
            print(f"  Top result similarity: {similarity:.4f}")
            print(f"  File: {first_result['entity']['file_name']}")
            return True
        else:
            print("Search returned no results")
            return False
            
    except Exception as e:
        print(f"Verification failed: {e}")
        return False


if __name__ == "__main__":
    try:
        # Load model
        model, vector_dim = load_model()
        
        # Create collection
        create_collection(vector_dim)
        
        # Create index
        create_index()
        
        # Process files
        if not process_files(model):
            print("\nERROR: File processing failed")
            exit(1)
        
        # Verify search
        print("\n" + "=" * 60)
        print("Database Creation Complete")
        print("=" * 60)
        
        if verify_search(model):
            print("\nDatabase is ready for use!")
            print("\nNext steps:")
            print("  python scripts/search-vecdb.py  # Test search")
            print("  python rag_example.py           # Run RAG application")
        else:
            print("\nWarning: Search verification failed")
            print("Database may have issues, please check logs")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        print("Please check the error and try again")
