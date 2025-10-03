#!/usr/bin/env python3
"""
Quick check of vector normalization in existing database
"""

import numpy as np
from pymilvus import MilvusClient

DB_FILE = "./milvus_lite.db"
COLLECTION_NAME = "doc_sections"

print("="*60)
print("Vector Normalization Check")
print("="*60)

try:
    client = MilvusClient(DB_FILE)
    
    if not client.has_collection(COLLECTION_NAME):
        print("ERROR: Collection not found!")
        print("Please run: python scripts/create-vecdb.py")
        exit(1)
    
    # Query sample vectors
    print("\nFetching sample vectors...")
    results = client.query(
        collection_name=COLLECTION_NAME,
        filter="id >= 0",
        output_fields=["id", "file_name", "vector"],
        limit=10
    )
    
    if not results:
        print("No vectors found in database")
        exit(1)
    
    print(f"Found {len(results)} sample vectors\n")
    
    # Check normalization
    print("Vector Normalization Status:")
    print("-" * 60)
    
    norms = []
    for i, result in enumerate(results):
        vec = np.array(result['vector'])
        norm = np.linalg.norm(vec)
        norms.append(norm)
        
        status = "✓ OK" if abs(norm - 1.0) < 0.01 else "✗ NOT NORMALIZED"
        print(f"{i+1}. {result['file_name'][:40]}")
        print(f"   Norm: {norm:.6f} {status}")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    avg_norm = np.mean(norms)
    std_norm = np.std(norms)
    
    print(f"Average norm: {avg_norm:.6f}")
    print(f"Std deviation: {std_norm:.6f}")
    print(f"Min norm: {min(norms):.6f}")
    print(f"Max norm: {max(norms):.6f}")
    
    if abs(avg_norm - 1.0) < 0.01 and std_norm < 0.01:
        print("\n✓ VECTORS ARE NORMALIZED")
        print("  Database is ready for use")
    else:
        print(f"\n✗ VECTORS ARE NOT NORMALIZED")
        print(f"  Expected norm: 1.0")
        print(f"  Actual average: {avg_norm:.6f}")
        print("\n  ACTION REQUIRED:")
        print("  1. Delete old database: rm milvus_lite.db")
        print("  2. Rebuild with normalization: python scripts/create-vecdb.py")
        print("\n  The updated create-vecdb.py now includes normalize_embeddings=True")

except Exception as e:
    print(f"\nError: {e}")
    import traceback
    traceback.print_exc()

