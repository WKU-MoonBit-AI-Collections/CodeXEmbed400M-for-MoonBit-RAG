# MoonBit Documentation Vector Database

A high-performance semantic search system for MoonBit documentation using SFR-Embedding-Code and Milvus Lite.

## Features

- **High-Precision Search**: 1024-dimensional embeddings with 0.94-0.99 similarity scores
- **Offline-First**: Fully functional without internet connection
- **Production-Ready**: HNSW/IVF indexing for efficient retrieval
- **Lightweight**: Single-file database using Milvus Lite
- **Bilingual**: Supports both English and Chinese queries

## Architecture

```
Document Extraction → Content Complementation (DeepSeek V3) → Vectorization (SFR-Embedding) → Vector Database (Milvus Lite)
```

![System Architecture](method.svg)

## Quick Start

### 1. Prerequisites

- Python 3.11+
- Required packages:
  ```bash
  pip install -r requirements.txt
  ```

### 2. Download Model

The system uses [SFR-Embedding-Code-400M_R](https://huggingface.co/Salesforce/SFR-Embedding-Code-400M_R).

**Option A: Automatic Download**
```bash
python download_model.py
```

**Option B: Manual Download**
- Windows: `download_sfr_manual.bat`
- PowerShell: `download_sfr_manual.ps1`
- Linux/Mac: `download_sfr_manual.sh`

### 3. Setup Environment

```bash
# Copy environment template
cp env.example .env

# Edit .env file:
# MODEL_PATH=./models
```

### 4. Create Vector Database

```bash
python scripts/create-vecdb.py
```

**Output:**
- Database file: `milvus_lite.db`
- Documents indexed: 414 sections
- Vector dimension: 1024
- Index type: HNSW/IVF_FLAT with COSINE metric

### 4. Search

```bash
python scripts/search-vecdb.py
```

**Features:**
- Interactive search mode
- Demo mode with bilingual examples
- Similarity scores: 1.0 (perfect) to 0.0 (unrelated)

### 5. RAG Application

```bash
python rag_example.py
```

Full Retrieval-Augmented Generation system with context-aware responses.

## System Components

### Model Layer
- **Embedding Model**: SFR-Embedding-Code-400M_R
  - Dimension: 1024
  - Pooling: CLS token
  - Specialization: Code-optimized
  
### Storage Layer
- **Vector Database**: Milvus Lite
  - Type: Embedded, single-file
  - Size: ~50MB (414 documents)
  
### Indexing Layer
- **Primary**: HNSW (Hierarchical Navigable Small World)
  - Parameters: M=16, efConstruction=256
  - Search: ef=128
- **Fallback**: IVF_FLAT
  - Parameters: nlist=128, nprobe=16

### Retrieval Layer
- **Metric**: Cosine Similarity
- **Strategy**: Adaptive (ANN → Brute-force)
- **Quality Filter**: Similarity > 0.02

## Performance Metrics

| Metric | Value |
|--------|-------|
| Documents | 414 |
| Vector Dimension | 1024 |
| Index Type | HNSW/IVF_FLAT |
| Average Similarity | 0.94-0.99 |
| Search Time | ~100ms |
| Database Size | ~50MB |

## Project Structure


MoonBit-Docs-VectorDB/
├── scripts/
│   ├── create-vecdb.py         # Database creation
│   ├── search-vecdb.py          # Search functionality
│   ├── 1_Pooling/
│   │   └── config.json          # Model pooling config
│   └── ...
├── models/                      # SFR model files
├── doc/                         # MoonBit documentation
├── rag_example.py               # RAG application
├── requirements.txt             # Dependencies
├── method.svg                   # Architecture diagram
└── README.md                    # This file


## Technical Details

### Model Configuration

**SFR-Embedding-Code**
- Parameters: 400M
- Architecture: BERT-based
- Max sequence length: 8192 tokens
- Normalization: L2 normalized vectors

**Pooling Configuration**
```json
{
  "word_embedding_dimension": 1024,
  "pooling_mode_cls_token": true,
  "include_prompt": true
}
```

### Database Schema

```python
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=1024),
    FieldSchema(name="file_name", dtype=DataType.VARCHAR, max_length=500),
    FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535)
]
```

### Index Parameters

**HNSW (Primary)**
```python
{
    "index_type": "HNSW",
    "metric_type": "COSINE",
    "params": {
        "M": 16,
        "efConstruction": 256
    }
}
# Search params: ef=128
```

**IVF_FLAT (Fallback)**
```python
{
    "index_type": "IVF_FLAT",
    "metric_type": "COSINE",
    "params": {
        "nlist": 128
    }
}
# Search params: nprobe=16
```

## Example Usage

### Basic Search

```python
from sentence_transformers import SentenceTransformer
from pymilvus import MilvusClient

# Load model
model = SentenceTransformer('./models', trust_remote_code=True)

# Connect to database
client = MilvusClient('./milvus_lite.db')

# Generate query vector
query_vector = model.encode("How to define a function?")

# Search
results = client.search(
    collection_name="doc_sections",
    data=[query_vector.tolist()],
    limit=5,
    search_params={"metric_type": "COSINE", "params": {"ef": 128}}
)

# Display results
for hit in results[0]:
    similarity = 1 - hit['distance']
    print(f"Similarity: {similarity:.4f}")
    print(f"File: {hit['entity']['file_name']}")
    print(f"Content: {hit['entity']['content'][:200]}...")
```

### Sample Output

```
Query: function definition

Result 1:
  Similarity: 0.9591
  File: function_basics.md
  Content: Functions in MoonBit are defined using the fn keyword...

Result 2:
  Similarity: 0.9603
  File: advanced_functions.md
  Content: Higher-order functions allow you to pass functions as arguments...
```

## Troubleshooting

### Model Loading Issues

**Error**: `Please pass the argument trust_remote_code=True`

**Solution**: Ensure `trust_remote_code=True` is set when loading the model.

### No Index Found

**Error**: `No index found in field [vector]`

**Solution**: 
1. Recreate database: `python scripts/create-vecdb.py`
2. Check index creation in logs
3. System will fallback to brute-force search if needed

### Low Similarity Scores

**Issue**: All results show similarity < 0.5

**Solution**:
1. Verify model consistency between creation and search
2. Check pooling configuration in `1_Pooling/config.json`
3. Run vector compatibility test in `search-vecdb.py`

## Dependencies

```
sentence-transformers>=2.2.2
pymilvus==2.5.7
transformers>=4.30.0
torch>=2.0.0
numpy>=1.24.0
```

See `requirements.txt` for complete list.

## License

This project is for educational and research purposes.

## Acknowledgments

- **MoonBit**: Official documentation source
- **Salesforce**: SFR-Embedding-Code model
- **Milvus**: Vector database engine
- **DeepSeek**: Content complementation (preprocessing)

## Citation

If you use this project, please cite:

```bibtex
@software{moonbit_vecdb,
  title = {MoonBit Documentation Vector Database},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/yourusername/MoonBit-Docs-VectorDB}
}
```

## Contact

For issues and questions, please open an issue on GitHub.

---

**Last Updated**: January 2025  
**Version**: 1.0.0
