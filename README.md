# CodeXEmbed400M-for-MoonBit-RAG

# MoonBit Documentation Vector Database

A high-performance semantic search system for MoonBit documentation using SFR-Embedding-Code and Milvus Lite.

## Features

- **High-Precision Search**: 1024-dimensional embeddings with 0.94-0.99 similarity scores
- **Offline-First**: Fully functional without internet connection
- **Production-Ready**: HNSW/IVF indexing for efficient retrieval
- **Lightweight**: Single-file database using Milvus Lite
- **Bilingual**: Supports both English and Chinese queries

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    MoonBit RAG System Architecture               │
└─────────────────────────────────────────────────────────────────┘

┌──────────────────┐       ┌──────────────────┐       ┌──────────────────┐
│  Raw Documents   │──────▶│  Preprocessing   │──────▶│  Final Sections  │
│  (Markdown)      │       │  Pipeline        │       │  (Processed)     │
└──────────────────┘       └──────────────────┘       └──────────────────┘
                                    │
                                    │ unity.py
                                    │ extract.js
                                    │ complement.py (DeepSeek V3)
                                    ▼
┌──────────────────────────────────────────────────────────────────┐
│                      Vector Database Layer                        │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │  Milvus Lite (Embedded Vector Database)                    │  │
│  │  • Collection: doc_sections                                │  │
│  │  • Index: IVF_FLAT (Milvus Lite compatible)               │  │
│  │  • Metric: COSINE similarity                               │  │
│  │  • Storage: milvus_lite.db (single file)                  │  │
│  └────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
                                    │
                                    │ create-vecdb.py
                                    ▼
┌──────────────────────────────────────────────────────────────────┐
│                      Embedding Model Layer                        │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │  SFR-Embedding-Code-400M_R                                 │  │
│  │  • Dimension: 1024                                         │  │
│  │  • Type: Sentence Transformer                              │  │
│  │  • Specialization: Code & Technical Documentation          │  │
│  │  • Pooling: CLS token (include_prompt=True)               │  │
│  └────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
                                    │
                                    │ search-vecdb.py
                                    │ rag_example.py
                                    ▼
┌──────────────────────────────────────────────────────────────────┐
│                      Application Layer                            │
│  ┌─────────────────────┐      ┌─────────────────────┐           │
│  │  Interactive Search │      │  Demo Search        │           │
│  │  • Real-time query  │      │  • Sample queries   │           │
│  │  • User interface   │      │  • Benchmarking     │           │
│  └─────────────────────┘      └─────────────────────┘           │
└──────────────────────────────────────────────────────────────────┘
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

*

```bash
# Copy environment template
cp env.example .env

# Edit .env file:
# MODEL_PATH=./models
```

### 4. Create Vector Database

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

## Core Components

### Document Processing Pipeline
- `scripts/unity.py` - Unify and merge document files
- `scripts/extract.js` - Extract structured sections from documents  
- `scripts/complement.py` - Enhance content using DeepSeek V3 LLM


## Project Structure

```
MoonBit-Docs-VectorDB/
├── scripts/
│   ├── create-vecdb.py         # Database creation
│   ├── search-vecdb.py         # Search functionality
│   ├── unity.py                # Document unification
│   ├── extract.js              # Section extraction
│   └── complement.py           # AI enhancement
├── models/                     # SFR model files
│   ├── 1_Pooling/
│   │   └── config.json         # Model pooling config
│   ├── config.json             # Model configuration
│   ├── model.safetensors       # Model weights
│   └── ...                     # Other model files
├── final_sections/             # Processed documents
├── rag_example.py              # RAG application
├── env.example                 # Environment template
├── requirements.txt            # Dependencies
├── method.svg                  # Architecture diagram
└── README.md                   # This file
```


## Data Flow

### Database Creation
1. Load SFR model with offline configuration
2. Create Milvus collection with auto-generated IDs
3. Build IVF_FLAT index for efficient search
4. Process documents and generate embeddings
5. Insert vectors with metadata into database

### Search Process
1. Generate query embedding using same model
2. Execute ANN search in Milvus with cosine similarity
3. Return ranked results with similarity scores
4. Format results with content and metadata

## Example Output

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
1. Recreate database: `python scripts/create-vecdb-lite.py`
2. Check index creation in logs
3. System will fallback to brute-force search if needed

### Low Similarity Scores

**Issue**: All results show similarity < 0.5

**Solution**:
1. Verify model consistency between creation and search
2. Check pooling configuration in `1_Pooling/config.json`
3. Run vector compatibility test in `search-vecdb.py`

## Requirements

- Python 3.8+
- 4GB+ RAM  
- 2GB+ storage

See `requirements.txt` for complete dependencies.

## License

MIT License - See LICENSE file for details.

## Acknowledgments

- **MoonBit Team**: Official documentation
- **Salesforce**: SFR-Embedding-Code model  
- **Milvus**: Vector database technology
