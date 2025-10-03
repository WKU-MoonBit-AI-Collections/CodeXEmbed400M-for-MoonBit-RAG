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
                                    │ rag.py
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


