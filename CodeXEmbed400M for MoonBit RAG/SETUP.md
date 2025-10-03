# Setup Guide

## Quick Start

### 1. Environment Setup

```bash
# Clone the repository
git clone <your-repo-url>
cd MoonBit-Docs-VectorDB

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
npm install  # For document processing
```

### 2. Model Setup

Download the SFR-Embedding-Code model:

```bash
# Option 1: Use download script
python download_model.py

# Option 2: Manual download
# Download from: https://huggingface.co/Salesforce/SFR-Embedding-Code-400M_R
# Extract to: ./models/
```

### 3. Environment Configuration

```bash
# Copy environment template
cp env.example .env

# Edit .env file with your settings:
# MODEL_PATH=./models
# DEEPSEEK_API_KEY=your-api-key-here (optional)
```

### 4. Document Processing (Optional)

If you have raw MoonBit documentation:

```bash
# Step 1: Unify documents
python scripts/unity.py

# Step 2: Extract sections
node scripts/extract.js processed

# Step 3: Enhance with AI (requires DEEPSEEK_API_KEY)
python scripts/complement.py
```

### 5. Create Vector Database

```bash
# Create vector database from processed documents
python scripts/create-vecdb.py
```

### 6. Test the System

```bash
# Test search functionality
python scripts/search-vecdb.py

# Run RAG application
python rag_example.py
```

## Configuration Options

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `./models` | Path to SFR-Embedding-Code model |
| `DEEPSEEK_API_KEY` | - | API key for content enhancement (optional) |
| `DB_FILE` | `./milvus_lite.db` | Vector database file path |
| `COLLECTION_NAME` | `doc_sections` | Milvus collection name |

### Model Requirements

- **Model**: Salesforce SFR-Embedding-Code-400M_R
- **Size**: ~828 MB
- **Dimensions**: 1024
- **Context Length**: 8192 tokens

### System Requirements

- **Python**: 3.8+
- **Memory**: 4GB+ RAM
- **Storage**: 2GB+ free space
- **OS**: Windows, macOS, Linux

## Troubleshooting

### Common Issues

1. **Model not found**
   ```
   Error: Model path not found: ./models
   ```
   **Solution**: Download the SFR model and place in `./models/` directory

2. **No documents found**
   ```
   ERROR: No files found in final_sections/
   ```
   **Solution**: Run document processing steps or provide processed documents

3. **API key missing**
   ```
   Error: DEEPSEEK_API_KEY not set
   ```
   **Solution**: Set API key in `.env` file or skip complement.py step

### Performance Tips

1. **GPU Acceleration**: Install `torch` with CUDA support for faster embedding
2. **Memory**: Close other applications if running out of memory
3. **Storage**: Use SSD for better database performance

## Project Structure

```
MoonBit-Docs-VectorDB/
├── scripts/
│   ├── create-vecdb.py      # Create vector database
│   ├── search-vecdb.py      # Search interface
│   ├── unity.py             # Document unification
│   ├── extract.js           # Section extraction
│   └── complement.py        # AI enhancement
├── rag_example.py           # RAG application
├── env.example              # Environment template
├── requirements.txt         # Python dependencies
├── package.json            # Node.js dependencies
└── README.md               # Project overview
```

## Next Steps

After setup, see:
- [README.md](README.md) - Project overview
- [ARCHITECTURE.md](ARCHITECTURE.md) - System architecture
- Run `python scripts/search-vecdb.py` and select mode 3 for quality evaluation

## Support

- **Issues**: Open GitHub issue
- **Documentation**: See project README and architecture docs
- **Model**: [SFR-Embedding-Code on HuggingFace](https://huggingface.co/Salesforce/SFR-Embedding-Code-400M_R)
