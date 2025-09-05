# AutoPatternChecker

Automated system that learns per-composite-key formats from a CSV database, validates new values, suggests corrections, and updates profiles automatically. Designed to run on Colab for development and on a server/VM for production.

## Features

- **Pattern Learning**: Automatically learns patterns from CSV data using signature analysis
- **Normalization**: Applies learned normalization rules to clean and standardize values
- **Clustering**: Detects sub-formats within composite keys using HDBSCAN and KMeans
- **Semantic Similarity**: Uses sentence transformers for duplicate detection
- **FastAPI Service**: RESTful API for runtime validation
- **Active Learning**: Human-in-the-loop feedback for continuous improvement
- **FAISS Indexing**: Efficient similarity search using vector indices

## Quick Start

### Development (Google Colab)

1. Open `colab_notebook_full_pipeline.ipynb` in Google Colab
2. Upload your CSV file
3. Run all cells to process data and start the API
4. Test validation using the provided examples

### Production (Docker)

```bash
# Clone the repository
git clone <repository-url>
cd autopatternchecker

# Build and run with Docker Compose
docker-compose up -d

# The API will be available at http://localhost:8000
```

### Local Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .

# Run the API
uvicorn autopatternchecker.api:app --host 0.0.0.0 --port 8000
```

## Data Format

Your CSV file should have the following structure:

```csv
key_part1,key_part2,key_part3,value,optional_metadata
Tools,Accessories,Accessory Type (184),SMB Mini Jack Right Angle,
Tools,Accessories,Accessory Type (184),SC Plug,
Electronics,Components,Resistor,1kΩ 1/4W,
Electronics,Components,Resistor,2.2kΩ 1/2W,
```

The system will create composite keys by concatenating the first three columns with `||` as separator.

## API Usage

### Validate a Value

```bash
curl -X POST "http://localhost:8000/validate" \
  -H "Content-Type: application/json" \
  -d '{
    "key_parts": ["Tools", "Accessories", "Accessory Type (184)"],
    "value": "SMB Mini Jack, Right Angle",
    "metadata": {"source": "user_input"}
  }'
```

Response:
```json
{
  "verdict": "accepted",
  "issues": [],
  "suggested_fix": null,
  "nearest_examples": ["SMB Mini Jack Right Angle"],
  "confidence": 0.92,
  "rules_applied": ["trim_whitespace"],
  "processing_time_ms": 15.3
}
```

### Get Key Profile

```bash
curl "http://localhost:8000/profile/Tools||Accessories||Accessory%20Type%20(184)"
```

### Get Review Items

```bash
curl "http://localhost:8000/review/next?limit=10"
```

### Submit Review Decision

```bash
curl -X POST "http://localhost:8000/review/submit" \
  -H "Content-Type: application/json" \
  -d '{
    "item_id": "review_123",
    "decision": "accept",
    "corrected_value": null,
    "notes": "Looks good"
  }'
```

## Configuration

The system can be configured using YAML files in the `configs/` directory:

- `pipeline_config.yaml`: Main configuration for data processing, clustering, and API
- `model_config.json`: Model-specific configurations for embeddings and clustering

Key configuration options:

```yaml
# Data processing
data:
  key_columns: ['key_part1', 'key_part2', 'key_part3']
  value_column: 'value'
  chunk_size: 10000

# Clustering
clustering:
  hdbscan_min_cluster_size: 5
  use_embeddings: true

# Embeddings
embeddings:
  model: 'all-MiniLM-L6-v2'  # or 'all-mpnet-base-v2'
  batch_size: 64

# Active Learning
active_learning:
  auto_accept_confidence_threshold: 0.95
  manual_review_threshold_lower: 0.6
```

## Architecture

### Core Components

1. **DataIngester**: Handles CSV reading and composite key creation
2. **PatternProfiler**: Analyzes patterns and generates signatures
3. **NormalizationEngine**: Applies learned normalization rules
4. **ClusterAnalyzer**: Detects sub-formats using clustering algorithms
5. **EmbeddingGenerator**: Generates semantic embeddings
6. **FAISSIndexer**: Builds vector indices for similarity search
7. **ActiveLearningManager**: Manages human feedback and continuous learning

### Pipeline Flow

1. **Ingestion**: Load CSV data and create composite keys
2. **Profiling**: Analyze patterns and generate signatures
3. **Normalization**: Learn and apply cleaning rules
4. **Clustering**: Detect sub-formats within keys
5. **Embeddings**: Generate semantic representations
6. **Indexing**: Build FAISS indices for search
7. **Validation**: Runtime API for value validation
8. **Learning**: Human feedback integration

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_utils.py

# Run with coverage
pytest --cov=autopatternchecker
```

### Code Structure

```
autopatternchecker/
├── __init__.py
├── ingest.py          # Data ingestion
├── profiling.py       # Pattern analysis
├── normalize.py       # Normalization rules
├── clustering.py      # Clustering algorithms
├── embeddings.py      # Embedding generation
├── indexing.py        # FAISS indexing
├── api.py            # FastAPI service
├── active_learning.py # Human feedback
└── utils.py          # Utility functions

configs/
├── pipeline_config.yaml
└── model_config.json

tests/
├── test_utils.py
├── test_ingest.py
└── test_api.py

colab_notebook_full_pipeline.ipynb
Dockerfile
docker-compose.yml
```

## Deployment

### Production Considerations

1. **Resource Requirements**:
   - CPU: 4+ vCPUs recommended
   - RAM: 16GB+ for medium datasets
   - Storage: SSD recommended for FAISS indices

2. **Scaling**:
   - Use multiple API instances behind a load balancer
   - Consider sharding FAISS indices for very large datasets
   - Use Redis for shared state in multi-instance deployments

3. **Monitoring**:
   - Monitor API response times and error rates
   - Track review queue size and processing time
   - Monitor embedding generation performance

### Environment Variables

```bash
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
LOG_LEVEL=INFO

# Data Paths
DATA_DIR=/app/data
OUTPUT_DIR=/app/output
INDICES_DIR=/app/indices

# Model Configuration
EMBEDDING_MODEL=all-MiniLM-L6-v2
USE_GPU=true
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For questions and support, please open an issue on GitHub or contact the development team.