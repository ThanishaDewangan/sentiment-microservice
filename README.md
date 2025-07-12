# Sentiment Analysis Microservice

A complete end-to-end microservice for binary sentiment analysis with fine-tuning capabilities using Hugging Face Transformers, FastAPI, and React.

## Setup & Run Instructions

### Quick Start (Docker Compose - Recommended)

```bash
git clone <repository-url>
cd sentiment-microservice
docker-compose up --build
```

**Access Points:**
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Documentation: http://localhost:8000/docs

### Manual Setup

**Backend:**
```bash
cd backend
pip install -r requirements.txt
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

**Frontend:**
```bash
cd frontend
npm install
npm start
```

**Fine-tuning:**
```bash
python finetune.py -data data/sample_data.jsonl -epochs 3 -lr 3e-5
```

## Design Decisions

### Architecture
- **Backend**: FastAPI chosen for high-performance async API with automatic OpenAPI documentation
- **Frontend**: React with minimal dependencies for fast loading and maintainability
- **Model**: `cardiffnlp/twitter-roberta-base-sentiment-latest` - pre-trained RoBERTa model optimized for sentiment analysis
- **Containerization**: Docker Compose for easy deployment and service orchestration

### Technical Choices
- **Binary Classification**: Simplified positive/negative sentiment for clear business value
- **Model Persistence**: Fine-tuned weights saved to `./model/` with automatic reload on API restart
- **Training Pipeline**: Implements ML best practices:
  - Cross-entropy loss for classification
  - Gradient clipping (max_norm=1.0) for training stability
  - Linear learning rate scheduling with warmup
  - 80/20 train/validation split
  - Deterministic seeding for reproducible results

### Data Format
- **JSONL**: One JSON object per line for efficient streaming and processing
- **Schema**: `{"text": "sample text", "label": "positive|negative"}`

## Performance Benchmarks

### Fine-tuning Times (10 samples, 3 epochs)
- **CPU Training**: ~2-3 minutes (Intel i5/i7 equivalent)
- **GPU Training**: ~30 seconds (CUDA-enabled GPU)
- **Memory Usage**: ~2GB RAM for training, ~1GB for inference

### Inference Performance
- **CPU Inference**: ~100-200ms per request
- **Batch Processing**: Supports single requests (can be extended for batching)
- **Model Size**: ~500MB (RoBERTa-base)

## API Documentation

### Endpoints

#### POST /predict
Analyze sentiment of input text.

**Request:**
```json
{
  "text": "I love this product! It works perfectly."
}
```

**Response:**
```json
{
  "label": "positive",
  "score": 0.9847
}
```

**Response Fields:**
- `label`: "positive" or "negative"
- `score`: Confidence score (0.0 to 1.0)

#### GET /health
Health check endpoint for monitoring.

**Response:**
```json
{
  "status": "healthy"
}
```

### OpenAPI Documentation
Interactive API documentation available at: http://localhost:8000/docs

## Project Structure

```
sentiment-microservice/
├── backend/
│   ├── app.py              # FastAPI application
│   ├── requirements.txt    # Python dependencies
│   └── Dockerfile         # Backend container config
├── frontend/
│   ├── src/
│   │   ├── App.js         # React main component
│   │   ├── App.css        # Styling
│   │   └── index.js       # React entry point
│   ├── public/
│   │   └── index.html     # HTML template
│   ├── package.json       # Node.js dependencies
│   └── Dockerfile         # Frontend container config
├── data/
│   └── sample_data.jsonl  # Sample training data
├── model/                 # Fine-tuned model storage (auto-created)
├── finetune.py           # Training CLI script
├── docker-compose.yml    # Service orchestration
├── requirements.txt      # Root Python dependencies
└── README.md            # This documentation
```

## Tech Stack

**Backend:**
- FastAPI (Python web framework)
- Hugging Face Transformers (ML models)
- PyTorch (deep learning framework)
- Uvicorn (ASGI server)

**Frontend:**
- React 18 (UI framework)
- Create React App (build tooling)
- CSS3 (styling)

**DevOps:**
- Docker & Docker Compose (containerization)
- Multi-stage builds for optimized images

**ML Pipeline:**
- scikit-learn (metrics and utilities)
- AdamW optimizer with linear scheduling
- Gradient clipping and validation monitoring

---

**Demo Video**: https://drive.google.com/file/d/1JTO-mBcIOwzqIwaD-iOGns7QtHP6y293/view?usp=sharing
