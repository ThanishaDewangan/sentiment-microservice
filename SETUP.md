# Setup Guide

## Prerequisites

- Docker and Docker Compose
- Python 3.9+ (for local development)
- Node.js 18+ (for local development)

## Quick Start (Docker)

1. **Clone/Download the project**
2. **Start services:**
   ```bash
   docker-compose up --build
   ```
3. **Access the application:**
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8000
   - API Docs: http://localhost:8000/docs

## Fine-tuning Example

1. **Prepare your data** (JSONL format):
   ```json
   {"text": "Great product!", "label": "positive"}
   {"text": "Poor quality", "label": "negative"}
   ```

2. **Run fine-tuning:**
   ```bash
   python finetune.py -data your_data.jsonl -epochs 3 -lr 3e-5
   ```

3. **Restart services** to load the new model:
   ```bash
   docker-compose restart backend
   ```

## Local Development

### Backend
```bash
cd backend
pip install -r requirements.txt
python app.py
```

### Frontend
```bash
cd frontend
npm install
npm start
```

### Fine-tuning
```bash
pip install -r requirements.txt
python finetune.py -data data/sample_data.jsonl -epochs 2 -lr 3e-5
```

## Testing

```bash
# Test API directly
python test_api.py

# Run demo workflow
python demo.py
```

## Troubleshooting

- **Port conflicts**: Change ports in docker-compose.yml
- **Memory issues**: Reduce batch size in finetune.py
- **CORS errors**: Check backend CORS settings in app.py
- **Model loading**: Ensure ./model directory has proper permissions