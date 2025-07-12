import os
import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Sentiment Analysis API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictRequest(BaseModel):
    text: str

class PredictResponse(BaseModel):
    label: str
    score: float

class SentimentModel:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.load_model()
    
    def load_model(self):
        model_path = "./model"
        if os.path.exists(model_path) and os.listdir(model_path):
            logger.info("Loading fine-tuned model from ./model")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        else:
            logger.info("Loading pre-trained model from HuggingFace")
            model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        
        self.model.eval()
    
    def predict(self, text: str) -> PredictResponse:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
        predicted_class = torch.argmax(predictions, dim=-1).item()
        confidence = predictions[0][predicted_class].item()
        
        # Map to binary sentiment
        if hasattr(self.model.config, 'id2label'):
            label_map = self.model.config.id2label
            raw_label = label_map[predicted_class].lower()
            if 'positive' in raw_label or raw_label == 'label_2':
                label = "positive"
            else:
                label = "negative"
        else:
            label = "positive" if predicted_class == 1 else "negative"
        
        return PredictResponse(label=label, score=confidence)

sentiment_model = SentimentModel()

@app.post("/predict", response_model=PredictResponse)
async def predict_sentiment(request: PredictRequest):
    return sentiment_model.predict(request.text)

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)