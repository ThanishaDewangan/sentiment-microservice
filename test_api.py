import requests
import json

def test_api():
    url = "http://localhost:8000/predict"
    
    test_cases = [
        "I love this product!",
        "This is terrible.",
        "It's okay, nothing special.",
        "Amazing experience, highly recommend!"
    ]
    
    print("Testing Sentiment Analysis API...")
    print("-" * 40)
    
    for text in test_cases:
        try:
            response = requests.post(url, json={"text": text})
            result = response.json()
            print(f"Text: {text}")
            print(f"Prediction: {result['label']} (confidence: {result['score']:.3f})")
            print()
        except Exception as e:
            print(f"Error testing '{text}': {e}")

if __name__ == "__main__":
    test_api()