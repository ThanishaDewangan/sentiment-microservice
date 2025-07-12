#!/usr/bin/env python3
"""
Demo script showing the complete workflow:
1. Fine-tune the model
2. Test the API
3. Show results
"""

import subprocess
import time
import requests
import json
import os

def run_finetuning():
    print("🔧 Starting fine-tuning process...")
    try:
        result = subprocess.run([
            "python", "finetune.py", 
            "-data", "data/sample_data.jsonl", 
            "-epochs", "2", 
            "-lr", "3e-5"
        ], capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("✅ Fine-tuning completed successfully!")
            return True
        else:
            print(f"❌ Fine-tuning failed: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("⏰ Fine-tuning timed out (5 minutes)")
        return False
    except Exception as e:
        print(f"❌ Error during fine-tuning: {e}")
        return False

def test_predictions():
    print("\n🧪 Testing predictions...")
    
    test_cases = [
        "I absolutely love this product! It's fantastic!",
        "This is the worst thing I've ever bought.",
        "The service was excellent and very professional.",
        "Completely disappointed with the quality."
    ]
    
    url = "http://localhost:8000/predict"
    
    for i, text in enumerate(test_cases, 1):
        try:
            response = requests.post(url, json={"text": text}, timeout=10)
            if response.status_code == 200:
                result = response.json()
                emoji = "😊" if result['label'] == 'positive' else "😞"
                print(f"{i}. {emoji} '{text[:50]}...' → {result['label']} ({result['score']:.3f})")
            else:
                print(f"{i}. ❌ API error: {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"{i}. ❌ Connection error: {e}")

def main():
    print("🚀 Sentiment Analysis Demo")
    print("=" * 50)
    
    # Check if model directory exists
    if os.path.exists("./model") and os.listdir("./model"):
        print("📁 Found existing fine-tuned model")
        use_existing = input("Use existing model? (y/n): ").lower().strip()
        if use_existing != 'y':
            run_finetuning()
    else:
        print("📁 No fine-tuned model found")
        if input("Run fine-tuning? (y/n): ").lower().strip() == 'y':
            run_finetuning()
    
    print("\n🌐 Make sure the API is running (docker-compose up)")
    input("Press Enter when ready to test...")
    
    test_predictions()
    
    print("\n✨ Demo completed!")
    print("💡 Try the web interface at http://localhost:3000")

if __name__ == "__main__":
    main()