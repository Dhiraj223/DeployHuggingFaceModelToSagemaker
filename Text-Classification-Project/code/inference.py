import json
import logging
import torch
from typing import List
from sagemaker_inference import encoder
from transformers import AutoTokenizer, AutoModelForSequenceClassification

SENTIMENT = "sentiment"
sentiment_labels = ['negative', 'neutral', 'positive']

class DhirajSentimentAnalysis :

    def __init__(self) -> None:
        model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        logging.info(f"Commander I am Here")

    def __call__(self, text :str) -> str :
        with torch.inference_mode():
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=64)
            outputs = self.model(**inputs)
            predicted_class = torch.argmax(outputs.logits).item()
            # Map predicted class to sentiment label
            predicted_sentiment = sentiment_labels[predicted_class]
        
        return predicted_sentiment
    
def model_fn(model_dir: str) -> DhirajSentimentAnalysis:
    try:
        return DhirajSentimentAnalysis()
    except Exception :
        logging.exception(f"Failed to load model from: {model_dir}")
        raise

def transform_fn(sentiment_analysis: DhirajSentimentAnalysis, input_data: bytes, content_type: str, accept: str) -> bytes:
    payload = json.loads(input_data)
    text = payload["text"]
    model_output = sentiment_analysis(text=text)
    output = {SENTIMENT : model_output}
    return encoder.encode(output,accept)