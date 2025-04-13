from transformers import AutoTokenizer
import torch
from backend.model import load_trained_model

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = load_trained_model()

def predict(text):
    try:
        # Tokenize the input text
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']

        # Perform inference
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            prediction = torch.argmax(logits, dim=1).item()

        return prediction
    except Exception as e:
        print(f"Error during inference: {e}")
        raise