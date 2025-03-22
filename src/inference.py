# src/inference.py
# Purpose: Predict sentiment using the trained model
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def predict(text, model_path="./models"):
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    # Tokenize input
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)
    # Return the predicted label (0: negative, 1: neutral, 2: positive)
    return outputs.logits.argmax().item()

if __name__ == "__main__":
    # Test with a sample text
    sample_text = "This movie is amazing!"
    result = predict(sample_text)
    label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
    print(f"Sentiment for '{sample_text}': {label_map[result]}")
