# Purpose: Handle text cleaning, tokenization, and data splitting
import pandas as pd
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from .data_extraction import load_data

def clean_text(text):
    # Convert to lowercase and remove non-alphanumeric characters except spaces
    return "".join(c.lower() for c in text if c.isalnum() or c.isspace())

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize_data(df):
    # Tokenize text using BERT tokenizer
    tokens = tokenizer(df['content'].tolist(), padding=True, truncation=True, max_length=128)
    return tokens, df['label'].tolist()

def split_data(df):
    # Split data into training and validation sets
    return train_test_split(df, test_size=0.2, random_state=42)

if __name__ == "__main__":
    # Test the pipeline
    df = load_data()
    train_df, val_df = split_data(df)
    train_encodings, train_labels = tokenize_data(train_df)
    print("Train encodings sample:", train_encodings['input_ids'][0])
    print("Train size:", len(train_df), "Validation size:", len(val_df))