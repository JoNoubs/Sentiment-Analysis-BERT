# Purpose: Load and preprocess the sentiment dataset from a CSV file
import pandas as pd
import os

def load_data():
    # Use a relative path to the dataset
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "dataset.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} does not exist")
    # Load the CSV and preprocess
    df = pd.read_csv(path)
    # Ensure required columns exist and handle missing values
    if 'content' not in df.columns or 'score' not in df.columns:
        raise ValueError("CSV must contain 'content' and 'score' columns")
    df = df[['content', 'score']].dropna()
    # Map scores to labels: 0 (negative <= 2), 1 (neutral = 3), 2 (positive > 3)
    df['label'] = df['score'].apply(lambda x: 0 if x <= 2 else (1 if x == 3 else 2))
    return df[['content', 'label']]

if __name__ == "__main__":
    # Test the loading function
    try:
        df = load_data()
        print("Loaded and preprocessed data:\n", df.head())
    except Exception as e:
        print(f"Error: {e}")