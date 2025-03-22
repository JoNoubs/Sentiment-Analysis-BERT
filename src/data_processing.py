# src/data_processing.py
# Purpose: Clean text data for sentiment analysis
def clean_text(text):
    # Convert to lowercase and remove non-alphanumeric characters except spaces
    return "".join(c.lower() for c in text if c.isalnum() or c.isspace())
