# app.py
# Purpose: Create an enhanced web interface for sentiment analysis using Streamlit

import streamlit as st
from src.inference import predict

# App configuration
st.set_page_config(page_title="BERT Sentiment Analyzer", layout="centered")

# Title and subtitle
st.title("🧠 Sentiment Analysis with BERT")
st.subheader("Analyze the sentiment of a sentence using a fine-tuned BERT model")

# Text input
text = st.text_area("✍️ Enter your text below:", height=150, placeholder="Type a review, tweet, or comment...")

# Predict button
if st.button("🔍 Analyze Sentiment"):
    if text.strip():
        try:
            result = predict(text)
            label_map = {0: "😠 Negative", 1: "😐 Neutral", 2: "😊 Positive"}
            st.success(f"**Sentiment:** {label_map.get(result, 'Unknown')}")

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

    else:
        st.warning("Please enter some text to analyze.")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ by **Johanna** and **Armel** | Powered by 🤗 HuggingFace BERT")
