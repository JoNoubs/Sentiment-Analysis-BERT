import streamlit as st
import requests

st.set_page_config(page_title="Sentiment Analyzer", page_icon="🧠", layout="centered")

# 🎨 Custom CSS for style
st.markdown("""
    <style>
        .title {
            text-align: center;
            font-size: 2.5em;
            color: #4CAF50;
            margin-bottom: 0.5em;
        }
        .stTextInput > div > div > input {
            font-size: 1.2em;
        }
        .result {
            text-align: center;
            font-size: 1.8em;
            margin-top: 1.5em;
            color: #333;
        }
    </style>
""", unsafe_allow_html=True)

# 🧠 Title
st.markdown('<div class="title">Sentiment Analysis with BERT 💬</div>', unsafe_allow_html=True)

# 📝 Input
text = st.text_input("✍️ Enter a sentence to analyze", "")

# 🚀 Predict
if st.button("🔍 Analyze Sentiment"):
    if text.strip():
        with st.spinner("Analyzing... 🔄"):
            try:
                response = requests.post("http://backend:5000/predict", json={"text": text})
                result = response.json()
                if "sentiment" in result:
                    emoji_map = {
                        "Negative": "😞",
                        "Neutral": "😐",
                        "Positive": "😊"
                    }
                    sentiment = result["sentiment"]
                    emoji = emoji_map.get(sentiment, "❓")
                    st.markdown(f'<div class="result">{emoji} Sentiment: <strong>{sentiment}</strong></div>', unsafe_allow_html=True)
                else:
                    st.error(f"Error: {result.get('error', 'Unknown error')}")
            except Exception as e:
                st.error(f"❌ Connection error: {str(e)}")
    else:
        st.warning("🚨 Please enter some text before submitting.")