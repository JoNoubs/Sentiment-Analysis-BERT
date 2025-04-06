import streamlit as st
import requests

st.title("Sentiment Analysis with BERT")
text = st.text_input("Enter Text")
if st.button("Predict"):
    if text:
        try:
            response = requests.post("http://backend:5000/predict", json={"text": text})
            result = response.json()
            if 'sentiment' in result:
                label_map = {0: "😞", 1: "😐", 2: "😊"}
                st.write(f"Sentiment: {result['sentiment']} {label_map[int(result['sentiment'][-1])]}")
            else:
                st.write(f"Error: {result.get('error', 'Unknown error')}")
        except Exception as e:
            st.write(f"Error: {str(e)}")
    else:
        st.write("Please enter some text!")