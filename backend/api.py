from flask import Flask, request, jsonify
from .inference import predict

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict_sentiment():
    data = request.get_json()
    text = data.get('text', '')
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    try:
        result = predict(text)
        label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
        return jsonify({'sentiment': label_map[result]})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)