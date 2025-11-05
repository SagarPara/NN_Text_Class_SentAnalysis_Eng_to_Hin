from flask import Flask, request, jsonify, render_template
from inference import predict_sentiment, translate_to_hindi  # assuming your model code is in BERT_model.py

app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    #return "Welcome to the Sentiment and Translation API."
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict_handler():
    data = request.get_json()
    texts = data.get("texts")

    if not texts or not isinstance(texts, list):
        return jsonify({"error": "Please provide a list of texts under 'texts' key."}), 400

    try:
        sentiments, confidences = predict_sentiment(texts)
        translations = translate_to_hindi(texts)

        response = []
        for text, hi, label, prob in zip(texts, translations, sentiments, confidences):
            response.append({
                "text": text,
                "translation": hi,
                "sentiment": "Positive" if label == 1 else "Negative",
                "confidence": float(prob[label])
            })

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
