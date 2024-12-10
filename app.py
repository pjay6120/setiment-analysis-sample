from flask import Flask, request, jsonify
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import requests
from bs4 import BeautifulSoup
import re


def sentiment(txt):
    tokens = tokenizer.encode(txt, return_tensors='pt')
    result = model(tokens)
    result.logits
    return int(torch.argmax(result.logits))+1

app = Flask(__name__)

@app.route('/', methods=['POST'])
def post_example():
    data = request.get_json()
    if data:
        txt = data.get('message', 'No message provided')
        ans = sentiment(txt) 

        response = {
            "status": "success",
            "message": f"Sentiment score is '{ans}'/5."
        }
        return jsonify(response), 200  
    else:
        return jsonify({"status": "error", "message": "No data received"}), 400 


if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
    model = AutoModelForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
    app.run(debug=True)