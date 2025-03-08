import json

import flask
import requests
import torch
from flask import Flask, jsonify, request

from web import model

app = Flask(__name__)


def generate_query_vector_search(vector, size, min_score):
    return {
        "knn": {
            "field": "embedding",
            "query_vector": vector,
            "k": size,
            "num_candidates": 100
        },
        "size": size,
        "min_score": min_score
    }

def generate_all_multi_match_queries(word):
    search_type = "most_fields"
    fields = [
        "name^2",
        "categories^3",
        "params_str",
        "n_grams"
    ]
    return {
        "multi_match": {
            "query": word,
            "fields": fields,
            "type": search_type,
            "operator": "or",
        }
    }

#def generate_query_vector_search(vector, size, min_score):
#    return {
#        "query": {
#            "script_score": {
#                "query": {
#                    "match_all": {}
#                },
#                "script": {
#                    "source": "cosineSimilarity(params.vector, 'vector') + 1",
#                    "lang": "painless",
#                    "params": {
#                        "vector": vector
#                    }
#                }
#            }
#        },
#        "size": size,
#        "min_score": min_score
#    }
#

def encode_text(text):
    """ Кодирует текст в эмбеддинг """
    embeddings = model.encode(text)

    # Перемещаем результат обратно на CPU (если нужно)
    return embeddings.squeeze().tolist()


@app.route('/search', methods=['POST'])
def search():
    data = request.json
    query = data.get('query', '')
    url = "http://localhost:9200/products2/_search"
    headers = {'Content-Type': 'application/json'}

    payload = generate_query_vector_search(
        min_score=0,
        size=10,
        vector=encode_text(query)
    )
    response = requests.post(url, headers=headers, json=payload)

    if response.status_code != 200:
        return jsonify({'error': 'Failed to fetch results'}), response.status_code

    return jsonify(response.json())

@app.get("/")
def get():
    return flask.render_template('index.html')

if __name__ == '__main__':
    app.run(port=5000)