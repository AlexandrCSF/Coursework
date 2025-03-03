import flask
import requests
import torch
from flask import Flask, jsonify, request

#from web import tokenizer, model, device

app = Flask(__name__)


def generate_query_search(min_match, query_words, min_score, size, vector):
    return {
        "query": {
            "script_score": {
                "query": {
                    "bool": {
                        "should": [
                            {
                                "bool": {
                                    "should": generate_all_multi_match_queries(word)
                                }
                            } for word in query_words
                        ],
                        "minimum_should_match": min_match
                    }
                },
                "script": {
                    "source": "cosineSimilarity(params.vector, 'vector') + 1",
                    "lang": "painless",
                    "params": {
                        "vector": vector
                    }
                }
            }
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

def generate_query_vector_search(vector, size, min_score):
    return {
        "query": {
            "script_score": {
                "query": {
                    "match_all": {}
                },
                "script": {
                    "source": "cosineSimilarity(params.vector, 'vector') + 1",
                    "lang": "painless",
                    "params": {
                        "vector": vector
                    }
                }
            }
        },
        "size": size,
        "min_score": min_score
    }


def encode_text(text):
    """ Кодирует текст в эмбеддинг """
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

    # Перемещаем входные данные на GPU
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state[:, 0, :]  # Берем первый токен [CLS]

    # Перемещаем результат обратно на CPU (если нужно)
    return embeddings.cpu().squeeze().tolist()


@app.route('/search', methods=['POST'])
def search():
    data = request.json
    query = data.get('query', '')

    # Формируем поисковый запрос
    payload = generate_query_search(
        min_match=len(query),
        query_words=query.split(),
        min_score=0,
        size=10,
        vector=[0] * 4096
    )

    url = "http://localhost:9200/products/_search"
    headers = {'Content-Type': 'application/json'}

    # Отправляем запрос в Elasticsearch
    response = requests.post(url, headers=headers, json=payload)

    # Проверяем успешность запроса
    if response.status_code != 200:
        return jsonify({'error': 'Failed to fetch results'}), response.status_code

    return jsonify(response.json())
@app.get("/")
def get():
    return flask.render_template('index.html')

if __name__ == '__main__':
    app.run(port=5000)