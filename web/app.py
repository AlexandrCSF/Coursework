import json
import flask
import requests
from flask import Flask, jsonify, request
from sentence_transformers import SentenceTransformer
import torch

app = Flask(__name__)

# Список моделей для эмбеддингов
MODELS = {
    'mpnet': 'sentence-transformers/all-mpnet-base-v2',
    'minilm': 'sentence-transformers/all-MiniLM-L6-v2',
    'qa-mpnet': 'sentence-transformers/multi-qa-mpnet-base-dot-v1',
    'multilingual': 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'
}

# Загрузка моделей
models = {}
for model_name, model_path in MODELS.items():
    print(f"Загрузка модели {model_name}...")
    models[model_name] = SentenceTransformer(model_path)
    if torch.cuda.is_available():
        models[model_name] = models[model_name].to('cuda')

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

def encode_text(text, model_name):
    """ Кодирует текст в эмбеддинг с использованием указанной модели """
    model = models[model_name]
    embeddings = model.encode(text)
    return embeddings.squeeze().tolist()


@app.route('/search', methods=['POST'])
def search():
    data = request.json
    query = data.get('query', '')
    model_name = data.get('model', 'mpnet')  # По умолчанию используем mpnet
    
    if model_name not in MODELS:
        return jsonify({'error': f'Model {model_name} not found'}), 400
        
    url = f"http://localhost:9200/products_{model_name}/_search"
    headers = {'Content-Type': 'application/json'}

    # Создаем комбинированный запрос с взвешенной суммой оценок
    payload = {
        "query": {
            "script_score": {
                "query": generate_all_multi_match_queries(query),
                "script": {
                    "source": """
                    // Нормализация векторного поиска (косинусное сходство в диапазоне [-1, 1])
                    double vectorScore = (cosineSimilarity(params.query_vector, 'embedding') + 1.0) / 2.0;
                    
                    // Нормализация полнотекстового поиска
                    // Используем min-max нормализацию для _score
                    double textScore = _score;
                    if (textScore > params.max_score) {
                        textScore = params.max_score;
                    }
                    textScore = textScore / params.max_score;
                    
                    // Взвешенная сумма нормализованных оценок
                    return params.vector_weight * vectorScore + params.text_weight * textScore;
                    """,
                    "params": {
                        "query_vector": encode_text(query, model_name),
                        "vector_weight": 0.7,  # Вес для векторного поиска
                        "text_weight": 0.3,    # Вес для полнотекстового поиска
                        "max_score": 10.0      # Максимальная ожидаемая оценка для полнотекстового поиска
                    }
                }
            }
        },
        "size": 10
    }
    
    response = requests.post(url, headers=headers, json=payload)

    if response.status_code != 200:
        return jsonify({'error': 'Failed to fetch results'}), response.status_code

    return jsonify(response.json())

@app.get("/")
def get():
    return flask.render_template('index.html')

if __name__ == '__main__':
    app.run(port=5000)