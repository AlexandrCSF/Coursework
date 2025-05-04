import json
import flask
import requests
from flask import Flask, jsonify, request
from web import models, MODELS, es, dataset_amazon, dataset_wildberries

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

def encode_text(text, model_name):
    """Кодирует текст в эмбеддинг с использованием указанной модели"""
    model = models[model_name]
    embeddings = model.encode(text)
    return embeddings.squeeze().tolist()

@app.route('/search', methods=['POST'])
def search():
    data = request.json
    query = data.get('query', '')
    model_name = data.get('model', 'mpnet')  # По умолчанию используем mpnet
    dataset = data.get('dataset', 'amazon')  # По умолчанию используем amazon
    
    if model_name not in MODELS:
        return jsonify({'error': f'Model {model_name} not found'}), 400
        
    if dataset not in ['amazon', 'wildberries']:
        return jsonify({'error': f'Dataset {dataset} not found. Available datasets: amazon, wildberries'}), 400
        
    url = f"http://localhost:9200/products_{dataset}_{model_name}/_search"
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

@app.route('/all_products', methods=['GET'])
def get_all_products():
    page = int(request.args.get('page', 1))
    size = int(request.args.get('size', 20))
    
    # Получаем товары из Amazon
    amazon_url = f"http://localhost:9200/products_mpnet_amazon/_search"
    amazon_payload = {
        "query": {"match_all": {}},
        "from": (page - 1) * size,
        "size": size
    }
    
    # Получаем товары из Wildberries
    wildberries_url = f"http://localhost:9200/products_mpnet_wildberries/_search"
    wildberries_payload = {
        "query": {"match_all": {}},
        "from": (page - 1) * size,
        "size": size
    }
    
    headers = {'Content-Type': 'application/json'}
    
    try:
        amazon_response = requests.post(amazon_url, headers=headers, json=amazon_payload)
        wildberries_response = requests.post(wildberries_url, headers=headers, json=wildberries_payload)
        
        if amazon_response.status_code != 200 or wildberries_response.status_code != 200:
            return jsonify({'error': 'Failed to fetch results'}), 500
            
        amazon_data = amazon_response.json()
        wildberries_data = wildberries_response.json()
        
        # Объединяем результаты
        all_products = {
            'amazon': amazon_data['hits']['hits'],
            'wildberries': wildberries_data['hits']['hits'],
            'total': {
                'amazon': amazon_data['hits']['total']['value'],
                'wildberries': wildberries_data['hits']['total']['value']
            }
        }
        
        return jsonify(all_products)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.get("/")
def get():
    return flask.render_template('index.html')

if __name__ == '__main__':
    app.run(port=5000)