from web import models, MODELS, es, dataset_amazon, dataset_wildberries
from tqdm import tqdm

def create_index(dataset_name, model_name):
    """Создает индекс с dense_vector для указанной модели и датасета"""
    index_name = f"products_{dataset_name}_{model_name}"
    if not es.indices.exists(index=index_name):
        es.indices.create(
            index=index_name,
            body={
                "mappings": {
                    "properties": {
                        "embedding": {"type": "dense_vector", "dims": models[model_name].get_sentence_embedding_dimension()}
                    }
                }
            }, request_timeout=1000
        )

def encode_product(product_dict, model):
    """Кодирует текст товара в эмбеддинг"""
    text = f"{product_dict['name']} {product_dict['description']} {product_dict['brand']} {product_dict['categories']} {product_dict['params_str']}"
    embeddings = model.encode(text)
    return embeddings.squeeze().tolist()

def load_and_index_dataset():
    """Загружает датасет и индексирует товары в Elasticsearch с использованием Bulk API"""
    print("Indexing datasets...")
    
    # Создание индексов для каждой модели
    for model_name in MODELS.keys():
        create_index('amazon', model_name)
    create_index('wildberries', 'multilingual')
    
    bulk_data = []
    i = 1
    
    # Обработка Amazon датасета
    if dataset_amazon is not None:
        print("Processing Amazon dataset...")
        for index, product in enumerate(tqdm(dataset_amazon, desc="Подготовка данных Amazon")):
            try:
                product_dict = {
                    "id": product.get('uniq_id', index),
                    "name": product.get('title', ''),
                    "brand": product.get('store', ''),
                    "description": product.get('description', ''),
                    "categories": product.get('main_category', ''),
                    "params_str": product.get('details', ''),
                    "picture": product.get('image', '')
                }

                # Индексация для каждой модели
                for model_name, model in models.items():
                    payload = product_dict.copy()
                    payload["embedding"] = encode_product(product_dict, model)

                    bulk_data.append({
                        "index": {
                            "_index": f"products_amazon_{model_name}",
                            "_id": i
                        }
                    })
                    bulk_data.append(payload)
                    i += 1
            except Exception as e:
                print(f"Ошибка при подготовке товара {index}: {e}")
    else:
        print("Amazon dataset is None, skipping...")
    
    # Обработка Wildberries датасета
    if dataset_wildberries is not None:
        print("Processing Wildberries dataset...")
        for index, product in enumerate(tqdm(dataset_wildberries, desc="Подготовка данных Wildberries")):
            try:
                product_dict = {
                    "id": product.get('id', f"wb_{index}"),
                    "name": product.get('imt_name', ''),
                    "brand": product.get('brand_name', ''),
                    "description": product.get('description', ''),
                    "categories": product.get('subj_name', ''),
                    "params_str": product.get('бежевый', ''),
                    "picture": 'https://www.google.com/url?sa=i&url=https%3A%2F%2Fwww.pixsy.com%2Fimage-theft%2Fverify-image-source-copyright-owner&psig=AOvVaw3sptq6uKBUX8dL051JtPC8&ust=1741552195372000&source=images&cd=vfe&opi=89978449&ved=0CBQQjRxqFwoTCKC90veo-4sDFQAAAAAdAAAAABAO'
                }
                
                # Индексация только для multilingual модели
                payload = product_dict.copy()
                payload["embedding"] = encode_product(product_dict, models['multilingual'])
                
                bulk_data.append({
                    "index": {
                        "_index": "products_wildberries_multilingual",
                        "_id": i
                    }
                })
                bulk_data.append(payload)
                i += 1
            except Exception as e:
                print(f"Ошибка при подготовке товара {index}: {e}")
    else:
        print("Wildberries dataset is None, skipping...")
    
    # Индексация данных только если есть что индексировать
    if bulk_data:
        print("Indexing data...")
        print(es.bulk(body=bulk_data, request_timeout=1000)['errors'])
    else:
        print("No data to index")

# Запуск процесса загрузки и индексации
if __name__ == "__main__":
    load_and_index_dataset()