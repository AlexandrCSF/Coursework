from datasets import load_dataset
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import torch

MODELS = {
    'mpnet': 'sentence-transformers/all-mpnet-base-v2',
    'minilm': 'sentence-transformers/all-MiniLM-L6-v2',
    'qa-mpnet': 'sentence-transformers/multi-qa-mpnet-base-dot-v1',
    'multilingual': 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'
}
# Подключение к Elasticsearch
es = Elasticsearch(["http://localhost:9200"])

# Проверка соединения с Elasticsearch
if not es.ping():
    raise ValueError("Ошибка подключения к Elasticsearch")

# Загрузка моделей
models = {}
for model_name, model_path in MODELS.items():
    print(f"Загрузка модели {model_name}...")
    models[model_name] = SentenceTransformer(model_path)
    if torch.cuda.is_available():
        models[model_name] = models[model_name].to('cuda')

def encode_text(text, model):
    """ Кодирует текст в эмбеддинг с использованием указанной модели """
    embeddings = model.encode(text)
    return embeddings.squeeze().tolist()

def encode_product(product, model):
    """ Кодирует текстовую информацию о товаре в эмбеддинг """
    text_data = f"{product.get('name', '')} {product.get('brand', '')} {product.get('description', '')} " \
                f"{product.get('categories', '')} {product.get('params_str', '')}"
    return encode_text(text_data, model)

def create_index(dataset_name, model_name):
    """ Создает индекс с dense_vector для указанной модели и датасета """
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

def load_and_index_dataset():
    """ Загружает датасет и индексирует товары в Elasticsearch с использованием Bulk API """
    datasets = ['breadlicker45/products','nyuuzyou/wb-products']
    print('Loading first dataset...')
    dataset1 = load_dataset(datasets[0], split='train')
    print('Loading second dataset...')
    dataset2 = load_dataset(datasets[1], split='train').select(range(1000))
    print('Datasets loaded')

    for model_name in MODELS.keys():
        create_index('amazon', model_name)  # для первого датасета
    create_index('wildberries', 'multilingual')  # для второго датасета
    
    bulk_data = []
    i = 1
    
    # Обработка первого датасета
    print("Processing first dataset...")
    for index, product in enumerate(tqdm(dataset1, desc="Подготовка данных первого датасета")):
        try:
            product_dict = {
                "id": product.get('uniq_id', index),
                "name": product.get('product_name', ''),
                "brand": product.get('manufacturer', ''),
                "description": product.get('description', ''),
                "categories": product.get('amazon_category_and_sub_category', ''),
                "params_str": product.get('product_information', ''),
                "picture": 'https://www.google.com/url?sa=i&url=https%3A%2F%2Fwww.pixsy.com%2Fimage-theft%2Fverify-image-source-copyright-owner&psig=AOvVaw3sptq6uKBUX8dL051JtPC8&ust=1741552195372000&source=images&cd=vfe&opi=89978449&ved=0CBQQjRxqFwoTCKC90veo-4sDFQAAAAAdAAAAABAO'
            }

            # Индексируем товар для каждой модели
            for model_name, model in models.items():
                payload = product_dict.copy()
                payload["embedding"] = encode_product(product_dict, model)

                # Добавляем действие для Bulk API
                bulk_data.append(
                    {
                        "index": {
                            "_index": f"products_amazon_{model_name}",
                            "_id": i
                        }
                    },
                )
                bulk_data.append(payload)
                i += 1
        except Exception as e:
            print(f"Ошибка при подготовке товара {index}: {e}")
    
    # Обработка второго датасета только через multilingual модель
    print("Processing second dataset with multilingual model...")
    for index, product in enumerate(tqdm(dataset2, desc="Подготовка данных второго датасета")):
        try:
            product_dict = {
                "id": product.get('id', f"wb_{index}"),
                "name": product.get('name', ''),
                "brand": product.get('brand', ''),
                "description": product.get('description', ''),
                "categories": product.get('category', ''),
                "params_str": product.get('characteristics', ''),
                "picture": product.get('image', '')
            }

            # Индексируем товар только для multilingual модели
            payload = product_dict.copy()
            payload["embedding"] = encode_product(product_dict, models['multilingual'])

            # Добавляем действие для Bulk API
            bulk_data.append(
                {
                    "index": {
                        "_index": "products_wildberries_multilingual",
                        "_id": i
                    }
                },
            )
            bulk_data.append(payload)
            i += 1
        except Exception as e:
            print(f"Ошибка при подготовке товара {index}: {e}")
    
    print("Indexing data...")
    print(es.bulk(body=bulk_data, request_timeout=1000)['errors'])

# Запуск процесса загрузки и индексации
if __name__ == "__main__":
    load_and_index_dataset()