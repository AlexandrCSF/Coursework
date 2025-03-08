import torch
from datasets import load_dataset
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Подключение к Elasticsearch
es = Elasticsearch(["http://localhost:9200"])

# Проверка доступности GPU
device = torch.device("cpu")
print("GPU недоступен, используется CPU.")

# Проверка соединения с Elasticsearch
if not es.ping():
    raise ValueError("Ошибка подключения к Elasticsearch")
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

def encode_text(text):
    """ Кодирует текст в эмбеддинг """
    embeddings = model.encode(text)

    # Перемещаем результат обратно на CPU (если нужно)
    return embeddings.squeeze().tolist()


def encode_product(product):
    """ Кодирует текстовую информацию о товаре в эмбеддинг """
    text_data = f"{product.get('name', '')} {product.get('brand', '')} {product.get('description', '')} " \
                f"{' '.join(product.get('categories', []))} {product.get('params_str', '')}"
    return encode_text(text_data)


def create_index():
    """ Создает индекс с dense_vector """
    if not es.indices.exists(index="products2"):
        es.indices.create(
            index="products2",
            body={
                "mappings": {
                    "properties": {
                        "embedding": {"type": "dense_vector", "dims": 768}
                    }
                }
            }
        )


def load_and_index_dataset():
    """ Загружает датасет и индексирует товары в Elasticsearch с использованием Bulk API """
    # Загрузка датасета
    print('load_dataset')
    dataset = load_dataset("breadlicker45/products",split='train')
    dataset = dataset.select(range(5))
    print('end load_dataset')

    # Создание индекса в Elasticsearch
    create_index()

    # Подготовка данных для Bulk API
    bulk_data = []
    for index,product in enumerate(tqdm(dataset, desc="Подготовка данных для индексации")):
        try:
            # Преобразуем товар в нужный формат
            product_dict = {
                "id": product["uniq_id"],
                "name": product["product_name"],
                "brand": product["manufacturer"],
                "description": product["description"],
                "categories": product["amazon_category_and_sub_category"],
                "params_str": product["product_information"],
            }
            # Кодируем текст в эмбеддинг
            product_dict["embedding"] = encode_product(product_dict)

            # Добавляем действие для Bulk API
            bulk_data.append({
                "index": {
                    "_index": "products2",
                    "_id": index
                }
            })
            bulk_data.append(product_dict)
        except Exception as e:
            print(f"Ошибка при подготовке товара {index}: {e}")
    es.bulk(body=bulk_data, index="products2")

# Запуск процесса загрузки и индексации
load_and_index_dataset()