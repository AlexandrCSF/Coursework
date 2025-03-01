from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
from elasticsearch import Elasticsearch, helpers
from tqdm import tqdm
from huggingface_hub import login
import torch

# Логин в Hugging Face Hub (если требуется)
login()

# Подключение к Elasticsearch
es = Elasticsearch(["http://localhost:9200"])

# Проверка доступности GPU
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"GPU доступен: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("GPU недоступен, используется CPU.")

# Проверка соединения с Elasticsearch
if not es.ping():
    raise ValueError("Ошибка подключения к Elasticsearch")

# Инициализация токенайзера и модели
tokenizer = AutoTokenizer.from_pretrained('Salesforce/SFR-Embedding-Mistral')
tokenizer.add_eos_token = True

# Перемещаем модель на GPU
model = AutoModel.from_pretrained('Salesforce/SFR-Embedding-Mistral').to(device)


def encode_text(text):
    """ Кодирует текст в эмбеддинг """
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

    # Перемещаем входные данные на GPU
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state[:, 0, :]  # Берем первый токен [CLS]

    # Перемещаем результат обратно на CPU (если нужно)
    return embeddings.cpu().squeeze().tolist()


def encode_product(product):
    """ Кодирует текстовую информацию о товаре в эмбеддинг """
    text_data = f"{product.get('name', '')} {product.get('brand', '')} {product.get('description', '')} " \
                f"{' '.join(product.get('categories', []))} {product.get('params_str', '')}"
    return encode_text(text_data)


def create_index():
    """ Создает индекс с dense_vector """
    if not es.indices.exists(index="products"):
        es.indices.create(
            index="products",
            body={
                "mappings": {
                    "properties": {
                        "embedding": {"type": "dense_vector", "dims": 1024}  # Mistral выдаёт 1024-мерные эмбеддинги
                    }
                }
            }
        )


def load_and_index_dataset():
    """ Загружает датасет и индексирует товары в Elasticsearch с использованием Bulk API """
    # Загрузка датасета
    print('load_dataset')
    dataset = load_dataset("nyuuzyou/wb-products", split="train")
    print('end load_dataset')

    # Создание индекса в Elasticsearch
    create_index()

    # Подготовка данных для Bulk API
    actions = []
    for product in tqdm(dataset, desc="Подготовка данных для индексации"):
        try:
            # Преобразуем товар в нужный формат
            product_dict = {
                "id": product["id"],
                "name": product["name"],
                "brand": product["brand"],
                "description": product["description"],
                "categories": product["categories"],
                "params_str": product["params_str"],
            }
            # Кодируем текст в эмбеддинг
            product_dict["embedding"] = encode_product(product_dict)

            # Добавляем действие для Bulk API
            actions.append({
                "_index": "products",
                "_id": product_dict["id"],
                "_source": product_dict
            })
        except Exception as e:
            print(f"Ошибка при подготовке товара {product['id']}: {e}")

    # Индексация товаров с использованием Bulk API
    try:
        helpers.bulk(es, actions)
        print("Индексация завершена успешно!")
    except Exception as e:
        print(f"Ошибка при индексации: {e}")


# Запуск процесса загрузки и индексации
load_and_index_dataset()