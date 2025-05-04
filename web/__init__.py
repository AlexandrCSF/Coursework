import json

import pyzstd
import torch
from datasets import load_dataset, Dataset
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer

MODELS = {
    'mpnet': 'sentence-transformers/all-mpnet-base-v2',
    'minilm': 'sentence-transformers/all-MiniLM-L6-v2',
    'qa-mpnet': 'sentence-transformers/multi-qa-mpnet-base-dot-v1',
    'multilingual': 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'
}
es = Elasticsearch(["http://localhost:9200"])
if not es.ping():
    raise ValueError("Ошибка подключения к Elasticsearch")

models = {}
for model_name, model_path in MODELS.items():
    print(f"Загрузка модели {model_name}...")
    models[model_name] = SentenceTransformer(model_path)
    if torch.cuda.is_available():
        models[model_name] = models[model_name].to('cuda')


# Загрузка датасетов из Hugging Face
datasets = ['breadlicker45/products', 'nyuuzyou/wb-products']
print('load wildberries dataset')
filepath='basket-01.json.zst'
records = []
with pyzstd.ZstdFile(filepath, 'rb') as zf:
    for i, line in enumerate(zf):
        if i >= 10_000:
            break
        records.append(json.loads(line.decode('utf-8')))

dataset_wildberries = Dataset.from_list(records).select(range(1000))
print('load amazon dataset')
dataset_amazon = None#load_dataset(datasets[0], split='train').select(range(1000))
print('Datasets loaded')