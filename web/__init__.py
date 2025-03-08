import torch
from sentence_transformers import SentenceTransformer

device = torch.device("cpu")
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
