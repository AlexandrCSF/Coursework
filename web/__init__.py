import torch
from transformers import AutoTokenizer, AutoModel

device = torch.device("cpu")
#tokenizer = AutoTokenizer.from_pretrained('Salesforce/SFR-Embedding-Mistral')
#tokenizer.add_eos_token = True
#model = AutoModel.from_pretrained('Salesforce/SFR-Embedding-Mistral').to(device)