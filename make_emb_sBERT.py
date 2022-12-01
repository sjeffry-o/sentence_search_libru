from glob import glob
from transformers import AutoTokenizer, AutoModel
import torch
import pickle

#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

device = 'cuda'
#Load AutoModel from huggingface model repository
tokenizer = AutoTokenizer.from_pretrained("sberbank-ai/sbert_large_mt_nlu_ru")
model = AutoModel.from_pretrained("sberbank-ai/sbert_large_mt_nlu_ru").to(device)


data_files = glob('data/*')
for book_pkl_filename in data_files:
    with open(book_pkl_filename, 'rb') as f:
        data = pickle.load(f)
    embs = []
    sent_texts = [case[0] for case in data]
    sent_links = [case[1] for case in data]
    for sentence in sent_texts:
        #Tokenize
        encoded_input = tokenizer([sentence], padding=True, truncation=True, max_length=24, return_tensors='pt').to(device)
        #Compute token embeddings
        with torch.no_grad():
            model_output = model(**encoded_input)
        #Perform pooling. In this case, mean pooling
        sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask']).to('cpu')
        embs.append(sentence_embeddings)
    book_payload = zip(sent_texts, sent_links, embs)
    with open(book_pkl_filename.replace('data', 'data_emb'), 'wb') as f:
        pickle.dump(book_payload, f)
