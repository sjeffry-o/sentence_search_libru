from glob import glob
from transformers import AutoTokenizer, AutoModel
import torch
import pickle
import ruclip

device = 'cuda'
model, processor = ruclip.load('ruclip-vit-base-patch16-224', device=device)

def computeTextVectors(text_list, save=True):
    text_vectors = []
    device = 'cuda'
    count = 0
    last_shape = 0

    with torch.no_grad():
        for text in text_list:
            text = text[0]
            # if count % 20000 == 0:
            #     print(count, "embs acquired with ruCLIP")
            input_ids = processor.encode_text(text).unsqueeze(0).cuda()
            embedding = model.encode_text(input_ids)
            text_vectors.append(embedding.to('cpu'))
            count += 1
    if save:
        with open('data_emb/text_embeds_ruclip.pkl', 'wb') as f:
            pickle.dump(text_vectors, f)
    return text_vectors

data_files = glob('data/*')
for book_pkl_filename in data_files:
    with open(book_pkl_filename, 'rb') as f:
        data = pickle.load(f)
    embs = []
    sent_texts = [case[0] for case in data]
    sent_links = [case[1] for case in data]
    embs = computeTextVectors(sent_texts, save=False)
    book_payload = zip(sent_texts, sent_links, embs)
    print('book path', book_pkl_filename)
    with open(book_pkl_filename.replace('data', 'data_emb'), 'wb') as f:
        pickle.dump(book_payload, f)
