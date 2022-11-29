import ngtpy
import pickle
from glob import glob
from torch import cat

index_name = b'libru_vanilla'
emb_data_files = sorted(glob('data_emb/*'))
# create an index framwork in filesystem.
ngtpy.create(path=index_name, dimension=1024, distance_type="L2")

for emb_file in emb_data_files:
    objects = []
    index = ngtpy.Index(index_name)
    with open(emb_file, 'rb') as f:
        data = pickle.load(f)
    for (sent_text, sent_link, sent_emb) in data:
        objects.append(sent_emb)
    objects = cat(objects)
    # insert the objects.
    index.batch_insert(objects)
    index.save()
    index.close()
