from datasets import load_dataset
from tqdm import tqdm
import os

dataset = load_dataset('Skylion007/openwebtext', streaming=True, trust_remote_code=True)
data = dataset['train']

os.makedirs('data', exist_ok=True)
with open('data/data.txt', 'a', encoding='utf-8') as f:
    for document in tqdm(data.take(250_000), total = 250_000):
        f.write(document['text'])

print('saved data')