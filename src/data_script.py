from datasets import load_dataset
from tqdm import tqdm
import os

def download_data(PATH, TAKE = 100_000):
    dataset = load_dataset('Skylion007/openwebtext', streaming=True, trust_remote_code=True)
    data = dataset['train']

    with open(PATH, 'a', encoding='utf-8') as f:
        for document in tqdm(data.take(TAKE), total = TAKE):
            f.write('<|endoftext|>' + document['text'])

    print('saved data')