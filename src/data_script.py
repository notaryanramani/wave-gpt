from datasets import load_dataset
from tqdm import tqdm
import os
import tiktoken
import multiprocessing as mp
import numpy as np

def download_data(FOLDER_PATH='data/', TAKE = 500000, max_file_size=500, validation_split=0.1):
    dataset = load_dataset('Skylion007/openwebtext', streaming=True, trust_remote_code=True)
    data = dataset['train'] # type: ignore
    max_size = max_file_size * 1024 * 1024

    train_idx = 1
    val_idx = 1
    current_train_size = 0
    current_val_size = 0

    current_train_filename = f"{FOLDER_PATH}train{train_idx}.txt"
    current_val_filename = f"{FOLDER_PATH}val{val_idx}.txt"


    for i, example in enumerate(tqdm(data)):
        text = '<|endoftext|>' + example['text'] + "\n" # type: ignore
        encoded_text = text.encode('utf-8')

        if i % int(1/validation_split) == 0:
            # Write to validation file
            if current_val_size + len(encoded_text) > max_size:
                val_idx += 1
                current_val_filename = f"{FOLDER_PATH}val{val_idx}.txt"
                current_val_size = 0

            with open(current_val_filename, 'a') as val_file:
                val_file.write(text)
                current_val_size += len(encoded_text)
        else:
            # Write to train file
            if current_train_size + len(encoded_text) > max_size:
                train_idx += 1
                current_train_filename = f"{FOLDER_PATH}train{train_idx}.txt"
                current_train_size = 0

            with open(current_train_filename, 'a') as train_file:
                train_file.write(text)
                current_train_size += len(encoded_text)
        
    print('saved data')

tokenizer = tiktoken.get_encoding('r50k_base')
eot = tokenizer._special_tokens['<|endoftext|>']
def tokenize(doc):
        tokens = [eot] + tokenizer.encode(doc['text'])
        token_np = np.array(tokens, dtype=np.uint16)
        return token_np

def download_shards(folderPath='data/'):
    data = load_dataset('Skylion007/openwebtext', trust_remote_code=True, split='train')
    shard_size = int(1e8)

    os.makedirs(folderPath, exist_ok=True)
    
    n_procs = max(1, mp.cpu_count() // 2)
    with mp.Pool(n_procs) as pool:
        shard_index = 0
        all_np_tokens = np.empty((shard_size,), dtype=np.uint16)
        current_size = 0
        progress_bar = None
        for tokens in pool.imap(tokenize, data, chunksize=16):
            if current_size + len(tokens) < shard_size:
                all_np_tokens[current_size: current_size + len(tokens)] = tokens
                current_size += len(tokens)
                if progress_bar is None:
                    progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_index}")
                progress_bar.update(len(tokens))
            else:
                split = 'val' if shard_index == 0 else 'train'
                remainder = shard_size - current_size
                progress_bar.update(remainder) # type: ignore
                all_np_tokens[current_size: current_size + remainder] = tokens[:remainder]
                np.save(f'{folderPath}/shard_{split}_{shard_index}.npy', all_np_tokens)
                shard_index += 1
                progress_bar = None
                all_np_tokens = np.empty((shard_size,), dtype=np.uint16)
                remainder = shard_size - current_size
                all_np_tokens[0:len(tokens)-remainder] = tokens[remainder:]
                current_size = len(tokens)-remainder
            