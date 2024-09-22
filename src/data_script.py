from datasets import load_dataset
from tqdm import tqdm
import os

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