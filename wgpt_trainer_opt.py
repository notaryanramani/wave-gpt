import torch
from src_opt.preprocess import OpenWebText
from src_opt.models import WaveGPT
from dataclasses import dataclass
import tiktoken
from src_opt.transformer import ModelHyperParams
from tqdm import tqdm
import time


params = ModelHyperParams()
tokenizer = tiktoken.get_encoding('r50k_base')


@dataclass
class TrainParams:
    epochs:int = 5
    eval_step:int = 80
    learning_rate:float = 3e-4

train_params = TrainParams()


# data preparation
data_file_path = 'data/data.txt'
train_dataset = OpenWebText(data_file_path, split='train')
val_dataset = OpenWebText(data_file_path, split = 'val')


# model initialization
m = WaveGPT(vocab_size=tokenizer.n_vocab)
opt = torch.optim.AdamW(m.parameters(), lr=train_params.learning_rate)
m.to(params.device)
# m = torch.compile(m)
print(f'model has {sum(p.numel() for p in m.parameters() if p.requires_grad)} parameters')


#training loop
print(f'running on {params.device}')
print('starting training...')
for epoch in range(train_params.epochs):
    pb = tqdm(range(len(train_dataset)), leave=False)
    pb.set_description(f'Train Epoch {epoch}/{train_params.epochs}')
    for step in pb:
        x, y = train_dataset[step]
        x = x.to(params.device)
        y = y.to(params.device)

        t1 = time.time()
        logits, loss = m(x, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        torch.cuda.synchronize()
        t2 = time.time()
        t3 = (t2-t1) * 1000
        toks_per_sec = (len(x.view(-1)))/ t3
        pb.set_postfix({'loss':loss.item(), 'time(ms)' : f'{t3:.4f}', 'toks/ms' : f'{toks_per_sec:.4f}'})


    m.eval()
    pb = tqdm(range(len(val_dataset)), leave=False)
    pb.set_description(f'Val Epoch {epoch}/{train_params.epochs}')
    for step in pb:
        x, y = val_dataset[step]
        x = x.to(params.device)
        y = y.to(params.device)


        with torch.no_grad():
            logits, loss = m(x, y)
        pb.set_postfix({'loss':loss.item()})
    m.train()