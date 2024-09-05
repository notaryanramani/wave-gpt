import torch
from torch.utils.data import DataLoader
from src import GPT, Shakespeare, ModelHyperParams
from dataclasses import dataclass
from tqdm import tqdm
import time


params = ModelHyperParams()


@dataclass
class TrainParams:
    epochs:int = 5
    eval_step:int = 80
    learning_rate:float = 3e-4

train_params = TrainParams()


# data preparation
train = Shakespeare('data/train.txt')
val = Shakespeare('data/test.txt')
train_dataloader = DataLoader(train, batch_size=params.batch_size, shuffle=True)
val_dataloader = DataLoader(val, batch_size=params.batch_size, shuffle=True)

with open('data/input.txt') as f:
    text = f.read()
    tokens = set(list(text))

chtoi = {s:i for i, s in enumerate(tokens)}
itoch = {i:s for s, i in chtoi.items()}

def encode(strings):
    stacks = []
    for string in strings:
        stack = torch.tensor([chtoi[ch] for ch in string], dtype=torch.int64)
        stacks.append(stack)
    out = torch.stack(stacks, dim=1)
    return out

# model initialization
m = GPT(vocab_size=len(tokens))
opt = torch.optim.AdamW(m.parameters(), lr=train_params.learning_rate)
m.to(params.device)
print(f'model has {sum(p.numel() for p in m.parameters() if p.requires_grad)} parameters')


#training loop
print(f'running on {params.device}')
print('starting training...')
for epoch in range(train_params.epochs):
    pb = tqdm(train_dataloader, leave=False, position=0)
    pb.set_description(f'Train Epoch {epoch}/{train_params.epochs}')
    for x, y in pb:
        x, y = encode(x), encode(y)
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
    pb = tqdm(val_dataloader, leave=False)
    pb.set_description(f'Val Epoch {epoch}/{train_params.epochs}')
    for x, y in pb:
        x, y = encode(x), encode(y)
        x = x.to(params.device)
        y = y.to(params.device)

        with torch.no_grad():
            logits, loss = m(x, y)
        pb.set_postfix({'loss':loss.item()})
    m.train()