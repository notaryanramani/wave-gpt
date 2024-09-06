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

metrics = {
    'tl' : [],
    'vl' : []
}

#training loop
print(f'running on {params.device}')
print('starting training...')
for epoch in range(train_params.epochs):
    if epoch > 7:
        for group in opt.param_groups:
            group['lr'] = train_params.learning_rate / 10
    pb = tqdm(enumerate(train_dataloader), leave=False, position=0)
    pb.set_description(f'Train Epoch {epoch}/{train_params.epochs}')
    for i, (x, x_prev, y) in pb:
        x, x_prev, y = encode(x), encode(x_prev), encode(y)
        x = x.to(params.device)
        x_prev = x_prev.to(params.device)
        y = y.to(params.device)
        t1 = time.time()
        logits, loss = m(x, x_prev, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        metrics['train_loss'].append(loss.item())
        torch.cuda.synchronize()
        t2 = time.time()
        t3 = (t2-t1) * 1000
        toks_per_sec = (len(x.view(-1)))/ t3
        pb.set_postfix({'loss':loss.item(), 'time(ms)' : f'{t3:.4f}', 'toks/ms' : f'{toks_per_sec:.4f}'})
        if i % train_params.eval_step == 0:
            m.eval()
            temp_losses = []
            for i, (x, x_prev, y) in enumerate(val_dataloader):
                if i > 50:
                    break
                x, x_prev, y = encode(x), encode(x_prev), encode(y)
                x = x.to(params.device)
                x_prev = x_prev.to(params.device)
                y = y.to(params.device)

                with torch.no_grad():
                    logits, loss = m(x, x_prev, y)
                temp_losses.append(loss.item())
            batch_loss = sum(temp_losses) / len(temp_losses)
            metrics['val_loss'].append(batch_loss)
            m.train()

torch.save(metrics, 'artifacts/shakespeare_metrics.pth')