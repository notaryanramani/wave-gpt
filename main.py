import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from dataclasses import dataclass
import tiktoken
from tqdm import tqdm
from datetime import datetime
import os
import time

from src import WaveGPT, OpenWebText, ModelHyperParams

torch.set_float32_matmul_precision('high')


params = ModelHyperParams()
tokenizer = tiktoken.get_encoding('r50k_base')
today_date = datetime.today().strftime('%d_%m_%y')
os.makedirs('artifacts', exist_ok=True)


@dataclass
class TrainParams:
    epochs:int = 10
    learning_rate:float = 3e-4
    warmup_epochs:int = 3

train_params = TrainParams()


# data preparation
os.makedirs('data', exist_ok=True)
DATA_PATH = os.path.join(os.getcwd(), 'data/data.txt')
if not os.path.exists(DATA_PATH):
    from src import download_data
    download_data(DATA_PATH)
train_dataset = OpenWebText(DATA_PATH, split='train')
val_dataset = OpenWebText(DATA_PATH, split = 'val')

BATCH_TOKENS = train_dataset.block_size * train_dataset.batch_size


# model initialization
m = WaveGPT(vocab_size=tokenizer.n_vocab)
opt = torch.optim.AdamW(m.parameters(), lr=train_params.learning_rate)
lrs = CosineAnnealingLR(opt, T_max= train_params.epochs - train_params.warmup_epochs, eta_min=3e-8)
m.to(params.device)
m = torch.compile(m)
torch.cuda.synchronize()
print(f'model has {sum(p.numel() for p in m.parameters() if p.requires_grad)} parameters')


PATH = 'model-path'
if os.path.exists(PATH):
    checkpoint = torch.load(PATH)
    m.load_state_dict(checkpoint['model'])
    m.to(params.device)
    opt.load_state_dict(checkpoint['opt'])
    last_epoch = checkpoint['epoch']
else:
    last_epoch = -1


if os.path.exists(f'artifacts-path'):
    metrics = torch.load(f'artifacts-path')
    metrics = {key:value.tolist() for key, value in metrics.items()}
else:
    metrics = {
        'tl': [],
        'vl': [],
    }


#training loop
print(f'running on {params.device}')
print('starting training...')
for epoch in range(last_epoch + 1, train_params.epochs + last_epoch + 1):
    pb = tqdm(range(len(train_dataset)), leave=False)
    pb.set_description(f'Train Epoch {epoch+1}/{train_params.epochs + last_epoch + 1}')
    if epoch > train_params.warmup_epochs:
        lrs.step()
    for step in pb:
        x, x_prev, y = train_dataset[step]
        x = x.to(params.device)
        y = y.to(params.device)
        x_prev = x_prev.to(params.device)
        if torch.rand(1).item() < 0.1:
            x_prev = None
        t1 = time.time()
        with torch.autocast(device_type=params.device, dtype=torch.bfloat16):
            logits, loss = m(x, x_prev, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        torch.cuda.synchronize()
        t2 = time.time()
        t3 = (t2-t1) * 1000
        toks_per_sec = (BATCH_TOKENS) / (t2 - t1)

        metrics['tl'].append(loss.item())

        pb.set_postfix({'loss':loss.item(), 'time' : f'{t3:.4f}ms', 'toks/s' : f'{toks_per_sec:.4f}'})


    m.eval()
    pb = tqdm(range(len(val_dataset)), leave=False)
    pb.set_description(f'Val Epoch {epoch+1}/{train_params.epochs + last_epoch + 1}')
    for step in pb:
        x, x_prev, y = train_dataset[step]
        x = x.to(params.device)
        y = y.to(params.device)
        x_prev = x_prev.to(params.device)

        with torch.no_grad():
            logits, loss = m(x, x_prev, y)
        y_pred = torch.argmax(logits, dim=-1)
        
        metrics['vl'].append(loss.item())

        pb.set_postfix({'loss':loss.item()})


    m.train()


    PATH = f'artifacts/wavegpt_{today_date}_cp{epoch}.pth'
    torch.save({
        'epoch': epoch,
        'model' : m.state_dict(),
        'opt' : opt.state_dict()
    }, PATH)

    METRICS_PATH = f'artifacts/training_metrics.pt'
    torch.save({key:torch.tensor(metrics[key]) for key in metrics.keys()}, METRICS_PATH)
