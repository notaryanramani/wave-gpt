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
    eval_every:int = 1000

train_params = TrainParams()


# data preparation
train_files = [f for f in os.listdir('data/') if f.startswith('train')]
val_files = [f for f in os.listdir('data/') if f.startswith('val')]
BATCH_TOKENS = params.block_size * params.batch_size


# model initialization
m = WaveGPT(vocab_size=tokenizer.n_vocab)
opt = torch.optim.AdamW(m.parameters(), lr=train_params.learning_rate)
lrs = CosineAnnealingLR(opt, T_max= train_params.epochs - train_params.warmup_epochs, eta_min=3e-8)
m.to(params.device)
m = torch.compile(m)
torch.cuda.synchronize()
print(f'model has {sum(p.numel() for p in m.parameters() if p.requires_grad)} parameters') # type: ignore


metrics = {
    'tl': [],
    'vl': [],
}


#training loop
print(f'running on {params.device}')
print('starting training...')
for epoch in range(10):
    if not epoch < train_params.warmup_epochs:
        lrs.step()
    for train_path in train_files:
        train_dataset = OpenWebText(f"data/{train_path}")
        
        pb = tqdm(range(len(train_dataset)), leave=False)
        pb.set_description(f'Train Epoch {epoch+1}/10')
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
            if step % train_params.eval_every == 0:
                with torch.no_grad():
                    val_loss = 0
                    val_idx = torch.randperm(len(val_files))[0]
                    val_path = val_files[val_idx]
                    
                    val_dataset = OpenWebText(f"data/{val_path}")
                    vn = len(val_dataset)
                    for val_step in range(vn):
                        if not val_step < 32:
                            break
                        x, x_prev, y = val_dataset[val_step]
                        x = x.to(params.device)
                        y = y.to(params.device)
                        x_prev = x_prev.to(params.device)
                        if torch.rand(1).item() < 0.1:
                            x_prev = None
                        logits, loss = m(x, x_prev, y)
                        val_loss += loss.item()
                    val_loss /= vn
                    metrics['vl'].append(val_loss)
        
    MODEL_PATH = f'artifacts/model_{today_date}_{epoch}.pt'
    torch.save({
        'model' : m.state_dict(), # type: ignore
        'optimizer' : opt.state_dict(),
    }, MODEL_PATH) 
    
    METRICS_PATH = f'artifacts/training_metrics.pt'
    torch.save(
        {key:torch.tensor(metrics[key]) for key in metrics.keys()}, 
        METRICS_PATH
    )
     
