import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import warnings
from datetime import datetime
import os
from dataclasses import dataclass
import tiktoken

from src.data_loader import OpenWebText
from src.models import GPT, WaveGPT
from src.transformer import ModelHyperParams

warnings.filterwarnings('ignore')


params = ModelHyperParams()
tokenizer = tiktoken.get_encoding('r50k_base')
today_date = datetime.today().strftime('%d_%m_%y')
os.makedirs('artifacts', exist_ok=True)


@dataclass
class TrainParams:
    epochs:int = 10
    warmup_epochs:int = 3
    eval_step:int = 100
    learning_rate:float = 3e-4

train_params = TrainParams()


# data preparation
os.makedirs('data', exist_ok=True)
DATA_PATH = os.path.join(os.getcwd(), 'data/data.txt')
if not os.path.exists(DATA_PATH):
    from src import download_data
    download_data(DATA_PATH, TAKE=250_000)
train_dataset = OpenWebText(DATA_PATH, split='train')
val_dataset = OpenWebText(DATA_PATH, split = 'val')

# model initialization
gpt = GPT(vocab_size=tokenizer.n_vocab)
gpt_opt = torch.optim.AdamW(gpt.parameters(), lr=train_params.learning_rate)
gpt_lrs = CosineAnnealingLR(gpt_opt, T_max= train_params.epochs - train_params.warmup_epochs, eta_min=3e-8)
gpt.to(params.device)
gpt = torch.compile(gpt)

wave_gpt = WaveGPT(vocab_size=tokenizer.n_vocab)
wave_opt = torch.optim.AdamW(wave_gpt.parameters(), lr=train_params.learning_rate)
wave_lrs = CosineAnnealingLR(wave_opt, T_max= train_params.epochs - train_params.warmup_epochs, eta_min=3e-8)
wave_gpt.to(params.device)
wave_gpt = torch.compile(wave_gpt)

print(f'gpt model has {sum(p.numel() for p in gpt.parameters() if p.requires_grad)} parameters')
print(f'wavegpt model has {sum(p.numel() for p in wave_gpt.parameters() if p.requires_grad)} parameters')


GPT_PATH = 'gpt-model-path'
WAVEGPT_PATH = 'wave-gpt-model-path'
if os.path.exists(GPT_PATH) and os.path.exists(WAVEGPT_PATH):
    gpt_dict = torch.load(GPT_PATH)
    wgpt_dict = torch.load(WAVEGPT_PATH)

    if gpt_dict['epoch'] != wgpt_dict['epoch']:
        raise Exception("Both models are at different training epoch")

    gpt.load_state_dict(gpt_dict['model'])
    gpt_opt.load_state_dict(gpt_dict['opt'])

    wave_gpt.load_state_dict(wgpt_dict['model'])
    wave_opt.load_state_dict(wgpt_dict['opt'])

    last_epoch = gpt_dict['epoch']

else:
    last_epoch = -1


#training loop
if os.path.exists(f'artifacts-path'):
    metrics = torch.load(f'artifacts-path')
    metrics = {key:value.tolist() for key, value in metrics.items()}
else:
    metrics = {
        'gpt_tl' : [],
        'gpt_vl' : [],
        'wave_tl': [],
        'wave_vl': [],
    }

print(f'running on {params.device}')
print('starting training...')
for epoch in range(last_epoch + 1, train_params.epochs + last_epoch + 1):
    pb = tqdm(range(len(train_dataset)), leave=False)
    pb.set_description(f'Train Epoch {epoch}/{train_params.epochs}')
    if epoch > train_params.warmup_epochs:
        gpt_lrs.step()
        wave_lrs.step()
    for step in pb:
        x, x_prev, y = train_dataset[step]
        x = x.to(params.device)
        y = y.to(params.device)
        x_prev = x_prev.to(params.device)

        with torch.autocast(device_type=params.device, dtype=torch.bfloat16):
            g_logits, g_loss = gpt(x, y)
        gpt_opt.zero_grad()
        g_loss.backward()
        gpt_opt.step()
        metrics['gpt_tl'].append(g_loss.item())

        if torch.rand(1).item() < 0.1:
            x_prev = None

        with torch.autocast(device_type=params.device, dtype=torch.bfloat16):
            w_logits, w_loss = wave_gpt(x, x_prev, y)
        wave_opt.zero_grad()
        w_loss.backward()
        wave_opt.step()
        metrics['wave_tl'].append(w_loss.item())

        pb.set_postfix({'wavegpt_loss':w_loss.item(), 'gpt_loss': g_loss.item()})


    gpt.eval()
    wave_gpt.eval()
    pb = tqdm(range(len(val_dataset)), leave=False)
    pb.set_description(f'Val Epoch {epoch}/{train_params.epochs}')
    for step in pb:
        x, x_prev, y = val_dataset[step]
        x = x.to(params.device)
        y = y.to(params.device)
        x_prev = x_prev.to(params.device)

        with torch.no_grad():
            logits, g_loss = gpt(x, y)
        metrics['gpt_vl'].append(g_loss.item())

        if torch.rand(1).item() < 0.1:
            x_prev = None


        with torch.no_grad():
            logits, w_loss = wave_gpt(x, x_prev, y)

        pb.set_postfix({'wavegpt_loss':w_loss.item(), 'gpt_loss': g_loss.item(),})

    gpt.train()
    wave_gpt.train()

    GPT_PATH = f'artifacts/gpt_{today_date}_cp{epoch+1}.pth'
    WAVEGPT_PATH = f'artifacts/wavegpt_{today_date}_cp{epoch+1}.pth'
    torch.save({
        'epoch': epoch,
        'model' : gpt.state_dict(),
        'opt' : gpt_opt.state_dict()
    }, GPT_PATH)
    torch.save({
        'epoch': epoch,
        'model' : wave_gpt.state_dict(),
        'opt' : wave_opt.state_dict()
    }, WAVEGPT_PATH)

    METRICS_PATH = f'artifacts/training_metrics.pt'
    torch.save({key:torch.tensor(metrics[key]) for key in metrics.keys()}, METRICS_PATH)
