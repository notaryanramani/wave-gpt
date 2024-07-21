import torch
from src.preprocess import OpenWebText
from src.models import GPT, WaveGPT
from dataclasses import dataclass
import tiktoken
from src.transformer import ModelHyperParams
from tqdm import tqdm
import warnings
from datetime import datetime
import os
from sklearn.metrics import accuracy_score
warnings.filterwarnings('ignore')


params = ModelHyperParams()
tokenizer = tiktoken.get_encoding('r50k_base')
today_date = datetime.today().strftime('%d_%m_%y')
os.makedirs('artifacts', exist_ok=True)


@dataclass
class TrainParams:
    epochs:int = 5
    eval_step:int = 100
    learning_rate:float = 3e-4

train_params = TrainParams()


# data preparation
data_file_path = 'data/data.txt'
train_dataset = OpenWebText(data_file_path, split='train')
val_dataset = OpenWebText(data_file_path, split = 'val')

# model initialization
gpt = GPT(vocab_size=tokenizer.n_vocab)
gpt_opt = torch.optim.AdamW(gpt.parameters(), lr=train_params.learning_rate)
gpt.to(params.device)

wave_gpt = WaveGPT(vocab_size=tokenizer.n_vocab)
wave_opt = torch.optim.AdamW(wave_gpt.parameters(), lr=train_params.learning_rate)
wave_gpt.to(params.device)

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
if os.path.exists(f'artifacts/training_metrics.pt'):
    metrics = torch.load(f'artifacts/training_metrics.pt')
    metrics = {key:value.tolist() for key, value in metrics.items()}
else:
    metrics = {
        'gpt_tl' : [],
        'gpt_vl' : [],
        'gpt_ta' : [],
        'gpt_va' : [],
        'wave_tl': [],
        'wave_vl': [],
        'wave_ta': [],
        'wave_va': [],
    }

print(f'running on {params.device}')
print('starting training...')
for epoch in range(last_epoch + 1, train_params.epochs + last_epoch + 1):
    pb = tqdm(range(len(train_dataset)), leave=False)
    pb.set_description(f'Train Epoch {epoch}/{train_params.epochs}')
    for step in pb:
        x, y = train_dataset[step]
        x = x.to(params.device)
        y = y.to(params.device)

        g_logits, g_loss = gpt(x, y)
        gpt_opt.zero_grad()
        g_loss.backward()
        gpt_opt.step()
        y_pred = torch.argmax(g_logits, dim=-1)
        g_acc = float(accuracy_score(y.view(-1).tolist(), y_pred.view(-1).tolist()))
        metrics['gpt_tl'].append(g_loss.item())
        metrics['gpt_ta'].append(g_acc)

        w_logits, w_loss = wave_gpt(x, y)
        wave_opt.zero_grad()
        w_loss.backward()
        wave_opt.step()
        y_pred = torch.argmax(w_logits, dim=-1)
        w_acc = float(accuracy_score(y.view(-1).tolist(), y_pred.view(-1).tolist()))
        metrics['wave_tl'].append(w_loss.item())
        metrics['wave_ta'].append(w_acc)

        pb.set_postfix({'gpt_loss': g_loss.item(), 'gpt_acc': g_acc,'wgpt_loss':w_loss.item(), 'wgpt_acc' : w_acc})

        if step % train_params.eval_step == 0:
            pass
            # add your code if you want a different evaluation at a time step


    gpt.eval()
    wave_gpt.eval()
    pb = tqdm(range(len(val_dataset)), leave=False)
    pb.set_description(f'Val Epoch {epoch}/{train_params.epochs}')
    for step in pb:
        x, y = val_dataset[step]
        x = x.to(params.device)
        y = y.to(params.device)

        with torch.no_grad():
            logits, g_loss = gpt(x, y)
        y_pred = torch.argmax(logits, dim=-1)
        g_acc = float(accuracy_score(y.view(-1).tolist(), y_pred.view(-1).tolist()))
        metrics['gpt_vl'].append(g_loss.item())
        metrics['gpt_va'].append(g_acc)

        with torch.no_grad():
            logits, w_loss = wave_gpt(x, y)
        y_pred = torch.argmax(logits, dim=-1)
        w_acc = float(accuracy_score(y.view(-1).tolist(), y_pred.view(-1).tolist()))
        metrics['wave_vl'].append(w_loss.item())
        metrics['wave_va'].append(w_acc)

        pb.set_postfix({'gpt_loss': g_loss.item(), 'gpt_acc': g_acc,'wgpt_loss':w_loss.item(), 'wgpt_acc' : w_acc})

        if step % train_params.eval_step == 0:
            pass
            # add your code if you want a different evaluation at a time step


    gpt.train()
    wave_gpt.train()

    GPT_PATH = f'artifacts/gpt_{today_date}_cp{epoch}.pth'
    WAVEGPT_PATH = f'artifacts/wavegpt_{today_date}_cp{epoch}.pth'
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
