import torch
from src import WaveGPT, OpenWebText, ModelHyperParams
from dataclasses import dataclass
import tiktoken
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
    epochs:int = 4
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
print(f'model has {sum(p.numel() for p in m.parameters() if p.requires_grad)} parameters')


PATH = 'artifacts\wavegpt_e_20_07_24_cp6.pth'
if os.path.exists(PATH):
    checkpoint = torch.load(PATH)
    m.load_state_dict(checkpoint['model'])
    m.to(params.device)
    opt.load_state_dict(checkpoint['opt'])
    last_epoch = checkpoint['epoch']
else:
    last_epoch = -1


if os.path.exists(f'artifacts/training_metrics_e.pt'):
    metrics = torch.load(f'artifacts/training_metrics_e.pt')
    metrics = {key:value.tolist() for key, value in metrics.items()}
else:
    metrics = {
        'tl': [],
        'vl': [],
        'ta': [],
        'va': [],
    }


#training loop
print(f'running on {params.device}')
print('starting training...')
for epoch in range(last_epoch + 1, train_params.epochs + last_epoch + 1):
    pb = tqdm(range(len(train_dataset)), leave=False)
    pb.set_description(f'Train Epoch {epoch}/{train_params.epochs + last_epoch + 1}')
    for step in pb:
        x, y = train_dataset[step]
        x = x.to(params.device)
        y = y.to(params.device)

        logits, loss = m(x, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        y_pred = torch.argmax(logits, dim=-1)
        acc = float(accuracy_score(y.view(-1).tolist(), y_pred.view(-1).tolist()))
        metrics['tl'].append(loss.item())
        metrics['ta'].append(acc)

        pb.set_postfix({'loss':loss.item(), 'acc' : acc})

        if step % train_params.eval_step == 0:
            pass


    m.eval()
    pb = tqdm(range(len(val_dataset)), leave=False)
    pb.set_description(f'Val Epoch {epoch}/{train_params.epochs + last_epoch + 1}')
    for step in pb:
        x, y = val_dataset[step]
        x = x.to(params.device)
        y = y.to(params.device)


        with torch.no_grad():
            logits, loss = m(x, y)
        y_pred = torch.argmax(logits, dim=-1)
        acc = float(accuracy_score(y.view(-1).tolist(), y_pred.view(-1).tolist()))
        metrics['vl'].append(loss.item())
        metrics['va'].append(acc)

        pb.set_postfix({'loss':loss.item(), 'acc' : acc})

        if step % train_params.eval_step == 0:
            pass

    m.train()


    PATH = f'artifacts/wavegpt_e_{today_date}_cp{epoch}.pth'
    torch.save({
        'epoch': epoch,
        'model' : m.state_dict(),
        'opt' : opt.state_dict()
    }, PATH)

    METRICS_PATH = f'artifacts/training_metrics_e.pt'
    torch.save({key:torch.tensor(metrics[key]) for key in metrics.keys()}, METRICS_PATH)
