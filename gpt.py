import torch
from preprocess import OpenWebText
from models import GPT
import mlflow
from dataclasses import dataclass, fields
import tiktoken
from transformer import ModelHyperParams
from tqdm import tqdm
from utils import accuracy
import warnings
from datetime import datetime
import os
warnings.filterwarnings('ignore')


params = ModelHyperParams()
tokenizer = tiktoken.get_encoding('r50k_base')
today_date = datetime.today().strftime('%d_%m_%y')
os.makedirs('artifacts', exist_ok=True)

@dataclass
class TrainParams:
    epochs:int = 2
    eval_step:int = 100
    learning_rate:float = 3e-4

train_params = TrainParams()

# setting up mlflow
mlflow.set_tracking_uri('http://localhost:5000')
mlflow.set_experiment('gpt-training')
mlflow.start_run()

# data preparation
data_file_path = 'data/data.txt'
train_dataset = OpenWebText(data_file_path, split='train')
val_dataset = OpenWebText(data_file_path, split = 'val')

# model initialization
m = GPT(vocab_size=tokenizer.n_vocab)
print(f'running on {params.device}')

opt = torch.optim.AdamW(m.parameters(), lr=train_params.learning_rate)

# checkpoint = torch.load('artifacts/gpt_11_07_24_cp0.pth')
# m.load_state_dict(checkpoint['model'])
# m.to(params.device)
# opt.load_state_dict(checkpoint['opt'])
# last_epoch = checkpoint['epoch']
last_epoch = -1

# logging mkflow model params
model_params_dict = {field.name: getattr(params, field.name)  for field in fields(ModelHyperParams)}
train_params_dict = {field.name: getattr(train_params, field.name)  for field in fields(TrainParams)}

params_dict = {**model_params_dict, **train_params_dict}
mlflow.log_params(params_dict)

#training loop
if os.path.exists(f'artifacts/gpt_metrics.pt'):
    metrics = torch.load(f'artifacts/gpt_metrics.pt')
    train_losses = metrics['train_losses'].tolist()
    val_losses = metrics['val_losses'].tolist()
    train_accs = metrics['train_accs'].tolist()
    val_accs = metrics['val_accs'].tolist()
else:
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []



print('starting training...')
for epoch in range(last_epoch + 1, train_params.epochs + last_epoch + 1):
    pb = tqdm(range(len(train_dataset)), ncols=100, leave=False)
    for step in pb:
        x, y = train_dataset[step]
        x = x.to(params.device)
        y = y.to(params.device)
        logits, loss = m(x, y)
        opt.zero_grad()
        loss.backward()
        opt.step()

        y_pred = torch.argmax(logits, dim=-1)
        acc = accuracy(y.view(-1).tolist(), y_pred.view(-1).tolist())

        pb.set_description(f'Train Epoch {epoch}/{train_params.epochs}')
        pb.set_postfix({'loss':loss.item(), 'accuracy' : acc})

        if step % train_params.eval_step == 0:
        
            mlflow.log_metric("train_loss", loss.item(), step=(step//train_params.eval_step))
            mlflow.log_metric("train_accuract", acc, step=(step//train_params.eval_step))

        train_losses.append(loss.item())
        train_accs.append(acc)
    
    m.eval()
    pb = tqdm(range(len(val_dataset)), ncols=100, leave=False)
    for step in pb:
        x, y = val_dataset[step]
        x = x.to(params.device)
        y = y.to(params.device)

        with torch.no_grad():
            logits, loss = m(x, y)

        y_pred = torch.argmax(logits, dim=-1)
        acc = accuracy(y.view(-1).tolist(), y_pred.view(-1).tolist())

        pb.set_description(f'Val Epoch {epoch}/{train_params.epochs}')
        pb.set_postfix({'loss':loss.item(), 'accuracy' : acc})

        if step % train_params.eval_step == 0:
            mlflow.log_metric("val_loss", loss.item(), step=(step//train_params.eval_step))
            mlflow.log_metric("val_accuract", acc, step=(step//train_params.eval_step))

        val_losses.append(loss.item())
        val_accs.append(acc)
    
    m.train()

    PATH = f'artifacts/gpt_{today_date}_cp{epoch}.pth'
    torch.save({
        'epoch': epoch,
        'model' : m.state_dict(),
        'opt' : opt.state_dict()
    }, PATH)

    METRICS_PATH = f'artifacts/gpt_metrics.pt'
    torch.save({
        'train_losses' : torch.tensor(train_losses),
        'train_accs' : torch.tensot(train_accs),
        'val_losses' : torch.tensor(val_losses),
        'val_accs' : torch.tensor(val_accs)
    })

# ending ml flow run
mlflow.end_run()