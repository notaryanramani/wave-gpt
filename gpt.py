import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from preprocess import train_val_split, OpenWebText
from models import GPT
import mlflow
from dataclasses import dataclass, fields
import tiktoken
from transformer import ModelHyperParams
from tqdm import tqdm
from utils import accuracy

params = ModelHyperParams()
tokenizer = tiktoken.get_encoding('r50k_base')

@dataclass
class TrainParams:
    epochs:int = 5
    eval_step:int = 1000
    learning_rate:float = 3e-4

train_params = TrainParams()

# setting up mlflow
mlflow.set_tracking_uri('http://localhost:5000')
mlflow.set_experiment('gpt-training')

mlflow.start_run()


# data preparation
('starting data preparation...')
data_file_path = 'data/data.txt'

train_dataset = OpenWebText(data_file_path, split='train')
val_dataset = OpenWebText(data_file_path, split = 'val')

# model initialization
m = GPT(vocab_size=tokenizer.n_vocab)
print(f'running on {params.device}')
m.to(params.device)
opt = torch.optim.AdamW(m.parameters(), lr=train_params.learning_rate)

# logging mkflow model params
model_params_dict = {field.name: getattr(params, field.name)  for field in fields(ModelHyperParams)}
train_params_dict = {field.name: getattr(train_params, field.name)  for field in fields(TrainParams)}

params_dict = {**model_params_dict, **train_params_dict}
mlflow.log_params(params_dict)

#training loop
# train_losses = []
# val_losses = []
# train_accs = []
# val_accs = []
print('starting training...')
for epoch in range(train_params.epochs):
    pb = tqdm(range(len(train_dataset)), ncols=100)
    for step in pb:
        x, y = train_dataset[step]
        x = x.to(params.device)
        y = y.to(params.device)
        logits, loss = m(x, y)

        opt.zero_grad()
        loss.backward()
        opt.step()

        y_pred = torch.argmax(logits, dim=-1)
        acc = accuracy(y.tolist(), y_pred.tolist())

        pb.set_description(f'Train Epoch {epoch}/{train_params.epochs}')
        pb.set_postfix({'loss':loss.item(), 'accuracy' : acc})

        if step % train_params.eval_step == 0:
        
            mlflow.log_metric("train_loss", loss.item(), step=(step//train_params.eval_step))
            mlflow.log_metric("train_accuract", acc, step=(step//train_params.eval_step))

            # train_losses.append(loss.item())
            # train_accs.append(acc)
    
    m.eval()
    pb = tqdm(range(len(val_dataset)), ncols=100)
    for step in pb:
        x, y = val_dataset[step]
        x = x.to(params.device)
        y = y.to(params.device)

        with torch.no_grad():
            logits, loss = m(x, y)

        y_pred = torch.argmax(logits, dim=-1)
        acc = accuracy(y.tolist(), y_pred.tolist())

        pb.set_description(f'Val Epoch {epoch}/{train_params.epochs}')
        pb.set_postfix({'loss':loss.item(), 'accuracy' : acc})

        if step % train_params.eval_step == 0:
            mlflow.log_metric("val_loss", loss.item(), step=(step//train_params.eval_step))
            mlflow.log_metric("val_accuract", acc, step=(step//train_params.eval_step))

            # val_losses.append(loss.item())
            # val_accs.append(acc)
    
    m.train()



# ending ml flow run
mlflow.end_run()