from dataclasses import dataclass
import os
from datetime import datetime
import time
import logging

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from tqdm import tqdm
import tiktoken

from src import WaveGPT, ShardsLoader, ModelHyperParams


# Parameters setup
params = ModelHyperParams()

@dataclass
class TrainParams:
    steps:int = 20000
    learning_rate:float = 3e-4
    warmup_epochs:int = 3
    eval_every:int = 1000
    val_batch_size:int = 50
    checkpoint:bool = True
    checkpoint_every:int = 1000

train_params = TrainParams()

# ddp setup
use_ddp = int(os.environ.get('WORLD_SIZE', 1)) > 1
if use_ddp:
    assert torch.cuda.is_available(), 'cuda is not available for ddp'
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
else:
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    elif hasattr(torch, 'backends') and torch.backends.mps.is_available():
        device = 'mps'
    
device_type = 'cuda' if device.startswith('cuda') else 'cpu'

total_batch_size = 131072
assert total_batch_size % (params.batch_size * params.block_size * ddp_world_size) == 0, 'batch size must be divisible by world size'
grad_accum = total_batch_size // (params.batch_size * params.block_size * ddp_world_size)


# data preparation
train_loader = ShardsLoader(
    process_rank=ddp_rank,
    num_processes=ddp_world_size,
    split='train'
)

val_loader = ShardsLoader(
    process_rank=ddp_rank,
    num_processes=ddp_world_size,
    split='val'
)


# model initialization
model = WaveGPT(vocab_size=50304)
model.to(device) 
if use_ddp:
    init_process_group(backend='nccl', init_method='env://')
    model = DDP(model, device_ids=[ddp_local_rank], output_device=device)
    torch.cuda.synchronize()
raw_model = model.module if use_ddp else model 
optim = torch.optim.AdamW(model.parameters(), lr=train_params.learning_rate) 
lrs = CosineAnnealingLR(optim, T_max=train_params.steps - train_params.warmup_epochs, eta_min=3e-8)

if master_process:
    os.makedirs('logs', exist_ok=True)
    logging.basicConfig(
        filename=f'logs/{datetime.today()}.log',
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        datefmt='%d-%b-%y %H:%M:%S'
    )
    if train_params.checkpoint:
        os.makedirs('checkpoints', exist_ok=True)
        metrics = {
            'tl': [],
            'vl': [],
        }
    logging.info(f'checkpointing: {train_params.checkpoint}')
    logging.info(f'total batch size: {total_batch_size}')
    logging.info(f'grad accum step: {grad_accum}')
    logging.info(f'model has {sum(p.numel() for p in raw_model.parameters() if p.requires_grad)} parameters') # type: ignore
    logging.info(f'running on {device_type}')
    logging.info('starting training...')
    

# training loop
pb = tqdm(range(train_params.steps), leave=False) if master_process else range(train_params.steps)
for step in pb:
    t0 = time.time()
    loss_accum = 0.
    optim.zero_grad()
    for mini_step in range(grad_accum):
        x, x_prev, y = train_loader.get_batch()
        x, x_prev, y = x.to(device), x_prev.to(device), y.to(device)
        if torch.rand(1).item() < 0.1:
            x_prev = None
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            logits, loss = model(x, x_prev, y) # type: ignore
        loss = loss / grad_accum
        loss_accum += loss.detach()
        loss.backward()
    if use_ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
    if not step < train_params.warmup_epochs:
        lrs.step()
    optim.step()
    if device_type == 'cuda':
        torch.cuda.synchronize()
    t1 = time.time()
    dt = t1 - t0
    if master_process:
        tokens_per_sec = total_batch_size / dt
        metrics['tl'].append(loss_accum)
        logging.info(f'step: {step}, loss: {loss_accum}')
        pb.set_postfix_str(f'step: {step}, loss: {loss_accum}, time: {dt:.4f}s, toks/s: {tokens_per_sec:.4f}')

if use_ddp:
    destroy_process_group()
