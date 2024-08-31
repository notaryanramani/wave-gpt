from torch.optim.lr_scheduler import _LRScheduler, CosineAnnealingLR

class CustomCosineAnnealingLR(_LRScheduler):
    def __init__(self, optimizer, T_max, eta_min=3e-6, warmup_epochs=3, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.cosine_scheduler = CosineAnnealingLR(optimizer, T_max, eta_min, last_epoch)
        super(CustomCosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            return [group['lr'] for group in self.optimizer.param_groups]
        self.cosine_scheduler.step()
        return self.cosine_scheduler.get_lr()
    