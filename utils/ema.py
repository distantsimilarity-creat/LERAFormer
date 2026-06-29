import copy
import torch


class ModelEMA:
    def __init__(self, model, decay=0.9999, device=None):
        self.decay = float(decay)
        self.device = device
        self.shadow = {}
        self.backup = {}
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[name] = p.detach().clone().to(device=device) if device is not None else p.detach().clone()

    @torch.no_grad()
    def update(self, model):
        d = self.decay
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if name not in self.shadow:
                self.shadow[name] = p.detach().clone()
            new = p.detach()
            if self.device is not None:
                new = new.to(self.device)
            self.shadow[name].mul_(d).add_(new, alpha=1.0 - d)

    @torch.no_grad()
    def apply_to(self, model):
        self.backup = {}
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            self.backup[name] = p.detach().clone()
            ema_p = self.shadow.get(name, None)
            if ema_p is not None:
                p.data.copy_(ema_p.to(p.device))

    @torch.no_grad()
    def restore(self, model):
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if name in self.backup:
                p.data.copy_(self.backup[name].to(p.device))
        self.backup = {}

    def state_dict(self):
        return {k: v.detach().cpu() for k, v in self.shadow.items()}

    def load_state_dict(self, state_dict):
        self.shadow = {k: v.detach().clone() for k, v in state_dict.items()}
