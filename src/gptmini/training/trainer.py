import os, math
import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from tqdm.auto import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def evaluate(model, loader, device) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(input_ids=xb, labels=yb)
            losses.append(out.loss.item())
    return float(np.mean(losses))


def perplexity(loss: float) -> float:
    return math.exp(loss)


def plot_losses(train_losses, val_losses, out_path: str):
    plt.figure()
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Validation")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Training Curve")
    plt.legend(); plt.savefig(out_path); plt.close()


class Trainer:
    def __init__(self, model, device, lr: float, out_dir: str, use_amp: bool):
        self.model = model.to(device)
        self.device = device
        self.opt = torch.optim.AdamW(model.parameters(), lr=lr)
        self.amp_enabled = (device.type == "cuda" and use_amp)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp_enabled)
        os.makedirs(out_dir, exist_ok=True)
        self.out_dir = out_dir


    def fit(self, train_loader, val_loader, epochs: int, clip_norm: float = 1.0):
        tr_losses, va_losses = [], []
        for ep in range(epochs):
            self.model.train()
            loop = tqdm(train_loader, desc=f"Epoch {ep+1}/{epochs}")
            batch_losses = []
            for xb, yb in loop:
                xb, yb = xb.to(self.device), yb.to(self.device)
                self.opt.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=self.amp_enabled):
                    out = self.model(input_ids=xb, labels=yb)
                    loss = out.loss
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.opt)
                clip_grad_norm_(self.model.parameters(), clip_norm)
                self.scaler.step(self.opt)
                self.scaler.update()
                loop.set_postfix(loss=loss.item())
                batch_losses.append(loss.item())


            avg_tr = float(np.mean(batch_losses))
            avg_va = evaluate(self.model, val_loader, self.device)
            tr_losses.append(avg_tr); va_losses.append(avg_va)
            print(f"Epoch {ep+1} | Train {avg_tr:.4f} | Val {avg_va:.4f} | Val PPL {perplexity(avg_va):.4f}")


            torch.save(self.model.state_dict(), os.path.join(self.out_dir, f"model_epoch_{ep+1}.pth"))


        plot_losses(tr_losses, va_losses, os.path.join(self.out_dir, "training_curve.png"))
        return tr_losses, va_losses