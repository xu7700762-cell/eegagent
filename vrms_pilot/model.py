"""Small CNN trained from scratch with path-balanced weak window supervision."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import random
import time

import numpy as np
import torch
from torch import nn


class CompactEEGCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(30, 16, 1, bias=False), nn.BatchNorm1d(16), nn.GELU(),
            nn.Conv1d(16, 16, 33, stride=4, padding=16, groups=16, bias=False),
            nn.BatchNorm1d(16), nn.GELU(),
            nn.Conv1d(16, 24, 9, stride=4, padding=4, bias=False),
            nn.BatchNorm1d(24), nn.GELU())
        self.head = nn.Linear(48, 1)

    def forward(self, x):
        z = self.features(x)
        pooled = torch.cat((z.mean(dim=-1), z.std(dim=-1, correction=0)), dim=1)
        return self.head(pooled).squeeze(-1)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(2)


def train_cnn(windows, paths, train_indices, device, epochs=12, seed=2026):
    seed_everything(seed)
    model = CompactEEGCNN().to(device)
    ids, y, weights = [], [], []
    usable = [paths[i] for i in train_indices if paths[i]["accepted_windows"]]
    counts = np.bincount([p["label"] for p in usable], minlength=2)
    if min(counts) == 0:
        raise ValueError("Base training requires both path classes")
    total_windows = sum(p["accepted_windows"] for p in usable)
    for path in usable:
        ix = list(range(path["window_start"], path["window_end"]))
        ids.extend(ix)
        y.extend([path["label"]] * len(ix))
        weight = total_windows / (2 * counts[path["label"]] * len(ix))
        weights.extend([weight] * len(ix))
    x = torch.from_numpy(np.array(windows[ids], dtype=np.float32)).to(device)
    mean = x.mean(dim=(0, 2), keepdim=True)
    scale = torch.sqrt(((x - mean) ** 2).mean(dim=(0, 2), keepdim=True)).clamp_min(0.01)
    x = ((x - mean) / scale).clamp(-20, 20)
    targets = torch.tensor(y, dtype=torch.float32, device=device)
    sample_weights = torch.tensor(weights, dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.001)
    begin = time.perf_counter()
    losses = []
    model.train()
    for _ in range(epochs):
        order = torch.randperm(len(x), device=device)
        loss_sum = 0.0
        for start in range(0, len(order), 128):
            batch = order[start:start + 128]
            optimizer.zero_grad(set_to_none=True)
            logits = model(x[batch])
            loss = (nn.functional.binary_cross_entropy_with_logits(logits, targets[batch], reduction="none")
                    * sample_weights[batch]).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_sum += float(loss.detach()) * len(batch)
        losses.append(loss_sum / len(x))
    if device.type == "cuda":
        torch.cuda.synchronize()
    model.eval()
    del x, targets, sample_weights
    return model, mean.detach(), scale.detach(), dict(losses=losses, train_seconds=time.perf_counter() - begin,
                                                    train_windows=len(ids), train_paths=len(usable),
                                                    parameters=sum(p.numel() for p in model.parameters()))


def predict_windows(model, mean, scale, windows, device, batch_size=128):
    predictions = []
    with torch.inference_mode():
        for start in range(0, len(windows), batch_size):
            x = torch.from_numpy(np.array(windows[start:start + batch_size], dtype=np.float32)).to(device)
            logits = model(((x - mean) / scale).clamp(-20, 20))
            predictions.extend(logits.cpu().numpy().astype(float).tolist())
    return np.asarray(predictions)
