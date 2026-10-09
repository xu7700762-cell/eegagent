# -*- coding: utf-8 -*-
import hashlib
import time
import numpy as np
import torch
from torch import nn
from .model import PathMIL
SEED=2026
EPOCHS=12

def seed_everything(seed=SEED):
    if seed != 2026:
        raise ValueError("The active training recipe fixes seed2026")
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def state_digest(model):
    sha = hashlib.sha256()
    for key, tensor in sorted(model.state_dict().items()):
        sha.update(key.encode())
        sha.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return sha.hexdigest()


def base_window_indices(paths, base):
    parts = [np.arange(paths[i]["window_start"], paths[i]["window_end"]) for i in base]
    return np.concatenate(parts).astype(int)


def fit_mil(features, paths, base, device, epochs=EPOCHS):
    seed_everything(SEED)
    usable = np.asarray([i for i in base if paths[i]["accepted_windows"]], int)
    yy = np.asarray([paths[i]["label"] for i in usable])
    counts = np.bincount(yy, minlength=2)
    if len(usable) < 6 or min(counts) == 0:
        raise ValueError("Base14 MIL training requires both path classes")
    reference = features[base_window_indices(paths, usable)]
    center, scale = reference.mean(axis=0), np.maximum(reference.std(axis=0), 1e-4)
    head = PathMIL(features.shape[1], center, scale).to(device)
    initial_sha = state_digest(head)
    optimizer = torch.optim.AdamW(head.parameters(), lr=.001, weight_decay=.001)
    feature_tensor = torch.from_numpy(np.asarray(features)).to(device)
    rng = np.random.default_rng(SEED)
    history = []
    begin = time.perf_counter()
    head.train()
    for epoch in range(epochs):
        order = rng.permutation(usable)
        losses = []
        for start in range(0, len(order), 4):
            optimizer.zero_grad(set_to_none=True)
            batch_losses = []
            for i in order[start:start + 4]:
                path = paths[i]
                logit = head(feature_tensor[path["window_start"]:path["window_end"]])
                target = torch.tensor(float(path["label"]), device=device)
                weight = len(usable) / (2 * counts[path["label"]])
                batch_losses.append(nn.functional.binary_cross_entropy_with_logits(logit, target) * weight)
            loss = torch.stack(batch_losses).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(head.parameters(), 1.)
            optimizer.step()
            losses.extend(float(v.detach()) for v in batch_losses)
        history.append(dict(epoch=epoch + 1, path_weighted_train_bce=float(np.mean(losses))))
    head.eval()
    return head, dict(epochs=epochs, selected_epoch=epochs, selection="fixed_final_epoch_no_validation_selection",
                      training_indices=usable.tolist(), train_paths=len(usable),
                      train_subjects=sorted({paths[i]["subject_key"] for i in usable}),
                      feature_normalization_subjects=sorted({paths[i]["subject_key"] for i in usable}),
                      parameters=sum(p.numel() for p in head.parameters()),
                      initial_state_sha256=initial_sha, final_state_sha256=state_digest(head),
                      history=history, seconds=time.perf_counter() - begin)
