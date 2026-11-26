"""Fine-tune ImageNet-pretrained ResNet-18 on CIFAR-10 at 224 px.

Intended to run on a Kaggle T4 (see notebooks/kaggle_01_finetune.ipynb), but works on any
CUDA / MPS / CPU device. Mixed-precision (AMP) training is used on CUDA only.
Monitoring uses the ``spare`` split (from CIFAR train), never val/test, so the selection
and final-report splits stay untouched by training decisions.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import torch
import torch.nn as nn

from precisionpilot.data import iterate_batches, preprocess
from precisionpilot.evaluate import accuracy
from precisionpilot.model import build_resnet18, save_checkpoint
from precisionpilot.utils import save_json, set_seed


def train(cfg, data, device: torch.device, out_path: str | Path, pretrained: bool = True,
          max_steps_per_epoch: int | None = None, log=print) -> dict:
    t = cfg.train
    set_seed(cfg.seed)
    model = build_resnet18(num_classes=10, pretrained=pretrained).to(device)
    use_amp = bool(t.amp) and device.type == "cuda"
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
        torch.backends.cudnn.benchmark = True

    x_tr, y_tr = data.get("finetune")
    x_mon, y_mon = data.get("spare")
    steps_per_epoch = math.ceil(len(y_tr) / t.batch_size)
    if max_steps_per_epoch:
        steps_per_epoch = min(steps_per_epoch, max_steps_per_epoch)
    total_steps = max(1, steps_per_epoch * t.epochs)

    opt = torch.optim.SGD(model.parameters(), lr=t.lr, momentum=t.momentum,
                          weight_decay=t.weight_decay, nesterov=True)
    warmup = max(1, int(0.05 * total_steps))

    def lr_at(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        prog = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * prog))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    crit = nn.CrossEntropyLoss(label_smoothing=t.label_smoothing)
    gen = torch.Generator().manual_seed(cfg.seed)

    history = []
    step = 0
    t0 = time.time()
    for epoch in range(t.epochs):
        model.train()
        run_loss, run_correct, run_n = 0.0, 0, 0
        e0 = time.time()
        for i, (xb, yb) in enumerate(iterate_batches(x_tr, y_tr, t.batch_size, shuffle=True,
                                                     seed=cfg.seed + epoch, drop_last=True)):
            if i >= steps_per_epoch:
                break
            x = preprocess(xb, device, cfg.data.image_size, augment=True, generator=gen)
            if device.type == "cuda":
                x = x.contiguous(memory_format=torch.channels_last)
            yb = yb.to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                logits = model(x)
                loss = crit(logits, yb)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            run_loss += loss.item() * len(yb)
            run_correct += (logits.argmax(1) == yb).sum().item()
            run_n += len(yb)
        mon_acc = accuracy(model, x_mon, y_mon, device, cfg.data.eval_batch_size,
                           cfg.data.image_size)
        rec = {"epoch": epoch + 1, "train_loss": run_loss / max(1, run_n),
               "train_acc": 100.0 * run_correct / max(1, run_n), "spare_acc": mon_acc,
               "lr": sched.get_last_lr()[0], "epoch_s": time.time() - e0}
        history.append(rec)
        log(f"epoch {rec['epoch']}/{t.epochs}  loss {rec['train_loss']:.4f}  "
            f"train {rec['train_acc']:.2f}%  spare {rec['spare_acc']:.2f}%  "
            f"({rec['epoch_s']:.0f}s)")

    model = model.to(memory_format=torch.contiguous_format)
    meta = {"epochs": t.epochs, "steps": step, "train_seconds": time.time() - t0,
            "final_spare_acc": history[-1]["spare_acc"] if history else None,
            "pretrained": pretrained, "seed": cfg.seed, "image_size": cfg.data.image_size,
            "device": str(device), "amp": use_amp, "history": history}
    save_checkpoint(model.cpu(), out_path, meta)
    save_json(meta, Path(out_path).with_suffix(".json"))
    log(f"saved checkpoint -> {out_path}")
    return meta
