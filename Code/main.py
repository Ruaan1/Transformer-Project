import argparse
import csv
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt

# -----------------------------
# Constants
# -----------------------------
PAD_ID = 70
SEP_ID = 68
C1, C2, C3, C4 = 64, 65, 66, 67

# -----------------------------
# Small helpers
# -----------------------------
def no_decay(name: str) -> bool:
    return ("bias" in name) or ("LayerNorm.weight" in name) or ("norm.weight" in name)


def current_lr(optimizer):
    for g in optimizer.param_groups:
        return g.get("lr", None)


def save_confusion_png(cm, out_path, normalize=True, title="Confusion Matrix"):
    cm = np.array(cm, dtype=float)
    labels = ["C1", "C2", "C3", "C4"]

    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm = cm / row_sums

    plt.figure(figsize=(5, 4))
    im = plt.imshow(cm, interpolation="nearest")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(range(4), labels)
    plt.yticks(range(4), labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)

    for i in range(4):
        for j in range(4):
            txt = f"{cm[i, j]*100:.1f}%" if normalize else f"{int(cm[i, j])}"
            plt.text(j, i, txt, ha="center", va="center")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def gather_sep_logits(h, ttids):
    # Cosine similarity between each timestep and the first 4 "category" slots
    # h: (B,T,D)
    cats_h = h[:, :4, :]                                  # (B,4,D)
    h_n    = F.normalize(h, dim=-1)
    cats_n = F.normalize(cats_h, dim=-1)
    all_logits = torch.einsum("btd,bkd->btk", h_n, cats_n)  # (B,T,4)
    return all_logits


def select_masked_logits(all_logits, targets):
    # targets has -100 everywhere except SEP positions where next token is 64..67
    mask = (targets != -100)                               # (B,T)
    logits_sel = all_logits[mask]                          # (N,4)
    targets_sel = (targets[mask] - 64).long()              # (N,)
    return logits_sel, targets_sel, mask


# -----------------------------
# Loss & weights
# -----------------------------
def compute_class_weights_4(dataloader):
    cnt = Counter({64: 0, 65: 0, 66: 0, 67: 0})
    for batch in dataloader:
        t = batch[1]  # (inputs, targets, [ttids])
        labels = t[t != -100].flatten().tolist()
        for c in (64, 65, 66, 67):
            cnt[c] += sum(1 for x in labels if x == c)

    eps = 1e-6
    inv = {c: 1.0 / (cnt[c] + eps) for c in (64, 65, 66, 67)}
    w4 = torch.tensor([inv[64], inv[65], inv[66], inv[67]], dtype=torch.float32)
    w4 = w4 * (4.0 / w4.sum())
    print("4-way class weights:", {c: float(w4[i]) for i, c in enumerate((64, 65, 66, 67))})
    return w4


# -----------------------------
# Transformer
# -----------------------------
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dModel, numHeads, dropout=0.1):
        super().__init__()
        assert dModel % numHeads == 0, "dModel must be divisible by numHeads"
        self.dModel = dModel
        self.numHeads = numHeads
        self.dK = dModel // numHeads

        self.Wq = nn.Linear(dModel, dModel)
        self.Wk = nn.Linear(dModel, dModel)
        self.Wv = nn.Linear(dModel, dModel)
        self.Wo = nn.Linear(dModel, dModel)
        self.dropout = nn.Dropout(dropout)

    def splitHeads(self, x):
        b, t, _ = x.size()
        return x.view(b, t, self.numHeads, self.dK).transpose(1, 2)

    def combineHeads(self, x):
        b, h, t, dK = x.size()
        return x.transpose(1, 2).contiguous().view(b, t, h * dK)

    def forward(self, x, mask=None, returnAttention=False):
        Q = self.splitHeads(self.Wq(x))
        K = self.splitHeads(self.Wk(x))
        V = self.splitHeads(self.Wv(x))
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(self.dK, dtype=torch.float32, device=x.device)
        )
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        context = torch.matmul(attn, V)
        out = self.Wo(self.combineHeads(context))
        if returnAttention:
            return out, attn
        return out


class FeedForward(nn.Module):
    def __init__(self, dModel, dFF, dropout=0.1):
        super().__init__()
        self.lin1 = nn.Linear(dModel, dFF)
        self.lin2 = nn.Linear(dFF, dModel)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.lin2(self.dropout(F.relu(self.lin1(x))))


class TransformerBlock(nn.Module):
    def __init__(self, dModel, numHeads, dFF, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadSelfAttention(dModel, numHeads, dropout)
        self.ff = FeedForward(dModel, dFF, dropout)
        self.norm1 = nn.LayerNorm(dModel)
        self.norm2 = nn.LayerNorm(dModel)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None, returnAttention=False):
        if returnAttention:
            a, w = self.attn(self.norm1(x), mask, returnAttention=True)
            x = x + self.dropout(a)
            x = x + self.dropout(self.ff(self.norm2(x)))
            return x, w
        else:
            a = self.attn(self.norm1(x), mask)
            x = x + self.dropout(a)
            x = x + self.dropout(self.ff(self.norm2(x)))
            return x


class WCSTTransformer(nn.Module):
    def __init__(self, vocabSize=71, dModel=128, numHeads=4, numLayers=4, dFF=512, maxSeqLen=512, dropout=0.1):
        super().__init__()
        self.dModel = dModel
        self.tokenEmbedding = nn.Embedding(vocabSize, dModel, padding_idx=PAD_ID)
        self.positionEmbedding = nn.Embedding(maxSeqLen, dModel)
        self.segment_embedding = nn.Embedding(2, dModel)
        self.blocks = nn.ModuleList([TransformerBlock(dModel, numHeads, dFF, dropout) for _ in range(numLayers)])
        self.dropout = nn.Dropout(dropout)

        self.outputProjection = nn.Linear(dModel, vocabSize)
        self.choice_head = nn.Sequential(
            nn.Linear(dModel, dModel),
            nn.ReLU(),
            nn.Linear(dModel, 4)
        )

    def forward(self, x, token_type=None, returnAttention=False):
        batchSize, seqLen = x.size()

        mask = torch.tril(torch.ones(seqLen, seqLen, device=x.device, dtype=torch.bool)).unsqueeze(0).unsqueeze(0)
        keys_with_no_padding = (x != PAD_ID).unsqueeze(1).unsqueeze(2)
        mask = (mask & keys_with_no_padding).expand(batchSize, self.blocks[0].attn.numHeads, seqLen, seqLen)

        positions = torch.arange(seqLen, device=x.device).unsqueeze(0).expand(batchSize, -1)
        tok = self.tokenEmbedding(x)
        pos = self.positionEmbedding(positions)
        seg = self.segment_embedding(token_type) if token_type is not None else 0
        x = self.dropout(tok + pos + seg)

        attentionWeightsList = []
        for block in self.blocks:
            if returnAttention:
                x, attnWeights = block(x, mask, returnAttention=True)
                attentionWeightsList.append(attnWeights)
            else:
                x = block(x, mask)

        logits = self.outputProjection(x)
        choice_logits = self.choice_head(x)
        if returnAttention:
            return logits, choice_logits, x, attentionWeightsList
        return logits, choice_logits, x


# -----------------------------
# Dataset
# -----------------------------
class WCSTDataset(Dataset):
    def __init__(self, wcst, wcstGen, numBatches, switch_period=None, supervise="all"):
        self.data = []
        self.supervise=supervise
        steps = 0
        for _ in range(numBatches):
            if switch_period is not None and steps > 0 and (steps % switch_period) == 0:
                wcst.context_switch()
                print(f"[Dataset] Context switched after {steps} batches.")
            batchInputs, batchTargets = next(wcstGen)
            steps += 1
            for inp, tgt in zip(batchInputs, batchTargets):
                full = np.concatenate([inp, tgt])
                inputSeq = full[:-1]
                targetSeq = full[1:]

                targetMask = np.full_like(targetSeq, -100)
                sep_positions = np.where(inputSeq == SEP_ID)[0]
                def is_class_token(v): 
                    return v in (C1, C2, C3, C4)

                if self.supervise == "all":
                    for p in sep_positions:
                        nxt = targetSeq[p]
                        if is_class_token(nxt):
                            targetMask[p] = nxt

                elif self.supervise in ("last", "query"):
                    if len(sep_positions) > 0:
                        p = sep_positions[-1]  # last SEP ~ query boundary in this dataset
                        nxt = targetSeq[p]
                        if is_class_token(nxt):
                            targetMask[p] = nxt

                self.data.append({
                    "input": torch.tensor(inputSeq, dtype=torch.long),
                    "target": torch.tensor(targetMask, dtype=torch.long),
                })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        d = self.data[idx]
        return d["input"], d["target"]


def collateFn(batch):
    inputs, targets = zip(*batch)
    maxLen = max(len(x) for x in inputs)

    B = len(batch)
    batchInputs = torch.full((B, maxLen), PAD_ID, dtype=torch.long)
    batchTargets = torch.full((B, maxLen), -100, dtype=torch.long)

    for i, (inp, tgt) in enumerate(zip(inputs, targets)):
        L = len(inp)
        batchInputs[i, :L] = torch.as_tensor(inp, dtype=torch.long)
        batchTargets[i, :L] = torch.as_tensor(tgt, dtype=torch.long)

    # token_type: 0=examples, 1=query/answer
    token_type_ids = torch.zeros((B, maxLen), dtype=torch.long)
    for i in range(B):
        seq = batchInputs[i]
        valid = (seq != PAD_ID).nonzero(as_tuple=True)[0]
        if len(valid) == 0:
            continue
        last_t = valid[-1].item()
        seps = (seq[: last_t + 1] == SEP_ID).nonzero(as_tuple=True)[0]
        last_sep = seps[-1].item() if len(seps) > 0 else 0
        token_type_ids[i, last_sep:] = 1

    return batchInputs, batchTargets, token_type_ids


# -----------------------------
# Train / Eval
# -----------------------------
def trainEpoch(model, dataloader, optimizer, device, class_weights_4, maxGradNorm=1.0, warmup_steps=500, base_lr=3e-4, global_step_start=0):
    model.train()
    totalLoss = 0.0
    totalCorrect = 0
    totalTokens = 0
    global_step = global_step_start

    for batch in tqdm(dataloader, desc="Training"):
        if len(batch) == 3:
            inputs, targets, ttids = batch
        else:
            inputs, targets = batch
            ttids = torch.zeros_like(inputs)

        inputs, targets, ttids = inputs.to(device), targets.to(device), ttids.to(device)
        assert (targets != -100).sum().item() >= inputs.size(0), "Expected at least 1 supervised label per seq"

        optimizer.zero_grad()
        out = model(inputs, token_type=ttids)
        if   len(out) == 3:  logits, choice_logits, h = out
        elif len(out) == 4:  logits, choice_logits, h, _ = out
        else:                raise RuntimeError("Unexpected model outputs")

        all_logits = gather_sep_logits(h, ttids)
        logits_sel, targets_sel, _ = select_masked_logits(all_logits, targets)

        loss = F.cross_entropy(logits_sel, targets_sel, label_smoothing=0.0, weight=class_weights_4)
        preds = logits_sel.argmax(dim=-1)
        acc_step = (preds == targets_sel).float().mean().item()
        n_step = len(targets_sel)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), maxGradNorm)

        global_step += 1
        if global_step <= warmup_steps:
            warm_lr = base_lr * (global_step / float(warmup_steps))
            for g in optimizer.param_groups:
                g["lr"] = warm_lr
        optimizer.step()

        totalCorrect += acc_step * n_step
        totalTokens += n_step
        totalLoss += loss.item()

    return totalLoss / len(dataloader), totalCorrect / max(totalTokens, 1), global_step


def evaluate(model, dataloader, device, class_weights_4):
    model.eval()
    totalLoss = 0.0
    totalCorrect = 0
    totalTokens = 0

    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 3:
                inputs, targets, ttids = batch
            else:
                inputs, targets = batch
                ttids = torch.zeros_like(inputs)

            inputs, targets, ttids = inputs.to(device), targets.to(device), ttids.to(device)
            out = model(inputs, token_type=ttids)
            if   len(out) == 3:  logits, choice_logits, h = out
            elif len(out) == 4:  logits, choice_logits, h, _ = out
            else:                raise RuntimeError("Unexpected model outputs")

            all_logits = gather_sep_logits(h, ttids)
            logits_sel, targets_sel, _ = select_masked_logits(all_logits, targets)

            loss = F.cross_entropy(logits_sel, targets_sel, label_smoothing=0.0, weight=class_weights_4)
            preds = logits_sel.argmax(dim=-1)
            acc_step = (preds == targets_sel).float().mean().item()
            n_step = len(targets_sel)

            totalCorrect += acc_step * n_step
            totalTokens += n_step
            totalLoss += loss.item()

    return totalLoss / len(dataloader), totalCorrect / max(totalTokens, 1)


# -----------------------------
# Logging
# -----------------------------
class RunLogger:
    def __init__(self, tag: str, config: dict):
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.dir = Path("runs") / f"{ts}_{tag}"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.history = {"train": [], "val": []}
        (self.dir / "config.json").write_text(json.dumps(config, indent=2))

    def log_epoch(self, epoch, train_loss, train_acc, val_loss, val_acc, lr):
        rec = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "train_acc": float(train_acc),
            "val_loss": float(val_loss),
            "val_acc": float(val_acc),
            "lr": float(lr) if lr is not None else None,
        }
        with open(self.dir / "epoch_log.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        self.history["train"].append({"epoch": epoch, "loss": float(train_loss), "acc": float(train_acc)})
        self.history["val"].append({"epoch": epoch, "loss": float(val_loss), "acc": float(val_acc)})

    def finalize(self, test_loss, test_acc, label_counts, confusion_matrix):
        final = {
            "test_loss": float(test_loss),
            "test_acc": float(test_acc),
            "label_counts": {str(k): int(v) for k, v in label_counts.items()},
            "confusion": confusion_matrix,
        }
        (self.dir / "metrics.json").write_text(json.dumps(final, indent=2))
        (self.dir / "history.json").write_text(json.dumps(self.history, indent=2))

    def save_model(self, model_path: str):
        dst = self.dir / "model.pt"
        with open(model_path, "rb") as src, open(dst, "wb") as out:
            out.write(src.read())

    def append_global_csv(self, tag, test_loss, test_acc):
        row = [datetime.now().isoformat(timespec="seconds"), tag, f"{test_loss:.4f}", f"{test_acc:.4f}"]
        csv_path = Path("results.csv")
        new_file = not csv_path.exists()
        with open(csv_path, "a", newline="") as f:
            w = csv.writer(f)
            if new_file:
                w.writerow(["timestamp", "tag", "test_loss", "test_acc"])
            w.writerow(row)


# -----------------------------
# Utility analysis (optional)
# -----------------------------
def count_supervised_targets(dataloader):
    cnt = Counter({0: 0, 1: 0, 2: 0, 3: 0})
    for batch in dataloader:
        t = batch[1]
        lab = t[t != -100].flatten()
        for z in (0, 1, 2, 3):
            cnt[z] += int((lab == (64 + z)).sum().item())
    print("Supervised target counts (val):", dict(cnt))


def prediction_histogram_4way(model, dataloader, device):
    model.eval()
    ph = Counter({0: 0, 1: 0, 2: 0, 3: 0})
    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 3:
                x, t, tt = batch
            else:
                x, t = batch
                tt = torch.zeros_like(x)
            x, t, tt = x.to(device), t.to(device), tt.to(device)
            out = model(x, token_type=tt)
            if   len(out) == 3:  _, _, h = out
            elif len(out) == 4:  _, _, h, _ = out
            else:                raise RuntimeError("Unexpected model outputs")

            all_logits = gather_sep_logits(h, tt)
            logits_sel, targets_sel, _ = select_masked_logits(all_logits, t)
            preds4 = logits_sel.argmax(dim=-1).cpu().tolist()
            for p in preds4:
                ph[p] += 1
    print("Pred histogram (val) 0:C1 1:C2 2:C3 3:C4:", dict(ph))


def confusion_matrix_4way(model, dataloader, device):
    model.eval()
    cm = torch.zeros(4, 4, dtype=torch.long)
    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 3:
                inputs, targets, ttids = batch
            else:
                inputs, targets = batch
                ttids = torch.zeros_like(inputs)
            inputs, targets, ttids = inputs.to(device), targets.to(device), ttids.to(device)

            out = model(inputs, token_type=ttids)
            if   len(out) == 3:  _, _, h = out
            elif len(out) == 4:  _, _, h, _ = out
            else:                raise RuntimeError("Unexpected model outputs")

            all_logits = gather_sep_logits(h, ttids)
            logits_sel, targets_sel, _ = select_masked_logits(all_logits, targets)
            preds4 = logits_sel.argmax(dim=-1).cpu().tolist()
            trues4 = targets_sel.cpu().tolist()
            for t, p in zip(trues4, preds4):
                cm[t, p] += 1
    return cm.tolist()


def inspect_one(wcst, gen):
    batchInputs, batchTargets = next(gen)
    inp, tgt = batchInputs[0], batchTargets[0]
    full = np.concatenate([inp, tgt])
    inputSeq = full[:-1]
    targetSeq = full[1:]
    seps = np.where(inputSeq == SEP_ID)[0]
    print("inputSeq:", inputSeq.tolist())
    print("targetSeq:", targetSeq.tolist())
    print("SEP positions in inputSeq:", seps.tolist())
    for p in seps:
        print(f" at SEP idx {p}, targetSeq[p] = {targetSeq[p]}")


def scan_vocab_usage(dataloader, limit=2000):
    seen = Counter()
    k = 0
    for batch in dataloader:
        x = batch[0]
        vals = x.flatten().tolist()
        for v in vals:
            if v in (64, 65, 66, 67):
                seen[v] += 1
        k += 1
        if k >= limit:
            break
    print("Occurrences of 64..67 in *inputs*:", dict(seen))


# -----------------------------
# Argparse & main
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--switch_period", type=str, default="64")  # "none" or int
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--train_batches", type=int, default=2000)
    p.add_argument("--val_batches", type=int, default=300)
    p.add_argument("--test_batches", type=int, default=300)
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--num_layers", type=int, default=4)
    p.add_argument("--num_heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--supervise", type=str,default="all", choices =["all","last","query"])
    return p.parse_args()


def main():
    from wcst import WCST  # your generator
    args = parse_args()

    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    sp = None if str(args.switch_period).lower() == "none" else int(args.switch_period)

    # Data
    batch_size = args.batch_size
    wcst_train = WCST(batch_size=batch_size)
    wcst_val = WCST(batch_size=batch_size + 1)
    wcst_test = WCST(batch_size=batch_size + 2)
    trainGen = wcst_train.gen_batch()
    valGen = wcst_val.gen_batch()
    testGen = wcst_test.gen_batch()

    trainDataset = WCSTDataset(wcst_train, trainGen, numBatches=args.train_batches, switch_period=sp, supervise=args.supervise)
    valDataset = WCSTDataset(wcst_val, valGen, numBatches=args.val_batches, supervise=args.supervise)
    testDataset = WCSTDataset(wcst_test, testGen, numBatches=args.test_batches, supervise=args.supervise)

    trainLoader = DataLoader(trainDataset, batch_size=min(32, batch_size*4), shuffle=True, collate_fn=collateFn)
    valLoader = DataLoader(valDataset, batch_size=32, shuffle=False, collate_fn=collateFn)
    testLoader = DataLoader(testDataset, batch_size=32, shuffle=False, collate_fn=collateFn)

    # Model
    model = WCSTTransformer(
        vocabSize=71,
        dModel=args.d_model,
        numHeads=args.num_heads,
        numLayers=args.num_layers,
        dFF=4 * args.d_model,
        dropout=args.dropout,
    ).to(device)

    with torch.no_grad():
        model.outputProjection.bias[64:68] = torch.log(torch.tensor([0.25, 0.25, 0.25, 0.25], device=device))
        for m in model.choice_head:
            if isinstance(m, nn.Linear):
                nn.init.zeros_((m.bias))
        nn.init.xavier_uniform_(model.choice_head[-1].weight, gain=0.1)

    # Logging config & tag
    config = {
        "seed": 42,
        "device": str(device),
        "batch_size": batch_size,
        "epochs": args.epochs,
        "model": {
            "vocabSize": 71,
            "dModel": args.d_model,
            "numHeads": args.num_heads,
            "numLayers": args.num_layers,
            "dFF": 4 * args.d_model,
            "dropout": args.dropout,
        },
        "optimizer": {"type": "AdamW", "lr": args.lr, "betas": [0.9, 0.95], "weight_decay": 0.01},
        "loss": {"type": "CrossEntropy", "ignore_index": -100, "label_smoothing": 0.0},
        "data": {
            "switch_period": sp,
            "train_batches": args.train_batches,
            "val_batches": args.val_batches,
            "test_batches": args.test_batches,
        },
        "pad_id": PAD_ID,
        "sep_id": SEP_ID,
        "class_ids": [C1, C2, C3, C4],
    }
    tag = f"sp{'none' if sp is None else sp}-d{args.d_model}-L{args.num_layers}-H{args.num_heads}"
    logger = RunLogger(tag, config)

    # class balance (for info)
    cnt = Counter()
    for batch in valLoader:
        _, t, *_ = batch
        lab = t[t != -100].flatten()
        for c in (64, 65, 66, 67):
            cnt[c] += (lab == c).sum().item()
    print("Label counts (val C1..C4):", dict(cnt))

    # Optimizer (decay / no-decay)
    pointer_params, other_decay, other_nodecay = [], [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "choice_head" in n:
            pointer_params.append(p)
        elif no_decay(n):
            other_nodecay.append(p)
        else:
            other_decay.append(p)

    optimizer = torch.optim.AdamW(
        [
            {"params": other_decay, "weight_decay": 0.01},
            {"params": other_nodecay, "weight_decay": 0.0},
            {"params": pointer_params, "weight_decay": 0.0},
        ],
        lr=args.lr,
        betas=(0.9, 0.95),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=1)

    bestValLoss = float("inf")
    history = {"trainLoss": [], "trainAcc": [], "valLoss": [], "valAcc": []}
    global_step = 0
    patience = 3
    bad = 0

    # Main training
    for epoch in range(args.epochs):
        # rebuild train stream every epoch (fresh rules and samples)
        wcst_train = WCST(batch_size=batch_size)
        trainGen = wcst_train.gen_batch()
        trainDataset = WCSTDataset(wcst_train, trainGen, numBatches=args.train_batches, switch_period=sp, supervise=args.supervise)
        trainLoader = DataLoader(trainDataset, batch_size=min(32, batch_size*4), shuffle=True, collate_fn=collateFn)

        weights4 = compute_class_weights_4(trainLoader).to(device)

        trainLoss, trainAcc, global_step = trainEpoch(
            model, trainLoader, optimizer, device, class_weights_4=weights4,
            maxGradNorm=1.0, warmup_steps=500, base_lr=args.lr, global_step_start=global_step
        )
        valLoss, valAcc = evaluate(model, valLoader, device, class_weights_4=weights4)

        print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {trainLoss:.4f} | Train Acc: {trainAcc:.4f} | Val Loss: {valLoss:.4f} | Val Acc: {valAcc:.4f}")

        logger.log_epoch(epoch=epoch + 1, train_loss=trainLoss, train_acc=trainAcc, val_loss=valLoss, val_acc=valAcc, lr=current_lr(optimizer))
        scheduler.step(valLoss)

        if valLoss < bestValLoss - 1e-4:
            bestValLoss = valLoss
            bad = 0
            torch.save(model.state_dict(), "model.pt")
        else:
            bad += 1
            if bad >= patience:
                print("Early stopping triggered.")
                break

        history["trainLoss"].append(trainLoss)
        history["trainAcc"].append(trainAcc)
        history["valLoss"].append(valLoss)
        history["valAcc"].append(valAcc)

    # Final eval
    model.load_state_dict(torch.load("model.pt", map_location=device))
    prediction_histogram_4way(model, valLoader, device)
    testLoss, testAcc = evaluate(model, testLoader, device, class_weights_4=weights4)
    print(f"\nTest Loss: {testLoss:.4f} | Test Acc: {testAcc:.4f}")

    cm = confusion_matrix_4way(model, testLoader, device)
    logger.finalize(testLoss, testAcc, cnt, cm)
    print("\nConfusion matrix (rows=true C1..C4, cols=pred C1..C4):")
    for row in cm:
        print(row)

    with open(logger.dir / "confusion.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["", "pred_C1", "pred_C2", "pred_C3", "pred_C4"])
        for i, row in enumerate(cm, start=1):
            w.writerow([f"true_C{i}"] + row)

    logger.save_model("model.pt")
    logger.append_global_csv(tag, testLoss, testAcc)
    save_confusion_png(cm, logger.dir / "confusion.png", normalize=True, title=f"Confusion (rows=true, cols=pred) — {tag}")

    with open("trainingHistory.json", "w") as f:
        json.dump(history, f)

    return model, history


if __name__ == "__main__":
    main()
