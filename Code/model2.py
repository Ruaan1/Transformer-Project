import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import builtins
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from datetime import datetime
from pathlib import Path
import csv
import matplotlib.pyplot as plt
import numpy as np

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--switch", type=int, default=None, help="switch_period for training. Use None for no switching.")
parser.add_argument("--dmodel", type=int, default=128, help="Transformer hidden size")
parser.add_argument("--heads", type=int, default=4, help="Number of attention heads")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
args = parser.parse_args()

torch.manual_seed(args.seed); np.random.seed(args.seed)
assert args.dmodel % args.heads == 0, "dmodel must be divisible by heads"

PAD_ID = 70
SEP_ID = 68
C1,C2,C3,C4 = 64,65,66,67

def current_lr(optimizer):
    for g in optimizer.param_groups:
        return g.get('lr', None)
    
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
            txt = f"{cm[i,j]*100:.1f}%" if normalize else f"{int(cm[i,j])}"
            plt.text(j, i, txt, ha="center", va="center")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

# ============================================================================
# TRANSFORMER COMPONENTS
# ============================================================================

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
        batchSize, seqLen, _ = x.size()
        return x.view(batchSize, seqLen, self.numHeads, self.dK).transpose(1, 2)

    def combineHeads(self, x):
        batchSize, numHeads, seqLen, dK = x.size()
        return x.transpose(1, 2).contiguous().view(batchSize, seqLen, self.dModel)

    def forward(self, x, mask=None, returnAttention=False):
        Q = self.splitHeads(self.Wq(x))
        K = self.splitHeads(self.Wk(x))
        V = self.splitHeads(self.Wv(x))
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(self.dK, dtype=torch.float32, device=x.device)
        )
        if mask is not None:
            scores = scores.masked_fill(~mask, float('-inf'))
        attentionWeights = F.softmax(scores, dim=-1)
        attentionWeights = self.dropout(attentionWeights)
        context = torch.matmul(attentionWeights, V)
        output = self.Wo(self.combineHeads(context))
        if returnAttention:
            return output, attentionWeights
        return output


class FeedForward(nn.Module):
    def __init__(self, dModel, dFF, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(dModel, dFF)
        self.linear2 = nn.Linear(dFF, dModel)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class TransformerBlock(nn.Module):
    def __init__(self, dModel, numHeads, dFF, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadSelfAttention(dModel, numHeads, dropout)
        self.feedForward = FeedForward(dModel, dFF, dropout)
        self.norm1 = nn.LayerNorm(dModel)
        self.norm2 = nn.LayerNorm(dModel)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None, returnAttention=False):
        if returnAttention:
            attnOutput, attnWeights = self.attention(self.norm1(x), mask, returnAttention=True)
            x = x + self.dropout(attnOutput)
            x = x + self.dropout(self.feedForward(self.norm2(x)))
            return x, attnWeights
        else:
            attnOutput = self.attention(self.norm1(x), mask)
            x = x + self.dropout(attnOutput)
            x = x + self.dropout(self.feedForward(self.norm2(x)))
            return x


class WCSTTransformer(nn.Module):
    def __init__(self, vocabSize=71, dModel=128, numHeads=4, numLayers=4, dFF=512, maxSeqLen=512, dropout=0.1):
        super().__init__()
        self.dModel = dModel
        self.tokenEmbedding = nn.Embedding(vocabSize, dModel, padding_idx=PAD_ID)
        self.positionEmbedding = nn.Embedding(maxSeqLen, dModel)
        self.segment_embedding = nn.Embedding(2, dModel)
        self.blocks = nn.ModuleList([
            TransformerBlock(dModel, numHeads, dFF, dropout) for _ in range(numLayers)
        ])
        self.outputProjection = nn.Linear(dModel, vocabSize)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, token_type=None, returnAttention=False):
        batchSize, seqLen = x.size()
        mask = torch.tril(torch.ones(seqLen, seqLen, device=x.device, dtype = torch.bool)).unsqueeze(0).unsqueeze(0)
        keys_with_no_padding = (x != PAD_ID).unsqueeze(1).unsqueeze(2)
        combination = mask & keys_with_no_padding
        mask = combination.expand(batchSize, self.blocks[0].attention.numHeads, seqLen, seqLen)
        
        positions = torch.arange(seqLen, device=x.device).unsqueeze(0).expand(batchSize, -1)
        tok = self.tokenEmbedding(x) 
        pos = self.positionEmbedding(positions)
        if token_type is None:
            seg = 0
        else:
            seg = self.segment_embedding(token_type)
        x = self.dropout(tok + pos + seg)

        attentionWeightsList = []
        for block in self.blocks:
            if returnAttention:
                x, attnWeights = block(x, mask, returnAttention=True)
                attentionWeightsList.append(attnWeights)
            else:
                x = block(x, mask)

        logits = self.outputProjection(x)
        if returnAttention:
            return logits, attentionWeightsList
        return logits


# ============================================================================
# WCST DATASET AND COLLATE FUNCTION
# ============================================================================

class WCSTDataset(Dataset):
    def __init__(self, wcst, wcstGen, numBatches, switch_period=None):
        self.data = []
        self.inputs = []
        self.targets = []
        self.lengths = []
        steps = 0
        for _ in range(numBatches):
            if switch_period is not None and steps > 0 and (steps % switch_period) ==0:
                wcst.context_switch()
                print(f"[Dataset] Context switched after {steps} batches.")
                

            batchInputs, batchTargets = builtins.next(wcstGen)
            steps += 1
            for inp, tgt in zip(batchInputs, batchTargets):
                # inp contains: [4 category cards, 1 example card, SEP, example_label, EOS]
                # tgt contains: [question card, SEP, question_label]
                
                # Full sequence: inp + tgt
                fullSeq = np.concatenate([inp, tgt])
                
                # Create input (all tokens except the last one for next-token prediction)
                inputSeq = fullSeq[:-1]
                
                # Create target (shifted by 1, predicting next token)
                targetSeq = fullSeq[1:]
                
                targetMask = np.full_like(targetSeq, -100)
                
                sep_position = np.where(inputSeq == SEP_ID)[0]
                if len(sep_position) > 0:
                    p = sep_position[-1]
                    next_value = targetSeq[p]
                    if next_value in (C1,C2,C3,C4):
                        targetMask[p] = next_value
                
                self.data.append({
                    'input': torch.tensor(inputSeq, dtype=torch.long),
                    'target': torch.tensor(targetMask, dtype=torch.long)
                })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]['input'], self.data[idx]['target']


def collateFn(batch):
    inputs, targets = zip(*batch)
    maxLen = max(len(x) for x in inputs)

    batchInputs  = torch.full((len(batch), maxLen), PAD_ID, dtype=torch.long)
    batchTargets = torch.full((len(batch), maxLen), -100,   dtype=torch.long)

    for i, (inp, tgt) in enumerate(zip(inputs, targets)):
        L = len(inp)
        batchInputs[i, :L]  = torch.as_tensor(inp, dtype=torch.long)
        batchTargets[i, :L] = torch.as_tensor(tgt, dtype=torch.long)

    # build token_type (0=examples, 1=query/answer block)
    B, T = batchInputs.shape
    token_type_ids = torch.zeros((B, T), dtype=torch.long)
    for i in range(B):
        seq = batchInputs[i]
        valid = (seq != PAD_ID).nonzero(as_tuple=True)[0]
        if len(valid) == 0:
            continue
        last_t = valid[-1].item()
        seps = (seq[:last_t+1] == SEP_ID).nonzero(as_tuple=True)[0]
        last_sep = seps[-1].item() if len(seps) > 0 else 0
        token_type_ids[i, last_sep:] = 1   # mark query/answer segment

    return batchInputs, batchTargets, token_type_ids


# ============================================================================
# TRAINING AND EVALUATION
# ============================================================================

def trainEpoch(model, dataloader, optimizer, device, maxGradNorm=1.0, warmup_steps=500, base_lr=3e-4, global_step_start=0):
    model.train()
    totalLoss = 0.0
    totalCorrect = 0
    totalTokens = 0
    global_step = global_step_start
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=0.05)

    for batch in tqdm(dataloader, desc="Training"):
        if len(batch) ==3:
            inputs, targets, ttids = batch
        else:
            inputs, targets = batch
            ttids = torch.zeros_like(inputs)

        inputs, targets, ttids = inputs.to(device), targets.to(device), ttids.to(device)

        assert (targets != -100).sum().item() == inputs.size(0), "Expected exactly 1 supervised label per seq"

        optimizer.zero_grad()
        logits = model(inputs, token_type=ttids)

        loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        loss.backward()

        # clip
        torch.nn.utils.clip_grad_norm_(model.parameters(), maxGradNorm)

        global_step += 1
        if global_step <= warmup_steps:
            warm_lr = base_lr * (global_step / float(warmup_steps))
            for g in optimizer.param_groups:
                g['lr'] = warm_lr

        optimizer.step()

        # accuracy on the single supervised position per seq
        logitsFlat = logits.reshape(-1, logits.size(-1))
        targetsFlat = targets.reshape(-1)
        mask = targetsFlat != -100
        totalCorrect += (logitsFlat.argmax(dim=-1)[mask] == targetsFlat[mask]).sum().item()
        totalTokens += mask.sum().item()
        totalLoss += loss.item()

    return totalLoss / len(dataloader), totalCorrect / totalTokens, global_step


def evaluate(model, dataloader, device):
    model.eval()
    totalLoss = 0
    totalCorrect = 0
    totalTokens = 0
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=0.05)

    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 3:
                inputs, targets, ttids = batch
            else:
                inputs, targets = batch
                ttids = torch.zeros_like(inputs)

            inputs, targets, ttids = inputs.to(device), targets.to(device), ttids.to(device)
            logits = model(inputs, token_type =ttids)
            
            loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            logitsFlat = logits.reshape(-1, logits.size(-1))
            targetsFlat = targets.reshape(-1)

            mask = targetsFlat != -100
            totalCorrect += (logitsFlat.argmax(dim=-1)[mask] == targetsFlat[mask]).sum().item()
            totalTokens += mask.sum().item()
            totalLoss += loss.item()

    return totalLoss / len(dataloader), totalCorrect / totalTokens


# ============================================================================
# Saved Experiments
# ============================================================================

class RunLogger:
    def __init__(self, tag: str, config: dict):
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.dir = Path("runs") / f"{ts}_{tag}"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.history = {"train": [], "val": []}
        # pretty config dump
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
        # jsonlines for easy streaming
        with open(self.dir / "epoch_log.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")

        self.history["train"].append({"epoch": epoch, "loss": float(train_loss), "acc": float(train_acc)})
        self.history["val"].append({"epoch": epoch, "loss": float(val_loss), "acc": float(val_acc)})

    def finalize(self, test_loss, test_acc, label_counts, confusion_matrix):
        final = {
            "test_loss": float(test_loss),
            "test_acc": float(test_acc),
            "label_counts": {str(k): int(v) for k, v in label_counts.items()},
            "confusion": confusion_matrix,  # 4x4 list of ints
        }
        (self.dir / "metrics.json").write_text(json.dumps(final, indent=2))
        (self.dir / "history.json").write_text(json.dumps(self.history, indent=2))

    def save_model(self, model_path: str):
        # copy best model file into this run dir
        dst = self.dir / "model1.pt"
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



# ============================================================================
# MAIN SCRIPT
# ============================================================================

def confusion_matrix_4way(model, dataloader, device):
    model.eval()
    cm = torch.zeros(4, 4, dtype=torch.long)  # rows=true, cols=pred
    for batch in dataloader:
        if len(batch) == 3:
            inputs, targets, ttids = batch
        else:
            inputs, targets = batch
            ttids = torch.zeros_like(inputs)

        inputs, targets, ttids = inputs.to(device), targets.to(device), ttids.to(device)
        logits = model(inputs, token_type=ttids)
        logitsFlat = logits.reshape(-1, logits.size(-1))
        targetsFlat = targets.reshape(-1)

        idxs = (targetsFlat != -100).nonzero(as_tuple=True)[0]
        if idxs.numel() == 0:
            continue
        preds = logitsFlat[idxs].argmax(dim=-1)
        trues = targetsFlat[idxs]

        for t, p in zip(trues.tolist(), preds.tolist()):
            if t in (64, 65, 66, 67):
                ti = t - 64
                pi = p - 64 if p in (64, 65, 66, 67) else None
                if pi is not None and 0 <= pi < 4:
                    cm[ti, pi] += 1
    return cm.tolist()

def main():
    import numpy as np
    from wcst import WCST
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    batch_size = 8
    wcst_train = WCST(batch_size=batch_size)
    wcst_val = WCST(batch_size=batch_size+1)
    wcst_test = WCST(batch_size=batch_size+2)
    trainGen = wcst_train.gen_batch()
    valGen = wcst_val.gen_batch()
    testGen = wcst_test.gen_batch()

    trainDataset = WCSTDataset(wcst_train, trainGen, numBatches=2000, switch_period = switch_period)
    valDataset = WCSTDataset(wcst_val, valGen, numBatches=300)     
    testDataset = WCSTDataset(wcst_test, testGen, numBatches=300) 

    trainLoader = DataLoader(trainDataset, batch_size=32, shuffle=True, collate_fn=collateFn)
    valLoader = DataLoader(valDataset, batch_size=32, shuffle=False, collate_fn=collateFn)
    testLoader = DataLoader(testDataset, batch_size=32, shuffle=False, collate_fn=collateFn)

    switch_period = args.switch
    numEpochs = 10
    config = {
        "seed": args.seed,
        "device": str(device),
        "batch_size": batch_size,
        "numEpochs": numEpochs,
        "model": {"vocabSize": 71, "dModel": args.dmodel, "numHeads": args.heads, "numLayers": 4, "dFF": 512, "dropout": 0.1},
        "optimizer": {"type": "AdamW", "lr": 3e-4, "betas": [0.9, 0.95], "weight_decay": 0.1},
        "loss": {"type": "CrossEntropy", "ignore_index": -100, "label_smoothing": 0.05},
        "data": {"switch_period": switch_period, "train_batches": 2000, "val_batches": 300, "test_batches": 300},
        "pad_id": PAD_ID, "sep_id": SEP_ID, "class_ids": [C1, C2, C3, C4]
    }
    tag = f"switch_{'none' if switch_period is None else switch_period}-d{args.dmodel}-h{args.heads}-ep{numEpochs}-seed{args.seed}"
    logger = RunLogger(tag, config)


    print("Initializing model...")
    model = WCSTTransformer(vocabSize=71, dModel=args.dmodel, numHeads=args.head, numLayers=4, dFF=512, dropout=0.1).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1)
    scheduler = None

    numEpochs = 10
    bestValLoss = float('inf')
    history = {'trainLoss': [], 'trainAcc': [], 'valLoss': [], 'valAcc': []}

    from collections import Counter
    cnt = Counter()
    for batch in valLoader:  
        if len(batch) == 3:
            _, t, _ = batch
        else:
            _, t = batch
        lab = t[t != -100].flatten()
        for c in (64, 65, 66, 67):
            cnt[c] += (lab == c).sum().item()
    print("Label counts (C1..C4):", dict(cnt))

    patience = 3
    bad = 0
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=1
    )

    global_step = 0

    for epoch in range(numEpochs):
        wcst_train = WCST(batch_size=batch_size)
        trainGen   = wcst_train.gen_batch()
        trainDataset = WCSTDataset(wcst_train, trainGen, numBatches=2000, switch_period=64)
        trainLoader  = DataLoader(trainDataset, batch_size=32, shuffle=True, collate_fn=collateFn)

        trainLoss, trainAcc, global_step = trainEpoch(
            model, trainLoader, optimizer, device,
            maxGradNorm=1.0, warmup_steps=500, base_lr=3e-4, global_step_start=global_step
        )
        valLoss, valAcc = evaluate(model, valLoader, device)

        logger.log_epoch(
            epoch=epoch + 1,
            train_loss=trainLoss, train_acc=trainAcc,
            val_loss=valLoss, val_acc=valAcc,
            lr=current_lr(optimizer)
        )

        scheduler.step(valLoss)

        if valLoss < bestValLoss - 1e-4:
            bestValLoss = valLoss
            bad = 0
            torch.save(model.state_dict(), 'model2.pt')
        else:
            bad += 1
            if bad >= patience:
                print("Early stopping triggered.")
                break

        history['trainLoss'].append(trainLoss)
        history['trainAcc'].append(trainAcc)
        history['valLoss'].append(valLoss)
        history['valAcc'].append(valAcc)

        print(f"Epoch {epoch+1}/{numEpochs} | "
            f"Train Loss: {trainLoss:.4f} | Train Acc: {trainAcc:.4f} | "
            f"Val Loss: {valLoss:.4f} | Val Acc: {valAcc:.4f}")

    model.load_state_dict(torch.load('model2.pt'))
    testLoss, testAcc = evaluate(model, testLoader, device)
    print(f"\nTest Loss: {testLoss:.4f} | Test Acc: {testAcc:.4f}")

    # confusion matrix (on test)
    cm = confusion_matrix_4way(model, testLoader, device)

    # finalize logs
    logger.finalize(testLoss, testAcc, cnt, cm)  # 'cnt' is your label counts dict

    print("\nConfusion matrix (rows=true C1..C4, cols=pred C1..C4):")
    for row in cm:
        print(row)

    # also save a CSV for quick eyeballing
    import csv
    with open(logger.dir / "confusion.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["", "pred_C1", "pred_C2", "pred_C3", "pred_C4"])
        for i, row in enumerate(cm, start=1):
            w.writerow([f"true_C{i}"] + row)
        logger.save_model('model2.pt')
        logger.append_global_csv(tag, testLoss, testAcc)
        save_confusion_png(cm, logger.dir / "confusion.png", normalize=True,
                   title=f"Confusion (rows=true, cols=pred) — {tag}")

    # record where the dataset switches happened
    history["switch_points"] = list(range(64, 10000, 64))

    with open('trainingHistory.json', 'w') as f:
        json.dump(history, f)

    return model, history


if __name__ == "__main__":
    model, history = main()