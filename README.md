# Transformer-Based WCST: Context Switching & Supervision Experiments

This project implements a compact Transformer trained on a **synthetic Wisconsin Card Sorting Test (WCST)** task to study **context adaptation, supervision granularity, and inductive flexibility** in rule-changing environments.

Rather than optimizing for raw predictive performance, the goal is to understand **how and when Transformer models begin to exhibit adaptive behaviour** under controlled distribution shifts.

This work was completed as part of an advanced NLP project and is research-oriented in nature.

---

## 🧠 Motivation

The WCST is a classical cognitive task used to measure **cognitive flexibility** in humans — the ability to infer and adapt to changing rules based on feedback.

Recent large language models exhibit *in-context learning*, where rule inference emerges without explicit parameter updates.  
This project explores whether similar behaviour can arise in a **lightweight, scratch-built Transformer**, and under what training conditions.

Key questions explored:
- Does periodic context switching improve generalization?
- How does supervision granularity affect stability?
- Does increasing model capacity help or hurt in small reasoning tasks?

---

## 🧪 Experimental Overview

Three Transformer variants were developed and compared:

- **Model 1**: No context switching (stationary rule)
- **Model 2**: Context switching every fixed number of batches
- **Main Model (Model 3)**: Refined architecture with:
  - Pointer-style 4-way classification head
  - Supervision at all separator (SEP) positions
  - Class-balanced loss and stabilized initialization

Experiments include:
- Context switching frequency sweeps
- Supervision ablations
- Model capacity scaling
- Training stream size analysis

---

## 📊 Key Findings (High-Level)

- Context switching acts as an implicit curriculum signal and improves robustness
- Supervision at intermediate SEP positions significantly stabilizes learning
- Increasing model size degrades performance in this regime (over-parameterization)
- Performance remains sensitive to initialization and data scale

Absolute accuracy remains modest (~30–75% depending on regime), but **learning dynamics and failure modes are interpretable and consistent**.

---

## ⚠️ Limitations

- Synthetic task may not fully capture human cognitive flexibility
- Training is sensitive to random initialization
- Performance variance remains high across runs
- Results emphasize *behavioural trends*, not benchmark dominance

These limitations are discussed explicitly in the accompanying report.

---

## 👥 Collaboration

This project was completed as part of a group assignment.

My contributions included:
- Transformer architecture design and implementation
- Training pipeline and supervision masking
- Experimental design and ablation studies
- Result analysis and report writing

---

## 🔁 Reproducibility & Execution

**Note:** Due to stochastic data generation and random initialization, exact numerical results may vary between runs.  
Trends, however, remain consistent.

### Quick Start (Main Model)
```bash
python main.py --switch_period 64 --supervise all

### Key Arguments
Argument	Description
--switch_period	Context switch frequency (none, 32, 64, 128)
--supervise	SEP supervision scope (all, last, query)
--train_batches	Training batch count
--d_model	Model hidden size
--num_layers	Transformer depth
--num_heads	Attention heads
--epochs	Training epochs
