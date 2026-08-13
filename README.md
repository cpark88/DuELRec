# DuELRec

**A Dual-Expert Strategy Integrating LLMs to Mitigate Negative Transfer in Cross-Domain Sequential Recommendation**

<p align="center">
  <img alt="Venue" src="https://img.shields.io/badge/CIKM%202026-Full%20Research%20Track-2E6BE6">
  <img alt="Task" src="https://img.shields.io/badge/Task-Cross--Domain%20Sequential%20Recommendation-555555">
  <img alt="Backbone" src="https://img.shields.io/badge/Backbone-LLM-orange">
</p>

> 🎉 **Accepted at CIKM 2026 (Full Research Track).** This repository provides the official implementation of DuELRec.

---

## 📌 Overview

**DuELRec** is an LLM-based **Cross-Domain Sequential Recommendation (CDSR)** model that **mitigates negative transfer** across domains while jointly exploiting **semantic** and **collaborative** signals.

CDSR predicts the next item a user will interact with, given historical interaction sequences that span multiple domains. Recent LLM-based recommenders (LLMRec) model item text as token-level sequences, but they largely **overlook item-level collaborative signals**, which leads to **semantic misalignment** and to performance degradation caused by negative transfer.

DuELRec addresses this with three components:

- **Domain-Gated Dual Experts** — a single-domain expert and a cross-domain expert, combined through a gating mechanism that suppresses harmful cross-domain interference.
- **Item-Aware Attention Transformation** — aggregates textual subtokens into item-level representations and applies **block-level attention masking**.
- **Dual-Sampling Token-to-Item Contrastive Learning** — captures collaborative signals from both single-domain and cross-domain views via stochastic negative sampling.

## 📊 Key Results

- Outperforms roughly 30 competitive baselines across **10 domains** drawn from **2 real-world datasets**.
- Deployed in a commercial personal-assistant application, delivering a **47.6% relative CTR improvement**.

---

## 🛠 Getting Started

All commands below are run from the `duelrec/` directory, since the training scripts resolve
`input_data/`, `token_mapping/` and `pretrained_tokenizer_*/` as paths relative to the working directory.

### 1. Installation

```bash
cd duelrec
pip install -r requirements.txt
```

### 2. Data download & placement

The dataset is too large to be hosted in this repository, so the **pre-extracted data** is provided via Google Drive.

- **Drive link**: https://drive.google.com/drive/folders/1tjBP1VSSDEYkSOckemZYfRXudyR9Jzqt?usp=share_link
- **Provided folders**: `input_data/`, `token_mapping/`

Download both folders and place them inside `duelrec/`:

```
DuELRec/
├── README.md
└── duelrec/
    ├── input_data/
    ├── token_mapping/
    ├── pretrained_tokenizer_amazon/   # optional, see below
    ├── requirements.txt
    ├── train_amazon.sh
    └── *.py
```

### 3. (Optional) Amazon-specific tokenizer

We additionally define Amazon-specific tokens and train their embeddings so that they are merged into the existing LLM tokenizer embedding layer.

- This option corresponds to `pretrained_tokenizer_yn='y'` in `train_amazon.sh`.
- To enable it, download the `pretrained_tokenizer_amazon/` folder from the Drive link above and place it in `duelrec/`.

### 4. Train

```bash
cd duelrec
bash train_amazon.sh
```

`train_amazon.sh` launches via `torchrun --nproc_per_node=2`; set it to the number of GPUs you have.
Training is followed by a distributed leave-one-out evaluation that reports HIT/NDCG@{1,5,10} and MRR,
both overall and per domain.
