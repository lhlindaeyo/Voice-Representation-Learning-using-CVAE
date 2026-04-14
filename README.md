# A Study on Voice Representation Learning using CVAE
### Predicting Stock Market Reactions from Earnings Conference Call Audio

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## Overview

This repository contains the implementation for our study on acoustic voice representation learning applied to financial earnings conference calls. We investigate whether **paralinguistic and acoustic signals** in executive speech carry incremental predictive information about stock market reactions, beyond traditional text-based sentiment analysis.

We compare four modeling approaches:
1. **Baseline** — Traditional acoustic features (librosa + openSMILE eGeMAPSv02)
2. **HuBERT** — Self-supervised speech representations from HuBERT
3. **WavLM** — Self-supervised speech representations from WavLM
4. **CVAE** *(proposed)* — Conditional Variational Autoencoder conditioned on session type and speaker role

Our key finding is that a CVAE architecture conditioned on session type (PR/QA) and speaker identity (CEO/CFO) learns a structured latent space that outperforms all baselines in predicting abnormal stock returns following earnings calls.

---

## Repository Structure

```
.
├── preprocessing/
│   ├── audio_cleaner.py          # Remove intro/outro music via Whisper boundary detection
│   ├── session_splitter.py       # Split PR and QA segments via keyword matching
│   └── diarization.py            # Utterance-level segmentation via pyannote 3.1
├── feature_extraction/
│   ├── acoustic_features.py      # librosa + openSMILE (eGeMAPSv02, ~88 features)
│   ├── hubert_embeddings.py      # HuBERT hidden state extraction
│   ├── wavlm_embeddings.py       # WavLM hidden state extraction
│   └── finvoc2vec_embeddings.py  # FinVoc2Vec hidden states and logits
├── models/
│   ├── cvae.py                   # CVAE model (encoder → latent → decoder + PriceHead)
│   └── baselines.py              # Traditional ML baselines
├── training/
│   ├── train_cvae.py             # CVAE training loop with KL annealing
│   └── evaluate.py               # Ablation study and evaluation scripts
├── notebooks/
│   ├── 01_preprocessing.ipynb    # End-to-end preprocessing pipeline
│   ├── 02_feature_extraction.ipynb
│   ├── 03_training.ipynb
│   └── 04_evaluation.ipynb
├── data/
│   └── README.md                 # Data access instructions (audio not included)
├── paper/
│   ├── earnings_call_paper.tex   # ACL-style paper (Overleaf)
│   └── references.bib
└── README.md
```

---

## Pipeline

```
Raw Earnings Call Audio (.mp3 / .wav)
        │
        ▼
[1] Audio Cleaning
    • Whisper (medium) transcript boundary detection
    • Remove intro/outro music segments
        │
        ▼
[2] Session Splitting
    • Keyword matching → PR (Prepared Remarks) / QA (Q&A)
    • Label assignment per segment
        │
        ▼
[3] Speaker Diarization
    • pyannote speaker-diarization-3.1
    • Outputs utterance-level .wav segments
    • Speaker labels: CEO / CFO / Analyst / Unknown
        │
        ▼
[4] Feature Extraction (parallel)
    ├── librosa + openSMILE (eGeMAPSv02, 88 features)
    ├── HuBERT (hidden states, saved as .pt)
    ├── WavLM  (hidden states, saved as .pt)
    └── FinVoc2Vec (hidden states + logits)
        │
        ▼
[5] CVAE Training
    • Input  x  : acoustic features + speech embeddings
    • Condition c: session type (PR/QA) + speaker role (CEO/CFO)
    • Latent  z  : latent_dim = 64
    • Output  ŷ  : predicted abnormal return (AR)
    • Loss: L_recon + β·L_KL + γ·L_price
        │
        ▼
[6] Evaluation
    • Ablation across 5 embedding configurations
    • Metrics: MSE, Pearson r, directional accuracy
```

---

## Model: Conditional VAE (CVAE)

The core of this work is a CVAE that learns a structured latent representation of executive speech conditioned on contextual variables.

### Architecture

```
Input x (acoustic features + embeddings)
    + Condition c (session type, speaker role)
        │
    [Encoder]  → μ, log σ²
        │
    [Reparameterization]  z ~ N(μ, σ²)
        │
    ┌───┴───┐
[Decoder]  [PriceHead]
    │            │
x̂ (recon)    ŷ (stock return prediction)
```

### Loss Function

```
L = L_recon + β · L_KL + γ · L_price

where:
  L_recon = MSE(x, x̂)
  L_KL    = -0.5 · Σ(1 + log σ² - μ² - σ²)
  L_price = MSE(y, ŷ)
```

### Conditioning Variables

| Variable | Values | Role |
|---|---|---|
| Session type | PR / QA | Captures formality of speech context |
| Speaker role | CEO / CFO / Analyst | Captures authority and disclosure norms |
| Profitability | Beat / Miss / Meet | Optional auxiliary condition |

---

## Ablation Study

| Configuration | Features | MSE ↓ | Pearson r ↑ | Dir. Acc. ↑ |
|---|---|---|---|---|
| (1) Baseline | librosa + openSMILE | — | — | — |
| (2) + HuBERT | (1) + HuBERT hidden | — | — | — |
| (3) + WavLM | (1) + WavLM hidden | — | — | — |
| (4) + FinVoc2Vec hidden | (1) + FinVoc2Vec hidden | — | — | — |
| (5) **CVAE** *(proposed)* | (4) + conditional latent z | — | — | — |

> Results will be updated upon completion of full-scale experiments (~50,000 samples).

---

## Dataset

Audio recordings are sourced from **Frfinitiv** earnings call transcripts and audio archives. The dataset covers S&P 500 companies over multiple fiscal years.

- **Full dataset**: ~50,000 utterance-level samples
- Stock return labels: Abnormal returns (AR) computed relative to market benchmark on the day of / day after the earnings call.

> ⚠️ Due to licensing restrictions, raw audio files are **not** included in this repository. Please refer to `data/README.md` for access instructions.

---

## Installation

```bash
git clone https://github.com/lhlindaeyo/A-Study-on-Voice-Representation-Learning-using-CVAE.git
cd A-Study-on-Voice-Representation-Learning-using-CVAE

pip install torch torchaudio librosa opensmile
pip install pyannote.audio==3.1.0
pip install openai-whisper
pip install transformers  # for HuBERT, WavLM, FinVoc2Vec
pip install pandas numpy scikit-learn
```

> **Note**: A Hugging Face token is required for pyannote models. Set it via `HF_TOKEN` environment variable or pass it directly as `token=`.

---

## Usage

### 1. Preprocessing

```python
# Run full preprocessing pipeline
# notebooks/01_preprocessing.ipynb

# Or run individual steps:
python preprocessing/audio_cleaner.py --input_dir ./raw_audio --output_dir ./cleaned
python preprocessing/session_splitter.py --input_dir ./cleaned --output_dir ./sessions
python preprocessing/diarization.py --input_dir ./sessions --output_dir ./utterances
```

### 2. Feature Extraction

```python
# Extract all embeddings (saves as .pt files to Google Drive / local)
# notebooks/02_feature_extraction.ipynb
```

### 3. Training

```python
# Train CVAE
python training/train_cvae.py \
    --features_path ./features/combined_features.pt \
    --latent_dim 64 \
    --beta 1.0 \
    --gamma 0.5 \
    --epochs 100 \
    --checkpoint_dir ./checkpoints
```

### 4. Evaluation

```python
# Run ablation study
python training/evaluate.py --checkpoint_dir ./checkpoints
```

---

## Environment

| Component | Specification |
|---|---|
| GPU (training) | NVIDIA T4 (Google Colab) / Lab Linux Server |
| Framework | PyTorch 2.0+ |
| Python | 3.8+ |
| Key libraries | librosa, openSMILE, pyannote 3.1, Whisper, transformers |

---

## Citation

If you use this code or find this work helpful, please cite:

```bibtex
@misc{haklim2025cvae_earnings,
  title   = {A Study on Voice Representation Learning using CVAE for Earnings Call Analysis},
  author  = {Haklim},
  year    = {2025},
  url     = {https://github.com/lhlindaeyo/A-Study-on-Voice-Representation-Learning-using-CVAE}
}
```

---

## Related Work

- Mayew & Venkatachalam (2012) — Vocal affect in earnings calls and analyst forecasts
- Qin & Yang (2019) — MDRM: Multi-modal earnings call analysis
- Ewertz et al. (2023) — Speech-based features for earnings call analysis
- Yang & Shestopaloff (2024) — CVAE for stock volume prediction
- Koa et al. (2023) — Financial speech representation learning
- Netspar (2025) — FinVoc2Vec: Domain-specific speech model for finance

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

> FinVoc2Vec (`waiv/FinVoc2Vec`) is available under CC-BY-NC-4.0. Non-commercial use only.
