# Disaster Tweets Classification with BERT (Top 1% on Kaggle)

This repository contains a PyTorch-based Deep Learning solution for the well-known Kaggle competition: **"Natural Language Processing with Disaster Tweets"**. 

Using a robust data-centric preprocessing pipeline combined with a fine-tuned **BERT** (`bert-base-uncased`) architecture under a **5-Fold Cross-Validation Ensemble** strategy, this implementation achieved an elite milestone inside the global leaderboard.

## 📊 Key Achievements & Methodology

- **Leaderboard Position:** Ranked **66th** globally (Top 1% of competitors).
- **Data-Centric AI Focus:** Shifted the strategy from model complexity to text preprocessing quality. Retained critical context clues such as semantic conjunctions ("but", "and") and expressive emojis, preventing the network from losing contextual nuances.
- **Robust Validation:** Employed a 5-Fold `StratifiedKFold` split to ensure stable training evaluation and mitigate any overfitting hazards.
- **Model Architecture:** Fine-tuned `bert-base-uncased` powered by the Hugging Face ecosystem and **PyTorch**.

## 🏗️ Workflow & Infrastructure

1. **Preprocessing:** Raw text was structured through a tailored pipeline to protect text connectors and structural punctuation crucial for disaster sentiment evaluation.
2. **Training Setup:** Distributed training across multiple folds, utilizing dynamic tokenization padding, optimized learning rates ($2 \times 10^{-5}$), and custom PyTorch dataset loaders.
3. **Ensembling:** Model logits from all 5 distinct cross-validation folds were averaged at test-time to achieve higher generalizability during label prediction.

## 🚀 Quick Start

### Installation

Clone the repository and install the required dependencies:

```bash
git clone [https://github.com/YOUR_USERNAME/disaster-tweets-bert.git](https://github.com/YOUR_USERNAME/disaster-tweets-bert.git)
cd disaster-tweets-bert
pip install -r requirements.txt