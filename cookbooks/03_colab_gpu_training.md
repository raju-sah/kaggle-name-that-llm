# Cookbook 03: Colab Pro CLI GPU Training

## Purpose
Offloads heavy transformer fine-tuning (`deberta-v3-base/large`, `ModernBERT`) and LLM perplexity extraction to Google Colab Pro via `colab` CLI.

## Workflow Overview
1. **Prepare script & data:** Bundle `src/models/neural_deberta.py` and `data/processed/cv_folds.parquet`.
2. **Execute remote job:**
   ```bash
   colab run --gpu A100 src/models/neural_deberta.py --folds 0,1,2,3,4
   ```
3. **Sync back artifacts:**
   - Retrieve `exp003_deberta_v3.parquet` into `artifacts/oof/`.
   - Upload pre-trained fold checkpoints to Kaggle Dataset `rajucode/name-that-llm-models` using Kaggle CLI:
   ```bash
   kaggle datasets version -p artifacts/kaggle_dataset -m "Add DeBERTa-v3 5-fold checkpoints"
   ```
