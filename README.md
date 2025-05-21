# PPISeq

## Installation

1. Clone `ppiseq` repo.
2. `cd ppiseq`
3. `python -m pip install -U -e .`

## Training

```bash
export CUDA_VISIBLE_DEVICES=0,1

# For full finetuning with BFloat16:
accelerate launch --mixed-precision bf16 train.py

# For ConvBERT training with frozen embeddings:
accelerate launch train.py

# Important Note:
# Set `ddp_find_unused_parameters` parameter to `true` when training ESM2 or ESM3,
# this parameter is in the `train` key in the training/config/config.yaml
```

## Results

**Table 1: Top 10 runs ranked by test set Spearman correlation ($\rho$). Metrics (mean $\pm$ standard deviation, 3 seeds): Spearman $\rho$, Pearson r, and RMSE ($pK_d$). PAD: Pooled attention addition; HP: Hierarchical pooling**

| PLM        | Setup           | Spearman       | Pearson        | RMSE         | Spearman       | Pearson        | RMSE         |
|------------|-----------------|----------------|----------------|--------------|----------------|----------------|--------------|
|            |                 | (Validation)   | (Validation)   | (Validation) | (Test)         | (Test)         | (Test)         |
|------------|-----------------|----------------|----------------|--------------|----------------|----------------|--------------|
| Prot-T5    | ConvBERT-PAD    | **0.48 $\pm$ 0.02** | 0.48 $\pm$ 0.01 | 1.52 $\pm$ 0.09 | **0.48 $\pm$ 0.03** | **0.51 $\pm$ 0.02** | 1.42 $\pm$ 0.10 |
| Ankh2-Ext1 | Finetuning-HP   | **0.48 $\pm$ 0.01** | **0.49 $\pm$ 0.01** | 1.50 $\pm$ 0.10 | 0.47 $\pm$ 0.01 | 0.48 $\pm$ 0.01 | 1.45 $\pm$ 0.13 |
| ESM2-650M  | ConvBERT-HP     | 0.44 $\pm$ 0.03 | 0.43 $\pm$ 0.03 | 1.86 $\pm$ 0.29 | 0.47 $\pm$ 0.02 | 0.48 $\pm$ 0.02 | 1.74 $\pm$ 0.33 |
| Ankh2-Ext2 | Finetuning-HP   | 0.47 $\pm$ 0.01 | 0.47 $\pm$ 0.01 | 1.51 $\pm$ 0.07 | 0.45 $\pm$ 0.01 | 0.46 $\pm$ 0.02 | 1.43 $\pm$ 0.06 |
| ESM2-650M  | Finetuning-HP   | 0.45 $\pm$ 0.02 | 0.44 $\pm$ 0.02 | 1.75 $\pm$ 0.29 | 0.44 $\pm$ 0.02 | 0.45 $\pm$ 0.01 | 1.68 $\pm$ 0.29 |
| Ankh-Base  | Finetuning-HP   | 0.47 $\pm$ 0.01 | 0.47 $\pm$ 0.00 | **1.47 $\pm$ 0.03** | 0.44 $\pm$ 0.01 | 0.45 $\pm$ 0.01 | **1.41 $\pm$ 0.03** |
| Prot-T5    | ConvBERT-HP     | 0.42 $\pm$ 0.02 | 0.42 $\pm$ 0.02 | 1.66 $\pm$ 0.23 | 0.44 $\pm$ 0.01 | 0.44 $\pm$ 0.01 | 1.59 $\pm$ 0.25 |
| Prot-T5    | Finetuning-PAD  | 0.44 $\pm$ 0.00 | 0.44 $\pm$ 0.00 | 1.65 $\pm$ 0.13 | 0.44 $\pm$ 0.01 | 0.45 $\pm$ 0.01 | 1.57 $\pm$ 0.14 |
| Prot-T5    | Finetuning-HP   | 0.47 $\pm$ 0.02 | 0.46 $\pm$ 0.02 | 1.51 $\pm$ 0.05 | 0.44 $\pm$ 0.01 | 0.45 $\pm$ 0.01 | 1.47 $\pm$ 0.08 |
| ESM2-3B    | Finetuning-PAD  | 0.45 $\pm$ 0.01 | 0.45 $\pm$ 0.01 | 1.49 $\pm$ 0.07 | 0.44 $\pm$ 0.01 | 0.46 $\pm$ 0.00 | 1.43 $\pm$ 0.08 |

**Note:** PAD: Pooled attention addition; HP: Hierarchical pooling