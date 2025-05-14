# ppi_research

## Installation

1. Clone `ppi_research` repo.
2. `cd ppi_research`
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
