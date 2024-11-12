# ppi_research

## Installation

1. Clone this repo
2. `cd ppi_research`
3. `python -m pip install -U -e .`

## Running Ankh/ESM

1. `cd ppi_research`
2. `cd training`
3. `cd ankh # or cd esm`
4. `cd lora`
5. `python attention_training.py --ckpt ankh_base --ds_name skempi # or --ckpt esm_650m`

## If you want to know what are the available checkpoints to run use the following:

1. `cd ppi_research`
2. `cd training`
3. `cd ankh # or cd esm`
4. `cd lora`
5. `python attention_training.py --help # this will show you what is the available datasets and checkpoints to pass`
