import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['WANDB_PROJECT'] = 'PPIRefExperiments'
# os.environ['WANDB_MODE'] = 'disabled'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"

from transformers import AutoTokenizer
from transformers import T5EncoderModel
from ppi_research.data_adapters import ppi_datasets
from ppi_research.data_adapters.collators import SequenceConcatCollator
from ppi_research.models import SequenceConcatModel
from ppi_research.utils import create_run_name
from transformers import Trainer
from transformers import TrainingArguments
from ppi_research.metrics import compute_ppi_metrics
from ppi_research.utils import set_seed
from ppi_research.utils import ankh_checkpoint_mapping
from ppi_research.utils import ankh_checkpoints
import argparse


seed = 7
set_seed(seed=seed)


def main():
    ckpt = args.ckpt
    ds_name = args.ds_name
    print("Checkpoint:", ckpt)
    tokenizer = AutoTokenizer.from_pretrained(ckpt)
    model = T5EncoderModel.from_pretrained(ckpt)
    downstream_model = SequenceConcatModel(model)

    run_name = create_run_name(
        backbone=ckpt,
        setup="convbert_sequence_concat_randomized",
    )

    training_args = TrainingArguments(
        output_dir=run_name + "_weights",
        run_name=run_name,
        num_train_epochs=20,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        warmup_steps=1000,
        learning_rate=1e-3,
        weight_decay=0.0,
        logging_dir=f"./logs_{run_name}",
        logging_steps=1,
        do_train=True,
        do_eval=True,
        eval_strategy='epoch',
        gradient_accumulation_steps=16,
        fp16=False,
        fp16_opt_level="02",
        seed=seed,
        load_best_model_at_end=True,
        save_total_limit=1,
        metric_for_best_model='eval_validation_rmse',
        greater_is_better=False,
        save_strategy="epoch",
        report_to="wandb",
        remove_unused_columns=False,
        save_safetensors=False,
    )

    train_ds, val_ds, test_ds = ppi_datasets.load_ppi_dataset(ds_name)

    trainer = Trainer(
        model=downstream_model,
        args=training_args,
        data_collator=SequenceConcatCollator(
            tokenizer=tokenizer,
            randomized=True,
        ),
        train_dataset=train_ds,
        eval_dataset={'validation': val_ds, "test": test_ds},
        compute_metrics=compute_ppi_metrics
    )

    trainer.train()


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--ckpt",
        type=str,
        required=True,
        choices=ankh_checkpoints(),
    )
    argparser.add_argument(
        "--ds_name",
        type=str,
        required=True,
        choices=list(ppi_datasets.available_datasets.keys()),
    )
    args = argparser.parse_args()
    args.ckpt = ankh_checkpoint_mapping(args.ckpt)
    main(args)
