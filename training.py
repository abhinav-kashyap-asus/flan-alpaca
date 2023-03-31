import argparse
import functools
import os

import pytorch_lightning as pl
import torch
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import FSDPStrategy
from torch.distributed.fsdp import (
    MixedPrecision,
    FullyShardedDataParallel,
    StateDictType,
    FullStateDictConfig,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.utils.data import DataLoader
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Adafactor
from transformers.models.t5.modeling_t5 import T5Block
from peft import get_peft_model, LoraConfig, TaskType
from pytorch_lightning.loggers.wandb import WandbLogger
from data_loading import TextToTextDataset
import multiprocessing

os.environ["TOKENIZERS_PARALLELISM"] = "true"


class MyFSDPStrategy(FSDPStrategy):
    @staticmethod
    def clean_up_state_names(state: dict, prefix="_forward_module.") -> dict:
        """
        To restore original transformer state dict, remove FSDP name prefix from keys
        """
        new = {}
        for k in state.keys():
            assert k.startswith(prefix)
            new[k[slice(len(prefix), len(k))]] = state[k]
        return new

    def lightning_module_state_dict(self):
        """
        Returns model state for checkpointing.
        Original FSDPStrategy returns state of unwrapped lightning module which is incomplete
        But we need the FSDP-wrapped module to get the full state dict to load checkpoints properly
        See: https://pytorch.org/tutorials/intermediate/FSDP_adavnced_tutorial.html?highlight=transformer
        """
        # model = self.lightning_module
        model = self.model
        assert model is not None

        with FullyShardedDataParallel.state_dict_type(
            module=model,
            state_dict_type=StateDictType.FULL_STATE_DICT,
            state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
        ):
            state = model.state_dict()
            state = self.clean_up_state_names(state)
            print(dict(my_fsdp=type(model), state=len(state), io=self.checkpoint_io))
            return state

    def save_checkpoint(self, checkpoint: dict, filepath: str, **kwargs) -> None:
        """
        Save model/training states as a checkpoint file through state-dump and file-write.
        Default TorchCheckpointIO saves dict to bytes and bytes to file, which may take up more cpu memory
        So we bypass it and save direct from dict to file
        """
        if self.is_global_zero:
            print(dict(save_checkpoint_unused_kwargs=kwargs))
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            torch.save(checkpoint, filepath)


def init_args(raw_args):
    # Training args should follow FlanT5 (Scaling Instruction-Finetuned Language Models)
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="google/flan-t5-base")
    parser.add_argument("--max_source_length", type=int, default=40)
    parser.add_argument("--max_target_length", type=int, default=160)
    parser.add_argument("--data_path", type=str, default="data/train.json")
    parser.add_argument("--train_epochs", type=int, default=3)
    parser.add_argument("--train_batch_size", type=int, default=64)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--use_compile", action="store_true")
    parser.add_argument("--use_gradient_checkpointing", action="store_true")
    parser.add_argument("--use_fsdp", action="store_true")
    parser.add_argument("--debug", action="store_true")
    # Training args related to LoRA
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.1)

    args = parser.parse_args(raw_args)
    return args


class LightningModel(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        print(self.hparams)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.hparams.model_name_or_path
        )
        print(dict(orig_state_dict=len(self.model.state_dict())))

        # Get the lora model
        if self.hparams.use_lora:
            peft_config = LoraConfig(
                task_type=TaskType.SEQ_2_SEQ_LM,
                inference_mode=False,
                r=self.hparams.lora_r,
                lora_alpha=self.hparams.lora_alpha,
                lora_dropout=self.hparams.lora_dropout,
            )
            self.model = get_peft_model(self.model, peft_config)
            self.model.print_trainable_parameters()

        if self.hparams.use_compile:
            self.model = torch.compile(self.model)
        if self.hparams.use_gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        self.tokenizer = AutoTokenizer.from_pretrained(self.hparams.model_name_or_path)

    def forward(
        self,
        input_ids,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        labels=None,
    ):
        return self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
        )

    def _step(self, batch):
        lm_labels = batch["target_ids"]
        lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100

        outputs = self(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            labels=lm_labels,
            decoder_attention_mask=batch["target_mask"],
        )

        loss = outputs[0]
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log(
            "train/loss",
            loss,
            prog_bar=True,
            logger=True,
            on_epoch=True,
            on_step=True,
        )
        return loss

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        params = self.trainer.model.named_parameters()
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in params if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in params if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        # noinspection PyTypeChecker
        optimizer = Adafactor(
            optimizer_grouped_parameters,
            lr=self.hparams.learning_rate,
            relative_step=False,
        )
        return [optimizer]

    def train_dataloader(self):
        dataset = TextToTextDataset(
            path=self.hparams.data_path,
            max_source_length=self.hparams.max_source_length,
            max_target_length=self.hparams.max_target_length,
            tokenizer=self.tokenizer,
        )

        return DataLoader(
            dataset,
            batch_size=self.hparams.train_batch_size,
            drop_last=True,
            shuffle=True,
            num_workers=multiprocessing.cpu_count(),
        )


def main(raw_args=None):
    torch.set_float32_matmul_precision("high")
    args = init_args(raw_args)
    seed_everything(args.seed)
    model = LightningModel(args)

    saver = ModelCheckpoint(
        verbose=True,
        dirpath=args.output_dir,
        save_weights_only=True,
    )

    strategy = "auto"
    if args.use_fsdp:
        # https://pytorch.org/blog/efficient-large-scale-training-with-pytorch/
        # https://lightning.ai/docs/pytorch/stable/advanced/model_parallel.html
        strategy = MyFSDPStrategy(
            auto_wrap_policy=functools.partial(
                transformer_auto_wrap_policy,
                transformer_layer_cls={T5Block},
            ),
            mixed_precision=MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
                buffer_dtype=torch.bfloat16,
            ),
            activation_checkpointing=T5Block,
            cpu_offload=True,
        )

    # Instantiate the loggers
    import wandb

    run_id = wandb.util.generate_id()
    logger = WandbLogger(project="declare-lab-alpaca", id=run_id)

    assert logger is not None

    trainer = pl.Trainer(
        logger=logger,
        precision=32,
        accelerator="gpu",
        strategy=strategy,
        accumulate_grad_batches=1 if args.debug else args.gradient_accumulation_steps,
        default_root_dir=args.output_dir,
        gradient_clip_val=None if args.use_fsdp else 1.0,
        max_epochs=args.train_epochs,
        callbacks=[saver],
        overfit_batches=10 if args.debug else 0,
        devices=2,
        log_every_n_steps=100,
    )

    trainer.fit(model)


"""
p training.py --output_dir outputs/model/base

p training.py --output_dir outputs/model/xl \
--use_compile \
--model_name_or_path "google/flan-t5-xl" \
--train_batch_size 1 \
--gradient_accumulation_steps 64

python training.py --output_dir outputs/model/xxl \
--use_fsdp \
--model_name_or_path "google/flan-t5-xxl" \
--train_batch_size 1 \
--gradient_accumulation_steps 64

"""


if __name__ == "__main__":
    main()
