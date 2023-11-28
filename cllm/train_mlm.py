# This code is based on tatsu-lab/stanford_alpaca. Below is the original copyright:
#
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from dataclasses import dataclass, field
import json
import math
import pathlib
from typing import Dict, Optional, Sequence

import wandb

import random

import os

import datasets
import evaluate

import numpy as np
import torch
from torch.utils.data import Dataset
import transformers
from transformers import Trainer
from transformers.trainer_pt_utils import LabelSmoother

from fastchat.conversation import SeparatorStyle
from fastchat.model.model_adapter import get_conversation_template

from functools import partial

# local import
from distill_trainer import DistillTrainer, DistillTrainerCallback
from distill_trainer_seq2seq import Seq2SeqDistillTrainer, Seq2SeqDistillTrainerCallback
from load_dataset import load_dataset

IGNORE_TOKEN_ID = LabelSmoother.ignore_index


@dataclass
class ModelArguments:
    student_model_path: Optional[str] = field(
        default="facebook/opt-125m",  metadata={"help": "Path to student model"})
    teacher_model_path: str = field(
        default=None, metadata={"help": "Path to teacher model"})


@dataclass
class DataArguments:
    dataset_name: str = field(
        default=None, metadata={"help": "Dataset's name."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    metric: str = field(
        default='rouge', metadata={"help": "Metric to use for evaluation."}
    )
    source_max_length: int = field(
        default=1024,
        metadata={
            "help": "Maximum sequence length for the source. Sequences will be right padded (and possibly truncated)."
        },
    )
    train_target_max_length: int = field(
        default=64,
        metadata={
            "help": "Maximum sequence length for the target during training. Sequences will be right padded."
        },
    )
    val_target_max_length: int = field(
        default=128,
        metadata={
            "help": "Maximum sequence length for the target during validation. Sequences will be right padded."
        },
    )
    test_target_max_length: int = field(
        default=128,
        metadata={
            "help": "Maximum sequence length for the target during testing. Sequences will be right padded."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
                "which is used during ``evaluate`` and ``predict``."
            )
        },
    )
    source_prefix: Optional[str] = field(
        default="", metadata={"help": "A prefix to add before every source text (useful for T5 models)."}
    )
    fast_eval: bool = field(
        default=True,
        metadata={"help": "Fast evaluation strategy for logging."}
    )


@dataclass
class Seq2SeqTrainingArguments(transformers.Seq2SeqTrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    max_propose_num: int = field(
        default=5,
        metadata={
            "help": "gamma, number of tokens the student model proposes for each step."
        }
    )
    mode: str = field(
        default="offline",
        metadata={
            "help": "online mode or offline mode"
        }
    )
    online_eval_interval: int = field(
        default=10,
        metadata={
            "help": "evaluation interval for online training"
        }
    )
    online_update_interval: int = field(
        default=1,
        metadata={
            "help": "parameter update interval for online training"
        }
    )
    sample_source: str = field(
        default="student",
        metadata = {
            "choices" : ["student", "teacher", "mix_request", "mix_token"]
        }
    )
    kl_method: str = field(
        default="forward",
        metadata = {
            "choices" : ["forward", "reverse", "jsd"]
        }
    )
    all_token_mask: bool = field(
        default=False,
        metadata = {
            "help": "meaningful only for online learning. use all tokens to compute loss. otherwise use only wrong tokens"
        }
    )

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu()
                          for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def train():
    global local_rank
    
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, Seq2SeqTrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank

    # Set RoPE scaling factor
    config = transformers.AutoConfig.from_pretrained(
        model_args.student_model_path,
        cache_dir=training_args.cache_dir,
    )
    orig_ctx_len = getattr(config, "max_position_embeddings", None)
    if orig_ctx_len and data_args.source_max_length > orig_ctx_len:
        scaling_factor = float(
            math.ceil(data_args.source_max_length / orig_ctx_len))
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}
    config.use_cache = False

    # Load model and tokenizer
    # student model
    model = transformers.AutoModelForSeq2SeqLM.from_pretrained(
        model_args.student_model_path,
        config=config,
        cache_dir=training_args.cache_dir,
    )

    # teacher model
    teacher_config = transformers.AutoConfig.from_pretrained(
        model_args.teacher_model_path,
        cache_dir=training_args.cache_dir,
    )
    teacher_model = transformers.AutoModelForSeq2SeqLM.from_pretrained(
        model_args.teacher_model_path,
        config=teacher_config
    )
    teacher_model.cuda()

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.teacher_model_path,
    )
    tokenizer.pad_token = tokenizer.unk_token

    training_args.max_source_length = data_args.source_max_length
    training_args.max_target_length = data_args.train_target_max_length
    
    # Load data
    train_dataset, eval_dataset, predict_dataset = load_dataset(data_args, training_args, tokenizer)

    # Embedding size, start token sanity check
    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    if (
        hasattr(model.config, "max_position_embeddings")
        and model.config.max_position_embeddings < data_args.source_max_length
    ):
        if model_args.resize_position_embeddings is None:
            model.resize_position_embeddings(data_args.msource_max_length)
        elif model_args.resize_position_embeddings:
            model.resize_position_embeddings(data_args.source_max_length)
        else:
            raise ValueError(
                f"`--source_max_length` is set to {data_args.source_max_length}, but the model only has"
                f" {model.config.max_position_embeddings} position encodings. Consider either reducing"
                f" `--source_max_length` to {model.config.max_position_embeddings} or to automatically resize the"
                " model's position encodings by passing `--resize_position_embeddings`."
            )

    training_args.generation_num_beams = (
        data_args.num_beams if data_args.num_beams is not None else training_args.generation_num_beams
    )

    # Data collator
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    data_collator = transformers.DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if training_args.fp16 else None,
    )

    metric = evaluate.load("rouge")
    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        # rougeLSum expects newline after each sentence
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

        return preds, labels

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        # Replace -100s used for padding as we can't decode them
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        result = {k: round(v * 100, 4) for k, v in result.items()}
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        return result
    
    trainer = Seq2SeqDistillTrainer(
            model=model, 
            teacher_model=teacher_model, 
            tokenizer=tokenizer, 
            train_dataset=train_dataset if training_args.do_train else None,
            eval_dataset=eval_dataset if training_args.do_eval else None,
            data_collator=data_collator,
            compute_metrics=compute_metrics if training_args.predict_with_generate else None,
            propose_num=training_args.max_propose_num,
            args=training_args,
        )
    trainer.add_callback(Seq2SeqDistillTrainerCallback)

    if training_args.do_train:
        if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
            train_result = trainer.train(resume_from_checkpoint=True)
        else:
            train_result = trainer.train()
        trainer.save_model()
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        model.config.use_cache = True
        trainer.save_state()

    # Evaluation
    results = {}
    if training_args.do_eval:
        if isinstance(eval_dataset, dict):
            metrics = {}
            for eval_ds_name, eval_ds in eval_dataset.items():
                dataset_metrics = trainer.evaluate(eval_dataset=eval_ds, metric_key_prefix=f"eval_{eval_ds_name}")
                metrics.update(dataset_metrics)
        else:
            metrics = trainer.evaluate(metric_key_prefix="eval")
        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Testing
    if training_args.do_predict:

        predict_results = trainer.predict(predict_dataset, metric_key_prefix="predict")
        metrics = predict_results.metrics
        max_predict_samples = (
            data_args.max_predict_samples if data_args.max_predict_samples is not None else len(predict_dataset)
        )
        metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)
    
    
    safe_save_model_for_hf_trainer(
        trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
