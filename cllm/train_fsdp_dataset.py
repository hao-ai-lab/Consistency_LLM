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

import os
import sys

import functools
import numpy as np
import torch
from torch.utils.data import Dataset
import transformers
from transformers.trainer_pt_utils import LabelSmoother, get_module_class_from_name
from fastchat.model.model_adapter import get_conversation_template
import datasets

from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from torch.distributed.fsdp import (
   FullyShardedDataParallel,
   CPUOffload,
)

from torch.distributed.fsdp.wrap import (
   size_based_auto_wrap_policy,
   transformer_auto_wrap_policy,
)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from typing import Dict

from dataset_distill_trainer import DistillTrainer, DistillTrainerCallback

import logging
logger = logging.getLogger(__name__)

IGNORE_TOKEN_ID = LabelSmoother.ignore_index


@dataclass
class ModelArguments:
    student_model_path: Optional[str] = field(
        default="models/vicuna-7b-v1.5",  metadata={"help": "Path to student model"})
    teacher_model_path: Optional[str] = field(
        default="models/vicuna-7b-v1.5",  metadata={"help": "Path to teacher model"})

@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    eval_data_path: str = field(
        default=None, metadata={"help": "Path to the evaluation data."}
    )
    lazy_preprocess: bool = False


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    max_propose_num: int = field(
        default=5,
        metadata={
            "help": "gamma, number of tokens the student model proposes for each step"
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
    max_new_tokens: int = field(
        default=8,
        metadata={
            "help": "the size of distill interval"
        }
    )
    max_new_seq_len: int = field(
        default=128,
        metadata={
            "help": "the max new generated sequence length"
        }
    )
    consistency_loss: str = field(
        default="global",
        metadata = {
            "choices" : ["global", "local", "both", "self_with_supervised"]
        }
    )

    report_to: str = field(
        default='tensorboard',
        metadata={
            'help': 'The list of integrations to report the results and logs to.'
        }
    )

    logging_dir: str = field(
        default='/liymai24/sjtu/siqi/experiments',
        metadata={
            'help': 'where to save logging'
        }
    )

    logging_steps: int = field(
        default=10,
        metadata={
            "help": "logging steps"
        }
    )

def rank0_print(local_rank, *args):
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


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for consistency training."""
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = [torch.tensor(x) for x in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = [torch.tensor(x) for x in labels]
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

def preprocess_distill_data(
    jacobian_prompt,
    answer_trajectory_ids,
    labels_ids,
    tokenizer: transformers.PreTrainedTokenizer,
    model: str
) -> Dict:
    
    jacobian_trajectory_ids = []
    for answer_ids in answer_trajectory_ids:
        jacobian_prompt_ids = tokenizer(jacobian_prompt, return_tensors="pt")["input_ids"][0]
        answer_ids = torch.tensor(answer_ids).to(jacobian_prompt_ids.device)
        trajectory_ids = torch.cat((jacobian_prompt_ids, answer_ids), dim=0)
        # print(trajectory_ids.shape) # torch.Size([228])
        jacobian_trajectory_ids.append(trajectory_ids)

    return dict(
        jacobian_trajectory=jacobian_trajectory_ids,
        attention_mask=jacobian_trajectory_ids[0].ne(tokenizer.pad_token_id),
        labels_ids=labels_ids
    ) 
    
class JacobianDataset(Dataset):
    """Dataset for consistency training."""

    def __init__(self, raw_data,
                 tokenizer: transformers.PreTrainedTokenizer,
                 model: str,
                 do_eval: bool = False,
                 local_rank: int = -1):
        super(JacobianDataset, self).__init__()
        self.tokenizer = tokenizer

        rank0_print(local_rank, "Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.raw_data = raw_data
        self.cached_data_dict = {}
        self.do_eval = do_eval
        self.model = model

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i) -> Dict:
        if i in self.cached_data_dict:
            return self.cached_data_dict[i]

        ret = preprocess_distill_data([self.raw_data[i]["jacobian_prompt"]],
                         self.raw_data[i]["answer_trajectory_ids"],
                         self.raw_data[i]["labels_ids"],
                         self.tokenizer,
                         self.model)
        self.cached_data_dict[i] = ret

        return ret


def make_jacobian_data_module(
    tokenizer: transformers.PreTrainedTokenizer,
    data_args,
    model: str,
    local_rank: int,
) -> Dict:
    """Make dataset and collator for consistency training."""
    assert data_args.lazy_preprocess, "only support lazy process"
    dataset_cls = JacobianDataset
    rank0_print("Loading data...")

    train_json_part1 = json.load(open('/liymai24/sjtu/siqi/wizard_dataset/WizardLM_evol_instruct_V2_143k_jacobian16_augTrue_labels_True_max_seq_len_256.json', "r"))
    train_json_part2 = json.load(open('/liymai24/sjtu/siqi/wizard_dataset/WizardLM_evol_instruct_V2_143k_jacobian16_augTrue_labels_True_max_seq_len_256_part2.json', "r"))
    truncated_train_json = []
    for data in train_json_part1:
        if tokenizer(data['jacobian_prompt'],return_tensors="pt")['input_ids'].shape[1]<500 :
            truncated_train_json.append(data)
    for data in train_json_part2:
        if tokenizer(data['jacobian_prompt'],return_tensors="pt")['input_ids'].shape[1]<500 :
            truncated_train_json.append(data)
    train_dataset = dataset_cls(truncated_train_json,
                                tokenizer=tokenizer,
                                model=model,
                                local_rank=local_rank)

    if data_args.eval_data_path:
        eval_json = json.load(open(data_args.eval_data_path, "r"))
        eval_dataset = dataset_cls(eval_json,
                                   tokenizer=tokenizer,
                                   model=model,
                                   do_eval=True,
                                   local_rank=local_rank)
    else:
        eval_dataset = None

    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset)


def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = int(os.environ["LOCAL_RANK"])
    training_args.local_rank = local_rank

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set RoPE scaling factor
    config = transformers.AutoConfig.from_pretrained(
        model_args.student_model_path,
        cache_dir=training_args.cache_dir,
    )
    orig_ctx_len = getattr(config, "max_position_embeddings", None)
    if orig_ctx_len and training_args.model_max_length > orig_ctx_len:
        scaling_factor = float(
            math.ceil(training_args.model_max_length / orig_ctx_len))
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}
    config.use_cache = False

    # Load model and tokenizer
    # student model
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.student_model_path,
        config=config,
        cache_dir=training_args.cache_dir,
    )
    # teacher model
    teacher_config = transformers.AutoConfig.from_pretrained(
        model_args.teacher_model_path,
        cache_dir=training_args.cache_dir,
    )
    teacher_model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.teacher_model_path,
        config=teacher_config,
        cache_dir=training_args.cache_dir,
        # device_map='cuda',
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.student_model_path,
        padding_side="right",
        use_fast=False,
    )
    if 'vicuna' in model_args.student_model_path:
        tokenizer.pad_token = tokenizer.unk_token

    # Load data
    data_module = make_jacobian_data_module(tokenizer=tokenizer,
                                              data_args=data_args,
                                              model=model_args.student_model_path,
                                              local_rank=training_args.local_rank)

    trainer = DistillTrainer(
        model=model, tokenizer=tokenizer,
        teacher_model=teacher_model, args=training_args, **data_module
    )
    trainer.add_callback(DistillTrainerCallback)

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=False)
    else:
        trainer.train()
    model.config.use_cache = True
    trainer.save_state()
    safe_save_model_for_hf_trainer(
        trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
