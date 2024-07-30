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
from typing import Dict, Optional

import os
import sys
import torch
from torch.utils.data import Dataset
import transformers
from transformers.trainer_pt_utils import LabelSmoother, get_module_class_from_name
import datasets

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from typing import Dict

from cllm_trainer_global import CllmTrainer

from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

import logging
logger = logging.getLogger(__name__)

IGNORE_TOKEN_ID = LabelSmoother.ignore_index


@dataclass
class ModelArguments:
    target_model_path: Optional[str] = field(
        default="models/vicuna-7b-v1.5",  metadata={"help": "Path to target model"})
    qlora: Optional[bool] = field(default=False, metadata={"help": "Enable QLoRA processing"})

@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
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
    max_new_tokens: int = field(
        default=16,
        metadata={
            "help": "Size of n_token_sequence in Jacobi trajectory."
        },
    )
    use_gt_labels: bool = False
    report_to: str = field(
        default='wandb',
        metadata={
            'help': 'The list of integrations to report the results and logs to.'
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

def preprocess_distill_data(
    prompt_ids,
    answer_trajectory_ids,
    teacher_output_ids,
    complete_teacher_output_ids,
    tokenizer: transformers.PreTrainedTokenizer,
    model: str,
    labels_ids=None,
) -> Dict:
    
    jacobian_trajectory_ids = []
    # only take batch size 1 for now
    # TODO: support bsz > 1 from the generation script. for now, only prompt ids is in (bsz, seq_len)
    jacobian_prompt_ids = torch.tensor(prompt_ids[0], dtype=torch.int64)
    teacher_output_ids = torch.tensor(teacher_output_ids[0], dtype=torch.int64)
    complete_teacher_output_ids = torch.tensor(complete_teacher_output_ids, dtype=torch.int64)
    for answer_ids in answer_trajectory_ids:
        answer_ids = torch.tensor(answer_ids, dtype=torch.int64)
        #print(answer_ids)
        #print(jacobian_prompt_ids)
        if len(jacobian_prompt_ids.shape) == len(answer_ids.shape):
            trajectory_ids = torch.cat((jacobian_prompt_ids, answer_ids), dim=-1)
        elif len(jacobian_prompt_ids.shape) > len(answer_ids.shape):
            #print(f'prompt: {jacobian_prompt_ids.shape}')
            #print(f'answer: {answer_ids.shape}')
            trajectory_ids = torch.cat((jacobian_prompt_ids[0], answer_ids), dim=-1)
        # print(trajectory_ids.shape) # torch.Size([228])
        jacobian_trajectory_ids.append(trajectory_ids)
   
    if labels_ids:
        return dict(
            jacobian_trajectory=jacobian_trajectory_ids,
            attention_mask=jacobian_trajectory_ids[0].ne(tokenizer.pad_token_id),
            labels_ids=labels_ids,
            teacher_output_ids=teacher_output_ids,
            complete_teacher_output_ids=complete_teacher_output_ids
        )
    else:
        return dict(
            jacobian_trajectory=jacobian_trajectory_ids,
            attention_mask=jacobian_trajectory_ids[0].ne(tokenizer.pad_token_id),
            teacher_output_ids=teacher_output_ids,
            complete_teacher_output_ids=complete_teacher_output_ids
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
        if 'labels_ids' in self.raw_data[i].keys():
            ret = preprocess_distill_data(self.raw_data[i]["prompt_ids"],
                         self.raw_data[i]["answer_trajectory_ids"],
                         self.raw_data[i]["teacher_output_ids"],
                         self.raw_data[i]["complete_teacher_output_ids"],
                         self.tokenizer,
                         self.model,
                         labels_ids=self.raw_data[i]["labels_ids"])
        else:
            ret = preprocess_distill_data(self.raw_data[i]["prompt_ids"],
                         self.raw_data[i]["answer_trajectory_ids"],
                         self.raw_data[i]["teacher_output_ids"],
                         self.raw_data[i]["complete_teacher_output_ids"],
                         self.tokenizer,
                         self.model)
        self.cached_data_dict[i] = ret

        return ret


def make_jacobian_data_module(
    tokenizer: transformers.PreTrainedTokenizer,
    trajectory_path,
    data_args,
    model: str,
    local_rank: int,
) -> Dict:
    """Make dataset and collator for consistency training."""
    assert data_args.lazy_preprocess, "only support lazy process"
    dataset_cls = JacobianDataset
    rank0_print("Loading data...")

    train_json = json.load(open(trajectory_path, "r"))
    truncated_train_json = []
    
    for data in train_json:
        # take prompt lengths with limited size if necessary
        truncated_train_json.append(data)
    train_dataset = dataset_cls(truncated_train_json,
                                tokenizer=tokenizer,
                                model=model,
                                local_rank=local_rank)
    eval_dataset = None

    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset)


def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = int(os.environ["LOCAL_RANK"])
    training_args.local_rank = local_rank
    training_args.qlora = model_args.qlora
    
    torch.set_default_dtype(torch.float)

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
        model_args.target_model_path,
        cache_dir=training_args.cache_dir,
    )
    orig_ctx_len = getattr(config, "max_position_embeddings", None)
    if orig_ctx_len and training_args.model_max_length > orig_ctx_len:
        scaling_factor = float(
            math.ceil(training_args.model_max_length / orig_ctx_len))
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}
    config.use_cache = False
    
    # Load model and tokenizer
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.target_model_path,
        config=config,
        cache_dir=training_args.cache_dir,
        attn_implementation='flash_attention_2',
        device_map='cuda',
        torch_dtype=torch.bfloat16,
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.target_model_path,
        padding_side="right",
        use_fast=False,
    )
    if 'vicuna' in model_args.target_model_path:
        tokenizer.pad_token = tokenizer.unk_token

    if model_args.qlora:
        # Runs w/ qLoRA when qlora tag is enabled is enabled
        model = prepare_model_for_kbit_training(model)
        config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=32,
            lora_alpha=16,
            lora_dropout=0.05,
        )
    
        model = get_peft_model(model, config)
        model.config.use_cache = False

    # Load data
    data_module = make_jacobian_data_module(tokenizer=tokenizer,
                                              trajectory_path=data_args.data_path,
                                              data_args=data_args,
                                              model=model_args.target_model_path,
                                              local_rank=training_args.local_rank)

    trainer = CllmTrainer(
        model=model, tokenizer=tokenizer, args=training_args, **data_module
    )

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
