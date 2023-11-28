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
from data import load_gsm8k, preprocess_function_gsm8k, preprocess_function_spider, preprocess_function_ende, preprocess_function_finance, preprocess_function_python, preprocess_function_generic, load_dataset_with_answers

IGNORE_TOKEN_ID = LabelSmoother.ignore_index


def load_dataset(data_args, training_args, tokenizer):
    train_dataset = None
    eval_dataset = None
    predict_dataset = None

    global preprocess_function_gsm8k
    global preprocess_function_spider
    global preprocess_function_finance
    global preprocess_function_python
    global load_dataset_with_answers
    
    if data_args.dataset_name == 'gsm8k':
        global load_gsm8k
        preprocess_function = preprocess_function_gsm8k
        train_dataset, predict_dataset = load_gsm8k(train_path=os.path.join(os.getcwd(), 'data/gsm8k/train.jsonl'), test_path=os.path.join(os.getcwd(), 'data/gsm8k/test.jsonl'))
    elif data_args.dataset_name == 'wmt16' and data_args.dataset_config_name == 'de-en':
        global preprocess_function_ende
        # support english to german only
        preprocess_function = preprocess_function_ende
        raw_datasets = datasets.load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
        )
    elif data_args.dataset_name == 'spider':
        preprocess_function = preprocess_function_spider
        raw_datasets = datasets.load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
        )
    elif data_args.dataset_name == 'code_search_net':
        preprocess_function = preprocess_function_python
        raw_datasets = datasets.load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
        )
    elif data_args.dataset_name == 'gbharti/finance-alpaca':
        preprocess_function = preprocess_function_finance
        raw_datasets = datasets.load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
        )
    elif data_args.dataset_name == 'gsm8k_with_answers':
        preprocess_function = preprocess_function_gsm8k
        train_dataset = load_dataset_with_answers(train_path=os.path.join(os.getcwd(), 'data/gsm8k_with_answer_t5.jsonl'))
    elif data_args.dataset_name == 'spider_with_answers':
        preprocess_function = preprocess_function_spider
        train_dataset = load_dataset_with_answers(train_path=os.path.join(os.getcwd(), 'data/spider_with_answer_t5.jsonl'))
    elif data_args.dataset_name == 'finance_with_answers':
        preprocess_function = preprocess_function_finance
        train_dataset = load_dataset_with_answers(train_path=os.path.join(os.getcwd(), 'data/gbharti/finance-alpaca_with_answer_t5.jsonl'))
    elif data_args.dataset_name == 'python_with_answers':
        preprocess_function = preprocess_function_python
        train_dataset = load_dataset_with_answers(train_path=os.path.join(os.getcwd(), 'data/code_search_net_with_answer_t5.jsonl'))
    else:
        # generic data preprocessing
        global preprocess_function_generic
        preprocess_function = preprocess_function_generic
        raw_datasets = datasets.load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
        )

    prefix = data_args.source_prefix if data_args.source_prefix is not None else ""
    print(f'prefix: {prefix}')

    if 'with_answers' in data_args.dataset_name:
        partial_preprocess_function = partial(
            preprocess_function,
            tokenizer=tokenizer,
            args=data_args,
            train_args=training_args
        )
    else:
        partial_preprocess_function = partial(
            preprocess_function,
            tokenizer=tokenizer,
            args=data_args,
            train_args=training_args,
            prefix=prefix
        )

    # preprocess datasets that don't have validation sets
    require_val_tasks = [
        'gsm8k',
        'gsm8k_with_answers',
        'spider_with_answers',
        'python_with_answers',
        'finance_with_answers',
    ]
    if data_args.dataset_name in require_val_tasks and (training_args.do_eval or training_args.do_predict):
        if training_args.do_predict:
            # take 1/5 of training set to be prediction dataset
            train_indices = range(len(train_dataset)//5 * 4)
            predict_indices = range(len(train_dataset)//5 * 4, len(train_dataset))

            predict_dataset = datasets.Dataset.from_dict(train_dataset[predict_indices])  
            train_dataset = datasets.Dataset.from_dict(train_dataset[train_indices])

            print('train dataset size: {}'.format(len(train_dataset)))
            print('predict dataset size: {}'.format(len(predict_dataset)))
            if training_args.do_eval:
                # take 1/5 of training set to be eval dataset
                train_indices = range(len(train_dataset)//5 * 4)
                eval_indices = range(len(train_dataset)//5 * 4, len(train_dataset))

                eval_dataset = datasets.Dataset.from_dict(train_dataset[eval_indices])  
                train_dataset = datasets.Dataset.from_dict(train_dataset[train_indices])

                print('train dataset size: {}'.format(len(train_dataset)))
                print('eval dataset size: {}'.format(len(eval_dataset)))
        elif training_args.do_eval:
            train_indices = range(len(train_dataset)//5 * 4)
            eval_indices = range(len(train_dataset)//5 * 4, len(train_dataset))

            eval_dataset = datasets.Dataset.from_dict(train_dataset[eval_indices])  
            train_dataset = datasets.Dataset.from_dict(train_dataset[train_indices])

            print('train dataset size: {}'.format(len(train_dataset)))
            print('eval dataset size: {}'.format(len(eval_dataset)))



    num_proc = 16
    # Preprocessing the datasets for summarization tasks.
    # We need to tokenize inputs and targets.
    if training_args.do_train:
        if not train_dataset:
            train_dataset = raw_datasets["train"]
        
        if data_args.dataset_name == 'wmt16':
            # handle special case: WMT16 dataset is too large and too slow to tokenize, not the entire set is needed
            train_dataset = train_dataset.select(range(len(train_dataset) // 2))

        print('train dataset size: {}'.format(len(train_dataset)))

        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_dataset.map(
                partial_preprocess_function,
                num_proc=num_proc,
                batched=True,
                remove_columns=train_dataset.column_names,
                desc="Running tokenizer on train dataset",
            )
    if training_args.do_eval:
        if not eval_dataset:
            eval_dataset = raw_datasets["validation"]
        print('eval dataset size: {}'.format(len(eval_dataset)))
        max_target_length = data_args.val_target_max_length
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            eval_dataset = eval_dataset.map(
                partial_preprocess_function,
                num_proc=num_proc,
                batched=True,
                remove_columns=eval_dataset.column_names,
                desc="Running tokenizer on eval dataset",
            )
    if training_args.do_predict:
        if not predict_dataset:
            predict_dataset = raw_datasets["test"]
        print('test dataset size: {}'.format(len(predict_dataset)))
        max_target_length = data_args.test_target_max_length
        if data_args.max_predict_samples is not None:
            max_predict_samples = min(len(predict_dataset), data_args.max_predict_samples)
            predict_dataset = predict_dataset.select(range(max_predict_samples))
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            predict_dataset = predict_dataset.map(
                partial_preprocess_function,
                num_proc=num_proc,
                batched=True,
                remove_columns=predict_dataset.column_names,
                desc="Running tokenizer on test dataset",
            )  

    # data partitioning for fast evaluation model (only a subset of the validation set is used)
    if data_args.fast_eval:  
        if training_args.do_eval:
            if len(eval_dataset) < 200:
                eval_length = len(eval_dataset)
            else:
                eval_length = 200

            eval_random_indices = random.sample(range(len(eval_dataset)), eval_length)
            eval_dataset = datasets.Dataset.from_dict(eval_dataset[eval_random_indices])
            print('fast eval... updated eval dataset size: {}'.format(len(eval_dataset)))

        if training_args.do_predict:
            if len(predict_dataset) < 200:
                eval_length = len(predict_dataset)
            else:
                eval_length = 200
            
            predict_random_indices = random.sample(range(len(predict_dataset)), eval_length)
            predict_dataset = datasets.Dataset.from_dict(predict_dataset[predict_random_indices])
            print('fast eval... updated test dataset size: {}'.format(len(predict_dataset)))
    
    return train_dataset, eval_dataset, predict_dataset
