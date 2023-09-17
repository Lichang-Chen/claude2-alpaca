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

import io
import json
import copy
import logging
import random
from dataclasses import dataclass
from typing import Optional, Dict, Sequence
import pickle

import torch
import transformers
from torch.utils.data import Dataset
from datasets import load_dataset
from datasets import Dataset as HFDataset

IGNORE_INDEX = -100
PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}

def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f

def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict

def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )

def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)
from io_utils import read_jsonlines
class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer, data_fraction: float=1.0, seed: int=42, efficient_load: bool=False, filtering_method: str='random'):
        super().__init__()
        logging.warning("Loading data...")
        if efficient_load:
            data_dict = load_dataset('json', data_files=data_path, split='train')
            used_data_count = int(len(data_dict)*data_fraction)
            if filtering_method == 'random':
                data_dict = data_dict.shuffle(seed=seed).select(range(used_data_count))
            elif filtering_method == 'no_shuffle':
                data_dict = data_dict.select(range(used_data_count))
            else:
                raise ValueError(f"Unexpected filtering method: {filtering_method}, choose from ['random', 'no_shuffle']")
            print(f"using {used_data_count} data out of {len(data_dict)}")
            columns = data_dict.column_names
            # changing column names
            data_dict = data_dict.rename_column('Instruction', 'instruction')
            data_dict = data_dict.rename_column('Input', 'input')
            data_dict = data_dict.rename_column('Response', 'output')
            logging.warning("Formatting inputs...")
            prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
            def _format_data(examples):
                output = []
                for row in zip(examples["instruction"], examples["input"], examples["output"]):
                    examples = {"instruction": row[0], "input": row[1], "output": row[2]}
                    if examples.get("instruction", "") != "":
                        output += [prompt_input.format_map(examples)]
                    else:
                        output += [prompt_no_input.format_map(examples)]
                return {"source": output}
            self.sources = data_dict.map(_format_data, remove_columns=data_dict.column_names, batched=True)['source']
            self.targets = data_dict.map(lambda examples: {"target": [f"{example}{tokenizer.eos_token}" for example in examples['output']]}, remove_columns=data_dict.column_names, batched=True)['target']
            return
        else:
            list_data_dict = jload(data_path)
        used_data_count = int(len(list_data_dict)*data_fraction)
        print(f"using {used_data_count} data out of {len(list_data_dict)}")
        random.seed(seed)
        random.shuffle(list_data_dict)
        list_data_dict = list_data_dict[:used_data_count]

        logging.warning("Formatting inputs...")
        prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
        
        sources = [
            prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
            for example in list_data_dict
        ]
        targets = [f"{example['output']}{tokenizer.eos_token}" for example in list_data_dict]

        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(sources, targets, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        # return len(self.input_ids)
        return len(self.sources)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        # return dict(input_ids=self.input_ids[i], labels=self.labels[i])
        return dict(sources=self.sources[i], targets=self.targets[i])

@dataclass
class DataCollatorForSupervisedDataset:
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        sources = []
        targets = []
        for instance in instances:
            sources.append(instance['sources'])
            targets.append(instance['targets'])
        instances = preprocess(sources, targets, self.tokenizer)
        # input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids, labels = tuple((instances['input_ids'], instances['labels']))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_path, data_fraction: float=1.0, seed: int=42, efficient_load: bool=False, filtering_method: str='random') -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=data_path, data_fraction=data_fraction, seed=seed, efficient_load=efficient_load, filtering_method=filtering_method)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)
