# -*- coding:utf-8 -*-

from utils import neg_sample, neg_sample_set, get_sample_scores, neg_sample_unigram
from typing import Union, Dict, Optional, Sequence
import numpy as np
import os

import copy
import logging
from dataclasses import dataclass

import torch
import transformers
import utils
from torch.utils.data import Dataset

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_int8_training,
    set_peft_model_state_dict
)

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import wandb
import numpy as np
import random
import json
from utils import smart_tokenizer_and_embedding_resize_v3, dict_str_key_to_int
from outputs import ModelArguments, DataArguments, TrainingArguments, MyCallback

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

DEFAULT_NEXT_TOKEN = "<|n|>" 
DEFAULT_QUERY_TOKEN = "<q>" # user query 구분자

PROMPT_DICT = {
    "prompt_input": (
        "You are a an artificial intelligence assistant that recommend useful items to customers based on their profiles and interests.\n\n"
        "The assistant gives helpful, detailed, and sequential item sets to the customers.\n\n"
        "### Instruction:\n{instruction}%s\n\n### Input:\n{input}\n\n### Response:"%DEFAULT_QUERY_TOKEN
    ),
    "prompt_no_input": (
        "You are a an artificial intelligence assistant that recommend useful items to customers based on their profiles and interests.\n\n"
        "The assistant gives helpful, detailed, and sequential item sets to the customers.\n\n"
        "### Instruction:\n{instruction}%s\n\n### Response:"%DEFAULT_QUERY_TOKEN
    ),
}


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """
    Tokenize a list of strings.
    Set the final length of the input sequence (tokenizer.model_max_length).    
    """
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


def custom_replace_neg(tensor, next_index, ignore_index):
    # we create a copy of the original tensor, 
    # because of the way we are replacing them.
    res = tensor.clone()
    res[tensor!=next_index] = ignore_index    
    return res

def custom_replace_pos(tensor, next_index, ignore_index):
    # we create a copy of the original tensor, 
    # because of the way we are replacing them.
    res = tensor.clone()
    res[tensor==next_index] = ignore_index    
    return res

def preprocess_v2(
    sources: Sequence[str],
    targets: Sequence[str],
    targets_id: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing.
       Add the labels_id which indicates the real indices of items.
    """
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)] 
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX

    target_id_split_original = [example.split(',') for example in targets_id]
    target_id_split = [torch.tensor([int(j) for j in i][1:]) for i in target_id_split_original ] 
    next_token_id=tokenizer.convert_tokens_to_ids(DEFAULT_NEXT_TOKEN)
    labels_id = [custom_replace_neg(i, next_token_id, tokenizer.pad_token_id) for i in labels]
    for i,k in enumerate(labels_id):
        k[-1]=next_token_id
        index_s=(k==next_token_id ).nonzero(as_tuple=True)[0] -1 
        k[index_s]=target_id_split[i][:len(index_s)] # then replace the next token id with the labels_id.
    labels_id = [custom_replace_pos(i, next_token_id, tokenizer.pad_token_id) for i in labels_id] 
  
    return dict(input_ids=input_ids, labels=labels, labels_id=labels_id)


class SupervisedDatasetv3(Dataset):
    """Dataset for supervised fine-tuning.
       Add the "labels_id"
       Add the negative samples and the test negative samples.
    """

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer, model_type: str, len_vocab_dict_tokenized:int, vocab_id_type, vocab_type_set, neg_sample_type:str):
        super(SupervisedDatasetv3, self).__init__()
        self.model_type = model_type
        self.len_vocab_dict_tokenized = len_vocab_dict_tokenized
        self.vocab_id_type = vocab_id_type
        self.vocab_type_set = vocab_type_set
        
        self.neg_sample_type = neg_sample_type


        logging.warning("Loading data...")
        list_data_dict = utils.jload(data_path)

        logging.warning("Formatting inputs...")
        prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
        sources = [prompt_input.format_map(example) for example in list_data_dict]

        self.answer_id=[]
        if self.model_type=='inference':
            targets = [f"{ '<|n|>'.join(example['output'].split('<|n|>')[:-1]) }{tokenizer.eos_token}" for example in list_data_dict]
            targets_id = [f"{ ','.join(example['output_id'].split(',')[:]) }" for example in list_data_dict]
            self.answer_id = [ int(example['output_id'].split(',')[-1])   for example in list_data_dict]

        elif self.model_type=='train':
            targets = [f"{ '<|n|>'.join(example['output'].split('<|n|>')[:-3]) }{tokenizer.eos_token}" for example in list_data_dict]
            targets_id = [f"{ ','.join(example['output_id'].split(',')[:-2]) }" for example in list_data_dict]
            self.answer_id = [ int(example['output_id'].split(',')[-1])   for example in list_data_dict]
            

        elif self.model_type=='valid':
            targets = [f"{ '<|n|>'.join(example['output'].split('<|n|>')[:-2]) }{tokenizer.eos_token}" for example in list_data_dict]
            targets_id = [f"{ ','.join(example['output_id'].split(',')[:-1]) }" for example in list_data_dict]
            self.answer_id = [ int(example['output_id'].split(',')[-1])   for example in list_data_dict]

        elif self.model_type=='test':
            targets = [f"{ '<|n|>'.join(example['output'].split('<|n|>')[:-1]) }{tokenizer.eos_token}" for example in list_data_dict]
            targets_id = [f"{ ','.join(example['output_id'].split(',')[:]) }" for example in list_data_dict]
            self.answer_id = [ int(example['output_id'].split(',')[-1])   for example in list_data_dict] # only for test

        else:
            raise ValueError("model_type should be either 'inference' or 'train' or 'valid' or 'test', but got {}".format(self.model_type))

        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = preprocess_v2(sources, targets, targets_id, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.labels_id = data_dict["labels_id"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        test_neg = []
        if self.model_type=='test':
            seq_set = set(self.labels_id[i])
            for _ in range(100):
                test_neg.append(neg_sample(seq_set, self.len_vocab_dict_tokenized)) # vocab

        neg_sample_id = []
        seq_set = set(self.labels_id[i])
        for num in range(len(self.labels_id[i])):
            if self.neg_sample_type == 'basic':
                neg_sample_id.append(neg_sample(seq_set, self.len_vocab_dict_tokenized))
            
            elif self.neg_sample_type == 'type_neg':
                type_index = self.vocab_id_type[str(self.labels_id[i][num].item())]
                type_item_set = self.vocab_type_set[type_index]
                neg_sample_id.append(neg_sample_set(seq_set, type_item_set))

            elif self.neg_sample_type == 'hybrid':
                prob = random.random()
                if prob < 0.1: # p hyperparameter
                    neg_sample_id.append(neg_sample(seq_set, self.len_vocab_dict_tokenized))
                else:
                    type_index = self.vocab_id_type[str(self.labels_id[i][num].item())]
                    type_item_set = self.vocab_type_set[type_index]
                    neg_sample_id.append(neg_sample_set(seq_set, type_item_set))
            else:
                pass

        return dict(input_ids=self.input_ids[i], labels=self.labels[i], labels_id=self.labels_id[i], answer_id=self.answer_id[i], test_neg=test_neg, neg_sample_id=torch.tensor(neg_sample_id))



def customized_pad_sequence(
    sequences: Union[torch.Tensor, list[torch.Tensor]],
    batch_first: bool = False,
    padding_value: float = 0.0,
    pos: str = 'right',
) -> torch.Tensor:

    """
    This function returns a Tensor of size T x B x * or B x T x * where T is the length of the longest sequence. This function assumes trailing dimensions and type of all the Tensors in sequences are same.
    """
    if pos=='right':
        padded_sequence = torch._C._nn.pad_sequence(sequences, batch_first, padding_value)
    elif pos=='left':
        sequences = tuple(map(lambda s: s.flip(0), sequences))
        padded_sequence = torch._C._nn.pad_sequence(sequences, batch_first, padding_value)
        _seq_dim = padded_sequence.dim()
        padded_sequence = padded_sequence.flip(-_seq_dim+batch_first)
    else:
        raise ValueError("pos should be either 'right' or 'left', but got {}".format(pos))
    return padded_sequence

@dataclass
class DataCollatorForSupervisedDatasetv3(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer
    vocab_dict_tokenized: Dict[int, Sequence[int]]

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        vocab_dict_tokenized = self.vocab_dict_tokenized
        input_ids, labels, labels_id, answer_id, test_neg, neg_sample_id = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels","labels_id", "answer_id", "test_neg", "neg_sample_id")) #full

        answer_id = torch.tensor(answer_id) # B
        test_neg = torch.tensor(test_neg) # B x 0
        # print("first:",answer_id.shape, test_neg.shape)
        input_ids = customized_pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id, pos='left') 
        labels = customized_pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX, pos='left')
        
        labels_id = customized_pad_sequence(labels_id, batch_first=True, padding_value=self.tokenizer.pad_token_id, pos='left')
        labels_token_id = labels_id.clone().detach()
        neg_sample_id = customized_pad_sequence(neg_sample_id, batch_first=True, padding_value=self.tokenizer.pad_token_id, pos='left')
        labels_id = torch.stack([torch.stack([torch.tensor(vocab_dict_tokenized[int(item)]).long() for item in instance]) for instance in labels_id]) # B x S x #sub-token
        neg_sample_id = torch.stack([torch.stack([torch.tensor(vocab_dict_tokenized[int(item)]).long() for item in instance]) for instance in neg_sample_id]) # B x S x #sub-token
        answer_id_token = torch.stack([torch.stack([torch.tensor(vocab_dict_tokenized[int(item)]).long() for item in instance]) for instance in answer_id.unsqueeze(1)]) # B x 1 x #sub-token
        
        if test_neg.size()[1]>0: #test일때만
            test_neg = torch.stack([torch.stack([torch.tensor(vocab_dict_tokenized[int(item)]).long() for item in instance]) for instance in test_neg]) # B x 100 x #sub-token # B x 1 xsubtoken + B x 100 x subtoken -> B x 100 x subtoken
        
        return dict(
            input_ids=input_ids,
            labels=labels,
            labels_id=labels_id, 
            answer_id=answer_id, 
            answer_id_token=answer_id_token,
            test_neg=test_neg, 
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            neg_sample_id=neg_sample_id, 
            labels_token_id=labels_token_id
        )

def make_supervised_data_module_v3(tokenizer: transformers.PreTrainedTokenizer, data_args, vocab_id_type, vocab_type_set, vocab_dict_tokenized) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = SupervisedDatasetv3(tokenizer=tokenizer, data_path=data_args.data_path, model_type=data_args.model_type, len_vocab_dict_tokenized=data_args.len_vocab_dict_tokenized, vocab_id_type=vocab_id_type, vocab_type_set=vocab_type_set, neg_sample_type=data_args.neg_sample_type)
    data_collator = DataCollatorForSupervisedDatasetv3(tokenizer=tokenizer, vocab_dict_tokenized=vocab_dict_tokenized)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)

