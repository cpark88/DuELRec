# -*- coding:utf-8 -*-


from utils import neg_sample, neg_sample_set, get_sample_scores
from customized_model import OneModelV3
# from customized_model_noise import OneModelV3
import json
from typing import Union, Dict, Optional, Sequence
import tqdm
import numpy as np
import os

import copy
import logging

import torch
import torch.distributed as dist
import transformers
import utils
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
from transformers import Trainer

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_int8_training,
    set_peft_model_state_dict, prepare_model_for_kbit_training,
    PromptEncoderConfig # p-tuning
)
from peft.utils import PeftType # p-tuning

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import wandb
import numpy as np
from transformers import LlamaModel, LlamaForCausalLM, LlamaTokenizer, BitsAndBytesConfig
from safetensors import safe_open

from outputs import ModelArguments, DataArguments, TrainingArguments, MyCallback
from dataset import make_supervised_data_module_v3
from utils import smart_tokenizer_and_embedding_resize_v3, dict_str_key_to_int


def safe_save_model_for_hf_trainer(trainer: Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict) 

def train():

    wandb.init(mode="offline") #offline disabled
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    training_args.label_names=['labels','labels_id', 'answer_id','answer_id_token', 'test_neg', 'neg_sample_id', 'labels_token_id'] # add inputs (multi-task)
    model_path = '_'.join(model_args.model_name_or_path.split('/'))

    # add quantization 
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    
    backbone_model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        quantization_config=bnb_config,
    )
    
    
    if training_args.pretrained_tokenizer_yn=='n':
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right", #right
            use_fast=False,
        )
        print("*"*20)
        print("Using non-pretrained tokenizer!")
        
    elif training_args.pretrained_tokenizer_yn=='y':
        # tokenizer 새롭게 추가
        tokenizer_save_path = f"pretrained_tokenizer_{data_args.data_name}/"
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            tokenizer_save_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right", #right
            use_fast=False,
        )
        emb_save_path = f"pretrained_tokenizer_{data_args.data_name}/new_embedding.pth"
        lm_head_save_path = f"pretrained_tokenizer_{data_args.data_name}/new_lm_head.pth"
        emb_dict = torch.load(emb_save_path, map_location='cuda')
        lm_head_dict = torch.load(lm_head_save_path, map_location='cuda')
        len_tokenizer=max(list(emb_dict.values())[0].shape[0], len(tokenizer))
        print("model_embedding_size:",list(emb_dict.values())[0].shape[0], "added_tokenizer_size:", len(tokenizer))
        backbone_model.resize_token_embeddings(len_tokenizer)
        backbone_model.load_state_dict(emb_dict, strict=False)
        backbone_model.load_state_dict(lm_head_dict, strict=False)
        print("*"*20)
        print("Using pretrained tokenizer!")
    else:
        pass
    
    
    IGNORE_INDEX = -100
    DEFAULT_PAD_TOKEN = "[PAD]"
    DEFAULT_EOS_TOKEN = "</s>"
    DEFAULT_BOS_TOKEN = "<s>"
    DEFAULT_UNK_TOKEN = "<unk>"

    DEFAULT_NEXT_TOKEN = data_args.default_next_token # "<|n|>" 
    DEFAULT_QUERY_TOKEN = data_args.default_query_token # "<q>" # user query

    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    print("Before:",len(tokenizer))
    smart_tokenizer_and_embedding_resize_v3(
        special_tokens_dict=special_tokens_dict,
        added_tokens=[DEFAULT_NEXT_TOKEN, DEFAULT_QUERY_TOKEN],
        tokenizer=tokenizer,
        model=backbone_model,
    )

    query_token_idx = tokenizer.convert_tokens_to_ids(DEFAULT_QUERY_TOKEN)
    print("query_token_idx:", query_token_idx)
    vocab_size = tokenizer.vocab_size
    print("After:",len(tokenizer))
    tokenizer.pad_token_id=tokenizer.eos_token_id#added 20240517
    print(tokenizer)

    index_set = [IGNORE_INDEX, DEFAULT_PAD_TOKEN, DEFAULT_EOS_TOKEN, DEFAULT_BOS_TOKEN, DEFAULT_UNK_TOKEN, DEFAULT_NEXT_TOKEN, DEFAULT_QUERY_TOKEN]

    if training_args.peft_method=='p_tuning':
        num_virtual_tokens=16#32
        peft_config = PromptEncoderConfig(
            peft_type = PeftType.P_TUNING,
            task_type="CAUSAL_LM",
            num_virtual_tokens=num_virtual_tokens,
            token_dim=backbone_model.config.hidden_size,
            num_transformer_submodules=1,
            #num_attention_heads=6,
            #num_layers=6,
            encoder_reparameterization_type="MLP",
            encoder_hidden_size=backbone_model.config.hidden_size//2,
            base_model_name_or_path=model_args.model_name_or_path,
        )

        print(f"Peft Method is {training_args.peft_method}!")

        smart_tokenizer_and_embedding_resize_v3(
            special_tokens_dict=special_tokens_dict,
            added_tokens=[DEFAULT_NEXT_TOKEN, DEFAULT_QUERY_TOKEN],
            tokenizer=tokenizer,
            model=backbone_model,
        )
        query_token_idx = tokenizer.convert_tokens_to_ids(DEFAULT_QUERY_TOKEN)
        print("query_token_idx:", query_token_idx)
        vocab_size = tokenizer.vocab_size
        print("After:",len(tokenizer))
        tokenizer.pad_token_id=tokenizer.eos_token_id#added 20240517
        print("pad_token_idx:", tokenizer.pad_token_id)
        print("unk_token_idx:", tokenizer.unk_token_id)
        # print(tokenizer)

    elif training_args.peft_method=='lora':
        num_virtual_tokens=0
        peft_config = LoraConfig(
            task_type='CAUSAL_LM',#'FEATURE_EXTRACTION', CAUSAL_LM
            r=model_args.lora_r, #16
            lora_alpha=model_args.lora_alpha, #16
            lora_dropout=model_args.lora_dropout, #0.05
            target_modules=["q_proj", "k_proj", "v_proj"],
            bias='none',
        )
        print(f"Peft Method is {training_args.peft_method}!")
    else:
        raise ValueError("peft_method should be either 'p_tuning' or 'lora', but got {}".format(training_args.peft_method))

    with open(f'token_mapping/{model_path}_vocab_tokenizer_mapping_{data_args.data_name}_v3.json', 'r') as f:
        vocab_dict_tokenized = json.load(f)
        vocab_dict_tokenized = dict_str_key_to_int(vocab_dict_tokenized)
    with open(f'token_mapping/{model_path}_vocab_id_type_mapping_{data_args.data_name}_v3.json', 'r') as f:
        vocab_id_type = json.load(f)
    with open(f'token_mapping/{model_path}_vocab_type_set_mapping_{data_args.data_name}_v3.json', 'r') as f:
        vocab_type_set = json.load(f)
    data_args.len_vocab_dict_tokenized = len(vocab_dict_tokenized)
    
    #contextual masking
    training_args.next_token_ids = tokenizer.convert_tokens_to_ids(DEFAULT_NEXT_TOKEN)
    
    one_model = OneModelV3(model_args, data_args, training_args, vocab_dict_tokenized, bnb_config, backbone_model, tokenizer, index_set, peft_config, num_virtual_tokens)
    
    if training_args.peft_method=='lora':
        smart_tokenizer_and_embedding_resize_v3(
            special_tokens_dict=special_tokens_dict,
            added_tokens=[DEFAULT_NEXT_TOKEN, DEFAULT_QUERY_TOKEN],
            tokenizer=tokenizer,
            model=backbone_model,
        )
        query_token_idx = tokenizer.convert_tokens_to_ids(DEFAULT_QUERY_TOKEN)
        print("query_token_idx:", query_token_idx)
        vocab_size = tokenizer.vocab_size
        print("After:",len(tokenizer))
        tokenizer.pad_token_id=tokenizer.eos_token_id
        print("pad_token_idx:", tokenizer.pad_token_id)
        print("unk_token_idx:", tokenizer.unk_token_id)
        # print(tokenizer)

    data_args.model_type = 'train'
    data_module = make_supervised_data_module_v3(tokenizer=tokenizer, data_args=data_args, vocab_id_type=vocab_id_type, vocab_type_set=vocab_type_set, vocab_dict_tokenized=vocab_dict_tokenized)
    trainer = Trainer(model=one_model, tokenizer=tokenizer, args=training_args, **data_module)

    trainer.train()
    print("Training End!")
    trainer.save_state()
    print("Save State!")

    #save lora
    one_model.model.save_pretrained(training_args.output_dir)

    #save seq_encoder
    model_save_path = os.path.join(training_args.output_dir, "adapter.pth")
    seq_encoder = one_model.seq_encoder.state_dict()
    emb_head = one_model.emb_head.state_dict()


    torch.save({'seq_encoder': seq_encoder, 'emb_head': emb_head}, model_save_path) 
    print("Save Model!")

    # =========================
    # DDP TEST (one shard per GPU, as launched by torchrun --nproc_per_node)
    # =========================
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(local_rank)

    one_model.eval()
    one_model.to(device)

    data_args.model_type = 'test'
    data_module = make_supervised_data_module_v3(
        tokenizer=tokenizer, data_args=data_args,
        vocab_id_type=vocab_id_type, vocab_type_set=vocab_type_set,
        vocab_dict_tokenized=vocab_dict_tokenized,
    )
    test_dataset = data_module['train_dataset']

    test_sampler = DistributedSampler(
        test_dataset, num_replicas=world_size, rank=rank,
        shuffle=False, drop_last=False
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=8,
        sampler=test_sampler,
        collate_fn=data_module['data_collator'],
        pin_memory=True,
        num_workers=4,
    )

    str_code = 'test'
    epoch = 0
    rec_data_iter = tqdm.tqdm(
        enumerate(test_dataloader),
        desc=f"Recommendation EP_{str_code}:{epoch} (rank={rank})",
        total=len(test_dataloader),
        disable=(rank != 0),
        bar_format="{l_bar}{r_bar}"
    )

    local_preds = []
    local_types_pos = []

    with torch.no_grad():
        for i, batch in rec_data_iter:
            input_ids       = batch['input_ids'].to(device, non_blocking=True)
            labels          = batch['labels'].to(device, non_blocking=True)
            attention_mask  = batch['attention_mask'].to(device, non_blocking=True)
            answer_id_token = batch['answer_id_token'].to(device, non_blocking=True)
            test_neg        = batch['test_neg'].to(device, non_blocking=True)
            answer_id       = batch['answer_id']  # cpu tensor ok

            # 추론
            logits = one_model.evaluate(
                input_ids=input_ids,
                labels=labels,
                attention_mask=attention_mask,
                answer_id_token=answer_id_token,
                test_neg=test_neg
            )
            local_preds.append(logits.detach().cpu().numpy().copy())

            
            ans_ids_list = answer_id.cpu().numpy().tolist()
            local_types_pos.append([int(vocab_id_type[str(k)]) for k in ans_ids_list])
           
    
    
    # Empty shards must keep the 2-D (num_samples x num_candidates) shape, otherwise
    # the rank-0 concatenate below fails on mismatched ndim.
    num_candidates = local_preds[0].shape[-1] if len(local_preds) else 0
    pred_local = np.concatenate(local_preds, axis=0) if len(local_preds) else np.zeros((0, num_candidates))
    type_pos_local = np.array(sum(local_types_pos, []), dtype=np.int64) if len(local_types_pos) else np.zeros((0,), dtype=np.int64)

    gathered_preds = [None] * world_size
    gathered_pos = [None] * world_size
    dist.all_gather_object(gathered_preds, pred_local)
    dist.all_gather_object(gathered_pos, type_pos_local)

    if rank == 0:
        # Drop empty shards so ranks that received no samples cannot break the concatenate.
        pred_shards = [p for p in gathered_preds if p is not None and p.size]
        pos_shards = [p for p in gathered_pos if p is not None and p.size]
        pred_list_llm = np.concatenate(pred_shards, axis=0) if pred_shards else np.zeros((0, 0))
        type_pos_final = np.concatenate(pos_shards, axis=0) if pos_shards else np.zeros((0,), dtype=np.int64)

        print("Model Performance for LLM")
        print("================================================")
        print(get_sample_scores(epoch, pred_list_llm))

        for domain_type in range(5, 10):
            domain_pred = pred_list_llm[type_pos_final == domain_type]
            print("================================================")
            if len(domain_pred) == 0:
                print(f"{domain_type}: no test samples for this domain")
            else:
                print(f"{domain_type}:", get_sample_scores(epoch, domain_pred))

    dist.barrier()
    # =========================
    # END DDP TEST
    # =========================
        


if __name__ == "__main__":
    train()