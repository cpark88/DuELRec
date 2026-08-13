import dataclasses
import logging
import math
import os
import io
import sys
import time
import json
from typing import Optional, Sequence, Union
import tqdm
import copy
import random
# StrOrOpenAIObject = Union[str, openai_object.OpenAIObject]
from typing import Dict, Optional, Sequence
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers import LlamaModel, LlamaForCausalLM, LlamaTokenizer, BitsAndBytesConfig
from transformers.modeling_outputs import SequenceClassifierOutputWithPast, BaseModelOutput, CausalLMOutput, CausalLMOutputWithPast
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import Trainer, TrainerCallback

# def init_weights(module):
#     """ Initialize the weights.
#     """
#     if isinstance(module, (nn.Linear, nn.Embedding)):
#         # Slightly different from the TF version which uses truncated_normal for initialization
#         # cf https://github.com/pytorch/pytorch/pull/5617
#         module.weight.data.normal_(mean=0.0, std=0.02)
#     elif isinstance(module, torch.nn.LayerNorm):
#         module.bias.data.zero_()
#         module.weight.data.fill_(1.0)
#     if isinstance(module, nn.Linear) and module.bias is not None:
#         module.bias.data.zero_()

def init_weights(module):
    """ Initialize the weights.
    """
    if isinstance(module, (nn.Linear)):
        # Slightly different from the TF version which uses truncated_normal for initialization
        # cf https://github.com/pytorch/pytorch/pull/5617
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, torch.nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)
    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()

def smart_tokenizer_and_embedding_resize_v3(
    special_tokens_dict: Dict,
    added_tokens : str,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """
    Resize tokenizer and embedding.
    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    num_added_tokens = tokenizer.add_tokens(added_tokens)
    num_new_tokens = num_added_tokens+num_new_tokens

    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def sequential_loss(pooled_output, labels_id, neg_sample_id, labels_token_id, loss_type, training_args, num_virtual_tokens, tokenizer, item_encoder, item_enc_type, backbone_model):
    if loss_type == 'bpr':
        if training_args.peft_method=='p_tuning':
            tensor = torch.full((labels_id.size(0), num_virtual_tokens, labels_id.size(2)), tokenizer.pad_token_id).to(labels_id.device) #추가#20240904
            # tensor = torch.full((labels_id.size(0), num_virtual_tokens), tokenizer.pad_token_id).to(labels_id.device) #추가
            labels_id = torch.cat([tensor, labels_id], dim = 1) #추가 Bx16xsubtoken |BxSxsubtoken 
            neg_sample_id = torch.cat([tensor, neg_sample_id], dim = 1) #추가
            
            tensor = torch.full((labels_token_id.size(0), num_virtual_tokens), tokenizer.pad_token_id).to(labels_token_id.device) #추가
            labels_token_id = torch.cat([tensor, labels_token_id], dim = -1)
            del tensor

        pos_emb = item_encoder(padded_instances=labels_id, item_enc_type=item_enc_type)
        neg_emb = item_encoder(padded_instances=neg_sample_id, item_enc_type=item_enc_type)

        # [B*S X H]
        pos = pos_emb.view(-1, pos_emb.size(2))
        neg = neg_emb.view(-1, neg_emb.size(2))

        seq_emb = pooled_output.view(-1, backbone_model.config.hidden_size) # [B*S X H]
        pos_logits = torch.sum(pos * seq_emb, -1) # [B*S]
        neg_logits = torch.sum(neg * seq_emb, -1) # [B*S]

        istarget = (labels_token_id > tokenizer.pad_token_id).view(labels_token_id.size(0) * labels_token_id.size(1)).float() # [B*S]#20240904

        loss_seq = torch.sum(
            - torch.log(torch.sigmoid(pos_logits) + 1e-24) * istarget -
            torch.log(1 - torch.sigmoid(neg_logits) + 1e-24) * istarget
        ) / (torch.sum(istarget)+1e-24)

    elif loss_type == 'ce':
        # DuELRec scores items against trainable item embeddings rather than a vocab-sized
        # classification head, so there is no logit matrix to run cross-entropy over.
        raise NotImplementedError(
            "loss_type='ce' is not supported: the model has no vocab-sized scoring head. Use loss_type='bpr'."
        )
    else:
        raise ValueError("loss_type should be either 'bpr' or 'ce', but got {}".format(loss_type))
    return loss_seq

def clm_loss(outputs, clm_loss, training_args):
    if clm_loss=='y':
        loss_clm = outputs.loss
    elif clm_loss=='n':
        loss_clm = torch.tensor(0.0, device=outputs.logits.device)
    else:
        raise ValueError("clm_loss should be either 'y' or 'n', but got {}".format(training_args.clm_loss))
    return loss_clm


def dict_str_key_to_int(target_dict):
    """
    String key --> Int key
    """
    return {int(k):v for k,v in target_dict.items()}
    
def get_metric(pred_list, topk=10):
    NDCG = 0.0
    HIT = 0.0
    MRR = 0.0
    if len(pred_list) == 0:
        return 0.0, 0.0, 0.0
    # [batch] the answer's rank
    for rank in pred_list:
        MRR += 1.0 / (rank + 1.0)
        if rank < topk:
            NDCG += 1.0 / np.log2(rank + 2.0)
            HIT += 1.0
    return HIT /len(pred_list), NDCG /len(pred_list), MRR /len(pred_list)


def get_sample_scores(epoch, pred_list):
    # pred_list is [B x num_candidates] with the ground-truth item at column 0.
    if pred_list is None or len(pred_list) == 0 or pred_list.shape[-1] == 0:
        pred_list = np.zeros((0,), dtype=np.int64)
    else:
        pred_list = (-pred_list).argsort().argsort()[:, 0]
    HIT_1, NDCG_1, MRR = get_metric(pred_list, 1)
    HIT_5, NDCG_5, MRR = get_metric(pred_list, 5)
    HIT_10, NDCG_10, MRR = get_metric(pred_list, 10)
    post_fix = {
        "Epoch": epoch,
        "HIT@1": '{:.4f}'.format(HIT_1), "NDCG@1": '{:.4f}'.format(NDCG_1),
        "HIT@5": '{:.4f}'.format(HIT_5), "NDCG@5": '{:.4f}'.format(NDCG_5),
        "HIT@10": '{:.4f}'.format(HIT_10), "NDCG@10": '{:.4f}'.format(NDCG_10),
        "MRR": '{:.4f}'.format(MRR),
    }
    print(post_fix)
    # with open(self.args.log_file, 'a') as f:
    #     f.write(str(post_fix) + '\n')
    return ([HIT_1, NDCG_1, HIT_5, NDCG_5, HIT_10, NDCG_10, MRR], str(post_fix))

    
    
def neg_sample(item_set, item_size):  
    item = random.randint(2, item_size - 1)
    while item in item_set:
        item = random.randint(2, item_size - 1)
    return item

def neg_sample_set(item_set, item_total):  
    item = int(random.choice(item_total))
    while (item in item_set) | (item == 0) | (item == 1):
        item = int(random.choice(item_total))
    return item
    
def _make_w_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f_dirname = os.path.dirname(f)
        if f_dirname != "":
            os.makedirs(f_dirname, exist_ok=True)
        f = open(f, mode=mode)
    return f


def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f


def jdump(obj, f, mode="w", indent=4, default=str):
    """Dump a str or dictionary to a file in json format.

    Args:
        obj: An object to be written.
        f: A string path to the location on disk.
        mode: Mode for opening the file.
        indent: Indent for storing json dictionaries.
        default: A function to handle non-serializable entries; defaults to `str`.
    """
    f = _make_w_io_base(f, mode)
    if isinstance(obj, (dict, list)):
        json.dump(obj, f, indent=indent, default=default)
    elif isinstance(obj, str):
        f.write(obj)
    else:
        raise ValueError(f"Unexpected type: {type(obj)}")
    f.close()


def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict

def weighted_sample(totals, sample_size):
    # totals = np.cumsum(weights)
    rnd = random.random() * totals[-1]
    idx = np.searchsorted(totals,rnd,'right')
    sample = idx
    return sample

def neg_sample_unigram(item_set, item_size, weight):  
    item = weighted_sample(weight,1)
    while item in item_set:
        item = weighted_sample(weight,1)
    return item




class CustomTrainer(Trainer):
    """Optional Trainer that logs the individual loss terms of CustomModelOutput."""

    # Loss terms defined on CustomModelOutput; anything absent is simply skipped.
    LOSS_KEYS = ('loss', 'sequential_loss', 'llm_loss')

    def compute_loss(self, model, inputs, return_outputs=False):
        # Forward pass
        outputs = model(**inputs)
        total_loss = outputs.loss

        # Log whichever loss terms this output actually carries.
        logs = {}
        for key in self.LOSS_KEYS:
            value = getattr(outputs, key, None)
            if value is not None:
                name = 'total_loss' if key == 'loss' else key
                logs[name] = round(value.item(), 2)
        self.log(logs)

        return (total_loss, outputs) if return_outputs else total_loss

# Custom callback to log additional losses and round epoch
class CustomCallback(TrainerCallback):
    def on_step_begin(self, args, state, control, **kwargs):
        # # Compute the current epoch as a fraction
        # current_epoch = state.global_step / state.max_steps * args.num_train_epochs
        # if current_epoch - math.floor(current_epoch) < 0.5:
        #     # Log if the current epoch is near the 0.01 mark
        #     control.should_log = True

        # Ensure logging happens only on the main process (local_rank == 0)
        if args.local_rank != 0:
            return  # Skip logging for non-main processes
        
        # Compute the current epoch as a fraction
        current_epoch = state.global_step / state.max_steps * args.num_train_epochs
        
        # Check if the current epoch is close to an integer (1.0, 2.0, etc.)
        if args.local_rank == 0 and abs(current_epoch % 10.0) < (1 / state.max_steps):  # Small tolerance to capture 1.0 intervals
            # Force logging if the current epoch is near the 1.0 mark
            control.should_log = True
        else:
            # Suppress logging for other steps
            control.should_log = False


    def on_log(self, args, state, control, logs=None, **kwargs):
        # if logs is not None and 'epoch' in logs:
        #     # Round the epoch to 0.01
        #     logs['epoch'] = round(logs['epoch'], 2)
        if args.local_rank != 0:
            return  # Skip logging for non-main processes        

        if logs is not None and 'epoch' in logs:
            # Round the epoch to 0.5
            # logs['epoch'] = round(logs['epoch'] * 2) / 2  # Round to nearest 0.5
            logs['epoch'] = round(logs['epoch'])# Round to nearest 0.5

        # Loss terms emitted by CustomTrainer (total_loss / sequential_loss / llm_loss)
        # are already present in `logs` and pass through untouched.

# # GradNorm 클래스 정의
# class GradNorm:
#     def __init__(self, num_losses, alpha=0.5):
#         self.weights = nn.Parameter(torch.ones(num_losses))  # 초기 가중치는 1로 설정
#         self.alpha = alpha  # 하이퍼파라미터: 기울기 균형 조정을 위한 지수값

#     def compute_grad_norm(self, model, losses):
#         # 각 손실의 기울기 norm 계산
#         norms = []
#         for loss in losses:
#             model.zero_grad()
#             loss.backward(retain_graph=True)
#             grad_norm = 0
#             for param in model.parameters():
#                 if param.grad is not None:
#                     grad_norm += torch.norm(param.grad, p=2)  # L2 norm
#             norms.append(grad_norm.item())
#         return torch.tensor(norms)

#     def update_weights(self, losses, initial_grad_norm, current_grad_norm):
#         # 각 손실의 상대적인 기울기 norm 계산
#         relative_norms = current_grad_norm / initial_grad_norm
#         mean_relative_norm = relative_norms.mean()

#         # 가중치 업데이트 규칙 적용
#         target_norms = mean_relative_norm * (relative_norms ** self.alpha)
#         self.weights.data = self.weights.data * (target_norms / current_grad_norm)

#     def get_weighted_loss(self, losses):
#         weighted_losses = 0
#         for i, loss in enumerate(losses):
#             weighted_losses += self.weights[i] * loss
#         return weighted_losses