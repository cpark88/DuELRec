# -*- coding:utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers import LlamaModel, LlamaForCausalLM, LlamaTokenizer, BitsAndBytesConfig
from transformers.modeling_outputs import SequenceClassifierOutputWithPast, BaseModelOutput, CausalLMOutput, CausalLMOutputWithPast
from transformers import AutoModelForCausalLM, AutoTokenizer

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_int8_training,
    set_peft_model_state_dict, prepare_model_for_kbit_training,
    PromptEncoderConfig # p-tuning
)

from peft.utils import PeftType
import copy
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence
import json
import numpy as np
import copy

from sequential_reco import Encoder, EmbHead
from outputs import CustomModelOutput
from utils import sequential_loss, clm_loss

class OneModelV3(nn.Module):
    def __init__(self, model_args, data_args, training_args, vocab_dict_tokenized, bnb_config, backbone_model, tokenizer, index_set, peft_config, num_virtual_tokens):
        super(OneModelV3, self).__init__()
        self.model_args, self.data_args, self.training_args = model_args, data_args, training_args
        print(f'Initializing language decoder ...')
        
        self.model_path = '_'.join(self.model_args.model_name_or_path.split('/'))
        
        self.vocab_dict_tokenized = vocab_dict_tokenized
        self.bnb_config = bnb_config
        self.peft_config = peft_config
        self.model = backbone_model
        self.tokenizer = tokenizer
        self.index_set = index_set
        self.IGNORE_INDEX = index_set[0]
        self.DEFAULT_PAD_TOKEN = index_set[1]
        self.DEFAULT_EOS_TOKEN = index_set[2]
        self.DEFAULT_BOS_TOKEN = index_set[3]
        self.DEFAULT_UNK_TOKEN = index_set[4]
        self.DEFAULT_NEXT_TOKEN = index_set[5]
        self.DEFAULT_QUERY_TOKEN = index_set[6]
        self.query_token_idx = self.tokenizer.convert_tokens_to_ids(self.DEFAULT_QUERY_TOKEN)
        self.next_token_idx = self.tokenizer.convert_tokens_to_ids(self.DEFAULT_NEXT_TOKEN)
        self.num_virtual_tokens = num_virtual_tokens
        
        self.model = prepare_model_for_kbit_training(self.model)
        self.model = get_peft_model(self.model, self.peft_config)
        self.model.print_trainable_parameters()
        self.model.config.use_cache = False
        print(f"Peft Method is {self.training_args.peft_method}!")


        # Add Sequential Reco Model on LLM
        self.training_args.hidden_size =  self.model.config.hidden_size
        self.training_args.num_attention_heads =  16
        self.training_args.attention_probs_dropout_prob =  0.5
        self.training_args.hidden_act = 'gelu'
        self.training_args.hidden_dropout_prob =  0.5
        self.training_args.num_hidden_layers =  2
        self.seq_encoder = Encoder(self.training_args)

        # Item embedding for the set of sub-tokens
        self.emb_head = EmbHead(self.training_args)
        self.item_enc_type = self.training_args.item_enc_type

        print('Language decoder initialized.')

    def forward(self, input_ids, labels, labels_id, attention_mask, test_neg, neg_sample_id, answer_id, answer_id_token, labels_token_id):
        """
        Note that all elements in DataCollector should be used as input in forward function.  
        Extracting the last hidden state and user embedding
        """
        # 1. llm output 추출
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, output_hidden_states=True)
        pooled_output = outputs.hidden_states[-1] # B x S x H

        # 3. LLM-based sequence reco modeling
        if self.training_args.peft_method=='p_tuning':
            tensor = torch.full((labels.size(0), self.num_virtual_tokens), self.IGNORE_INDEX).to(labels.device) #추가
            labels = torch.cat((tensor, labels), dim=1)

        # pooled_output = self.seq_encoder(pooled_output, labels, self.IGNORE_INDEX, output_all_encoded_layers=True) # seq_encoder upon LLM backbone
        pooled_output = self.seq_encoder(pooled_output, labels, self.IGNORE_INDEX, output_all_encoded_layers=True) # contexual masking
        

        
        pooled_output = pooled_output[-1] # B x S x H
        pooled_last_output = pooled_output[:,-2,:] # B X H (-1: eos_token) : llm-based user embedding (last) #### user embedding
        
        
        if labels_id is None:
            return outputs, pooled_output, pooled_last_output
            
        else:
            loss_llm_seq = sequential_loss(pooled_output, labels_id, neg_sample_id, labels_token_id, self.model_args.loss_type, self.training_args, self.num_virtual_tokens, self.tokenizer, self.item_encoder, self.item_enc_type, self.model) 
            loss_llm_clm = clm_loss(outputs, self.training_args.clm_loss, self.training_args)

            loss = None
            eta = 0.5
            loss = (1-eta)*loss_llm_seq + eta*loss_llm_clm 

            return CustomModelOutput(
                loss=loss,
                sequential_loss=loss_llm_seq,
                llm_loss=loss_llm_clm,
                logits=pooled_output,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )


    def evaluate(self, input_ids, labels, attention_mask, answer_id_token, test_neg):
        """
        Leave-one-out Evaluation
        """
        _, _, seq_out = self.forward(input_ids=input_ids, labels=labels, labels_id=None, attention_mask=attention_mask, test_neg=None, neg_sample_id=None, answer_id=None, answer_id_token=None, labels_token_id=None)
        test_items = torch.cat((answer_id_token, test_neg), 1) # B x 1 x subtoken + B x 100 x subtoken -> B x 101 x subtoken
        test_items_emb = self.item_encoder(test_items, item_enc_type=self.item_enc_type)
        test_logits_llm = torch.bmm(test_items_emb, seq_out.unsqueeze(-1)).squeeze(-1)

        return test_logits_llm


    def item_encoder(self, padded_instances, item_enc_type): 
        """
        Extracting trainable item embedding (pos/neg)
        """
        padded_att = torch.logical_not(torch.isin(padded_instances, torch.tensor([self.tokenizer.pad_token_id, self.IGNORE_INDEX, self.tokenizer.unk_token_id]).cuda() )).long()
        
        if item_enc_type=='mean_pooling':
            item_emb = self.model.get_input_embeddings()(padded_instances.cuda()) # B x S x #sub_token x H
            padded_att = padded_att.unsqueeze(-1).expand(item_emb.size()) # B x S x #sub_token x H
            sum_embeddings = torch.sum(item_emb * padded_att, 2) # B x S x H
            sum_mask = torch.clamp(padded_att.sum(2), min=1e-9) # B x S x H
            item_emb = (sum_embeddings/sum_mask).cuda()  # B x S x H

        elif item_enc_type=='mean_fc':
            item_emb = self.model.get_input_embeddings()(padded_instances.cuda()) # B x S x #sub_token x H
            padded_att = padded_att.unsqueeze(-1).expand(item_emb.size()) # B x S x #sub_token x H
            sum_embeddings = torch.sum(item_emb * padded_att, 2) # B x S x H
            sum_mask = torch.clamp(padded_att.sum(2), min=1e-9) # B x S x H
            item_emb = sum_embeddings/sum_mask   # B x S x H
            item_emb = self.emb_head(item_emb)
            
        elif item_enc_type=='fc_layer':
            item_emb = self.model.get_input_embeddings()(padded_instances.cuda()) # B x S x #sub_token x H
            item_emb = self.emb_head(item_emb) # B x S x #sub_token x H
            padded_att = padded_att.unsqueeze(-1).expand(item_emb.size()) # B x S x #sub_token x H
            sum_embeddings = torch.sum(item_emb * padded_att, 2) # B x S x H
            sum_mask = torch.clamp(padded_att.sum(2), min=1e-9) # B x S x H
            item_emb = sum_embeddings/sum_mask   # B x S x H
        
        elif item_enc_type == 'shared_llm':
            # padded_instances --> B x S x #sub-token
            num_sub_tokens = padded_instances.shape[-1] # #sub-token
            seq_len = padded_instances.shape[1] # S

            padded_instances = padded_instances.view(-1, num_sub_tokens) # BS x #sub-token
            attention_mask = padded_instances.ne(self.tokenizer.pad_token_id)

            item_emb = self.model(input_ids=padded_instances, attention_mask=attention_mask, labels=None, output_hidden_states=True)
            item_emb = item_emb.hidden_states[-1] # BS x #sub-token x H
            item_emb = item_emb.detach()

            #option 1
            item_emb = item_emb[:,-1,:] # BS x H
            item_emb = item_emb.view(-1, seq_len, self.model.config.hidden_size) # B x S x H
        
        else:
            raise ValueError("item_enc_type should be either 'mean_pooling' or 'mean_transformer' or 'mean_fc' or 'llm_share', but got {}".format(self.training_args.item_enc_type))

        return item_emb