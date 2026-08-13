# -*- coding:utf-8 -*-

import numpy as np

import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, BatchNorm1d, ReLU, Tanh
from torch.distributions.normal import Normal

def initialize_non_glu(module, input_dim, output_dim):
    gain_value = np.sqrt((input_dim + output_dim) / np.sqrt(4 * input_dim))
    torch.nn.init.xavier_normal_(module.weight, gain=gain_value)
    # torch.nn.init.zeros_(module.bias)
    return

def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different
        (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) *
        (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": gelu, "relu": F.relu, "swish": swish}


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias
    
class EmbHead(nn.Module):
    """Construct the embeddings from item, position.
    """
    def __init__(self, training_args):
        super(EmbHead, self).__init__()
        layers = [
            torch.nn.Linear(training_args.hidden_size, training_args.hidden_size),
            torch.nn.LeakyReLU(),#torch.nn.GELU(),#torch.nn.SiLU(),#torch.nn.ReLU(),
            torch.nn.Linear(training_args.hidden_size, training_args.hidden_size),
        ]
        self.mlp_head = torch.nn.Sequential(*layers)


    def forward(self, embeddings):
        embeddings = self.mlp_head(embeddings)
        
        return embeddings
    
class SelfAttention(nn.Module):
    def __init__(self, training_args):
        super(SelfAttention, self).__init__()
        if training_args.hidden_size % training_args.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (training_args.hidden_size, training_args.num_attention_heads))
        self.num_attention_heads = training_args.num_attention_heads
        self.attention_head_size = int(training_args.hidden_size / training_args.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(training_args.hidden_size, self.all_head_size)
        self.key = nn.Linear(training_args.hidden_size, self.all_head_size)
        self.value = nn.Linear(training_args.hidden_size, self.all_head_size)

        self.attn_dropout = nn.Dropout(training_args.attention_probs_dropout_prob)

        self.dense = nn.Linear(training_args.hidden_size, training_args.hidden_size)
        self.LayerNorm = LayerNorm(training_args.hidden_size, eps=1e-12)
        self.out_dropout = nn.Dropout(training_args.hidden_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, input_tensor, attention_mask):
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        # [batch_size heads seq_len seq_len] scores
        # [batch_size 1 1 seq_len]
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states

class Intermediate(nn.Module):
    def __init__(self, training_args):
        super(Intermediate, self).__init__()
        self.dense_1 = nn.Linear(training_args.hidden_size, training_args.hidden_size * 4)
        if isinstance(training_args.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[training_args.hidden_act]
        else:
            self.intermediate_act_fn = training_args.hidden_act

        self.dense_2 = nn.Linear(training_args.hidden_size * 4, training_args.hidden_size)
        self.LayerNorm = LayerNorm(training_args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(training_args.hidden_dropout_prob)

    def forward(self, input_tensor):
        hidden_states = self.dense_1(input_tensor)
        hidden_states = self.intermediate_act_fn(hidden_states)

        hidden_states = self.dense_2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states
    
class Layer(nn.Module):
    """
    Self-Attention + Fully-Connected Layer
    """
    def __init__(self, training_args):
        super(Layer, self).__init__()
        self.attention = SelfAttention(training_args)
        self.intermediate = Intermediate(training_args)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        return intermediate_output


    
class WeightedSum(nn.Module):
    def __init__(self, sequence_length):
        super(WeightedSum, self).__init__()
        # 가중치를 sequence_length 크기로 초기화
        self.weights = nn.Parameter(torch.randn(sequence_length))

    def forward(self, tensor1, tensor2):
        # 가중치를 softmax를 통해 sum-to-one으로 변환 (sequence_length 축에 대해)
        softmax_weights = torch.softmax(self.weights, dim=0)
        
        # 두 텐서를 각각 가중치로 곱하고 가중 합 수행
        weighted_sum = softmax_weights.view(1, -1, 1) * tensor1 + (1 - softmax_weights.view(1, -1, 1)) * tensor2
        
        return weighted_sum

# Gating 및 가중 합 모듈 정의
class GatedWeightedSum(nn.Module):
    def __init__(self, hidden_size):
        super(GatedWeightedSum, self).__init__()
        # hidden_size에서 1로 매핑하는 선형 변환 레이어
        self.gate_fc1 = nn.Linear(hidden_size, hidden_size//2)
        self.gate_fc2 = nn.Linear(hidden_size//2, 2)
        self.noise_fc1 = nn.Linear(hidden_size, hidden_size//2)
        self.noise_fc2 = nn.Linear(hidden_size//2, 2)
        self.activation = torch.nn.GELU()
        self.LayerNorm = LayerNorm(hidden_size, eps=1e-12)
        self.noise_epsilon = 1e-3
        self.softplus = nn.Softplus()
        # self.normal = Normal(torch.tensor([0.0]), torch.tensor([1.0]))
        # self.final_fc = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, tensor1, tensor2):
        # 두 텐서를 더함 (batch_size, sequence_length, hidden_size)
        combined_tensor = (tensor1 + tensor2)/2
        # gating 값 계산 (batch_size, sequence_length, 1), softmax로 sum-to-one 제약 적용
        gate_logits = self.gate_fc1(combined_tensor)  # (batch_size, sequence_length, 1)
        gate_logits = self.activation(gate_logits)
        gate_logits = self.gate_fc2(gate_logits)
        
        # Noisy gating is a training-time regularizer only; sampling it at eval time
        # would make evaluation non-deterministic across runs.
        if self.training:
            noise_logits = self.noise_fc1(combined_tensor)  # (batch_size, sequence_length, 1)
            noise_logits = self.activation(noise_logits)
            noise_logits = self.noise_fc2(noise_logits)

            noise_stddev = ((self.softplus(noise_logits) + self.noise_epsilon))
            final_logits = gate_logits + ( torch.randn_like(gate_logits) * noise_stddev)
        else:
            final_logits = gate_logits

        gate = torch.softmax(final_logits, axis=2) # batch x seq x 2
        
        # 두 텐서를 gating 값을 이용해 가중합
        weighted_sum = gate[:,:,0].unsqueeze(-1) * tensor1 + gate[:,:,1].unsqueeze(-1) * tensor2
        
        return weighted_sum    
    
class Encoder(nn.Module):
    """
    Final Self-Attention Layer
    """
    def __init__(self, training_args):
        super(Encoder, self).__init__()
        layer = Layer(training_args)
        self.layer_cross = nn.ModuleList([copy.deepcopy(layer)
                                    for _ in range(training_args.num_hidden_layers)])
        self.layer_single = nn.ModuleList([copy.deepcopy(layer)
                                    for _ in range(training_args.num_hidden_layers)])
        self.training_args=training_args
        self.gated_weight_sum =  GatedWeightedSum(hidden_size=training_args.hidden_size)
        
    def create_attention_mask(self, input_tensor, ignore_token, special_token, mask_value):
        batch_size, sequence_length = input_tensor.shape
        # Initialize the attention mask with large negative values (masking by default)
        attention_mask = torch.full((batch_size, sequence_length, sequence_length), mask_value, device=input_tensor.device)
        # Find all positions of the special token (35777) and calculate block IDs
        special_token_mask = (input_tensor == special_token).float()
        block_ids = torch.cumsum(special_token_mask, dim=1)
        # Mask for valid tokens (exclude -100 tokens)
        valid_token_mask = (input_tensor != ignore_token).float()
        block_ids = block_ids * valid_token_mask  # Ensure -100 tokens are ignored in block calculations

        # Create an upper triangular mask (no look-ahead for future blocks)
        causal_mask = torch.triu(torch.ones((sequence_length, sequence_length), device=input_tensor.device))

        # Allow full attention within blocks by making all tokens within a block able to see each other
        block_ids_expanded = block_ids.unsqueeze(1) == block_ids.unsqueeze(2)  # Compare block IDs for attention
        attention_mask = torch.where(block_ids_expanded, torch.zeros_like(causal_mask), torch.full_like(causal_mask, mask_value))

        # Make it so that each block can see all previous blocks (but not future blocks)
        attention_mask = torch.where(causal_mask == 0, torch.zeros_like(attention_mask), attention_mask)

        # Mask out all positions related to special tokens (35777)
        special_token_mask_row = (input_tensor == special_token).unsqueeze(2).expand(-1, -1, sequence_length)
        special_token_mask_col = (input_tensor == special_token).unsqueeze(1).expand(-1, sequence_length, -1)
        attention_mask = torch.where(special_token_mask_row | special_token_mask_col, torch.full_like(attention_mask, mask_value), attention_mask)

        # Mask out all positions related to -100 in both row and column directions
        invalid_token_mask = (input_tensor == ignore_token).unsqueeze(1).expand(-1, sequence_length, -1) | (input_tensor == ignore_token).unsqueeze(2).expand(-1, -1, sequence_length)
        attention_mask = torch.where(invalid_token_mask, torch.full_like(attention_mask, mask_value), attention_mask)

        extended_attention_mask = attention_mask.unsqueeze(1)

        return extended_attention_mask


    
    def create_attention_mask_per_domain(self, input_tensor, ignore_token, special_token, mask_value):
        batch_size, sequence_length = input_tensor.size()
        device = input_tensor.device
        positions = torch.arange(sequence_length, device=device)

        # Identify masked positions (-100 or 35777)
        invalid_tokens = (input_tensor == ignore_token) | (input_tensor == special_token)

        # Create block IDs using cumsum of (tokens == 35777)
        block_separators = (input_tensor == special_token).int()
        block_ids = torch.cumsum(block_separators, dim=1)

        # Set block IDs for invalid tokens to -1
        block_ids[invalid_tokens] = -1

        # Get maximum block ID across all batches
        max_block_id = block_ids.max().item()

        # Reverse the tokens and block IDs along the sequence dimension
        reversed_tokens = input_tensor.flip(dims=[1])
        reversed_block_ids = block_ids.flip(dims=[1])

        # Flatten batch and sequence dimensions
        batch_indices = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, sequence_length)
        valid_mask = reversed_block_ids != -1

        batch_indices_flat = batch_indices[valid_mask]
        block_ids_flat = reversed_block_ids[valid_mask]
        tokens_flat = reversed_tokens[valid_mask]

        # Create combined indices
        combined_indices = batch_indices_flat * (max_block_id + 1) + block_ids_flat

        # Sort combined_indices and tokens_flat accordingly
        sorted_indices, sorted_order = combined_indices.sort()
        tokens_sorted = tokens_flat[sorted_order]

        # Find the first occurrence of each combined index
        first_occurrence_mask = torch.ones_like(sorted_indices, dtype=torch.bool)
        first_occurrence_mask[1:] = sorted_indices[1:] != sorted_indices[:-1]

        unique_combined_indices = sorted_indices[first_occurrence_mask]
        tokens_first = tokens_sorted[first_occurrence_mask]

        # Create last_tokens_per_block
        total_blocks = batch_size * (max_block_id + 1)
        last_tokens_per_block = torch.full((total_blocks,), -1, dtype=input_tensor.dtype, device=device)
        last_tokens_per_block[unique_combined_indices] = tokens_first

        # Reshape last_tokens_per_block to (batch_size, max_block_id + 1)
        last_tokens_per_block = last_tokens_per_block.view(batch_size, max_block_id + 1)

        # Map block IDs to domain IDs
        domain_ids = torch.full_like(block_ids, -1)
        valid_positions = block_ids != -1
        batch_indices_valid = batch_indices[valid_positions]
        block_ids_valid = block_ids[valid_positions]

        domain_ids_valid = last_tokens_per_block[batch_indices_valid, block_ids_valid]
        domain_ids[valid_positions] = domain_ids_valid

        # Initialize the attention mask with zeros
        attention_mask = torch.zeros((batch_size, sequence_length, sequence_length), dtype=torch.float32, device=device)

        # Mask out invalid query positions
        attention_mask[invalid_tokens.unsqueeze(2).expand(-1, -1, sequence_length)] = mask_value

        # Mask out invalid key positions
        invalid_tokens_k = invalid_tokens.unsqueeze(1).expand(-1, sequence_length, -1)
        attention_mask[invalid_tokens_k] = mask_value

        # Prepare expanded tensors for comparisons
        block_ids_expanded_i = block_ids.unsqueeze(2).expand(-1, -1, sequence_length)
        block_ids_expanded_j = block_ids.unsqueeze(1).expand(-1, sequence_length, -1)

        domain_ids_expanded_i = domain_ids.unsqueeze(2).expand(-1, -1, sequence_length)
        domain_ids_expanded_j = domain_ids.unsqueeze(1).expand(-1, sequence_length, -1)

        positions_i = positions.unsqueeze(0).unsqueeze(2).expand(batch_size, -1, sequence_length)
        positions_j = positions.unsqueeze(0).unsqueeze(1).expand(batch_size, sequence_length, -1)

        # Apply the masking rules
        same_block = (block_ids_expanded_i == block_ids_expanded_j) & (block_ids_expanded_i != -1)
        future_tokens = positions_j > positions_i
        same_domain = (domain_ids_expanded_i == domain_ids_expanded_j) & (domain_ids_expanded_i != -1)

        # Mask positions where tokens are not in the same block and in the future
        mask_cond1 = (~same_block) & future_tokens
        attention_mask[mask_cond1] = mask_value

        # Mask positions where tokens are not in the same block, not in the future, but have different domains
        mask_cond2 = (~same_block) & (~future_tokens) & (~same_domain)
        attention_mask[mask_cond2] = mask_value
        
        extended_attention_mask = attention_mask.unsqueeze(1)

        return extended_attention_mask
    
    
    def forward(self, hidden_states, input_ids, pad_token_id, output_all_encoded_layers=True, expert_type='both'):
        attention_mask = self.create_attention_mask(input_tensor=input_ids, ignore_token=pad_token_id, special_token=self.training_args.next_token_ids, mask_value=-10000) #ignore_token,  special_token, mask_value
        attention_mask_domain = self.create_attention_mask_per_domain(input_tensor=input_ids, ignore_token=pad_token_id, special_token=self.training_args.next_token_ids, mask_value=-10000) #ignore_token,  special_token, mask_value        
        
        all_encoder_layers = []

        if expert_type == "only_cross":
            for layer_module_cross, layer_module_single in zip(self.layer_cross, self.layer_single):
                hidden_states = layer_module_cross(hidden_states, attention_mask)
                if output_all_encoded_layers:
                    all_encoder_layers.append(hidden_states)
            if not output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)  
        elif expert_type == 'only_single':
            for layer_module_cross, layer_module_single in zip(self.layer_cross, self.layer_single):
                hidden_states = layer_module_single(hidden_states, attention_mask_domain)
                if output_all_encoded_layers:
                    all_encoder_layers.append(hidden_states)
            if not output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)  

        else: #expert_type == 'both' 
        
            for layer_module_cross, layer_module_single in zip(self.layer_cross, self.layer_single):
                hidden_states_cross = layer_module_cross(hidden_states, attention_mask)
                hidden_states_single = layer_module_single(hidden_states, attention_mask_domain)
                hidden_states = self.gated_weight_sum(hidden_states_cross, hidden_states_single)
                if output_all_encoded_layers:
                    all_encoder_layers.append(hidden_states)
            if not output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        return all_encoder_layers
    
class CrossAttention(nn.Module):
    def __init__(self, training_args):
        super(CrossAttention, self).__init__()
        if training_args.hidden_size % training_args.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (training_args.hidden_size, training_args.num_attention_heads))
        self.num_attention_heads = training_args.num_attention_heads
        self.attention_head_size = int(training_args.hidden_size / training_args.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(training_args.hidden_size, self.all_head_size)
        self.key = nn.Linear(training_args.hidden_size, self.all_head_size)
        self.value = nn.Linear(training_args.hidden_size, self.all_head_size)

        self.attn_dropout = nn.Dropout(training_args.attention_probs_dropout_prob)

        self.dense = nn.Linear(training_args.hidden_size, training_args.hidden_size)
        self.LayerNorm = LayerNorm(training_args.hidden_size, eps=1e-12)
        self.out_dropout = nn.Dropout(training_args.hidden_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, query_tensor, input_tensor, attention_mask):
        mixed_query_layer = self.query(query_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        # [batch_size heads seq_len seq_len] scores
        # [batch_size 1 1 seq_len]
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        # Fixme
        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states

class CrossLayer(nn.Module):
    def __init__(self, training_args):
        super(CrossLayer, self).__init__()
        self.cross_attention = CrossAttention(training_args)
        self.intermediate = Intermediate(training_args)

    def forward(self, query_hidden_states, hidden_states, attention_mask):
        attention_output = self.cross_attention(query_hidden_states, hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        return intermediate_output


class CrossEncoder(nn.Module):
    def __init__(self, training_args):
        super(CrossEncoder, self).__init__()
        layer = CrossLayer(training_args)
        self.layer = nn.ModuleList([copy.deepcopy(layer)
                                    for _ in range(training_args.num_cross_hidden_layers)])

    def make_att_mask(self, input_ids, pad_token_id):
        seq_attention_mask = (input_ids > pad_token_id).long()
        extended_attention_mask = seq_attention_mask.unsqueeze(1).unsqueeze(2) # torch.int64
        max_len = seq_attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1) # torch.uint8
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1) # 0 or 1
        subsequent_mask = subsequent_mask.long().to(input_ids.device)

        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def forward(self, query_hidden_states, hidden_states, input_ids, pad_token_id, output_all_encoded_layers=True):
        attention_mask = self.make_att_mask(input_ids, pad_token_id)
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(query_hidden_states, hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers