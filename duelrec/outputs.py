from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch
from transformers.utils import ModelOutput
import transformers
from transformers import TrainerCallback

@dataclass
class CustomModelOutput(ModelOutput):
    loss: Optional[torch.Tensor] = None
    sequential_loss: Optional[torch.Tensor] = None
    llm_loss: Optional[torch.Tensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default=None)
    loss_type: str = field(default=None)#added
    lora_r: int = field(default=None)#added #16,#self.args['lora_r'],
    lora_alpha: int= field(default=None)  #16,#self.args['lora_alpha'],
    lora_dropout: float = field(default=None) #added

@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    model_type: str = field(default=None, metadata={"help": "full_train, train, valid, test."})
    len_vocab_dict_tokenized: int = field(default=None, metadata={"help": "num of original tokens."})
    data_name: str = field(default=None, metadata={"help": "Name of the dataset, e.g. 'amazon'."})
    default_next_token : str = field(default=None, metadata={"help": "<|n|>"})
    default_query_token : str = field(default=None, metadata={"help": "<q>"})
    neg_sample_type : str = field(default='basic', metadata={"help": "basic"})
    pad_token_id : int = field(default=None, metadata={"help": "2"})
    

    
@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    lora_yn: str = field(default=None)#added
    clm_loss: str = field(default=None)#added
    item_enc_type: str = field(default=None)#added
    peft_method: str = field(default=None)#added
    pretrained_tokenizer_yn: str = field(default=None)#added
    
    
    
class MyCallback(TrainerCallback):
    "A callback that prints a grad at every step"
       
    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_local_process_zero:
            print(logs)