torchrun --nproc_per_node=2 --master_port=1234 train_reco.py \
--model_name_or_path 'TinyLlama/TinyLlama-1.1B-Chat-v1.0' \
--data_name='amazon' \
--bf16 False \
--num_train_epochs 2 \
--per_device_train_batch_size 2 \
--per_device_eval_batch_size 1 \
--gradient_accumulation_steps 16 \
--evaluation_strategy "no" \
--save_strategy "steps" \
--save_steps 200000 \
--save_total_limit 1 \
--weight_decay 0. \
--warmup_ratio 0.03 \
--lr_scheduler_type "cosine" \
--logging_steps 1 \
--tf32 False \
--peft_method='p_tuning' \
--loss_type='bpr' \
--lora_r=16 \
--lora_alpha=16 \
--lora_dropout=0.05 \
--learning_rate 2e-4 \
--model_max_length=4096 \
--default_next_token='<|n|>' \
--default_query_token='<q>' \
--clm_loss='y' \
--item_enc_type='fc_layer' \
--neg_sample_type='hybrid' \
--pretrained_tokenizer_yn='y' \
--data_path ./input_data/amazon_one_model_sequence_v3_temp.json \
--output_dir output_dir/TinyLlama/TinyLlama-1.1B-Chat-v1.0_v3_amazon




