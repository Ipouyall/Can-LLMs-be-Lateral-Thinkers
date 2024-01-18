python3 train.py \
    --save_steps 25 \
    --logging_steps 25 \
    --learning_rate 2e-4 \
    --weight_decay 0.001 \
    --warmup_ratio 0.03 \
    --fp16 \
    --bf16 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --save_total_limit 5 \
    --lr_scheduler_type "cosine" \
    --dataset_address "../SentencePuzzleKD/KD_train_gpt-4_revised.csv" \
    --output_dir "./brain_teaser_explain_llama2_checkpoints"
