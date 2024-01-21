python3 inference2.py \
    --max_length 1000 \
    --temperature 0 \
    --model_name_or_path "NousResearch/Llama-2-7b-chat-hf" \
    --dataset_address "./infer.jsonl" \
    --output_path "./infer2.jsonl"
