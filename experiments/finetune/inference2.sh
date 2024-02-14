python3 inference2.py \
    --max_length 1000 \
    --temperature 0 \
    --model_name_or_path "HuggingFaceH4/zephyr-7b-beta" \
    --dataset_address "./infer.jsonl" \
    --output_path "./infer2.jsonl"
