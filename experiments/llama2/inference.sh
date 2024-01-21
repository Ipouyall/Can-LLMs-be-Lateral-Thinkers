python3 inference.py \
    --max_length 2000 \
    --temperature 0 \
    --model_name_or_path "./brain_teaser_explain_llama2_checkpoints/checkpoint-final/" \
    --dataset_address "./sp-test.csv" \
    --output_path "./infer.jsonl"
