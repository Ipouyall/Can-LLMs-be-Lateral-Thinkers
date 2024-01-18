python3 your_inference_script.py \
    --max_length 500 \
    --temperature 0 \
    --model_name_or_path "./brain_teaser_explain_llama2_checkpoints/checkpoint-final/" \
    --dataset_address "../SentencePuzzleKD/KD_train_gpt-4_revised.csv" \
    --output_path "./infer.jsonl"
