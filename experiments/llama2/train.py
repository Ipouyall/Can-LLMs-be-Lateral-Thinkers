import pandas as pd
from tqdm import tqdm
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig, TaskType
from trl import SFTTrainer
import argparse
import os

def parse_arguments():
    parser = argparse.ArgumentParser(description="Training Llama-2-7b on Brain Teaser Explain dataset")

    parser.add_argument("--save_steps", type=int, default=25, help="Number of steps to save the model")
    parser.add_argument("--logging_steps", type=int, default=25, help="Number of steps to log training information")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate for training")
    parser.add_argument("--weight_decay", type=float, default=0.001, help="Weight decay for training")
    parser.add_argument("--warmup_ratio", type=float, default=0.03, help="Warmup ratio")
    parser.add_argument("--fp16", action="store_true", help="Enable mixed precision training (FP16)")
    parser.add_argument("--bf16", action="store_true", help="Enable mixed precision training (BF16)")
    parser.add_argument("--output_dir", type=str, default="./brain_teaser_explain_llama2_checkpoints", help="Output directory for checkpoints")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--per_device_train_batch_size", type=int, default=2, help="Batch size per GPU/CPU")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of steps to accumulate gradients")
    parser.add_argument("--save_total_limit", type=int, default=None, help="Total number of checkpoints to save (None for unlimited)")
    parser.add_argument("--dataset_address", type=str, default="../SentencePuzzleKD/KD_train_gpt-4_revised.csv", help="Path to the dataset CSV file")
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine", help="Learning rate scheduler type")

    args = parser.parse_args()
    return args

def create_dataset(dataset_address):
    df = pd.read_csv(dataset_address)
    prompt = """ \
Your task is to generate a descriptive explanation from a question to an answer option. \
In the following, a question and an option as the answer to the question is provided. \
The answer might be or not be a correct answer. \
Write a descriptive explanation in at most one paragraph and 200 words to show that path from question to the answer.
Question: "{question}"
Answer Option: "{option}"
    """
    prompt = prompt.replace("\n", " \n ")
    prompt = prompt.strip()


    custom_dataset = []

    for index, row in tqdm(df.iterrows(), total=len(df)):
        for option_number in range(1, 4):
            custom_dataset.append(
                {
                    'text': '<s>[INST] ' + prompt.replace("{question}", row["question"]).replace("{option}", row[f"option_{option_number}"]) \
                    + " [/INST]" + row[f"hypothesis_{option_number}"] + " </s>"
                }
            )

    converted_dict = {key: [item[key] for item in custom_dataset] for key in custom_dataset[0]}
    return Dataset.from_dict(converted_dict)

def train(args, dataset):
    base_model = "NousResearch/Llama-2-7b-chat-hf"
    compute_dtype = getattr(torch, "bfloat16")

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=quant_config,
        device_map="auto",
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    lora_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.05,
        r=8,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        optim="paged_adamw_32bit",
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        fp16=args.fp16,
        bf16=args.bf16,
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_ratio=args.warmup_ratio,
        group_by_length=True,
        lr_scheduler_type=args.lr_scheduler_type,
        save_total_limit=args.save_total_limit
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=lora_config,
        dataset_text_field="text",
        max_seq_length=None,
        tokenizer=tokenizer,
        args=training_args,
        packing=False,
    )

    resume_from_checkpoint = any(
        os.path.isdir(os.path.join(training_args.output_dir, folder))
        and folder.startswith("checkpoint-") and folder[11:].isdigit()
        for folder in os.listdir(training_args.output_dir)
    )

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model(os.path.join(args.output_dir, "checkpoint-final"))


if __name__ == "__main__":
    args = parse_arguments()

    dataset = create_dataset(args.dataset_address)
    train(args, dataset)
