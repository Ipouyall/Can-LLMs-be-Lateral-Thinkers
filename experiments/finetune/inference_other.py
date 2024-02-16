from langchain import PromptTemplate,  LLMChain
from langchain import HuggingFacePipeline
from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import pandas as pd
import transformers
import torch
import json
import argparse
import numpy as np


def parse_arguments():
    parser = argparse.ArgumentParser(description="Getting inference from Llama-2-7b on Brain Teaser Explain dataset")

    parser.add_argument("--max_length", type=int, default=500, help="Maximum number of generated tokens")
    parser.add_argument("--temperature", type=float, default=0, help="Temperature for sampling")
    parser.add_argument("--model_name_or_path", type=str, default="./brain_teaser_explain_zephyr_checkpoints/checkpoint-final/", help="HuggingFace model name or path to the checkpoint")

    args = parser.parse_args()
    return args


def read_data(address):
    df = pd.read_csv(address)
    return df


def get_inference_csqa(args):
    def extract_csqa_answer(result: str):
        o1 = result.rfind("Option 1")
        o2 = result.rfind("Option 2")
        o3 = result.rfind("Option 3")
        o4 = result.rfind("Option 4")
        o5 = result.rfind("Option 5")

        answer = np.argmax([o1, o2, o3, o4, o5])
        option_to_answer = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E"}
        return option_to_answer[answer], answer

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    pipeline = transformers.pipeline(
        "text-generation",
        model=args.model_name_or_path,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
        max_length=args.max_length,
        eos_token_id=tokenizer.eos_token_id
    )

    llm = HuggingFacePipeline(pipeline=pipeline, model_kwargs={'temperature': args.temperature})

    template = """ \
I would provide you a question and five options. \
The question is designed in common sense reasoning evaluation format and to answer question, \
you need to choose the best option that is related to the question and is logical. \
You may need to think of the problem from another perspective to find the best answer.

Question: "{question}"

Option 1: "{option_1}"
Option 2: "{option_2}"
Option 3: "{option_3}"
Option 4: "{option_4}"
Option 5: "{option_5}"

o answer this question, you should exactly mention one option, \
so announce the option you think is the best one in the format: \
'Option 1' or 'Option 2' or 'Option 3' or 'Option 4' or 'Option 5':
"""
    template = template.strip()

    prompt = PromptTemplate(
        template=template,
        input_variables=["question", "option_1", "option_2", "option_3", "option_4", "option_5"],
    )
    llm_chain = LLMChain(prompt=prompt, llm=llm)

    df = load_dataset("tau/commonsense_qa", split='validation')[:150]
    itr = zip(df['question'], df['choices']['text'], df['answerKey'])

    output_data = []

    for que, options, key in tqdm(itr, total=150, desc="Inference (CSQA)"):
        data = {"question": que}
        for i, opt in enumerate(options, start=1):
            data[f"option_{i+1}"] = opt
        result = llm_chain.run(data).strip()
        data["zephyr_raw"] = result
        pred_key, pred = extract_csqa_answer(result)
        data["zephyr_pred"] = pred
        data["answer"] = key
        data["score"] = 1 if pred_key == key else 0
        output_data.append(data)
    save_inference(output_data, args.output_path)


def save_inference(data, address):
    with open(address, 'w') as jsonl_file:
        for item in data:
            jsonl_file.write(json.dumps(item) + '\n')


if __name__ == "__main__":
    args = parse_arguments()
    get_inference_csqa(args)
