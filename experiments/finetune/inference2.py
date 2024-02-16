from langchain import PromptTemplate,  LLMChain
from langchain import HuggingFacePipeline
from transformers import AutoTokenizer
from tqdm import tqdm
import pandas as pd
import transformers
import torch
import json
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description="Getting inference from zephyr on Brain Teaser Explain dataset")

    parser.add_argument("--max_length", type=int, default=500, help="Maximum number of generated tokens")
    parser.add_argument("--temperature", type=float, default=0, help="Temperature for sampling")
    parser.add_argument("--model_name_or_path", type=str, default="HuggingFaceH4/zephyr-7b-beta", help="HuggingFace model name or path to the checkpoint")
    parser.add_argument("--dataset_address", type=str, default="./infer.jsonl", help="Path to the dataset jsonl file")
    parser.add_argument("--output_path", type=str, default="./infer2.jsonl", help="Path to save output.jsonl")

    args = parser.parse_args()
    return args


def read_data(address):
    json_list = []
    with open(address, 'r') as file:
        sample = []
        for i, line in enumerate(file, start=1):
            data = json.loads(line)
            sample.append(data)
            if i % 3 == 0:
                json_list.append(sample)
                sample = []
    return json_list

def get_inference(args):
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
You are given a riddle and four options to choose the answer amongst them. \
The fourth option is "None of the above options". \
Your final task is choosing the best option that is related to the riddle. \
For the first three options, you are given a context that explains a path between the question and the answer. \
Although these contexts may try to say their option is true, you should compare all the options based on the question \
and options' context to choose the one that has the most logical answer. If none of them seem logical, 
choose the fourth option: "None of the above options." \
Now, consider the riddle below and the context provided for you, and tell me which option is \
the best answer to the riddle due to the context. \

Riddle: "{question}"

Options:
Option 1: "{option 1}"
Option 2: "{option 2}"
Option 3: "{option 3}"
Option 4: "None of the above options."

Contexts:
Context about option 1: "{context 1}"
Context about option 2: "{context 2}"
Context about option 3: "{context 3}"

To answer this riddle, you should exactly mention one option, \
so announce the option you think is the best one in the format: 'Option 1' or 'Option 2' or 'Option 3' or 'Option 4':
"""

    template = template.replace("\n", " \n ")
    template = template.strip()

    prompt = PromptTemplate(
        template=template,
        input_variables=["question", "option 1", "option 2", "option 3", "context 1", "context 2", "context 3"]
    )
    llm_chain = LLMChain(prompt=prompt, llm=llm)

    df = read_data(args.dataset_address)

    output_data = []

    for data in tqdm(df, total=len(df)):
        answer = llm_chain.run(
            {
                'question': data[0]["question"],
                'option 1': data[0]["option"],
                'option 2': data[1]["option"],
                'option 3': data[2]["option"],
                'context 1': data[0]["hypothesis"],
                'context 2': data[1]["hypothesis"],
                'context 3': data[2]["hypothesis"],
            }
        )
        output_data.append(
            {
                'question': data[0]["question"],
                'option 1': data[0]["option"],
                'option 2': data[1]["option"],
                'option 3': data[2]["option"],
                'answer': answer
            }
        )
    
    save_inference(output_data, args.output_path)


def save_inference(data, address):
    with open(address, 'w') as jsonl_file:
        for item in data:
            jsonl_file.write(json.dumps(item) + '\n')


if __name__ == "__main__":
    args = parse_arguments()
    get_inference(args)
