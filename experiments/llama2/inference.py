from langchain import PromptTemplate,  LLMChain
from langchain import HuggingFacePipeline
from transformers import AutoTokenizer
from tqdm import tqdm
import pandas as pd
import transformers
import torch
import json


def parse_arguments():
    parser = argparse.ArgumentParser(description="Getting inference from Llama-2-7b on Brain Teaser Explain dataset")

    parser.add_argument("--max_length", type=int, default=500, help="Maximum number of generated tokens")
    parser.add_argument("--temperature", type=float, default=0, help="Temperature for sampling")
    parser.add_argument("--model_name_or_path", type=str, default="./brain_teaser_explain_llama2_checkpoints/checkpoint-final/", help="HuggingFace model name or path to the checkpoint")
    parser.add_argument("--dataset_address", type=str, default="../SentencePuzzleKD/KD_train_gpt-4_revised.csv", help="Path to the dataset CSV file")
    parser.add_argument("--output_path", type=str, default="./infer.jsonl", help="Path to save output.jsonl")

    args = parser.parse_args()
    return args

def read_data(address):
    df = pd.read_csv(address)
    return df

def get_inference(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    pipeline = transformers.pipeline(
        "text-generation",
        model=args.model_name_or_path,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map={"": 0},
        max_length=args.max_length,
        eos_token_id=tokenizer.eos_token_id
    )

    llm = HuggingFacePipeline(pipeline = pipeline, model_kwargs = {'temperature': args.temperature})

    template = """ \
    Your task is to generate a descriptive explanation from a question to an answer option. \
    In the following, a question and an option as the answer to the question is provided. \
    The answer might be or not be a correct answer. \
    Write a descriptive explanation in at most one paragraph and 200 words to show that path from question to the answer.
    Question: "{question}"
    Answer Option: "{option}"
    """
    template = template.replace("\n", " \n ")
    template = template.strip()
    print({'question': df.iloc[0]["question"], 'option': df.iloc[0]["option_2"]})

    prompt = PromptTemplate(template=template, input_variables=["question", "option"])
    llm_chain = LLMChain(prompt=prompt, llm=llm)

    df = read_data(args.dataset_address)

    output_data = []

    for index, row in tqdm(df.iterrows(), total=len(df)):
        for option_number in range(1, 4):
            hypothesis = llm_chain.run({'question': row["question"], 'option': row[f"option_{option_number}"]})
            output_data.append(
                {
                    'question': row["question"],
                    'option': row[f"option_{option_number}"],
                    'hypothesis': hypothesis
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
