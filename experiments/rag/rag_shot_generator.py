import pandas as pd
import json
import os
import glob


def generate_prompt_sample(data_row):
    return f"""\
Riddle: "{data_row[1]}"\n\
Answer: "{data_row[2]}"\n\
Explanation: "{data_row[3]}"\n"""


def convert_rag_shots_to_prompt(rag_df, data_df):
    sample_list = []
    shot_cols = [col for col in rag_df.columns if 'shot' in col]

    for index, row in rag_df.iterrows():
        prompts = {'id': row['id']}
        for shot in shot_cols:
            sample_id = row[shot]
            sample = data_df.loc[data_df['id'] == sample_id].values[0]
            prompt = generate_prompt_sample(sample)
            prompts[shot] = prompt
        sample_list.append(prompts)

    return sample_list


def save_shots_as_jsonl(shots_list, output_path):
    with open(output_path, 'w') as f:
        for item in shots_list:
            f.write(json.dumps(item) + "\n")


def main(data_df_path, rag_df_path):
    data_id = data_df_path.split("/")[-1].split(".")[0].split("_")[-1]
    output_path = ("1" + rag_df_path.split("/")[-1].split("_")[0][1:] + "_" +
                   rag_df_path.split("/")[-1].split("_")[1] + "_prompts" +
                   data_id + ".jsonl")

    print(output_path)

    data_df = pd.read_csv(data_df_path)
    rag_df = pd.read_csv(rag_df_path)
    prompt_df = convert_rag_shots_to_prompt(rag_df, data_df)
    save_shots_as_jsonl(prompt_df, output_path)


if __name__ == '__main__':
    DATA_DF_PATH = "./../logical_relation/SE2024/train_logical_relation.csv"
    RAG_DF_PATHS = glob.glob("./0*.csv")
    for RAG_DF_PATH in RAG_DF_PATHS:
        main(DATA_DF_PATH, RAG_DF_PATH)



