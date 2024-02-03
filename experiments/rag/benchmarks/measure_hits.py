import pandas as pd
import glob


def find_files(dir: str, ends_with: str):
    return glob.glob(f"{dir}/*{ends_with}", recursive=True)


def hit_ratio(df: pd.DataFrame):
    variations = ["SR", "CR"]
    ret_cols = [col for col in df.columns if 'retrieve' in col]
    hits = 0
    for _, row in df.iterrows():
        org = row['org id'].split("_")[0]
        sample_variations = [org] + [f"{org}_{var}" for var in variations]
        for col in ret_cols:
            if row[col] in sample_variations:
                hits += 1
    hit_rate = 3 * len(df)
    hit_rate = hits / hit_rate
    return hits, hit_rate, len(ret_cols)


if __name__ == "__main__":
    files = sorted(find_files(".", ".csv"))
    print(f"{'File':<32}| {'Hits':<6}| {'Hit Rate':<9}| Retrieved")
    for file in files:
        df = pd.read_csv(file)
        hits, hit_rate, retrieved = hit_ratio(df)
        print(f"{file[-30:]:<32}| {hits:<6}| {round(hit_rate,3):<9}| {retrieved}")