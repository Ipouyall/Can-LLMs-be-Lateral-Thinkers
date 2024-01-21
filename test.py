import json


def save_inference(data, address):
    with open(address, 'w') as jsonl_file:
        for item in data:
            jsonl_file.write(json.dumps(item) + '\n')


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



if __name__ == "__main__":
    data = [
        {"q": 123, "o": 456, "h": 789},
        {"q": 24, "o": 13, "h": 45},
        {"q": 45, "o": 23, "h": 546},
        {"q": 13, "o": 43, "h": 56},
        {"q": 43, "o": 54, "h": 67},
        {"q": 54, "o": 34, "h": 65},
    ]
    file = "./tempalte_test.jsonl"
    save_inference(data, file)

    data2 = read_data(file)
    print(json.dumps(data2, indent=4))