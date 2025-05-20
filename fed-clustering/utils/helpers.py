import json


def get_hash_experiment(file_path="./experiment_info.json"):
    with open(file_path, "r") as f:
        data = json.load(f)
    return data["hash_experimento"]
