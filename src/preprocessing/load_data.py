import os
from datasets import load_dataset
import pandas as pd
import yaml

def load_config(path="configs/config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)

def main():
    cfg = load_config()
    print(f"Loading {cfg['data']['dataset_config']}...")

    ds = load_dataset(cfg["data"]["dataset_name"], cfg["data"]["dataset_config"])
    split = ds["train"]

    records = []
    for row in split:
        records.append({
            "pubid": row["pubid"],
            "question": row["question"],
            "context": " ".join(row["context"]["contexts"]),
            "long_answer": row["long_answer"],
            "final_decision": row.get("final_decision"),
        })

    df = pd.DataFrame(records)
    os.makedirs(cfg["data"]["processed_dir"], exist_ok=True)
    df.to_json(cfg["data"]["qa_pairs_file"], orient="records", lines=True)

    print(f"Saved {len(df)} records to {cfg['data']['qa_pairs_file']}")
    print(df["final_decision"].value_counts())

if __name__ == "__main__":
    main()