import os

import pandas as pd
import yaml
from datasets import load_dataset


def load_config(path: str = "configs/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main() -> None:
    cfg = load_config()
    os.makedirs(cfg["data"]["raw_dir"], exist_ok=True)

    ds = load_dataset(
        cfg["data"]["dataset_name"],
        cfg["data"]["dataset_config"],
        cache_dir=cfg["data"]["raw_dir"],
    )

    split = ds["train"]

    records = []

    for row in split:
        records.append(
            {
                "pubid": row["pubid"],
                "question": row["question"],
                "context": " ".join(row["context"]["contexts"]),
                "long_answer": row["long_answer"],
                "final_decision": row.get("final_decision"),
            }
        )

    df = pd.DataFrame(records)
    os.makedirs(cfg["data"]["processed_dir"], exist_ok=True)
    df.to_json(cfg["data"]["qa_pairs_file"], orient="records", lines=True)


if __name__ == "__main__":
    main()
