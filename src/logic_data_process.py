import argparse
import os
import re
import json

import datasets

from verl.utils.hdfs_io import copy, makedirs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sub_set", default="easy")
    parser.add_argument("--local_dir", default="~/data/synlogic")
    parser.add_argument("--hdfs_dir", default=None)

    args = parser.parse_args()

    data_source = "MiniMaxAI/SynLogic"

    dataset = datasets.load_dataset(data_source, args.sub_set)

    train_dataset = dataset["train"]
    test_dataset = dataset["validation"]
    
    def make_map_fn(split):
        def process_fn(example, idx):
            prompt_raw = example["prompt"][0].pop("content")
            prompt = prompt_raw
            
            extra_info = example.pop("extra_info")
            game_data_str = extra_info["game_data_str"]
            if game_data_str is None:
                question = extra_info["original_question"]
                answer = extra_info["original_answer"]
                difficulty = 1
                metadata = {}
                game_data_str = json.dumps(
                    {
                        "question": question,
                        "answer": answer,
                        "difficulty": difficulty,
                        "metadata": metadata
                    }
                )
                
            
            data = {
                "data_source": example["data_source"],
                "prompt": [
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                "ability": example.pop("ability"),
                "reward_model": {
                    "style": "",
                    "solution": "",
                    "answer": "",
                    "ground_truth": "",
                },
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "game_data_str": game_data_str,
                }
            }
            
            return data

        return process_fn
    
    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, f"train_{args.sub_set}.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, f"test_{args.sub_set}.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)