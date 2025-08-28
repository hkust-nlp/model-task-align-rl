import argparse
import os
import re
import json

import datasets

from verl.utils.hdfs_io import copy, makedirs


prompt_template = "Solve the following problem step by step. First, think about the reasoning process in the mind and then provide the answer. The reasoning process is enclosed within <think> </think> and the final answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here</answer>.\n\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="~/data/math")
    parser.add_argument("--hdfs_dir", default=None)

    args = parser.parse_args()

    data_source = "agentica-org/DeepScaleR-Preview-Dataset"

    # 加载数据集并只取前2条数据，构成一个新的数据集
    dataset = datasets.load_dataset(data_source)["train"]
    
    def make_map_fn(split):
        def process_fn(example, idx):
            problem = example.pop("problem")
            prompt = prompt_template + problem
            answer = example.pop("answer")
            
            data = {
                "data_source": "deepscaler",
                "prompt": [
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                "ability": "math",
                "reward_model": {
                    "style": "rule-based",
                    "solution": example.pop("solution"),
                    "answer": answer,
                    "ground_truth": answer,
                },
                "extra_info": {
                    "split": split,
                    "index": idx,
                }
            }
            
            return data

        return process_fn
    
    dataset = dataset.map(function=make_map_fn("train"), with_indices=True)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    dataset.to_parquet(os.path.join(local_dir, f"deepscaler.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)