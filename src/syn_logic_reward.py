import re
import sys
sys.path = sys.path + ["SynLogic"]
import random
from task2verifier import verifier_classes

from collections import defaultdict
import torch
from verl import DataProto


import json

class Data:
    """
    Data class for game/corpus
    @param question: question of the game/corpus
    @param answer: answer of the game/corpus
    @param difficulty: difficulty of the game/corpus, from 1 to 10
    """
    def __init__(self, question: str, answer: str, difficulty: int = 1, metadata: dict = None, **kwargs):
        self.question = question
        self.answer = answer
        self.difficulty = difficulty
        self.metadata = metadata
        self.gpt_response = ""
        
    def to_json(self):
        return {
            "question": self.question,
            "answer": self.answer,
            "difficulty": self.difficulty,
            "metadata": self.metadata,
            "gpt_response": self.gpt_response
        }
    
    def to_json_str(self):
        return json.dumps(self.to_json(), ensure_ascii=False)
    
    @classmethod
    def from_json_str(cls, json_str):
        json_data = json.loads(json_str)
        return cls(**json_data)
    
    @classmethod
    def from_json_dict(cls, json_dict):
        instance = cls(**json_dict)
        if 'gpt_response' in json_dict:
            instance.gpt_response = json_dict['gpt_response']
        return instance
    
    @classmethod
    def from_jsonl_file(cls, file_path):
        data_list = []
        with open(file_path, "r") as f:
            for line in f:
                json_data = json.loads(line)
                instance = cls(**json_data)
                if 'gpt_response' in json_data:
                    instance.gpt_response = json_data['gpt_response']
                data_list.append(instance)
        return data_list


def _extract_answer(solution_str):
    pattern = r'<answer>(.*?)</answer>'
    answers = re.findall(pattern, solution_str)
    if len(answers) == 0:
        raise ValueError("could not extract answer")
    return answers[0]


def compute_score_wo_think_format(data_source, solution_str, ground_truth, extra_info=None):
    if data_source == "deepscaler":
        return compute_math_score_wo_think_format(data_source, solution_str, ground_truth, extra_info)
    
    verifier = verifier_classes[data_source]()
    game_data = extra_info["game_data"]
    try:
        solution = _extract_answer(solution_str)
    except ValueError as e:
        score = 0.0
    else:
        score = float(verifier.verify(game_data, solution))
    return score


def compute_math_score_wo_think_format(data_source, solution_str, ground_truth, extra_info=None):
    try:
        prediction = _extract_answer(solution_str)
    except ValueError as e:
        score = 0
    else:
        score = float(prediction == ground_truth)
    return score
    
    
def compute_think_format_score(data_source, solution_str, ground_truth, extra_info=None):
    def check_format(s):
        pattern = r'^<think>.*?</think>\s*<answer>.*?</answer>$'
        if re.match(pattern, s):
            return 1.0
        else:
            return 0.0
    return check_format(solution_str.strip())
    

def compute_score_w_think_format_score(data_source, solution_str, ground_truth, extra_info=None):
    format_reward = compute_think_format_score(data_source, solution_str, ground_truth, extra_info)
    accuracy_reward = compute_score_wo_think_format(data_source, solution_str, ground_truth, extra_info)
    
    return {
        "format_reward": format_reward,
        "accuracy_reward": accuracy_reward,
        "final_reward": 1 if format_reward * accuracy_reward > 0 else 0
    }


def compute_math_score_w_think_format_score(data_source, solution_str, ground_truth, extra_info=None):
    format_reward = compute_think_format_score(data_source, solution_str, ground_truth, extra_info)
    accuracy_reward = compute_math_score_wo_think_format(data_source, solution_str, ground_truth, extra_info)
    return {
        "format_reward": format_reward,
        "accuracy_reward": accuracy_reward,
        "final_reward": 1 if format_reward * accuracy_reward > 0 else 0
    }


def random_score(data_source, solution_str, ground_truth, extra_info=None):
    return 1.0 if random.uniform(0, 1) >= 0.5 else 0.0


def random_score_with_format(data_source, solution_str, ground_truth, extra_info=None):
    format_reward = compute_think_format_score(data_source, solution_str, ground_truth, extra_info)
    accuracy_reward = random_score(data_source, solution_str, ground_truth, extra_info)
    return {
        "format_reward": format_reward,
        "accuracy_reward": accuracy_reward,
        "final_reward": 1 if format_reward * accuracy_reward > 0 else 0
    }

def incorrect_format_score(data_source, solution_str, ground_truth, extra_info=None):
    format_reward = compute_think_format_score(data_source, solution_str, ground_truth, extra_info)
    accuracy_reward = compute_score_wo_think_format(data_source, solution_str, ground_truth, extra_info)
    accuracy_reward = 1.0 - accuracy_reward
    return {
        "format_reward": format_reward,
        "accuracy_reward": accuracy_reward,
        "final_reward": 1 if format_reward * accuracy_reward > 0 else 0
    }


def math_incorrect_format_score(data_source, solution_str, ground_truth, extra_info=None):
    format_reward = compute_think_format_score(data_source, solution_str, ground_truth, extra_info)
    accuracy_reward = compute_math_score_wo_think_format(data_source, solution_str, ground_truth, extra_info)
    accuracy_reward = 1.0 - accuracy_reward
    return {
        "format_reward": format_reward,
        "accuracy_reward": accuracy_reward,
        "final_reward": 1 if format_reward * accuracy_reward > 0 else 0
    }


def math_incorrect_format_score(data_source, solution_str, ground_truth, extra_info=None):
    format_reward = compute_think_format_score(data_source, solution_str, ground_truth, extra_info)
    accuracy_reward = compute_math_score_wo_think_format(data_source, solution_str, ground_truth, extra_info)
    return {
        "format_reward": format_reward,
        "accuracy_reward": 1.0 - accuracy_reward,
        "final_reward": 1 if format_reward * accuracy_reward > 0 else 0
    }

# reward for rule-base RL training
_SynLogic_Scores = {
    "acc_without_think_format": compute_score_wo_think_format,
    "think_format": compute_think_format_score,
    "acc_with_think_format": compute_score_w_think_format_score,
    "random_score_with_format": random_score_with_format,
    "incorrect_format_score": incorrect_format_score,
    "math_acc_with_think_format": compute_math_score_w_think_format_score,
    "math_incorrect_format_score": math_incorrect_format_score,
    "math_think_format": compute_think_format_score,
    "math_random_score_with_format": random_score_with_format,
    "math_incorrect_format_score": math_incorrect_format_score,
    "extropy_min": None,
}
# score for validation
_SynLogic_Scores["validation_acc"] = compute_score_wo_think_format
_SynLogic_Scores["math_validation_acc"] = compute_math_score_wo_think_format


def _preprocess_game_data(game_data_str):
    # conver str to dict
    return Data.from_json_str(game_data_str)
        

class SynLogicRewardManager:
    """The SynLogic reward manager."""
    def __init__(
        self,
        tokenizer,
        num_examine,
        syn_logic_score_type="acc_with_think_format",
        reward_fn_key="data_source",
        max_resp_len=None,
        overlong_buffer_cfg=None,
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.score_type = syn_logic_score_type
        self.compute_score = _SynLogic_Scores[syn_logic_score_type]
        self.reward_fn_key = reward_fn_key
        self.overlong_buffer_cfg = overlong_buffer_cfg
        self.max_resp_len = max_resp_len

        if self.overlong_buffer_cfg is not None:
            assert self.max_resp_len is not None, f"max_resp_len must be provided if {overlong_buffer_cfg=}, but got None"

    def __call__(self, data: DataProto, return_dict: bool = False):
        """We will expand this function gradually based on the available datasets"""

        if self.score_type == "extropy_min":
            return self.entropy_min(data, return_dict)

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch["prompts"]

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            eos_token = self.tokenizer.eos_token
            if response_str.endswith(eos_token):
                response_str = response_str[: -len(eos_token)]

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]

            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            if data_source.startswith("val/"):
                data_source = data_source[4:]

            extra_info = data_item.non_tensor_batch.get("extra_info", None)
            
            if extra_info is not None and "game_data_str" in extra_info:
                extra_info["game_data"] = _preprocess_game_data(extra_info["game_data_str"])
            
            import copy

            result = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=copy.deepcopy(extra_info),
            )

            score: float
            if isinstance(result, dict):
                score = result["final_reward"]
                # Store the information including original reward
                for key, value in result.items():
                    reward_extra_info[key].append(value)
            else:
                score = result
                
            reward = score

            if self.overlong_buffer_cfg.enable:
                overlong_buffer_len = self.overlong_buffer_cfg.len
                expected_len = self.max_resp_len - overlong_buffer_len
                exceed_len = valid_response_length - expected_len
                overlong_penalty_factor = self.overlong_buffer_cfg.penalty_factor
                overlong_reward = min(-exceed_len / overlong_buffer_len * overlong_penalty_factor, 0)
                reward += overlong_reward
                if self.overlong_buffer_cfg.log:
                    reward_extra_info["overlong_reward"].append(overlong_reward)
                    reward_extra_info["overlong"].append(overlong_reward < 0)

            reward_tensor[i, valid_response_length - 1] = reward

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                if isinstance(result, dict):
                    for key, value in result.items():
                        print(f"[{key}]", value)
                else:
                    print("[score]", score)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor

    def entropy_min(self, data: DataProto, return_dict: bool = False):
        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
    
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']

            prompt_length = prompt_ids.shape[-1]

            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            joint_prob = data_item.batch['old_log_probs'][:valid_response_length].mean()
            reward_tensor[i, valid_response_length - 1] = joint_prob

        return reward_tensor