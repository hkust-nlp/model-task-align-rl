import random
from typing import List

# 中文提示模板
chinese_prompt_candidates = [
    """你是一位精通dyck语言的专家，其中你必须完成所有类型的未闭合括号（例如[]，{}，<>）的语言序列。你需要根据dyck语言规则，分析括号配对的步骤是否正确。

给定一个初始dyck语言序列和用于推导闭合括号序列的步骤（以思考过程的形式给出），你的任务是确定dyck语言中存在错误推理的位置，可能存在多个错误。

这可能是忘记闭合某个括号、使用错误的闭合括号或者在下一步中不正确地复制闭合括号的子序列。

任务：检查序列，确保括号正确闭合。
输入：{dyck_sequence}
{thoughts}
问题：这个序列中有推理错误吗？如果没有错误，请直接输出空字符串""；如果有错误，请直接输出错误步骤的编号，错误出现在"思考过程N"中。

附加条件：如果有多个错误，请按格式输出：1,3,9""",
    """作为dyck语言专家，你需要完成所有类型括号（如[]，{}，<>）的闭合配对。你将分析一个初始序列，以及用于得出闭合括号序列的推理步骤。

你的任务是找出推理过程中所有错误（如果存在），可能是遗漏了闭合括号、使用了错误的闭合括号或者错误地复制了前一步的闭合序列。

题目：分析以下括号序列的推理步骤是否正确。
序列：{dyck_sequence}
{thoughts}
请问：上述推理过程中是否存在错误？若无错误，请直接回答""；若有错误，请直接输出错误的思考步骤编号。

补充说明：若有多个错误，请用逗号分隔列出所有错误步骤编号，如"7,9,12"""
]

# 英文提示模板
english_prompt_candidates = [
    """You are an expert in a language called dyck where you must complete the language sequence of unclosed brackets of all types (e.g., [], {}, <>). You are given an initial dyck language sequence and the steps, provided as thoughts, that were used to arrive at the closing bracket sequence in the dyck language.

Your job is to identify any steps that were mistakes in reasoning about the closing bracket sequence in dyck language, as there may be multiple errors. This can be forgetting to close a bracket or closing with the wrong bracket or incorrectly copying the prior subsequence of closing brackets to the next step.

Task: Examine the sequence, making sure that the parentheses are closed properly.
Input: {dyck_sequence}
{thoughts}
Q: Is there any mistake in this sequence? If there are no mistakes, output an empty string ""; if there is a mistake in Thought N, output "N".

Additional Conditions: If there were multiple mistakes, then output like "7,9,12" (comma-separated numbers)""",
    """As a dyck language expert, you need to analyze a sequence where all types of brackets ([], {}, <>, etc.) must be properly closed. You'll examine an initial sequence and the reasoning steps used to verify bracket closure.

Your task is to identify all mistakes in the reasoning process, as there may be several errors. Mistakes might include failing to close brackets properly, using incorrect closing brackets, or misrepresenting the stack state.

Examine: {dyck_sequence}
{thoughts}
Question: Are there any errors in the reasoning above? If everything is correct, output an empty string ""; if there are errors, identify which thought(s) contain errors by directly outputting their numbers.

Note: If multiple thoughts contain errors, list them all separated by commas (e.g., "3,7,9")"""
]

def prompt_dyck_language_reasoning_errors(dyck_sequence: str, thoughts: List[str], n_types: int = 3, is_chinese: bool = None):
    """
    生成Dyck语言推理错误识别游戏的提示语
    
    @param dyck_sequence: 括号序列
    @param thoughts: 思考步骤列表
    @param n_types: 括号种类数量
    @param is_chinese: 是否使用中文提示，如果为None则随机选择
    @return: 格式化后的提示语
    """
    # 如果未指定语言，随机选择
    if is_chinese is None:
        is_chinese = random.choice([True, False])
    
    # 选择提示模板
    if is_chinese:
        prompt_template = random.choice(chinese_prompt_candidates)
    else:
        prompt_template = random.choice(english_prompt_candidates)
    
    # 格式化思考步骤
    thoughts_text = "\n".join(thoughts)
    
    # 替换模板中的占位符，不使用format方法
    prompt = prompt_template.replace("{dyck_sequence}", dyck_sequence)
    prompt = prompt.replace("{thoughts}", thoughts_text)
    
    return prompt 