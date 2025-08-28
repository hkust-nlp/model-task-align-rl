import random

# 中文提示模板
chinese_prompt_candidates = [
    "你是一个括号语言专家，需要分析括号的闭合情况。\n这是一个包含{n_types}种括号的序列，总长度为{total_length}。请找出第一个出现错误的位置（以1为起始索引）。错误定义为第一个破坏预期模式的字符，例如未匹配的括号或顺序错误的括号。如果序列中间没有错误，但末尾存在未闭合的括号，则错误位置被认为是在序列末尾的下一个位置。\n如果序列合法，请直接输出-1；如果不合法，请直接输出第一个错误的位置编号。\n\n括号序列：{bracket_string}",
    "你是一个编程语言解析器，需要找出括号不匹配的位置。\n下面是一个由{n_types}种不同类型括号组成的序列，总长度为{total_length}。你的任务是找出第一个不匹配的括号位置（位置从1开始计数）。特别注意：如果序列前面的所有括号都正确匹配，但末尾有未闭合的括号，错误位置将被视为序列长度加1。\n如果序列完全合法（所有括号都正确配对），则直接输出-1；否则，直接输出第一个错误出现的位置（数字）。\n\n括号序列：{bracket_string}",
    "作为括号匹配专家，请分析以下序列中的第一个错误。\n给定一个长度为{total_length}、包含{n_types}种不同类型括号的序列。你需要指出序列中第一个括号错误的位置（位置从1开始）。错误可能是未匹配的括号、错误的闭合顺序等。如果序列中间没有匹配错误，但末尾存在未闭合的括号，错误位置将是序列长度加1。\n如果序列完全合法，请直接输出-1；如果不合法，请直接输出第一个错误的位置数字。\n\n序列：{bracket_string}",
    "你现在需要检查一个括号序列是否合法。\n这个序列包含{n_types}种括号类型，总长度为{total_length}。请判断这个序列中最早出现错误的位置（从1开始计数）。注意：只有当序列前面的括号都正确匹配时，对于末尾的未闭合括号，我们才将其错误位置定义为序列长度加1。\n如果序列合法（所有括号都正确闭合），直接输出-1；如果有错误，直接输出第一个错误的位置数字。\n\n括号序列：{bracket_string}",
    "你需要对括号配对进行分析。\n给你一个由{n_types}种不同括号组成的序列，总长度为{total_length}。请找出序列中第一个出错的位置（以1为索引起始）。出错位置定义为第一个破坏有效括号配对规则的字符所在位置。特别说明：如果序列中间的所有括号都正确匹配，但在末尾有未闭合的括号，错误位置将是序列长度加1。\n如果整个序列是有效的括号配对，则直接输出-1；否则直接输出第一个错误字符的位置数字。\n\n序列：{bracket_string}"
]

# 英文提示模板
english_prompt_candidates = [
    "You are an expert in a language called dyck where you must complete the language sequence of unclosed brackets of all types (e.g., [], {}, <>).\nYou are given a Dyck language sequence with a total length of {total_length}. Identify the first position in the string where an error occurs. The error is defined as the first character that breaks the expected pattern, such as an unmatched or incorrectly ordered bracket. Only if there are no matching errors in the middle of the sequence but there are unclosed brackets at the end, the error position is considered to be the sequence length plus 1.\nIf the string is valid, output -1; if it's invalid, output only the number of the first error position.\n\nBracket sequence: {bracket_string}",
    "As a bracket matching analyzer, examine the following bracket sequence that contains {n_types} different types of brackets with a total length of {total_length}.\nYour task is to find the very first position where a bracket error occurs (using 1-based indexing). An error could be an unmatched bracket or incorrectly ordered closing bracket. Note: For unclosed brackets at the end of the sequence, the error position is sequence length plus 1, but only if all brackets before are correctly matched.\nIf the sequence is valid (all brackets properly matched), output -1; otherwise, output only the position number of the first error.\n\nSequence: {bracket_string}",
    "You are working with a sequence of brackets from {n_types} different types, with a total length of {total_length}.\nYour job is to identify the first position where an error in bracket matching occurs. Positions are counted starting from 1. Important: If all brackets in the sequence are correctly matched except for unclosed brackets at the end, the error position will be the sequence length plus 1.\nIf the sequence is completely valid with all brackets properly matched and closed, output -1; otherwise, output only the number of the first error position.\n\nBracket sequence: {bracket_string}",
    "As a parser specialist, analyze this bracket sequence containing {n_types} types of brackets with a total length of {total_length}.\nFind the first position where the bracket matching rule is violated (use 1-based indexing). Valid bracket sequences have all opening brackets matched with their corresponding closing brackets in the correct order. For any unclosed brackets at the end, the error position is defined as sequence length plus 1, but this only applies if there are no matching errors earlier in the sequence.\nIf the sequence follows all rules, output -1; otherwise, output only the number of the first error position.\n\nSequence to analyze: {bracket_string}",
    "You are evaluating a bracket sequence with {n_types} different bracket types and a total length of {total_length}.\nDetermine if the sequence is valid according to bracket matching rules, where each opening bracket must have a matching closing bracket in the correct order. If the sequence is invalid, identify the position of the first error (using 1-based indexing). Special note: The error position for unclosed brackets at the end of the sequence is the sequence length plus 1, but this rule only applies when all previous brackets are correctly matched.\nIf valid, output -1; if invalid, output only the number of the first error position.\n\nBracket sequence: {bracket_string}"
]

def prompt_dyck_language_errors(bracket_string: str, n_types: int, total_length: int, is_chinese: bool = False):
    """
    生成括号闭合错误识别游戏的提示语
    
    @param bracket_string: 括号序列字符串
    @param n_types: 括号种类数量
    @param total_length: 序列总长度
    @param is_chinese: 是否生成中文提示
    @return: 格式化后的提示语
    """
    if is_chinese:
        prompt = random.choice(chinese_prompt_candidates)
    else:
        prompt = random.choice(english_prompt_candidates)
    
    # 填充参数
    prompt = prompt.format(
        bracket_string=bracket_string, 
        n_types=n_types, 
        total_length=total_length
    )
    return prompt 