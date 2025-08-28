english_prompts = [
    """You are an expert in word sorting. You will be provided with a list of words and the steps taken, in the form of thoughts, to arrange them in alphabetical order.
Your job is to identify the first step that was a mistake in reasoning about the order of the words. This can be misunderstanding the order of
the alphabet or getting the word or letter wrong or missing a word entirely.
Q: Sort the following words alphabetically: List: {word_list}
{think_process}
Is there a mistake in this sequence? Write \\boxed{{No}} if there are no mistakes, or the number N if there is a mistake in Thought N, write the answer in \\boxed{{}}.""",
    
    """As a word sorting expert, you need to analyze a sequence of thoughts describing how a list of words was alphabetically sorted.
Your task is to find the first error in the reasoning process, if any exists. Errors could include misunderstanding alphabetical order,
misreading words or letters, or completely overlooking words.
Question: Put these words in alphabetical order: {word_list}
{think_process}
Identify the first mistake in the sequence above. Answer \\boxed{{No}} if everything is correct, or write the number of the thought containing the first error in \\boxed{{}}.""",
    
    """You're a specialist in alphabetical ordering. Below is a list of words followed by a step-by-step thought process of sorting them.
Examine this process carefully and identify the first step where an error was made. Mistakes might involve incorrect understanding of
alphabetical sequence, incorrect word or letter identification, or omission of words.
Problem: Sort alphabetically: {word_list}
{think_process}
Did you find any mistakes? If all steps are correct, write \\boxed{{No}}. Otherwise, indicate the number of the first incorrect thought in \\boxed{{}}.""",
    
    """As a lexicographical sorting expert, your challenge is to detect errors in the sorting process described below.
You must identify the first step where a mistake occurs in the reasoning about arranging words alphabetically.
This could be an error in understanding alphabet order, misreading words, or overlooking entries.
Task: Alphabetize the following words: {word_list}
{think_process}
Is there an error in the above procedure? Write \\boxed{{No}} if all steps are correct, or provide the number of the first flawed thought in \\boxed{{}}.""",
    
    """You have expertise in alphabetical arrangement. You will review a detailed thought process of sorting words alphabetically.
Your goal is to identify the first incorrect step in this process, if any exists. Errors may include misunderstanding of
alphabetical order, misreading of words or letters, or completely missing words.
Exercise: Arrange these words alphabetically: {word_list}
{think_process}
Identify any mistakes in the sequence. Write \\boxed{{No}} if everything is correct, or the number of the first erroneous thought in \\boxed{{}}.""",
    
    """With your knowledge of word sorting, analyze the thinking steps below that attempt to arrange words in alphabetical order.
Find the first point where an error occurs in the reasoning. Such errors might involve incorrect understanding of alphabetical sequence,
misidentification of letters or words, or omission of entries.
Instructions: Sort these words alphabetically: {word_list}
{think_process}
Is there a mistake in this thinking process? If not, write \\boxed{{No}}. If yes, indicate the number of the first incorrect step in \\boxed{{}}.""",
    
    """As an alphabetical ordering specialist, your assignment is to identify errors in the following sorting process.
You need to find the first step where a mistake was made in reasoning about the order of words.
Mistakes could include wrong understanding of alphabet sequence, incorrect reading of words or letters, or missing words.
Question: Arrange these words in alphabetical order: {word_list}
{think_process}
Did you spot any errors in the sequence? Write \\boxed{{No}} if all is correct, or the number of the first problematic thought in \\boxed{{}}.""",
    
    """Using your expertise in word arrangement, review the following thought process for sorting words alphabetically.
Your task is to detect the first point where an error occurs in the reasoning, if any. This could be a misunderstanding of
alphabetical order, incorrect identification of words or letters, or overlooking certain words.
Task: Put these words in alphabetical sequence: {word_list}
{think_process}
Is there a mistake in this sequence? If all steps are correct, write \\boxed{{No}}. Otherwise, indicate the number of the first flawed thought in \\boxed{{}}.""",
    
    """As a lexicographical expert, examine the following thought process of alphabetically arranging a list of words.
Your challenge is to identify the first step where a mistake was made. Potential errors include incorrect understanding of
the alphabet, misreading words or letters, or completely overlooking certain words.
Problem: Sort these words into alphabetical order: {word_list}
{think_process}
Identify any errors in the sequence. Write \\boxed{{No}} if all thoughts are correct, or the number of the first incorrect thought in \\boxed{{}}.""",
    
    """With your knowledge of alphabetical ordering, analyze the reasoning below that attempts to sort words in alphabetical order.
Your job is to find the first error in this process, if any exists. Such errors could include misunderstanding of alphabet sequence,
mistakes in identifying letters or words, or missing entries entirely.
Exercise: Arrange alphabetically: {word_list}
{think_process}
Is there a mistake in this reasoning? If all steps are correct, write \\boxed{{No}}. Otherwise, write the number of the first incorrect thought in \\boxed{{}}."""
]

chinese_prompts = [
    """你是单词排序专家。你将获得一个单词列表和以思考形式呈现的排序步骤，这些步骤试图将单词按字母顺序排列。
你的任务是找出推理过程中的第一个错误步骤。错误可能包括对字母表顺序的误解、错误理解单词或字母，或完全遗漏某个单词。
问题：按字母顺序排列以下单词：{word_list}
{think_process}
这个排序过程中有错误吗？如果没有错误，请写\\boxed{{No}}；如果有错误，请写出第一个错误出现在第几个思考步骤，将答案写在\\boxed{{}}中。""",

    """作为字母排序专家，你需要分析一系列思考步骤，这些步骤描述了如何将单词按字母顺序排列。
你的任务是找出推理过程中的第一个错误，如果存在的话。错误可能包括对字母顺序的误解、错误读取单词或字母，或者完全忽略某些单词。
问题：将这些单词按字母顺序排列：{word_list}
{think_process}
上述排序过程中有错误吗？如果全部正确，请写\\boxed{{No}}；如果有错误，请在\\boxed{{}}中写出第一个错误思考的编号。""",
    
    """你是词语排序专家。下面是一个单词列表，以及将它们按字母顺序排列的详细思考过程。
仔细检查这个过程，找出第一个出错的步骤，如果有的话。错误可能涉及对字母表顺序的错误理解、不正确识别单词或字母，或者遗漏单词。
任务：按字母顺序排序：{word_list}
{think_process}
你发现错误了吗？如果所有步骤都正确，请写\\boxed{{No}}；否则，请在\\boxed{{}}中指出第一个不正确思考的编号。""",
    
    """作为词典排序专家，你的挑战是检测下面描述的排序过程中的错误。
你必须找出在关于按字母顺序排列单词的推理中第一个出错的步骤。
这可能是理解字母顺序的错误、错误读取单词，或忽略条目。
问题：将以下单词按字母顺序排列：{word_list}
{think_process}
上述过程中有错误吗？如果所有步骤都正确，请写\\boxed{{No}}；或者在\\boxed{{}}中提供第一个有缺陷思考的编号。""",
    
    """你在字母排序方面有专业知识。你将审查一个将单词按字母顺序排列的详细思考过程。
你的目标是找出这个过程中的第一个不正确步骤，如果存在的话。错误可能包括对字母顺序的误解、错误读取单词或字母，或者完全遗漏单词。
练习：按字母顺序排列这些单词：{word_list}
{think_process}
识别序列中的任何错误。如果一切正确，请写\\boxed{{No}}；或者在\\boxed{{}}中写出第一个错误思考的编号。""",
    
    """凭借你对单词排序的了解，分析下面试图按字母顺序排列单词的思考步骤。
找出推理中第一个出错的点。这些错误可能涉及对字母顺序的错误理解、对字母或单词的错误识别，或者遗漏条目。
指示：按字母顺序排序这些单词：{word_list}
{think_process}
这个思考过程中有错误吗？如果没有，请写\\boxed{{No}}；如果有，请在\\boxed{{}}中指出第一个不正确步骤的编号。""",
    
    """作为字母排序专家，你的任务是识别以下排序过程中的错误。
你需要找出在关于单词顺序推理中第一个出错的步骤。
错误可能包括对字母表顺序的错误理解、不正确读取单词或字母，或遗漏单词。
问题：按字母顺序排列这些单词：{word_list}
{think_process}
你发现序列中有错误吗？如果全部正确，请写\\boxed{{No}}；或者在\\boxed{{}}中写出第一个有问题思考的编号。""",
    
    """使用你在单词排列方面的专业知识，审查以下按字母顺序排序单词的思考过程。
你的任务是检测推理中第一个出错的点，如果有的话。这可能是对字母顺序的误解、对单词或字母的错误识别，或者忽略某些单词。
任务：将这些单词按字母顺序排列：{word_list}
{think_process}
这个序列中有错误吗？如果所有步骤都正确，请写\\boxed{{No}}；否则，请在\\boxed{{}}中指出第一个有缺陷思考的编号。""",
    
    """作为词典专家，审查以下按字母顺序排列单词列表的思考过程。
你的挑战是找出第一个出错的步骤。潜在错误包括对字母表的错误理解、错误读取单词或字母，或者完全忽略某些单词。
问题：将这些单词按字母顺序排序：{word_list}
{think_process}
识别序列中的任何错误。如果所有思考都正确，请写\\boxed{{No}}；或者在\\boxed{{}}中写出第一个不正确思考的编号。""",
    
    """凭借你对字母排序的了解，分析下面试图按字母顺序排序单词的推理。
你的工作是找出这个过程中的第一个错误，如果存在的话。这些错误可能包括对字母表顺序的误解、在识别字母或单词时的错误，或者完全遗漏条目。
练习：按字母顺序排列：{word_list}
{think_process}
这个推理中有错误吗？如果所有步骤都正确，请写\\boxed{{No}}；否则，请在\\boxed{{}}中写出第一个不正确思考的编号。"""
]
