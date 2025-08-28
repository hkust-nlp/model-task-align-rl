# 英文提示词模板
en_prompts = [
    """
    You are given a tracking problem about a collection of items with various properties. 
    {context}
    
    Based on the description above, answer the following question:
    {problem}
    
    Provide step-by-step reasoning to track all changes to the collection and count the items satisfying the specified conditions.
    
    Place your final answer in a LaTeX box like this: \\boxed{{answer}}. Only include the box with your answer, no additional explanation.
    """,
    
    """
    I need help solving a logical reasoning problem about objects with multiple attributes.
    {context}
    
    Please answer this question:
    {problem}
    
    Explain your process carefully, tracking each change to the collection before calculating the final answer.
    
    After your explanation, write only your final answer in a LaTeX box like this: \\boxed{{answer}}
    """,
    
    # ... 其余英文提示词 ...
    """
    This is a reasoning challenge about tracking objects through a series of transformations.
    {context}
    
    Now I need to know:
    {problem}
    
    Walk through each step of the transformations, tracking how the collection changes, then calculate your answer.
    
    Conclude with your final answer in this format: \\boxed{{answer}}
    """,
    
    """
    You are a logical reasoning assistant. I have a problem about tracking object properties through various changes.
    {context}
    
    Question:
    {problem}
    
    Break down your approach: First identify the initial collection, then track each transformation carefully, and finally count the objects meeting the criteria.
    
    Your final answer should be presented in a LaTeX box like this: \\boxed{{answer}}
    """,
    
    """
    In this object tracking puzzle, you need to follow the changes to a collection of items with different properties.
    {context}
    
    Based on the above scenario, answer:
    {problem}
    
    To solve this, create a table or list to track the items and their properties after each transformation, then count the items meeting the specified conditions.
    
    Put your final numerical answer in a LaTeX box: \\boxed{{answer}}
    """,
    
    """
    I'm working on a logical problem that requires precise tracking of objects and their changing attributes.
    {context}
    
    My question is:
    {problem}
    
    Please help by methodically following each transformation and keeping track of all items and their properties before determining the final count.
    
    Express your final answer using the LaTeX notation: \\boxed{{answer}}
    """,
    
    """
    This is an attribute tracking problem where items undergo several transformations.
    {context}
    
    Please determine:
    {problem}
    
    Approach this by tracking the specific properties of each item after every change, then identify which items satisfy the given criteria.
    
    Provide your answer in this format: \\boxed{{answer}}
    """,
    
    """
    Below is a problem about tracking objects with multiple properties through a series of transformations.
    {context}
    
    The question is:
    {problem}
    
    Start by listing the initial items and their properties, then apply each transformation step-by-step before counting the items that match the required attributes.
    
    End your solution with: \\boxed{{answer}}
    """,
    
    """
    I need your help with a logical reasoning challenge involving a changing collection of items.
    {context}
    
    Please answer:
    {problem}
    
    For accuracy, consider creating a representation of each item and its properties, updating it with each transformation, then checking which items meet the specified criteria.
    
    Give your final answer as: \\boxed{{answer}}
    """,
    
    """
    In this object property tracking exercise, you'll need to follow how items and their attributes change over time.
    {context}
    
    Your task:
    {problem}
    
    Solve this methodically by tracking each item individually through all transformations, noting how properties change, then identifying which items have the requested attributes.
    
    Format your final answer as: \\boxed{{answer}}
    """
]

# 中文提示词模板
cn_prompts = [
    """
    这是一个关于物品属性跟踪的问题，涉及多个具有不同特征的物品。
    {context}
    
    根据上述描述，请回答以下问题：
    {problem}
    
    请提供逐步推理过程，跟踪收藏品的所有变化，并计算满足指定条件的物品数量。
    
    将你的最终答案放在LaTeX框中，格式如下：\\boxed{{answer}}。只需给出带有答案的框，不需要额外解释。
    """,
    
    """
    我需要帮助解决一个关于多属性物品的逻辑推理问题。
    {context}
    
    请回答这个问题：
    {problem}
    
    请仔细解释你的思考过程，跟踪收藏品的每次变化，然后计算最终答案。
    
    在你的解释之后，仅将最终答案写在LaTeX框中，如：\\boxed{{answer}}
    """,
    
    # ... 其余中文提示词 ...
    """
    这是一个关于通过一系列转变跟踪物品的推理挑战。
    {context}
    
    现在我需要知道：
    {problem}
    
    请逐步分析每次转变，跟踪收藏品的变化，然后计算你的答案。
    
    以这种格式给出你的最终答案：\\boxed{{answer}}
    """,
    
    """
    你是一个逻辑推理助手。我有一个关于跟踪物品属性在各种变化中的问题。
    {context}
    
    问题：
    {problem}
    
    请拆解你的方法：首先确定初始收藏，然后仔细跟踪每次转变，最后计算满足条件的物品数量。
    
    你的最终答案应该用LaTeX框表示，如：\\boxed{{answer}}
    """,
    
    """
    在这个物品跟踪谜题中，你需要跟随具有不同属性的物品收藏的变化。
    {context}
    
    基于上述情景，回答：
    {problem}
    
    要解决这个问题，请创建一个表格或列表来跟踪每次转变后的物品及其属性，然后计算满足指定条件的物品数量。
    
    将你的最终数值答案放在LaTeX框中：\\boxed{{answer}}
    """,
    
    """
    我正在研究一个需要精确跟踪物品及其变化属性的逻辑问题。
    {context}
    
    我的问题是：
    {problem}
    
    请通过有条理地跟踪每次转变并记录所有物品及其属性，然后确定最终数量来帮助我。
    
    使用LaTeX符号表达你的最终答案：\\boxed{{answer}}
    """,
    
    """
    这是一个属性跟踪问题，其中物品经历了几次转变。
    {context}
    
    请确定：
    {problem}
    
    解决这个问题的方法是跟踪每次变化后每个物品的特定属性，然后识别哪些物品满足给定条件。
    
    按照这种格式提供你的答案：\\boxed{{answer}}
    """,
    
    """
    以下是一个关于通过一系列转变跟踪具有多种属性的物品的问题。
    {context}
    
    问题是：
    {problem}
    
    首先列出初始物品及其属性，然后逐步应用每次转变，最后计算匹配所需属性的物品数量。
    
    以下面的方式结束你的解答：\\boxed{{answer}}
    """,
    
    """
    我需要你帮助解决一个涉及变化物品收藏的逻辑推理挑战。
    {context}
    
    请回答：
    {problem}
    
    为了准确性，考虑创建每个物品及其属性的表示，随着每次转变更新它，然后检查哪些物品满足指定条件。
    
    将你的最终答案表示为：\\boxed{{answer}}
    """,
    
    """
    在这个物品属性跟踪练习中，你需要跟随物品及其属性如何随时间变化。
    {context}
    
    你的任务：
    {problem}
    
    通过有条理地跟踪每个物品通过所有转变的过程，记录属性如何变化，然后确定哪些物品具有所请求的属性来解决这个问题。
    
    将你的最终答案格式化为：\\boxed{{answer}}
    """
]
