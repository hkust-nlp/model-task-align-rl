import random

prompt_candidates = {
    # 英文提示
    "You are given a {n} x {m} grid representing a Kukurasu puzzle. In this puzzle, you need to place 1s in the grid so that the weighted sum of each row and column matches the given constraints. The row sums are {row_sums} and the column sums are {col_sums}.\n1. Rules:\n  1. Each cell can contain either a 1 or an X.\n  2. The weight of a 1 in a row is its column position (1 to {m}).\n  3. The weight of a 1 in a column is its row position (1 to {n}).\n  4. The weighted sum of each row must match the corresponding row constraint.\n  5. The weighted sum of each column must match the corresponding column constraint.\n2. Input:\n{puzzle}": "en",
    
    "This is a {n} x {m} Kukurasu puzzle grid. Your task is to fill in the grid with 1s and Xs such that the weighted sums match the given constraints. The row sums are {row_sums} and the column sums are {col_sums}.\n1. Rules:\n  1. Each cell must contain either a 1 or an X.\n  2. In each row, a 1 in position j contributes j points to that row's sum (positions are 1-indexed).\n  3. In each column, a 1 in position i contributes i points to that column's sum (positions are 1-indexed).\n  4. The weighted sum of each row must equal its constraint value.\n  5. The weighted sum of each column must equal its constraint value.\n2. Input:\n{puzzle}": "en",
    
    "You're presented with a {n} x {m} Kukurasu puzzle grid. The goal is to place 1s in the grid so that the weighted sums of rows and columns match the given constraints: row sums {row_sums} and column sums {col_sums}.\n1. Rules:\n  1. Each cell must be filled with either a 1 or an X.\n  2. A 1 in column j of any row contributes j points to that row's sum (j ranges from 1 to {m}).\n  3. A 1 in row i of any column contributes i points to that column's sum (i ranges from 1 to {n}).\n  4. Each row's weighted sum must match its constraint value.\n  5. Each column's weighted sum must match its constraint value.\n2. Input:\n{puzzle}": "en",
    
    "Below is a {n} x {m} Kukurasu puzzle grid. Your objective is to place 1s in the grid such that the weighted sums of rows and columns match the given constraints. Row sums: {row_sums}. Column sums: {col_sums}.\n1. Rules:\n  1. Each cell must contain either a 1 or an X.\n  2. The weight of a 1 in a row equals its column number (1 to {m}).\n  3. The weight of a 1 in a column equals its row number (1 to {n}).\n  4. The sum of weighted 1s in each row must equal the row constraint.\n  5. The sum of weighted 1s in each column must equal the column constraint.\n2. Input:\n{puzzle}": "en",
    
    "Here's a {n} x {m} Kukurasu logic puzzle. You need to place 1s in the grid so that the weighted sums match the constraints. Row sums: {row_sums}. Column sums: {col_sums}.\n1. Rules:\n  1. Each cell can be filled with either a 1 or an X.\n  2. A 1 in the jth position of a row contributes j points to that row's sum.\n  3. A 1 in the ith position of a column contributes i points to that column's sum.\n  4. The weighted sum of each row must equal its constraint value.\n  5. The weighted sum of each column must equal its constraint value.\n2. Input:\n{puzzle}": "en",
    
    "I'm presenting you with a {n} x {m} Kukurasu puzzle. Your task is to place 1s in the grid so that the weighted sums match the given constraints: row sums {row_sums} and column sums {col_sums}.\n1. Rules:\n  1. Each cell must be filled with either a 1 or an X.\n  2. In each row, a 1 in position j has a weight of j (where j ranges from 1 to {m}).\n  3. In each column, a 1 in position i has a weight of i (where i ranges from 1 to {n}).\n  4. The weighted sum of each row must match its constraint.\n  5. The weighted sum of each column must match its constraint.\n2. Input:\n{puzzle}": "en",
    
    "Consider this {n} x {m} Kukurasu puzzle grid. You need to place 1s in the grid such that the weighted sums match the constraints. Row sums: {row_sums}. Column sums: {col_sums}.\n1. Rules:\n  1. Each cell must contain either a 1 or an X.\n  2. A 1 in column position j contributes j points to its row's sum.\n  3. A 1 in row position i contributes i points to its column's sum.\n  4. Each row's weighted sum must equal its constraint value.\n  5. Each column's weighted sum must equal its constraint value.\n2. Input:\n{puzzle}": "en",
    
    "You have a {n} x {m} Kukurasu puzzle grid. Your goal is to place 1s in the grid so that the weighted sums match the given constraints: row sums {row_sums} and column sums {col_sums}.\n1. Rules:\n  1. Each cell must be filled with either a 1 or an X.\n  2. The weight of a 1 in a row is its column position (1 to {m}).\n  3. The weight of a 1 in a column is its row position (1 to {n}).\n  4. The weighted sum of each row must match its constraint.\n  5. The weighted sum of each column must match its constraint.\n2. Input:\n{puzzle}": "en",
    
    "This {n} x {m} grid represents a Kukurasu puzzle. Your task is to place 1s in the grid so that the weighted sums match the constraints. Row sums: {row_sums}. Column sums: {col_sums}.\n1. Rules:\n  1. Each cell must contain either a 1 or an X.\n  2. A 1 in the jth position of a row contributes j points to that row's sum.\n  3. A 1 in the ith position of a column contributes i points to that column's sum.\n  4. The weighted sum of each row must equal its constraint value.\n  5. The weighted sum of each column must equal its constraint value.\n2. Input:\n{puzzle}": "en",
    
    "Examine this {n} x {m} Kukurasu puzzle grid. Your objective is to place 1s in the grid such that the weighted sums match the given constraints: row sums {row_sums} and column sums {col_sums}.\n1. Rules:\n  1. Each cell must be filled with either a 1 or an X.\n  2. The weight of a 1 in a row equals its column number (1 to {m}).\n  3. The weight of a 1 in a column equals its row number (1 to {n}).\n  4. The weighted sum of each row must match its constraint.\n  5. The weighted sum of each column must match its constraint.\n2. Input:\n{puzzle}": "en",
    
    # 中文提示
    "这是一个 {n} x {m} 的 Kukurasu 谜题。在这个谜题中，你需要在网格中放置数字1，使每行和每列的加权和匹配给定的约束。行和为 {row_sums}，列和为 {col_sums}。\n1. 规则：\n  1. 每个单元格可以包含数字1或X。\n  2. 行中数字1的权重是其列位置（1到{m}）。\n  3. 列中数字1的权重是其行位置（1到{n}）。\n  4. 每行的加权和必须匹配对应的行约束。\n  5. 每列的加权和必须匹配对应的列约束。\n2. 输入：\n{puzzle}": "zh",
    
    "给你一个 {n} x {m} 的 Kukurasu 谜题网格。你的任务是在网格中填入数字1和X，使加权和匹配给定的约束。行和为 {row_sums}，列和为 {col_sums}。\n1. 规则：\n  1. 每个单元格必须包含数字1或X。\n  2. 在每行中，位置j的数字1贡献j分（位置从1开始编号）。\n  3. 在每列中，位置i的数字1贡献i分（位置从1开始编号）。\n  4. 每行的加权和必须等于其约束值。\n  5. 每列的加权和必须等于其约束值。\n2. 输入：\n{puzzle}": "zh",
    
    "这是一个 {n} x {m} 的 Kukurasu 逻辑谜题。目标是在网格中放置数字1，使行和列的加权和匹配给定的约束：行和 {row_sums} 和列和 {col_sums}。\n1. 规则：\n  1. 每个单元格必须填入数字1或X。\n  2. 任何行中列位置j的数字1贡献j分（j从1到{m}）。\n  3. 任何列中行位置i的数字1贡献i分（i从1到{n}）。\n  4. 每行的加权和必须匹配其约束值。\n  5. 每列的加权和必须匹配其约束值。\n2. 输入：\n{puzzle}": "zh",
    
    "下面是一个 {n} x {m} 的 Kukurasu 谜题网格。你的目标是在网格中放置数字1，使行和列的加权和匹配给定的约束。行和：{row_sums}。列和：{col_sums}。\n1. 规则：\n  1. 每个单元格必须包含数字1或X。\n  2. 行中数字1的权重等于其列号（1到{m}）。\n  3. 列中数字1的权重等于其行号（1到{n}）。\n  4. 每行加权1的总和必须等于行约束。\n  5. 每列加权1的总和必须等于列约束。\n2. 输入：\n{puzzle}": "zh",
    
    "这是一个 {n} x {m} 的 Kukurasu 谜题。你需要在网格中放置数字1，使加权和匹配约束。行和：{row_sums}。列和：{col_sums}。\n1. 规则：\n  1. 每个单元格可以填入数字1或X。\n  2. 行中第j个位置的数字1贡献j分。\n  3. 列中第i个位置的数字1贡献i分。\n  4. 每行的加权和必须等于其约束值。\n  5. 每列的加权和必须等于其约束值。\n2. 输入：\n{puzzle}": "zh",
    
    "我给你一个 {n} x {m} 的 Kukurasu 谜题。你的任务是在网格中放置数字1，使加权和匹配给定的约束：行和 {row_sums} 和列和 {col_sums}。\n1. 规则：\n  1. 每个单元格必须填入数字1或X。\n  2. 在每行中，位置j的数字1权重为j（j从1到{m}）。\n  3. 在每列中，位置i的数字1权重为i（i从1到{n}）。\n  4. 每行的加权和必须匹配其约束。\n  5. 每列的加权和必须匹配其约束。\n2. 输入：\n{puzzle}": "zh",
    
    "考虑这个 {n} x {m} 的 Kukurasu 谜题网格。你需要在网格中放置数字1，使加权和匹配约束。行和：{row_sums}。列和：{col_sums}。\n1. 规则：\n  1. 每个单元格必须包含数字1或X。\n  2. 列位置j的数字1为其行贡献j分。\n  3. 行位置i的数字1为其列贡献i分。\n  4. 每行的加权和必须等于其约束值。\n  5. 每列的加权和必须等于其约束值。\n2. 输入：\n{puzzle}": "zh",
    
    "你有一个 {n} x {m} 的 Kukurasu 谜题网格。你的目标是在网格中放置数字1，使加权和匹配给定的约束：行和 {row_sums} 和列和 {col_sums}。\n1. 规则：\n  1. 每个单元格必须填入数字1或X。\n  2. 行中数字1的权重是其列位置（1到{m}）。\n  3. 列中数字1的权重是其行位置（1到{n}）。\n  4. 每行的加权和必须匹配其约束。\n  5. 每列的加权和必须匹配其约束。\n2. 输入：\n{puzzle}": "zh",
    
    "这个 {n} x {m} 的网格代表一个 Kukurasu 谜题。你的任务是在网格中放置数字1，使加权和匹配约束。行和：{row_sums}。列和：{col_sums}。\n1. 规则：\n  1. 每个单元格必须包含数字1或X。\n  2. 行中第j个位置的数字1贡献j分。\n  3. 列中第i个位置的数字1贡献i分。\n  4. 每行的加权和必须等于其约束值。\n  5. 每列的加权和必须等于其约束值。\n2. 输入：\n{puzzle}": "zh",
    
    "请分析这个 {n} x {m} 的 Kukurasu 谜题网格。你的目标是在网格中放置数字1，使加权和匹配给定的约束：行和 {row_sums} 和列和 {col_sums}。\n1. 规则：\n  1. 每个单元格必须填入数字1或X。\n  2. 行中数字1的权重等于其列号（1到{m}）。\n  3. 列中数字1的权重等于其行号（1到{n}）。\n  4. 每行的加权和必须匹配其约束。\n  5. 每列的加权和必须匹配其约束。\n2. 输入：\n{puzzle}": "zh"
}


def prompt_kukurasu(grid, row_sums, col_sums):
    """Generate a prompt for the Kukurasu puzzle.
    
    Args:
        grid: A 2D list representing the initial grid (filled with "X")
        row_sums: A list of integers representing the target sum for each row
        col_sums: A list of integers representing the target sum for each column
    
    Returns:
        A string prompt for the puzzle
    """
    n = len(grid)
    m = len(grid[0]) if n > 0 else 0
    
    # 创建网格的字符串表示
    grid_str = ""
    for i, row in enumerate(grid):
        grid_str += "  ".join(row) + "  |  " + str(row_sums[i]) + "\n"
    
    grid_str += "-" * (m * 3 + 5) + "\n"
    
    for col_sum in col_sums:
        grid_str += f"{col_sum:2d} "
    
    # 创建完整的谜题描述
    puzzle = f"{grid_str}\n\nrow_sums = {row_sums}\ncol_sums = {col_sums}"
    
    # 选择一个随机提示模板
    prompt = random.choice(list(prompt_candidates.keys()))
    language = prompt_candidates[prompt]
    
    # 格式化提示
    prompt = prompt.format(n=n, m=m, row_sums=row_sums, col_sums=col_sums, puzzle=puzzle)
    
    # 添加任务说明和输出格式
    if language == "en":
        prompt += "\n3. Task:\n  - Place 1s in the grid following all rules.\n  - Replace Xs with 1s where appropriate, keeping Xs in other positions.\n  - Output the completed grid as a 2D list (list of lists) in Python format.\n4. Example Output Format:\n[\n    [\"X\", \"1\", \"X\", \"X\"],\n    [\"1\", \"X\", \"1\", \"X\"],\n    [\"X\", \"1\", \"X\", \"1\"],\n    [\"1\", \"X\", \"X\", \"1\"]\n]"
    elif language == "zh":
        prompt += "\n3. 任务：\n  - 按照所有规则在网格中放置数字1。\n  - 在适当的位置用1替换X，在其他位置保留X。\n  - 以Python格式的二维列表(列表的列表)输出完成的网格。\n4. 示例输出格式：\n[\n    [\"X\", \"1\", \"X\", \"X\"],\n    [\"1\", \"X\", \"1\", \"X\"],\n    [\"X\", \"1\", \"X\", \"1\"],\n    [\"1\", \"X\", \"X\", \"1\"]\n]"
    
    return prompt