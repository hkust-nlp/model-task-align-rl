#!/usr/bin/env python3
"""
Prompts for Wordscapes puzzles.
These prompts are used to generate questions for the LLM to solve.
"""

def format_grid(grid):
    """Format the grid as a string with proper spacing."""
    grid_str = []
    for row in grid:
        grid_str.append("        ".join(row))
    return "\n".join(grid_str)

def prompt_wordscapes(across_words, down_words, grid):
    """Generate a prompt for a wordscapes puzzle."""
    grid_str = format_grid(grid)
    across_str = ",".join(across_words)
    down_str = ",".join(down_words)
    
    # Get a random prompt variant
    prompt_idx = hash(str(across_words) + str(down_words)) % len(english_prompts)
    return english_prompts[prompt_idx].format(
        across=across_str,
        down=down_str,
        grid=grid_str
    )

# English prompt templates with clearer instructions
english_prompts = [
    "Fill in this crossword puzzle grid where:\nacross words: {across}\ndown words: {down}\nGrid layout (X=letter position to fill in, 0=empty cell):\n{grid}\n\nINSTRUCTIONS:\n1. Replace each X with a letter to form the given words.\n2. Keep 0 positions as empty spaces in your answer.\n3. Place across words from left to right and down words from top to bottom.\n4. Format your answer as a 2D grid wrapped in double square brackets: [[...]].\n5. Each row should be separated by commas.\n6. All letters in each row should have spaces between them.\n7. For example: [[\"A\", \"B\", \"C\"], [\"D\", \"E\", \"F\"], [\"G\", \"H\", \"I\"]]\n\nYour answer should look like [[row1 with letters in quotes, row2 with letters in quotes, ...]].",
    
    "Complete this crossword puzzle:\nacross words: {across}\ndown words: {down}\nPuzzle grid (X=letter position, 0=empty):\n{grid}\n\nINSTRUCTIONS:\n1. Fill in the grid with the given words.\n2. Across words read from left to right, down words read from top to bottom.\n3. Where words intersect, they must share the same letter.\n4. Format your answer as [[row1, row2, ...]] with letters in quotes.\n5. Replace Xs with appropriate letters to form the given words.\n6. Keep 0 positions as spaces (represented as \" \") in your answer.\n7. Example format: [[\"C\", \"A\", \"T\"], [\" \", \" \", \" \"], [\"D\", \"O\", \"G\"]]\n\nProvide your completed grid in the format [[row1, row2, ...]].",
    
    "Solve this wordscapes puzzle:\nacross words: {across}\ndown words: {down}\nGrid (X=letter position, 0=empty space):\n{grid}\n\nINSTRUCTIONS:\n1. Place the given words in the grid according to the clues.\n2. Across words go from left to right, down words go from top to bottom.\n3. Words must intersect correctly where they share positions.\n4. In your answer:\n   - Replace each X with the appropriate letter\n   - Use a space (\" \") for positions marked 0 or where no letter belongs\n   - Format as a 2D array with each character in quotes\n   - Separate rows with commas\n   - Wrap the entire answer in double square brackets [[ ]]\n5. For example: [[\"W\", \"O\", \"R\", \"D\"], [\" \", \"N\", \" \", \"O\"], [\"G\", \"A\", \"M\", \"E\"]]\n\nFormat your solution as [[row1, row2, ...]].",
    
    "Complete this word puzzle with:\nacross words: {across}\ndown words: {down}\nPuzzle layout (X=letter position, 0=empty):\n{grid}\n\nINSTRUCTIONS:\n1. Place all the given words into the grid correctly.\n2. Across words read horizontally (left to right).\n3. Down words read vertically (top to bottom).\n4. Words must share letters where they intersect.\n5. Answer format must be: [[row1, row2, ...]] with each character in quotes.\n6. Use spaces (\" \") for 0 positions or where no letter belongs.\n7. Example: [[\"H\", \"A\", \"T\"], [\" \", \"P\", \" \"], [\"D\", \"O\", \"G\"]]\n\nProvide your completed grid as [[row1, row2, ...]].",
    
    "Fill in the blanks in this crossword puzzle:\nacross words: {across}\ndown words: {down}\nGrid (X=letter position, 0=empty):\n{grid}\n\nINSTRUCTIONS:\n1. Each X should be replaced with a letter to form the given words.\n2. Each 0 should be represented as a space (\" \") in your answer.\n3. Across words run horizontally from left to right.\n4. Down words run vertically from top to bottom.\n5. Format your answer with:\n   - Double square brackets [[ ]] around the entire answer\n   - Commas separating each row\n   - Each character in quotes, including spaces\n   - Example: [[\"P\", \"A\", \"T\"], [\"A\", \" \", \"E\"], [\"N\", \"O\", \" \"]]\n\nYour final answer should look like: [[row1, row2, ...]]",
    
    "Solve this wordscapes crossword:\nacross words: {across}\ndown words: {down}\nGrid layout (X=letter position, 0=empty space):\n{grid}\n\nINSTRUCTIONS:\n1. Fill all X positions with the correct letters to form the given words.\n2. Across words go horizontally (left to right).\n3. Down words go vertically (top to bottom).\n4. Where words cross, they must share the same letter.\n5. Format your answer precisely as:\n   - A 2D array with each element in quotes\n   - Rows separated by commas\n   - Use a space character (\" \") where marked as 0 or where no letter belongs\n   - Example: [[\"C\", \"A\", \"T\"], [\" \", \"R\", \" \"], [\"B\", \"A\", \"T\"]]\n\nYour answer should be formatted exactly as [[row1, row2, ...]].",
    
    "Complete this crossword grid:\nacross words: {across}\ndown words: {down}\nPuzzle (X=letter position, 0=empty):\n{grid}\n\nINSTRUCTIONS:\n1. Fill in each X with the appropriate letter to complete all given words.\n2. Horizontal words run from left to right (across).\n3. Vertical words run from top to bottom (down).\n4. Letters must be consistent where words intersect.\n5. In your answer:\n   - Format as a 2D array with double brackets [[ ]]\n   - Each character should be in quotes, including spaces\n   - Rows separated by commas\n   - Use spaces (\" \") for any 0 positions or where no letter belongs\n   - Example format: [[\"F\", \"O\", \"X\"], [\"A\", \" \", \"Y\"], [\"T\", \" \", \"Z\"]]\n\nFormat your answer as: [[row1, row2, ...]]",
    
    "Fill in this word puzzle grid:\nacross words: {across}\ndown words: {down}\nGrid layout (X=letter position, 0=empty space):\n{grid}\n\nINSTRUCTIONS:\n1. Fill in the grid with the given words.\n2. Across words read horizontally (left to right).\n3. Down words read vertically (top to bottom).\n4. Words must share the same letter where they intersect.\n5. Your answer format must be:\n   - Enclosed in double square brackets [[ ]]\n   - Each row separated by commas\n   - Each character in quotes (including spaces)\n   - Use space characters (\" \") for 0 positions or where no letter belongs\n   - Example: [[\"B\", \"A\", \"T\"], [\" \", \"P\", \" \"], [\"E\", \"N\", \"D\"]]\n\nProvide your answer in the format: [[row1, row2, ...]]",
    
    "Solve this crossword puzzle grid:\nacross words: {across}\ndown words: {down}\nLayout (X=letter position, 0=empty space):\n{grid}\n\nINSTRUCTIONS:\n1. Replace each X with the appropriate letter to form all the given words.\n2. Across words run left to right, down words run top to bottom.\n3. Words must share letters correctly at intersections.\n4. Format your answer precisely as a 2D grid:\n   - Wrapped in double square brackets [[ ]]\n   - Rows separated by commas\n   - Each character in quotes, including spaces\n   - Use spaces (\" \") for 0 positions or where no letter belongs\n   - For example: [[\"M\", \"A\", \"P\"], [\"A\", \" \", \"A\"], [\"T\", \"O\", \"P\"]]\n\nYour answer should look exactly like: [[row1, row2, ...]]",
    
    "Complete this word grid puzzle:\nacross words: {across}\ndown words: {down}\nGrid (X=letter position, 0=empty space):\n{grid}\n\nINSTRUCTIONS:\n1. Place all given words correctly in the grid.\n2. Horizontal (across) words read from left to right.\n3. Vertical (down) words read from top to bottom.\n4. Letters must match at intersections between words.\n5. Format your answer as:\n   - A 2D grid inside double square brackets [[ ]]\n   - Each row separated by commas\n   - Each character (including spaces) in quotes\n   - Use space characters (\" \") for 0 positions or where no letter belongs\n   - Example: [[\"C\", \"A\", \"R\"], [\"A\", \" \", \"U\"], [\"T\", \"O\", \"N\"]]\n\nFormat your solution as: [[row1, row2, ...]]"
]

# Chinese prompt templates
chinese_prompts = [
    "横向词:{across}\n纵向词:{down}\n{grid}\n\n填写说明:\n1. 将每个X替换为适当的字母，形成给定的单词。\n2. 在答案中将0位置保留为空格。\n3. 横向词从左到右放置，纵向词从上到下放置。\n4. 将你的答案格式化为用双方括号包围的二维网格：[[...]]。\n5. 每行之间用逗号分隔。\n6. 每个字符（包括空格）都应放在引号中。\n7. 例如：[[\"A\", \"B\", \"C\"], [\"D\", \"E\", \"F\"]]\n\n你的答案应该如下所示：[[行1中带引号的字母, 行2中带引号的字母, ...]]。",
    
    "完成这个填字游戏：\n横向词: {across}\n纵向词: {down}\n谜题网格 (X=字母位置, 0=空白):\n{grid}\n\n填写说明:\n1. 用给定的单词填写网格。\n2. 横向词从左到右读，纵向词从上到下读。\n3. 单词相交处，它们必须共享相同的字母。\n4. 将你的答案格式化为[[行1, 行2, ...]]，每个字符都放在引号中。\n5. 用适当的字母替换X，形成给定的单词。\n6. 在答案中将0位置保留为空格（用\" \"表示）。\n7. 格式示例：[[\"C\", \"A\", \"T\"], [\"A\", \" \", \"O\"]]\n\n以[[行1, 行2, ...]]的格式提供你完成的网格。",
    
    "解决这个文字游戏谜题：\n横向词: {across}\n纵向词: {down}\n网格 (X=字母位置, 0=空白):\n{grid}\n\n填写说明:\n1. 根据线索在网格中放置给定的单词。\n2. 横向词从左到右，纵向词从上到下。\n3. 单词必须在共享位置正确相交。\n4. 在你的答案中：\n   - 用适当的字母替换每个X\n   - 对于标记为0的位置或不属于字母的位置使用空格字符（用\" \"表示）\n   - 每个字符（包括空格）都应放在引号中\n   - 用逗号分隔行\n   - 用双方括号[[ ]]包围整个答案\n5. 例如：[[\"W\", \"O\", \"R\", \"D\"], [\" \", \"N\", \" \", \"O\"], [\"G\", \"A\", \"M\", \"E\"]]\n\n将你的解决方案格式化为[[行1, 行2, ...]]。",
    
    "完成这个带有以下单词的文字谜题：\n横向词: {across}\n纵向词: {down}\n谜题布局 (X=字母位置, 0=空白):\n{grid}\n\n填写说明:\n1. 将所有给定的单词正确放入网格中。\n2. 横向词水平读取（从左到右）。\n3. 纵向词垂直读取（从上到下）。\n4. 单词在相交处必须共享字母。\n5. 答案格式必须是：[[行1, 行2, ...]]，每个字符都放在引号中。\n6. 对于0位置或不属于字母的地方使用空格（用\" \"表示）。\n7. 示例：[[\"H\", \"A\", \"T\"], [\" \", \"P\", \" \"], [\"D\", \"O\", \"G\"]]\n\n将你完成的网格提供为[[行1, 行2, ...]]。",
    
    "填写这个填字游戏中的空白：\n横向词: {across}\n纵向词: {down}\n网格 (X=字母位置, 0=空白):\n{grid}\n\n填写说明:\n1. 每个X应该被替换为一个字母，形成给定的单词。\n2. 每个0在你的答案中应表示为空格（用\" \"表示）。\n3. 横向词水平从左到右运行。\n4. 纵向词垂直从上到下运行。\n5. 格式化你的答案：\n   - 整个答案外围用双方括号[[ ]]\n   - 每行之间用逗号分隔\n   - 每个字符（包括空格）都放在引号中\n   - 示例：[[\"P\", \"A\", \"T\"], [\"A\", \" \", \"E\"], [\"N\", \"O\", \" \"]]\n\n你的最终答案应该看起来像：[[行1, 行2, ...]]",
    
    "解决这个文字游戏填字游戏：\n横向词: {across}\n纵向词: {down}\n网格布局 (X=字母位置, 0=空白):\n{grid}\n\n填写说明:\n1. 用正确的字母填充所有X位置，形成给定的单词。\n2. 横向词水平走向（从左到右）。\n3. 纵向词垂直走向（从上到下）。\n4. 单词交叉处，它们必须共享相同的字母。\n5. 精确格式化你的答案为：\n   - 表示为[[行1, 行2, ...]]的二维网格\n   - 每个字符都应放在引号中，包括空格\n   - 在标记为0的地方或不属于字母的地方使用空格字符（用\" \"表示）\n   - 示例：[[\"C\", \"A\", \"T\"], [\" \", \"R\", \" \"], [\"B\", \"A\", \"T\"]]\n\n你的答案应该精确格式化为[[行1, 行2, ...]]，每个字符都在引号中。",
    
    "完成这个填字网格：\n横向词: {across}\n纵向词: {down}\n谜题 (X=字母位置, 0=空白):\n{grid}\n\n填写说明:\n1. 用适当的字母填充每个X，完成所有给定的单词。\n2. 水平单词从左到右运行（横向）。\n3. 垂直单词从上到下运行（纵向）。\n4. 单词相交处的字母必须一致。\n5. 在你的答案中：\n   - 格式化为带有双括号[[ ]]的二维数组\n   - 每个字符（包括空格）都应放在引号中\n   - 用逗号分隔行\n   - 对于任何0位置或不属于字母的地方使用空格（用\" \"表示）\n   - 格式示例：[[\"F\", \"O\", \"X\"], [\"A\", \" \", \"Y\"], [\"T\", \" \", \"Z\"]]\n\n将你的答案格式化为：[[行1, 行2, ...]]",
    
    "填写这个文字谜题网格：\n横向词: {across}\n纵向词: {down}\n网格布局 (X=字母位置, 0=空白):\n{grid}\n\n填写说明:\n1. 用给定的单词填写网格。\n2. 横向词水平读取（从左到右）。\n3. 纵向词垂直读取（从上到下）。\n4. 单词在相交处必须共享相同的字母。\n5. 你的答案格式必须是：\n   - 用双方括号[[ ]]括起\n   - 每行用逗号分隔\n   - 每个字符（包括空格）都应放在引号中\n   - 对于0位置或不属于字母的地方使用空格字符（用\" \"表示）\n   - 示例：[[\"B\", \"A\", \"T\"], [\" \", \"P\", \" \"], [\"E\", \"N\", \"D\"]]\n\n以下格式提供你的答案：[[行1, 行2, ...]]",
    
    "解决这个填字游戏网格：\n横向词: {across}\n纵向词: {down}\n布局 (X=字母位置, 0=空白):\n{grid}\n\n填写说明:\n1. 用适当的字母替换每个X，形成所有给定的单词。\n2. 横向词从左到右运行，纵向词从上到下运行。\n3. 单词必须在交叉点正确共享字母。\n4. 将你的答案精确格式化为二维网格：\n   - 用双方括号[[ ]]包围\n   - 行之间用逗号分隔\n   - 每个字符（包括空格）都应放在引号中\n   - 对于0位置或不属于字母的地方使用空格（用\" \"表示）\n   - 例如：[[\"M\", \"A\", \"P\"], [\"A\", \" \", \"A\"], [\"T\", \"O\", \"P\"]]\n\n你的答案应该看起来完全像：[[行1, 行2, ...]]",
    
    "完成这个文字网格谜题：\n横向词: {across}\n纵向词: {down}\n网格 (X=字母位置, 0=空白):\n{grid}\n\n填写说明:\n1. 在网格中正确放置所有给定的单词。\n2. 水平（横向）单词从左到右读取。\n3. 垂直（纵向）单词从上到下读取。\n4. 单词之间的交叉点必须匹配字母。\n5. 将你的答案格式化为：\n   - 双方括号[[ ]]内的二维网格\n   - 每行用逗号分隔\n   - 每个字符（包括空格）都应放在引号中\n   - 对于0位置或不属于字母的地方使用空格字符（用\" \"表示）\n   - 示例：[[\"C\", \"A\", \"R\"], [\"A\", \" \", \"U\"], [\"T\", \"O\", \"N\"]]\n\n将你的解决方案格式化为：[[行1, 行2, ...]]"
] 