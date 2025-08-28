import random

# 中文名字集合
chinese_names = [
    "张三", "李四", "王五", "赵六", "钱七", "孙八", "周九", "吴十",
    "郑一", "冯二", "陈明", "楚天", "刘洋", "林峰", "黄河", "杨光",
    "朱红", "秦岭", "许诺", "何方", "吕梁", "施然", "张艺", "孔亮",
    "王伟", "李强", "张磊", "刘军", "陈亮", "杨勇", "赵阳", "钱明",
    # 新增人名
    "吴刚", "郑华", "周明", "徐亮", "胡军", "高峰", "林阳", "马超",
    "叶勇", "宋杰", "唐军", "韩磊", "曾强", "彭勇", "董明", "任阳",
    "谢军", "邓强", "卢亮", "蒋刚", "江明", "何强", "梁勇", "熊杰"
]

# 英文名字集合
english_names = [
    "Alice", "Bob", "Charlie", "Diana", "Ethan", "Fiona", "George", "Hannah",
    "Ian", "Julia", "Kevin", "Laura", "Mike", "Nina", "Oscar", "Penny",
    "Quincy", "Rachel", "Steve", "Tina", "Ulysses", "Victoria", "William", "Xena",
    "Yuri", "Zoe", "Adam", "Bella", "Carl", "Donna", "Erik", "Felicia",
    # 新增人名
    "Grace", "Henry", "Ivy", "Jack", "Katherine", "Leo", "Megan", "Nate",
    "Olivia", "Peter", "Quinn", "Ryan", "Sophia", "Thomas", "Ursula", "Vincent",
    "Wendy", "Xavier", "Yasmine", "Zachary", "Andrew", "Betty", "Cameron", "Daisy"
]

# 中文颜色集合
chinese_colors = [
    "红色", "蓝色", "绿色", "黄色", "紫色", "黑色", "白色", "橙色",
    "粉色", "棕色", "灰色", "青色", "金色", "银色", "米色", "桃色",
    # 新增颜色
    "天蓝色", "草绿色", "玫红色", "深蓝色", "浅绿色", "墨绿色", "暗红色", "深紫色",
    "明黄色", "淡蓝色", "深棕色", "浅灰色", "藏青色", "古铜色", "靛蓝色", "酒红色"
]

# 英文颜色集合
english_colors = [
    "red", "blue", "green", "yellow", "purple", "black", "white", "orange",
    "pink", "brown", "gray", "cyan", "golden", "silver", "beige", "peach",
    # 新增颜色
    "skyblue", "lime", "magenta", "navy", "mint", "olive", "maroon", "violet",
    "amber", "azure", "chocolate", "coral", "indigo", "ivory", "lavender", "turquoise"
]

# 中文物品类别集合
chinese_categories = [
    "书", "手表", "耳机", "笔", "杯子", "背包", "手机", "电脑",
    "相机", "键盘", "鼠标", "帽子", "眼镜", "项链", "手链", "戒指",
    "围巾", "手套", "伞", "玩具", "花盆", "盒子", "钱包", "地图",
    # 新增物品
    "画框", "台灯", "雕像", "枕头", "毯子", "香水", "扇子", "徽章",
    "发夹", "手绢", "口哨", "纪念币", "书签", "小刀", "指南针", "记事本",
    "茶叶", "明信片", "挂饰", "水壶", "吉他", "口琴", "玩偶", "摆件"
]

# 英文物品类别集合
english_categories = [
    "book", "watch", "headphones", "pen", "cup", "backpack", "phone", "laptop",
    "camera", "keyboard", "mouse", "hat", "glasses", "necklace", "bracelet", "ring",
    "scarf", "gloves", "umbrella", "toy", "flowerpot", "box", "wallet", "map",
    # 新增物品
    "frame", "lamp", "statue", "pillow", "blanket", "perfume", "fan", "badge",
    "hairpin", "handkerchief", "whistle", "coin", "bookmark", "knife", "compass", "notebook",
    "tea", "postcard", "ornament", "kettle", "guitar", "harmonica", "doll", "figurine"
]

# 中文交换操作描述模板
chinese_operations = {
    "operation1": "{name1}和{name2}互相交换他们的所有物品",
    "operation2": "{object1}和{object2}被它们的主人互相交换",
    "operation3": "持有{object1}的人和{name1}互相交换他们的所有物品",
    "operation4": "{name1}用{object1}换取了{object2}",
    "operation5": "{name1}把{object1}送给了{name2}，{name2}把{object2}送给了{name1}",
    "operation6": "{name1}把{object1}和{name2}的{object2}交换了",
    "operation7": "拥有{object1}的人和拥有{object2}的人交换了他们的物品",
    "operation8": "{name1}想和{name2}交换，但是{name2}拒绝了",  # 干扰操作
    "operation9": "{name1}很喜欢{object2}",  # 干扰操作
    "operation10": "{name1}正在寻找{object1}，但是找不到",  # 干扰操作
    # 新增操作
    "operation11": "{name1}主动和拥有{object1}的人交换了物品",
    "operation13": "{name1}、{name2}和{name3}进行了三方物品交换，{name1}把物品给了{name2}，{name2}把物品给了{name3}，{name3}把物品给了{name1}"
}

# 英文交换操作描述模板
english_operations = {
    "operation1": "{name1} and {name2} swapped their items",
    "operation2": "The owners of {object1} and {object2} exchanged these items",
    "operation3": "The person who had {object1} exchanged items with {name1}",
    "operation4": "{name1} traded {object1} for {object2}",
    "operation5": "{name1} gave {object1} to {name2}, and {name2} gave {object2} to {name1}",
    "operation6": "{name1} exchanged {object1} with {name2}'s {object2}",
    "operation7": "The person who owned {object1} and the person who owned {object2} swapped their items",
    "operation8": "{name1} wanted to trade with {name2}, but {name2} refused",  # 干扰操作
    "operation9": "{name1} really liked {object2}",  # 干扰操作
    "operation10": "{name1} was looking for {object1}, but couldn't find it",  # 干扰操作
    # 新增操作
    "operation11": "{name1} initiated an exchange with the owner of {object1}",
    "operation13": "{name1}, {name2}, and {name3} did a three-way exchange where {name1} gave to {name2}, {name2} gave to {name3}, and {name3} gave to {name1}"
}

# 中文提示模板
chinese_prompt_templates = [
    "有 {n} 个人, 分别是 {names}；有 {x} 件物品，分别是 {objects}，每个物品有且仅有1个；\n交换之前，物品和人的对应关系是 {owns_before}；\n规定依次发生 {z} 次交换动作： {operations}，求交换结束时，物品和人的对应关系。",
    "在一个小镇上住着 {n} 个人：{names}。他们各自拥有一件物品，共有 {x} 件物品：{objects}。\n初始时，物品的归属关系是：{owns_before}。\n现在，他们按照以下顺序进行了 {z} 次物品交换：{operations}。\n请计算最终每个人拥有的物品是什么？",
    "{n} 个朋友 {names} 每人拥有一件特别的物品，这些物品是：{objects}。\n开始时的物品分配是：{owns_before}。\n他们按照下列规则进行了 {z} 次交换：{operations}。\n交换结束后，每个人手中的物品是什么？",
    "在一次聚会上，有 {n} 个参与者：{names}。每人带了一件物品，这些物品是：{objects}。\n一开始，物品与人的对应关系为：{owns_before}。\n在聚会中，他们按顺序进行了以下 {z} 次交换：{operations}。\n请问聚会结束时，每个人拥有什么物品？",
    "{n} 位同学 {names} 各自拥有一件物品，这些物品分别是：{objects}。\n最初的物品归属是：{owns_before}。\n现在他们要进行 {z} 轮物品交换，交换规则如下：{operations}。\n所有交换结束后，每位同学最终拥有的物品是什么？",
    "在一个小组中有 {n} 个成员，他们是：{names}。每人持有一件独特的物品，这些物品是：{objects}。\n初始的物品分配情况是：{owns_before}。\n根据以下规则，他们进行了 {z} 次物品交换：{operations}。\n请问所有交换结束后，每个人拥有什么物品？",
    "有 {n} 个朋友参加了一个交换礼物的游戏，他们是：{names}。礼物包括：{objects}。\n游戏开始前，礼物的分配是：{owns_before}。\n游戏规则是依次进行 {z} 轮交换：{operations}。\n游戏结束后，每个人获得的礼物是什么？",
    "{n} 名玩家 {names} 正在玩一个物品交换游戏。游戏中的物品有：{objects}。\n游戏开始时，物品的分配如下：{owns_before}。\n玩家们按顺序执行了以下 {z} 次交换：{operations}。\n请计算游戏结束时各玩家拥有的物品。",
    "在一个交易市场上，有 {n} 个交易者：{names}。他们各自带来一件商品：{objects}。\n交易开始前，物品的所有权是：{owns_before}。\n他们按照下面的顺序进行了 {z} 次交易：{operations}。\n所有交易结束后，每个人拥有什么商品？",
    "班级里有 {n} 个学生：{names}，每人有一件物品：{objects}。\n最初物品的归属为：{owns_before}。\n他们按照老师的指令进行了 {z} 次物品交换：{operations}。\n请问交换后，每个学生手中的物品是什么？"
]

# 英文提示模板
english_prompt_templates = [
    "There are {n} people: {names}, and {x} items: {objects}, each item being unique and singular.\nBefore any exchanges, the ownership of items was as follows: {owns_before}.\nThrough a series of {z} exchanges: {operations}, determine the final ownership relationship between people and items.",
    "In a small town, there live {n} people: {names}. Each owns one item from a set of {x} items: {objects}.\nInitially, the ownership is as follows: {owns_before}.\nNow, they perform {z} exchanges in the following order: {operations}.\nCalculate what item each person ends up with.",
    "{n} friends {names} each possess a special item from the set: {objects}.\nThe initial distribution is: {owns_before}.\nThey proceed to swap items according to these {z} rules: {operations}.\nAfter all exchanges, what item does each person have?",
    "At a gathering, there are {n} participants: {names}. Each brought one item: {objects}.\nAt the beginning, the item-person pairs were: {owns_before}.\nDuring the event, they conducted {z} exchanges in sequence: {operations}.\nAt the end of the gathering, what item does each person possess?",
    "{n} classmates {names} each own one item from the collection: {objects}.\nThe initial ownership is: {owns_before}.\nThey are about to conduct {z} rounds of item exchanges, following these rules: {operations}.\nAfter all exchanges, what item does each classmate end up with?",
    "In a group of {n} members: {names}, each holds a unique item from: {objects}.\nThe initial distribution of items is: {owns_before}.\nAccording to the following rules, they perform {z} item exchanges: {operations}.\nAfter all exchanges, what item does each person have?",
    "{n} friends participate in a gift exchange game. They are: {names}. The gifts include: {objects}.\nBefore the game starts, the gifts are distributed as: {owns_before}.\nThe game rules involve {z} rounds of exchanges: {operations}.\nAfter the game ends, what gift does each person receive?",
    "{n} players {names} are playing an item exchange game. The items in the game are: {objects}.\nAt the start of the game, the items are distributed as follows: {owns_before}.\nThe players execute {z} exchanges in sequence: {operations}.\nCalculate what items each player has at the end of the game.",
    "At a trading market, there are {n} traders: {names}. Each brings one commodity: {objects}.\nBefore trading begins, the ownership is: {owns_before}.\nThey conduct {z} trades in the following order: {operations}.\nAfter all trading is complete, what commodity does each person own?",
    "Once upon a time, there were {n} friends named {names}. Each of them initially had a unique item from the set of {objects}, each one distinct and singular.\nBefore any exchanges took place, the ownership of items was as follows: {owns_before}.\nThroughout the day, they proceeded to swap their items according to these rules: {operations}.\nAfter a total of {z} swaps, determined by the sequence above, let's find out what item each person ends up with."
]

def format_operations(operations, is_chinese=False):
    """
    格式化交换操作描述
    
    @param operations: 交换操作列表
    @param is_chinese: 是否使用中文格式
    @return: 格式化的操作描述
    """
    # 直接连接操作，不添加交换次序描述
    return "\n".join(operations)

def format_owns_before(owns_before, is_chinese=False):
    """
    格式化初始物品归属关系
    
    @param owns_before: 初始物品归属关系字典
    @param is_chinese: 是否使用中文格式
    @return: 格式化的归属关系描述
    """
    if is_chinese:
        formatted = []
        for person, item in owns_before.items():
            formatted.append(f"{person}拥有{item}")
        return "；".join(formatted)
    else:
        formatted = []
        for person, item in owns_before.items():
            formatted.append(f"{person} owned the {item}")
        return ".\n".join(formatted) + "."

def prompt_goods_exchange(n, names, objects, owns_before, operations, is_chinese=False):
    """
    生成物品交换游戏的提示语
    
    @param n: 人数
    @param names: 人名列表
    @param objects: 物品列表
    @param owns_before: 初始物品归属关系
    @param operations: 交换操作列表
    @param is_chinese: 是否生成中文提示
    @return: 格式化后的提示语
    """
    # 格式化人名列表
    names_str = "、".join(names) if is_chinese else ", ".join(names)
    
    # 格式化物品列表
    objects_str = "、".join(objects) if is_chinese else ", ".join(objects)
    
    # 格式化初始物品归属关系
    owns_before_str = format_owns_before(owns_before, is_chinese)
    
    # 格式化交换操作
    operations_str = format_operations(operations, is_chinese)
    
    # 选择提示模板
    if is_chinese:
        prompt = random.choice(chinese_prompt_templates)
        # 添加格式要求
        prompt += " 请在回答的最后使用 Python markdown 代码块格式输出结果，结果应为 tuple 结构。例如：\n```python\n(('张三','红色书'),('李四','蓝色手表'),('王五','绿色耳机'),('赵六','黄色笔'),('钱七','紫色杯子'))\n```"
    else:
        prompt = random.choice(english_prompt_templates)
        # 添加格式要求
        prompt += " Please provide your answer at the end using a Python markdown code block format, with the result in a tuple structure. For example:\n```python\n(('Alice','red book'),('Bob','blue watch'),('Charlie','green headphones'),('Diana','yellow pen'),('Ethan','purple cup'))\n```"
    
    # 填充参数
    prompt = prompt.format(
        n=n,
        x=len(objects),
        z=len(operations),
        names=names_str,
        objects=objects_str,
        owns_before=owns_before_str,
        operations=operations_str
    )
    
    return prompt 