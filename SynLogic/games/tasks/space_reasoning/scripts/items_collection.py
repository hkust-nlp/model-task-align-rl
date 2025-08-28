# coding: utf-8

import random
from typing import List, Dict, Any, Optional

# 超大规模中文物品列表
ITEMS_CN = [
    # 家具类
    "桌子", "椅子", "沙发", "床", "柜子", "书架", "衣橱", "茶几", "电视柜", "餐桌", "梳妆台", "置物架", 
    "鞋柜", "办公桌", "休闲椅", "摇椅", "床头柜", "吧台", "酒柜", "衣帽架", "玄关柜", "屏风", "花架", 
    "挂墙书架", "折叠桌", "儿童书桌", "婴儿床", "懒人沙发", "躺椅", "换鞋凳", "行李架", "角落架", "衣帽钩",
    
    # 电子产品
    "手机", "电脑", "平板", "电视", "冰箱", "洗衣机", "空调", "微波炉", "电饭煲", "吸尘器", "音响", 
    "相机", "游戏机", "电热水壶", "烤箱", "电风扇", "加湿器", "打印机", "投影仪", "电动剃须刀", "蓝牙耳机", 
    "智能手表", "VR眼镜", "电子书阅读器", "智能音箱", "行车记录仪", "电动牙刷", "扫地机器人", "咖啡机", "榨汁机",
    
    # 厨房用品
    "锅", "碗", "筷子", "勺子", "刀", "叉", "杯子", "盘子", "砧板", "漏勺", "饭盒", "保温杯", "茶壶", 
    "咖啡杯", "餐垫", "烹饪铲", "料理钵", "调味瓶", "水果刀", "开瓶器", "量杯", "烘焙模具", "绞肉机", 
    "打蛋器", "搅拌碗", "饺子模具", "烧烤架", "厨房剪", "擀面杖", "冰块模具", "蒸笼", "压蒜器", "水果去核器",
    
    # 衣物饰品
    "衬衫", "裤子", "裙子", "外套", "帽子", "鞋子", "袜子", "手套", "围巾", "背包", "手表", "项链", 
    "耳环", "眼镜", "皮带", "领带", "连衣裙", "风衣", "牛仔裤", "T恤", "卫衣", "夹克", "西装", "毛衣", 
    "泳装", "睡衣", "运动裤", "运动鞋", "高跟鞋", "凉鞋", "戒指", "手镯", "太阳镜", "发夹", "钱包", "领结",
    
    # 食物饮品
    "苹果", "香蕉", "橙子", "葡萄", "西瓜", "番茄", "黄瓜", "胡萝卜", "土豆", "白菜", "牛肉", "猪肉", 
    "鸡肉", "鱼", "虾", "蛋糕", "面包", "饼干", "巧克力", "冰淇淋", "草莓", "蓝莓", "柠檬", "桃子", 
    "梨", "樱桃", "菠萝", "芒果", "豆腐", "鸡蛋", "奶酪", "酸奶", "牛奶", "咖啡", "茶", "可乐", "果汁",
    
    # 办公用品
    "钢笔", "铅笔", "尺子", "橡皮", "订书机", "夹子", "便签纸", "剪刀", "胶带", "文件夹", "笔记本", "日历", 
    "计算器", "复印纸", "信封", "名片盒", "印章", "墨水", "报纸", "杂志", "书本", "字典", "记事本", "白板", 
    "白板笔", "图钉", "回形针", "打孔器", "笔筒", "文件柜", "签字笔", "荧光笔", "文件袋", "标签纸", "便利贴",
    
    # 工具
    "锤子", "螺丝刀", "扳手", "钳子", "卷尺", "锯子", "电钻", "手电筒", "胶水", "砂纸", "梯子", "绳子", 
    "铲子", "钉子", "螺丝", "电线", "管钳", "焊接工具", "刷子", "切割刀", "锉刀", "螺栓", "钢丝钳", "万用表", 
    "电烙铁", "水平仪", "电锤", "电刨", "气动工具", "油漆刷", "工具箱", "工具柜", "磁铁", "木工凿", "热熔胶枪",
    
    # 玩具游戏
    "积木", "玩偶", "拼图", "遥控车", "飞机模型", "毛绒玩具", "棋盘", "扑克牌", "陀螺", "弹珠", "风筝", 
    "气球", "水枪", "溜溜球", "橡皮泥", "画笔", "贴纸", "魔方", "望远镜", "跳绳", "遥控直升机", "泡泡机", 
    "玩具琴", "玩具鼓", "遥控船", "娃娃屋", "小汽车", "玩具火车", "轨道套装", "恐龙模型", "科学实验套装", "磁力积木",
    
    # 体育用品
    "篮球", "足球", "排球", "网球", "乒乓球", "羽毛球", "高尔夫球", "棒球", "保龄球", "橄榄球", "跑步机", 
    "健身车", "哑铃", "杠铃", "瑜伽垫", "仰卧板", "划船机", "健身球", "弹力带", "网球拍", "羽毛球拍", 
    "乒乓球拍", "高尔夫球杆", "护膝", "护腕", "泳镜", "泳帽", "滑雪板", "滑板", "轮滑鞋", "自行车", "头盔",
    
    # 美容护理
    "口红", "粉底", "眼影", "腮红", "睫毛膏", "眉笔", "眼线笔", "高光", "遮瑕", "定妆粉", "粉刷", 
    "美妆蛋", "化妆镜", "发膏", "发蜡", "洗发水", "护发素", "沐浴露", "洗面奶", "面膜", "爽肤水", "乳液", 
    "精华液", "防晒霜", "卸妆水", "香水", "指甲油", "指甲刀", "剃须刀", "电吹风", "卷发棒", "发梳", "洗漱杯",
    
    # 植物花卉
    "玫瑰", "郁金香", "百合", "向日葵", "菊花", "兰花", "牡丹", "绿萝", "仙人掌", "芦荟", "吊兰", 
    "多肉植物", "盆栽", "松树", "竹子", "橡树", "樱花树", "榕树", "椰子树", "薰衣草", "薄荷", "迷迭香", 
    "绿茶", "红茶", "菊花茶", "玉米", "小麦", "水稻", "向日葵", "棉花", "大豆", "油菜", "葵花籽", "花生"
]

# 对应的英文物品列表
ITEMS_EN = [
    # Furniture
    "table", "chair", "sofa", "bed", "cabinet", "bookshelf", "wardrobe", "coffee table", "TV stand", "dining table",
    "dressing table", "shelf", "shoe cabinet", "desk", "lounge chair", "rocking chair", "nightstand", "bar counter",
    "wine cabinet", "coat rack", "console table", "screen divider", "flower stand", "wall shelf", "folding table",
    "children's desk", "crib", "bean bag", "chaise longue", "bench", "luggage rack", "corner shelf", "coat hook",
    
    # Electronics
    "phone", "computer", "tablet", "television", "refrigerator", "washing machine", "air conditioner", "microwave",
    "rice cooker", "vacuum cleaner", "speaker", "camera", "game console", "kettle", "oven", "fan", "humidifier",
    "printer", "projector", "electric shaver", "bluetooth headphones", "smartwatch", "VR glasses", "e-reader",
    "smart speaker", "dash cam", "electric toothbrush", "robot vacuum", "coffee machine", "juicer",
    
    # Kitchen Items
    "pot", "bowl", "chopsticks", "spoon", "knife", "fork", "cup", "plate", "cutting board", "strainer",
    "lunch box", "thermos", "teapot", "coffee mug", "placemat", "spatula", "mixing bowl", "spice bottle",
    "fruit knife", "bottle opener", "measuring cup", "baking mold", "meat grinder", "egg beater", "mixing bowl",
    "dumpling mold", "grill rack", "kitchen scissors", "rolling pin", "ice cube tray", "steamer", "garlic press",
    "fruit corer",
    
    # Clothing & Accessories
    "shirt", "pants", "skirt", "coat", "hat", "shoes", "socks", "gloves", "scarf", "backpack", "watch", "necklace",
    "earrings", "glasses", "belt", "tie", "dress", "trench coat", "jeans", "t-shirt", "hoodie", "jacket", "suit",
    "sweater", "swimsuit", "pajamas", "sweatpants", "sneakers", "high heels", "sandals", "ring", "bracelet",
    "sunglasses", "hair clip", "wallet", "bow tie",
    
    # Food & Drinks
    "apple", "banana", "orange", "grape", "watermelon", "tomato", "cucumber", "carrot", "potato", "cabbage",
    "beef", "pork", "chicken", "fish", "shrimp", "cake", "bread", "cookie", "chocolate", "ice cream", "strawberry",
    "blueberry", "lemon", "peach", "pear", "cherry", "pineapple", "mango", "tofu", "egg", "cheese", "yogurt",
    "milk", "coffee", "tea", "cola", "juice",
    
    # Office Supplies
    "pen", "pencil", "ruler", "eraser", "stapler", "clip", "sticky note", "scissors", "tape", "folder", "notebook",
    "calendar", "calculator", "copy paper", "envelope", "business card holder", "seal", "ink", "newspaper", "magazine",
    "book", "dictionary", "notepad", "whiteboard", "whiteboard marker", "pushpin", "paperclip", "hole puncher",
    "pen holder", "filing cabinet", "marker", "highlighter", "document bag", "label", "post-it",
    
    # Tools
    "hammer", "screwdriver", "wrench", "pliers", "tape measure", "saw", "drill", "flashlight", "glue", "sandpaper",
    "ladder", "rope", "shovel", "nail", "screw", "wire", "pipe wrench", "welding tool", "brush", "cutter", "file",
    "bolt", "wire cutter", "multimeter", "soldering iron", "level", "electric hammer", "electric planer",
    "pneumatic tool", "paint brush", "toolbox", "tool cabinet", "magnet", "wood chisel", "hot glue gun",
    
    # Toys & Games
    "building blocks", "doll", "puzzle", "remote control car", "airplane model", "plush toy", "chess board",
    "playing cards", "top", "marble", "kite", "balloon", "water gun", "yo-yo", "clay", "paint brush", "sticker",
    "magic cube", "telescope", "jump rope", "RC helicopter", "bubble machine", "toy piano", "toy drum", "RC boat",
    "dollhouse", "toy car", "toy train", "track set", "dinosaur model", "science kit", "magnetic blocks",
    
    # Sports Equipment
    "basketball", "football", "volleyball", "tennis", "ping pong ball", "badminton", "golf ball", "baseball",
    "bowling ball", "rugby", "treadmill", "exercise bike", "dumbbell", "barbell", "yoga mat", "sit-up bench",
    "rowing machine", "fitness ball", "resistance band", "tennis racket", "badminton racket", "ping pong paddle",
    "golf club", "knee pad", "wrist guard", "swimming goggles", "swimming cap", "ski board", "skateboard",
    "roller skates", "bicycle", "helmet",
    
    # Beauty & Personal Care
    "lipstick", "foundation", "eyeshadow", "blush", "mascara", "eyebrow pencil", "eyeliner", "highlighter",
    "concealer", "setting powder", "makeup brush", "beauty blender", "makeup mirror", "hair gel", "hair wax",
    "shampoo", "conditioner", "shower gel", "facial cleanser", "facial mask", "toner", "lotion", "serum",
    "sunscreen", "makeup remover", "perfume", "nail polish", "nail clipper", "razor", "hair dryer", "curling iron",
    "hair comb", "toothbrush cup",
    
    # Plants & Flowers
    "rose", "tulip", "lily", "sunflower", "chrysanthemum", "orchid", "peony", "pothos", "cactus", "aloe vera",
    "hanging plant", "succulent", "potted plant", "pine tree", "bamboo", "oak tree", "cherry blossom tree",
    "banyan tree", "coconut tree", "lavender", "mint", "rosemary", "green tea", "black tea", "chrysanthemum tea",
    "corn", "wheat", "rice", "sunflower", "cotton", "soybean", "canola", "sunflower seed", "peanut"
]

# 合并所有中文物品到一个大列表
ALL_ITEMS_CN = ITEMS_CN

# 合并所有英文物品到一个大列表
ALL_ITEMS_EN = ITEMS_EN

def get_random_items(count: int, language: str = "cn", exclude: Optional[List[str]] = None) -> List[str]:
    """
    从物品集合中随机选择指定数量的物品
    
    Args:
        count: 需要的物品数量
        language: 语言，"cn"为中文，"en"为英文
        exclude: 需要排除的物品列表
    
    Returns:
        随机选择的物品列表
    """
    items_list = ALL_ITEMS_CN if language.lower() == "cn" else ALL_ITEMS_EN
    available_items = items_list.copy()
    
    if exclude:
        available_items = [item for item in available_items if item not in exclude]
    
    # 如果请求的数量超过可用物品数量，则返回所有可用物品
    if count >= len(available_items):
        raise ValueError("请求的物品数量超过可用物品数量")
    
    return random.sample(available_items, count)

def distribute_items_to_nodes(nodes: List[Any], language: str = "cn") -> None:
    """
    将物品随机分配给节点
    
    Args:
        nodes: 节点列表，每个节点必须有item属性
        language: 语言，"cn"为中文，"en"为英文
    """
    # 随机选择与节点数量相同的物品
    items = get_random_items(len(nodes), language)
    
    # 随机打乱顺序
    random.shuffle(items)
    
    # 分配物品给节点
    for i, node in enumerate(nodes):
        node.item = items[i]
