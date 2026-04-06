import re

''' 验证身份证号是否有效'''

# 身份证号码正则表达式
id_card_pattern = r'^(?:\d{15}|\d{17}[\dXx])$'

# 测试身份证号码
test_id_cards = [
    '123456789012345',  # 15位
    '12345678901234567X',  # 18位，最后一位是X
    '123456789012345678',  # 18位，错误示例
    '12345678901234X',  # 错误示例
    '440603198608083037'  
]

def calculate_check_digit(id_card):
    # 计算18位身份证的校验码
    if len(id_card) != 17:
        return None

    # 加权因子，长度应为17
    weights = [7, 9, 10, 5, 8, 4, 2, 1, 6, 3, 7, 9, 10, 5, 8, 4, 2]
    # 校验码对应的值
    check_digits = '10X98765432'
    
    # 计算加权和
    total = sum(int(id_card[i]) * weights[i] for i in range(17))
    check_index = total % 11
    
    return check_digits[check_index]

def is_valid_id_card(id_card):
    # 校验身份证号码的逻辑
    if len(id_card) == 15:
        return True  # 这里可以添加15位身份证的具体校验逻辑
    elif len(id_card) == 18:
        # 校验18位身份证的校验码
        if id_card[:-1].isdigit() and id_card[-1].upper() == calculate_check_digit(id_card[:-1]):
            return True
    return False

# 匹配并输出结果
for id_card in test_id_cards:
    if re.match(id_card_pattern, id_card) and is_valid_id_card(id_card):
        print(f"{id_card} 是有效的身份证号码")
    else:
        print(f"{id_card} 不是有效的身份证号码")