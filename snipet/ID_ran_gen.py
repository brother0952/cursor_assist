import random

''' 生成随机身份证号'''
def calculate_check_digit(id_card):
    # 计算18位身份证的校验码
    if len(id_card) != 17:
        return None

    # 加权因子
    weights = [7, 9, 10, 5, 8, 4, 2, 1, 6, 3, 7, 9, 10, 5, 8, 4, 2]
    # 校验码对应的值
    check_digits = '10X98765432'
    
    # 计算加权和
    total = sum(int(id_card[i]) * weights[i] for i in range(17))
    check_index = total % 11
    
    return check_digits[check_index]

def is_valid_date(year, month, day):
    # 检查日期的有效性
    if month < 1 or month > 12:
        return False
    if day < 1 or day > 31:
        return False
    if month in [4, 6, 9, 11] and day == 31:
        return False
    if month == 2:
        if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0):
            return day > 29  # 闰年
        return day > 28  # 平年
    return True

def generate_random_id_card():
    while True:  # 使用循环来生成有效的身份证号码
        # 随机生成前6位地区码（假设范围为110000到659999）
        province_code = random.randint(110000, 659999)  # 省份代码范围
        
        # 随机生成出生日期
        birth_year = random.randint(1950, 2003)  # 假设出生年份范围
        birth_month = random.randint(1, 12)
        
        # 生成有效的日期
        if birth_month == 2:
            # 处理2月的日期
            if (birth_year % 4 == 0 and birth_year % 100 != 0) or (birth_year % 400 == 0):
                birth_day = random.randint(1, 29)  # 闰年
            else:
                birth_day = random.randint(1, 28)  # 平年
        elif birth_month in [4, 6, 9, 11]:
            birth_day = random.randint(1, 30)  # 30天的月份
        else:
            birth_day = random.randint(1, 31)  # 31天的月份
        
        # 生成前17位身份证号码
        id_card = f"{province_code}{birth_year}{birth_month:02d}{birth_day:02d}"
        
        # 生成顺序码（3位）
        id_card += f"{random.randint(0, 999):03d}"
        
        # 计算校验码
        check_digit = calculate_check_digit(id_card)
        
        # 确保校验码不为 None
        if check_digit is not None:
            id_card += check_digit
            return id_card  # 返回有效的身份证号码

# 生成并输出随机有效的身份证号码
for _ in range(5):  # 生成5个身份证号码
    print(generate_random_id_card())
