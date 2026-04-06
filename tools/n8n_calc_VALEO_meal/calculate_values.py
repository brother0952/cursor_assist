import datetime

# 法定节假日列表 (示例，可根据实际情况更新)
HOLIDAYS = [
    # 元旦
    datetime.date(2025, 1, 1),
    # 春节
    datetime.date(2025, 1, 29),  # 春节
    datetime.date(2025, 1, 30),  # 春节
    datetime.date(2025, 1, 31),  # 春节
    datetime.date(2025, 2, 1),   # 春节
    datetime.date(2025, 2, 2),   # 春节
    datetime.date(2025, 2, 3),   # 春节
    # 清明节
    datetime.date(2025, 4, 4),
    # 劳动节
    datetime.date(2025, 5, 1),
    datetime.date(2025, 5, 2),
    datetime.date(2025, 5, 3),
    # 端午节
    datetime.date(2025, 5, 31),
    # 中秋节
    datetime.date(2025, 10, 6),
    # 国庆节
    datetime.date(2025, 10, 1),
    datetime.date(2025, 10, 2),
    datetime.date(2025, 10, 3),
    datetime.date(2025, 10, 4),
    datetime.date(2025, 10, 5),
    datetime.date(2025, 10, 6),
    datetime.date(2025, 10, 7),
]

def is_weekend_or_holiday(date):
    """
    判断给定日期是否为周末或法定节假日
    """
    # 周六(5)或周日(6)
    if date.weekday() >= 5:
        return True
    
    # 法定节假日
    if date in HOLIDAYS:
        return True
    
    return False

def get_time_value(date=None):
    """
    根据时间获取数值：
    - 如果没超过12点，返回2
    - 如果超过12点但没超过18点，返回1
    - 如果超过18点，返回0
    - 如果是周末或节假日，返回0
    """
    if date is None:
        now = datetime.datetime.now()
        date = now.date()
    else:
        now = datetime.datetime.combine(date, datetime.time(0, 0, 0))
    
    # 如果是周末或节假日，返回0
    if is_weekend_or_holiday(date):
        return 0
    
    hour = now.hour if hasattr(now, 'hour') else 0
    
    if hour < 12:
        return 2
    elif hour < 18:
        return 1
    else:
        return 0

def next_21st():
    """
    计算下一个21日的日期
    """
    today = datetime.date.today()
    
    # 如果今天是本月21日或之前，下一个21日就是本月21日
    # 否则，下一个21日是下个月21日
    if today.day <= 21:
        next_date = today.replace(day=21)
    else:
        # 需要计算下个月的21日
        if today.month == 12:
            next_date = today.replace(year=today.year + 1, month=1, day=21)
        else:
            next_date = today.replace(month=today.month + 1, day=21)
    
    return next_date

def count_values_until_next_21st():
    """
    计算从现在到下一个21日之间有多少个数值
    跳过周末和法定节假日（这些日子被认为是没有数值的）
    """
    today = datetime.date.today()
    target_date = next_21st()
    
    total_values = 0
    
    # 逐天检查从今天到目标日期前一天的所有日期
    current_date = today
    while current_date < target_date:
        # 只有非周末和非节假日才计算数值
        if not is_weekend_or_holiday(current_date):
            # 如果是今天，需要根据当前时间计算剩余值
            if current_date == today:
                current_value = get_time_value()
                total_values+=current_value
         
            else:
                # 非今天的一整天天都有3个值
                total_values += 2 
        
        current_date += datetime.timedelta(days=1)
    
    # 检查目标日期本身（21日）
    # 如果目标日期不是周末或节假日，则加上该天的第一个值（2）
    if not is_weekend_or_holiday(target_date):
        total_values += 0  # 目标日期的值2
    
    return total_values

if __name__ == "__main__":
    print(f"当前时间: {datetime.datetime.now()}")
    print(f"当前时间对应的数值: {get_time_value()}")
    print(f"下一个21日: {next_21st()}")
    print(f"到下一个21日的数值数量: {count_values_until_next_21st()}")
    
    # 显示期间的周末和节假日
    today = datetime.date.today()
    target_date = next_21st()
    current_date = today
    non_working_days = []
    
    while current_date < target_date:
        if is_weekend_or_holiday(current_date):
            if current_date.weekday() >= 5:
                day_type = "周末"
            else:
                day_type = "节假日"
            non_working_days.append(f"{current_date} ({day_type})")
        current_date += datetime.timedelta(days=1)
    
    if non_working_days:
        print("\n期间的周末和节假日:")
        for day in non_working_days:
            print(f"- {day}")
    else:
        print("\n期间没有周末或节假日")