



import re


pattern = r'(\d{4})年(\d{1,2})月份70个大中城市商品住宅销售价格变动情况'

def match_title(s):
    match = re.fullmatch(pattern, s)
    year = ""
    month = ""
    if match:
        year = match.group(1)
        month = match.group(2).zfill(2)
        print(f"匹配成功: '{s}' → 年份: {year}, 月份: {month}")
    else:
        print(f"不匹配: '{s}'")
    
    return year+month

res= match_title("2023年10月份70个大中城市商品住宅销售价格变动情况")
print(res)