import sys
import re
from bs4 import BeautifulSoup

# 用法: python clean_html.py input.html output.html
# if len(sys.argv) != 3:
#     print("用法: python clean_html.py 输入文件 输出文件")
#     sys.exit(1)

# input_path = sys.argv[1]
input_path = r"input\2025年5月份70个大中城市商品住宅销售价格变动情况.html"
output_path = r"input\2025年5月份70个大中城市商品住宅销售价格变动情况2.html"

with open(input_path, 'r', encoding='utf-8') as f:
    html = f.read()

# 1. 删除 <div class="detail-text-content mhide"> 之前的内容
start_tag = '<div class="detail-text-content mhide">'
start_idx = html.find(start_tag)
if start_idx != -1:
    html = html[start_idx:]

# 2. 删除 <div class="mobile-content pchide"> 之后的内容
end_tag = '<div class="mobile-content pchide">'
end_idx = html.find(end_tag)
if end_idx != -1:
    html = html[:end_idx]

# 3. 删除所有标签内的 style 属性
soup = BeautifulSoup(html, 'html.parser')
for tag in soup.find_all(True):
    if 'style' in tag.attrs:
        del tag.attrs['style']

cleaned_html = str(soup)

with open(output_path, 'w', encoding='utf-8') as f:
    f.write(cleaned_html)

print(f"处理完成，输出文件: {output_path}") 