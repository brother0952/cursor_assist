import pandas as pd
import re

# 读取 Excel 文件
def read_economic_data(file_path, month):
    df_raw = pd.read_excel(file_path, header=None)
    # 预处理：去除空行和全空列
    df_raw = df_raw.dropna(how='all').dropna(axis=1, how='all')

    # 解析表头和分组
    records = []
    current_group = None
    current_category = None
    current_parent = None
    parent_stack = []
    for idx, row in df_raw.iterrows():
        values = row.fillna('').tolist()
        # 跳过表头行
        if idx < 3:
            continue
        cell = str(values[0])
        # 检查分组（如 "按类别分"、"一、食品烟酒"）
        if re.match(r'^\s*按类别分', cell):
            current_group = '按类别分'
            continue
        if re.match(r'^\s*\d+、', cell):
            current_category = cell.strip()
            parent_stack = []
            continue
        # 处理父子关系：遇到“其中：”关键字，记录父类
        if '其中：' in cell:
            parent = parent_stack[-1] if parent_stack else None
            current_parent = parent if parent else None
            parent_stack.append(cell.replace('其中：', '').strip())
            continue
        # 检查缩进（tab 或空格）表示子类
        indent = len(cell) - len(cell.lstrip())
        name = cell.strip()
        if name == '':
            continue
        # 如果有缩进且有父类，则归属于父类
        if indent > 0 and parent_stack:
            parent = parent_stack[-1]
        else:
            parent = None
            # 如果不是缩进行，且不是“其中：”，则清空父类栈
            parent_stack = []
        # 解析指标
        indicators = values[1:]
        # 生成记录
        record = {
            '月份': month,
            '分组': current_group,
            '类别': current_category,
            '父分类': parent,
            '项目': name,
            '环比涨跌幅(%)': indicators[0] if len(indicators) > 0 else None,
            '同比涨跌幅(%)': indicators[1] if len(indicators) > 1 else None,
            '1-6月同比涨跌幅(%)': indicators[2] if len(indicators) > 2 else None
        }
        records.append(record)
    # 转为 DataFrame
    df = pd.DataFrame(records)
    return df

if __name__ == '__main__':
    # 文件路径和月份
    file_path = 'P020250709332852003794.xlsx'
    month = '2025-06'
    df = read_economic_data(file_path, month)
    # 保存为 csv，便于后续追加
    df.to_csv('economic_data_2025_06.csv', index=False, encoding='utf-8-sig')
    print(df.head())
